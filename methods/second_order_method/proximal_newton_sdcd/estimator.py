import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from utils.basis import create_basis_functions
from scipy.special import logsumexp
from methods.base_estimator import BaseEstimator


class ProximalNewtonScaledDiagonalCDEstimator(BaseEstimator):
    """
    Density estimator using Proximal Newton method with adaptive diagonal scaling and coordinate descent.
    
    This class implements a hybrid quasi-Newton algorithm that combines coordinate descent with 
    adaptive diagonal Hessian approximation for solving the regularized density estimation problem:
    
    min_θ  −ℓ(θ) + λ‖θ[1:]‖₁
    
    where ℓ is the HAL-basis log-likelihood and θ[0] is the intercept (unpenalized).
    The HAL basis functions are cumulative indicators I(x >= grid_point).
    
    Algorithm Overview:
    - Uses a scalar diagonal approximation H ≈ γₖ·I where γₖ is adaptively updated
    - Employs coordinate descent to solve the proximal Newton subproblem efficiently
    - Updates diagonal scaling γₖ using recent parameter and gradient differences
    - Uses efficient residual updates during coordinate descent sweeps
    
    Key Features:
    - Memory efficient: O(K) storage instead of O(K²) for full Hessian
    - Computationally efficient: coordinate descent often faster than linear solves
    - Adaptive scaling: γₖ provides some second-order curvature information
    - Robust: handles non-descent directions with fallback mechanisms
    
    Note: This is NOT a standard L-BFGS implementation, but rather a simplified 
    approach that uses adaptive diagonal scaling inspired by quasi-Newton methods.
    """
    
    def __init__(
        self,
        lam: float = 3.0,
        n_iterations: int = 100,
        tol: float = 1e-6,
        n_grid_points: int = 200,
        cd_sweeps: int = 2,  # coordinate descent sweeps per Newton step
        line_search_c: float = 1e-4,
        max_line_search_steps: int = 20,
        basis_order: int = 0,
        log_dir: str = "./local/logs/experiment.log",
        log_frequency: int = 10,
        non_desc_clip_alpha: bool = True,
        # Exposed hyperparameters
        initial_diagonal_scale: float = 1.0,  # Initial diagonal scaling factor
        cd_skip_tolerance: float = 1e-8,  # Skip coordinate if diagonal element too small
        line_search_beta: float = 0.5,  # Step size reduction factor for line search
        failed_line_search_step_size: float = 0.01,  # Step size when line search fails
        non_descent_step_size: float = 0.1,  # Fallback step size for non-descent directions
        scaling_update_tolerance: float = 1e-12,  # Tolerance for diagonal scaling updates
        diagonal_scale_clip_range: Tuple[float, float] = (1e-6, 1e3)  # Clipping range for γₖ
    ):
        """
        Initialize Proximal Newton Scaled Diagonal estimator.
        
        Args:
            lam: L1 regularization parameter
            n_iterations: Maximum number of Newton iterations
            tol: Convergence tolerance
            n_grid_points: Number of grid points for density evaluation
            cd_sweeps: Number of coordinate descent sweeps per Newton step
            line_search_c: Line search parameter (Armijo condition)
            max_line_search_steps: Maximum line search steps
            basis_order: Order of truncated power basis functions (0 for step functions, ≥1 for polynomial splines)
            log_dir: Directory for logging
            log_frequency: Frequency of logging (-1 means no logging)
            initial_diagonal_scale: Initial diagonal scaling factor γ₀
            cd_skip_tolerance: Skip coordinate if diagonal element too small
            line_search_beta: Step size reduction factor for line search
            failed_line_search_step_size: Step size when line search fails
            non_descent_step_size: Fallback step size for non-descent directions
            scaling_update_tolerance: Tolerance for diagonal scaling updates
            diagonal_scale_clip_range: Clipping range for γₖ to prevent numerical issues
        """
        # Initialize base class
        super().__init__(
            lam=lam,
            n_iterations=n_iterations,
            tol=tol,
            basis_order=basis_order,
            log_dir=log_dir,
            log_frequency=log_frequency
        )
        
        # Algorithm-specific parameters
        self.n_grid_points = n_grid_points
        self.cd_sweeps = cd_sweeps
        self.line_search_c = line_search_c
        self.max_line_search_steps = max_line_search_steps
        self.non_desc_clip_alpha = non_desc_clip_alpha
        self.initial_diagonal_scale = initial_diagonal_scale
        self.cd_skip_tolerance = cd_skip_tolerance
        self.line_search_beta = line_search_beta
        self.failed_line_search_step_size = failed_line_search_step_size
        self.non_descent_step_size = non_descent_step_size
        self.scaling_update_tolerance = scaling_update_tolerance
        self.diagonal_scale_clip_range = diagonal_scale_clip_range
        
    def _soft_threshold(self, z: float, tau: float) -> float:
        """Soft thresholding operator."""
        return np.sign(z) * max(abs(z) - tau, 0.0)
    
    def _objective_function(self, theta: np.ndarray, phi_data_full: np.ndarray, 
                           phi_grid_full: np.ndarray, delta_j: np.ndarray) -> float:
        """
        Compute the penalized objective function F(θ).
        
        Args:
            theta: parameter vector [θ₀, θ₁, ..., θₖ]
            phi_data_full: full basis matrix at data points (N, K) with intercept
            phi_grid_full: full basis matrix at grid midpoints (m, K) with intercept
            delta_j: grid widths for integration (m,)
            
        Returns:
            Objective function value
        """
        N = phi_data_full.shape[0]
        
        # First term: -∑ᵢ log f(xᵢ) = -∑ᵢ (φᵢᵀθ)
        log_f_data = phi_data_full @ theta  # shape (N,)
        term1 = -np.sum(log_f_data.astype(np.float64))
        
        # Second term: N * log(∫ f(x) dx) using Riemann sum
        log_f_grid = phi_grid_full @ theta  # shape (m,)
        # Use logsumexp for numerical stability
        max_log_f = np.max(log_f_grid)
        log_integral = max_log_f + np.log(np.sum(np.exp(log_f_grid - max_log_f) * delta_j))
        term2 = N * log_integral
        
        # Penalty term (only on θ[1:], intercept is unpenalized)
        penalty = self.lam * np.sum(np.abs(theta[1:]))
        
        return term1 + term2 + penalty
    
    def _compute_gradient(self, theta: np.ndarray, phi_data_full: np.ndarray, 
                         phi_grid_full: np.ndarray, delta_j: np.ndarray, 
                         n_samples: int) -> np.ndarray:
        """
        Compute gradient of the objective function.
        
        Args:
            theta: current parameter vector
            phi_data_full: basis matrix at data points with intercept (N, K)
            phi_grid_full: basis matrix at grid points with intercept (m, K)
            delta_j: grid widths for integration
            n_samples: number of data samples
            
        Returns:
            Gradient vector
        """
        # First term: -∑ᵢ φ(xᵢ) 
        grad_term1 = -np.sum(phi_data_full, axis=0)
        
        # Second term: N * E[φ(x)] where expectation is under current density
        # Compute weights for current density estimate
        log_f_grid = phi_grid_full @ theta
        max_log_f = np.max(log_f_grid)
        f_grid_unnorm = np.exp(log_f_grid - max_log_f)
        weights_unnorm = f_grid_unnorm * delta_j
        Z = np.sum(weights_unnorm)
        weights = weights_unnorm / Z
        
        # Weighted expectation
        grad_term2 = n_samples * np.sum(phi_grid_full * weights[:, None], axis=0)
        
        return grad_term1 + grad_term2
    
    def fit(self, data: pd.DataFrame, warm_start_coefficients: Optional[np.ndarray] = None,
            validation_data: Optional[pd.DataFrame] = None,
            validation_frequency: int = -1) -> 'ProximalNewtonScaledDiagonalCDEstimator':
        """
        Fit the Proximal Newton Scaled Diagonal density estimator.
        
        Uses coordinate descent with adaptive diagonal scaling to solve the regularized
        density estimation problem. The algorithm maintains a scalar diagonal approximation
        H ≈ γₖ·I that is updated using recent parameter and gradient differences.
        
        Args:
            data: DataFrame with column 'W1' containing the observations
            warm_start_coefficients: Optional initial coefficients for warm starting
            validation_data: Optional validation data for tracking performance
            validation_frequency: Frequency for validation logging (default -1 means no validation)
            
        Returns:
            Self for method chaining
        """


        n_samples = len(data)
        
        # 1) HAL basis on observed W1 (same as first-order methods)
        grid_points_hal = np.unique(data['W1'].dropna())
        b_ik, basis_names = create_basis_functions(data, grid_points_hal, order=self.basis_order)  # (n, K−1)
        self.basis_names = basis_names

        # 2) midpoint grid for the log‐normaliser
        grid_eval = np.linspace(0, 1, self.n_grid_points)
        midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2
        delta_j = grid_eval[1:] - grid_eval[:-1]  # numpy array
        df_mid = pd.DataFrame({'W1': midpoints})
        b_jk, _ = create_basis_functions(df_mid, grid_points_hal, order=self.basis_order)  # (m, K−1)
        
        # Convert to numpy arrays - these already include intercept in the new basis
        phi_data_full = b_ik.numpy()  # (N, K)
        phi_grid_full = b_jk.numpy()  # (m, K)
        
        # 3) Initialize θ = [θ₀, θ₁, ..., θₖ₋₁]
        # K is determined by the basis function output dimensions
        K = phi_data_full.shape[1]
        
        if warm_start_coefficients is None:
            theta = np.zeros(K)
        elif len(warm_start_coefficients) == K:
            theta = warm_start_coefficients.copy()
        else:
            print(f"Warm start failed: expected {K} coefficients, got {len(warm_start_coefficients)}\n\n")
            theta = np.zeros(K)

        # Initialize diagonal scaling storage
        gamma_k = self.initial_diagonal_scale  # current diagonal scaling factor
        
        print(f"Starting Proximal Newton + Scaled Diagonal with K={K} parameters")
        
        # Simple initialization log
        if self.do_log:
            self.logger.info(f"ProximalNewtonScaledDiagonal: lam={self.lam}, n_grid={self.n_grid_points}, basis_order={self.basis_order}, n_samples={n_samples}, K={K}")
        
        # Main proximal Newton loop
        for iter_k in range(self.n_iterations):
            # 1. Compute gradient
            gradient = self._compute_gradient(theta, phi_data_full, phi_grid_full, 
                                            delta_j, n_samples)
            
            # 2. Use scalar diagonal approximation H ≈ γₖ·I
            # All diagonal entries use the same scaling factor γₖ
            h_diag = np.full(K, max(gamma_k, self.cd_skip_tolerance))
            
            # 3. Solve Newton subproblem via coordinate descent
            d_newton = np.zeros(K)
            
            # Coordinate descent with proper residual updates
            residual = gradient.copy()  # r = g + H @ d (initially d=0, so r=g)
            
            for sweep in range(self.cd_sweeps):
                for r in range(K):
                    if h_diag[r] <= self.cd_skip_tolerance:  # Skip if diagonal is too small
                        continue
                        
                    # Current residual at coordinate r
                    r_r = residual[r]
                    
                    old_d_r = d_newton[r]
                    
                    if r == 0:  # Intercept is never penalized
                        # Standard Newton step: d_r = -r_r / h_rr
                        d_newton[r] = -r_r / h_diag[r]
                    else:
                        # Penalized coordinate: solve the proximal subproblem
                        # Newton direction: θ - g/H (moving against gradient scaled by curvature)
                        raw = theta[r] - r_r / h_diag[r]  # θ - g/H
                        thresh = self.lam / h_diag[r]
                        z = self._soft_threshold(raw, thresh)
                        d_newton[r] = z - theta[r]
                    
                    # Update residual efficiently: only update r-th coordinate
                    # Since H ≈ γₖ·I, updating d[r] affects residual as: r += γₖ·(d[r] - d_old[r])·eᵣ
                    delta_d = d_newton[r] - old_d_r
                    if abs(delta_d) > 1e-12:
                        residual += h_diag[r] * delta_d  # Broadcast to all coordinates
            
            # 4. Line search with Armijo condition
            alpha = 1.0
            obj_current = self._objective_function(theta, phi_data_full, phi_grid_full, delta_j)
            
            # Early stopping if objective function explodes
            if self._check_objective_explosion(obj_current, iter_k):
                break
            
            # Directional derivative for line search
            directional_deriv = np.dot(gradient, d_newton)
            
            if directional_deriv >= -1e-12:  # Not a descent direction
                if iter_k % self.log_frequency == 0:
                    print(f"Warning: Non-descent direction at iteration {iter_k}, directional_deriv={directional_deriv:.2e}, setting alpha from {alpha} to 0.1")
                if self.non_desc_clip_alpha:
                    alpha = self.non_descent_step_size
            else:
                # Backtracking line search with Armijo condition
                for ls_step in range(self.max_line_search_steps):
                    theta_trial = theta + alpha * d_newton
                    obj_trial = self._objective_function(theta_trial, phi_data_full, phi_grid_full, delta_j)
                    
                    # Armijo condition
                    if obj_trial <= obj_current + self.line_search_c * alpha * directional_deriv:
                        break
                    alpha *= self.line_search_beta
                else:
                    if iter_k % self.log_frequency == 0:
                        print(f"Line search failed at iteration {iter_k}, using small step")
                    alpha = self.failed_line_search_step_size
            
            # 5. Update
            theta_new = theta + alpha * d_newton

            # Intercept correction for exact normalization
            # logZ = np.log(np.sum(np.exp(phi_grid_full @ theta_new) * delta_j))
            logZ = logsumexp(phi_grid_full @ theta_new, b=delta_j, axis=0)
            if self.do_log and iter_k % self.log_frequency == 0:
                print(f"Iteration {iter_k}: logZ={logZ:.6f}")
            if not np.isfinite(logZ):
                print(f"Warning: logZ became {logZ} at iteration {iter_k}, stopping optimization"); break
            theta_new[0] -= logZ  # Subtract from intercept only
            
            # 6. Compute new gradient for diagonal scaling update
            gradient_new = self._compute_gradient(theta_new, phi_data_full, phi_grid_full, 
                                                delta_j, n_samples)
            
            # 7. Update diagonal scaling using most recent (s,y) pair
            s_k = theta_new - theta
            y_k = gradient_new - gradient
            
            # Update γₖ for next iteration using current (s,y) pair
            # This provides adaptive diagonal scaling based on recent curvature information
            if np.dot(s_k, y_k) > self.scaling_update_tolerance:
                gamma_k = np.dot(y_k, y_k) / np.dot(y_k, s_k)
            # Clip γₖ to prevent numerical issues
            gamma_k = np.clip(gamma_k, self.diagonal_scale_clip_range[0], self.diagonal_scale_clip_range[1])

            
            # 8. Check convergence
            change = np.max(np.abs(theta_new - theta))
            
            # Validation and logging
            if validation_data is not None and validation_frequency > 0 and iter_k % validation_frequency == 0:
                # Update parameters for validation
                old_fitted = self.is_fitted
                self.theta_hat = theta_new
                self.grid_midpoints = midpoints
                self.delta_j = delta_j
                self._grid_points_hal = grid_points_hal
                self.is_fitted = True  # Temporarily set to fitted for validation

                
                validation_pts = validation_data['W1'].values
                validation_sum_log_likelihood = self.get_sum_log_likelihood_for_points(validation_pts)
                if self.do_log:
                    self.logger.info(f"Validation at iter {iter_k}: sum_log_likelihood={validation_sum_log_likelihood:.6f}")
                print(f"Validation at iter {iter_k}: sum_log_likelihood={validation_sum_log_likelihood:.6f}")

                # Restore previous fitted state
                self.is_fitted = old_fitted
            
            if iter_k % self.log_frequency == 0 or self.do_log and iter_k % self.log_frequency == 0:  # Report every 10 iterations
                l1_norm = np.sum(np.abs(theta_new[1:]))  # Only penalized coefficients
                num_knots = np.sum(np.abs(theta_new[1:]) > self.tol)
                
                if self.do_log and iter_k % self.log_frequency == 0:
                    self.logger.info(f"Iter {iter_k:3d}: obj={obj_current:.6f}, change={change:.2e}, α={alpha:.3f}, ‖θ[1:]‖₁={l1_norm:.3f}, γ={gamma_k:.3f}, num_knots={num_knots}")
                
                if iter_k % self.log_frequency == 0:
                    print(f"Iter {iter_k:3d}: obj={obj_current:.6f}, change={change:.2e}, "
                          f"α={alpha:.3f}, ‖θ[1:]‖₁={l1_norm:.3f}, γ={gamma_k:.3f}, num_knots={num_knots}")
            
            if change < self.tol:
                if self.do_log:
                    self.logger.info(f"Converged at iteration {iter_k}")
                print(f"Converged at iteration {iter_k}")
                break
                
            theta = theta_new
        
        # Store results
        self.theta_hat = theta
        self.grid_midpoints = midpoints
        self.delta_j = delta_j
        self._grid_points_hal = grid_points_hal
        
        # Select non-zero knots (only for truncated power terms, not polynomial terms)
        if self.basis_order == 0:
            # For order 0: theta = [intercept, step_functions...]
            truncated_power_coeffs = self.theta_hat[1:]
        else:
            # For order k≥1: theta = [intercept, x, x^2, ..., x^k, (x-ξ₁)₊^k, ...]
            truncated_power_coeffs = self.theta_hat[1 + self.basis_order:]
        
        mask = np.abs(truncated_power_coeffs) > self.tol
        self.grid_points_hal_selected = grid_points_hal[mask]
        
        # Create evaluation grid for density
        self.grid_points = np.linspace(0, 1, 200)
        
        # Final log
        final_selected_knots = np.sum(np.abs(self.theta_hat[1:]) > self.tol)
        if self.do_log:
            self.logger.info(f"Final: selected_knots={final_selected_knots}, iterations={iter_k}")
        
        self.is_fitted = True

        # Store the fitted theta as a dictionary for inspection
        assert len(self.basis_names) == len(self.theta_hat), "Basis names count does not match theta_hat length"
        self.fitted_theta_dict = {name: value for name, value in zip(self.basis_names, self.theta_hat.tolist())}

        return self
    
    def get_density(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the estimated density on the evaluation grid.
        
        Returns:
            Tuple of (grid_points, density_values)
            
        Raises:
            ValueError: If the estimator hasn't been fitted yet
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting density. Call fit() first.")
        
        # Create evaluation grid
        grid_points = np.linspace(0, 1, self.n_grid_points)
        
        # Create DataFrame for evaluation points
        df_eval = pd.DataFrame({'W1': grid_points})
        
        # Create basis functions at evaluation points using the same grid points as training
        b_eval, _ = create_basis_functions(df_eval, self._grid_points_hal, order=self.basis_order)
        phi_eval_full = b_eval.numpy()  # (200, K) - already includes intercept
        
        # Compute log density
        log_density = phi_eval_full @ self.theta_hat
        
        # Use the same normalization approach as in training (midpoints)
        df_mid = pd.DataFrame({'W1': self.grid_midpoints})
        b_mid, _ = create_basis_functions(df_mid, self._grid_points_hal, order=self.basis_order)
        phi_mid_full = b_mid.numpy()  # already includes intercept
        
        log_f_mid = phi_mid_full @ self.theta_hat
        max_log_f = np.max(log_f_mid)
        Z = np.sum(np.exp(log_f_mid - max_log_f) * self.delta_j)
        
        # Compute normalized density
        density = np.exp(log_density - max_log_f) / Z

        # make sure density sums to 1
        density /= np.sum(density * (np.linspace(0, 1+1/self.n_grid_points, 1+self.n_grid_points)[1:] - np.linspace(0, 1+1/self.n_grid_points, 1+self.n_grid_points)[:-1]))
        
        return grid_points, density
    
    def get_results(self) -> Dict:
        """
        Get comprehensive results from the fitting process.
        
        Returns:
            Dictionary containing all relevant results
            
        Raises:
            ValueError: If the estimator hasn't been fitted yet
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting results. Call fit() first.")
        
        return self._get_common_results()