import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from utils.basis import create_basis_functions, hessian
from methods.base_estimator import BaseEstimator


class ProximalNewtonEstimator(BaseEstimator):
    """
    Density estimator using Proximal Newton method with coordinate descent.
    
    This class implements the proximal Newton algorithm for solving the regularized 
    density estimation problem:
    
    min_θ  −ℓ(θ) + λ‖θ[1:]‖₁
    
    where ℓ is the HAL-basis log-likelihood and θ[0] is the intercept (unpenalized).
    The HAL basis functions are cumulative indicators I(x >= grid_point).
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
        log_dir: str = "./local/logs/experiment.log",  # Directory for logging (default is local)
        log_frequency: int = 10,  # Frequency of logging (-1 means no logging)
        non_desc_clip_alpha: bool = True,
        hessian_regularization: float = 1e-8,  # Regularization added to Hessian diagonal
        cd_skip_tolerance: float = 1e-8,  # Skip coordinate if diagonal element too small
        line_search_beta: float = 0.5,  # Step size reduction factor for line search
        non_descent_step_size: float = 0.1  # Fallback step size for non-descent directions
    ):
        """
        Initialize Proximal Newton estimator.
        
        Args:
            lam: L1 regularization parameter
            n_iterations: Maximum number of Newton iterations
            tol: Convergence tolerance
            n_grid_points: Number of grid points for density evaluation
            cd_sweeps: Number of coordinate descent sweeps per Newton step
            line_search_c: Line search parameter (Armijo condition)
            max_line_search_steps: Maximum line search steps
            basis_order: Order of the truncated power basis
            log_dir: Directory for logging
            log_frequency: Frequency of logging (-1 means no logging)
            hessian_regularization: Regularization added to Hessian diagonal
            cd_skip_tolerance: Skip coordinate if diagonal element too small
            line_search_beta: Step size reduction factor for line search
            non_descent_step_size: Fallback step size for non-descent directions
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
        
        # ProximalNewton-specific parameters
        self.n_grid_points = n_grid_points
        self.cd_sweeps = cd_sweeps
        self.line_search_c = line_search_c
        self.max_line_search_steps = max_line_search_steps
        self.non_desc_clip_alpha = non_desc_clip_alpha
        self.hessian_regularization = hessian_regularization
        self.cd_skip_tolerance = cd_skip_tolerance
        self.line_search_beta = line_search_beta
        self.non_descent_step_size = non_descent_step_size

        # Will be set during fitting
        self.warm_start_coefficients: Optional[np.ndarray] = None
        
    def _soft_threshold(self, z: float, tau: float) -> float:
        """Soft thresholding operator."""
        return np.sign(z) * max(abs(z) - tau, 0.0)
    
    def _objective_function(self, theta: np.ndarray, phi_data: np.ndarray, 
                           phi_grid: np.ndarray, delta_j: np.ndarray) -> float:
        """
        Compute the penalized objective function F(θ).
        
        Args:
            theta: parameter vector [θ₀, θ₁, ..., θₖ]
            phi_data: basis matrix at data points (N, K) 
            phi_grid: basis matrix at grid midpoints (m, K)
            delta_j: grid widths for integration (m,)
            
        Returns:
            Objective function value
        """
        N = phi_data.shape[0]
        
        # First term: -∑ᵢ log f(xᵢ) = -∑ᵢ (φᵢᵀθ)
        log_f_data = phi_data @ theta  # shape (N,)
        term1 = -np.sum(log_f_data)
        
        # Second term: N * log(∫ f(x) dx) using Riemann sum
        log_f_grid = phi_grid @ theta  # shape (m,)
        # Use logsumexp for numerical stability
        max_log_f = np.max(log_f_grid)
        log_integral = max_log_f + np.log(np.sum(np.exp(log_f_grid - max_log_f) * delta_j))
        term2 = N * log_integral
        
        # Penalty term (only on θ[1:], exclude intercept)
        penalty = self.lam * np.sum(np.abs(theta[1:]))
        
        return term1 + term2 + penalty
    
    def fit(
            self, 
            data: pd.DataFrame, 
            warm_start_coefficients: Optional[np.ndarray] = None,
            validation_data: Optional[pd.DataFrame] = None,
            validation_frequency: int = -1 # Frequency for validation logging (default -1 means no validation)
        ) -> 'ProximalNewtonEstimator':
        """
        Fit the Proximal Newton density estimator.
        
        Args:
            data: DataFrame with column 'W1' containing the observations
            
        Returns:
            Self for method chaining
        """
        n_samples = len(data)
        
        # 1) HAL basis on observed W1 (same as first-order methods)
        grid_points_hal = np.unique(data['W1'].dropna())
        b_ik, basis_names = create_basis_functions(data, grid_points_hal, order=self.basis_order)  # (n, K)
        self.basis_names = basis_names

        # 2) midpoint grid for the log‐normaliser
        grid_eval = np.linspace(0, 1, self.n_grid_points)
        midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2
        delta_j = grid_eval[1:] - grid_eval[:-1]  # numpy array
        df_mid = pd.DataFrame({'W1': midpoints})
        b_jk, _ = create_basis_functions(df_mid, grid_points_hal, order=self.basis_order)  # (m, K)
        
        # Convert to numpy arrays
        phi_data = b_ik.numpy()  # (N, K)
        phi_grid = b_jk.numpy()  # (m, K)
        
        # 3) Initialize θ = [θ₀, θ₁, ..., θₖ₋₁]
        K = phi_data.shape[1]

        if warm_start_coefficients is None:
            theta = np.zeros(K)
        elif len(warm_start_coefficients) == K:
            theta = warm_start_coefficients.copy()
        else:
            print(f"Warm start failed: expected {K} coefficients, got {len(warm_start_coefficients)}\n\n")
            theta = np.zeros(K)

        
        print(f"Starting Proximal Newton with K={K} parameters")
        
        # Simple initialization log
        if self.do_log:
            self.logger.info(f"ProximalNewton: lam={self.lam}, n_grid={self.n_grid_points}, cd_sweeps={self.cd_sweeps}, basis_order={self.basis_order}, n_samples={n_samples}, K={K}")
        
        # Main proximal Newton loop
        for iter_k in range(self.n_iterations):
            # 1. Compute gradient and Hessian
            # The basis functions already include intercept, so use them directly
            
            # Gradient computation
            # First term: -∑ᵢ φ(xᵢ) 
            grad_term1 = -np.sum(phi_data, axis=0)
            
            # Second term: N * E[φ(x)] where expectation is under current density
            # Compute weights for current density estimate
            log_f_grid = phi_grid @ theta
            max_log_f = np.max(log_f_grid)
            f_grid_unnorm = np.exp(log_f_grid - max_log_f)
            weights_unnorm = f_grid_unnorm * delta_j
            Z = np.sum(weights_unnorm)
            weights = weights_unnorm / Z
            
            # Weighted expectation
            grad_term2 = n_samples * np.sum(phi_grid * weights[:, None], axis=0)
            
            gradient = grad_term1 + grad_term2
            
            # 2. Hessian computation
            # The Hessian of the log-likelihood is n * Cov[φ(x)] under current density
            # Compute weighted expectation E[φ(x)] under current density
            weighted_mean = np.sum(phi_grid * weights[:, None], axis=0)  # (K,)
            
            # Compute weighted covariance matrix
            centered_phi = phi_grid - weighted_mean[None, :]  # (m, K)
            H = n_samples * np.sum(weights[:, None, None] * centered_phi[:, :, None] * centered_phi[:, None, :], axis=0)
            
            # Add regularization for numerical stability
            H += self.hessian_regularization * np.eye(K)
            
            # 3. Solve Newton subproblem via coordinate descent
            d_newton = np.zeros(K)
            h_diag = np.diag(H)
            
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
                        # For the proximal Newton step, we solve:
                        # min_d  (1/2) h_rr d² + r_r d + λ |θ_r + d|
                        # This gives: d = prox_{λ/h_rr}(-r_r/h_rr) - θ_r
                        unthresholded = -r_r / h_diag[r]
                        proximal_arg = unthresholded + theta[r]  # Current parameter + Newton step
                        thresholded = self._soft_threshold(proximal_arg, self.lam / h_diag[r])
                        d_newton[r] = thresholded - theta[r]
                    
                    # Update residual efficiently: r += H[:, r] * (d_new - d_old)
                    delta_d = d_newton[r] - old_d_r
                    if abs(delta_d) > 1e-12:
                        residual += H[:, r] * delta_d
            
            # 4. Line search with Armijo condition
            alpha = 1.0
            obj_current = self._objective_function(theta, phi_data, phi_grid, delta_j)
            
            # Early stopping if objective function explodes
            if self._check_objective_explosion(obj_current, iter_k):
                break
            
            # Directional derivative for line search
            directional_deriv = np.dot(gradient, d_newton)
            
            if directional_deriv >= -1e-12:  # Not a descent direction
                if iter_k % self.log_frequency == 0:
                    print(f"Warning: Non-descent direction at iteration {iter_k}, directional_deriv={directional_deriv:.2e}")
                # Use fallback step size
            else:
                # Backtracking line search with Armijo condition
                for ls_step in range(self.max_line_search_steps):
                    theta_trial = theta + alpha * d_newton
                    obj_trial = self._objective_function(theta_trial, phi_data, phi_grid, delta_j)
                    
                    # Armijo condition
                    if obj_trial <= obj_current + self.line_search_c * alpha * directional_deriv:
                        break
                    alpha *= self.line_search_beta
                else:
                    if iter_k % self.log_frequency == 0:
                        print(f"Warning: Line search failed at iteration {iter_k}, setting alpha from {alpha} to {self.non_descent_step_size}")
                    if self.non_desc_clip_alpha:
                        alpha = self.non_descent_step_size

            # 5. Update
            theta_new = theta + alpha * d_newton
            
            # Intercept correction for exact normalization
            logZ = np.log(np.sum(np.exp(phi_grid @ theta_new) * delta_j))
            if not np.isfinite(logZ):
                print(f"Warning: logZ became {logZ} at iteration {iter_k}, stopping optimization"); break
            theta_new[0] -= logZ
            
            # 6. Check convergence and logging
            change = np.max(np.abs(theta_new - theta))
            l1_norm = np.sum(np.abs(theta_new[1:]))  # Only penalized coefficients
            num_selected_knots = np.sum(np.abs(theta_new[1:]) > self.tol)

            # If validation data is provided, compute validation sum log-likelihood
            if validation_data is not None and validation_frequency > 0 and iter_k % validation_frequency == 0:
                # update self parameters from current parameters
                if iter_k == 0:
                    continue  # Skip first iteration for validation
                # Temporarily update parameters for validation
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
            else:
                validation_sum_log_likelihood = None
            
            # Log every iteration
            if self.do_log:
                self.logger.info(f"Iter {iter_k:3d}: obj={obj_current:.6f}, change={change:.2e}, "
                           f"α={alpha:.3f}, ‖θ[1:]‖₁={l1_norm:.3f}, num_selected_knots={num_selected_knots}")
            
            if iter_k % self.log_frequency == 0:
                print(f"Iter {iter_k:3d}: obj={obj_current:.6f}, change={change:.2e}, "
                      f"α={alpha:.3f}, ‖θ[1:]‖₁={l1_norm:.3f}, num_selected_knots={num_selected_knots}")
                
            
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
        
        # Select non-zero knots
        # select non-zero knots (only for truncated power terms, not polynomial terms)
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
        
        # Simple final log
        final_obj = self._objective_function(self.theta_hat, phi_data, phi_grid, delta_j)
        final_selected_knots = np.sum(np.abs(self.theta_hat[1:]) > self.tol)
        if self.do_log:
            self.logger.info(f"Final: obj={final_obj:.6f}, selected_knots={final_selected_knots}, iterations={iter_k + 1}")
        
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
        
        if self.theta_hat is None:
            raise ValueError("theta_hat is None, fitting may have failed.")
        
        # Create evaluation grid
        grid_points = np.linspace(0, 1, self.n_grid_points)
        
        # Create DataFrame for evaluation points
        df_eval = pd.DataFrame({'W1': grid_points})
        
        # Create basis functions at evaluation points using the same grid points as training
        b_eval, _ = create_basis_functions(df_eval, self._grid_points_hal, order=self.basis_order)
        phi_eval = b_eval.numpy()  # (200, K)
        
        # Compute log density
        log_density = phi_eval @ self.theta_hat
        
        # Use the same normalization approach as in training (midpoints)
        df_mid = pd.DataFrame({'W1': self.grid_midpoints})
        b_mid, _ = create_basis_functions(df_mid, self._grid_points_hal, order=self.basis_order)
        phi_mid = b_mid.numpy()
        
        log_f_mid = phi_mid @ self.theta_hat
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