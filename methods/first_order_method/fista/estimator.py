import numpy as np
import torch
import pandas as pd
from typing import Dict, Optional, Tuple
from utils.basis import create_basis_functions
from methods.base_estimator import BaseEstimator


class FISTAEstimator(BaseEstimator):
    """
    Density estimator using FISTA (accelerated proximal gradient descent).
    
    This class implements the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
    for solving the regularized density estimation problem:
    
    min_θ  −ℓ(θ) + λ‖θ[1:]‖₁
    
    where ℓ is the HAL-basis log-likelihood and θ[0] is the intercept (unpenalized).
    """
    
    def __init__(
        self,
        lam: float = 3.0,
        L: float = 2000.0,
        n_iterations: int = 8000,
        tol: float = 1e-6,
        n_grid_points: int = 200,
        basis_order: int = 0,
        log_dir: str = "./local/logs/experiment.log",
        log_frequency: int = 500
    ):
        """
        Initialize FISTA estimator.
        
        Args:
            lam: L1 regularization parameter
            L: Lipschitz constant estimate for step size (step = 1/L)
            n_iterations: Maximum number of iterations
            tol: Convergence tolerance
            n_grid_points: Number of grid points for density evaluation
            basis_order: Order of the truncated power basis
            log_dir: Directory for logging
            log_frequency: Frequency of logging (-1 means no logging)
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
        
        # FISTA-specific parameters
        self.L = L
        self.n_grid_points = n_grid_points
        
        # Will be set during fitting
        self.grid_points_hal_selected: Optional[np.ndarray] = None
        
    def fit(self, data: pd.DataFrame, warm_start_coefficients: Optional[np.ndarray] = None,
            validation_data: Optional[pd.DataFrame] = None,
            validation_frequency: int = -1) -> 'FISTAEstimator':
        """
        Fit the FISTA density estimator.
        
        Args:
            data: DataFrame with column 'W1' containing the observations
            warm_start_coefficients: Optional initial coefficients for warm starting
            validation_data: Optional validation data for tracking performance
            validation_frequency: Frequency for validation logging (default -1 means no validation)
            
        Returns:
            Self for method chaining
        """
        n_samples = len(data)
        
        # 1) HAL basis on observed W1
        grid_points_hal = np.unique(data['W1'].dropna())
        b_ik, basis_names = create_basis_functions(data, grid_points_hal, order=self.basis_order)  # (n, K)
        self.basis_names = basis_names
        
        # 2) midpoint grid for the log‐normaliser
        grid_eval = np.linspace(0, 1, self.n_grid_points)
        midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2
        delta_j = torch.tensor(grid_eval[1:] - grid_eval[:-1], dtype=torch.float32)
        log_delta_j = torch.log(delta_j)
        df_mid = pd.DataFrame({'W1': midpoints})
        b_jk, _ = create_basis_functions(df_mid, grid_points_hal, order=self.basis_order)  # (m, K)
        
        # 3) initialise θ⁽⁰⁾ and θ⁽⁻¹⁾ = θ⁽⁰⁾
        K = b_ik.shape[1]  # number of parameters
        
        if warm_start_coefficients is None:
            theta_old = torch.zeros(K)    # θ^(k-2), also θ^(0) at start
            theta_cur = torch.zeros(K)    # θ^(k-1), also θ^(0) at start
        elif len(warm_start_coefficients) == K:
            # Warm start coefficients should include intercept
            theta_old[1:] = torch.tensor(warm_start_coefficients, dtype=torch.float32, requires_grad=True)
            theta_cur[1:] = torch.tensor(warm_start_coefficients, dtype=torch.float32, requires_grad=True)
        else:
            print(f"Warm start failed: expected {K} coefficients, got {len(warm_start_coefficients)}\n\n")

            theta_old = torch.zeros(K, requires_grad=True)
            theta_cur = torch.zeros(K, requires_grad=True)
        
        step = 1.0 / self.L
        
        print(f"Starting FISTA with K={K} parameters")
        
        # Simple initialization log
        if self.do_log:
            self.logger.info(f"FISTA: lam={self.lam}, L={self.L}, n_grid={self.n_grid_points}, basis_order={self.basis_order}, n_samples={n_samples}, K={K}")
        
        for k in range(1, self.n_iterations + 1):
            # 4a) form momentum point v = θ^(k-1) + ((k-2)/(k+1)) * (θ^(k-1) - θ^(k-2))
            momentum = (k - 2) / (k + 1)
            v = theta_cur + momentum * (theta_cur - theta_old)
            
            # 4b) compute gradient ∇(−ℓ) at v
            v_tens = v.clone().detach().requires_grad_(True)
            
            # negative log-likelihood at v
            log_f_data = b_ik @ v_tens
            term1 = -log_f_data.sum()
            log_f_grid = b_jk @ v_tens
            term2 = n_samples * torch.logsumexp(log_delta_j + log_f_grid, dim=0)
            loss_nopen = term1 + term2
            
            # Early stopping if objective function explodes
            if self._check_objective_explosion(loss_nopen.item(), k):
                break
            
            loss_nopen.backward()
            grad_v = v_tens.grad      # ∇(−ℓ)(v)
            
            if grad_v is None:
                raise RuntimeError("Gradient computation failed - grad_v is None")
            
            # 4c) gradient step + soft-threshold
            with torch.no_grad():
                theta_next = v - step * grad_v
                # soft-threshold on θ[1:] (exclude intercept)
                u = theta_next[1:]
                shrunk = torch.clamp(u.abs() - self.lam * step, min=0.0)
                theta_next[1:] = torch.sign(u) * shrunk
                
                # Intercept correction for exact normalization
                logZ = torch.log(torch.sum(torch.exp(b_jk @ theta_next) * delta_j))
                if not torch.isfinite(logZ):
                    print(f"Warning: logZ became {logZ.item()} at iteration {k}, stopping optimization"); break
                theta_next[0] -= logZ  # Subtract from intercept only
            
            # 4d) check convergence on max parameter change
            change = torch.max(torch.abs(theta_next - theta_cur)).item()
            if change < self.tol:
                theta_old, theta_cur = theta_cur, theta_next
                if self.do_log:
                    self.logger.info(f"Converged at iteration {k}")
                print(f"Converged at iter {k}")
                break
            
            # rotate for next iteration
            theta_old, theta_cur = theta_cur, theta_next
            
            # Validation and logging
            if validation_data is not None and validation_frequency > 0 and k % validation_frequency == 0:
                # Update parameters for validation
                self.theta_hat = theta_cur.cpu().numpy()
                self.grid_midpoints = midpoints
                self.delta_j = delta_j.numpy()
                self._grid_points_hal = grid_points_hal
                
                validation_pts = validation_data['W1'].values
                validation_sum_log_likelihood = self.get_sum_log_likelihood_for_points(validation_pts)
                if self.do_log:
                    self.logger.info(f"Validation at iter {k}: sum_log_likelihood={validation_sum_log_likelihood:.6f}")
                print(f"Validation at iter {k}: sum_log_likelihood={validation_sum_log_likelihood:.6f}")
            
            if k % self.log_frequency == 0:
                l1norm = theta_cur[1:].abs().sum().item()
                num_selected_knots = (theta_cur[1:].abs() > self.tol).sum().item()
                loss_val = loss_nopen.item()
                
                if self.do_log:
                    self.logger.info(f"Iter {k:4d}: loss={loss_val:.4f}, change={change:.2e}, ‖θ[1:]‖₁={l1norm:.3f}, num_selected_knots={num_selected_knots}")
                
                print(f"Iter {k:4d}: loss={loss_val:.4f}, change={change:.2e}, ‖θ[1:]‖₁={l1norm:.3f}, num_selected_knots={num_selected_knots}")
        
        # Store results
        self.theta_hat = theta_cur.cpu().numpy()
        self.grid_midpoints = midpoints
        self.delta_j = delta_j.numpy()
        
        # Store grid points for density computation
        self._grid_points_hal = grid_points_hal
        
        # select non-zero knots (only for truncated power terms, not polynomial terms)
        if self.basis_order == 0:
            # For order 0: theta = [intercept, step_functions...]
            truncated_power_coeffs = self.theta_hat[1:]
        else:
            # For order k≥1: theta = [intercept, x, x^2, ..., x^k, (x-ξ₁)₊^k, ...]
            truncated_power_coeffs = self.theta_hat[1 + self.basis_order:]
        
        mask = np.abs(truncated_power_coeffs) > self.tol
        self.grid_points_hal_selected = grid_points_hal[mask]
        
        # Final log
        final_selected_knots = np.sum(np.abs(self.theta_hat[1:]) > self.tol)
        if self.do_log:
            self.logger.info(f"Final: selected_knots={final_selected_knots}, iterations={k}")
        
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
        
        if self.grid_midpoints is None or self.delta_j is None or self.theta_hat is None:
            raise ValueError("Missing required data - ensure fit() completed successfully.")
        
        # rebuild design for midpoints:
        b_jk, _ = create_basis_functions(
            pd.DataFrame({'W1': self.grid_midpoints}),
            self._grid_points_hal,
            order=self.basis_order
        )

        b_jk_np = b_jk.numpy()

        log_f_grid = b_jk_np @ self.theta_hat
        f_u = np.exp(log_f_grid)
        Z = np.sum(f_u * self.delta_j)
        density_hat = f_u / Z

        # make sure density sums to 1
        density_hat /= np.sum(density_hat * (np.linspace(0, 1, self.n_grid_points)[1:] - np.linspace(0, 1, self.n_grid_points)[:-1]))
        
        return self.grid_midpoints, density_hat
    
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