import numpy as np
import torch
import pandas as pd
from typing import Dict, Optional, Tuple
from utils.basis import create_basis_functions
from methods.base_estimator import BaseEstimator


class ProximalGDEstimator(BaseEstimator):
    """
    Density estimator using ISTA (proximal gradient descent).
    
    This class implements the Iterative Shrinkage-Thresholding Algorithm (ISTA)
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
        Initialize ISTA estimator.
        
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
        
        # ProximalGD-specific parameters
        self.L = L
        self.n_grid_points = n_grid_points
        
        # Will be set during fitting
        self.grid_points_hal_selected: Optional[np.ndarray] = None
        
    def fit(self, data: pd.DataFrame, warm_start_coefficients: Optional[np.ndarray] = None,
            validation_data: Optional[pd.DataFrame] = None,
            validation_frequency: int = -1) -> 'ProximalGDEstimator':
        """
        Fit the ISTA density estimator.
        
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
        
        # 3) initialise θ
        K = b_ik.shape[1]  # number of parameters
        
        if warm_start_coefficients is None:
            theta = torch.zeros(K, requires_grad=True)
        elif len(warm_start_coefficients) == K:
            theta = torch.tensor(warm_start_coefficients, dtype=torch.float32, requires_grad=True)
        else:
            print(f"Warm start failed: expected {K} coefficients, got {len(warm_start_coefficients)}\n\n")
            theta = torch.zeros(K, requires_grad=True)
        
        step = 1.0 / self.L
        
        print(f"Starting ProximalGD (ISTA) with K={K} parameters")
        
        # Simple initialization log
        if self.do_log:
            self.logger.info(f"ProximalGD: lam={self.lam}, L={self.L}, n_grid={self.n_grid_points}, basis_order={self.basis_order}, n_samples={n_samples}, K={K}")
        
        # 4) ISTA loop (manual GD + prox)
        for it in range(self.n_iterations):
            # zero previous grad
            if theta.grad is not None:
                theta.grad.zero_()
            
            # compute neg‐log‐lik (without penalty)
            log_f_data = b_ik @ theta           # (n,)
            term1 = -log_f_data.sum()
            log_f_grid = b_jk @ theta           # (m,)
            term2 = n_samples * torch.logsumexp(log_delta_j + log_f_grid, dim=0)
            loss_nopen = term1 + term2
            
            # Early stopping if objective function explodes
            if self._check_objective_explosion(loss_nopen.item(), it):
                break
            
            # gradient step
            loss_nopen.backward()
            
            if theta.grad is None:
                raise RuntimeError("Gradient computation failed - theta.grad is None")
            
            with torch.no_grad():
                theta.data -= step * theta.grad
                
                # proximal: soft‐threshold θ[1:]
                v = theta.data[1:]
                shrunk = torch.clamp(v.abs() - self.lam * step, min=0.0)
                theta.data[1:] = torch.sign(v) * shrunk
                
                # Intercept correction for exact normalization
                logZ = torch.log(torch.sum(torch.exp(b_jk @ theta) * delta_j))
                if not torch.isfinite(logZ):
                    print(f"Warning: logZ became {logZ.item()} at iteration {it}, stopping optimization"); break
                theta.data[0] -= logZ  # Subtract from intercept only
            
            # stopping rule: max change in grad small
            max_grad = theta.grad.abs().max().item()
            
            # Validation and logging
            if validation_data is not None and validation_frequency > 0 and it % validation_frequency == 0:
                # Update parameters for validation
                self.theta_hat = theta.detach().cpu().numpy()
                self.grid_midpoints = midpoints
                self.delta_j = delta_j.numpy()
                self._grid_points_hal = grid_points_hal
                
                validation_pts = validation_data['W1'].values
                validation_sum_log_likelihood = self.get_sum_log_likelihood_for_points(validation_pts)
                if self.do_log:
                    self.logger.info(f"Validation at iter {it}: sum_log_likelihood={validation_sum_log_likelihood:.6f}")
                print(f"Validation at iter {it}: sum_log_likelihood={validation_sum_log_likelihood:.6f}")
            
            if it % self.log_frequency == 0:
                l1norm = theta.data[1:].abs().sum().item()
                num_selected_knots = (theta.data[1:].abs() > self.tol).sum().item()
                loss_val = loss_nopen.item()
                
                if self.do_log:
                    self.logger.info(f"Iter {it:4d}: loss={loss_val:.4f}, max_grad={max_grad:.2e}, ‖θ[1:]‖₁={l1norm:.3f}, num_selected_knots={num_selected_knots}")
                
                print(f"Iter {it:4d}: loss={loss_val:.4f}, max_grad={max_grad:.2e}, ‖θ[1:]‖₁={l1norm:.3f}, num_selected_knots={num_selected_knots}")
            
            if max_grad < self.tol:
                if self.do_log:
                    self.logger.info(f"Converged at iteration {it}")
                print(f"Converged at iter {it}")
                break
        
        # Store results
        self.theta_hat = theta.detach().cpu().numpy()
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
        
        mask = np.abs(truncated_power_coeffs) > 1e-4
        self.grid_points_hal_selected = grid_points_hal[mask]
        
        # Final log
        final_selected_knots = np.sum(np.abs(self.theta_hat[1:]) > self.tol)
        if self.do_log:
            self.logger.info(f"Final: selected_knots={final_selected_knots}, iterations={it}")
        
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