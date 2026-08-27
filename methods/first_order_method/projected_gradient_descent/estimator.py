import numpy as np
import torch
import pandas as pd
from typing import Dict, Optional, Tuple
from utils.basis import create_basis_functions, project_onto_l1_ball
from methods.base_estimator import BaseEstimator


class ProjectedGDEstimator(BaseEstimator):
    """
    Density estimator using Projected Gradient Descent (with L1 ball projection).
    
    This class implements projected gradient descent for solving the constrained
    density estimation problem:
    
    min_θ  −ℓ(θ)  subject to  ‖θ[1:]‖₁ ≤ norm_constraint
    
    where ℓ is the HAL-basis log-likelihood and θ[0] is the intercept (unconstrained).
    """
    
    def __init__(
        self,
        norm_constraint: float = 8.0,
        learning_rate: float = 1e-2,
        n_iterations: int = 30000,
        weight_decay: float = 1e-2,
        tol: float = 1e-4,
        n_grid_points: int = 200,
        basis_order: int = 0,
        log_dir: str = "./local/logs/experiment.log",
        log_frequency: int = 500
    ):
        """
        Initialize Projected GD estimator.
        
        Args:
            norm_constraint: L1 ball radius for projection
            learning_rate: Learning rate for Adam optimizer
            n_iterations: Maximum number of iterations
            weight_decay: Weight decay for Adam optimizer
            tol: tol for pruning small coefficients
            n_grid_points: Number of grid points for density evaluation
            basis_order: Order of the truncated power basis
            log_dir: Directory for logging
            log_frequency: Frequency of logging (-1 means no logging)
        """
        # Initialize base class
        super().__init__(
            lam=norm_constraint,  # Use norm_constraint as lam for base class
            n_iterations=n_iterations,
            tol=tol,
            basis_order=basis_order,
            log_dir=log_dir,
            log_frequency=log_frequency
        )
        
        # ProjectedGD-specific parameters
        self.norm_constraint = norm_constraint
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.tol = tol
        self.n_grid_points = n_grid_points
        
        # Will be set during fitting
        self.theta_final: Optional[np.ndarray] = None
        self.theta_pruned: Optional[np.ndarray] = None
        self.grid_points_hal_selected: Optional[np.ndarray] = None
        
    def fit(self, data: pd.DataFrame, warm_start_coefficients: Optional[np.ndarray] = None,
            validation_data: Optional[pd.DataFrame] = None,
            validation_frequency: int = -1) -> 'ProjectedGDEstimator':
        """
        Fit the Projected GD density estimator.
        
        Args:
            data: DataFrame with column 'W1' containing the observations
            warm_start_coefficients: Optional initial coefficients for warm starting
            validation_data: Optional validation data for tracking performance
            validation_frequency: Frequency for validation logging (default -1 means no validation)
            
        Returns:
            Self for method chaining
        """
        n_samples = len(data)
        
        # Use unique observed data points as grid for basis functions
        grid_points_hal = np.unique(data['W1'].dropna())
        
        # Build basis for the data points
        b_ik, basis_names = create_basis_functions(data, grid_points_hal, order=self.basis_order)
        self.basis_names = basis_names

        # Create evaluation grid and corresponding basis functions
        grid_eval = np.linspace(0, 1, self.n_grid_points)
        grid_midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2
        data_grid = pd.DataFrame({'W1': grid_midpoints})
        b_jk, _ = create_basis_functions(data_grid, grid_points_hal, order=self.basis_order)
        
        delta_j = grid_eval[1:] - grid_eval[:-1]
        log_delta_j = torch.tensor(np.log(delta_j), dtype=torch.float32)
        
        # Initialise θ
        K = b_ik.shape[1]  # number of parameters
        
        if warm_start_coefficients is None:
            theta = torch.zeros(K, requires_grad=True)
        elif len(warm_start_coefficients) == K:
            theta = torch.tensor(warm_start_coefficients, dtype=torch.float32, requires_grad=True)
        else:
            print(f"Warm start failed: expected {K} coefficients, got {len(warm_start_coefficients)}\n\n")
            theta = torch.zeros(K, requires_grad=True)
        
        optimizer = torch.optim.Adam(
            [theta], 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        print(f"Starting ProjectedGD with K={K} parameters")
        
        # Simple initialization log
        if self.do_log:
            self.logger.info(f"ProjectedGD: norm_constraint={self.norm_constraint}, lr={self.learning_rate}, n_grid={self.n_grid_points}, basis_order={self.basis_order}, n_samples={n_samples}, K={K}")
        
        for it in range(self.n_iterations):
            optimizer.zero_grad()
            # Compute linear predictors for data and grid evaluation
            
            # First term: sum over data points: -(b_ik @ theta)
            first_term = -torch.sum(b_ik @ theta)
            
            # For the evaluation grid:
            log_density_grid = b_jk @ theta
            log_terms = log_delta_j + log_density_grid
            log_Z = torch.logsumexp(log_terms, dim=0)
            
            loss = first_term + n_samples * log_Z
            
            # Early stopping if objective function explodes
            if self._check_objective_explosion(loss.item(), it):
                break
            
            loss.backward()
            optimizer.step()
            
            # Intercept correction for exact normalization
            with torch.no_grad():
                delta_j_tensor = torch.tensor(delta_j, dtype=torch.float32)
                logZ = torch.log(torch.sum(torch.exp(b_jk @ theta) * delta_j_tensor))
                if not torch.isfinite(logZ):
                    print(f"Warning: logZ became {logZ.item()} at iteration {it}, stopping optimization"); break
                theta[0] -= logZ
            
            # Project theta[1:] onto the L1 ball of radius norm_constraint (exclude intercept)
            with torch.no_grad():
                theta[1:] = project_onto_l1_ball(theta[1:], z=self.norm_constraint)
            
            # Validation and logging
            if validation_data is not None and validation_frequency > 0 and it % validation_frequency == 0:
                # Update parameters for validation
                self.theta_hat = theta.detach().clone().numpy()
                self.grid_midpoints = grid_midpoints
                self.delta_j = delta_j
                self._grid_points_hal = grid_points_hal
                
                validation_pts = validation_data['W1'].values
                validation_sum_log_likelihood = self.get_sum_log_likelihood_for_points(validation_pts)
                if self.do_log:
                    self.logger.info(f"Validation at iter {it}: sum_log_likelihood={validation_sum_log_likelihood:.6f}")
                print(f"Validation at iter {it}: sum_log_likelihood={validation_sum_log_likelihood:.6f}")
            
            if it % self.log_frequency == 0:
                l1_norm = torch.sum(torch.abs(theta[1:])).item()
                num_selected_knots = torch.sum(torch.abs(theta[1:]) > self.tol).item()
                loss_val = loss.item()
                
                if self.do_log:
                    self.logger.info(f"Iter {it}: loss={loss_val:.4f}, ‖θ[1:]‖₁={l1_norm:.3f}, num_selected_knots={num_selected_knots}")
                
                print(f"Iteration {it}: loss={loss_val:.4f}, ‖θ[1:]‖₁={l1_norm:.3f}, num_selected_knots={num_selected_knots}")
        
        # After optimization, we can prune small coefficients
        self.theta_final = theta.detach().clone().numpy()
        self.theta_pruned = np.copy(self.theta_final)
        self.theta_hat = self.theta_final  # Add for compatibility with test script
        self.theta_pruned[1:] = np.where(
            np.abs(self.theta_final[1:]) > self.tol, 
            self.theta_final[1:], 
            0
        )
        
        # Identify nonzero indices and map them correctly to grid points
        non_zero_indices = np.nonzero(self.theta_pruned)[0]
        
        # For basis functions, we need to separate:
        # - Index 0: intercept (always included)
        # - Indices 1 to basis_order: polynomial terms (don't correspond to grid points)
        # - Indices (basis_order+1) onwards: truncated power basis (correspond to grid_points_hal)
        
        if self.basis_order == 0:
            # All non-zero indices (except intercept) correspond to grid points
            truncated_power_indices = non_zero_indices[non_zero_indices > 0]
            grid_point_indices = truncated_power_indices - 1  # Map to grid_points_hal indices
        else:
            # Only indices > basis_order correspond to grid points
            truncated_power_indices = non_zero_indices[non_zero_indices > self.basis_order]
            grid_point_indices = truncated_power_indices - (self.basis_order + 1)  # Map to grid_points_hal indices
        
        # Select grid points corresponding to non-zero truncated power coefficients
        self.grid_points_hal_selected = (
            grid_points_hal[grid_point_indices] 
            if len(grid_point_indices) > 0 
            else np.array([])
        )
        
        # Incorporate the grid points corresponding to nonzero coefficients into the evaluation grid
        grid_eval = np.sort(np.unique(np.concatenate((grid_eval, self.grid_points_hal_selected))))
        self.grid_midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2
        self.delta_j = grid_eval[1:] - grid_eval[:-1]
        
        # Store grid points for density computation
        self._grid_points_hal = grid_points_hal
        self._non_zero_indices = non_zero_indices
        
        # Final log
        final_selected_knots = len(self.grid_points_hal_selected) if self.grid_points_hal_selected is not None else 0
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
        
        # Rebuild basis functions for the final evaluation grid
        # using all original HAL knot points (self._grid_points_hal)
        data_grid = pd.DataFrame({'W1': self.grid_midpoints})
        basis_grid_tensor, _ = create_basis_functions(data_grid, self._grid_points_hal, order=self.basis_order)
        b_jk_full = basis_grid_tensor.numpy()  # Shape (num_midpoints, K)
        
        # self.theta_pruned has shape (K,) and contains the coefficients 
        # (including intercept, polynomial, and knot-based terms, with some potentially zeroed out).
        # The columns of b_jk_full should align with the elements of self.theta_pruned.
        estimated_log_density = b_jk_full @ self.theta_pruned
        
        estimated_density = np.exp(estimated_log_density)
        # Ensure density integrates to 1 over the grid
        Z = np.sum(estimated_density * self.delta_j)
        density_hat = estimated_density / Z

        # make sure density sums to 1
        density_hat /= np.sum(density_hat * self.delta_j)
        
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
