import numpy as np
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple, Union
import logging
import os

from utils.basis import create_basis_functions
from methods.base_estimator import BaseEstimator
from .optim import SingleDeviceMuon, SingleDeviceMuonWithAuxAdam


class HALLinearModel(nn.Module):
    """
    Simple linear model for HAL basis functions.
    
    This is essentially a linear layer that computes φ(x)ᵀθ where:
    - φ(x) are the HAL basis functions
    - θ are the learnable coefficients
    """
    
    def __init__(self, input_dim: int):
        """
        Initialize the linear model.
        
        Args:
            input_dim: Number of basis functions (K)
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)
        
        # Initialize weights to zero (common for HAL)
        nn.init.zeros_(self.linear.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute φ(x)ᵀθ
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        return self.linear(x).squeeze(-1)  # Remove last dimension to get (batch_size,)


class AutoDiffEstimator(BaseEstimator):
    """
    HAL-MLE density estimator using deep learning optimization.
    
    This class frames the HAL-MLE optimization problem as a deep learning problem:
    
    min_θ  Loss(θ) = -∑ᵢ φ(xᵢ)ᵀθ + N * log(∫ exp(φ(u)ᵀθ) du) + λ‖θ[1:]‖₁
    
    The model is a simple linear layer that maps basis functions to log-density values.
    It can be used with various PyTorch optimizers like Adam, SGD, or Muon.
    """
    
    def __init__(
        self,
        lam: float = 3.0,
        n_iterations: int = 100000,
        tol: float = 1e-6,
        n_grid_points: int = 200,
        basis_order: int = 0,
        optimizer: str = "AdamW",
        learning_rate: float = 0.01,
        muon_momentum: float = 0.95,
        muon_ns_steps: int = 5,
        muon_nesterov: bool = True,
        weight_decay: float = 0.0,
        scheduler: Optional[str] = "ReduceLROnPlateau",
        scheduler_patience: int = 50,
        scheduler_factor: float = 0.5,
        device: str = "auto",
        log_dir: str = "./local/logs/experiment.log",
        log_frequency: int = 1000,
    ):
        """
        Initialize AutoDiff HAL estimator.
        
        Args:
            lam: L1 regularization parameter
            n_iterations: Maximum number of optimization steps (epochs)
            tol: Convergence tolerance
            n_grid_points: Number of grid points for density evaluation
            basis_order: Order of the truncated power basis
            optimizer: Optimizer type ("Adam", "SGD", "AdamW", "RMSprop", "Muon", "MuonWithAuxAdam")
            learning_rate: Learning rate for optimizer
            muon_momentum: Momentum for MUON optimizer (default 0.95)
            muon_ns_steps: Newton-Schulz steps for MUON (default 5)
            muon_nesterov: Whether to use Nesterov momentum for MUON (default True)
            weight_decay: L2 regularization (for optimizer)
            scheduler: Learning rate scheduler ("ReduceLROnPlateau", "StepLR", None)
            scheduler_patience: Patience for ReduceLROnPlateau scheduler
            scheduler_factor: Factor for learning rate reduction
            device: Device to use ("auto", "cpu", "cuda", "mps")
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
        
        # MUON-specific parameters
        self.n_grid_points = n_grid_points
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.muon_momentum = muon_momentum
        self.muon_ns_steps = muon_ns_steps
        self.muon_nesterov = muon_nesterov
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        
        # Set device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Will be set during fitting
        self.model: Optional[HALLinearModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.loss_history: list = []
        self.phi_data: Optional[torch.Tensor] = None
        self.phi_grid: Optional[torch.Tensor] = None
        self.log_delta_j: Optional[torch.Tensor] = None
        self.n_samples: Optional[int] = None
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.model is None:
            raise ValueError("Model must be initialized before creating optimizer")
            
        if self.optimizer_name.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        elif self.optimizer_name.lower() == "rmsprop":
            return optim.RMSprop(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "muon":
            # Use SingleDeviceMuon for 2D parameters only
            # For HAL, we have a single linear layer with 2D weight matrix
            matrix_params = [p for p in self.model.parameters() if p.ndim >= 2]
            if not matrix_params:
                raise ValueError("No 2D parameters found for MUON optimizer")
            return SingleDeviceMuon(
                matrix_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.muon_momentum
            )
        elif self.optimizer_name.lower() == "muonwithauxadam":
            # Use MuonWithAuxAdam with parameter groups
            matrix_params = [p for p in self.model.parameters() if p.ndim >= 2]
            scalar_params = [p for p in self.model.parameters() if p.ndim < 2]
            
            param_groups = []
            if matrix_params:
                param_groups.append({
                    'params': matrix_params,
                    'use_muon': True,
                    'lr': self.learning_rate,
                    'weight_decay': self.weight_decay,
                    'momentum': self.muon_momentum
                })
            if scalar_params:
                param_groups.append({
                    'params': scalar_params,
                    'use_muon': False,
                    'lr': self.learning_rate,
                    'weight_decay': self.weight_decay,
                    'betas': (0.9, 0.999),
                    'eps': 1e-8
                })
            
            if not param_groups:
                raise ValueError("No parameters found for MuonWithAuxAdam optimizer")
            return SingleDeviceMuonWithAuxAdam(param_groups)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        if self.scheduler_name is None or self.optimizer is None:
            return None
        elif self.scheduler_name.lower() == "reducelronplateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.scheduler_patience,
                factor=self.scheduler_factor
            )
        elif self.scheduler_name.lower() == "steplr":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.scheduler_patience,
                gamma=self.scheduler_factor
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_name}")
    
    def _compute_loss(self) -> torch.Tensor:
        """
        Compute the HAL-MLE loss function.
        
        Returns:
            Total loss value
        """
        if self.model is None:
            raise ValueError("Model not initialized")
            
        # Data term: -∑ᵢ φ(xᵢ)ᵀθ
        log_f_data = self.model(self.phi_data)  # (N,)
        data_term = -torch.sum(log_f_data)
        
        # Normalization term: N * log(∫ exp(φ(u)ᵀθ) du)
        log_f_grid = self.model(self.phi_grid)  # (m,)
        # Add log(delta_j) for proper Riemann sum
        log_terms = log_f_grid + self.log_delta_j
        log_integral = torch.logsumexp(log_terms, dim=0)
        normalization_term = self.n_samples * log_integral
        
        # L1 regularization term: λ‖θ[1:]‖₁ (exclude intercept)
        weights = self.model.linear.weight.squeeze()  # (K,)
        l1_penalty = self.lam * torch.norm(weights[1:], p=1)
        
        total_loss = data_term + normalization_term + l1_penalty
        
        return total_loss
    
    def _log_progress(self, iteration: int, loss: float, lr: float):
        """Log training progress."""
        if self.do_log and (iteration % self.log_frequency == 0 or iteration == 0):
            # Calculate number of selected knots (non-zero coefficients excluding intercept)
            if self.model is not None:
                with torch.no_grad():
                    weights = self.model.linear.weight.squeeze().cpu().numpy()
                    num_knots = np.sum(np.abs(weights[1:]) > 1e-4)  # Exclude intercept
                    l1_norm = np.sum(np.abs(weights[1:]))
            else:
                num_knots = 0
                l1_norm = 0.0
            
            # Format the log message to match other estimators' format
            self.logger.info(f"Iter {iteration:4d}: obj={-loss:.6f}, change=0.00e+00, α=1.000, ‖θ[1:]‖₁={l1_norm:.3f}, lr={lr:.3e}, num_knots={num_knots}")
    
    def fit(
        self,
        data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
        validation_frequency: int = -1
    ) -> 'AutoDiffEstimator':
        """
        Fit the AutoDiff HAL density estimator.
        
        Args:
            data: DataFrame with column 'W1' containing the observations
            validation_data: Optional validation data for monitoring
            validation_frequency: Frequency for validation logging
            
        Returns:
            Self for method chaining
        """
        self.n_samples = len(data)
        
        # 1) Create HAL basis functions on observed data
        grid_points_hal = np.unique(data['W1'].dropna())
        b_ik, basis_names = create_basis_functions(data, grid_points_hal, order=self.basis_order)
        self.basis_names = list(basis_names)  # Ensure it's a list for type checking
        self._grid_points_hal = grid_points_hal
        
        # 2) Create midpoint grid for normalization integral
        grid_eval = np.linspace(0, 1, self.n_grid_points)
        midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2
        delta_j = grid_eval[1:] - grid_eval[:-1]
        
        df_mid = pd.DataFrame({'W1': midpoints})
        b_jk, _ = create_basis_functions(df_mid, grid_points_hal, order=self.basis_order)
        
        # 3) Convert to PyTorch tensors and move to device
        self.phi_data = b_ik.to(self.device)  # (N, K)
        self.phi_grid = b_jk.to(self.device)  # (m, K)
        self.log_delta_j = torch.log(torch.from_numpy(delta_j).float()).to(self.device)  # (m,)
        
        # 4) Initialize model
        K = self.phi_data.shape[1]
        self.model = HALLinearModel(K).to(self.device)
        
        # 5) Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        print(f"Starting AutoDiff optimization with K={K} parameters on device={self.device}")
        
        if self.do_log:
            self.logger.info(f"Starting AutoDiff HAL-MLE with {K} basis functions")
            self.logger.info(f"Device: {self.device}, Optimizer: {self.optimizer_name}, LR: {self.learning_rate}")
        
        # 6) Training loop
        self.loss_history = []
        prev_loss = float('inf')
        
        for iteration in range(self.n_iterations):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            loss = self._compute_loss()
            
            # Early stopping if objective function explodes
            loss_value = loss.item()
            if self._check_objective_explosion(loss_value, iteration):
                break
            
            # Backward pass
            loss.backward()
            
            # Optimization step
            self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(loss)
                else:
                    self.scheduler.step()
            
            # Store loss and log progress
            loss_value = loss.item()
            self.loss_history.append(loss_value)
            current_lr = self.optimizer.param_groups[0]['lr']
            self._log_progress(iteration, loss_value, current_lr)
            
            # Print progress to console (similar to ProximalNewton)
            if iteration % self.log_frequency == 0:
                with torch.no_grad():
                    weights = self.model.linear.weight.squeeze().cpu().numpy()
                    num_knots = np.sum(np.abs(weights[1:]) > 1e-4)  # Exclude intercept
                    l1_norm = np.sum(np.abs(weights[1:]))
                print(f"Iter {iteration:3d}: obj={-loss_value:.6f}, change=0.00e+00, "
                      f"α=1.000, ‖θ[1:]‖₁={l1_norm:.3f}, lr={current_lr:.3e}, num_knots={num_knots}")
            
            # Check convergence
            # if abs(prev_loss - loss_value) < self.tol:
            #     if self.do_log:
            #         self.logger.info(f"Converged at iteration {iteration}")
            #     break
            
            # prev_loss = loss_value
        
        # 7) Extract results and move to CPU
        with torch.no_grad():
            self.theta_hat = self.model.linear.weight.squeeze().cpu().numpy()
        
        # 8) Apply coefficient pruning following HAL methodology
        # Set coefficients with absolute value < 10^-4 to 0
        pruning_threshold = 1e-4
        theta_pruned = self.theta_hat.copy()
        
        # Count coefficients before pruning
        non_zero_before = np.sum(np.abs(theta_pruned[1:]) > pruning_threshold)
        
        # Apply pruning to all coefficients except intercept
        if self.basis_order == 0:
            # For order 0: theta = [intercept, step_functions...]
            theta_pruned[1:] = np.where(np.abs(theta_pruned[1:]) < pruning_threshold, 0.0, theta_pruned[1:])
        else:
            # For order k≥1: theta = [intercept, x, x^2, ..., x^k, (x-ξ₁)₊^k, ...]
            theta_pruned[1:] = np.where(np.abs(theta_pruned[1:]) < pruning_threshold, 0.0, theta_pruned[1:])
        
        # Count coefficients after pruning
        non_zero_after = np.sum(np.abs(theta_pruned[1:]) > 0)
        pruned_count = non_zero_before - non_zero_after
        
        if self.do_log:
            self.logger.info(f"Coefficient pruning: {pruned_count} coefficients set to zero (threshold={pruning_threshold})")
            self.logger.info(f"Non-zero coefficients: {non_zero_before} → {non_zero_after}")
        
        # 9) Renormalize the density to ensure it integrates to 1
        # Create basis functions at midpoints for normalization
        df_mid = pd.DataFrame({'W1': midpoints})
        b_mid, _ = create_basis_functions(df_mid, grid_points_hal, order=self.basis_order)
        phi_mid = b_mid.numpy()
        
        # Compute log density at midpoints with pruned coefficients
        log_f_mid = phi_mid @ theta_pruned
        max_log_f_val = np.max(log_f_mid)
        
        # Compute normalization constant
        Z = np.sum(np.exp(log_f_mid - max_log_f_val) * delta_j)
        
        # Adjust intercept to ensure density integrates to 1
        intercept_adjustment = -max_log_f_val - np.log(Z)
        theta_pruned[0] = theta_pruned[0] + intercept_adjustment
        
        if self.do_log:
            self.logger.info(f"Intercept adjusted by {intercept_adjustment:.6f} for renormalization")
        
        # Store the pruned and renormalized coefficients
        self.theta_hat = theta_pruned
        
        # 10) Store standard attributes
        self.grid_midpoints = midpoints
        self.delta_j = delta_j
        
        # Select non-zero knots (for HAL basis functions, excluding polynomial terms)
        if self.basis_order == 0:
            truncated_power_coeffs = self.theta_hat[1:]  # Exclude intercept
        else:
            truncated_power_coeffs = self.theta_hat[self.basis_order + 1:]  # Exclude intercept + polynomials
        
        mask = np.abs(truncated_power_coeffs) > self.tol
        self.grid_points_hal_selected = grid_points_hal[mask]
        
        # Create evaluation grid for density
        self.grid_points = np.linspace(0, 1, 200)
        
        final_loss = self.loss_history[-1] if self.loss_history else float('inf')
        final_selected_knots = np.sum(np.abs(self.theta_hat[1:]) > pruning_threshold)
        
        if self.do_log:
            self.logger.info(f"Final loss: {final_loss:.6f}")
            self.logger.info(f"Coefficient pruning applied: threshold={pruning_threshold}")
            self.logger.info(f"Selected knots after pruning: {final_selected_knots}")
        
        self.is_fitted = True
        
        # Store fitted theta as dictionary for inspection
        if self.basis_names is not None:
            assert len(self.basis_names) == len(self.theta_hat), "Basis names count does not match theta_hat length"
            self.fitted_theta_dict = {name: value for name, value in zip(self.basis_names, self.theta_hat.tolist())}
        else:
            self.fitted_theta_dict = {}
        
        return self
    
    def get_density(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the estimated density on the evaluation grid.
        
        Returns:
            Tuple of (grid_points, density_values)
            
        Raises:
            ValueError: If the estimator hasn't been fitted yet
        """
        if not self.is_fitted or self.theta_hat is None:
            raise ValueError("Estimator must be fitted before getting density. Call fit() first.")
        
        if self._grid_points_hal is None:
            raise ValueError("Grid points not initialized")
        
        if self.grid_midpoints is None or self.delta_j is None:
            raise ValueError("Grid attributes not initialized")
        
        # Create evaluation grid
        grid_points = np.linspace(0, 1, self.n_grid_points)
        
        # Create DataFrame for evaluation points
        df_eval = pd.DataFrame({'W1': grid_points})
        
        # Create basis functions at evaluation points
        b_eval, _ = create_basis_functions(df_eval, self._grid_points_hal, order=self.basis_order)
        phi_eval = b_eval.numpy()  # (n_eval, K)
        
        # Compute log density
        log_density = phi_eval @ self.theta_hat
        
        # Since coefficients are already pruned and renormalized, we can directly 
        # compute the density without additional normalization
        density = np.exp(log_density)
        
        # Final check: ensure density integrates to 1 (should already be satisfied)
        dx = grid_points[1] - grid_points[0]
        integral = np.sum(density * dx)
        if not np.isclose(integral, 1.0, rtol=1e-6):
            if self.do_log:
                self.logger.warning(f"Density integral is {integral:.6f}, renormalizing")
            density /= integral
        
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
        
        # Get common results from base class
        results = self._get_common_results()
        
        # Add AutoDiff-specific results
        results.update({
            "loss_history": self.loss_history,
            "final_loss": self.loss_history[-1] if self.loss_history else None,
            "optimizer": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "device": str(self.device),
            "convergence_iterations": len(self.loss_history),
        })
        
        return results