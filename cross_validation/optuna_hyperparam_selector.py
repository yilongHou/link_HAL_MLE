"""
Optuna-based hyperparameter tuning for density estimation methods.

This script implements automated hyperparameter optimization using Optuna for various
density estimators. It uses cross-validation to evaluate hyperparameters on held-out
data to avoid overfitting.

Usage:
    uv run test_optuna.py --estimator CVXPYEstimator --n-trials 50 --cv-folds 3
    uv run test_optuna.py --estimator FISTAEstimator --n-trials 100 --metric bic

Example commands:
    uv run test_optuna.py --estimator CVXPYEstimator --n-trials 50
    uv run test_optuna.py --estimator FISTAEstimator --n-trials 50 --max-iter 10000
    uv run test_optuna.py --estimator ProximalNewtonEstimator --n-trials 50 --max-iter 500
    uv run test_optuna.py --estimator ProximalNewtonLBFGSFullEstimator --n-trials 50 --max-iter 10000
    uv run test_optuna.py --estimator AutoDiffEstimator --n-trials 50
    uv run test_optuna.py --estimator KDEEstimator --n-trials 50
    uv run test_optuna.py --estimator TrendFilteringADMMEstimator --n-trials 50
    uv run test_optuna.py --estimator LogSplinesEstimator --n-trials 1


Available estimators:
    - CVXPYEstimator
    - FISTAEstimator
    - ProximalNewtonEstimator
    - ProximalNewtonLBFGSFullEstimator
    - AutoDiffEstimator  
    - KDEEstimator
    - TrendFilteringADMMEstimator
    - LogSplinesEstimator (no hyperparameters - runs single evaluation)

Metrics:
    - sll: Sum Log-Likelihood (maximization)
    - bic: Bayesian Information Criterion (minimization)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from sklearn.model_selection import KFold
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Set random seed for reproducibility
np.random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Import estimators
from methods import (
    CVXPYEstimator,
    FISTAEstimator, 
    ProximalNewtonEstimator,
    ProximalNewtonLBFGSFullEstimator,
    AutoDiffEstimator,
    KDEEstimator,
    TrendFilteringADMMEstimator,
    TrendFilteringCVXPYEstimator,
    TrendFilteringCVXPYPP,
    TrendFilteringCVXPYPPA2Layered,
    # LogSplinesEstimator,
)

# Import samplers and utilities
from utils import TruncatedGMM
from utils.sampler.truncated_normal import TruncatedNormal
from utils.sampler.sinusoidal import Sinusoidal
from utils.sampler.step_function import StepFunction
from utils.plotting import plot_density

# Available samplers
SAMPLERS = {
    "TruncatedGMM": TruncatedGMM,
    "TruncatedNormal": TruncatedNormal,
    "Sinusoidal": Sinusoidal,
    "StepFunction": StepFunction,
}

# Available estimators for tuning
ESTIMATORS = {
    "CVXPYEstimator": CVXPYEstimator,
    "FISTAEstimator": FISTAEstimator,
    "ProximalNewtonEstimator": ProximalNewtonEstimator,
    "ProximalNewtonLBFGSFullEstimator": ProximalNewtonLBFGSFullEstimator,
    "AutoDiffEstimator": AutoDiffEstimator,
    "KDEEstimator": KDEEstimator,
    "TrendFilteringADMMEstimator": TrendFilteringADMMEstimator,
    "TrendFilteringCVXPYEstimator": TrendFilteringCVXPYEstimator,
    "TrendFilteringCVXPYPP": TrendFilteringCVXPYPP,
    "TrendFilteringCVXPYPPA2Layered": TrendFilteringCVXPYPPA2Layered,
    # "LogSplinesEstimator": LogSplinesEstimator,
}

# ThreePeak GMM setup (from the JSON file)
THREEPEAKS_SAMPLER_CONFIG = {
    "components": [
        {"mean": 0.2, "std": 0.05, "lower": 0, "upper": 1},
        {"mean": 0.5, "std": 0.05, "lower": 0, "upper": 1}, 
        {"mean": 0.8, "std": 0.05, "lower": 0, "upper": 1}
    ],
    "weights": [0.33, 0.34, 0.33]
}


class OptunaHyperparameterTuner:
    """
    Optuna-based hyperparameter tuner for density estimation methods.
    
    Uses cross-validation to evaluate hyperparameters on held-out validation sets
    to prevent overfitting and ensure robust parameter selection.
    """
    
    def __init__(
        self,
        estimator_name: str,
        data: pd.DataFrame,
        sampler_setup: Dict[str, Any],
        cv_folds: int = 3,
        metric: str = "sll",
        random_state: int = 42,
        max_iter: int = 50_000,
        tolerance: float = 1e-6,
        log_frequency: int = -1,  # No logging during CV
        n_grid_points: int = 200,
        fixed_params: Optional[Dict[str, Any]] = None,
        silent: bool = True,
        show_progress: bool = False,
    ):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            estimator_name: Name of the estimator to tune
            data: Full dataset for cross-validation
            sampler_setup: Complete sampler configuration (sampler name + params)
            cv_folds: Number of cross-validation folds
            metric: Optimization metric ('sll' or 'bic')
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for iterative estimators
            tolerance: Convergence tolerance
            log_frequency: Logging frequency (-1 = no logging)
            n_grid_points: Number of grid points for density evaluation
            silent: Whether to suppress all print statements
        """
        if estimator_name not in ESTIMATORS:
            raise ValueError(f"Estimator '{estimator_name}' not supported. "
                           f"Available: {list(ESTIMATORS.keys())}")
        
        if metric not in ["sll", "bic"]:
            raise ValueError(f"Metric '{metric}' not supported. Use 'sll' or 'bic'.")
        
        self.estimator_name = estimator_name
        self.estimator_class = ESTIMATORS[estimator_name]
        self.data = data
        self.sampler_setup = sampler_setup
        self.cv_folds = cv_folds
        self.metric = metric
        self.random_state = random_state
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.log_frequency = log_frequency
        self.n_grid_points = n_grid_points
        self.fixed_params = dict(fixed_params or {})
        self.silent = silent
        self.show_progress = show_progress
        
        # Cross-validation setup
        self.kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # Results storage
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_metric_value: Optional[float] = None
        
        # Create true sampler for density plotting
        sampler_name = sampler_setup['sampler']
        sampler_params = sampler_setup['sampler_params']
        
        if sampler_name not in SAMPLERS:
            raise ValueError(f"Sampler '{sampler_name}' not supported. "
                           f"Available: {list(SAMPLERS.keys())}")
        
        sampler_class = SAMPLERS[sampler_name]
        self.true_sampler = sampler_class(**sampler_params)
        
        if not silent:
            print(f"Initialized tuner for {estimator_name}")
            print(f"Dataset size: {len(data)} samples")
            print(f"Cross-validation: {cv_folds} folds")
            print(f"Optimization metric: {metric} ({'maximize' if metric == 'sll' else 'minimize'})")
    
    def _compute_log_likelihood_from_density(self, estimator: Any, validation_points: np.ndarray) -> float:
        """
        Compute average log-likelihood using the density output from any estimator.
        
        Args:
            estimator: Fitted estimator
            validation_points: Points to evaluate log-likelihood on
            
        Returns:
            Average log-likelihood
        """
        try:
            # Get density from the estimator
            grid_points, density_values = estimator.get_density()
            
            # Interpolate density at validation points
            # Create interpolator with bounds handling
            density_interp = interp1d(
                grid_points, 
                density_values, 
                kind="linear",
                bounds_error=False, 
                fill_value=(density_values[0], density_values[-1])
            )
            
            # Get interpolated density values
            interpolated_density = density_interp(validation_points)
            
            # Ensure densities are positive (avoid log of zero/negative)
            interpolated_density = np.maximum(interpolated_density, 1e-10)
            
            # Compute average log-likelihood
            log_likelihood = np.log(interpolated_density)
            avg_log_likelihood = np.mean(log_likelihood)
            
            return float(avg_log_likelihood)
            
        except Exception as e:
            if not self.silent:
                print(f"Warning: Failed to compute log-likelihood: {e}")
            return -np.inf
    
    def _compute_bic_from_density(self, estimator: Any, validation_data: pd.DataFrame) -> float:
        """
        Compute BIC using the density output from any estimator.
        
        Args:
            estimator: Fitted estimator
            validation_data: Validation dataset
            
        Returns:
            BIC value (lower is better)
        """
        try:
            # Get number of parameters from the estimator results
            results = estimator.get_results()
            
            # For HAL-based methods, count non-zero coefficients
            if 'theta_hat' in results and results['theta_hat'] is not None:
                # Count non-zero coefficients (excluding intercept)
                theta_hat = results['theta_hat']
                n_params = np.sum(np.abs(theta_hat[1:]) > 1e-8) + 1  # +1 for intercept
            elif 'n_selected_knots' in results:
                n_params = results['n_selected_knots'] + 1  # +1 for intercept
            else:
                # Fallback: estimate parameters from density complexity
                n_params = 10  # Conservative estimate
            
            # Compute log-likelihood
            validation_points = validation_data['W1'].values
            avg_log_likelihood = self._compute_log_likelihood_from_density(estimator, validation_points)
            
            # BIC = -2 * log_likelihood + k * log(n)
            n_samples = len(validation_data)
            sum_log_likelihood = avg_log_likelihood * n_samples
            bic = -2 * sum_log_likelihood + n_params * np.log(n_samples)
            
            return float(bic)
            
        except Exception as e:
            if not self.silent:
                print(f"Warning: Failed to compute BIC: {e}")
            return np.inf
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for the given estimator based on Optuna trial.
        
        Args:
            trial: Optuna trial object for parameter suggestion
            
        Returns:
            Dictionary of hyperparameters for the estimator
        """
        def _fixed(name: str) -> Tuple[bool, Any]:
            if name in self.fixed_params:
                return True, self.fixed_params[name]
            return False, None

        # Estimator-specific parameter ranges
        if self.estimator_name == "KDEEstimator":
            params = {
                "kernel": trial.suggest_categorical("kernel", ["gaussian", "tophat", "epanechnikov"]),
                "bandwidth": trial.suggest_float("bandwidth", 1e-3, 1e6, log=True),
            }
            
        elif self.estimator_name == "TrendFilteringADMMEstimator":
            k_is_fixed, k_val = _fixed("k")
            nc_is_fixed, nc_val = _fixed("norm_constraint")
            params = {
                "k": int(k_val) if k_is_fixed else trial.suggest_categorical("k", [0, 1, 2]),
                "norm_constraint": float(nc_val) if nc_is_fixed else trial.suggest_float("norm_constraint", 1e-3, 1e6, log=True),
                # "lam": trial.suggest_float("lam", 5e-2, 5e3, log=True),
                # "rho": trial.suggest_float("rho", 1e-3, 10.0, log=True),
                # "eps_pri": trial.suggest_float("eps_pri", 1e-6, 1e-2, log=True),
                # "eps_dual": trial.suggest_float("eps_dual", 1e-6, 1e-2, log=True),
            }
        elif self.estimator_name == "CVXPYEstimator":
            bo_is_fixed, bo_val = _fixed("basis_order")
            nc_is_fixed, nc_val = _fixed("norm_constraint")
            us_is_fixed, us_val = _fixed("use_secondary_solver")
            params = {
                "basis_order": int(bo_val) if bo_is_fixed else trial.suggest_categorical("basis_order", [0, 1, 2]),
                # "basis_order": trial.suggest_categorical("basis_order", [2]),
                "norm_constraint": float(nc_val) if nc_is_fixed else trial.suggest_float("norm_constraint", 1e-3, 1e6, log=True),
                # Default to allowing fallback (e.g., SCS) if MOSEK fails.
                "use_secondary_solver": bool(us_val) if us_is_fixed else True
            }
        elif self.estimator_name == "TrendFilteringCVXPYEstimator":
            k_is_fixed, k_val = _fixed("k")
            nc_is_fixed, nc_val = _fixed("norm_constraint")
            ng_is_fixed, ng_val = _fixed("n_grid_points")
            us_is_fixed, us_val = _fixed("use_secondary_solver")
            params = {
                "k": int(k_val) if k_is_fixed else trial.suggest_categorical("k", [0, 1, 2]),
                "norm_constraint": float(nc_val) if nc_is_fixed else trial.suggest_float("norm_constraint", 1e-3, 1e6, log=True),
                "n_grid_points": int(ng_val) if ng_is_fixed else self.n_grid_points,
                "use_secondary_solver": bool(us_val) if us_is_fixed else True
            }
        elif self.estimator_name == "TrendFilteringCVXPYPP":
            k_is_fixed, k_val = _fixed("k")
            nc_is_fixed, nc_val = _fixed("norm_constraint")
            ng_is_fixed, ng_val = _fixed("n_grid_points")
            us_is_fixed, us_val = _fixed("use_secondary_solver")
            params = {
                "k": int(k_val) if k_is_fixed else trial.suggest_categorical("k", [0, 1, 2]),
                "norm_constraint": float(nc_val) if nc_is_fixed else trial.suggest_float("norm_constraint", 1e-3, 1e6, log=True),
                "n_grid_points": int(ng_val) if ng_is_fixed else self.n_grid_points,
                "use_secondary_solver": bool(us_val) if us_is_fixed else True
            }
        elif self.estimator_name == "TrendFilteringCVXPYPPA2Layered":
            # Algorithm 2 layered constraints with data-adaptive knots
            k_is_fixed, k_val = _fixed("k")
            nc_is_fixed, nc_val = _fixed("norm_constraint")
            ng_is_fixed, ng_val = _fixed("n_grid_points")
            us_is_fixed, us_val = _fixed("use_secondary_solver")
            params = {
                "k": int(k_val) if k_is_fixed else trial.suggest_categorical("k", [0, 1, 2]),
                "norm_constraint": float(nc_val) if nc_is_fixed else trial.suggest_float("norm_constraint", 1e-3, 1e6, log=True),
                "n_grid_points": int(ng_val) if ng_is_fixed else self.n_grid_points,
                "use_secondary_solver": bool(us_val) if us_is_fixed else True
            }
            
        elif self.estimator_name == "FISTAEstimator":
            ni_is_fixed, ni_val = _fixed("n_iterations")
            lam_is_fixed, lam_val = _fixed("lam")
            bo_is_fixed, bo_val = _fixed("basis_order")
            l_is_fixed, l_val = _fixed("L")
            params = {
                "n_iterations": int(ni_val) if ni_is_fixed else self.max_iter,
                "lam": float(lam_val) if lam_is_fixed else trial.suggest_float("lam", 1e-4, 1e4, log=True),
                "basis_order": int(bo_val) if bo_is_fixed else trial.suggest_categorical("basis_order", [0, 1, 2]),
                "L": float(l_val) if l_is_fixed else trial.suggest_float("L", 1e0, 1e5, log=True),
            }
            
        elif self.estimator_name == "ProximalNewtonEstimator":
            ni_is_fixed, ni_val = _fixed("n_iterations")
            lam_is_fixed, lam_val = _fixed("lam")
            bo_is_fixed, bo_val = _fixed("basis_order")
            cs_is_fixed, cs_val = _fixed("cd_sweeps")
            c_is_fixed, c_val = _fixed("line_search_c")
            mls_is_fixed, mls_val = _fixed("max_line_search_steps")
            beta_is_fixed, beta_val = _fixed("line_search_beta")
            clip_is_fixed, clip_val = _fixed("non_desc_clip_alpha")
            hr_is_fixed, hr_val = _fixed("hessian_regularization")
            nd_is_fixed, nd_val = _fixed("non_descent_step_size")
            params = {
                "n_iterations": int(ni_val) if ni_is_fixed else self.max_iter,
                "lam": float(lam_val) if lam_is_fixed else trial.suggest_float("lam", 1e-4, 1.0, log=True),
                "basis_order": int(bo_val) if bo_is_fixed else trial.suggest_categorical("basis_order", [0, 1, 2]),
                "cd_sweeps": int(cs_val) if cs_is_fixed else trial.suggest_int("cd_sweeps", 1, 5),
                "line_search_c": float(c_val) if c_is_fixed else trial.suggest_float("line_search_c", 1e-6, 1e-1, log=True),
                "max_line_search_steps": int(mls_val) if mls_is_fixed else trial.suggest_int("max_line_search_steps", 10, 50),
                "line_search_beta": float(beta_val) if beta_is_fixed else trial.suggest_float("line_search_beta", 0.3, 0.8),
                "non_desc_clip_alpha": bool(clip_val) if clip_is_fixed else trial.suggest_categorical("non_desc_clip_alpha", [True, False]),
                "hessian_regularization": float(hr_val) if hr_is_fixed else trial.suggest_float("hessian_regularization", 1e-10, 1e-6, log=True),
                "non_descent_step_size": float(nd_val) if nd_is_fixed else trial.suggest_float("non_descent_step_size", 0.01, 0.5),
            }
            
        elif self.estimator_name == "ProximalNewtonLBFGSFullEstimator":
            ni_is_fixed, ni_val = _fixed("n_iterations")
            lam_is_fixed, lam_val = _fixed("lam")
            bo_is_fixed, bo_val = _fixed("basis_order")
            c_is_fixed, c_val = _fixed("line_search_c")
            mls_is_fixed, mls_val = _fixed("max_line_search_steps")
            mem_is_fixed, mem_val = _fixed("lbfgs_memory")
            clip_is_fixed, clip_val = _fixed("non_desc_clip_alpha")
            gmin_is_fixed, gmin_val = _fixed("lbfgs_gamma_clip_min")
            gmax_is_fixed, gmax_val = _fixed("lbfgs_gamma_clip_max")
            params = {
                "n_iterations": int(ni_val) if ni_is_fixed else self.max_iter,
                "lam": float(lam_val) if lam_is_fixed else trial.suggest_float("lam", 1e-4, 1.0, log=True),
                "basis_order": int(bo_val) if bo_is_fixed else trial.suggest_categorical("basis_order", [0, 1, 2]),
                "line_search_c": float(c_val) if c_is_fixed else trial.suggest_float("line_search_c", 1e-5, 1e-1, log=True),
                "max_line_search_steps": int(mls_val) if mls_is_fixed else trial.suggest_int("max_line_search_steps", 10, 50),
                "lbfgs_memory": int(mem_val) if mem_is_fixed else trial.suggest_int("lbfgs_memory", 3, 15),
                "non_desc_clip_alpha": bool(clip_val) if clip_is_fixed else trial.suggest_categorical("non_desc_clip_alpha", [True, False]),
                "lbfgs_gamma_clip_range": ( # Note: lbfgs_gamma_clip_range is a tuple, so we suggest both bounds
                    float(gmin_val) if gmin_is_fixed else trial.suggest_float("lbfgs_gamma_clip_min", 1e-5, 1e-1, log=True),
                    float(gmax_val) if gmax_is_fixed else trial.suggest_float("lbfgs_gamma_clip_max", 1e1, 1e5, log=True)
                ),
            }
            
        elif self.estimator_name == "AutoDiffEstimator": # Default to AdamW optimizer, not Muon
            ni_is_fixed, ni_val = _fixed("n_iterations")
            lam_is_fixed, lam_val = _fixed("lam")
            bo_is_fixed, bo_val = _fixed("basis_order")
            lr_is_fixed, lr_val = _fixed("learning_rate")
            wd_is_fixed, wd_val = _fixed("weight_decay")
            sp_is_fixed, sp_val = _fixed("scheduler_patience")
            sf_is_fixed, sf_val = _fixed("scheduler_factor")
            dev_is_fixed, dev_val = _fixed("device")
            params = {
                "n_iterations": int(ni_val) if ni_is_fixed else self.max_iter,
                "lam": float(lam_val) if lam_is_fixed else trial.suggest_float("lam", 1e-4, 1.0, log=True),
                "basis_order": int(bo_val) if bo_is_fixed else trial.suggest_categorical("basis_order", [0, 1, 2]),
                "learning_rate": float(lr_val) if lr_is_fixed else trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
                "weight_decay": float(wd_val) if wd_is_fixed else trial.suggest_float("weight_decay", 0.0, 1e-1, log=True),
                "scheduler_patience": int(sp_val) if sp_is_fixed else trial.suggest_int("scheduler_patience", 0, 1000),
                "scheduler_factor": float(sf_val) if sf_is_fixed else trial.suggest_float("scheduler_factor", 0.1, 0.9, step=0.1),
                "device": str(dev_val) if dev_is_fixed else "cpu"
            }
            
        elif self.estimator_name == "LogSplinesEstimator":
            # LogSplines has no hyperparameters to tune - use default configuration
            lb_is_fixed, lb_val = _fixed("lower_bound")
            ub_is_fixed, ub_val = _fixed("upper_bound")
            ng_is_fixed, ng_val = _fixed("n_grid_points")
            ir_is_fixed, ir_val = _fixed("install_r_package")
            params = {
                "lower_bound": float(lb_val) if lb_is_fixed else 0.0,
                "upper_bound": float(ub_val) if ub_is_fixed else 1.0,
                "n_grid_points": int(ng_val) if ng_is_fixed else self.n_grid_points,
                "install_r_package": bool(ir_val) if ir_is_fixed else True
            }
            
        else:
            raise ValueError(f"Hyperparameter ranges not defined for {self.estimator_name}")
        
        return params

    def _evaluate_params_cv(self, params: Dict[str, Any]) -> float:
        """
        Evaluate hyperparameters using cross-validation.
        
        Args:
            params: Hyperparameters to evaluate
            
        Returns:
            Average metric value across CV folds
        """
        cv_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.kfold.split(self.data)):
            try:
                # Split data
                train_data = self.data.iloc[train_idx].reset_index(drop=True)
                val_data = self.data.iloc[val_idx].reset_index(drop=True)
                
                # Create and fit estimator
                params_local = dict(params)
                # Always allow fallback for CVXPY-based estimators unless explicitly fixed otherwise.
                if (
                    "use_secondary_solver" in self.estimator_class.__init__.__code__.co_varnames
                    and "use_secondary_solver" not in self.fixed_params
                ):
                    params_local["use_secondary_solver"] = True
                estimator = self.estimator_class(**params_local)
                estimator.fit(train_data)
                
                # Compute metric on validation set using density-based approach
                if self.metric == "sll":
                    # Average Log-Likelihood (higher is better)
                    score = self._compute_log_likelihood_from_density(estimator, val_data['W1'].values)
                else:  # bic
                    # Bayesian Information Criterion (lower is better)
                    score = self._compute_bic_from_density(estimator, val_data)
                
                if np.isfinite(score):
                    cv_scores.append(score)
                else:
                    if not self.silent:
                        print(f"Warning: Non-finite score {score} in fold {fold_idx}")
                    
            except Exception as e:
                if not self.silent:
                    print(f"Warning: Error in fold {fold_idx}: {e}")
                continue
        
        if len(cv_scores) == 0:
            # If all folds failed, return worst possible score
            return -np.inf if self.metric == "sll" else np.inf
        
        return np.mean(cv_scores)
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Metric value to optimize (Optuna always minimizes)
        """
        # Suggest hyperparameters
        params = self._suggest_hyperparameters(trial)
        
        # Evaluate using cross-validation
        metric_value = self._evaluate_params_cv(params)
        
        # Optuna minimizes, so negate SLL (which we want to maximize)
        if self.metric == "sll":
            return -metric_value  # Convert maximization to minimization
        else:
            return metric_value  # BIC is already for minimization
    
    def optimize(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Maximum optimization time in seconds (None for no limit)
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary containing optimization results
        """
        # Special case for LogSplines - no hyperparameters to tune
        if self.estimator_name == "LogSplinesEstimator":
            if not self.silent:
                print(f"\nLogSplinesEstimator has no hyperparameters to tune.")
                print(f"Running single evaluation with default parameters...")
            n_trials = 1  # Force only 1 trial
        
        if not self.silent:
            print(f"\nStarting hyperparameter optimization...")
            print(f"Estimator: {self.estimator_name}")
            print(f"Trials: {n_trials}")
            print(f"Metric: {self.metric}")
        
        # Create Optuna study
        direction = "minimize"  # Always minimize (we negate SLL if needed)
        study_name = f"{self.estimator_name}_{self.metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        self.study = study
        
        # Run optimization with tqdm progress bar
        if self.show_progress and n_trials > 1:
            # Use tqdm progress bar for trials (only if more than 1 trial)
            with tqdm(total=n_trials, desc=f"Optimizing {self.estimator_name}", 
                     unit="trial", dynamic_ncols=True) as pbar:
                def callback(study, trial):
                    pbar.update(1)
                    # Update progress bar description with current best value
                    if study.best_value is not None:
                        best_display = f"{-study.best_value:.4f}" if self.metric == "sll" else f"{study.best_value:.4f}"
                        pbar.set_postfix_str(f"Best {self.metric.upper()}: {best_display}")
                
                study.optimize(
                    self._objective,
                    n_trials=n_trials,
                    timeout=timeout,
                    callbacks=[callback],
                    show_progress_bar=False  # Disable optuna's progress bar
                )
        else:
            # Run without progress bar
            study.optimize(
                self._objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=False
            )
        
        # Extract results
        assert self.study is not None
        # Build a *complete* best parameter dict (including any fixed / constant params)
        # by replaying the suggestion logic with a FixedTrial.
        self.best_params = self._suggest_hyperparameters(
            optuna.trial.FixedTrial(self.study.best_params.copy())
        )
        best_objective_value = self.study.best_value
        
        # Convert back to original metric scale
        if self.metric == "sll":
            self.best_metric_value = -best_objective_value  # Convert back from negated
        else:
            self.best_metric_value = best_objective_value
        
        if not self.silent:
            print(f"\nOptimization completed!")
            print(f"Best {self.metric.upper()}: {self.best_metric_value:.6f}")
            print(f"Best parameters:")
            for param, value in self.best_params.items():
                print(f"  {param}: {value}")
        
        return {
            "best_params": self.best_params,
            "best_metric_value": self.best_metric_value,
            "n_trials": len(self.study.trials),
            "study": self.study
        }
    
    def plot_optimization_results(self, save_dir: str = "local/optuna_plots") -> None:
        """
        Create and save optimization visualization plots.
        
        Args:
            save_dir: Directory to save plots
        """
        if self.study is None:
            raise ValueError("No optimization study found. Run optimize() first.")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Optimization history plot
        try:
            fig = plot_optimization_history(self.study)
            fig.update_layout(
                title=f"{self.estimator_name} - Optimization History ({self.metric.upper()})"
            )
            fname = f"{save_dir}/{self.estimator_name}_optimization_history.html"
            fig.write_html(fname)
            print(f"Saved optimization history to {fname}")
        except Exception as e:
            print(f"Warning: Could not create optimization history plot: {e}")
        
        # 2. Parameter importance plot
        try:
            if len(self.study.trials) > 10:  # Need sufficient trials
                fig = plot_param_importances(self.study)
                fig.update_layout(
                    title=f"{self.estimator_name} - Parameter Importance ({self.metric.upper()})"
                )
                fname = f"{save_dir}/{self.estimator_name}_param_importance.html"
                fig.write_html(fname)
                print(f"Saved parameter importance to {fname}")
        except Exception as e:
            print(f"Warning: Could not create parameter importance plot: {e}")
    
    def fit_best_model(self) -> Any:
        """
        Fit the estimator with the best found hyperparameters on the full dataset.
        
        Returns:
            Fitted estimator with best hyperparameters
        """
        if self.best_params is None:
            raise ValueError("No best parameters found. Run optimize() first.")
        
        if not self.silent:
            print(f"\nFitting best model on full dataset...")
            print(f"Parameters: {self.best_params}")
        
        # Create estimator with best params
        best_params = self.best_params.copy()
        # Ensure CVXPY-based estimators can fall back if the primary solver fails.
        # If user explicitly fixed use_secondary_solver, respect it; otherwise force fallback on.
        if (
            "use_secondary_solver" in self.estimator_class.__init__.__code__.co_varnames
            and "use_secondary_solver" not in self.fixed_params
        ):
            best_params["use_secondary_solver"] = True
        best_estimator = self.estimator_class(**best_params)
        
        # Fit on full data
        best_estimator.fit(self.data)
        
        if not self.silent:
            print("Best model fitted successfully!")
        return best_estimator
    
    def evaluate_best_model(
        self,
        best_estimator: Any,
        save_plot: bool = True,
        plot_dir: str = "local/optuna_plots"
    ) -> Dict[str, float]:
        """
        Evaluate and visualize the best model.
        
        Args:
            best_estimator: Fitted estimator with best hyperparameters
            save_plot: Whether to save the density plot
            plot_dir: Directory to save plots
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.silent:
            print(f"\nEvaluating best model...")
        
        # Get density estimation
        grid_points, estimated_density = best_estimator.get_density()
        
        # Compute metrics on full data using density-based approach
        sll = self._compute_log_likelihood_from_density(best_estimator, self.data['W1'].values)
        bic = self._compute_bic_from_density(best_estimator, self.data)
        
        # Get model info
        results = best_estimator.get_results()
        n_knots = results.get('n_selected_knots', 'N/A')
        
        evaluation_metrics = {
            'sum_log_likelihood': sll,
            'bic': bic,
            'n_selected_knots': n_knots
        }
        
        if not self.silent:
            print(f"Final model evaluation:")
            print(f"  Sum Log-Likelihood: {sll:.6f}")
            print(f"  BIC: {bic:.6f}")
            print(f"  Selected knots: {n_knots}")
        
        # Create visualization (only if not silent)
        if save_plot and not self.silent:
            os.makedirs(plot_dir, exist_ok=True)
            
            plt.figure(figsize=(12, 8))
            
            # Plot estimated density
            plt.plot(grid_points, estimated_density, 
                    color='#007acc', linewidth=2, label='Estimated Density')
            
            # Plot true density
            true_density = self.true_sampler.compute_density(grid_points)
            plt.plot(grid_points, true_density, 
                    color='#d62728', linestyle='--', linewidth=2, label='True Density')
            
            # Plot data histogram
            plt.hist(self.data['W1'], bins=50, density=True, alpha=0.6, 
                    label='Data Histogram', color='#2ca02c')
            
            # Formatting
            title = (f'{self.estimator_name} - Best Model\n'
                    f'{self.metric.upper()}={self.best_metric_value:.4f}, '
                    f'Knots={n_knots}')
            plt.title(title, fontsize=14)
            plt.xlabel('x', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_fname = f"{plot_dir}/{self.estimator_name}_best_model.png"
            plt.savefig(plot_fname, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Saved density plot to {plot_fname}")
        
        return evaluation_metrics
    
    def save_results(self, filepath: str) -> None:
        """
        Save optimization results to JSON file.
        
        Args:
            filepath: Path to save results JSON file
        """
        if self.study is None or self.best_params is None:
            raise ValueError("No optimization results to save. Run optimize() first.")
        
        # Prepare results dictionary
        results = {
            "estimator_name": self.estimator_name,
            "optimization_config": {
                "cv_folds": self.cv_folds,
                "metric": self.metric,
                "n_trials": len(self.study.trials),
                "random_state": self.random_state
            },
            "best_hyperparameters": self.best_params,
            "best_metric_value": self.best_metric_value,
            "sampler_setup": self.sampler_setup,
            "data_size": len(self.data),
            "optimization_timestamp": datetime.now().isoformat()
        }
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        if not self.silent:
            print(f"Results saved to {filepath}")

