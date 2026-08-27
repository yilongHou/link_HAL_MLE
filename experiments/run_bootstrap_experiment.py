"""
Run bootstrap experiments for nonparametric variance estimation.

This script takes a single experiment setup file and runs bootstrap resampling
to estimate the variance of the density estimator using quantile-based confidence intervals.
Each bootstrap sample includes:
1. Bootstrap resampling from the original dataset
2. Hyperparameter optimization using Optuna with cross-validation
3. Model fitting with optimal hyperparameters
4. Density estimation on a fixed grid
5. Results saving for nonparametric variance computation using empirical quantiles (2.5% and 97.5%)

Example usage:
    uv run python experiments/run_bootstrap_experiment.py \
        experiments/uniform_convergence/setups/TruncatedGMMFiveSpikes_CVXPYEstimator_N800.json \
        --B 1000 --n-workers 5
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Suppress optuna logging
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import samplers
from utils.sampler.truncated_gmm import TruncatedGMM
from utils.sampler.truncated_normal import TruncatedNormal
from utils.sampler.sinusoidal import Sinusoidal
from utils.sampler.step_function import StepFunction

SAMPLERS = {
    "TruncatedGMM": TruncatedGMM,
    "TruncatedNormal": TruncatedNormal,
    "Sinusoidal": Sinusoidal,
    "StepFunction": StepFunction,
}

# Import estimators
from methods import (
    CVXPYEstimator,
    FISTAEstimator,
    ProximalNewtonEstimator,
    ProximalNewtonLBFGSFullEstimator,
    AutoDiffEstimator,
    KDEEstimator,
    TrendFilteringADMMEstimator,
    # LogSplinesEstimator,
)

ESTIMATORS = {
    "CVXPYEstimator": CVXPYEstimator,
    "FISTAEstimator": FISTAEstimator,
    "ProximalNewtonEstimator": ProximalNewtonEstimator,
    "ProximalNewtonLBFGSFullEstimator": ProximalNewtonLBFGSFullEstimator,
    "AutoDiffEstimator": AutoDiffEstimator,
    "KDEEstimator": KDEEstimator,
    "TrendFilteringADMMEstimator": TrendFilteringADMMEstimator,
    # "LogSplinesEstimator": LogSplinesEstimator,
}

# Import hyperparameter tuner
from cross_validation.optuna_hyperparam_selector import OptunaHyperparameterTuner


def serialize_dict_to_json(data_dict: dict) -> dict:
    """Serialize dictionary with numpy arrays and pandas objects to JSON-compatible format."""
    serialized = {}
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        elif isinstance(value, np.integer):
            serialized[key] = int(value)
        elif isinstance(value, np.floating):
            serialized[key] = float(value)
        elif isinstance(value, pd.Series):
            serialized[key] = value.tolist()
        elif isinstance(value, pd.DataFrame):
            serialized[key] = value.to_dict(orient='list')
        else:
            serialized[key] = value
    return serialized


def bootstrap_resample(data: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """
    Bootstrap resample from the original data.
    
    Args:
        data: Original dataset
        random_state: Random seed for reproducibility
    
    Returns:
        Bootstrap resampled dataset
    """
    np.random.seed(random_state)
    n_samples = len(data)
    bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return data.iloc[bootstrap_indices].reset_index(drop=True)


def interpolate_to_fixed_grid(grid_points, density_values, 
                             eval_grid: np.ndarray) -> np.ndarray:
    """
    Interpolate density estimates to a fixed evaluation grid.
    
    Args:
        grid_points: Original grid points from estimator
        density_values: Density values at grid points
        eval_grid: Fixed evaluation grid (201 points from 0 to 1)
    
    Returns:
        Interpolated density values on fixed grid
    """
    # Convert to numpy arrays
    grid_points = np.array(grid_points)
    density_values = np.array(density_values)
    
    # Remove any NaN or invalid values
    valid_mask = ~(np.isnan(density_values) | np.isnan(grid_points))
    if not np.any(valid_mask):
        return np.full(len(eval_grid), np.nan)
    
    valid_grid = grid_points[valid_mask]
    valid_density = density_values[valid_mask]
    
    # Sort by grid points
    sort_idx = np.argsort(valid_grid)
    valid_grid = valid_grid[sort_idx]
    valid_density = valid_density[sort_idx]
    
    # Interpolate to fixed grid
    interpolated = np.interp(eval_grid, valid_grid, valid_density)
    
    return interpolated


def run_single_bootstrap(args_tuple: Tuple[Dict[str, Any], pd.DataFrame, int, str, str, 
                                          int, int, np.ndarray]) -> Dict[str, Any]:
    """
    Run a single bootstrap experiment.
    
    Args:
        args_tuple: Tuple containing (setup, original_data, bootstrap_seed, results_dir, 
                                    experiment_name, n_trials, cv_folds, eval_grid)
    
    Returns:
        Dictionary with bootstrap results or error info
    """
    setup, original_data, bootstrap_seed, results_dir, experiment_name, n_trials, cv_folds, eval_grid = args_tuple
    
    try:
        # Set random seed for this bootstrap
        np.random.seed(bootstrap_seed)
        
        # Bootstrap resample the data
        bootstrap_data = bootstrap_resample(original_data, bootstrap_seed)
        
        # Extract setup components
        sampler_setup = setup['sampler_setup']
        estimator_setup = setup['estimator_setup']
        estimator_name = estimator_setup['estimator']
        
        # Hyperparameter tuning on bootstrap sample
        tuner = OptunaHyperparameterTuner(
            estimator_name=estimator_name,
            data=bootstrap_data,
            sampler_setup=sampler_setup,
            cv_folds=cv_folds,
            metric="sll",
            random_state=bootstrap_seed,
            max_iter=50_000,
            silent=True  # Suppress all output
        )
        
        # Run hyperparameter optimization
        tuner.optimize(
            n_trials=n_trials,
            timeout=None,
            show_progress=False
        )
        
        # Fit best model
        best_estimator = tuner.fit_best_model()
        
        # Get density estimation results
        estimator_results = best_estimator.get_results()
        grid_points, estimated_density = best_estimator.get_density()
        
        # Interpolate to fixed evaluation grid
        density_on_eval_grid = interpolate_to_fixed_grid(grid_points, estimated_density, eval_grid)
        
        # Prepare bootstrap results
        bootstrap_results = {
            "bootstrap_seed": bootstrap_seed,
            "hyperparams": tuner.best_params,
            "eval_grid": eval_grid.tolist(),
            "density_on_eval_grid": density_on_eval_grid.tolist(),
            "original_grid_points": grid_points.tolist() if isinstance(grid_points, np.ndarray) else grid_points,
            "original_density": estimated_density.tolist() if isinstance(estimated_density, np.ndarray) else estimated_density,
            "HAL_results": serialize_dict_to_json(estimator_results),
            "success": True
        }
        
        # Save individual bootstrap result
        bootstrap_file = os.path.join(results_dir, f"bootstrap_{bootstrap_seed:03d}.json")
        os.makedirs(results_dir, exist_ok=True)
        with open(bootstrap_file, 'w') as f:
            json.dump(bootstrap_results, f, indent=2)
        
        return {
            "bootstrap_seed": bootstrap_seed,
            "success": True,
            "density_on_eval_grid": density_on_eval_grid
        }
        
    except Exception as e:
        # For failed bootstrap, save empty result
        try:
            bootstrap_file = os.path.join(results_dir, f"bootstrap_{bootstrap_seed:03d}.json")
            os.makedirs(results_dir, exist_ok=True)
            error_result = {
                "bootstrap_seed": bootstrap_seed,
                "success": False,
                "error": str(e),
                "eval_grid": eval_grid.tolist(),
                "density_on_eval_grid": [np.nan] * len(eval_grid)
            }
            with open(bootstrap_file, 'w') as f:
                json.dump(error_result, f, indent=2)
        except:
            pass
        
        return {
            "bootstrap_seed": bootstrap_seed,
            "success": False,
            "density_on_eval_grid": np.full(len(eval_grid), np.nan)
        }


def compute_bootstrap_variance_and_ci_around_original(successful_densities: np.ndarray, eval_grid: np.ndarray,
                                                     original_density: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Compute bootstrap standard errors and confidence intervals around the original point estimate.
    
    This function computes standard errors from the bootstrap distribution and creates
    confidence intervals centered around the original (true) point estimate, not the bootstrap mean.
    
    Args:
        successful_densities: Array of shape (n_successful_bootstrap, n_eval_points)
        eval_grid: Evaluation grid points
        original_density: Original point estimate density values
        alpha: Significance level for confidence intervals (default: 0.05 for 95% CI)
    
    Returns:
        Dictionary with bootstrap standard errors and CI bounds around original estimate
    """
    n_bootstrap, n_points = successful_densities.shape
    
    # Filter out any bootstrap samples that contain NaN values
    nan_mask = np.isnan(successful_densities).any(axis=1)
    if np.any(nan_mask):
        print(f"Warning: Found {np.sum(nan_mask)} bootstrap samples with NaN values, filtering them out...")
        successful_densities = successful_densities[~nan_mask]
        n_bootstrap = successful_densities.shape[0]
        print(f"After filtering NaN: {n_bootstrap} successful bootstrap samples")
        
        if n_bootstrap == 0:
            raise ValueError("No valid bootstrap samples after filtering NaN values")
    
    # Filter out bootstrap samples with problematic density estimates
    zero_threshold = 1e-8
    max_zero_fraction = 0.80  # Allow at most 80% of points to be zero
    
    # Count zero or near-zero densities for each bootstrap sample
    zero_counts = np.sum(successful_densities <= zero_threshold, axis=1)
    zero_fractions = zero_counts / n_points
    
    # Also check for samples with extremely low overall density
    mean_densities = np.mean(successful_densities, axis=1)
    min_mean_density = 0.1
    
    # Create combined mask for problematic samples
    problematic_mask = (zero_fractions > max_zero_fraction) | (mean_densities < min_mean_density)
    n_problematic = np.sum(problematic_mask)
    
    if n_problematic > 0:
        print(f"Warning: Found {n_problematic} bootstrap samples with problematic density estimates, filtering them out...")
        print(f"  - Samples with >{max_zero_fraction:.1%} zero densities: {np.sum(zero_fractions > max_zero_fraction)}")
        print(f"  - Samples with mean density <{min_mean_density}: {np.sum(mean_densities < min_mean_density)}")
        
        successful_densities = successful_densities[~problematic_mask]
        n_bootstrap = successful_densities.shape[0]
        print(f"After filtering problematic estimates: {n_bootstrap} valid bootstrap samples")
        
        if n_bootstrap == 0:
            raise ValueError("No valid bootstrap samples after filtering problematic estimates")
    
    # Compute bootstrap standard errors (standard deviation across bootstrap samples)
    bootstrap_se = np.nanstd(successful_densities, axis=0, ddof=1)
    
    # Create confidence intervals around the ORIGINAL point estimate
    z_critical = 1.96  # For 95% CI
    bootstrap_ci_lower = original_density - z_critical * bootstrap_se
    bootstrap_ci_upper = original_density + z_critical * bootstrap_se
    
    # Ensure CI bounds are non-negative (densities must be >= 0)
    bootstrap_ci_lower = np.maximum(bootstrap_ci_lower, 0)
    
    return {
        "eval_grid": eval_grid,
        "bootstrap_se": bootstrap_se,
        "bootstrap_ci_lower": bootstrap_ci_lower,
        "bootstrap_ci_upper": bootstrap_ci_upper,
        "n_bootstrap_successful": n_bootstrap,
        "confidence_level": 1 - alpha
    }


def compute_bootstrap_variance_and_ci(successful_densities: np.ndarray, eval_grid: np.ndarray,
                                     alpha: float = 0.05) -> Dict[str, Any]:
    """
    Compute bootstrap variance estimates and nonparametric confidence intervals using quantiles.
    
    For nonparametric variance bootstrap, we use the empirical quantiles from the bootstrap
    distribution rather than standard error-based intervals. This provides the 95% CI by
    taking the 2.5% and 97.5% percentiles directly from the bootstrap samples.
    
    Args:
        successful_densities: Array of shape (n_successful_bootstrap, n_eval_points)
        eval_grid: Evaluation grid points
        alpha: Significance level for confidence intervals (default: 0.05 for 95% CI)
    
    Returns:
        Dictionary with variance estimates and quantile-based CI bounds
    """
    n_bootstrap, n_points = successful_densities.shape
    
    # Filter out any bootstrap samples that contain NaN values
    nan_mask = np.isnan(successful_densities).any(axis=1)
    if np.any(nan_mask):
        print(f"Warning: Found {np.sum(nan_mask)} bootstrap samples with NaN values, filtering them out...")
        successful_densities = successful_densities[~nan_mask]
        n_bootstrap = successful_densities.shape[0]
        print(f"After filtering NaN: {n_bootstrap} successful bootstrap samples")
        
        if n_bootstrap == 0:
            raise ValueError("No valid bootstrap samples after filtering NaN values")
    
    # Filter out bootstrap samples with problematic density estimates (zero densities in key regions)
    # Check for samples that have large regions of zero density, which indicates failed estimation
    zero_threshold = 1e-8
    max_zero_fraction = 0.80  # Allow at most 80% of points to be zero
    
    # Count zero or near-zero densities for each bootstrap sample
    zero_counts = np.sum(successful_densities <= zero_threshold, axis=1)
    zero_fractions = zero_counts / n_points
    
    # Also check for samples with extremely low overall density (likely estimation failures)
    mean_densities = np.mean(successful_densities, axis=1)
    min_mean_density = 0.1  # Density should integrate to 1, so mean should be reasonable
    
    # Create combined mask for problematic samples
    problematic_mask = (zero_fractions > max_zero_fraction) | (mean_densities < min_mean_density)
    n_problematic = np.sum(problematic_mask)
    
    if n_problematic > 0:
        print(f"Warning: Found {n_problematic} bootstrap samples with problematic density estimates, filtering them out...")
        print(f"  - Samples with >{max_zero_fraction:.1%} zero densities: {np.sum(zero_fractions > max_zero_fraction)}")
        print(f"  - Samples with mean density <{min_mean_density}: {np.sum(mean_densities < min_mean_density)}")
        
        successful_densities = successful_densities[~problematic_mask]
        n_bootstrap = successful_densities.shape[0]
        print(f"After filtering problematic estimates: {n_bootstrap} valid bootstrap samples")
        
        if n_bootstrap == 0:
            raise ValueError("No valid bootstrap samples after filtering problematic estimates")
    
    # Compute statistics across bootstrap samples, using nanmean/nanvar for extra safety
    mean_density = np.nanmean(successful_densities, axis=0)
    variance_density = np.nanvar(successful_densities, axis=0, ddof=1)
    std_density = np.sqrt(variance_density)
    
    # Compute percentile-based confidence intervals using quantiles from bootstrap distribution
    # For nonparametric bootstrap, we use the empirical quantiles directly
    lower_percentile = (alpha / 2) * 100  # 2.5% for 95% CI
    upper_percentile = (1 - alpha / 2) * 100  # 97.5% for 95% CI
    
    ci_lower = np.nanpercentile(successful_densities, lower_percentile, axis=0)
    ci_upper = np.nanpercentile(successful_densities, upper_percentile, axis=0)
    
    return {
        "eval_grid": eval_grid,
        "mean_density": mean_density,
        "variance_density": variance_density,
        "std_density": std_density,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_bootstrap_successful": n_bootstrap,
        "confidence_level": 1 - alpha
    }


def run_bootstrap_experiment(setup_path: str, B: int = 100, n_workers: int = 7, 
                           n_trials: int = 50, cv_folds: int = 5) -> None:
    """
    Run bootstrap experiment for nonparametric variance estimation.
    
    This function:
    1. First fits the original data with cross-validation to get the true point estimate
    2. Then performs bootstrap resampling to estimate the standard error around this true estimate
    3. Computes both bootstrap-based and delta method confidence intervals for comparison
    
    Args:
        setup_path: Path to the experiment setup JSON file
        B: Number of bootstrap samples
        n_workers: Number of parallel workers
        n_trials: Number of Optuna trials for hyperparameter tuning
        cv_folds: Number of cross-validation folds
    """
    # Load setup
    with open(setup_path, 'r') as f:
        setup = json.load(f)
    
    # Extract experiment name from filename
    experiment_name = os.path.splitext(os.path.basename(setup_path))[0]
    base_dir = os.path.dirname(os.path.dirname(setup_path))
    
    results_dir = os.path.join(base_dir, 'bootstrap_variance_estimation', 'results', experiment_name)
    
    print(f"Running bootstrap experiment: {experiment_name}")
    print(f"Number of bootstrap samples: {B}")
    print(f"Parallel workers: {n_workers}")
    print(f"Results directory: {results_dir}")
    
    # Generate original dataset once
    sampler_setup = setup['sampler_setup']
    estimator_setup = setup['estimator_setup']
    sampler_name = sampler_setup['sampler']
    sampler_params = sampler_setup['sampler_params']
    n_samples = sampler_setup['n_samples']
    
    # Use seed 42 for original data generation
    original_seed = 42
    np.random.seed(original_seed)
    
    sampler_class = SAMPLERS[sampler_name]
    sampler = sampler_class(**sampler_params)
    original_data = pd.DataFrame({'W1': sampler.generate_samples(n_samples)})
    
    print(f"Generated original dataset with {n_samples} samples using seed {original_seed}")
    
    # Create fixed evaluation grid (201 points from 0 to 1)
    eval_grid = np.linspace(0.0, 1.0, 201)
    
    # STEP 1: Fit the original data to get the TRUE POINT ESTIMATE
    print("\n" + "="*60)
    print("STEP 1: Computing TRUE POINT ESTIMATE from original data")
    print("="*60)
    
    estimator_name = estimator_setup['estimator']
    
    # Run hyperparameter optimization on original data
    print("Running hyperparameter optimization on original data...")
    original_tuner = OptunaHyperparameterTuner(
        estimator_name=estimator_name,
        data=original_data,
        sampler_setup=sampler_setup,
        cv_folds=cv_folds,
        metric="sll",
        random_state=original_seed,
        max_iter=50_000,
        silent=True
    )
    
    original_tuner.optimize(
        n_trials=n_trials,
        timeout=None,
        show_progress=True  # Show progress for original fit
    )
    
    # Fit best model on original data
    print("Fitting best model on original data...")
    original_estimator = original_tuner.fit_best_model()
    original_results = original_estimator.get_results()
    original_grid_points, original_density = original_estimator.get_density()
    
    # Interpolate original density to fixed grid
    original_density_on_grid = interpolate_to_fixed_grid(original_grid_points, original_density, eval_grid)
    
    print(f"Original point estimate computed successfully!")
    print(f"Best hyperparameters: {original_tuner.best_params}")
    
    # STEP 2: Compute delta method confidence intervals
    print("\n" + "="*60)
    print("STEP 2: Computing DELTA METHOD confidence intervals")
    print("="*60)
    
    try:
        # Import delta method functions
        from density_variance.density_variance import estimate_covariance_beta, density_confidence_interval
        
        # Adapt result structure for density variance functions
        best_params = original_tuner.best_params or {}
        adapted_results = {
            'results': original_results,
            'estimator_setup': {
                'estimator_params': {
                    'basis_order': best_params.get('basis_order', 0)
                }
            }
        }
        
        # Compute covariance matrix and confidence intervals
        print("Computing covariance matrix...")
        cov_beta = estimate_covariance_beta(original_data, adapted_results, ridge_param=1e-6)
        
        print("Computing delta method confidence intervals...")
        ci_df = density_confidence_interval(eval_grid, adapted_results, cov_beta, alpha=0.05)
        
        delta_method_results = {
            'eval_grid': eval_grid.tolist(),
            'density': ci_df['density'].values.tolist(),
            'ci_lower': ci_df['lower'].values.tolist(),
            'ci_upper': ci_df['upper'].values.tolist(),
            'se': ci_df['se'].values.tolist(),
            'success': True
        }
        print("Delta method confidence intervals computed successfully!")
        
    except Exception as e:
        print(f"Error computing delta method CI: {e}")
        delta_method_results = {'success': False, 'error': str(e)}
    
    # STEP 3: Run bootstrap experiments
    print("\n" + "="*60)
    print("STEP 3: Running BOOTSTRAP experiments for standard error estimation")
    print("="*60)
    
    # STEP 3: Run bootstrap experiments
    print("\n" + "="*60)
    print("STEP 3: Running BOOTSTRAP experiments for standard error estimation")
    print("="*60)
    
    # Generate bootstrap seeds: [42, 43, 44, ..., 41+B]
    bootstrap_seeds = list(range(42, 42 + B))
    
    # Prepare arguments for multiprocessing
    args_list = []
    for seed in bootstrap_seeds:
        if os.path.exists(os.path.join(results_dir, f"bootstrap_{seed:03d}.json")):
            continue  # Skip if already computed
        args_list.append((setup, original_data, seed, results_dir, experiment_name, 
                         n_trials, cv_folds, eval_grid))
    
    print(f"Running {len(args_list)} bootstrap samples (skipping {B - len(args_list)} existing results)")
    
    # Run bootstrap experiments in parallel
    successful_results = []
    failed_results = []
    
    with Pool(n_workers) as pool:
        with tqdm(total=len(args_list), desc="Running bootstrap samples") as pbar:
            for result in pool.imap(run_single_bootstrap, args_list):
                if result["success"]:
                    successful_results.append(result)
                else:
                    failed_results.append(result)
                pbar.update(1)
    
    # Also collect any pre-existing successful results
    for seed in bootstrap_seeds:
        bootstrap_file = os.path.join(results_dir, f"bootstrap_{seed:03d}.json")
        if os.path.exists(bootstrap_file) and seed not in [r["bootstrap_seed"] for r in successful_results + failed_results]:
            try:
                with open(bootstrap_file, 'r') as f:
                    existing_result = json.load(f)
                if existing_result.get("success", False):
                    density_on_grid = np.array(existing_result["density_on_eval_grid"])
                    successful_results.append({
                        "bootstrap_seed": seed,
                        "success": True,
                        "density_on_eval_grid": density_on_grid
                    })
                else:
                    failed_results.append({
                        "bootstrap_seed": seed,
                        "success": False,
                        "density_on_eval_grid": np.full(len(eval_grid), np.nan)
                    })
            except:
                failed_results.append({
                    "bootstrap_seed": seed,
                    "success": False,
                    "density_on_eval_grid": np.full(len(eval_grid), np.nan)
                })
    
    n_successful = len(successful_results)
    n_failed = len(failed_results)
    
    print(f"\nBootstrap experiment completed!")
    print(f"Successful bootstrap samples: {n_successful}")
    print(f"Failed bootstrap samples: {n_failed}")
    
    if n_successful == 0:
        print("No successful bootstrap samples. Cannot compute variance estimates.")
        return
    
    print(f"Success rate: {n_successful/(n_successful + n_failed)*100:.1f}%")
    
    # STEP 4: Compute bootstrap-based confidence intervals using original point estimate
    print("\n" + "="*60)
    print("STEP 4: Computing BOOTSTRAP-based confidence intervals")
    print("="*60)
    
    # Stack successful density estimates
    successful_densities = np.array([r["density_on_eval_grid"] for r in successful_results])
    successful_seeds = [r["bootstrap_seed"] for r in successful_results]
    
    # Compute bootstrap standard errors around the original point estimate
    bootstrap_stats = compute_bootstrap_variance_and_ci_around_original(
        successful_densities, eval_grid, original_density_on_grid
    )
    
    # Track which seeds were actually used in the final computation
    n_used = bootstrap_stats["n_bootstrap_successful"]
    if n_used < len(successful_results):
        n_filtered = len(successful_results) - n_used
        print(f"Note: {n_filtered} bootstrap samples were filtered out due to problematic density estimates")
        print(f"Final analysis uses {n_used} valid bootstrap samples")
    
    # Compute true density for comparison (if possible)
    try:
        true_density = sampler.compute_density(eval_grid)
        bootstrap_stats["true_density"] = true_density
    except Exception as e:
        print(f"Could not compute true density: {e}")
        bootstrap_stats["true_density"] = None
    
    # STEP 5: Save comprehensive results
    print("\n" + "="*60)
    print("STEP 5: Saving comprehensive results")
    print("="*60)
    
    # Save bootstrap summary with both bootstrap and delta method results
    summary_result = {
        "experiment_setup": setup,
        "bootstrap_config": {
            "B": B,
            "n_workers": n_workers,
            "n_trials": n_trials,
            "cv_folds": cv_folds,
            "bootstrap_seeds": bootstrap_seeds,
            "original_seed": original_seed
        },
        "original_point_estimate": {
            "eval_grid": eval_grid.tolist(),
            "density": original_density_on_grid.tolist(),
            "hyperparams": original_tuner.best_params or {},
            "grid_points": original_grid_points.tolist() if isinstance(original_grid_points, np.ndarray) else original_grid_points,
            "original_density": original_density.tolist() if isinstance(original_density, np.ndarray) else original_density,
            "HAL_results": serialize_dict_to_json(original_results)
        },
        "delta_method_ci": delta_method_results if delta_method_results['success'] else None,
        "bootstrap_summary": {
            "n_successful": n_successful,
            "n_failed": n_failed,
            "success_rate": n_successful/(n_successful + n_failed),
            "successful_seeds": [r["bootstrap_seed"] for r in successful_results],
            "failed_seeds": [r["bootstrap_seed"] for r in failed_results]
        },
        "variance_estimates": {
            "eval_grid": bootstrap_stats["eval_grid"].tolist(),
            "bootstrap_se": bootstrap_stats["bootstrap_se"].tolist(),
            "bootstrap_ci_lower": bootstrap_stats["bootstrap_ci_lower"].tolist(),
            "bootstrap_ci_upper": bootstrap_stats["bootstrap_ci_upper"].tolist(),
            "confidence_level": bootstrap_stats["confidence_level"],
            "true_density": bootstrap_stats["true_density"].tolist() if bootstrap_stats["true_density"] is not None else None
        }
    }
    
    summary_file = os.path.join(results_dir, "bootstrap_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_result, f, indent=2)
    
    print(f"Saved comprehensive bootstrap summary to: {summary_file}")
    print(f"Mean bootstrap standard error: {np.mean(bootstrap_stats['bootstrap_se']):.6f}")
    print(f"Max bootstrap standard error: {np.max(bootstrap_stats['bootstrap_se']):.6f}")
    
    if delta_method_results['success']:
        print(f"Mean delta method standard error: {np.mean(delta_method_results['se']):.6f}")
        print(f"Max delta method standard error: {np.max(delta_method_results['se']):.6f}")
    
    print("\nBootstrap experiment completed successfully!")
    print("="*60)


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run bootstrap experiments for variance estimation"
    )
    parser.add_argument(
        "setup_path",
        type=str,
        help="Path to the experiment setup JSON file"
    )
    parser.add_argument(
        "--B",
        type=int,
        default=100,
        help="Number of bootstrap samples (default: 100)"
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=7,
        help="Number of parallel workers (default: 7)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for hyperparameter tuning (default: 50)"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.setup_path):
        print(f"Error: Setup file not found: {args.setup_path}")
        sys.exit(1)
    
    run_bootstrap_experiment(args.setup_path, args.B, args.n_workers, args.n_trials, args.cv_folds)


if __name__ == "__main__":
    main()