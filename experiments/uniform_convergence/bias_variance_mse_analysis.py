#!/usr/bin/env python3
"""
Bias-Variance-MSE Analysis for Four Density Estimation Methods

This script compares four density estimation methods (HAL-MLE, KDE, TF, LogSplines)
across 6 DGPs by computing pointwise bias, variance, and MSE at 21 evaluation points.
"""


"""
HAL-MLE Results:
- experiments/uniform_convergence/results/Sinusoidal_CVXPYEstimator_N800
- experiments/uniform_convergence/results/StepFunction_CVXPYEstimator_N800
- experiments/uniform_convergence/results/TruncatedGMMAsymmetricThree_CVXPYEstimator_N800
- experiments/uniform_convergence/results/TruncatedGMMFiveSpikes_CVXPYEstimator_N800
- experiments/uniform_convergence/results/TruncatedGMMSymmetricThree_CVXPYEstimator_N800
- experiments/uniform_convergence/results/TruncatedNormal_CVXPYEstimator_N800

KDE Results:
- experiments/uniform_convergence/results/Sinusoidal_KDEEstimator_N800
- experiments/uniform_convergence/results/StepFunction_KDEEstimator_N800
- experiments/uniform_convergence/results/TruncatedGMMAsymmetricThree_KDEEstimator_N800
- experiments/uniform_convergence/results/TruncatedGMMFiveSpikes_KDEEstimator_N800
- experiments/uniform_convergence/results/TruncatedGMMSymmetricThree_KDEEstimator_N800
- experiments/uniform_convergence/results/TruncatedNormal_KDEEstimator_N800

TF Results: 
- experiments/uniform_convergence/results/Sinusoidal_TrendFilteringADMMEstimator_N800
- experiments/uniform_convergence/results/StepFunction_TrendFilteringADMMEstimator_N800
- experiments/uniform_convergence/results/TruncatedGMMAsymmetricThree_TrendFilteringADMMEstimator_N800
- experiments/uniform_convergence/results/TruncatedGMMFiveSpikes_TrendFilteringADMMEstimator_N800
- experiments/uniform_convergence/results/TruncatedGMMSymmetricThree_TrendFilteringADMMEstimator_N800
- experiments/uniform_convergence/results/TruncatedNormal_TrendFilteringADMMEstimator_N800

LogSpline Results:
- experiments/uniform_convergence/results/Sinusoidal_LogSplinesEstimator_N800
- experiments/uniform_convergence/results/StepFunction_LogSplinesEstimator_N800
- experiments/uniform_convergence/results/TruncatedGMMAsymmetricThree_LogSplinesEstimator_N800
- experiments/uniform_convergence/results/TruncatedGMMFiveSpikes_LogSplinesEstimator_N800
- experiments/uniform_convergence/results/TruncatedGMMSymmetricThree_LogSplinesEstimator_N800
- experiments/uniform_convergence/results/TruncatedNormal_LogSplinesEstimator_N800
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
# Optional pretty-name mapping for DGPs (was historically provided as a JSON file).
# Fall back to identity mapping if the file is absent.
_mapping_candidates = [
    os.path.join(_HERE, "dpg-name-mapping.json"),
    os.path.join(_HERE, "dgp-name-mapping.json"),
]
for _p in _mapping_candidates:
    if os.path.exists(_p):
        with open(_p, "r") as f:
            DGP_NAME_MAPPING = json.load(f)
        break
else:
    DGP_NAME_MAPPING = {}

warnings.filterwarnings('ignore')

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import samplers for true density computation
from utils import (
    TruncatedNormal,
    TruncatedGMM,
    Sinusoidal,
    StepFunction,
)

# Define samplers mapping
SAMPLERS = {
    "TruncatedNormal": TruncatedNormal,
    "TruncatedGMM": TruncatedGMM,
    "Sinusoidal": Sinusoidal,
    "StepFunction": StepFunction,
}

# Define DGP configurations used in experiments
DGP_CONFIGS = {
    "TruncatedNormal": {
        "sampler": "TruncatedNormal",
        "sampler_params": {
            "mean": 0.5,
            "std": 0.1,
            "lower": 0,
            "upper": 1
        }
    },
    "TruncatedGMMSymmetricThree": {
        "sampler": "TruncatedGMM",
        "sampler_params": {
            "components": [
                {"mean": 0.2, "std": 0.05, "lower": 0, "upper": 1},
                {"mean": 0.5, "std": 0.05, "lower": 0, "upper": 1},
                {"mean": 0.8, "std": 0.05, "lower": 0, "upper": 1}
            ],
            "weights": [0.33, 0.34, 0.33]
        }
    },
    "TruncatedGMMAsymmetricThree": {
        "sampler": "TruncatedGMM",
        "sampler_params": {
            "components": [
                {"mean": 0.35, "std": 0.1, "lower": 0.0, "upper": 1.0},
                {"mean": 0.65, "std": 0.05, "lower": 0.0, "upper": 1.0},
                {"mean": 0.9, "std": 0.2, "lower": 0.0, "upper": 1.0}
            ],
            "weights": [0.4, 0.4, 0.2]
        }
    },
    "TruncatedGMMFiveSpikes": {
        "sampler": "TruncatedGMM",
        "sampler_params": {
            "components": [
                {"mean": 0.45, "std": 0.005, "lower": 0, "upper": 1},
                {"mean": 0.475, "std": 0.005, "lower": 0, "upper": 1},
                {"mean": 0.5, "std": 0.005, "lower": 0, "upper": 1},
                {"mean": 0.525, "std": 0.005, "lower": 0, "upper": 1},
                {"mean": 0.55, "std": 0.005, "lower": 0, "upper": 1},
                {"mean": 0.5, "std": 0.05, "lower": 0, "upper": 1}
            ],
            "weights": [0.06666667, 0.06666667, 0.06666667, 0.06666667, 0.06666667, 0.66666667]
        }
    },
    "StepFunction": {
        "sampler": "StepFunction",
        "sampler_params": {
            "level1": 1.0,
            "level2": 0.5,
            "breakpoint": 0.7
        }
    },
    "Sinusoidal": {
        "sampler": "Sinusoidal",
        "sampler_params": {}
    }
}

# Method mapping
METHODS = {
    "HAL-MLE": "CVXPYEstimator",
    # Alias used in some writeups/plots.
    "HAL": "CVXPYEstimator",
    "KDE": "KDEEstimator",
    "TF": "TrendFilteringADMMEstimator",
    "TFPP": "TrendFilteringCVXPYPP",
    # Trend filtering Algorithm 2 (layered PPA2) results.
    "TFA2": "TrendFilteringCVXPYPPA2Layered",
    # Pseudo-method: uses TFPP files but adjusts/renormalizes densities on a common grid
    # before evaluating at the bias/variance/MSE evaluation points.
    "TFPP-adjusted": "TrendFilteringCVXPYPP",
    "LogSplines": "LogSplinesEstimator",
}

def compute_true_density(dgp_name, eval_points):
    """
    Compute true density for a given DGP at evaluation points.
    """
    if dgp_name not in DGP_CONFIGS:
        raise ValueError(f"Unknown DGP: {dgp_name}")
    
    config = DGP_CONFIGS[dgp_name]
    sampler_class = SAMPLERS[config["sampler"]]
    sampler = sampler_class(**config["sampler_params"])
    
    return sampler.compute_density(eval_points)

def _adjust_density_on_common_grid(
    gridpoints,
    density,
    *,
    common_grid_n: int = 2000,
):
    """
    Adjust a (gridpoints, density) curve by:
      - clamping density to >= 0
      - interpolating to a common grid on [0,1]
      - renormalizing so integral over [0,1] is 1

    This matches the adjustment used by TF_TFPP_HAL overlay plots.
    Returns (common_x, adjusted_density_on_common_x).
    """
    grid_x = np.asarray(gridpoints, dtype=float).ravel()
    grid_f = np.asarray(density, dtype=float).ravel()
    if grid_x.shape != grid_f.shape:
        raise ValueError(f"gridpoints/density shape mismatch: {grid_x.shape} vs {grid_f.shape}")

    # Sort + clamp
    order = np.argsort(grid_x)
    grid_x = grid_x[order]
    grid_f = np.maximum(grid_f[order], 0.0)

    common_x = np.linspace(0.0, 1.0, int(common_grid_n))
    # Endpoint fill outside grid range
    common_f = np.interp(common_x, grid_x, grid_f, left=float(grid_f[0]), right=float(grid_f[-1]))
    common_f = np.maximum(common_f, 0.0)
    area = float(np.trapz(common_f, common_x))
    if area > 0:
        common_f = common_f / area
    return common_x, common_f

def load_method_results(
    method_name,
    dgp_name,
    sample_size=800,
    results_dir="experiments/uniform_convergence/results",
    *,
    method_dir_suffix_map=None,
    density_key_map=None,
):
    """
    Load all results for one method-DGP combination.
    
    Args:
        method_name: One of the method names from METHODS keys
        dgp_name: DGP name
        sample_size: Sample size (default 800)
        results_dir: Directory containing results
    
    Returns:
        list: List of density estimates from all experiments
    """
    method_code = METHODS[method_name]
    method_dir_suffix_map = method_dir_suffix_map or {}
    density_key_map = density_key_map or {}

    dir_suffix = method_dir_suffix_map.get(method_name, "")
    result_dir = os.path.join(results_dir, f"{dgp_name}_{method_code}_N{sample_size}{dir_suffix}")
    
    if not os.path.exists(result_dir):
        print(f"Warning: Directory not found: {result_dir}")
        return []
    
    results = []
    json_files = [f for f in os.listdir(result_dir) if f.endswith('.json')]
    
    print(f"Loading {len(json_files)} files for {method_name} on {dgp_name}")
    
    for json_file in tqdm(json_files, desc=f"{method_name}-{dgp_name}"):
        file_path = os.path.join(result_dir, json_file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract density estimates
            density_key = density_key_map.get(method_name, "estimated_density")
            if density_key in data and 'density' in data[density_key]:
                density_values = np.array(data[density_key]['density'], dtype=float)
                gridpoints = np.array(data[density_key]['gridpoints'], dtype=float)

                # Robustness: if saved density has non-finite values (often from exp overflow),
                # attempt to reconstruct a stable piecewise density from HAL_results when possible.
                if (not np.all(np.isfinite(gridpoints))) or (not np.all(np.isfinite(density_values))):
                    hr = data.get("HAL_results", {}) if isinstance(data, dict) else {}
                    theta_hat = hr.get("theta_hat")
                    bin_widths = hr.get("bin_widths")
                    knots = hr.get("knots")
                    if theta_hat is not None and bin_widths is not None and knots is not None:
                        try:
                            theta = np.asarray(theta_hat, dtype=float).ravel()
                            dx = np.asarray(bin_widths, dtype=float).ravel()
                            kts = np.asarray(knots, dtype=float).ravel()
                            if theta.size != dx.size or dx.size != (kts.size - 1):
                                raise ValueError("shape mismatch in theta/bin_widths/knots")
                            log_terms = theta + np.log(dx)
                            m = float(np.max(log_terms))
                            logZ = m + float(np.log(np.sum(np.exp(log_terms - m))))
                            log_levels = theta - logZ
                            bin_idx = np.searchsorted(kts, gridpoints, side="right") - 1
                            bin_idx = np.clip(bin_idx, 0, theta.size - 1)
                            density_values = np.exp(log_levels[bin_idx])
                            if (not np.all(np.isfinite(density_values))) or (not np.all(np.isfinite(gridpoints))):
                                raise ValueError("reconstructed density still non-finite")
                            print(
                                f"Warning: Non-finite saved density in {json_file} (key={density_key}); "
                                "reconstructed density from HAL_results."
                            )
                        except Exception:
                            print(f"Warning: Non-finite density/gridpoints in {json_file} (key={density_key}); skipping.")
                            continue
                    else:
                        print(f"Warning: Non-finite density/gridpoints in {json_file} (key={density_key}); skipping.")
                        continue
                # Parse seed from filename like seed_123.json when possible
                seed = None
                if json_file.startswith("seed_") and json_file.endswith(".json"):
                    try:
                        seed = int(json_file.split("_")[1].split(".")[0])
                    except Exception:
                        seed = None
                
                results.append({
                    'density': density_values,
                    'gridpoints': gridpoints,
                    'file': json_file,
                    'seed': seed,
                    'max_density': float(np.max(density_values)),
                })
            else:
                print(f"Warning: No density data found in {json_file} (key={density_key})")
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return results

def interpolate_density_to_eval_points(density_values, gridpoints, eval_points):
    """
    Interpolate density estimates to common evaluation points.
    
    Args:
        density_values: Array of density values
        gridpoints: Array of gridpoints where density was estimated
        eval_points: Target evaluation points
    
    Returns:
        Array of interpolated density values at eval_points
    """
    # Create interpolation function
    f_interp = interp1d(gridpoints, density_values, kind='linear', 
                       bounds_error=False, fill_value=0.0)
    
    return f_interp(eval_points)

def compute_bias_variance_mse(all_density_estimates, true_density, eval_points):
    """
    Compute bias, variance, and MSE at each evaluation point.
    
    Args:
        all_density_estimates: List of density estimates (2D array: n_experiments x n_eval_points)
        true_density: True density values at eval_points
        eval_points: Evaluation points
    
    Returns:
        dict: Contains bias_squared, variance, mse arrays
    """
    # Convert to numpy array if not already
    density_matrix = np.array(all_density_estimates)  # Shape: (n_experiments, n_eval_points)
    
    # Compute empirical mean across experiments
    empirical_mean = np.mean(density_matrix, axis=0)  # Shape: (n_eval_points,)
    
    # Compute bias (not squared)
    bias = empirical_mean - true_density
    bias_squared = bias**2
    
    # Compute variance
    variance = np.var(density_matrix, axis=0, ddof=1)  # Shape: (n_eval_points,)
    
    # Compute MSE
    mse = bias_squared + variance
    
    return {
        'bias': bias,  # Return bias, not bias_squared
        'bias_squared': bias_squared,
        'variance': variance,
        'mse': mse,
        'empirical_mean': empirical_mean,
        'n_experiments': density_matrix.shape[0]
    }

def analyze_dgp(
    dgp_name,
    eval_points,
    sample_size=800,
    results_dir="experiments/uniform_convergence/results",
    method_names=None,
    *,
    method_dir_suffix_map=None,
    density_key_map=None,
    exclude_max_density_gt=None,
):
    """
    Analyze all methods for a single DGP.
    
    Args:
        dgp_name: Name of the DGP
        eval_points: Evaluation points
        sample_size: Sample size to analyze
        results_dir: Directory containing results
    
    Returns:
        dict: Analysis results for all methods
    """
    print(f"\n{'='*60}")
    print(f"Analyzing DGP: {DGP_NAME_MAPPING.get(dgp_name, dgp_name)}")
    print(f"{'='*60}")
    
    # Compute true density
    true_density = compute_true_density(dgp_name, eval_points)
    
    results = {
        'dgp_name': dgp_name,
        'eval_points': eval_points,
        'true_density': true_density,
        'methods': {}
    }
    
    # Analyze each method
    if method_names is None:
        method_names = list(METHODS.keys())

    # First load all requested methods (so we can optionally filter to a common seed set)
    per_method_results = {}
    for method_name in method_names:
        if method_name not in METHODS:
            print(f"Warning: Unknown method '{method_name}', skipping.")
            continue
        print(f"\nProcessing method: {method_name}")

        method_results = load_method_results(
            method_name,
            dgp_name,
            sample_size,
            results_dir,
            method_dir_suffix_map=method_dir_suffix_map,
            density_key_map=density_key_map,
        )
        if not method_results:
            print(f"No results found for {method_name} on {DGP_NAME_MAPPING.get(dgp_name, dgp_name)}")
            continue
        per_method_results[method_name] = method_results

    if not per_method_results:
        return results

    # Optional filtering: exclude seeds with max grid density > threshold and
    # keep only the intersection of remaining seeds across methods.
    if exclude_max_density_gt is not None:
        thr = float(exclude_max_density_gt)
        kept_sets = []
        for method_name, method_results in per_method_results.items():
            with_seed = [r for r in method_results if r.get("seed") is not None]
            if len(with_seed) != len(method_results):
                print(
                    f"Warning: {method_name} has {len(method_results) - len(with_seed)} files without parseable seed; "
                    "excluding them for filtering."
                )
            kept = {r["seed"] for r in with_seed if float(r.get("max_density", np.inf)) <= thr}
            kept_sets.append(kept)
            print(f"{method_name}: keeping {len(kept)}/{len(with_seed)} seeds with max_density <= {thr:g}")

        common_kept = set.intersection(*kept_sets) if kept_sets else set()
        print(f"Common kept seeds across methods: {len(common_kept)}")
        for method_name in list(per_method_results.keys()):
            per_method_results[method_name] = [r for r in per_method_results[method_name] if r.get("seed") in common_kept]
            if not per_method_results[method_name]:
                print(f"Warning: after filtering, no results remain for {method_name} on {dgp_name}")

    # Now compute metrics for each method using (optionally) filtered results
    for method_name, method_results in per_method_results.items():
        # Interpolate all density estimates to common evaluation points
        interpolated_densities = []
        for result in method_results:
            # Special-case: TFPP-adjusted uses a common-grid adjustment/renormalization first.
            if method_name == "TFPP-adjusted":
                cx, cf = _adjust_density_on_common_grid(
                    result["gridpoints"], result["density"], common_grid_n=2000
                )
                interpolated = interpolate_density_to_eval_points(cf, cx, eval_points)
            else:
                interpolated = interpolate_density_to_eval_points(
                    result['density'], result['gridpoints'], eval_points
                )
            interpolated_densities.append(interpolated)

        bias_var_mse = compute_bias_variance_mse(interpolated_densities, true_density, eval_points)
        results['methods'][method_name] = bias_var_mse

        print(f"  - Loaded {len(method_results)} experiments")
        print(f"  - Mean bias: {np.mean(bias_var_mse['bias']):.6f}")
        print(f"  - Mean variance: {np.mean(bias_var_mse['variance']):.6f}")
        print(f"  - Mean MSE: {np.mean(bias_var_mse['mse']):.6f}")
    
    return results

def create_dgp_comparison_plot(
    dgp_results,
    save_dir="paper/resources/density_bias_variance_mse_analysis",
    filename_suffix="",
    sample_size=800,
):
    """
    Create 3-panel plot for one DGP comparing all 4 methods.
    
    Args:
        dgp_results: Results dictionary for one DGP
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    dgp_name = dgp_results['dgp_name']
    eval_points = dgp_results['eval_points']
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    
    # Colors and markers for methods
    colors = {
        'HAL-MLE': 'blue',
        'HAL': 'blue',
        'KDE': 'orange', 
        'TF': 'green',
        'TFPP': 'purple',
        'TFA2': 'brown',
        'TFPP-adjusted': 'magenta',
        'LogSplines': 'red'
    }
    
    markers = {
        'HAL-MLE': 'o',    # Circle marker for HAL-MLE
        'HAL': 'o',
        'TF': 's',  # Square marker for TF
        'TFPP': 'P',  # Plus (filled) marker for TFPP
        'TFA2': '*',  # Star marker for TF Algorithm 2
        'TFPP-adjusted': 'X',  # X marker for adjusted TFPP
        'LogSplines': '^',     # Triangle marker for LogSplines
        'KDE': 'D'             # Diamond marker for KDE
    }
    
    # Plot bias (not squared)
    ax = axes[0]
    for method_name, method_results in dgp_results['methods'].items():
        ax.plot(eval_points, method_results['bias'], 
               label=method_name, color=colors[method_name], linewidth=1, 
               marker=markers[method_name], markersize=4)
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=1)  # Add horizontal line at y=0
    ax.set_xlabel('Evaluation Point')
    ax.set_ylabel('Bias')
    ax.set_title('Bias Across Evaluation Points')
    ax.grid(True, alpha=0.3)
    
    # Plot variance
    ax = axes[1]
    for method_name, method_results in dgp_results['methods'].items():
        ax.plot(eval_points, method_results['variance'], 
               label=method_name, color=colors[method_name], linewidth=1,
               marker=markers[method_name], markersize=4)
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=1)  # Add horizontal line at y=0
    ax.set_xlabel('Evaluation Point')
    ax.set_ylabel('Variance')
    ax.set_title('Variance Across Evaluation Points')
    ax.grid(True, alpha=0.3)
    
    # Plot MSE
    ax = axes[2]
    for method_name, method_results in dgp_results['methods'].items():
        ax.plot(eval_points, method_results['mse'], 
               label=method_name, color=colors[method_name], linewidth=1,
               marker=markers[method_name], markersize=4)
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=1)  # Add horizontal line at y=0
    ax.set_xlabel('Evaluation Point')
    ax.set_ylabel('MSE')
    ax.set_title('MSE Across Evaluation Points')
    ax.grid(True, alpha=0.3)
    
    # Add single legend below the plot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.10), ncol=min(4, len(handles)))
    
    plt.suptitle(f'{DGP_NAME_MAPPING.get(dgp_name, dgp_name)}', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    
    # Save plot
    filename = f"bias_variance_mse_N{int(sample_size)}_{dgp_name}{filename_suffix}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filepath}")
    
    plt.close()

def create_metric_comparison_plots(
    all_results,
    save_dir="paper/resources/density_bias_variance_mse_analysis",
    sample_size=800,
    metrics_to_plot=None,
):
    """
    Create plots grouped by metric (bias, variance, MSE) where each plot shows all DGPs.
    
    Args:
        all_results: Dictionary containing results for all DGPs
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Colors and markers for methods
    colors = {
        'HAL-MLE': 'blue',
        'HAL': 'blue',
        'KDE': 'orange', 
        'TF': 'green',
        'TFPP': 'purple',
        'TFA2': 'brown',
        'TFPP-adjusted': 'magenta',
        'LogSplines': 'red'
    }
    
    markers = {
        'HAL-MLE': 'o',    # Circle marker for HAL-MLE
        'HAL': 'o',
        'TF': 's',         # Square marker for TF
        'TFPP': 'P',       # Plus (filled) marker for TFPP
        'TFA2': '*',
        'TFPP-adjusted': 'X',
        'LogSplines': '^', # Triangle marker for LogSplines
        'KDE': 'D'         # Diamond marker for KDE
    }
    
    # Define metrics to plot
    metrics = {
        'bias': 'Bias',
        'variance': 'Variance',
        'mse': 'MSE',
    }
    if metrics_to_plot is not None:
        metrics = {k: v for k, v in metrics.items() if k in set(metrics_to_plot)}
    
    # Get DGP order (consistent with experiment configuration)
    dgp_order = [name for name in DGP_CONFIGS.keys() if name in all_results]
    
    # Create a plot for each metric
    for metric_key, metric_title in metrics.items():
        # Create figure with 2x3 subplots for 6 DGPs - reduced size for paper
        fig, axes = plt.subplots(2, 3, figsize=(10, 5))
        axes = axes.flatten()  # Flatten to 1D array for easier indexing
        
        # Collect handles and labels for legend
        handles = []
        labels = []
        
        # Plot each DGP
        for idx, dgp_name in enumerate(dgp_order):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            dgp_results = all_results[dgp_name]
            eval_points = dgp_results['eval_points']
            
            # Plot all methods for this DGP and metric
            for method_name, method_results in dgp_results['methods'].items():
                line, = ax.plot(eval_points, method_results[metric_key], 
                       color=colors[method_name], linewidth=1,
                       marker=markers[method_name], markersize=3)
                # Add to legend only once (for the first DGP)
                if idx == 0:
                    handles.append(line)
                    labels.append(method_name)
            
            # Add horizontal line at y=0 for reference
            ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.7)
            
            # Customize subplot
            # Set y-label only for first column (idx 0, 3)
            if idx % 3 == 0:
                ax.set_ylabel(metric_title)
            # Set x-label only for bottom row (idx 3, 4, 5)
            if idx >= 3:
                ax.set_xlabel('Evaluation Point')
            ax.set_title(f'{DGP_NAME_MAPPING.get(dgp_name, dgp_name)}')
            ax.grid(True, alpha=0.3)
        
        # Hide any unused subplots
        for idx in range(len(dgp_order), len(axes)):
            axes[idx].set_visible(False)
        
        # Add legend at bottom
        fig.legend(
            handles,
            labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.07),
            # Force a single-row legend (avoid wrapping into 2 lines when we have 5 methods).
            ncol=len(handles),
            fontsize=8,
        )
        
        # Add overall title and layout
        plt.suptitle(f'{metric_title} Comparison Across Methods', fontsize=12)
        plt.tight_layout()
        # plt.subplots_adjust(top=0.93, bottom=0.15)  # Raised suptitle by increasing top margin
        
        # Save plot
        filename = f"methods_compare_{metric_key}_across_dgps_N{int(sample_size)}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {filepath}")
        
        plt.close()

def create_summary_table(all_results, sample_size=800):
    """
    Create a summary table of the bias-variance-MSE analysis.
    
    Args:
        all_results: Dictionary containing results for all DGPs
    """
    print("\n" + "="*120)
    print(f"BIAS-VARIANCE-MSE ANALYSIS SUMMARY (N={int(sample_size)})")
    print("="*120)
    
    # Header
    print(f"{'DGP':<30} {'Method':<20} {'Mean |Bias|':<12} {'Mean Variance':<15} {'Mean MSE':<12} {'N Experiments':<15}")
    print("-" * 120)
    
    for dgp_name, dgp_results in all_results.items():
        first_method = True
        for method_name, method_results in dgp_results['methods'].items():
            dgp_display = dgp_name if first_method else ""
            print(f"{dgp_display:<30} {method_name:<20} "
                  f"{np.mean(np.abs(method_results['bias'])):<12.6f} "
                  f"{np.mean(method_results['variance']):<15.6f} "
                  f"{np.mean(method_results['mse']):<12.6f} "
                  f"{method_results['n_experiments']:<15}")
            first_method = False
        if dgp_results['methods']:  # Only add separator if there were methods
            print("-" * 120)

def create_summary_latex_table(
    all_results,
    sample_size=800,
    save_path="paper/resources/density_bias_variance_mse_analysis/bias_variance_mse_table.tex",
):
    """
    Create and save a LaTeX table summarizing mean |bias|, variance, and MSE by DGP and method.

    The table follows the same formatting conventions as coverage_analysis_table.tex and omits the
    N experiments column. Mean |bias| is computed as the average absolute bias over evaluation points.

    Args:
        all_results: Dictionary containing results for all DGPs
        save_path: Output .tex file path
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Enforce DGP and method order consistent with experiment configuration
    dgp_order = [name for name in DGP_CONFIGS.keys() if name in all_results]
    method_order = list(METHODS.keys())

    lines = []
    lines.append("% Auto-generated bias-variance-MSE summary table")
    lines.append("\\begin{table}[H]")
    lines.append("  \\centering")
    lines.append("  \\scriptsize")
    lines.append("  \\setlength{\\tabcolsep}{2pt}")
    lines.append("  \\renewcommand{\\arraystretch}{1.1}")
    lines.append("  \\begin{tabular}{l l c c c}")
    lines.append("    \\toprule")
    lines.append("    DGP & Method & Mean $|$Bias$|$ & Mean Variance & Mean MSE \\\\")
    lines.append("    \\midrule")

    for dgp_name in dgp_order:
        dgp_results = all_results[dgp_name]
        first_method = True
        for method_name in method_order:
            if method_name not in dgp_results['methods']:
                continue
            method_results = dgp_results['methods'][method_name]
            mean_abs_bias = float(np.mean(np.abs(method_results['bias'])))
            mean_variance = float(np.mean(method_results['variance']))
            mean_mse = float(np.mean(method_results['mse']))

            dgp_display = DGP_NAME_MAPPING.get(dgp_name, dgp_name) if first_method else ""
            lines.append(
                f"    {dgp_display} & {method_name} & {mean_abs_bias:.6f} & {mean_variance:.6f} & {mean_mse:.6f} \\\\"
            )
            first_method = False

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append(
        f"  \\\\caption{{Bias--variance--MSE summary at $n={int(sample_size)}$ across DGPs and methods. "
        "Mean $|$Bias$|$ averages absolute bias over evaluation points; Variance and MSE are averaged over points.}}"
    )
    lines.append("  \\label{tab:bias_variance_mse_summary}")
    lines.append("\\end{table}")

    # ----------------------------------------------------------------------------------
    # Backup: include the pointwise data needed to reproduce methods_compare_{bias,variance,mse}
    # plots, without affecting LaTeX compilation when this file is included in a paper.
    # We emit this as TeX comments (% ...) so it's safe to keep alongside the summary table.
    # ----------------------------------------------------------------------------------
    def _fmt_arr(a, ndigits=10):
        a = np.asarray(a, dtype=float).ravel()
        return "[" + ", ".join(f"{float(x):.{ndigits}g}" for x in a) + "]"

    lines.append("%")
    lines.append("% =============================")
    lines.append("% BACKUP_POINTWISE_DATA_BEGIN")
    lines.append("% =============================")
    lines.append(f"% sample_size: {int(sample_size)}")
    # Eval points are shared across all DGPs in this analysis.
    any_dgp = next(iter(all_results.values()))
    lines.append(f"% eval_points: {_fmt_arr(any_dgp['eval_points'])}")
    lines.append("%")
    lines.append("% Data format per block:")
    lines.append("%   dgp: <name>")
    lines.append("%   method: <name>")
    lines.append("%   n_experiments: <int>")
    lines.append("%   bias: [..]        # pointwise bias at eval_points")
    lines.append("%   variance: [..]    # pointwise variance at eval_points")
    lines.append("%   mse: [..]         # pointwise mse at eval_points")
    lines.append("%")

    dgp_order = [name for name in DGP_CONFIGS.keys() if name in all_results]
    method_order = list(METHODS.keys())
    for dgp_name in dgp_order:
        dgp_results = all_results[dgp_name]
        for method_name in method_order:
            if method_name not in dgp_results["methods"]:
                continue
            mr = dgp_results["methods"][method_name]
            lines.append(f"% dgp: {dgp_name}")
            lines.append(f"% method: {method_name}")
            lines.append(f"% n_experiments: {int(mr.get('n_experiments', 0))}")
            lines.append(f"% bias: {_fmt_arr(mr['bias'])}")
            lines.append(f"% variance: {_fmt_arr(mr['variance'])}")
            lines.append(f"% mse: {_fmt_arr(mr['mse'])}")
            lines.append("%")

    lines.append("% ===========================")
    lines.append("% BACKUP_POINTWISE_DATA_END")
    lines.append("% ===========================")
    lines.append("%")

    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved LaTeX summary table: {save_path}")

def main(
    sample_size=800,
    save_dir="paper/resources/density_bias_variance_mse_analysis",
    results_dir="experiments/uniform_convergence/results",
    method_names=None,
    filename_suffix="",
    dgp_names=None,
    *,
    method_dir_suffix_map=None,
    density_key_map=None,
    exclude_max_density_gt=None,
):
    """
    Main function to run the bias-variance-MSE analysis.
    """
    print("Starting Bias-Variance-MSE Analysis for Four Density Estimation Methods")
    print("="*80)
    
    # Configuration
    eval_points = np.linspace(0.05, 0.95, 19)  # 19 points from 0.05 to 0.95
    dgp_names = list(DGP_CONFIGS.keys()) if dgp_names is None else list(dgp_names)
    
    print(f"Evaluation points: {len(eval_points)} points from {eval_points[0]} to {eval_points[-1]}")
    print(f"Sample size: {sample_size}")
    print(f"DGPs to analyze: {dgp_names}")
    print(f"Methods to compare: {list(METHODS.keys())}")
    
    # Analyze each DGP
    all_results = {}
    
    for dgp_name in dgp_names:
        dgp_results = analyze_dgp(
            dgp_name,
            eval_points,
            sample_size,
            results_dir=results_dir,
            method_names=method_names,
            method_dir_suffix_map=method_dir_suffix_map,
            density_key_map=density_key_map,
            exclude_max_density_gt=exclude_max_density_gt,
        )
        
        # Only store results if at least one method was successfully analyzed
        if dgp_results['methods']:
            all_results[dgp_name] = dgp_results
            
            # Create plot for this DGP
            create_dgp_comparison_plot(
                dgp_results,
                save_dir=save_dir,
                filename_suffix=filename_suffix,
                sample_size=sample_size,
            )
        else:
            print(f"Warning: No methods successfully analyzed for {dgp_name}")
    
    # Create summary table (console)
    if all_results:
        create_summary_table(all_results, sample_size=sample_size)
        # Create LaTeX summary table (file for paper inclusion)
        create_summary_latex_table(all_results, sample_size=sample_size, save_path=os.path.join(save_dir, "bias_variance_mse_table.tex"))
        # Create additional plots grouped by metric
        create_metric_comparison_plots(all_results, save_dir=save_dir, sample_size=sample_size)
    else:
        print("No results to summarize!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bias-Variance-MSE analysis and plotting for multiple density estimators."
    )
    parser.add_argument("--sample-size", type=int, default=800)
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/uniform_convergence/results",
        help="Root directory containing per-DGP per-method result folders.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="paper/resources/density_bias_variance_mse_analysis",
        help="Directory where plots and tables will be written.",
    )
    parser.add_argument(
        "--with-tfpp",
        action="store_true",
        help="Include TFPP (TrendFilteringCVXPYPP) as an additional method in plots.",
    )
    parser.add_argument(
        "--tfpp-recon",
        action="store_true",
        help=(
            "For TFPP only, load reconstructed densities from *_TrendFilteringCVXPYPP_N{N}_recon "
            "and use JSON key estimated_density_reconstructed."
        ),
    )
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        help=(
            "Restrict analysis to a specific method name (repeatable). "
            "Examples: --method HAL-MLE --method TFPP"
        ),
    )
    parser.add_argument(
        "--dgp",
        action="append",
        default=None,
        help="DGP name to analyze (repeatable). If omitted, analyzes all DGPs.",
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Skip writing LaTeX/table outputs (plots still generated).",
    )
    parser.add_argument(
        "--only-metric",
        type=str,
        choices=["bias", "variance", "mse"],
        default=None,
        help="If set, only generate the methods_compare_<metric>_across_dgps plot (no per-DGP plots, no tables).",
    )
    parser.add_argument(
        "--skip-dgp-plots",
        action="store_true",
        help="Skip per-DGP 3-panel plots (bias/variance/MSE).",
    )
    parser.add_argument(
        "--exclude-max-density-gt",
        type=float,
        default=None,
        help=(
            "Exclude any seed whose fitted max grid density exceeds this threshold, and use the "
            "intersection of kept seeds across methods for each DGP (e.g., 50)."
        ),
    )
    args = parser.parse_args()

    methods = ["HAL-MLE", "KDE", "TF", "LogSplines"]
    suffix = ""
    if args.with_tfpp:
        methods.insert(3, "TFPP")  # place next to TF
        suffix = "_with_TFPP"
        if args.tfpp_recon:
            suffix = "_with_TFPP_recon"

    if args.method:
        # Override methods list when explicit methods are provided.
        methods = args.method
        # Keep suffix behavior only for the default path; custom method selections should not
        # change filenames unexpectedly.
        suffix = ""

    dgp_list = args.dgp if args.dgp else list(DGP_CONFIGS.keys())
    method_dir_suffix_map = {}
    density_key_map = {}
    if args.tfpp_recon:
        method_dir_suffix_map["TFPP"] = "_recon"
        density_key_map["TFPP"] = "estimated_density_reconstructed"

    # If user requests only a single metric-comparison plot, run a minimal flow.
    if args.only_metric is not None:
        eval_points = np.linspace(0.05, 0.95, 19)
        all_results = {}
        for dgp_name in dgp_list:
            dgp_results = analyze_dgp(
                dgp_name,
                eval_points,
                sample_size=args.sample_size,
                results_dir=args.results_dir,
                method_names=methods,
                method_dir_suffix_map=method_dir_suffix_map,
                density_key_map=density_key_map,
                exclude_max_density_gt=args.exclude_max_density_gt,
            )
            if dgp_results["methods"]:
                all_results[dgp_name] = dgp_results
                if not args.skip_dgp_plots:
                    create_dgp_comparison_plot(
                        dgp_results,
                        save_dir=args.save_dir,
                        filename_suffix=suffix,
                        sample_size=args.sample_size,
                    )
        if all_results:
            create_metric_comparison_plots(
                all_results,
                save_dir=args.save_dir,
                sample_size=args.sample_size,
                metrics_to_plot=[args.only_metric],
            )
        else:
            print("No results to summarize!")
        raise SystemExit(0)

    # Run analysis and plots. Tables/metric-comparison plots can be expensive; allow skipping.
    if args.no_tables:
        # Minimal run: per-DGP plots only
        eval_points = np.linspace(0.05, 0.95, 19)
        all_results = {}
        for dgp_name in dgp_list:
            dgp_results = analyze_dgp(
                dgp_name,
                eval_points,
                sample_size=args.sample_size,
                results_dir=args.results_dir,
                method_names=methods,
                method_dir_suffix_map=method_dir_suffix_map,
                density_key_map=density_key_map,
                exclude_max_density_gt=args.exclude_max_density_gt,
            )
            if dgp_results["methods"]:
                all_results[dgp_name] = dgp_results
                if not args.skip_dgp_plots:
                    create_dgp_comparison_plot(
                        dgp_results,
                        save_dir=args.save_dir,
                        filename_suffix=suffix,
                        sample_size=args.sample_size,
                    )
    else:
        # Full run: include summary tables and metric-comparison plots
        main(
            sample_size=args.sample_size,
            save_dir=args.save_dir,
            results_dir=args.results_dir,
            method_names=methods,
            filename_suffix=suffix,
            dgp_names=dgp_list,
            method_dir_suffix_map=method_dir_suffix_map,
            density_key_map=density_key_map,
            exclude_max_density_gt=args.exclude_max_density_gt,
        )
