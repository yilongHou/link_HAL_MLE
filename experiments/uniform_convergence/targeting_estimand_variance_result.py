#!/usr/bin/env python3
"""
Targeting Estimand Variance vs Oracle (Monte Carlo) Comparison

This script compares EIC-based estimand variance estimates from targeting learners
against oracle (Monte Carlo) variance computed across multiple experimental runs.

The script:
1. Loads targeted results produced by asymptotic_efficiency_run_targeting_step.py
2. Computes per-run EIC-based estimand variance using targeting learners
3. Computes oracle variance across runs for the estimand(s)
4. Analyzes calibration, coverage, and produces summary tables and plots

Usage:
  uv run experiments/uniform_convergence/targeting_estimand_variance_result.py [args...]
"""
import argparse
import json
import os
import sys
import glob
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

# Project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Samplers and utils
from utils import (
    TruncatedNormal,
    TruncatedGMM,
    Sinusoidal,
    StepFunction,
)
from utils.density_computations import (
    generic_compute_moment_from_density,
    generic_compute_survival_from_density,
    generic_compute_cdf_from_density,
    generic_compute_median_from_density,
)

# Targeting learners: use estimand variance functions
from targeting.moments.learner import moments_estimand_variance
from targeting.survival.learner import survival_estimand_variance
from targeting.median.learner import median_estimand_variance

# DGP short-name mapping (for LaTeX tables)
with open('dpg-name-mapping.json', 'r') as _f:
    DGP_NAME_MAPPING = json.load(_f)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

SAMPLERS = {
    "TruncatedNormal": TruncatedNormal,
    "TruncatedGMM": TruncatedGMM,
    "Sinusoidal": Sinusoidal,
    "StepFunction": StepFunction,
}

VALID_TARGETERS = ['mean', 'second_moment', 'survival_0_5', 'median']

# Define DGP configurations used in experiments (from asymptotic_normality_results_parallel.py)
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

# True population parameters from population_parameters.md
TRUE_POPULATION_STATS = {
    "TruncatedNormal": {
        "mean": 0.5,
        "median": 0.5,
        "s_0_5": 0.5,
        "second_moment": 0.260000
    },
    "TruncatedGMMSymmetricThree": {
        "mean": 0.5,
        "median": 0.5,
        "s_0_5": 0.5,
        "second_moment": 0.311896
    },
    "TruncatedGMMFiveSpikes": {
        "mean": 0.5,
        "median": 0.5,
        "s_0_5": 0.5,
        "second_moment": 0.252092
    },
    "TruncatedGMMAsymmetricThree": {
        "mean": 0.559669,
        "median": 0.60837,
        "s_0_5": 0.61961,
        "second_moment": 0.354317
    },
    "StepFunction": {
        "mean": 0.438235,
        "median": 0.42500,
        "s_0_5": 0.41176,
        "second_moment": 0.263333
    },
    "Sinusoidal": {
        "mean": 0.5,
        "median": 0.5,
        "s_0_5": 0.5,
        "second_moment": 0.320145
    }
}

# ---------------------------------------------------------------------
# I/O and discovery
# ---------------------------------------------------------------------

def find_targeted_files(targeted_dir, dgp_filter=None, targeters=None):
    """
    Find targeted JSON files in targeted_dir, optionally filtered by DGP and targeter suffix.
    """
    files = []
    for root, _, filenames in os.walk(targeted_dir):
        if dgp_filter and dgp_filter not in root:
            continue
        if 'CVXPYEstimator' not in root:
            continue
        for fn in filenames:
            if not fn.endswith('.json'):
                continue
            if "6400" in fn:
                continue
            if targeters:
                # Require suffix __target=<name>.json
                if not any(f"__target={t}.json" in fn for t in targeters):
                    continue
            files.append(os.path.join(root, fn))
    return sorted(files)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# ---------------------------------------------------------------------
# Estimand computation helpers
# ---------------------------------------------------------------------

def compute_estimand_from_density(density, grid, targeter, targeting_points=None):
    """
    Compute estimand value(s) from density estimate.
    
    Returns:
      - value: scalar or vector np.ndarray
    """
    if targeter == 'mean':
        return generic_compute_moment_from_density(grid_points=grid, density=density, moment_order=1)
    elif targeter == 'second_moment':
        return generic_compute_moment_from_density(grid_points=grid, density=density, moment_order=2)
    elif targeter == 'median':
        return generic_compute_median_from_density(grid_points=grid, density=density)[0]
    elif targeter.startswith('survival'):
        # vector at targeting points
        surv, grid_s = generic_compute_survival_from_density(grid_points=grid, density=density)
        # interpolate to requested points
        tp = np.array(targeting_points, dtype=float)
        return np.interp(tp, grid_s, surv, left=1.0, right=0.0)
    else:
        raise ValueError(f"Unsupported targeter: {targeter}")

def compute_true_estimand(dgp_name, targeter, targeting_points=None):
    """
    Compute true estimand(s) from the DGP on a fine grid.
    """
    if dgp_name not in DGP_CONFIGS:
        raise ValueError(f"Unknown DGP: {dgp_name}")
    
    config = DGP_CONFIGS[dgp_name]
    sampler_class = SAMPLERS[config["sampler"]]
    sampler = sampler_class(**config["sampler_params"])
    grid = np.linspace(0, 1, 5001)
    dens = sampler.compute_density(grid)

    if targeter == 'mean':
        return generic_compute_moment_from_density(grid, dens, 1)
    elif targeter == 'second_moment':
        return generic_compute_moment_from_density(grid, dens, 2)
    elif targeter == 'median':
        return generic_compute_median_from_density(grid, dens)
    elif targeter.startswith('survival'):
        surv, grid_s = generic_compute_survival_from_density(grid, dens)
        tp = np.array(targeting_points, dtype=float)
        return np.interp(tp, grid_s, surv, left=1.0, right=0.0)
    else:
        raise ValueError(f"Unsupported targeter for truth: {targeter}")

def get_true_estimand_from_stats(dgp_name, targeter, targeting_points=None):
    """
    Get true estimand from precomputed population statistics when available.
    Falls back to compute_true_estimand if not available.
    """
    if dgp_name in TRUE_POPULATION_STATS:
        stats = TRUE_POPULATION_STATS[dgp_name]
        if targeter == 'mean' and 'mean' in stats:
            return stats['mean']
        elif targeter == 'second_moment' and 'second_moment' in stats:
            return stats['second_moment']
        elif targeter == 'median' and 'median' in stats:
            return stats['median']
        elif targeter == 'survival_0_5' and 's_0_5' in stats:
            return np.array([stats['s_0_5']])
    
    # Fallback to computation
    return compute_true_estimand(dgp_name, targeter, targeting_points)

# ---------------------------------------------------------------------
# EIC variance per file
# ---------------------------------------------------------------------

def estimate_variance_for_file(file_path, targeter, targeting_points=None, verbose=False):
    """
    Compute EIC-based estimand variance for a single targeted result file.
    Returns dict with 'value' (estimand), 'var' (variance vector or scalar), and metadata.
    """
    try:
        data = load_json(file_path)
        hal = data['HAL_results']
        
        # Check if targeting succeeded
        if hal.get('targeting_failed', False):
            return {'file': file_path, 'success': False, 'error': 'targeting_failed'}
        
        # Adapt targeted_fit schema for variance helpers 
        # Key: asymptotic_efficiency_run_targeting_step.py stores as 'grid_points' but variance functions expect 'grid_midpoints'
        targeted_fit = {
            'estimated_density': np.array(hal['estimated_density']),
            'grid_midpoints': np.array(hal.get('grid_midpoints', hal.get('grid_points'))),
        }
        # Uncensored data as DataFrame
        uncensored_df = pd.DataFrame({'W1': hal['data_points']})

        # Compute estimand value from targeted density
        est_value = compute_estimand_from_density(
            density=targeted_fit['estimated_density'],
            grid=targeted_fit['grid_midpoints'],
            targeter=targeter,
            targeting_points=targeting_points
        )

        # Compute EIC variance via the appropriate function
        if targeter == 'mean':
            var_vec = moments_estimand_variance(
                targeted_fit=targeted_fit,
                uncensored_data=uncensored_df,
                x_moment=1
            )
        elif targeter == 'second_moment':
            var_vec = moments_estimand_variance(
                targeted_fit=targeted_fit,
                uncensored_data=uncensored_df,
                x_moment=2
            )
        elif targeter == 'median':
            var_vec = median_estimand_variance(
                targeted_fit=targeted_fit,
                uncensored_data=uncensored_df
            )
        elif targeter.startswith('survival'):
            var_vec = survival_estimand_variance(
                targeted_fit=targeted_fit,
                uncensored_data=uncensored_df,
                targeting_points=np.array(targeting_points, dtype=float)
            )
        else:
            raise ValueError(f"Unsupported targeter: {targeter}")

        # Ensure np.ndarray
        var_vec = np.atleast_1d(np.asarray(var_vec, dtype=float))
        est_value = np.atleast_1d(np.asarray(est_value, dtype=float))

        return {
            'file': file_path,
            'estimand': est_value,
            'var': var_vec,
            'success': True
        }
    except Exception as e:
        if verbose:
            print(f"[WARN] {Path(file_path).name}: {e}")
        return {'file': file_path, 'success': False, 'error': str(e)}

# ---------------------------------------------------------------------
# Aggregation and comparison
# ---------------------------------------------------------------------

def aggregate_and_compare(results, true_value):
    """
    results: list of dicts with keys {'estimand': vec, 'var': vec}
    true_value: scalar or vector
    Returns dict of aggregate metrics and arrays.
    """
    # Stack
    est_stack = np.vstack([r['estimand'] for r in results if r['success']])
    var_stack = np.vstack([r['var'] for r in results if r['success']])
    n_runs, k = est_stack.shape

    # Oracle variance across runs
    oracle_var = np.var(est_stack, axis=0, ddof=1)
    oracle_se = np.sqrt(oracle_var)
    mean_eic_var = np.mean(var_stack, axis=0)

    # Calibration
    calib_ratio = mean_eic_var / np.maximum(oracle_var, 1e-18)
    
    # Coverage using EIC SE (original approach)
    se_stack = np.sqrt(np.maximum(var_stack, 0))
    true = np.atleast_1d(np.asarray(true_value, dtype=float))
    coverage_ind = (np.abs(est_stack - true[None, :]) <= 1.96 * se_stack).astype(int)
    coverage = coverage_ind.mean(axis=0)

    # Oracle coverage using oracle SE (following asymptotic normality scripts)
    oracle_lower = est_stack - 1.96 * oracle_se  # shape: (n_runs, k)
    oracle_upper = est_stack + 1.96 * oracle_se
    oracle_coverage_indicators = ((true >= oracle_lower) & 
                                 (true <= oracle_upper)).astype(int)
    oracle_coverage = np.mean(oracle_coverage_indicators, axis=0)

    # Oracle percentile-based coverage (95% CI using 2.5% and 97.5% percentiles)
    oracle_ci_lower_range_95 = np.percentile(est_stack, 97.5, axis=0) - np.percentile(est_stack, 2.5, axis=0)   # shape: (k,)

    oracle_ci_lower_percentile = est_stack - oracle_ci_lower_range_95 / 2
    oracle_ci_upper_percentile = est_stack + oracle_ci_lower_range_95 / 2
    oracle_percentile_coverage_indicators = ((true >= oracle_ci_lower_percentile) & 
                                            (true <= oracle_ci_upper_percentile)).astype(int)
    oracle_percentile_coverage = np.mean(oracle_percentile_coverage_indicators, axis=0)

    # CI widths for comparison
    eic_ci_width = 2 * 1.96 * np.sqrt(mean_eic_var)
    oracle_ci_width_se = 2 * 1.96 * oracle_se
    oracle_ci_width_percentile = oracle_ci_upper_percentile - oracle_ci_lower_percentile

    # RMSE of SE vs oracle SE
    rmse_se = np.sqrt(np.mean((np.sqrt(mean_eic_var) - oracle_se)**2))

    return {
        'n_runs': n_runs,
        'k': k,
        'oracle_var': oracle_var,
        'oracle_se': oracle_se,
        'mean_eic_var': mean_eic_var,
        'calibration_ratio': calib_ratio,
        'coverage': coverage,
        'oracle_coverage': oracle_coverage,
        'oracle_percentile_coverage': oracle_percentile_coverage,
        'eic_ci_width': eic_ci_width,
        'oracle_ci_width_se': oracle_ci_width_se,
        'oracle_ci_width_percentile': oracle_ci_width_percentile,
        'rmse_se': rmse_se,
        'est_stack': est_stack,
        'var_stack': var_stack
    }

# ---------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------

def save_results(results_dict, cache_dir, n_suffix=""):
    """Save results to cache files, one per DGP."""
    os.makedirs(cache_dir, exist_ok=True)
    
    for dgp_name in results_dict:
        cache_file = os.path.join(cache_dir, f"targeting_variance_{dgp_name}{n_suffix}.json")
        # Convert numpy arrays to lists for JSON serialization
        cache_data = {}
        for n_sample in results_dict[dgp_name]:
            cache_data[str(n_sample)] = {}
            for targeter in results_dict[dgp_name][n_sample]:
                agg = results_dict[dgp_name][n_sample][targeter]
                cache_data[str(n_sample)][targeter] = {
                    'n_runs': int(agg['n_runs']),
                    'k': int(agg['k']),
                    'oracle_var': agg['oracle_var'].tolist(),
                    'oracle_se': agg['oracle_se'].tolist(),
                    'mean_eic_var': agg['mean_eic_var'].tolist(),
                    'calibration_ratio': agg['calibration_ratio'].tolist(),
                    'coverage': agg['coverage'].tolist(),
                    'oracle_coverage': agg['oracle_coverage'].tolist(),
                    'oracle_percentile_coverage': agg['oracle_percentile_coverage'].tolist(),
                    'eic_ci_width': agg['eic_ci_width'].tolist(),
                    'oracle_ci_width_se': agg['oracle_ci_width_se'].tolist(),
                    'oracle_ci_width_percentile': agg['oracle_ci_width_percentile'].tolist(),
                    'rmse_se': float(agg['rmse_se']),
                    # Save raw data for box plots
                    'var_stack': agg['var_stack'].tolist(),
                    'est_stack': agg['est_stack'].tolist()
                }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)

def load_results(dgp_names, cache_dir, n_suffix=""):
    """Load results from cache files."""
    results_dict = {}
    
    for dgp_name in dgp_names:
        cache_file = os.path.join(cache_dir, f"targeting_variance_{dgp_name}{n_suffix}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            results_dict[dgp_name] = {}
            for n_sample_str in cache_data:
                n_sample = int(n_sample_str)
                results_dict[dgp_name][n_sample] = {}
                
                for targeter in cache_data[n_sample_str]:
                    agg_data = cache_data[n_sample_str][targeter]
                    results_dict[dgp_name][n_sample][targeter] = {
                        'n_runs': agg_data['n_runs'],
                        'k': agg_data['k'],
                        'oracle_var': np.array(agg_data['oracle_var']),
                        'oracle_se': np.array(agg_data.get('oracle_se', np.sqrt(agg_data['oracle_var']))),
                        'mean_eic_var': np.array(agg_data['mean_eic_var']),
                        'calibration_ratio': np.array(agg_data['calibration_ratio']),
                        'coverage': np.array(agg_data['coverage']),
                        'oracle_coverage': np.array(agg_data.get('oracle_coverage', agg_data['coverage'])),
                        'oracle_percentile_coverage': np.array(agg_data.get('oracle_percentile_coverage', agg_data['coverage'])),
                        'eic_ci_width': np.array(agg_data.get('eic_ci_width', 2 * 1.96 * np.sqrt(agg_data['mean_eic_var']))),
                        'oracle_ci_width_se': np.array(agg_data.get('oracle_ci_width_se', 2 * 1.96 * np.sqrt(agg_data['oracle_var']))),
                        'oracle_ci_width_percentile': np.array(agg_data.get('oracle_ci_width_percentile', agg_data.get('eic_ci_width', 2 * 1.96 * np.sqrt(agg_data['mean_eic_var'])))),
                        'rmse_se': agg_data['rmse_se'],
                        # Load raw data for box plots
                        'var_stack': np.array(agg_data.get('var_stack', [])),
                        'est_stack': np.array(agg_data.get('est_stack', []))
                    }
    
    return results_dict if results_dict else None

def clear_cache(dgp_names, cache_dir, n_suffix=""):
    """Clear cache files for specified DGPs."""
    cleared = []
    for dgp_name in dgp_names:
        cache_file = os.path.join(cache_dir, f"targeting_variance_{dgp_name}{n_suffix}.json")
        if os.path.exists(cache_file):
            os.remove(cache_file)
            cleared.append(dgp_name)
    
    if cleared:
        print(f"Cleared cache for: {', '.join(cleared)}")
    else:
        print("No cache files found to clear")

# ---------------------------------------------------------------------
# Plotting and tables
# ---------------------------------------------------------------------

def create_coverage_plots(results_dict, save_dir="paper/resources/target_estimand_variance", n_suffix=""):
    """
    Create coverage plots showing 95% CI coverage across sample sizes for both EIC and Oracle methods.
    Layout: 2 rows x 2 columns, one subplot per estimand (mean, second_moment, survival_0_5, median),
    each comparing EIC-based, Oracle SE-based, and Oracle Percentile coverages. 
    Optimized for publication with compact figure size (4.5x4) and adjusted fonts/line widths.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    targeters_order = ['mean', 'second_moment', 'survival_0_5', 'median']
    # Title mapping to match table captions
    title_map = {
        'mean': 'Mean',
        'second_moment': 'Second Moment',
        'survival_0_5': 'Survival at 0.5',
        'median': 'Median'
    }
    method_specs = [
        ("EIC-based", 'coverage', dict(marker='o', linestyle='-', linewidth=1.0, markersize=3)),
        ("Oracle SE-based", 'oracle_coverage', dict(marker='s', linestyle='--', linewidth=1.0, markersize=3)),
        ("Oracle Percentile", 'oracle_percentile_coverage', dict(marker='^', linestyle=':', linewidth=1.0, markersize=3)),
    ]

    for dgp_name in results_dict:
        print(f"Creating coverage plot for {dgp_name}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(6, 4))
        axes = axes.ravel()  # Flatten to 1D array for easier iteration
        
        for idx, targeter in enumerate(targeters_order):
            ax = axes[idx]
            sample_sizes = []
            series = {label: [] for (label, _, __) in method_specs}

            for n_sample in sorted(results_dict[dgp_name].keys()):
                if n_sample is None or n_sample >= 6400:
                    continue
                if targeter not in results_dict[dgp_name][n_sample]:
                    continue
                agg = results_dict[dgp_name][n_sample][targeter]

                sample_sizes.append(n_sample)
                for label, key, _style in method_specs:
                    vals = agg.get(key, None)
                    if vals is None:
                        # Fallback to EIC coverage when oracle keys not present (back-compat)
                        vals = agg.get('coverage', None)
                    # Take mean if vector (e.g., survival)
                    series[label].append(float(np.nanmean(vals)) * 100.0)

            if sample_sizes:
                positions = np.arange(len(sample_sizes))
                for (label, _key, style), y in zip(method_specs, series.values()):
                    ax.plot(positions, y, label=label, **style)

                # Nominal line
                ax.axhline(y=95.0, color='red', linestyle='--', alpha=0.7, linewidth=1.0, label='Nominal 95%')

                # X ticks as actual sample sizes, rotated 90 degrees
                ax.set_xticks(positions)
                ax.set_xticklabels([str(n) for n in sample_sizes], fontsize=6, rotation=90)

            # Only show Y label for first column (idx 0 and 2)
            if idx in [0, 2]:
                ax.set_ylabel('Coverage (%)', fontsize=7)
            ax.set_title(title_map.get(targeter, targeter.replace('_', ' ').title()), fontsize=8)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_ylim(75, 100)
            ax.tick_params(axis='both', labelsize=6)

        # Shared legend at the bottom
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.025))

        # Suptitle with DGP name mapping
        dgp_title = DGP_NAME_MAPPING.get(dgp_name, dgp_name)
        plt.suptitle(f'{dgp_title} - Coverage', fontsize=9, fontweight='bold', y=0.99)
        # Reduce gap to title and leave room for bottom legend (tighter):
        # more top (0.985) and slightly less bottom space
        # plt.tight_layout(rect=(0.03, 0.11, 0.97, 0.985))
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"Figure_Coverage_{dgp_name}{n_suffix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()

def create_ci_width_plots(results_dict, save_dir="paper/resources/target_estimand_variance", n_suffix=""):
    """
    Create CI width comparison plots showing distribution of CI widths for each seed.
    Uses box plots to show the distribution across all seeds/runs.
    Layout: 2 rows x 2 columns (one subplot per estimand).
    Optimized for publication with compact figure size (4.5x4) and adjusted fonts/line widths.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Title mapping to match table captions
    title_map = {
        'mean': 'Mean',
        'second_moment': 'Second Moment',
        'survival_0_5': 'Survival at 0.5',
        'median': 'Median'
    }
    
    for dgp_name in results_dict:
        print(f"Creating CI width plot for {dgp_name}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(6, 4))
        axes = axes.ravel()
        
        # Prepare shared legend elements once
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', edgecolor='darkblue', alpha=0.8, label='EIC-based'),
            Line2D([0], [0], color='red', linewidth=1.5, alpha=0.8, label='Oracle SE')
        ]

        for i, targeter in enumerate(['mean', 'second_moment', 'survival_0_5', 'median']):
            ax = axes[i]
            
            sample_sizes = []
            eic_ci_data = []
            oracle_ci_data = []
            
            for n_sample in sorted(results_dict[dgp_name].keys()):
                if n_sample >= 6400:
                    continue
                if targeter in results_dict[dgp_name][n_sample]:
                    agg = results_dict[dgp_name][n_sample][targeter]
                    
                    # Get raw variance data for each seed
                    var_stack = agg.get('var_stack')
                    if var_stack is not None and len(var_stack) > 0:
                        var_stack = np.array(var_stack)
                        
                        # Compute CI widths for each seed: 2 * 1.96 * sqrt(variance)
                        eic_ci_widths_per_seed = 2 * 1.96 * np.sqrt(np.maximum(var_stack, 0))
                        
                        # For survival targeters (vector case), take mean across targeting points
                        if eic_ci_widths_per_seed.ndim > 1:
                            eic_ci_widths_per_seed = np.mean(eic_ci_widths_per_seed, axis=1)
                        
                        sample_sizes.append(n_sample)
                        eic_ci_data.append(eic_ci_widths_per_seed)
                        
                        # Oracle CI width is constant for all seeds (computed from oracle variance)
                        oracle_se = agg.get('oracle_se', np.sqrt(agg.get('oracle_var', [0])))
                        oracle_ci_width = 2 * 1.96 * np.mean(oracle_se)  # Take mean for vector case
                        oracle_ci_data.append([oracle_ci_width] * len(eic_ci_widths_per_seed))
            
            if sample_sizes and eic_ci_data:
                # Create box plot positions centered on integer positions
                positions = np.arange(len(sample_sizes))
                
                # Create box plots for EIC CI widths (centered)
                bp1 = ax.boxplot(eic_ci_data, positions=positions, widths=0.6, 
                                patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.8, linewidth=0.8),
                                medianprops=dict(color='darkblue', linewidth=1.2),
                                whiskerprops=dict(color='darkblue', linewidth=0.8),
                                capprops=dict(color='darkblue', linewidth=0.8),
                                flierprops=dict(marker='o', markerfacecolor='lightblue', 
                                              markeredgecolor='darkblue', markersize=2, alpha=0.6))
                
                # Add horizontal lines for Oracle CI widths instead of box plots
                for j, n_sample in enumerate(sample_sizes):
                    agg = results_dict[dgp_name][n_sample][targeter]
                    oracle_se = agg.get('oracle_se', np.sqrt(agg.get('oracle_var', [0])))
                    oracle_ci_width = 2 * 1.96 * np.mean(oracle_se)
                    ax.hlines(oracle_ci_width, positions[j] - 0.4, positions[j] + 0.4, 
                             colors='red', linewidth=1.5, alpha=0.8, label='Oracle SE-based' if j == 0 else "")
                
                # Set x-axis labels to sample sizes, rotated 90 degrees
                ax.set_xticks(positions)
                ax.set_xticklabels([str(n) for n in sample_sizes], fontsize=6, rotation=90)
                # Only show Y label for first column (i=0 and i=2)
                if i in [0, 2]:
                    ax.set_ylabel('CI Width', fontsize=7)
                ax.set_title(title_map.get(targeter, targeter.replace("_", " ").title()), fontsize=8)
                ax.grid(True, alpha=0.3, linewidth=0.5)
                ax.tick_params(axis='both', labelsize=6)
                
                # Set log scale for y-axis if there are many sample sizes
                if len(sample_sizes) > 4:
                    ax.set_yscale('log')
        
        # Shared legend at the bottom
        fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.025))

        # Suptitle with DGP name mapping
        dgp_title = DGP_NAME_MAPPING.get(dgp_name, dgp_name)
        plt.suptitle(f'{dgp_title} - CI Width', fontsize=9, fontweight='bold', y=0.99)
        # Reduce gap to title and leave room for bottom legend (tighter)
        # plt.tight_layout(rect=(0.03, 0.11, 0.97, 0.985))
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"Figure_CI_Width_{dgp_name}{n_suffix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()

def create_coverage_latex_tables(results_dict, targeters=None, save_dir="paper/resources/target_estimand_variance", n_suffix=""):
    """
    Create LaTeX tables for coverage results (EIC vs Oracle) per estimand/targeter.
    Mirrors the style of coverage_analysis_table.tex used for delta-method density.

    One table per targeter is created:
      - coverage_targeting_<targeter>.tex

    Each table has 6 rows (DGPs) and 8 columns (sample sizes),
    with cells formatted as (EIC coverage, oracle coverage) in percent with 1 decimal.
    Missing combinations are printed as "--".
    """
    os.makedirs(save_dir, exist_ok=True)

    # Default targeters order
    if targeters is None:
        targeters = ['mean', 'second_moment', 'survival_0_5', 'median']

    # DGP order consistent with other resources
    dgp_order = [
        "TruncatedNormal",
        "TruncatedGMMSymmetricThree",
        "TruncatedGMMAsymmetricThree",
        "TruncatedGMMFiveSpikes",
        "StepFunction",
        "Sinusoidal",
    ]

    # Desired sample sizes (columns)
    sample_sizes = [25, 50, 100, 200, 400, 800, 1600, 3200]

    def _fmt_pair(eic_cov, oracle_cov):
        if eic_cov is None or oracle_cov is None:
            return "--"
        if np.isnan(eic_cov) or np.isnan(oracle_cov):
            return "--"
        return f"({eic_cov:.1f}, {oracle_cov:.1f})"

    for targeter in targeters:
        try:
            lines = []
            lines.append("% Auto-generated targeting coverage analysis table")
            lines.append("\\begin{table}[H]")
            lines.append("  \\centering")
            lines.append("  \\scriptsize")
            lines.append("  \\setlength{\\tabcolsep}{2pt}")
            lines.append("  \\renewcommand{\\arraystretch}{1.1}")
            lines.append("  \\begin{tabular}{l" + "c" * len(sample_sizes) + "}")
            lines.append("    \\toprule")
            header_row = ["DGP"] + [f"N={n}" for n in sample_sizes]
            lines.append("    " + " & ".join(header_row) + " \\\\")
            lines.append("    \\midrule")

            for dgp in dgp_order:
                row = [DGP_NAME_MAPPING.get(dgp, dgp)]
                d_results = results_dict.get(dgp, {}) if isinstance(results_dict, dict) else {}

                for n in sample_sizes:
                    cell = "--"
                    if d_results and n in d_results and targeter in d_results[n]:
                        agg = d_results[n][targeter]
                        # Mean across vector components when applicable
                        eic_cov_vals = agg.get('coverage', None)
                        oracle_cov_vals = agg.get('oracle_coverage', None)
                        if eic_cov_vals is not None and oracle_cov_vals is not None:
                            try:
                                eic_cov = float(np.nanmean(eic_cov_vals) * 100.0)
                                oracle_cov = float(np.nanmean(oracle_cov_vals) * 100.0)
                                cell = _fmt_pair(eic_cov, oracle_cov)
                            except Exception:
                                cell = "--"
                    row.append(cell)

                lines.append("    " + " & ".join(row) + " \\\\")

            lines.append("    \\bottomrule")
            lines.append("  \\end{tabular}")
            # Simple captions for each estimand
            caption_map = {
                'mean': 'Coverage for mean',
                'second_moment': 'Coverage for second moment',
                'survival_0_5': 'Coverage for survival at 0.5',
                'median': 'Coverage for median'
            }
            caption_text = caption_map.get(targeter, f'Coverage for {targeter}')
            lines.append(f"  \\caption{{{caption_text}}}")
            lines.append("  \\label{tab:coverage_targeting_" + targeter + "}")
            lines.append("\\end{table}")

            out_name = f"coverage_targeting_{targeter}{n_suffix}.tex"
            out_path = os.path.join(save_dir, out_name)
            with open(out_path, 'w') as f:
                f.write("\n".join(lines))

            print(f"Saved targeting coverage LaTeX table: {out_path}")
        except Exception as e:
            print(f"Failed to create LaTeX table for targeter {targeter}: {e}")

def print_summary_table(results_dict, save_csv=True, csv_dir="experiments/uniform_convergence/targeting_variance"):
    """
    Create a summary table of the targeting variance analysis including oracle coverage.
    Optionally save as CSV file.
    """
    print("\n" + "="*160)
    print("TARGETING ESTIMAND VARIANCE VS ORACLE COMPARISON")
    print("="*160)
    
    # Collect data for CSV
    csv_data = []
    
    for dgp_name in results_dict:
        print(f"\n{dgp_name}:")
        print("-" * 155)
        print(f"{'Sample Size':<12} {'Targeter':<16} {'N Runs':<8} {'EIC Cov':<10} {'Oracle Cov':<12} {'Oracle %ile':<12} {'Calib Ratio':<12} {'Oracle SE':<12} {'EIC SE':<12} {'RMSE SE':<12}")
        print("-" * 155)
        
        for n_sample in sorted(results_dict[dgp_name].keys()):
            for targeter in ['mean', 'second_moment', 'survival_0_5', 'median']:
                if targeter in results_dict[dgp_name][n_sample]:
                    agg = results_dict[dgp_name][n_sample][targeter]
                    
                    # EIC coverage
                    mean_eic_cov = float(np.mean(agg['coverage']))
                    
                    # Oracle coverage (with fallback for backward compatibility)
                    oracle_cov = agg.get('oracle_coverage', np.array([np.nan]))
                    mean_oracle_cov = float(np.mean(oracle_cov))
                    
                    # Oracle percentile coverage (with fallback for backward compatibility)  
                    oracle_pct_cov = agg.get('oracle_percentile_coverage', np.array([np.nan]))
                    mean_oracle_pct_cov = float(np.mean(oracle_pct_cov))
                    
                    # Other metrics
                    mean_calib = float(np.mean(agg['calibration_ratio']))
                    mean_oracle_se = float(np.mean(np.sqrt(agg['oracle_var'])))
                    mean_eic_se = float(np.mean(np.sqrt(agg['mean_eic_var'])))
                    
                    print(f"N={n_sample:<7} {targeter:<16} {agg['n_runs']:<8} "
                          f"{mean_eic_cov*100:>6.1f}% {mean_oracle_cov*100:>8.1f}% {mean_oracle_pct_cov*100:>8.1f}% "
                          f"{mean_calib:>9.3f} {mean_oracle_se:>9.6f} {mean_eic_se:>9.6f} {agg['rmse_se']:>9.6f}")
                    
                    # Collect data for CSV
                    csv_data.append({
                        'DGP': dgp_name,
                        'Sample_Size': n_sample,
                        'Targeter': targeter,
                        'N_Runs': agg['n_runs'],
                        'EIC_Coverage_Pct': mean_eic_cov * 100,
                        'Oracle_Coverage_Pct': mean_oracle_cov * 100,
                        'Oracle_Percentile_Coverage_Pct': mean_oracle_pct_cov * 100,
                        'Calibration_Ratio': mean_calib,
                        'Oracle_SE': mean_oracle_se,
                        'EIC_SE': mean_eic_se,
                        'RMSE_SE': agg['rmse_se']
                    })
    
    # Save CSV
    if save_csv and csv_data:
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, "targeting_variance_summary.csv")
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        print(f"\nSummary table saved to: {csv_path}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare EIC estimand variance vs oracle across targeted results")
    parser.add_argument("--targeted_dir", type=str, default="experiments/uniform_convergence/targeted_results")
    parser.add_argument("--dgp", type=str, default=None, help="Filter by DGP (folder name contains it)")
    parser.add_argument("--targeters", type=str, default="mean,second_moment,survival_0_5,median")
    parser.add_argument("--n-only", type=int, default=None, help="Process single sample size N")
    parser.add_argument("--targeting_points", type=str, default="0.5", help="Comma list for survival/CDF")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--cache-dir", type=str, default="experiments/uniform_convergence/cache_targeting_variance")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--latex-table", action="store_true", help="Generate LaTeX coverage tables from cached/computed results")
    
    args = parser.parse_args()

    targeters = [t.strip() for t in args.targeters.split(',') if t.strip()]
    for t in targeters:
        if t not in VALID_TARGETERS:
            print(f"Invalid targeter: {t}; valid: {VALID_TARGETERS}")
            sys.exit(1)

    # Parse targeting points if needed
    tp = [float(x) for x in args.targeting_points.split(',') if x.strip()] if any(t.startswith('survival') for t in targeters) else None

    # Determine DGP names
    if args.dgp:
        dgp_names = [args.dgp]
    else:
        dgp_names = ["TruncatedNormal", "TruncatedGMMSymmetricThree", "TruncatedGMMAsymmetricThree",
                     "TruncatedGMMFiveSpikes", "StepFunction", "Sinusoidal"]

    # Set up suffixes for caching
    n_suffix = f"_N{args.n_only}" if args.n_only is not None else ""
    
    # Set up save directories - plots now go to paper resources directory
    plot_dir = "paper/resources/target_estimand_variance"
    if args.n_only is not None:
        plot_dir = f"paper/resources/target_estimand_variance/plots_N{args.n_only}"

    # Clear cache if requested
    if args.clear_cache:
        clear_cache(dgp_names, args.cache_dir, n_suffix)
        return

    # Try to load existing results
    results_dict = None
    if not args.overwrite_cache:
        results_dict = load_results(dgp_names, args.cache_dir, n_suffix)
        if results_dict:
            print(f"Loaded cached results for {len(results_dict)} DGPs")

    if args.plots_only:
        if results_dict is None:
            print("No cached results found for plots-only mode")
            return
        print("Generating plots from cached results...")
        create_coverage_plots(results_dict, save_dir=plot_dir, n_suffix=n_suffix)
        create_ci_width_plots(results_dict, save_dir=plot_dir, n_suffix=n_suffix)
        print_summary_table(results_dict)
        if args.latex_table:
            create_coverage_latex_tables(results_dict, targeters=targeters, save_dir="paper/resources/target_estimand_variance", n_suffix=n_suffix)
        return

    if results_dict is not None and not args.overwrite_cache:
        print("Using cached results. Use --overwrite-cache to recompute.")
        create_coverage_plots(results_dict, save_dir=plot_dir, n_suffix=n_suffix)
        create_ci_width_plots(results_dict, save_dir=plot_dir, n_suffix=n_suffix)
        print_summary_table(results_dict)
        if args.latex_table:
            create_coverage_latex_tables(results_dict, targeters=targeters, save_dir="paper/resources/target_estimand_variance", n_suffix=n_suffix)
        return

    # Discover files
    targeted_files = find_targeted_files(args.targeted_dir, dgp_filter=args.dgp, targeters=targeters)
    if args.n_only is not None:
        targeted_files = [p for p in targeted_files if f"_N{args.n_only}" in p]
    if not targeted_files:
        print("No targeted files found with the given filters.")
        return

    print(f"Found {len(targeted_files)} targeted files to process")

    # Group by (dgp_name, N, targeter)
    groups = {}
    for fp in targeted_files:
        name = Path(fp).name
        # Expect suffix __target=<name>.json
        tgt = None
        for t in targeters:
            if f"__target={t}.json" in name:
                tgt = t
                break
        if tgt is None:
            continue
        
        # Extract DGP and N from directory structure
        # Expected: .../DGP_CVXPYEstimator_N<sample>/...
        parts = Path(fp).parts
        parent = Path(fp).parent.name
        if "_CVXPYEstimator_N" in parent:
            dgp_name = parent.split("_CVXPYEstimator_N")[0]
            try:
                n_sample = int(parent.split("_CVXPYEstimator_N")[1])
            except Exception:
                n_sample = None
        else:
            # Fallback: try to extract from path
            dgp_name = parent.replace("_CVXPYEstimator", "")
            n_sample = None

        key = (dgp_name, n_sample, tgt)
        groups.setdefault(key, []).append(fp)

    # Process groups
    print("\nTARGETING ESTIMAND VARIANCE VS ORACLE")
    print("="*80)
    
    results_dict = {}
    
    for (dgp_name, n_sample, tgt), file_list in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1] or 0, x[0][2])):
        if dgp_name not in results_dict:
            results_dict[dgp_name] = {}
        if n_sample not in results_dict[dgp_name]:
            results_dict[dgp_name][n_sample] = {}
        
        print(f"Processing {dgp_name} N={n_sample} {tgt}: {len(file_list)} files")
        
        # Compute per-file EIC variances
        eval_axis = np.array(tp, dtype=float) if tgt.startswith('survival') else np.array([0.0])
        
        with tqdm(total=len(file_list), desc=f"Processing {tgt}", leave=False) as pbar:
            per_file = Parallel(n_jobs=args.n_jobs)(
                delayed(estimate_variance_for_file)(fp, tgt, targeting_points=tp, verbose=args.verbose) 
                for fp in file_list
            )
            pbar.update(len(file_list))
        
        per_file = [r for r in per_file if r is not None and r.get('success')]
        if len(per_file) == 0:
            print(f"  No successful files for {dgp_name} N={n_sample} {tgt}")
            continue

        # True value
        try:
            true_val = get_true_estimand_from_stats(dgp_name, tgt, targeting_points=tp)
        except Exception as e:
            print(f"  Warning: Could not compute true value for {dgp_name} {tgt}: {e}")
            true_val = None

        # Aggregate and compare
        if true_val is None:
            # Use oracle variance only (no coverage analysis)
            est_stack = np.vstack([r['estimand'] for r in per_file])
            var_stack = np.vstack([r['var'] for r in per_file])
            oracle_var = np.var(est_stack, axis=0, ddof=1)
            mean_eic_var = np.mean(var_stack, axis=0)
            calib_ratio = mean_eic_var / np.maximum(oracle_var, 1e-18)
            rmse_se = np.sqrt(np.mean((np.sqrt(mean_eic_var) - np.sqrt(oracle_var))**2))
            
            results_dict[dgp_name][n_sample][tgt] = {
                'n_runs': len(per_file),
                'k': len(np.atleast_1d(oracle_var)),
                'oracle_var': oracle_var,
                'mean_eic_var': mean_eic_var,
                'calibration_ratio': calib_ratio,
                'coverage': np.array([np.nan]),  # No coverage without truth
                'rmse_se': rmse_se
            }
            print(f"  {len(per_file)} runs, calib={np.mean(calib_ratio):.3f} (no coverage - truth unavailable)")
        else:
            agg = aggregate_and_compare(per_file, true_val)
            results_dict[dgp_name][n_sample][tgt] = agg
            print(f"  {agg['n_runs']} runs, calib={np.mean(agg['calibration_ratio']):.3f}, cov={np.mean(agg['coverage'])*100:.1f}%")

    # Save results to cache
    save_results(results_dict, args.cache_dir, n_suffix)
    print(f"Results cached in {args.cache_dir}")

    # Create summary table
    print_summary_table(results_dict)

    # Create plots
    print("\nCreating plots...")
    create_coverage_plots(results_dict, save_dir=plot_dir, n_suffix=n_suffix)
    create_ci_width_plots(results_dict, save_dir=plot_dir, n_suffix=n_suffix)

    print("\nTargeting estimand variance analysis completed!")
    print(f"Cache files stored in {args.cache_dir}/")
    print(f"Plots saved in {plot_dir}/")
    print("Use --dgp <DGP_NAME> to run analysis for specific DGPs only")
    print("Use --n-only <N> to process a single sample size only")
    print("Use --n-jobs <N> to use parallel processing")
    print("Use --clear-cache to clear caches for the selected DGP(s)")
    print("Use --overwrite-cache to recompute from scratch")
    print("Use --plots-only to regenerate plots from cache without computation")
    if args.latex_table:
        create_coverage_latex_tables(results_dict, targeters=targeters, save_dir="paper/resources/target_estimand_variance", n_suffix=n_suffix)

if __name__ == "__main__":
    main()