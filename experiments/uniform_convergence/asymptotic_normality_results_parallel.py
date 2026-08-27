#!/usr/bin/env python3
"""
Asymptotic Normality Analysis for CVXPY Estimator

This script analyzes the asymptotic normality properties of the CVXPY estimator
by examining confidence interval widths and coverage probabilities across 
different sample sizes and DGPs.
"""

import os
import sys
import json
import glob
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.interpolate import interp1d
from scipy.stats import norm
from tqdm import tqdm
# NEW: parallel support
from joblib import Parallel, delayed

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

# Import density variance computation functions
from density_variance.density_variance import (
    estimate_covariance_beta,
    density_confidence_interval
)

with open('dpg-name-mapping.json', 'r') as f:
    DGP_NAME_MAPPING = json.load(f)

SCALING_RIDGE_PARAM = 1e-5

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

DISTRIBUTION_SCALE = {
    "TruncatedNormal": {
        "default": 10.0
    },
    "TruncatedGMMSymmetricThree": {
        "default": 0.1
    },
    "TruncatedGMMAsymmetricThree": {
        "default": 1.0
    },
    "TruncatedGMMFiveSpikes": {
        "default": 0.001
    },
    "StepFunction": {
        "default": 0.01
    },
    "Sinusoidal": {
        "default": 10.0
    }
}

def get_distribution_scale(distribution, sample_size):
    """
    Get the distribution scale for a specific distribution and sample size.
    
    Args:
        distribution: Distribution name (e.g., "TruncatedGMMFiveSpikes")
        sample_size: Sample size (e.g., 1600)
    
    Returns:
        float: Scale factor for this distribution and sample size
    """
    if distribution not in DISTRIBUTION_SCALE:
        return 1.0  # Default fallback
    
    dist_config = DISTRIBUTION_SCALE[distribution]
    
    # If it's a simple scalar (backward compatibility)
    if isinstance(dist_config, (int, float)):
        return dist_config
    
    # If it's a dictionary, look for sample size specific value
    if isinstance(dist_config, dict):
        # First try exact sample size match
        if sample_size in dist_config:
            return dist_config[sample_size]
        # Fallback to default
        elif "default" in dist_config:
            return dist_config["default"]
        else:
            return 1.0  # Ultimate fallback
    
    return 1.0  # Ultimate fallback

def load_cvxpy_results(results_dir="experiments/uniform_convergence/results", 
                       dgp_names=None, sample_sizes=None, max_files_per_combo=None):
    """
    Load CVXPY estimator results from JSON files.
    
    Args:
        results_dir: Directory containing results
        dgp_names: List of DGP names to include (default: all available)
        sample_sizes: List of sample sizes to include (default: specified range)
        max_files_per_combo: Maximum number of files to load per DGP/sample size combo
    
    Returns:
        dict: Dictionary with structure {dgp_name: {n_sample: [results_list]}}
    """
    if sample_sizes is None:
        sample_sizes = [25, 50, 100, 200, 400, 800, 1600, 3200]
    
    if dgp_names is None:
        dgp_names = ["Sinusoidal", "StepFunction", "TruncatedNormal", 
                     "TruncatedGMMSymmetricThree", "TruncatedGMMAsymmetricThree",
                     "TruncatedGMMFiveSpikes"]
    
    results = {}
    
    for dgp_name in tqdm(dgp_names, desc="Loading DGPs"):
        results[dgp_name] = {}
        
        for n_sample in sample_sizes:
            results[dgp_name][n_sample] = []
            
            # Find directory for this combination
            pattern = f"{dgp_name}_CVXPYEstimator_N{n_sample}"
            dir_path = os.path.join(results_dir, pattern)
            
            if not os.path.exists(dir_path):
                continue
            
            # Load all JSON files in this directory
            json_files = glob.glob(os.path.join(dir_path, "*.json"))
            
            # Limit number of files if specified
            if max_files_per_combo and len(json_files) > max_files_per_combo:
                json_files = json_files[:max_files_per_combo]
            
            for json_file in tqdm(json_files, desc=f"Loading {dgp_name} N{n_sample}", leave=False):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract seed from filename
                    filename = os.path.basename(json_file)
                    if filename.startswith('seed_') and filename.endswith('.json'):
                        seed = filename[5:-5]
                        data['filename_seed'] = seed
                    
                    # Check if we have the required data structure
                    if 'HAL_results' in data and 'hyperparams' in data:
                        # Check if any density value exceeds 20 - if so, skip this result
                        # if 'estimated_density' in data['HAL_results']:
                        #     estimated_density = np.array(data['HAL_results']['estimated_density'])
                        #     if np.any(estimated_density > 20):
                        #         continue  # Skip this result as density exceeds 20 at some point
                        
                        results[dgp_name][n_sample].append(data)
                        
                except (json.JSONDecodeError, Exception) as e:
                    pass  # Skip failed files silently
            
            # print(f"  Loaded {len(results[dgp_name][n_sample])} results for N={n_sample}")
    
    # Print summary of loaded data
    print("\nLoaded data summary:")
    for dgp_name in results.keys():
        total_experiments = sum(len(experiments) for experiments in results[dgp_name].values())
        print(f"  {dgp_name}: {total_experiments} total experiments")
        for n_sample, experiments in results[dgp_name].items():
            if len(experiments) > 0:
                print(f"    N={n_sample}: {len(experiments)} experiments")
    
    return results

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

def process_single_result(result_data, eval_points, true_density_values, distribution):
    """
    Process a single experimental result to compute confidence intervals and coverage.
    
    Args:
        result_data: Single experiment result from JSON
        eval_points: 201 evaluation points from 0 to 1
        true_density_values: True density at evaluation points
        distribution: Distribution name for getting the scale factor
    
    Returns:
        dict: Contains density estimates, standard errors, CI bounds, and coverage indicators
    """
    try:
        # Extract data points
        data_points = result_data['HAL_results']['data_points']
        data = pd.DataFrame({'W1': data_points})
        
        # Get sample size from data
        sample_size = len(data_points)
        
        # Adapt result structure for density variance functions
        adapted_results = {
            'results': result_data['HAL_results'],
            'estimator_setup': {
                'estimator_params': {
                    'basis_order': result_data['hyperparams']['basis_order']
                }
            }
        }
        n_knots = result_data['HAL_results']["n_selected_knots"]

        sample_size = len(data_points)
        
        # Get distribution scale for this specific distribution and sample size
        dist_scale = get_distribution_scale(distribution, sample_size)

        scaling_factor = SCALING_RIDGE_PARAM * dist_scale * (sample_size ** (-1/5))
        
        # Compute covariance matrix
        cov_beta = estimate_covariance_beta(
            data, 
            adapted_results, 
            ridge_param=scaling_factor
        )
        
        # Compute confidence intervals at evaluation points
        ci_df = density_confidence_interval(
            eval_points, 
            adapted_results, 
            cov_beta, 
            alpha=0.05
        )
        
        # Determine data range for valid coverage computation
        x_min = np.min(data_points)
        x_max = np.max(data_points)
        valid_mask = (eval_points >= x_min) & (eval_points <= x_max)
        
        # Check coverage: true density within CI bounds
        # Only compute coverage within the data domain [x_min, x_max]
        coverage = np.full(len(eval_points), np.nan)  # Initialize with NaN
        coverage[valid_mask] = ((true_density_values[valid_mask] >= ci_df['lower'].values[valid_mask]) & 
                                (true_density_values[valid_mask] <= ci_df['upper'].values[valid_mask])).astype(float)
        
        # Compute CI widths
        ci_widths = np.array(ci_df['upper']) - np.array(ci_df['lower'])
        
        return {
            'density': ci_df['density'].values,
            'se': ci_df['se'].values,
            'lower': ci_df['lower'].values,
            'upper': ci_df['upper'].values,
            'ci_width': ci_widths,
            'coverage': coverage,
            'success': True
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def analyze_asymptotic_normality(results_dict, eval_points, oracle_stats, n_jobs=1):
    """
    Analyze asymptotic normality across all DGPs and sample sizes.
    
    Args:
        results_dict: Dictionary of experimental results (limited by max_files_per_combo)
        eval_points: Evaluation points for density computation
        oracle_stats: Pre-computed oracle statistics using all data
        n_jobs: Number of parallel jobs to use (default: 1)
    
    Returns:
        dict: Analysis results with CI widths and coverage statistics
    """
    analysis_results = {}
    
    for dgp_name in tqdm(results_dict.keys(), desc="Analyzing DGPs"):
        # Compute true density for this DGP
        true_density = compute_true_density(dgp_name, eval_points)
        
        analysis_results[dgp_name] = {}
        
        for n_sample, experiments in results_dict[dgp_name].items():
            if len(experiments) == 0:
                continue
                
            # Parallel processing of experiments with explicit progress info
            print(f"  Processing {dgp_name} N{n_sample}: {len(experiments)} experiments with {n_jobs} jobs...")
            
            # Process experiments in parallel with progress tracking
            processed_list = []
            batch_size = max(1, len(experiments) // 10)  # Process in 10% batches for progress updates
            
            if len(experiments) <= 50 or n_jobs == 1:
                # For small numbers or single job, show individual progress
                for i, exp_result in enumerate(tqdm(experiments, desc=f"  {dgp_name} N{n_sample}", leave=False)):
                    processed = process_single_result(exp_result, eval_points, true_density, dgp_name)
                    processed_list.append(processed)
            else:
                # For large numbers with parallel processing, process all at once but show timing
                import time
                start_time = time.time()
                processed_list = Parallel(n_jobs=n_jobs)(
                    delayed(process_single_result)(exp_result, eval_points, true_density, dgp_name)
                    for exp_result in experiments
                )
                elapsed_time = time.time() - start_time
                print(f"    Completed {len(experiments)} experiments in {elapsed_time:.1f}s ({len(experiments)/elapsed_time:.1f} exp/s)")

            # Collect successful runs
            densities = []
            se_values = []
            ci_widths = []
            coverage_indicators = []
            successful_runs = 0
            
            for processed in processed_list:
                if processed.get('success'):
                    densities.append(processed['density'])
                    se_values.append(processed['se'])
                    ci_widths.append(processed['ci_width'])
                    coverage_indicators.append(processed['coverage'])
                    successful_runs += 1
            
            if successful_runs == 0:
                continue
            
            # Convert to arrays
            densities = np.array(densities)  # shape: (n_experiments, n_eval_points)
            se_values = np.array(se_values)
            ci_widths = np.array(ci_widths)
            coverage_indicators = np.array(coverage_indicators)
            
            # Use pre-computed oracle statistics (from all data)
            if (dgp_name in oracle_stats and 
                n_sample in oracle_stats[dgp_name] and
                oracle_stats[dgp_name][n_sample] is not None):
                oracle_data = oracle_stats[dgp_name][n_sample]
                oracle_se = oracle_data['oracle_se']
                oracle_ci_width_se = oracle_data['oracle_ci_width_se']
                oracle_ci_width_percentile = oracle_data['oracle_ci_width_percentile']
                oracle_n_experiments = oracle_data['n_experiments']
                
                print(f"  Using oracle stats for {dgp_name} N{n_sample} "
                      f"(based on {oracle_n_experiments} experiments)")
            else:
                # Fallback: compute oracle from current limited data
                print(f"  Warning: No oracle stats found for {dgp_name} N{n_sample}, "
                      f"computing from limited data")
                oracle_se = np.std(densities, axis=0, ddof=1)
                oracle_ci_width_se = 2 * 1.96 * oracle_se
                oracle_ci_width_percentile = (np.percentile(densities, 97.5, axis=0) - 
                                            np.percentile(densities, 2.5, axis=0))
                oracle_n_experiments = successful_runs
            
            # Coverage probability at each evaluation point (use nanmean to ignore out-of-range points)
            coverage_prob = np.nanmean(coverage_indicators, axis=0)
            
            # Compute oracle coverage using oracle SE
            oracle_lower = densities - 1.96 * oracle_se  # shape: (n_experiments, n_eval_points)
            oracle_upper = densities + 1.96 * oracle_se
            oracle_coverage_indicators = ((true_density >= oracle_lower) & 
                                        (true_density <= oracle_upper)).astype(float)
            # Full-range oracle coverage (no in-domain masking)
            oracle_coverage_prob = np.mean(oracle_coverage_indicators, axis=0)
            
            # Compute CI width percentiles for plotting (same as in create_ci_width_plots)
            ci_widths_mean = np.mean(ci_widths, axis=0)
            ci_widths_lower = np.percentile(ci_widths, 25, axis=0)  # 25th percentile
            ci_widths_upper = np.percentile(ci_widths, 75, axis=0)  # 75th percentile
            ci_widths_min = np.percentile(ci_widths, 10, axis=0)   # 10th percentile  
            ci_widths_max = np.percentile(ci_widths, 90, axis=0)   # 90th percentile
            
            analysis_results[dgp_name][n_sample] = {
                'eval_points': eval_points,
                'n_experiments': successful_runs,
                'oracle_n_experiments': oracle_n_experiments,
                'densities': densities,
                'se_values': se_values,
                'ci_widths': ci_widths,
                'coverage_indicators': coverage_indicators,
                'coverage_prob': coverage_prob,
                'oracle_coverage_prob': oracle_coverage_prob,  # NEW: Oracle coverage
                'oracle_se': oracle_se,
                'oracle_ci_width_se': oracle_ci_width_se,
                'oracle_ci_width_percentile': oracle_ci_width_percentile,
                # CI width statistics for plotting
                'ci_widths_mean': ci_widths_mean,
                'ci_widths_lower': ci_widths_lower,  # 25th percentile
                'ci_widths_upper': ci_widths_upper,  # 75th percentile
                'ci_widths_min': ci_widths_min,      # 10th percentile
                'ci_widths_max': ci_widths_max,      # 90th percentile
                # Summary statistics: estimated coverage uses in-domain mask; oracle uses full range
                'mean_coverage': np.nanmean(coverage_prob),
                'mean_oracle_coverage': np.mean(oracle_coverage_prob),  # NEW: Oracle coverage summary
                'mean_ci_width': np.mean(ci_widths_mean),
                'mean_oracle_ci_width_se': np.mean(oracle_ci_width_se),
                'mean_oracle_ci_width_percentile': np.mean(oracle_ci_width_percentile)
            }
            
            # Print progress summary for this combination
            tqdm.write(f"  {dgp_name} N{n_sample}: {successful_runs} successful runs, "
                      f"coverage={np.nanmean(coverage_prob):.3f}, "
                      f"oracle_coverage={np.mean(oracle_coverage_prob):.3f}, "
                      f"CI width={np.mean(np.mean(ci_widths, axis=0)):.4f}")
    
    return analysis_results

def augment_analysis_with_oracle_coverage(analysis_results, oracle_stats, eval_points):
    """
    Augment existing analysis results with oracle coverage computation.
    This is useful when you have cached analysis results but want to add oracle coverage.
    
    Args:
        analysis_results: Existing analysis results dictionary
        oracle_stats: Oracle statistics dictionary
        eval_points: Evaluation points used in the analysis
        
    Returns:
        dict: Updated analysis results with oracle coverage added
    """
    print("Augmenting existing analysis results with oracle coverage...")
    
    for dgp_name in analysis_results.keys():
        # Compute true density for this DGP
        true_density = compute_true_density(dgp_name, eval_points)
        
        for n_sample in analysis_results[dgp_name].keys():
            result = analysis_results[dgp_name][n_sample]
            
            # Skip if oracle coverage already exists
            if 'oracle_coverage_prob' in result:
                print(f"  Oracle coverage already exists for {dgp_name} N{n_sample}, skipping...")
                continue
            
            # Check if we have the necessary data
            if 'densities' not in result:
                print(f"  No densities found for {dgp_name} N{n_sample}, skipping...")
                continue
            
            # Get oracle SE
            oracle_se = None
            if (dgp_name in oracle_stats and 
                n_sample in oracle_stats[dgp_name] and
                oracle_stats[dgp_name][n_sample] is not None):
                oracle_se = oracle_stats[dgp_name][n_sample]['oracle_se']
                oracle_n_experiments = oracle_stats[dgp_name][n_sample]['n_experiments']
                print(f"  Using oracle stats for {dgp_name} N{n_sample} "
                      f"(based on {oracle_n_experiments} experiments)")
            elif 'oracle_se' in result:
                # Use oracle SE already in the result
                oracle_se = result['oracle_se']
                oracle_n_experiments = result.get('oracle_n_experiments', result['n_experiments'])
                print(f"  Using existing oracle SE for {dgp_name} N{n_sample}")
            else:
                # Fallback: compute oracle from current data
                densities = np.array(result['densities'])
                oracle_se = np.std(densities, axis=0, ddof=1)
                oracle_n_experiments = result['n_experiments']
                print(f"  Computing oracle SE from limited data for {dgp_name} N{n_sample}")
            
            # Compute oracle coverage
            densities = np.array(result['densities'])
            oracle_lower = densities - 1.96 * oracle_se
            oracle_upper = densities + 1.96 * oracle_se
            oracle_coverage_indicators = ((true_density >= oracle_lower) & 
                                        (true_density <= oracle_upper)).astype(float)
            # Full-range oracle coverage (no in-domain masking)
            oracle_coverage_prob = np.mean(oracle_coverage_indicators, axis=0)
            
            # Add oracle coverage to result
            result['oracle_coverage_prob'] = oracle_coverage_prob
            result['mean_oracle_coverage'] = np.mean(oracle_coverage_prob)
            if 'oracle_n_experiments' not in result:
                result['oracle_n_experiments'] = oracle_n_experiments
            
            print(f"    Added oracle coverage: {np.mean(oracle_coverage_prob):.3f}")
    
    return analysis_results

def create_ci_width_plots(analysis_results, save_dir="experiments/uniform_convergence", n_suffix=""):
    """
    Create confidence interval width plots. This is now the combined version.
    """
    return create_ci_width_plots_combined(analysis_results, save_dir, n_suffix)

def create_ci_width_plots_combined(analysis_results, save_dir="experiments/uniform_convergence", n_suffix=""):
    """
    Create combined confidence interval width plots showing CI widths at multiple evaluation points.
    Creates a single figure with 6 rows (DGPs) and 8 columns (sample sizes).
    """
    dgp_names = list(analysis_results.keys())
    
    # Find all sample sizes across all DGPs
    all_sample_sizes = set()
    for dgp_name in dgp_names:
        all_sample_sizes.update(analysis_results[dgp_name].keys())
    sample_sizes = sorted(list(all_sample_sizes))
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Creating combined CI width plot with {len(dgp_names)} DGPs and {len(sample_sizes)} sample sizes...")
    
    # Create combined figure: 6 rows (DGPs) x 8 columns (sample sizes) - reduced height
    fig, axes = plt.subplots(len(dgp_names), len(sample_sizes), 
                            figsize=(1.75*len(sample_sizes), 1.5*len(dgp_names)))
    
    # Handle case where there's only one DGP or one sample size
    if len(dgp_names) == 1:
        axes = axes.reshape(1, -1)
    if len(sample_sizes) == 1:
        axes = axes.reshape(-1, 1)
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.1, bottom=0.15)  # More space for bottom legend
    
    # Track row-specific y-limits for each DGP
    row_ylims = {}
    
    # First pass: determine row-specific y-limits for each DGP
    for i, dgp_name in enumerate(dgp_names):
        row_ylims[i] = [float('inf'), -float('inf')]
        
        for j, n_sample in enumerate(sample_sizes):
            if n_sample not in analysis_results[dgp_name]:
                continue
                
            result = analysis_results[dgp_name][n_sample]
            
            # Get CI width statistics
            if all(key in result for key in ['ci_widths_min', 'ci_widths_max']):
                ci_widths_min = result['ci_widths_min']
                ci_widths_max = result['ci_widths_max']
            else:
                ci_widths = result['ci_widths']
                ci_widths_min = np.percentile(ci_widths, 10, axis=0)
                ci_widths_max = np.percentile(ci_widths, 90, axis=0)
            
            # Update row-specific limits
            row_ylims[i][0] = min(row_ylims[i][0], np.min(ci_widths_min))
            row_ylims[i][1] = max(row_ylims[i][1], np.max(ci_widths_max))
        
        # Adjust row limits with buffer
        if row_ylims[i][0] != float('inf'):  # Only if we have data for this row
            row_ylims[i][0] = max(-0.2, row_ylims[i][0] - 0.1)  # Small buffer below
            row_ylims[i][1] = row_ylims[i][1] + 0.1  # Small buffer above
    
    # Second pass: create plots
    for i, dgp_name in enumerate(dgp_names):
        for j, n_sample in enumerate(sample_sizes):
            ax = axes[i, j]
            
            if n_sample not in analysis_results[dgp_name]:
                # Empty subplot for missing data
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_xlim(0, 1)
                if i in row_ylims and row_ylims[i][0] != float('inf'):
                    ax.set_ylim(row_ylims[i])
                ax.grid(True, alpha=0.3)
                continue
            
            result = analysis_results[dgp_name][n_sample]
            eval_points = result['eval_points']
            
            # Use cached CI width percentiles if available, otherwise compute them
            if all(key in result for key in ['ci_widths_mean', 'ci_widths_lower', 'ci_widths_upper', 
                                           'ci_widths_min', 'ci_widths_max']):
                # Use cached percentiles
                ci_widths_mean = result['ci_widths_mean']
                ci_widths_lower = result['ci_widths_lower']  # 25th percentile
                ci_widths_upper = result['ci_widths_upper']  # 75th percentile
                ci_widths_min = result['ci_widths_min']      # 10th percentile  
                ci_widths_max = result['ci_widths_max']      # 90th percentile
            else:
                # Fallback: compute percentiles from raw CI widths
                ci_widths = result['ci_widths']  # shape: (n_experiments, n_eval_points)
                ci_widths_mean = np.mean(ci_widths, axis=0)
                ci_widths_lower = np.percentile(ci_widths, 25, axis=0)  # 25th percentile
                ci_widths_upper = np.percentile(ci_widths, 75, axis=0)  # 75th percentile
                ci_widths_min = np.percentile(ci_widths, 10, axis=0)   # 10th percentile  
                ci_widths_max = np.percentile(ci_widths, 90, axis=0)   # 90th percentile
            
            # Oracle statistics
            oracle_percentile_widths = result['oracle_ci_width_percentile']  # shape: (n_eval_points,)
            
            # Plot confidence bands using fill_between
            ax.fill_between(eval_points, ci_widths_min, ci_widths_max, 
                          color='lightgray', alpha=0.65)
            ax.fill_between(eval_points, ci_widths_lower, ci_widths_upper, 
                          color='dodgerblue', alpha=0.35)
            
            # Plot mean CI width
            ax.plot(eval_points, ci_widths_mean, color='black', linewidth=1)
            
            # Add oracle line
            ax.plot(eval_points, oracle_percentile_widths, color='red', linestyle='--', 
                   linewidth=1)
            
            # Customize plot
            ax.set_xlim(0, 1)
            if i in row_ylims and row_ylims[i][0] != float('inf'):
                ax.set_ylim(row_ylims[i])
            ax.grid(True, alpha=0.3)
            
            # Remove x-ticks at the ends (0 and 1)
            xticks = ax.get_xticks()
            xticks = xticks[(xticks > 0) & (xticks < 1)]
            ax.set_xticks(xticks)
            
            # Title only for top row (sample sizes)
            if i == 0:
                ax.set_title(f'N={n_sample}', fontsize=11)
            
            # Y-axis label only for first column (DGP names)
            if j == 0:
                ax.set_ylabel(f'{DGP_NAME_MAPPING.get(dgp_name, dgp_name)}', fontsize=10, rotation=90)
            else:
                ax.set_yticklabels([])
            
            # X-axis labels only for bottom row
            if i == len(dgp_names) - 1:
                ax.set_xlabel('x', fontsize=9)
            else:
                ax.set_xticklabels([])
    
    # Add overall title
    fig.suptitle('95% Confidence Interval Widths Across DGPs and Sample Sizes', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Create legend elements manually
    legend_elements = [
        mpatches.Patch(color='lightgray', alpha=0.65, label='10th-90th percentile'),
        mpatches.Patch(color='dodgerblue', alpha=0.35, label='25th-75th percentile'),
        mlines.Line2D([], [], color='black', linewidth=1, label='Mean CI Width'),
        mlines.Line2D([], [], color='red', linestyle='--', linewidth=1, label='Oracle (Percentile-based)')
    ]
    
    # Add legend at the bottom
    fig.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, -0.035), ncol=4, fontsize=11)
    
    plt.tight_layout()
    
    # Save the combined plot
    save_path = os.path.join(save_dir, f"Figure_CI_Widths_Combined{n_suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.4)
    print(f"  Saved combined CI width plot: {save_path}")
    
    plt.close()  # Close the figure to free memory

def create_coverage_plots(analysis_results, save_dir="experiments/uniform_convergence", n_suffix=""):
    """
    Create coverage probability plots. This is now the combined version.
    """
    return create_coverage_plots_combined(analysis_results, save_dir, n_suffix)

def create_coverage_plots_combined(analysis_results, save_dir="experiments/uniform_convergence", n_suffix=""):
    """
    Create combined coverage probability plots showing coverage across DGPs and sample sizes.
    Creates a single figure with 6 rows (DGPs) and 8 columns (sample sizes).
    Shows both estimated coverage and oracle coverage.
    """
    dgp_names = list(analysis_results.keys())
    
    # Find all sample sizes across all DGPs
    all_sample_sizes = set()
    for dgp_name in dgp_names:
        all_sample_sizes.update(analysis_results[dgp_name].keys())
    sample_sizes = sorted(list(all_sample_sizes))
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Creating combined coverage plot with {len(dgp_names)} DGPs and {len(sample_sizes)} sample sizes...")
    
    # Create combined figure: 6 rows (DGPs) x 8 columns (sample sizes) - reduced height
    fig, axes = plt.subplots(len(dgp_names), len(sample_sizes), 
                            figsize=(1.75*len(sample_sizes), 1.5*len(dgp_names)))
    
    # Handle case where there's only one DGP or one sample size
    if len(dgp_names) == 1:
        axes = axes.reshape(1, -1)
    if len(sample_sizes) == 1:
        axes = axes.reshape(-1, 1)
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.1, bottom=0.15)  # More space for bottom legend
    
    # Create plots
    for i, dgp_name in enumerate(dgp_names):
        for j, n_sample in enumerate(sample_sizes):
            ax = axes[i, j]
            
            if n_sample not in analysis_results[dgp_name]:
                # Empty subplot for missing data
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_ylim(0, 101)
                ax.grid(True, alpha=0.3)
                continue
            
            result = analysis_results[dgp_name][n_sample]
            
            # Coverage probabilities across evaluation points
            coverage_prob = result['coverage_prob'] * 100  # Convert to percentage
            # Filter out NaNs so plotting/means work when coverage is undefined outside data domain
            coverage_valid = coverage_prob[~np.isnan(coverage_prob)]
            
            # Create a simple boxplot for this single sample size using valid values only
            if coverage_valid.size > 0:
                bp = ax.boxplot([coverage_valid], positions=[0], widths=0.6, 
                              patch_artist=True, showfliers=True)
            else:
                bp = None
            
            # Customize boxplot appearance
            if bp is not None:
                for box in bp['boxes']:
                    box.set(facecolor='lightblue', alpha=0.7)
            
            # Add mean point for estimated coverage
            if coverage_valid.size > 0:
                mean_coverage = np.mean(coverage_valid)
                ax.plot([0], [mean_coverage], color='blue', marker='o', 
                       markersize=6, label='Mean Estimated Coverage' if i == 0 and j == 0 else "")
            
            # Add oracle coverage (if available)
            if 'oracle_coverage_prob' in result:
                oracle_coverage = result['oracle_coverage_prob'] * 100  # Convert to percentage
                oracle_valid = oracle_coverage[~np.isnan(oracle_coverage)]
                if oracle_valid.size > 0:
                    mean_oracle_coverage = np.mean(oracle_valid)
                    ax.plot([0], [mean_oracle_coverage], color='green', marker='s', 
                           markersize=6, label='Oracle Coverage' if i == 0 and j == 0 else "")
            
            # Add 95% target line
            ax.axhline(y=95, color='red', linestyle='--', linewidth=1, 
                      label='95% Target' if i == 0 and j == 0 else "")
            
            # Customize plot
            ax.set_ylim(0, 101)
            ax.set_xlim(-0.5, 0.5)
            ax.grid(True, alpha=0.3)
            
            # Remove x-axis ticks and labels
            ax.set_xticks([])
            
            # # Title only for top row (sample sizes)
            # if i == 0:
            #     ax.set_title(f'N={n_sample}', fontsize=11)
            
            # Y-axis label only for first column (DGP names)
            if j == 0:
                ax.set_ylabel(f'{DGP_NAME_MAPPING.get(dgp_name, dgp_name)}', fontsize=10, rotation=90)
            else:
                ax.set_yticklabels([])
            
            # X-axis labels only for bottom row
            if i == len(dgp_names) - 1:
                ax.set_xlabel(f'{n_sample}', fontsize=9)
    
    # Add overall title
    fig.suptitle('95% Coverage Probabilities Across DGPs and Sample Sizes', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Create legend elements manually
    legend_elements = [
        mlines.Line2D([], [], color='blue', marker='o', linestyle='', markersize=6, label='Mean Estimated Coverage'),
        mlines.Line2D([], [], color='green', marker='s', linestyle='', markersize=6, label='Oracle Coverage'),
        mlines.Line2D([], [], color='red', linestyle='--', linewidth=1, label='95% Target')
    ]
    
    # Add legend at the bottom
    fig.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, -0.035), ncol=3, fontsize=11)
    
    plt.tight_layout()
    
    # Save the combined plot
    save_path = os.path.join(save_dir, f"Figure_Coverage_Combined{n_suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.4)
    print(f"  Saved combined coverage plot: {save_path}")
    
    plt.close()  # Close the figure to free memory

def create_coverage_latex_table(analysis_results, save_dir="paper/resources/density_asymptotic_normality_and_var_est"):
    """
    Create a LaTeX table for coverage results with 6 rows (DGPs) and 8 columns (sample sizes).
    Each cell contains (our coverage, oracle coverage) as a tuple.
    
    Args:
        analysis_results: Dictionary containing analysis results
        save_dir: Directory to save the LaTeX table
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Define DGP order for consistent presentation (same as uniform convergence)
    dgp_order = [
        "TruncatedNormal",
        "TruncatedGMMSymmetricThree", 
        "TruncatedGMMAsymmetricThree",
        "TruncatedGMMFiveSpikes",
        "StepFunction",
        "Sinusoidal"
    ]
    
    # Define sample sizes in order
    sample_sizes = [25, 50, 100, 200, 400, 800, 1600, 3200]
    
    try:
        latex_lines = []
        latex_lines.append("% Auto-generated coverage analysis table")
        latex_lines.append("\\begin{table}[H]")
        latex_lines.append("  \\centering")
        latex_lines.append("  \\scriptsize")  # Use scriptsize for more space
        latex_lines.append("  \\setlength{\\tabcolsep}{2pt}")  # Reduce column separation
        latex_lines.append("  \\renewcommand{\\arraystretch}{1.1}")
        latex_lines.append("  \\begin{tabular}{l" + "c" * len(sample_sizes) + "}")
        latex_lines.append("    \\toprule")
        
        # Column headers (sample sizes)
        header_row = ["DGP"] + [f"N={n}" for n in sample_sizes]
        latex_lines.append("    " + " & ".join(header_row) + " \\\\")
        latex_lines.append("    \\midrule")
        
        # Data rows for each DGP
        for dgp in dgp_order:
            if dgp not in analysis_results:
                # DGP not in results, fill with dashes
                row_data = [DGP_NAME_MAPPING.get(dgp, dgp)] + ["--"] * len(sample_sizes)
                latex_lines.append("    " + " & ".join(row_data) + " \\\\")
                continue
            
            dgp_results = analysis_results[dgp]
            row_data = [DGP_NAME_MAPPING.get(dgp, dgp)]
            
            for n_sample in sample_sizes:
                if n_sample in dgp_results:
                    result = dgp_results[n_sample]
                    our_coverage = result.get('mean_coverage', 0.0) * 100  # Convert to percentage
                    oracle_coverage = result.get('mean_oracle_coverage', 0.0) * 100  # Convert to percentage
                    
                    # Format as (our_coverage, oracle_coverage) with 1 decimal place
                    cell_content = f"({our_coverage:.1f}, {oracle_coverage:.1f})"
                    row_data.append(cell_content)
                else:
                    row_data.append("--")
            
            latex_lines.append("    " + " & ".join(row_data) + " \\\\")
        
        latex_lines.append("    \\bottomrule")
        latex_lines.append("  \\end{tabular}")
        latex_lines.append("  \\caption{Coverage probabilities (\\%) for 95\\% confidence intervals across DGPs and sample sizes. Each cell shows (estimated coverage, oracle coverage). Estimated coverage uses plug-in standard errors with ridge regularization. Oracle coverage uses the empirical standard deviation across Monte Carlo replicates.}")
        latex_lines.append("  \\label{tab:coverage_analysis}")
        latex_lines.append("\\end{table}")
        
        # Save LaTeX table
        latex_path = os.path.join(save_dir, "coverage_analysis_table.tex")
        with open(latex_path, 'w') as f:
            f.write("\n".join(latex_lines))
        
        print(f"Saved coverage LaTeX table: {latex_path}")
        
    except Exception as e:
        print(f"Failed to create coverage LaTeX table: {e}")

def create_summary_table(analysis_results):
    """
    Create a summary table of the asymptotic normality analysis.
    """
    print("\n" + "="*100)
    print("ASYMPTOTIC NORMALITY ANALYSIS SUMMARY")
    print("="*100)
    
    for dgp_name in analysis_results.keys():
        print(f"\n{dgp_name}:")
        print("-" * 90)
        print(f"{'Sample Size':<12} {'N Exp':<8} {'Oracle N':<10} {'Mean Coverage':<15} {'Oracle Cov':<12} {'CI Width':<12} {'Oracle SE':<12} {'Oracle %ile':<12}")
        print("-" * 90)
        
        for n_sample in sorted(analysis_results[dgp_name].keys(), key=lambda x: int(x)):
            result = analysis_results[dgp_name][n_sample]
            oracle_n = result.get('oracle_n_experiments', result['n_experiments'])
            oracle_coverage = result.get('mean_oracle_coverage', 0.0)
            print(f"{n_sample:<12} {result['n_experiments']:<8} {oracle_n:<10} "
                  f"{result['mean_coverage']:<15.3f} "
                  f"{oracle_coverage:<12.3f} "
                  f"{result['mean_ci_width']:<12.4f} "
                  f"{result['mean_oracle_ci_width_se']:<12.4f} "
                  f"{result['mean_oracle_ci_width_percentile']:<12.4f}")

def process_single_result_for_oracle(result_data, eval_points):
    """
    Process a single experimental result to extract density estimates only (for oracle computation).
    This is simpler than process_single_result since we don't need CIs, just the density estimates.
    
    Args:
        result_data: Single experiment result from JSON
        eval_points: 201 evaluation points from 0 to 1
    
    Returns:
        dict: Contains density estimates only, or error info
    """
    try:
        # Extract HAL results and interpolate to evaluation points
        stored_density = np.array(result_data['HAL_results']['estimated_density'])
        stored_grid = np.array(result_data['HAL_results']['grid_points'])
        
        # Check for NaN values in stored density
        if np.any(np.isnan(stored_density)) or np.any(np.isnan(stored_grid)):
            return {
                'success': False,
                'error': 'NaN values found in stored density or grid'
            }
        
        # Interpolate stored density to our evaluation points
        density_interp = interp1d(stored_grid, stored_density, 
                                kind='linear', bounds_error=False, fill_value=0)
        density_at_eval_points = density_interp(eval_points)
        
        # Check for NaN values in interpolated density
        if np.any(np.isnan(density_at_eval_points)):
            return {
                'success': False,
                'error': 'NaN values found in interpolated density'
            }
        
        return {
            'density': density_at_eval_points,
            'success': True
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def compute_oracle_statistics(results_dir="experiments/uniform_convergence/results", 
                             eval_points=None, dgp_names=None, sample_sizes=None,
                             cache_dir="experiments/uniform_convergence/cache",
                             n_jobs=1, n_suffix=""):
    """
    Compute oracle statistics using ALL available experimental results.
    Cache results separately for each DGP to avoid large files.
    
    Args:
        results_dir: Directory containing results
        eval_points: Evaluation points for density computation
        dgp_names: List of DGP names to include
        sample_sizes: List of sample sizes to include
        cache_dir: Directory to store cache files
        
    Returns:
        dict: Oracle statistics for each DGP and sample size
    """
    if eval_points is None:
        eval_points = np.linspace(0, 1, 201)
        
    if sample_sizes is None:
        sample_sizes = [25, 50, 100, 200, 400, 800, 1600, 3200]
        
    if dgp_names is None:
        dgp_names = ["Sinusoidal", "StepFunction", "TruncatedNormal", 
                     "TruncatedGMMSymmetricThree", "TruncatedGMMAsymmetricThree",
                     "TruncatedGMMFiveSpikes"]
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache key for validation
    cache_key = {
        'eval_points_hash': hash(tuple(eval_points)),
        'sample_sizes': sorted(sample_sizes),
        'n_eval_points': len(eval_points)
    }
    
    oracle_stats = {}
    
    for dgp_name in dgp_names:
        print(f"\nProcessing oracle statistics for {dgp_name}...")
        
        # DGP-specific cache file
        dgp_cache_file = os.path.join(cache_dir, f"oracle_{dgp_name}{n_suffix}.json")
        
        # Try to load cached results for this DGP
        dgp_oracle_stats = None
        if os.path.exists(dgp_cache_file):
            try:
                with open(dgp_cache_file, 'r') as f:
                    cached_data = json.load(f)
                    
                # Check if cache is valid
                if (cached_data.get('cache_key') == cache_key and 
                    'oracle_stats' in cached_data):
                    print(f"  Loading {dgp_name} oracle statistics from cache...")
                    
                    # Convert lists back to numpy arrays
                    dgp_oracle_stats = cached_data['oracle_stats']
                    for n_sample in dgp_oracle_stats:
                        data = dgp_oracle_stats[n_sample]
                        data['eval_points'] = np.array(data['eval_points'])
                        data['oracle_se'] = np.array(data['oracle_se'])
                        data['oracle_ci_width_se'] = np.array(data['oracle_ci_width_se'])
                        data['oracle_ci_width_percentile'] = np.array(data['oracle_ci_width_percentile'])
                    
                    oracle_stats[dgp_name] = dgp_oracle_stats
                    continue
                    
            except (json.JSONDecodeError, KeyError, Exception) as e:
                print(f"  Cache file for {dgp_name} corrupted or invalid, recomputing...")
        
        # Compute oracle statistics for this DGP
        print(f"  Computing oracle statistics for {dgp_name} using ALL available data...")
        
        # Load ALL results for this DGP only
        dgp_results = load_cvxpy_results(
            results_dir=results_dir,
            dgp_names=[dgp_name],  # Single DGP
            sample_sizes=sample_sizes,
            max_files_per_combo=None  # Load all files
        )
        
        if dgp_name not in dgp_results:
            print(f"  No results found for {dgp_name}, skipping...")
            continue
            
        dgp_oracle_stats = {}
        
        for n_sample, experiments in dgp_results[dgp_name].items():
            if len(experiments) == 0:
                continue
                
            print(f"    Processing {dgp_name} N{n_sample}: {len(experiments)} experiments")
            
            # Parallel processing of experiments with progress tracking
            if len(experiments) <= 50 or n_jobs == 1:
                # For small numbers or single job, show individual progress
                processed_list = []
                for exp_result in tqdm(experiments, desc=f"    Oracle {dgp_name} N{n_sample}", leave=False):
                    processed = process_single_result_for_oracle(exp_result, eval_points)
                    processed_list.append(processed)
            else:
                # For large numbers with parallel processing, process all at once but show timing
                import time
                start_time = time.time()
                processed_list = Parallel(n_jobs=n_jobs)(
                    delayed(process_single_result_for_oracle)(exp_result, eval_points)
                    for exp_result in experiments
                )
                elapsed_time = time.time() - start_time
                print(f"      Completed {len(experiments)} experiments in {elapsed_time:.1f}s ({len(experiments)/elapsed_time:.1f} exp/s)")
            
            densities = []
            successful_runs = 0
            for processed in processed_list:
                if processed.get('success'):
                    densities.append(processed['density'])
                    successful_runs += 1
            
            if successful_runs == 0:
                continue
            
            # Convert to arrays and filter NaN
            densities = np.array(densities)
            
            if np.any(np.isnan(densities)):
                print(f"      Warning: Found NaN values, filtering...")
                nan_mask = np.isnan(densities).any(axis=1)
                densities = densities[~nan_mask]
                successful_runs = densities.shape[0]
                print(f"      After filtering NaN: {successful_runs} successful experiments")
                
                if successful_runs == 0:
                    print(f"      No valid experiments after NaN filtering, skipping")
                    continue
            
            # Compute oracle statistics
            oracle_se = np.std(densities, axis=0, ddof=1)
            oracle_ci_width_se = 2 * 1.96 * oracle_se
            oracle_ci_width_percentile = (np.percentile(densities, 97.5, axis=0) - 
                                        np.percentile(densities, 2.5, axis=0))
            
            dgp_oracle_stats[n_sample] = {
                'eval_points': eval_points.tolist(),
                'n_experiments': successful_runs,
                'oracle_se': oracle_se.tolist(),
                'oracle_ci_width_se': oracle_ci_width_se.tolist(),
                'oracle_ci_width_percentile': oracle_ci_width_percentile.tolist(),
                'mean_oracle_ci_width_se': float(np.mean(oracle_ci_width_se)),
                'mean_oracle_ci_width_percentile': float(np.mean(oracle_ci_width_percentile))
            }
            
            print(f"      Oracle computed: {successful_runs} experiments, "
                  f"SE={np.mean(oracle_se):.4f}, "
                  f"SE width={np.mean(oracle_ci_width_se):.4f}, "
                  f"Percentile width={np.mean(oracle_ci_width_percentile):.4f}")
        
        # Cache the results for this DGP
        if dgp_oracle_stats:
            cache_data = {
                'cache_key': cache_key,
                'dgp_name': dgp_name,
                'oracle_stats': dgp_oracle_stats,
                'computed_at': pd.Timestamp.now().isoformat(),
                'total_experiments': sum(data['n_experiments'] for data in dgp_oracle_stats.values())
            }
            
            with open(dgp_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            print(f"    Oracle statistics for {dgp_name} cached to {dgp_cache_file}")
            
            # Convert lists back to numpy arrays for return
            for n_sample in dgp_oracle_stats:
                data = dgp_oracle_stats[n_sample]
                data['eval_points'] = np.array(data['eval_points'])
                data['oracle_se'] = np.array(data['oracle_se'])
                data['oracle_ci_width_se'] = np.array(data['oracle_ci_width_se'])
                data['oracle_ci_width_percentile'] = np.array(data['oracle_ci_width_percentile'])
            
            oracle_stats[dgp_name] = dgp_oracle_stats
    
    return oracle_stats

def save_analysis_results(analysis_results, cache_dir="experiments/uniform_convergence/cache", n_suffix=""):
    """
    Save analysis results to separate cache files for each DGP.
    
    Args:
        analysis_results: Dictionary containing analysis results
        cache_dir: Directory to store cache files
        n_suffix: Suffix to add to filenames (e.g., "_N25" for single N analysis)
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    for dgp_name in analysis_results:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        
        for n_sample in analysis_results[dgp_name]:
            result = analysis_results[dgp_name][n_sample].copy()
            
            # Convert numpy arrays to lists
            for key in ['eval_points', 'oracle_se', 'oracle_ci_width_se', 'oracle_ci_width_percentile',
                       'ci_widths_mean', 'ci_widths_lower', 'ci_widths_upper', 'ci_widths_min', 'ci_widths_max']:
                if key in result and isinstance(result[key], np.ndarray):
                    result[key] = result[key].tolist()
            
            # Convert larger arrays to lists
            for key in ['densities', 'se_values', 'ci_widths', 'coverage_indicators', 'coverage_prob', 'oracle_coverage_prob']:
                if key in result and isinstance(result[key], np.ndarray):
                    result[key] = result[key].tolist()
            
            serializable_results[n_sample] = result
        
        # Create cache data for this DGP
        cache_data = {
            'dgp_name': dgp_name,
            'analysis_results': serializable_results,
            'computed_at': pd.Timestamp.now().isoformat(),
            'cache_version': '1.0'
        }
        
        # Save to DGP-specific file
        dgp_cache_file = os.path.join(cache_dir, f"analysis_{dgp_name}{n_suffix}.json")
        with open(dgp_cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"Analysis results for {dgp_name} cached to {dgp_cache_file}")

def load_analysis_results(dgp_names, cache_dir="experiments/uniform_convergence/cache", n_suffix=""):
    """
    Load analysis results from separate cache files for each DGP.
    
    Args:
        dgp_names: List of DGP names to load
        cache_dir: Directory containing cache files
        n_suffix: Suffix to add to filenames (e.g., "_N25" for single N analysis)
        
    Returns:
        dict: Analysis results with numpy arrays restored, or None if no cache found
    """
    analysis_results = {}
    
    for dgp_name in dgp_names:
        dgp_cache_file = os.path.join(cache_dir, f"analysis_{dgp_name}{n_suffix}.json")
        
        if not os.path.exists(dgp_cache_file):
            print(f"No cache found for {dgp_name}")
            continue
        
        try:
            with open(dgp_cache_file, 'r') as f:
                cached_data = json.load(f)
            
            dgp_results = cached_data['analysis_results']
            
            # Convert string keys back to integers and lists back to numpy arrays
            converted_results = {}
            for n_sample_str in dgp_results:
                n_sample_int = int(n_sample_str)
                result = dgp_results[n_sample_str]
                
                # Convert specific arrays back to numpy
                for key in ['eval_points', 'oracle_se', 'oracle_ci_width_se', 'oracle_ci_width_percentile', 'coverage_prob', 'oracle_coverage_prob',
                           'ci_widths_mean', 'ci_widths_lower', 'ci_widths_upper', 'ci_widths_min', 'ci_widths_max']:
                    if key in result and isinstance(result[key], list):
                        result[key] = np.array(result[key])
                
                # Convert larger arrays back to numpy
                for key in ['densities', 'se_values', 'ci_widths', 'coverage_indicators']:
                    if key in result and isinstance(result[key], list):
                        result[key] = np.array(result[key])
                
                converted_results[n_sample_int] = result
            
            analysis_results[dgp_name] = converted_results
            print(f"Analysis results loaded for {dgp_name} from cache")
            
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Failed to load analysis results cache for {dgp_name}: {e}")
    
    return analysis_results if analysis_results else None

def clear_caches(dgp_names=None, oracle_cache=True, analysis_cache=True, cache_dir="experiments/uniform_convergence/cache", n_suffix=""):
    """
    Clear cached results for specific DGPs or all DGPs.
    
    Args:
        dgp_names: List of DGP names to clear cache for (None = all DGPs)
        oracle_cache: Whether to clear oracle statistics cache
        analysis_cache: Whether to clear analysis results cache
        cache_dir: Directory containing cache files
        n_suffix: Suffix to add to filenames (e.g., "_N25" for single N analysis)
    """
    if dgp_names is None:
        dgp_names = ["Sinusoidal", "StepFunction", "TruncatedNormal", 
                     "TruncatedGMMSymmetricThree", "TruncatedGMMAsymmetricThree",
                     "TruncatedGMMFiveSpikes"]
    
    cleared = []
    
    for dgp_name in dgp_names:
        if oracle_cache:
            oracle_file = os.path.join(cache_dir, f"oracle_{dgp_name}{n_suffix}.json")
            if os.path.exists(oracle_file):
                os.remove(oracle_file)
                cleared.append(f"oracle_{dgp_name}{n_suffix}")
                
        if analysis_cache:
            analysis_file = os.path.join(cache_dir, f"analysis_{dgp_name}{n_suffix}.json")
            if os.path.exists(analysis_file):
                os.remove(analysis_file)
                cleared.append(f"analysis_{dgp_name}{n_suffix}")
    
    if cleared:
        print(f"Cleared cached: {', '.join(cleared)}")
    else:
        print("No cache files found to clear")

def main():
    """
    Main function to run the asymptotic normality analysis.
    """
    parser = argparse.ArgumentParser(description="Asymptotic Normality Analysis for CVXPY Estimator")
    parser.add_argument("--clear-cache", choices=["oracle", "analysis", "all"], 
                       help="Clear cached results before running")
    parser.add_argument("--force-recompute", action="store_true",
                       help="Force recomputation of analysis results (ignores cache)")
    parser.add_argument("--plots-only", action="store_true",
                       help="Only generate plots (requires existing cache)")
    parser.add_argument("--max-files", type=int, default=1000,
                       help="Maximum files per DGP/sample size combo (default: 1000)")
    parser.add_argument("--add-oracle-coverage", action="store_true",
                       help="Add oracle coverage to existing cached analysis results")
    parser.add_argument("--dgp", type=str, default=None,
                       help="Run analysis for specific DGP only (e.g., 'TruncatedGMMFiveSpikes')")
    # NEW: specific N and parallel jobs
    parser.add_argument("--n-only", type=int, default=None,
                       help="Only process a single sample size N (overrides default list)")
    parser.add_argument("--n-jobs", type=int, default=1,
                       help="Number of parallel jobs to use (-1 for all cores)")

    args = parser.parse_args()
    
    print("Starting Asymptotic Normality Analysis for CVXPY Estimator")
    print("="*60)
    
    # Set DGP names based on command line argument
    if args.dgp:
        dgp_names = [args.dgp]
        print(f"Running analysis for single DGP: {args.dgp}")
    else:
        dgp_names = ["Sinusoidal", "StepFunction", "TruncatedNormal", 
                     "TruncatedGMMSymmetricThree", "TruncatedGMMAsymmetricThree",
                     "TruncatedGMMFiveSpikes"]
        print("Running analysis for all DGPs")
    
    # Configuration (moved up to set cache_dir first)
    eval_points = np.linspace(0, 1, 201)  # 201 equally spaced points from 0 to 1
    if args.n_only is not None:
        sample_sizes = [args.n_only]
        print(f"Processing only N={args.n_only}")
        # Create separate directories for single N analysis
        cache_dir = f"experiments/uniform_convergence/cache_N"
        plot_dir = f"experiments/uniform_convergence/plots_N"
        n_suffix = f"_N{args.n_only}"
    else:
        sample_sizes = [25, 50, 100, 200, 400, 800, 1600, 3200]  # Focus on larger sample sizes for asymptotic analysis
        # Use default directories for full analysis
        cache_dir = "experiments/uniform_convergence/cache"
        plot_dir = "paper/resources/density_asymptotic_normality_and_var_est"
        n_suffix = ""
    max_files_per_combo = args.max_files  # Limit number of files to process for efficiency
    n_jobs = args.n_jobs
    
    # Clear caches if requested (now DGP-specific)
    if args.clear_cache:
        if args.clear_cache == "oracle":
            clear_caches(dgp_names, oracle_cache=True, analysis_cache=False, cache_dir=cache_dir, n_suffix=n_suffix)
        elif args.clear_cache == "analysis":
            clear_caches(dgp_names, oracle_cache=False, analysis_cache=True, cache_dir=cache_dir, n_suffix=n_suffix)
        elif args.clear_cache == "all":
            clear_caches(dgp_names, oracle_cache=True, analysis_cache=True, cache_dir=cache_dir, n_suffix=n_suffix)

    
    # Check for existing analysis results cache
    analysis_results = None
    
    if not args.force_recompute and not args.clear_cache:
        analysis_results = load_analysis_results(dgp_names, cache_dir=cache_dir, n_suffix=n_suffix)
        # Filter to only requested DGP if specified (already handled by dgp_names)
    
    if args.plots_only:
        if analysis_results is None:
            print("ERROR: --plots-only requires existing cached analysis results!")
            print("Run without --plots-only first to generate the cache.")
            return
        print("Plots-only mode: Using cached results, skipping all computation...")
    elif args.add_oracle_coverage:
        if analysis_results is None:
            print("ERROR: --add-oracle-coverage requires existing cached analysis results!")
            print("Run without --add-oracle-coverage first to generate the cache.")
            return
        print("Adding oracle coverage to existing cached results...")
        
        # Load oracle statistics
        oracle_stats = compute_oracle_statistics(
            results_dir="experiments/uniform_convergence/results",
            eval_points=eval_points,
            sample_sizes=sample_sizes,
            dgp_names=dgp_names,
            cache_dir=cache_dir,
            n_jobs=n_jobs,
            n_suffix=n_suffix
        )
        
        # Augment analysis results with oracle coverage
        analysis_results = augment_analysis_with_oracle_coverage(analysis_results, oracle_stats, eval_points)
        
        # Save updated analysis results
        print("Saving updated analysis results with oracle coverage...")
        save_analysis_results(analysis_results, cache_dir=cache_dir, n_suffix=n_suffix)
    elif analysis_results is not None and not args.force_recompute:
        print("Found cached analysis results. Skipping computation and proceeding to plotting...")
        print("Use --force-recompute to ignore cache and recompute from scratch.")
    else:
        print("Computing analysis results from scratch...")
        
        # Step 1: Compute/load oracle statistics using ALL available data
        print("Step 1: Computing oracle statistics using all available data...")
        oracle_stats = compute_oracle_statistics(
            results_dir="experiments/uniform_convergence/results",
            eval_points=eval_points,
            sample_sizes=sample_sizes,
            dgp_names=dgp_names,  # Pass the filtered DGP names
            cache_dir=cache_dir,
            n_jobs=n_jobs,
            n_suffix=n_suffix
        )
        
        # Step 2: Load limited results for main analysis
        print("\nStep 2: Loading experimental results for main analysis...")
        results_dict = load_cvxpy_results(
            results_dir="experiments/uniform_convergence/results",
            dgp_names=dgp_names,  # Pass the filtered DGP names
            sample_sizes=sample_sizes,
            max_files_per_combo=max_files_per_combo  # Limited for efficiency
        )
        
        # Filter out DGPs with no results
        results_dict = {k: v for k, v in results_dict.items() if any(len(experiments) > 0 for experiments in v.values())}
        
        if not results_dict:
            print("No experimental results found!")
            return
        
        print(f"Found results for {len(results_dict)} DGPs")
        
        # Step 3: Analyze asymptotic normality using oracle statistics
        print("\nStep 3: Analyzing asymptotic normality...")
        analysis_results = analyze_asymptotic_normality(results_dict, eval_points, oracle_stats, n_jobs=n_jobs)
        
        # Step 4: Save analysis results to cache
        print("\nStep 4: Caching analysis results...")
        save_analysis_results(analysis_results, cache_dir=cache_dir, n_suffix=n_suffix)
    
    # Create summary table
    create_summary_table(analysis_results)
    
    # Create coverage LaTeX table
    print("\nCreating coverage LaTeX table...")
    create_coverage_latex_table(analysis_results, save_dir=plot_dir)
    
    # Create plots
    print("\nCreating plots...")
    create_ci_width_plots(analysis_results, save_dir=plot_dir, n_suffix=n_suffix)
    create_coverage_plots(analysis_results, save_dir=plot_dir, n_suffix=n_suffix)
    
    print("\nAsymptotic normality analysis completed!")
    print(f"Cache files are now stored separately for each DGP in {cache_dir}/")
    print(f"Plots saved in {plot_dir}/")
    print("Use --dgp <DGP_NAME> to run analysis for specific DGPs only")
    print("Use --n-only <N> to process a single sample size only")
    print("Use --n-jobs <N> to use parallel processing (-1 for all cores)")
    print("Use --clear-cache to clear caches for the selected DGP(s)")
    print("Use --force-recompute to ignore cache and recompute from scratch")
    print("Use --plots-only to regenerate plots from cache without computation")
    print("Use --add-oracle-coverage to add oracle coverage to existing cached results")
    print(f"Analysis used {max_files_per_combo} files per DGP/sample size combo (when computed from scratch)")
    if n_jobs > 1:
        print(f"Parallel processing used {n_jobs} jobs")
    if n_suffix:
        print(f"Single N analysis: files suffixed with {n_suffix}")

if __name__ == "__main__":
    main() 