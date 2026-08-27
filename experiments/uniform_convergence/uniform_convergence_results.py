#!/usr/bin/env python3
"""
Uniform Convergence Analysis for CVXPY Estimator

This script analyzes the uniform convergence properties of the CVXPY estimator
by examining maximum deviations between estimated and true densities across
different sample sizes.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import glob
from scipy.interpolate import interp1d
from scipy import stats
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

with open('dpg-name-mapping.json', 'r') as f:
    DGP_NAME_MAPPING = json.load(f)

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

# Define DGP configurations used in experiments (from actual experiment setup)
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

def load_cvxpy_results(results_dir="experiments/uniform_convergence/results", 
                       dgp_names=None, sample_sizes=None):
    """
    Load CVXPY estimator results from JSON files.
    
    Args:
        results_dir: Directory containing results
        dgp_names: List of DGP names to include (default: all available)
        sample_sizes: List of sample sizes to include (default: specified range)
    
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
    
    for dgp_name in dgp_names:
        results[dgp_name] = {}
        
        for n_sample in sample_sizes:
            results[dgp_name][n_sample] = []
            
            # Find directory for this combination
            pattern = f"{dgp_name}_CVXPYEstimator_N{n_sample}"
            dir_path = os.path.join(results_dir, pattern)
            
            if not os.path.exists(dir_path):
                print(f"Warning: Directory {dir_path} not found")
                continue
            
            # Load all JSON files in this directory
            json_files = glob.glob(os.path.join(dir_path, "*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract seed from filename (e.g., "seed_123456.json" -> "123456")
                    filename = os.path.basename(json_file)
                    if filename.startswith('seed_') and filename.endswith('.json'):
                        seed = filename[5:-5]  # Remove "seed_" prefix and ".json" suffix
                        data['filename_seed'] = seed
                    
                    # Check if we have the required data
                    if 'estimated_density' in data and 'density' in data['estimated_density']:
                        if data['estimated_density']['density'] is not None:
                            results[dgp_name][n_sample].append(data)
                        else:
                            print(f"Warning: Null density in {json_file}")
                    else:
                        print(f"Warning: Missing density data in {json_file}")
                        
                except (json.JSONDecodeError, Exception) as e:
                    print(f"Error loading {json_file}: {e}")
            
            print(f"Loaded {len(results[dgp_name][n_sample])} results for {dgp_name} N={n_sample}")
    
    return results

def compute_true_density(dgp_name, eval_points):
    """
    Compute true density for a given DGP at evaluation points.
    
    Args:
        dgp_name: Name of the data generating process
        eval_points: Points at which to evaluate density
    
    Returns:
        np.array: True density values
    """
    if dgp_name not in DGP_CONFIGS:
        raise ValueError(f"Unknown DGP: {dgp_name}")
    
    config = DGP_CONFIGS[dgp_name]
    sampler_class = SAMPLERS[config["sampler"]]
    sampler = sampler_class(**config["sampler_params"])
    
    return sampler.compute_density(eval_points)

def interpolate_density(gridpoints, density_values, eval_points):
    """
    Interpolate density values to uniform evaluation points.
    
    Args:
        gridpoints: Original grid points from the estimator
        density_values: Density values at gridpoints
        eval_points: Points where we want density estimates
    
    Returns:
        np.array: Interpolated density values
    """
    # Convert to numpy arrays and remove any NaN or None values
    gridpoints = np.array(gridpoints)
    density_values = np.array(density_values)
    
    # Create mask for valid values
    valid_mask = ~(np.isnan(density_values) | np.isnan(gridpoints))
    if not np.any(valid_mask):
        return np.full(len(eval_points), np.nan)
    
    valid_gridpoints = gridpoints[valid_mask]
    valid_density = density_values[valid_mask]
    
    # Sort by gridpoints
    sort_idx = np.argsort(valid_gridpoints)
    valid_gridpoints = valid_gridpoints[sort_idx]
    valid_density = valid_density[sort_idx]
    
    # Use numpy interp for simple linear interpolation
    # This handles extrapolation by using boundary values automatically
    interpolated = np.interp(eval_points, valid_gridpoints, valid_density)
    
    return interpolated

def compute_max_deviations(results_dict, eval_points=None):
    """
    Compute maximum deviations for each simulation across all DGPs and sample sizes.
    
    Args:
        results_dict: Results dictionary from load_cvxpy_results
        eval_points: Uniform evaluation points (default: 101 points from 0 to 1)
    
    Returns:
        dict: Dictionary with structure {dgp_name: {n_sample: [max_deviations]}}
    """
    if eval_points is None:
        eval_points = np.linspace(0.0, 1.00, 101)
    
    max_deviations = {}
    high_error_seeds = []  # Track seeds with high errors
    
    for dgp_name, dgp_results in results_dict.items():
        max_deviations[dgp_name] = {}
        
        # Compute true density once for this DGP
        try:
            true_density = compute_true_density(dgp_name, eval_points)
        except Exception as e:
            print(f"Error computing true density for {dgp_name}: {e}")
            continue
        
        for n_sample, simulations in dgp_results.items():
            max_devs = []
            
            for sim_idx, sim_data in enumerate(simulations):
                try:
                    # Extract estimated density
                    gridpoints = sim_data['estimated_density']['gridpoints']
                    density_values = sim_data['estimated_density']['density']
                    
                    # Interpolate to uniform evaluation points
                    estimated_density = interpolate_density(gridpoints, density_values, eval_points)
                    
                    # Skip if interpolation failed
                    if np.all(np.isnan(estimated_density)):
                        print(f"Warning: All NaN estimated density for {dgp_name} N={n_sample}")
                        continue
                    
                    # Compute absolute deviations
                    deviations = np.abs(estimated_density - true_density)
                    
                    # Skip NaN deviations
                    valid_deviations = deviations[~np.isnan(deviations)]
                    if len(valid_deviations) == 0:
                        print(f"Warning: No valid deviations for {dgp_name} N={n_sample}")
                        continue
                    
                    # Compute maximum deviation
                    max_dev = np.max(valid_deviations)
                    max_devs.append(max_dev)
                    
                    # Check if error is greater than 20 and record the seed
                    if max_dev > 20.0:
                        # Extract seed from the simulation data (added during loading)
                        seed = sim_data.get('filename_seed', f'unknown_seed_{sim_idx}')
                        high_error_seeds.append({
                            'dgp': dgp_name,
                            'sample_size': n_sample,
                            'seed': seed,
                            'max_deviation': max_dev
                        })
                        print(f"HIGH ERROR: {dgp_name} N={n_sample} seed={seed} max_deviation={max_dev:.6f}")
                    
                except Exception as e:
                    print(f"Error processing simulation for {dgp_name} N={n_sample}: {e}")
                    continue
            
            max_deviations[dgp_name][n_sample] = max_devs
            print(f"Computed {len(max_devs)} max deviations for {dgp_name} N={n_sample}")
    
    # Print summary of high error seeds
    if high_error_seeds:
        print(f"\n=== SUMMARY: Found {len(high_error_seeds)} seeds with maximum deviation > 20.0 ===")
        for error_info in high_error_seeds:
            print(f"  {error_info['dgp']} N={error_info['sample_size']} seed={error_info['seed']} error={error_info['max_deviation']:.6f}")
        print("=" * 80)
    else:
        print("\n=== No seeds found with maximum deviation > 20.0 ===")
    
    return max_deviations

def create_convergence_plots(max_deviations, save_dir="paper/resources/density_uniform_convergence"):
    """
    Create boxplots showing uniform convergence for each DGP.
    Creates both log-scale and linear-scale versions.
    
    Args:
        max_deviations: Dictionary from compute_max_deviations
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    for dgp_name, dgp_deviations in max_deviations.items():
        if not dgp_deviations:
            continue
        
        # Prepare data for plotting
        sample_sizes = []
        deviations_list = []
        
        for n_sample in sorted(dgp_deviations.keys()):
            if dgp_deviations[n_sample]:  # Only include if we have data
                sample_sizes.extend([str(n_sample)] * len(dgp_deviations[n_sample]))
                deviations_list.extend(dgp_deviations[n_sample])
        
        if not deviations_list:
            print(f"No data to plot for {dgp_name}")
            continue
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'Sample Size': sample_sizes,
            'Max Deviation': deviations_list
        })
        
        # Create log-scale plot
        plt.figure(figsize=(4, 2.5))
        ax = sns.boxplot(data=plot_data, x='Sample Size', y='Max Deviation', flierprops=dict(marker='o', markersize=3, alpha=0.5), boxprops=dict(facecolor='dodgerblue', alpha=0.5))
        plt.title(f'{DGP_NAME_MAPPING[dgp_name]}', fontsize=10, fontweight='bold')
        plt.xlabel('Sample Size', fontsize=10)
        plt.ylabel('Maximum Deviation', fontsize=10)
        plt.yscale('log')  # Log scale to better show convergence
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, fontsize=8)
        
        # Overlay fitted scaling line and median points on top of the boxplot (categorical x)
        try:
            # Compute medians per sample size
            ns_sorted = sorted([int(n) for n in dgp_deviations.keys() if dgp_deviations[n]])
            medians_sorted = [float(np.median(dgp_deviations[n])) for n in ns_sorted]
            # Keep strictly positive medians for log transform
            ns_pos = []
            meds_pos = []
            for n_val, med_val in zip(ns_sorted, medians_sorted):
                if med_val > 0:
                    ns_pos.append(float(n_val))
                    meds_pos.append(float(med_val))
            if len(ns_pos) >= 2:
                x_log = np.log(np.array(ns_pos, dtype=float))
                y_log = np.log(np.array(meds_pos, dtype=float))
                m_fit, b_fit = np.polyfit(x_log, y_log, 1)
                # Predicted medians at observed Ns
                y_pred = np.exp(b_fit) * (np.array(ns_pos) ** m_fit)
                # Map categories to x positions
                cat_labels = [str(n) for n in ns_sorted]
                xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
                pos_map = {label: i for i, label in enumerate(xtick_labels)}
                x_positions = [pos_map[str(int(n))] for n in ns_pos if str(int(n)) in pos_map]
                # Plot observed medians and fitted line across the same positions
                ax.scatter(x_positions, meds_pos, color='black', s=16, zorder=3)
                # To draw a line, sort by position
                order_idx = np.argsort(x_positions)
                x_sorted_pos = np.array(x_positions)[order_idx]
                y_sorted_pred = np.array(y_pred)[order_idx]
                ax.plot(x_sorted_pos, y_sorted_pred, color='crimson', linewidth=1.5)
                # Compute alpha and R^2 (computed on log-log scale)
                y_hat = m_fit * x_log + b_fit
                sse = float(np.sum((y_log - y_hat) ** 2))
                sst = float(np.sum((y_log - np.mean(y_log)) ** 2))
                r2 = 1.0 - sse / sst if sst > 0 else np.nan
                alpha = -m_fit
                # Create legend with alpha and R² values
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=6, label=f'α = {alpha:.4f}'),
                    Line2D([0], [0], color='crimson', linewidth=1.5, label=f'R² = {r2:.4f}')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)
        except Exception as e:
            print(f"Overlay fit failed for {dgp_name}: {e}")
        
        plt.tight_layout()
        
        # Save log-scale plot
        filename_log = f"uniform_convergence_{dgp_name.lower()}_cvxpy_log.png"
        filepath_log = os.path.join(save_dir, filename_log)
        plt.savefig(filepath_log, dpi=300, bbox_inches='tight')
        print(f"Saved log-scale plot: {filepath_log}")
        plt.close()
        
        # Create linear-scale plot
        plt.figure(figsize=(4, 2.5))
        ax = sns.boxplot(data=plot_data, x='Sample Size', y='Max Deviation', flierprops=dict(marker='o', markersize=3, alpha=0.5), boxprops=dict(facecolor='dodgerblue', alpha=0.5))
        plt.title(f'{DGP_NAME_MAPPING[dgp_name]}', fontsize=10, fontweight='bold')
        plt.xlabel('Sample Size', fontsize=10)
        plt.ylabel('Maximum Deviation', fontsize=10)
        
        # Set y-axis limit to 2 times the median of N=25 max deviation
        if 25 in dgp_deviations and dgp_deviations[25]:
            median_n25 = np.median(dgp_deviations[25])
            y_max = 2 * median_n25
            plt.ylim([0.00, y_max])
        else:
            plt.ylim([0.00, 1.00])  # fallback if N=25 data not available
        
        # No log scale for this plot
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=0)
        
        # Remove median annotations for linear plot
        # for i, n_sample in enumerate(sorted(dgp_deviations.keys())):
        #     if dgp_deviations[n_sample]:
        #         median_dev = np.median(dgp_deviations[n_sample])
        #         plt.text(i, median_dev, f'{median_dev:.4f}', 
        #                 ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save linear-scale plot
        filename_linear = f"uniform_convergence_{dgp_name.lower()}_cvxpy_linear.png"
        filepath_linear = os.path.join(save_dir, filename_linear)
        plt.savefig(filepath_linear, dpi=300, bbox_inches='tight')
        print(f"Saved linear-scale plot: {filepath_linear}")
        plt.close()

def create_summary_table(max_deviations):
    """
    Create a summary table of convergence statistics.
    
    Args:
        max_deviations: Dictionary from compute_max_deviations
    
    Returns:
        pd.DataFrame: Summary statistics table
    """
    summary_data = []
    
    for dgp_name, dgp_deviations in max_deviations.items():
        for n_sample, deviations in dgp_deviations.items():
            if deviations:
                summary_data.append({
                    'DGP': dgp_name,
                    'Sample Size': n_sample,
                    'Num Simulations': len(deviations),
                    'Median Max Deviation': np.median(deviations),
                    'Mean Max Deviation': np.mean(deviations),
                    'Std Max Deviation': np.std(deviations),
                    '25th Percentile': np.percentile(deviations, 25),
                    '75th Percentile': np.percentile(deviations, 75)
                })
    
    return pd.DataFrame(summary_data)

def create_uniform_convergence_comp_table(max_deviations):
    """
    Fit log-log scaling laws for each DGP using median max deviations across sample sizes
    and create a single combined figure with all DGPs.

    For each DGP, we:
    - Compute median maximum deviation for each available sample size
    - Fit linear model: log(median deviation) = a + b * log(N)
    - Plot observed medians (markers) and fitted line on one combined log-log plot
    - Collect compact summary statistics (slope b, alpha=-b, R^2) into a DataFrame

    Args:
        max_deviations: Dictionary from compute_max_deviations

    Returns:
        pd.DataFrame: Compact scaling-law summary per DGP
    """
    if not isinstance(max_deviations, dict) or not max_deviations:
        return None

    save_dir = "paper/resources/density_uniform_convergence"
    os.makedirs(save_dir, exist_ok=True)

    summaries = []
    plot_payload = []  # store per-DGP points and fit for combined plot

    # Set plotting style locally to avoid side-effects
    plt.style.use('default')
    sns.set_palette("husl")

    for dgp_name, dgp_deviations in max_deviations.items():
        # Gather (N, median deviation) pairs
        sample_sizes = []
        median_devs = []
        num_sims = []

        for n_sample, deviations in dgp_deviations.items():
            if deviations:
                med = float(np.median(deviations))
                # Require strictly positive median for log transformation
                if med > 0:
                    sample_sizes.append(int(n_sample))
                    median_devs.append(med)
                    num_sims.append(int(len(deviations)))

        if len(sample_sizes) < 2:
            # Not enough points to fit a line
            continue

        # Prepare log-log data
        x = np.log(np.array(sample_sizes, dtype=float))
        y = np.log(np.array(median_devs, dtype=float))

        # Fit linear regression via numpy (y = m x + b)
        try:
            m, b = np.polyfit(x, y, 1)
        except Exception as e:
            print(f"Regression failed for {dgp_name}: {e}")
            continue

        y_hat = m * x + b
        residuals = y - y_hat

        sse = float(np.sum(residuals ** 2))
        sst = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - sse / sst if sst > 0 else np.nan

        n = len(x)
        sxx = float(np.sum((x - np.mean(x)) ** 2))
        se_m = np.nan
        t_stat = np.nan
        p_value = np.nan
        if n > 2 and sxx > 0:
            sigma2 = sse / (n - 2)
            se_m = float(np.sqrt(sigma2 / sxx))
            # Hypothesis test: H0: slope = 0 vs H1: slope ≠ 0
            t_stat = m / se_m
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))

        # 95% CI using t-distribution
        ci_low = np.nan
        ci_high = np.nan
        if not np.isnan(se_m) and n > 2:
            t_crit = stats.t.ppf(0.975, df=n-2)
            ci_low = m - t_crit * se_m
            ci_high = m + t_crit * se_m

        # Predicted deviations at min/max N (on original scale)
        n_min = int(min(sample_sizes))
        n_max = int(max(sample_sizes))
        pred_min = float(np.exp(m * np.log(n_min) + b))
        pred_max = float(np.exp(m * np.log(n_max) + b))

        # Observed medians at min/max N
        med_at_min = float(median_devs[np.argmin(sample_sizes)])
        med_at_max = float(median_devs[np.argmax(sample_sizes)])

        # Save payload for combined plot (use original scale; we'll set log axes)
        n_min = int(min(sample_sizes))
        n_max = int(max(sample_sizes))
        n_line = np.logspace(np.log10(n_min), np.log10(n_max), 200)
        y_line_pred = np.exp(b) * (n_line ** m)
        plot_payload.append({
            'dgp': dgp_name,
            'N_obs': np.array(sample_sizes, dtype=float),
            'med_obs': np.array(median_devs, dtype=float),
            'N_line': n_line,
            'med_pred': y_line_pred,
            'slope': float(m),
            'r2': float(r2),
        })

        # Determine significance at α = 0.05
        is_significant = "Yes" if (not np.isnan(p_value) and p_value < 0.05) else "No"
        
        summaries.append({
            'DGP': dgp_name,
            'Num Points': n,
            'Slope': float(m),
            'Alpha (-Slope)': float(-m),
            'R^2': float(r2),
            'SE_Slope': float(se_m) if not np.isnan(se_m) else np.nan,
            'T_Stat': float(t_stat) if not np.isnan(t_stat) else np.nan,
            'P_Value': float(p_value) if not np.isnan(p_value) else np.nan,
            'Significant': is_significant,
            'CI_Low': float(ci_low) if not np.isnan(ci_low) else np.nan,
            'CI_High': float(ci_high) if not np.isnan(ci_high) else np.nan,
        })

    if not summaries:
        return None

    # Create combined plot with all DGPs
    try:
        plt.figure(figsize=(5, 4))
        colors = sns.color_palette("husl", n_colors=len(plot_payload))
        for idx, payload in enumerate(plot_payload):
            color = colors[idx]
            # Observed medians
            plt.scatter(payload['N_obs'], payload['med_obs'], s=15, color=color, alpha=0.9, label=f"{DGP_NAME_MAPPING[payload['dgp']]}")
            # Fitted line
            plt.plot(payload['N_line'], payload['med_pred'], color=color, linewidth=1.6)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Sample Size (N)', fontsize=10)
        plt.ylabel('Median Max Deviation', fontsize=10)
        plt.title('Uniform Convergence Scaling Law', fontsize=10, fontweight='bold')
        plt.xticks(fontsize=8, rotation=45)
        plt.grid(True, which='both', alpha=0.3)
        # Compact legend
        plt.legend(fontsize=7, framealpha=0.8, loc='upper right')
        plt.tight_layout()
        combined_path = os.path.join(save_dir, "uniform_convergence_scaling_all_cvxpy.png")
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined scaling-law plot: {combined_path}")
    except Exception as e:
        print(f"Failed to save combined scaling-law plot: {e}")
    finally:
        try:
            plt.close()
        except Exception:
            pass

    # Save comprehensive summary table
    summary_df = pd.DataFrame(summaries)
    out_path = os.path.join(save_dir, "uniform_convergence_scaling_law_summary_comprehensive.csv")
    try:
        summary_df.to_csv(out_path, index=False)
        print(f"Saved scaling-law comprehensive summary: {out_path}")
    except Exception as e:
        print(f"Failed to save scaling-law comprehensive summary CSV: {e}")

    # Create 3×6 LaTeX table for paper
    try:
        # Define DGP order for consistent presentation
        dgp_order = [
            "TruncatedNormal",
            "TruncatedGMMSymmetricThree", 
            "TruncatedGMMAsymmetricThree",
            "TruncatedGMMFiveSpikes",
            "StepFunction",
            "Sinusoidal"
        ]
        
        # Build three rows: Alpha, R², Significant
        alpha_row = ["$\\alpha$ (Decay Exponent)"]
        r2_row = ["$R^2$ (Goodness of Fit)"]
        sig_row = ["Significant ($p < 0.05$)"]
        
        for dgp in dgp_order:
            matching_rows = summary_df[summary_df['DGP'] == dgp]
            if len(matching_rows) == 1:
                # Get the first (and only) matching row as a Series
                row_data = matching_rows.iloc[0]
                alpha_val = float(row_data['Alpha (-Slope)'])
                r2_val = float(row_data['R^2'])
                sig_val = str(row_data['Significant'])
                alpha_row.append(f"{alpha_val:.4f}")
                r2_row.append(f"{r2_val:.4f}")
                sig_row.append(sig_val)
            else:
                alpha_row.append("--")
                r2_row.append("--")
                sig_row.append("--")
        
        # Create LaTeX table
        latex_lines = []
        latex_lines.append("% Auto-generated scaling law summary table")
        latex_lines.append("\\begin{table}[H]")
        latex_lines.append("  \\centering")
        latex_lines.append("  \\small")
        latex_lines.append("  \\setlength{\\tabcolsep}{4pt}")
        latex_lines.append("  \\renewcommand{\\arraystretch}{1.2}")
        latex_lines.append("  \\begin{tabular}{l" + "c" * len(dgp_order) + "}")
        latex_lines.append("    \\toprule")
        
        # Column headers (abbreviated DGP names)
        short_names = [
            DGP_NAME_MAPPING["TruncatedNormal"],
            DGP_NAME_MAPPING["TruncatedGMMSymmetricThree"],
            DGP_NAME_MAPPING["TruncatedGMMAsymmetricThree"],
            DGP_NAME_MAPPING["TruncatedGMMFiveSpikes"],
            DGP_NAME_MAPPING["StepFunction"],
            DGP_NAME_MAPPING["Sinusoidal"]
        ]
        header_row = ["Metric"] + short_names
        latex_lines.append("    " + " & ".join(header_row) + " \\\\")
        latex_lines.append("    \\midrule")
        
        # Data rows
        latex_lines.append("    " + " & ".join(alpha_row) + " \\\\")
        latex_lines.append("    " + " & ".join(r2_row) + " \\\\")
        latex_lines.append("    " + " & ".join(sig_row) + " \\\\")
        latex_lines.append("    \\bottomrule")
        latex_lines.append("  \\end{tabular}")
        latex_lines.append("  \\caption{Uniform convergence scaling analysis. $\\alpha$ represents the decay exponent in the power law $\\text{error} \\propto n^{-\\alpha}$. $R^2$ measures goodness of fit on the log-log scale. Significance tests $H_0: \\text{slope} = 0$ vs $H_1: \\text{slope} \\neq 0$ at $\\alpha = 0.05$.}")
        latex_lines.append("  \\label{tab:uniform_convergence_scaling}")
        latex_lines.append("\\end{table}")
        
        latex_path = os.path.join(save_dir, "uniform_convergence_scaling_table.tex")
        with open(latex_path, 'w') as f:
            f.write("\n".join(latex_lines))
        print(f"Saved 3×6 LaTeX table: {latex_path}")
        
    except Exception as e:
        print(f"Failed to create LaTeX table: {e}")

    return summary_df

def main():
    """Main function to run the uniform convergence analysis."""
    print("Loading CVXPY estimator results...")
    
    # Load results
    results = load_cvxpy_results()
    
    # Check if we have any data
    total_sims = sum(len(sims) for dgp_results in results.values() 
                    for sims in dgp_results.values())
    print(f"Total simulations loaded: {total_sims}")
    
    if total_sims == 0:
        print("No valid simulations found. Exiting.")
        return
    
    print("\nComputing maximum deviations...")
    
    # Compute maximum deviations
    max_deviations = compute_max_deviations(results)
    
    print("\nCreating convergence plots...")
    
    # Create plots
    create_convergence_plots(max_deviations)
    
    print("\nGenerating summary table...")
    
    # Create summary table
    summary_df = create_summary_table(max_deviations)
    
    # Save summary table
    summary_path = "experiments/uniform_convergence/uniform_convergence_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary table: {summary_path}")
    
    # Display summary
    print("\nConvergence Summary:")
    print(summary_df.to_string(index=False))
    
    # Scaling-law analysis
    print("\nFitting scaling-law models (log-log) and saving summary...")
    scaling_df = create_uniform_convergence_comp_table(max_deviations)
    if scaling_df is not None:
        print("\nScaling-Law Summary:")
        try:
            print(scaling_df.to_string(index=False))
        except Exception:
            pass
    else:
        print("No scaling-law summary generated (insufficient data).")

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()