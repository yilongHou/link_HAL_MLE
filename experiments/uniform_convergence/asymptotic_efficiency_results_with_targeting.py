#!/usr/bin/env python3
"""
Asymptotic Efficiency Analysis for Targeted HAL-MLE Results

This script analyzes the asymptotic efficiency of targeted HAL-MLE estimators by
comparing their estimates for population mean, median, second moment, and 
survival probability against asymptotically efficient estimators.

This is adapted from asymptotic_efficiency_results.py to work with targeted results.
"""
import os
import sys
import json
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy import integrate
import matplotlib.pyplot as plt

# Compatibility for trapz (deprecated in newer scipy versions)
try:
    from scipy.integrate import trapz
except ImportError:
    from scipy.integrate import trapezoid as trapz

warnings.filterwarnings('ignore')

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

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
        "mean": 0.43824,
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

def load_targeted_results(results_dir="experiments/uniform_convergence/targeted_results", 
                         dgp_names=None, sample_sizes=None, targeter_filter=None):
    """
    Load targeted estimator results from JSON files.
    
    Args:
        results_dir: Directory containing targeted results
        dgp_names: List of DGP names to include (default: all available)
        sample_sizes: List of sample sizes to include (default: specified range)
        targeter_filter: Specific targeter to load (e.g., 'mean', 'median')
    
    Returns:
        dict: Dictionary with structure {dgp_name: {n_sample: [results_list]}}
    """
    if sample_sizes is None:
        sample_sizes = [25, 50, 100, 200, 400, 800, 1600, 3200]
    
    if dgp_names is None:
        dgp_names = ["TruncatedNormal", "TruncatedGMMSymmetricThree", 
                    "TruncatedGMMAsymmetricThree", "TruncatedGMMFiveSpikes", 
                    "StepFunction", "Sinusoidal"]
    
    results = {}
    
    for dgp_name in tqdm(dgp_names, desc="Loading DGPs"):
        results[dgp_name] = {}
        
        for n_sample in sample_sizes:
            results[dgp_name][n_sample] = []
            pattern = f"{dgp_name}_CVXPYEstimator_N{n_sample}"
            dir_path = os.path.join(results_dir, pattern)
            
            if not os.path.exists(dir_path):
                continue
            
            # Filter files by targeter if specified
            if targeter_filter:
                json_pattern = f"*__target={targeter_filter}.json"
            else:
                json_pattern = "*__target=*.json"
                
            json_files = glob.glob(os.path.join(dir_path, json_pattern))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    # Verify targeting was successful and results are clean
                    if ('HAL_results' in data and 
                        'data_points' in data['HAL_results'] and 
                        'estimated_density' in data['HAL_results'] and
                        'grid_points' in data['HAL_results'] and
                        'targeting_info' in data and 
                        data['targeting_info'].get('success', False) and
                        not data['HAL_results'].get('targeting_failed', False)):
                        
                        # Additional validation for targeted results
                        hal_results = data['HAL_results']
                        targeting_info = data['targeting_info']
                        
                        # Check if this matches the requested targeter
                        if targeter_filter and targeting_info.get('targeter') != targeter_filter:
                            continue
                            
                        # Verify essential fields are present
                        if (isinstance(hal_results['estimated_density'], list) and 
                            isinstance(hal_results['data_points'], list) and
                            isinstance(hal_results['grid_points'], list) and
                            len(hal_results['estimated_density']) > 0 and
                            len(hal_results['data_points']) > 0 and
                            len(hal_results['grid_points']) > 0):
                            results[dgp_name][n_sample].append(data)
                except (json.JSONDecodeError, KeyError, IOError):
                    continue  # Skip corrupted or incomplete files
    
    # Print summary of loaded data
    print(f"\nLoaded targeted data summary (targeter: {targeter_filter or 'all'}):")
    for dgp_name in results.keys():
        dgp_summary = {n: len(results[dgp_name][n]) for n in sample_sizes 
                      if n in results[dgp_name] and results[dgp_name][n]}
        if dgp_summary:
            print(f"  {dgp_name}: {dgp_summary}")
    
    return results

def process_single_result(result_data, eval_points):
    """
    Process a single targeted experiment to compute HAL-MLE and efficient estimates.
    
    Args:
        result_data: Single experiment result from JSON
        eval_points: Fine grid for integration (e.g., 0 to 1 with 1001 points)
    
    Returns:
        dict: Contains HAL-MLE and efficient estimates, or error info
    """
    try:
        # Extract the raw data points for efficient estimates
        data_points = np.array(result_data['HAL_results']['data_points'])
        
        # Compute efficient estimates directly from sample
        efficient_mean = np.mean(data_points)
        efficient_median = np.median(data_points)  
        efficient_second_moment = np.mean(data_points**2)
        efficient_s_0_5 = np.mean(data_points > 0.5)  # Fixed: use > instead of >= to match original script
        
        # Get the targeted density from HAL results
        # The targeting step puts the targeted density in 'estimated_density'
        # and the corresponding grid points in 'grid_points'
        hal_density = np.array(result_data['HAL_results']['estimated_density'])
        hal_grid = np.array(result_data['HAL_results']['grid_points'])
        
        # Validate that we have matching lengths
        if len(hal_density) != len(hal_grid):
            raise ValueError(f"Density length ({len(hal_density)}) != grid length ({len(hal_grid)})")
        
        # Ensure density is normalized and non-negative
        hal_density = np.maximum(hal_density, 1e-12)
        
        # Interpolate density to evaluation grid
        interp_func = interp1d(hal_grid, hal_density, 
                             kind='linear', bounds_error=False, fill_value=0)
        density_interp = interp_func(eval_points)
        density_interp = np.maximum(density_interp, 1e-12)
        
        # Normalize interpolated density
        dx = eval_points[1] - eval_points[0]
        norm_constant = trapz(density_interp, dx=dx)
        if norm_constant > 1e-12:
            density_interp /= norm_constant
        
        # Compute HAL-MLE estimates from density
        hal_mean = trapz(eval_points * density_interp, dx=dx)
        
        # For median: find where CDF = 0.5
        cdf = np.cumsum(density_interp) * dx
        cdf = np.clip(cdf, 0, 1)
        median_idx = np.searchsorted(cdf, 0.5)
        if median_idx >= len(eval_points):
            median_idx = len(eval_points) - 1
        hal_median = eval_points[median_idx]
        
        # For second moment
        hal_second_moment = trapz(eval_points**2 * density_interp, dx=dx)
        
        # For survival at 0.5: integrate from 0.5 to 1
        survival_mask = eval_points >= 0.5
        if np.any(survival_mask):
            hal_s_0_5 = trapz(density_interp[survival_mask], dx=dx)
        else:
            hal_s_0_5 = 0.0
        
        return {
            'hal_mean': hal_mean,
            'hal_median': hal_median,
            'hal_second_moment': hal_second_moment,
            'hal_s_0_5': hal_s_0_5,
            'efficient_mean': efficient_mean,
            'efficient_median': efficient_median,
            'efficient_second_moment': efficient_second_moment,
            'efficient_s_0_5': efficient_s_0_5,
            'n_sample': len(data_points),
            'targeter': result_data['targeting_info'].get('targeter', 'unknown'),
            'success': True
        }
        
    except Exception as e:
        targeter_name = result_data.get('targeting_info', {}).get('targeter', 'unknown')
        print(f"Error processing result (targeter: {targeter_name}): {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'targeter': targeter_name
        }

def analyze_efficiency(results_dict, eval_points):
    """
    Analyze efficiency by computing Bias, Variance, MSE, and Bias/SE for HAL-MLE estimates.
    
    Args:
        results_dict: Dictionary of experimental results
        eval_points: Evaluation points for density integration
    
    Returns:
        dict: Analysis results with performance metrics for each estimator
    """
    analysis_results = {}
    parameters = ['mean', 'median', 's_0_5', 'second_moment']
    
    for dgp_name, dgp_results in tqdm(results_dict.items(), desc="Analyzing DGPs"):
        print(f"\nAnalyzing {dgp_name}...")
        
        dgp_analysis = {}
        sample_sizes = sorted([n for n in dgp_results.keys() if dgp_results[n]])
        
        for param in parameters:
            true_value = TRUE_POPULATION_STATS[dgp_name][param]
            
            # Initialize storage for this parameter
            param_results = {
                'sample_sizes': [],
                'hal_bias': [], 'hal_variance': [], 'hal_mse': [], 'hal_bias_over_se': [], 'hal_se': [],
                'efficient_bias': [], 'efficient_variance': [], 'efficient_mse': [], 'efficient_bias_over_se': [], 'efficient_se': []
            }
            
            for n_sample in sample_sizes:
                # Process all results for this sample size
                processed_results = []
                for result_data in dgp_results[n_sample]:
                    processed = process_single_result(result_data, eval_points)
                    if processed['success']:
                        processed_results.append(processed)
                
                if len(processed_results) < 5:  # Need minimum number of results
                    continue
                
                # Extract estimates for this parameter
                hal_estimates = [r[f'hal_{param}'] for r in processed_results]
                efficient_estimates = [r[f'efficient_{param}'] for r in processed_results]
                
                # Compute statistics
                def compute_stats(estimates, true_val):
                    estimates = np.array(estimates)
                    bias = np.abs(np.mean(estimates) - true_val)
                    variance = np.var(estimates, ddof=1)
                    mse = np.mean((estimates - true_val)**2)
                    se = np.sqrt(variance)
                    bias_over_se = bias / (se + 1e-9)  # Add epsilon for numerical stability
                    return bias, variance, mse, bias_over_se, se
                
                hal_bias, hal_var, hal_mse, hal_bias_se, hal_se = compute_stats(hal_estimates, true_value)
                eff_bias, eff_var, eff_mse, eff_bias_se, eff_se = compute_stats(efficient_estimates, true_value)
                
                # Store results
                param_results['sample_sizes'].append(n_sample)
                param_results['hal_bias'].append(hal_bias)
                param_results['hal_variance'].append(hal_var)
                param_results['hal_mse'].append(hal_mse)
                param_results['hal_bias_over_se'].append(hal_bias_se)
                param_results['hal_se'].append(hal_se)
                param_results['efficient_bias'].append(eff_bias)
                param_results['efficient_variance'].append(eff_var)
                param_results['efficient_mse'].append(eff_mse)
                param_results['efficient_bias_over_se'].append(eff_bias_se)
                param_results['efficient_se'].append(eff_se)
            
            dgp_analysis[param] = param_results
            
        analysis_results[dgp_name] = dgp_analysis
    
    return analysis_results

def create_summary_table(analysis_results, targeter_name):
    """
    Create and print a summary table of the efficiency analysis.
    """
    parameters = ['mean', 'median', 's_0_5', 'second_moment']
    
    print(f"\n{'='*80}")
    print(f"ASYMPTOTIC EFFICIENCY ANALYSIS SUMMARY - TARGETER: {targeter_name.upper()}")
    print(f"{'='*80}")
    
    for dgp_name, dgp_summary in analysis_results.items():
        print(f"\n{dgp_name}:")
        print("-" * 60)
        
        for param in parameters:
            if param in dgp_summary and dgp_summary[param]['sample_sizes']:
                param_data = dgp_summary[param]
                n_largest = param_data['sample_sizes'][-1]  # Largest sample size
                idx = -1  # Last entry
                
                hal_mse = param_data['hal_mse'][idx]
                eff_mse = param_data['efficient_mse'][idx]
                efficiency_ratio = hal_mse / eff_mse if eff_mse > 0 else np.inf
                
                print(f"  {param:15} (N={n_largest}): "
                      f"HAL-MSE={hal_mse:.6f}, Eff-MSE={eff_mse:.6f}, "
                      f"Ratio={efficiency_ratio:.3f}")

def create_efficiency_plots(analysis_results, targeter_name, save_dir):
    """
    Create and save 3x4 subplot grids for each DGP showing 3 metrics across 4 estimands.
    Each DGP gets one figure with 12 subplots (3 rows x 4 columns).
    """
    # All parameters to plot as columns
    parameters = ['mean', 'median', 's_0_5', 'second_moment']
    param_display_names = {
        'mean': 'Mean',
        'median': 'Median', 
        's_0_5': 'Survival at 0.5',
        'second_moment': 'Second Moment'
    }
    
    # Select 3 most important metrics as rows
    selected_metrics = ['mse', 'variance', 'bias_over_se']
    metric_display_names = {
        'mse': 'Mean Squared Error',
        'variance': 'Variance',
        'bias_over_se': '|Bias| / Standard Error'
    }
    
    # Create one figure per DGP
    for dgp_name, dgp_summary in analysis_results.items():
        # Check if we have data for all parameters
        available_params = [param for param in parameters if param in dgp_summary and dgp_summary[param]['sample_sizes']]
        if not available_params:
            continue
            
        # Create figure with 3x4 subplots
        fig, axes = plt.subplots(3, 4, figsize=(10, 12))
        fig.suptitle(f'Efficiency Analysis: {dgp_name}', fontsize=16, y=0.95)
        
        # Plot each metric (row) x parameter (column) combination
        for row, metric_key in enumerate(selected_metrics):
            for col, param in enumerate(parameters):
                ax = axes[row, col]
                
                if param not in dgp_summary or not dgp_summary[param]['sample_sizes']:
                    # No data for this parameter
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{param_display_names[param]}', fontsize=10)
                    continue
                
                param_data = dgp_summary[param]
                sample_sizes = param_data['sample_sizes']
                
                hal_values = param_data[f'hal_{metric_key}']
                eff_values = param_data[f'efficient_{metric_key}']
                
                # Plot the data
                ax.plot(sample_sizes, hal_values, 'o-', label=f'Targeted HAL-MLE', 
                       linewidth=2, markersize=4, color='C0')
                ax.plot(sample_sizes, eff_values, 's--', label='Efficient Estimator', 
                       linewidth=2, markersize=4, color='C1')
                
                # Set scales based on metric type
                if metric_key in ['mse', 'variance']:
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    ax.grid(True, which="both", ls="--", alpha=0.3)
                else:  # bias_over_se
                    ax.set_xscale('log')
                    ax.grid(True, ls="--", alpha=0.3)
                
                # Set title for top row (parameter names)
                if row == 0:
                    ax.set_title(f'{param_display_names[param]}', fontsize=12, pad=10)
                
                # Set ylabel for leftmost column (metric names)
                if col == 0:
                    ax.set_ylabel(metric_display_names[metric_key], fontsize=10)
                
                # Set xlabel only for bottom row
                if row == 2:
                    ax.set_xlabel('Sample Size (N)', fontsize=10)
                
                # Remove x-axis labels for non-bottom rows
                if row < 2:
                    ax.set_xticklabels([])
                
                # Remove y-axis labels for non-leftmost columns
                if col > 0:
                    ax.set_yticklabels([])
        
        # Add shared legend below the subplots
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, 
                  bbox_to_anchor=(0.5, 0.02), fontsize=11)
        
        # Adjust layout to make room for legend
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08, top=0.92, hspace=0.25, wspace=0.15)
        
        # Save the figure
        dgp_save_dir = os.path.join(save_dir, dgp_name)
        os.makedirs(dgp_save_dir, exist_ok=True)
        plot_filename = f"efficiency_analysis_{targeter_name}.png"
        plot_path = os.path.join(dgp_save_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved efficiency plot: {plot_path}")

    print(f"\nAll plots saved to {os.path.abspath(save_dir)}")

def main():
    """Main function to run the targeted asymptotic efficiency analysis."""
    parser = argparse.ArgumentParser(description="Asymptotic Efficiency Analysis for Targeted HAL-MLE")
    parser.add_argument("--dgp", type=str, help="Run for a specific DGP")
    parser.add_argument("--targeter", type=str, required=False, 
                       choices=['mean', 'second_moment', 'survival_0_5', 'median'],
                       help="Which targeter to analyze (optional - if not specified, uses all available data)")
    parser.add_argument("--results_dir", type=str,
                       default="experiments/uniform_convergence/targeted_results",
                       help="Directory containing targeted results")
    parser.add_argument("--plot_dir", type=str,
                       default="paper/resources/density_asymptotic_efficiency",
                       help="Directory to save plots")
    
    args = parser.parse_args()

    print(f"Starting Targeted Asymptotic Efficiency Analysis...")
    print(f"Targeter: {args.targeter or 'all available'}")
    print(f"Results directory: {args.results_dir}")
    
    dgp_names = [args.dgp] if args.dgp else None
    
    # Load targeted results
    eval_points = np.linspace(0, 1, 1001)

    targeter_suffix = args.targeter or "all_targeters"
    dgp_suffix = args.dgp or "all_dgps"

    try:
        with open(os.path.join(args.plot_dir, f"efficiency_analysis_result_{dgp_suffix}_{targeter_suffix}.json"), 'r') as f:
            analysis_results = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {os.path.join(args.plot_dir, f'efficiency_analysis_result_{dgp_suffix}_{targeter_suffix}.json')}\nLoading targeted results...")
        
    
        results_dict = load_targeted_results(
            results_dir=args.results_dir,
            dgp_names=dgp_names,
            targeter_filter=args.targeter  # None when not specified, loads all
        )
        
        if not any(any(results_dict[dgp].values()) for dgp in results_dict):
            print("No targeted results found!")
            return
        
        # Analyze efficiency
        print("\nAnalyzing efficiency...")
        analysis_results = analyze_efficiency(results_dict, eval_points)

        # Save analysis results
        filename = f"efficiency_analysis_result_{dgp_suffix}_{targeter_suffix}.json"
        with open(os.path.join(args.plot_dir, filename), 'w') as f:
            json.dump(analysis_results, f, indent=4)
    
    # Create summary table
    create_summary_table(analysis_results, args.targeter or "all_targeters")
    
    # Create plots
    print(f"\nCreating plots...")
    os.makedirs(args.plot_dir, exist_ok=True)
    create_efficiency_plots(analysis_results, args.targeter or "all_targeters", args.plot_dir)
    
    print(f"\nAnalysis completed!")
    print(f"Plots saved to: {os.path.abspath(args.plot_dir)}")

if __name__ == "__main__":
    main()