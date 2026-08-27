#!/usr/bin/env python3
"""
Asymptotic Efficiency Comparison for HAL-MLE vs HAL-MLE-Target vs Efficient Estimators

This script compares the asymptotic efficiency of three estimators:
1. Asymptotically Efficient Estimator
2. HAL-MLE Estimator  
3. HAL-MLE-Target Estimator

The analysis loads results from:
- Original HAL-MLE results: experiments/uniform_convergence/efficiency_analysis_results.json
- Targeted results: experiments/uniform_convergence/targeted_plots/efficiency_analysis_result_None_*.json

Creates comparison plots and saves to experiments/uniform_convergence/efficiency_targeting_plots/

Note: Only DGPs that appear in both the original and targeting results will be plotted.
Currently, this is typically limited to DGPs that have completed both phases of analysis.
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FuncFormatter

with open('dpg-name-mapping.json', 'r') as f:
    DGP_NAME_MAPPING = json.load(f)

# Define parameter display names
PARAM_DISPLAY_NAMES = {
    'mean': 'Mean',
    'median': 'Median', 
    's_0_5': 'Survival at 0.5',
    'second_moment': 'Second Moment'
}

# Define metrics to plot
METRICS = {
    'mse': 'Mean Squared Error (MSE)',
    'variance': 'Variance', 
    'bias': '|Bias|',
    'se': 'Standard Error',
    'bias_over_se': '|Bias| / Standard Error'
}

def load_original_results(filepath):
    """
    Load original HAL-MLE vs Efficient estimator results.
    
    Args:
        filepath: Path to efficiency_analysis_results.json
        
    Returns:
        dict: Results with structure {dgp_name: {sample_size: {summary: {...}}}}
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def load_targeting_results(base_dir):
    """
    Load targeting results from all parameter-specific JSON files.
    
    Args:
        base_dir: Directory containing targeting result files
        
    Returns:
        dict: Combined targeting results with structure {dgp_name: {param: {sample_sizes: [...], hal_*: [...], efficient_*: [...]}}}
    """
    # Map parameter names to their corresponding file names
    param_file_mapping = {
        'mean': 'mean',
        'median': 'median', 
        'second_moment': 'second_moment',
        's_0_5': 'survival_0_5'  # File uses 'survival_0_5' but we use 's_0_5' internally
    }
    
    targeting_results = {}
    
    for param_key, file_param in param_file_mapping.items():
        filepath = os.path.join(base_dir, f'efficiency_analysis_result_None_{file_param}.json')
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            for dgp_name, dgp_data in data.items():
                if dgp_name not in targeting_results:
                    targeting_results[dgp_name] = {}
                
                # Use the consistent parameter key (s_0_5 instead of survival_0_5)
                targeting_results[dgp_name][param_key] = dgp_data[param_key]
    
    return targeting_results

def extract_metrics_from_original(original_results, dgp_name, parameter):
    """
    Extract metrics for a specific DGP and parameter from original results.
    
    Args:
        original_results: Original results dictionary
        dgp_name: Name of the DGP
        parameter: Parameter name (mean, median, s_0_5, second_moment)
        
    Returns:
        tuple: (sample_sizes, hal_metrics, efficient_metrics) where metrics are dicts with keys: bias, variance, mse, se, bias_over_se
    """
    if dgp_name not in original_results:
        return [], {}, {}
    
    dgp_data = original_results[dgp_name]
    sample_sizes = sorted([int(k) for k in dgp_data.keys()])
    
    hal_metrics = {metric: [] for metric in METRICS.keys()}
    efficient_metrics = {metric: [] for metric in METRICS.keys()}
    
    for n in sample_sizes:
        n_str = str(n)
        if n_str in dgp_data and 'summary' in dgp_data[n_str]:
            summary = dgp_data[n_str]['summary']
            
            if 'hal_mle' in summary and parameter in summary['hal_mle']:
                hal_data = summary['hal_mle'][parameter]
                hal_metrics['bias'].append(hal_data.get('bias', 0))
                hal_metrics['variance'].append(hal_data.get('variance', 0))
                hal_metrics['mse'].append(hal_data.get('mse', 0))
                hal_metrics['se'].append(hal_data.get('s.e.', 0))  # Note: 's.e.' in original data
                hal_metrics['bias_over_se'].append(hal_data.get('bias_over_se', 0))
            else:
                # Fill with zeros if data missing
                for metric in hal_metrics:
                    hal_metrics[metric].append(0)
            
            if 'efficient' in summary and parameter in summary['efficient']:
                eff_data = summary['efficient'][parameter]
                efficient_metrics['bias'].append(eff_data.get('bias', 0))
                efficient_metrics['variance'].append(eff_data.get('variance', 0))
                efficient_metrics['mse'].append(eff_data.get('mse', 0))
                efficient_metrics['se'].append(eff_data.get('s.e.', 0))  # Note: 's.e.' in original data
                efficient_metrics['bias_over_se'].append(eff_data.get('bias_over_se', 0))
            else:
                # Fill with zeros if data missing
                for metric in efficient_metrics:
                    efficient_metrics[metric].append(0)
    
    return sample_sizes, hal_metrics, efficient_metrics

def extract_metrics_from_targeting(targeting_results, dgp_name, parameter):
    """
    Extract metrics for a specific DGP and parameter from targeting results.
    
    Args:
        targeting_results: Targeting results dictionary
        dgp_name: Name of the DGP
        parameter: Parameter name (mean, median, s_0_5, second_moment)
        
    Returns:
        tuple: (sample_sizes, hal_target_metrics, efficient_metrics) where metrics are dicts
    """
    if dgp_name not in targeting_results or parameter not in targeting_results[dgp_name]:
        return [], {}, {}
    
    param_data = targeting_results[dgp_name][parameter]
    
    sample_sizes = param_data.get('sample_sizes', [])
    
    hal_target_metrics = {
        'bias': param_data.get('hal_bias', []),
        'variance': param_data.get('hal_variance', []),
        'mse': param_data.get('hal_mse', []),
        'se': param_data.get('hal_se', []),
        'bias_over_se': param_data.get('hal_bias_over_se', [])
    }
    
    efficient_metrics = {
        'bias': param_data.get('efficient_bias', []),
        'variance': param_data.get('efficient_variance', []),
        'mse': param_data.get('efficient_mse', []),
        'se': param_data.get('efficient_se', []),
        'bias_over_se': param_data.get('efficient_bias_over_se', [])
    }
    
    return sample_sizes, hal_target_metrics, efficient_metrics

def create_comparison_plots(original_results, targeting_results, save_dir):
    """
    Create and save 3x4 subplot grids for each DGP comparing three estimators across 3 metrics and 4 estimands.
    Each DGP gets one figure with 12 subplots (3 rows x 4 columns).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all available DGPs (intersection of both datasets)
    original_dgps = set(original_results.keys())
    targeting_dgps = set(targeting_results.keys())
    available_dgps = original_dgps.intersection(targeting_dgps)
    
    if not available_dgps:
        print("Warning: No common DGPs found between original and targeting results!")
        print(f"Original DGPs: {list(original_dgps)}")
        print(f"Targeting DGPs: {list(targeting_dgps)}")
        return
    
    # All parameters to plot as columns
    parameters = ['mean', 'median', 's_0_5', 'second_moment']
    
    # Select 3 most important metrics as rows (same as targeting script)
    selected_metrics = ['mse', 'variance', 'bias_over_se']
    metric_display_names = {
        'mse': 'Mean Squared Error',
        'variance': 'Variance',
        'bias_over_se': '|Bias| / Standard Error'
    }
    
    # Create one figure per DGP
    for dgp_name in available_dgps:
        # Create figure with 3x4 subplots
        fig, axes = plt.subplots(3, 4, figsize=(10, 6))
        fig.suptitle(f'DGP: {DGP_NAME_MAPPING[dgp_name]}', fontsize=13, y=0.95)
        
        # Plot each metric (row) x parameter (column) combination
        for row, metric_key in enumerate(selected_metrics):
            for col, param in enumerate(parameters):
                ax = axes[row, col]
                
                # Extract data from original results
                orig_sample_sizes, orig_hal_metrics, orig_eff_metrics = extract_metrics_from_original(
                    original_results, dgp_name, param)
                
                # Extract data from targeting results
                target_sample_sizes, target_hal_metrics, target_eff_metrics = extract_metrics_from_targeting(
                    targeting_results, dgp_name, param)
                
                # Check if we have data
                if not orig_sample_sizes or not target_sample_sizes:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    if row == 0:
                        ax.set_title(f'{PARAM_DISPLAY_NAMES[param]}', fontsize=10, pad=10)
                    continue
                
                # Use original sample sizes as reference
                sample_sizes = orig_sample_sizes
                
                # Get metric values
                efficient_values = orig_eff_metrics[metric_key]
                hal_mle_values = orig_hal_metrics[metric_key]
                
                # For targeting values, align sample sizes
                hal_target_values = []
                for n in sample_sizes:
                    try:
                        idx = target_sample_sizes.index(n)
                        hal_target_values.append(target_hal_metrics[metric_key][idx])
                    except (ValueError, IndexError):
                        hal_target_values.append(np.nan)  # Missing data
                
                # Skip if all values are zero or nan
                all_values = efficient_values + hal_mle_values + [v for v in hal_target_values if not np.isnan(v)]
                if all(v == 0 for v in all_values):
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    if row == 0:
                        ax.set_title(f'{PARAM_DISPLAY_NAMES[param]}', fontsize=10, pad=10)
                    continue
                
                # Plot the three estimators
                ax.plot(sample_sizes, efficient_values, 's-', 
                       label='Asymptotically Efficient Estimator', linewidth=2.5, markersize=4, color='green')
                ax.plot(sample_sizes, hal_mle_values, 'o-', 
                       label='HAL-MLE', linewidth=2, markersize=4, color='blue')
                ax.plot(sample_sizes, hal_target_values, '^--', 
                       label='HAL-TMLE', linewidth=2, markersize=4, color='red')
                
                # Set scales based on metric type
                if metric_key in ['mse', 'variance']:
                    ax.set_xscale('log')
                    # ax.set_yscale('log')
                    ax.grid(True, ls="--", alpha=0.3)
                else:  # bias_over_se
                    ax.set_xscale('log')
                    # ax.set_yscale('log')
                    ax.grid(True, ls="--", alpha=0.3)
                
                # Set title for top row (parameter names)
                if row == 0:
                    ax.set_title(f'{PARAM_DISPLAY_NAMES[param]}', fontsize=10, pad=10)
                
                # Set ylabel for leftmost column (metric names)
                if col == 0:
                    ax.set_ylabel(metric_display_names[metric_key], fontsize=10)
                
                    # Format y-axis tick labels to 3 decimal places
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))

                # Set xlabel only for bottom row
                if row == 2:
                    ax.set_xlabel('Sample Size', fontsize=10)
                
                # Remove x-axis labels for non-bottom rows
                if row < 2:
                    ax.set_xticklabels([])
                
                # Remove y-axis labels for non-leftmost columns
                if col > 0:
                    ax.set_yticklabels([])
        
        # Add shared legend below the subplots
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, 
                  bbox_to_anchor=(0.5, -0.05), fontsize=10)
        
        # Adjust layout to make room for legend
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.10, top=0.875, hspace=0.20, wspace=0.10)
        
        # Save the figure
        dgp_save_dir = os.path.join(save_dir, dgp_name)
        os.makedirs(dgp_save_dir, exist_ok=True)
        plot_filename = f"efficiency_comparison.png"
        plot_path = os.path.join(dgp_save_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison plot: {plot_path}")

    print(f"\nAll comparison plots saved to {os.path.abspath(save_dir)}")

def main():
    """Main function to run the efficiency comparison."""
    parser = argparse.ArgumentParser(description="Compare Asymptotic Efficiency: HAL-MLE vs HAL-MLE-Target vs Efficient")
    parser.add_argument("--original_results", type=str, 
                       default="experiments/uniform_convergence/efficiency_analysis_results.json",
                       help="Path to original efficiency analysis results")
    parser.add_argument("--targeting_dir", type=str,
                       default="experiments/uniform_convergence/targeted_plots",
                       help="Directory containing targeting analysis results")
    parser.add_argument("--save_dir", type=str,
                       default="paper/resources/density_asymptotic_efficiency",
                       help="Directory to save comparison plots")
    
    args = parser.parse_args()
    
    print("Loading Asymptotic Efficiency Comparison...")
    print(f"Original results: {args.original_results}")
    print(f"Targeting directory: {args.targeting_dir}")
    print(f"Save directory: {args.save_dir}")
    print("-" * 50)
    
    # Load results
    print("Loading original results...")
    original_results = load_original_results(args.original_results)
    
    print("Loading targeting results...")  
    targeting_results = load_targeting_results(args.targeting_dir)
    
    print(f"Original DGPs: {list(original_results.keys())}")
    print(f"Targeting DGPs: {list(targeting_results.keys())}")
    
    # Find common DGPs
    original_dgps = set(original_results.keys())
    targeting_dgps = set(targeting_results.keys())
    common_dgps = original_dgps.intersection(targeting_dgps)
    print(f"Common DGPs for comparison: {list(common_dgps)}")
    
    if not common_dgps:
        print("ERROR: No common DGPs found between original and targeting results!")
        return
    
    # Create comparison plots
    print("Creating comparison plots...")
    create_comparison_plots(original_results, targeting_results, args.save_dir)
    
    print(f"\nComparison plots saved to: {os.path.abspath(args.save_dir)}")
    print("Asymptotic Efficiency Comparison complete!")

if __name__ == "__main__":
    main()