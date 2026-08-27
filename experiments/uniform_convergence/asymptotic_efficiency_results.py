#!/usr/bin/env python3
"""
Asymptotic Efficiency Analysis for HAL-MLE

This script analyzes the asymptotic efficiency of the HAL-MLE estimator by
comparing its estimates for population mean, median, second moment, and 
survival probability against asymptotically efficient estimators.

The analysis computes and compares Bias, Variance, MSE, and Bias/SE for each 
estimator across different sample sizes and Data Generating Processes (DGPs).
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

# Import BaseEstimator for direct density calculation
from methods.base_estimator import BaseEstimator

# Define DGP configurations (copied from asymptotic_normality_results.py)
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
        dgp_names = list(DGP_CONFIGS.keys())
    
    results = {}
    
    for dgp_name in tqdm(dgp_names, desc="Loading DGPs"):
        results[dgp_name] = {}
        
        for n_sample in sample_sizes:
            results[dgp_name][n_sample] = []
            pattern = f"{dgp_name}_CVXPYEstimator_N{n_sample}"
            dir_path = os.path.join(results_dir, pattern)
            
            if not os.path.exists(dir_path):
                continue
            
            json_files = glob.glob(os.path.join(dir_path, "*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    # We need both data points and estimated density
                    if ('HAL_results' in data and 
                        'data_points' in data['HAL_results'] and 
                        'estimated_density' in data['HAL_results']):
                        results[dgp_name][n_sample].append(data)
                except (json.JSONDecodeError, KeyError, IOError):
                    continue  # Skip corrupted or incomplete files
    
    # Print summary of loaded data
    print("\nLoaded data summary:")
    for dgp_name in results.keys():
        dgp_summary = {n: len(results[dgp_name][n]) for n in sample_sizes 
                      if n in results[dgp_name] and results[dgp_name][n]}
        if dgp_summary:
            print(f"  {dgp_name}: {dgp_summary}")
    
    return results

def process_single_result(result_data, eval_points, recompute_hal_density=False):
    """
    Process a single experiment to compute HAL-MLE and efficient estimates.
    
    Args:
        result_data: Single experiment result from JSON
        eval_points: Fine grid for integration (e.g., 0 to 1 with 1001 points)
    
    Returns:
        dict: Contains HAL-MLE and efficient estimates, or error info
    """
    try:
        # 1. Efficient Estimators (from raw data)
        data_points = np.array(result_data['HAL_results']['data_points'])
        efficient_estimates = {
            'mean': np.mean(data_points),
            'median': np.median(data_points),
            's_0_5': np.mean(data_points > 0.5),
            'second_moment': np.mean(data_points**2)
        }

        if not recompute_hal_density:
            # Use precomputed density if available
            density = np.array(result_data['HAL_results']['estimated_density'])
            grid = np.array(result_data['HAL_results']['grid_points'])

            # Interpolate to a finer grid for accurate integration
            f_hat = interp1d(grid, density, kind='linear', bounds_error=False, fill_value=0)
            density_fine = f_hat(eval_points)

            # Ensure density is non-negative and normalized
            density_fine = np.maximum(density_fine, 0)
            norm_const = trapz(density_fine, eval_points)
            if norm_const > 0:
                density_fine = density_fine / norm_const
        else:
            # 2. HAL-MLE Estimators (from estimated density using direct calculation)
            hal_results = result_data['HAL_results']
            hyperparams = result_data.get('hyperparams', {})
            
            # Use BaseEstimator's direct density calculation method
            density_fine = BaseEstimator.calculate_density_at_points(
                points=eval_points,
                theta_hat=np.array(hal_results['theta_hat']),
                basis_grid_points=np.array(hal_results['data_points']),  # This is _grid_points_hal
                basis_order=hyperparams.get('basis_order', 0),  # Get actual basis_order from hyperparams
                n_norm_grid_points=len(eval_points)  # Use same resolution as eval_points
            )

        # Calculate HAL-MLE Mean
        hal_mean = trapz(eval_points * density_fine, eval_points)
        
        # Calculate HAL-MLE Second Moment
        hal_second_moment = trapz(eval_points**2 * density_fine, eval_points)
        
        # Calculate HAL-MLE CDF and Median
        cdf_fine = np.cumsum(density_fine) * (eval_points[1] - eval_points[0])
        # Normalize CDF to ensure it goes to 1
        if cdf_fine[-1] > 0:
            cdf_fine = cdf_fine / cdf_fine[-1]
        
        # Find median by interpolation
        if np.any(cdf_fine >= 0.5):
            # Find first index where CDF >= 0.5
            idx = np.where(cdf_fine >= 0.5)[0][0]
            if idx > 0:
                # Linear interpolation
                x0, x1 = eval_points[idx-1], eval_points[idx]
                y0, y1 = cdf_fine[idx-1], cdf_fine[idx]
                hal_median = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
            else:
                hal_median = eval_points[idx]
        else:
            hal_median = 0.5  # fallback

        # Calculate HAL-MLE Survival at 0.5
        mask = eval_points >= 0.5
        hal_s_0_5 = trapz(density_fine[mask], eval_points[mask])

        hal_mle_estimates = {
            'mean': hal_mean,
            'median': hal_median,
            's_0_5': hal_s_0_5,
            'second_moment': hal_second_moment
        }
        
        return {'hal_mle': hal_mle_estimates, 'efficient': efficient_estimates, 'success': True}
        
    except Exception as e:
        # Add more context to the error for easier debugging
        error_info = f"Error in process_single_result: {str(e)}"
        return {'success': False, 'error': error_info}

def analyze_efficiency(results_dict, eval_points):
    """
    Analyze efficiency by computing Bias, Variance, MSE, and Bias/SE for all estimators.
    
    Args:
        results_dict: Dictionary of experimental results
        eval_points: Evaluation points for density integration
    
    Returns:
        dict: Analysis results with performance metrics for each estimator
    """
    analysis_results = {}
    parameters = ['mean', 'median', 's_0_5', 'second_moment']
    
    for dgp_name, dgp_results in tqdm(results_dict.items(), desc="Analyzing DGPs"):
        true_stats = TRUE_POPULATION_STATS[dgp_name]
        analysis_results[dgp_name] = {}
        
        for n_sample, experiments in dgp_results.items():
            if not experiments:
                continue

            # Store estimates from all experiments
            estimates = {
                'hal_mle': {p: [] for p in parameters},
                'efficient': {p: [] for p in parameters}
            }
            
            for exp in experiments:
                processed = process_single_result(exp, eval_points)
                if processed['success']:
                    for p in parameters:
                        estimates['hal_mle'][p].append(processed['hal_mle'][p])
                        estimates['efficient'][p].append(processed['efficient'][p])
            
            if not estimates['hal_mle']['mean']:  # Check if any results were processed
                continue

            # Compute Bias, Variance, MSE, and Bias/SE
            summary = {est_type: {} for est_type in ['hal_mle', 'efficient']}
            for est_type in summary:
                for p in parameters:
                    est_values = np.array(estimates[est_type][p])
                    true_val = true_stats[p]
                    
                    bias = np.abs(np.mean(est_values) - true_val)
                    variance = np.var(est_values, ddof=1)
                    mse = np.mean((est_values - true_val)**2)
                    se = np.sqrt(variance)
                    bias_over_se = bias / (se + 1e-9)  # Add epsilon for numerical stability
                    
                    summary[est_type][p] = {
                        'bias': bias,
                        's.e.': se,
                        'variance': variance,
                        'mse': mse,
                        'bias_over_se': bias_over_se
                    }
            
            analysis_results[dgp_name][n_sample] = {
                'summary': summary,
                'n_experiments': len(estimates['hal_mle']['mean'])
            }
    
    return analysis_results

def create_summary_table(analysis_results):
    """
    Create and print a summary table of the efficiency analysis.
    """
    parameters = ['mean', 'median', 's_0_5', 'second_moment']
    
    for dgp_name, dgp_summary in analysis_results.items():
        print("\n" + "="*120)
        print(f"DGP: {dgp_name}")
        print("="*120)
        
        # Header
        header = (f"{'N':<6} {'Param':<14} | {'Estimator':<12} | {'Bias':<15} | "
                  f"{'Variance':<15} | {'MSE':<15} | {'Bias/SE':<15}")
        print(header)
        print("-" * len(header))
        
        for n_sample in sorted(dgp_summary.keys()):
            results = dgp_summary[n_sample]['summary']
            n_exp = dgp_summary[n_sample]['n_experiments']
            
            for i, p in enumerate(parameters):
                n_str = f"{n_sample}" if i == 0 else ""
                
                # HAL-MLE Row
                hal_res = results['hal_mle'][p]
                print(f"{n_str:<6} {p:<14} | {'HAL-MLE':<12} | "
                      f"{hal_res['bias']:<15.6f} | "
                      f"{hal_res['variance']:<15.6f} | "
                      f"{hal_res['mse']:<15.6f} | "
                      f"{hal_res['bias_over_se']:<15.6f}")

                # Efficient Estimator Row
                eff_res = results['efficient'][p]
                print(f"{'':<6} {'':<14} | {'Efficient':<12} | "
                      f"{eff_res['bias']:<15.6f} | "
                      f"{eff_res['variance']:<15.6f} | "
                      f"{eff_res['mse']:<15.6f} | "
                      f"{eff_res['bias_over_se']:<15.6f}")
            
            if i < len(parameters) - 1 or n_sample != max(dgp_summary.keys()):
                print("-" * len(header))
            
            print(f"# Number of experiments: {n_exp}")

def create_efficiency_plots(analysis_results, save_dir="experiments/uniform_convergence/efficiency_plots"):
    """
    Create and save plots of MSE, Variance, and Bias/SE against sample size.
    
    Args:
        analysis_results: Dictionary containing analysis results
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    parameters = ['mean', 'median', 's_0_5', 'second_moment']

    param_disply_names = {
        'mean': 'Mean',
        'median': 'Median',
        's_0_5': 'Survival at 0.5',
        'second_moment': 'Second Moment'
    }

    metrics = {
        'mse': 'Mean Squared Error (MSE)',
        'variance': 'Variance',
        'bias': '|Bias|',
        's.e.': '|Standard Error|',
        'bias_over_se': '|Bias| / Standard Error'
    }

    for dgp_name, dgp_summary in analysis_results.items():
        dgp_save_dir = os.path.join(save_dir, dgp_name)
        os.makedirs(dgp_save_dir, exist_ok=True)

        sample_sizes = sorted(dgp_summary.keys())
        if not sample_sizes:
            continue

        for param in parameters:
            for metric_key, metric_name in metrics.items():
                
                # Extract metric values for both estimators
                hal_mle_values = []
                efficient_values = []
                
                for n in sample_sizes:
                    if n in dgp_summary:
                        hal_val = dgp_summary[n]['summary']['hal_mle'][param][metric_key]
                        eff_val = dgp_summary[n]['summary']['efficient'][param][metric_key]
                        hal_mle_values.append(hal_val)
                        efficient_values.append(eff_val)

                if not hal_mle_values:  # Skip if no data
                    continue

                plt.figure(figsize=(10, 6))
                
                plt.plot(sample_sizes, hal_mle_values, 'o-', label='HAL-MLE', linewidth=2, markersize=6)
                plt.plot(sample_sizes, efficient_values, 's--', label='Efficient Estimator', linewidth=2, markersize=6)

                # Use log scale for better visualization of convergence
                if metric_key in ['mse', 'variance']:
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.grid(True, which="both", ls="--", alpha=0.3)
                else:  # bias_over_se
                    plt.xscale('log')
                    plt.grid(True, ls="--", alpha=0.3)

                plt.xlabel("Sample Size (N)", fontsize=12)
                plt.ylabel(metric_name, fontsize=12)
                # plt.title(f"{metric_name} of {param.replace('_', ' ').title()} Estimator\nDGP: {dgp_name}", fontsize=14)
                plt.title(f"{metric_name} of {param_disply_names[param]} Estimator\nDGP: {dgp_name}", fontsize=14)
                plt.legend(fontsize=11)
                plt.tight_layout()
                
                # Save plot
                plot_filename = f"{param}_{metric_key}.png"
                plt.savefig(os.path.join(dgp_save_dir, plot_filename), dpi=300, bbox_inches='tight')
                plt.close()

    print(f"\nPlots saved to {os.path.abspath(save_dir)}")

def main():
    """Main function to run the asymptotic efficiency analysis."""
    parser = argparse.ArgumentParser(description="Asymptotic Efficiency Analysis for HAL-MLE")
    parser.add_argument("--dgp", type=str, help="Run for a specific DGP")
    parser.add_argument("--recompute_hal_density", action="store_true", help="Recompute HAL density using direct calculation instead of using precomputed density", default=False)
    args = parser.parse_args()

    print("Starting Asymptotic Efficiency Analysis...")
    
    dgp_names = [args.dgp] if args.dgp else list(DGP_CONFIGS.keys())
    eval_points = np.linspace(0, 1, 2001)  # Fine grid for integration

    # 1. Load results
    print("Step 1: Loading simulation results...")
    results_dict = load_cvxpy_results(dgp_names=dgp_names)

    # 2. Analyze efficiency
    print("Step 2: Analyzing efficiency...")
    analysis_results = analyze_efficiency(results_dict, eval_points)

    with open("experiments/uniform_convergence/efficiency_analysis_results.json", "w") as f:
        json.dump(analysis_results, f, indent=4)

    # 3. Display summary
    print("Step 3: Creating summary table...")
    create_summary_table(analysis_results)

    # 4. Create and save plots
    print("Step 4: Creating efficiency plots...")
    create_efficiency_plots(analysis_results)
    
    print("\nAsymptotic Efficiency Analysis complete!")

if __name__ == "__main__":
    main()