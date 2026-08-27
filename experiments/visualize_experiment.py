"""
Visualize the results of a single experiment.

This script takes a path to a JSON setup file, finds the corresponding
results and log files, and generates a series of plots to visualize
the experiment's outcome and the estimator's convergence behavior.

Example usage:
    python experiments/visualize_experiment.py experiments/single_trunc_gmm/setups/TruncatedGMM_FISTAEstimator_Order0.json
"""
import argparse
import json
import logging
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import Project Modules ---
from density_variance.density_variance import (
    estimate_covariance_beta,
    density_confidence_interval
)
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

def parse_log_file(log_fpath: str) -> pd.DataFrame:
    """Parses the experiment log file to extract convergence metrics."""
    log_entries = []
    # Regex to capture iteration and key=value pairs
    log_pattern = re.compile(r"Iter\s+(\d+):\s+(.*)")
    kv_pattern = re.compile(r"([\w|‖\[\]:]+)=(-?[\d.]+(?:[eE][+-]?\d+)?)")

    with open(log_fpath, 'r') as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                iteration = int(match.group(1))
                metrics_str = match.group(2)
                
                entry = {'iteration': iteration}
                kv_pairs = kv_pattern.findall(metrics_str)
                for key, value in kv_pairs:
                    # Clean up key and convert value to float
                    clean_key = key.replace('‖', '').replace('‖₁', '_l1').strip()
                    entry[clean_key] = float(value)
                log_entries.append(entry)
                
    return pd.DataFrame(log_entries)

def plot_density_with_ci(results: dict, data: pd.DataFrame, sampler: object, fpath: str):
    """Generates and saves the main density plot with confidence intervals."""
    logging.info("Generating density plot with confidence intervals...")
    
    # 1. Estimate covariance and confidence intervals
    cov_beta = estimate_covariance_beta(data, results)
    x_vals = np.linspace(0, 1, 200)
    ci_df = density_confidence_interval(x_vals, results, cov_beta, alpha=0.05)

    # 2. Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Estimated density and CI
    ax.plot(ci_df['x'], ci_df['density'], color='#007acc', label='Estimated Density')
    ax.fill_between(ci_df['x'], ci_df['lower'], ci_df['upper'], color='#007acc', alpha=0.5, label='95% Confidence Interval')

    # True density
    true_density = sampler.compute_density(x_vals)
    ax.plot(x_vals, true_density, color='#d62728', linestyle='--', label='True Density')

    # Data histogram
    ax.hist(data['W1'], bins=100, density=True, alpha=0.5, label='Training Data Histogram', color='#2ca02c')

    # 3. Formatting
    estimator_name = results['estimator_setup']['estimator']
    n_knots = results['results']['n_selected_knots']
    ax.set_title(f'{estimator_name} - Density Estimate (Selected Knots: {n_knots})', fontsize=16)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.savefig(fpath, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved density plot to {fpath}")

def plot_density_without_ci(results: dict, data: pd.DataFrame, sampler: object, fpath: str):
    """Generates and saves the main density plot without confidence intervals."""
    logging.info("Generating density plot without confidence intervals...")
    
    # 1. Extract results
    estimator_name = results['estimator_setup']['estimator']
    estimation_results = results['results']
    grid_points = estimation_results['grid_points']
    estimated_density = estimation_results['estimated_density']
    n_knots = estimation_results.get('n_selected_knots', 'N/A') # Use .get for non-HAL estimators

    # 2. Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Estimated density
    ax.plot(grid_points, estimated_density, color='#007acc', label='Estimated Density')

    # True density
    x_vals = np.linspace(0, 1, 200)
    true_density = sampler.compute_density(x_vals)
    ax.plot(x_vals, true_density, color='#d62728', linestyle='--', label='True Density')

    # Data histogram
    ax.hist(data['W1'], bins=100, density=True, alpha=0.5, label='Training Data Histogram', color='#2ca02c')

    # 3. Formatting
    if n_knots is not None:
        ax.set_title(f'{estimator_name} - Density Estimate (Selected Knots: {n_knots})', fontsize=16)
    else:
        ax.set_title(f'{estimator_name} - Density Estimate', fontsize=16)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.savefig(fpath, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved density plot to {fpath}")

def plot_convergence(log_df: pd.DataFrame, plot_dir: str, fname_base: str, estimator_name: str):
    """Generates and saves convergence plots on linear and log scales."""
    if log_df.empty:
        logging.warning("Log data is empty, skipping convergence plots.")
        return

    metrics = [col for col in log_df.columns if col != 'iteration']
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // n_cols

    for scale in ['linear', 'log']:
        logging.info(f"Generating {scale} scale convergence plots...")
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), squeeze=False)
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]
            ax.plot(log_df['iteration'], log_df[metric], marker='.', linestyle='-', markersize=3)
            ax.set_title(f'Iteration vs. {metric}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(metric)
            ax.set_xscale(scale)
            if scale == 'log' and metric in ['change', 'loss']:
                # Use symlog for metrics that can be negative or zero
                ax.set_yscale('symlog')
            else:
                ax.set_yscale('linear')

        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])

        fig.suptitle(f'{estimator_name} - Convergence Metrics ({scale.capitalize()} Scale)', fontsize=20)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        fpath = os.path.join(plot_dir, f'{fname_base}_convergence_{scale}_scale.png')
        plt.savefig(fpath, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved {scale} scale convergence plot to {fpath}")

def main():
    """Main function to run the visualization script."""
    parser = argparse.ArgumentParser(description="Visualize density estimation experiment results.")
    parser.add_argument("setup_fpath", type=str, help="Path to the experiment setup JSON file.")
    args = parser.parse_args()

    # --- Path Setup ---
    fname_no_ext = os.path.splitext(os.path.basename(args.setup_fpath))[0]
    base_dir = os.path.dirname(os.path.dirname(args.setup_fpath))

    result_fpath = os.path.join(base_dir, 'results', fname_no_ext + '.json')
    plot_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    if not os.path.exists(result_fpath):
        logging.error(f"Result file not found: {result_fpath}")
        sys.exit(1)

    # --- Load Data ---
    logging.info(f"Loading results from: {result_fpath}")
    with open(result_fpath, 'r') as f:
        results = json.load(f)

    # --- Recreate Sampler and Data ---
    seed = results.get('random_seed', 42)
    np.random.seed(seed)
    sampler_setup = results['sampler_setup']
    sampler_cls = SAMPLERS[sampler_setup['sampler']]
    sampler = sampler_cls(**sampler_setup['sampler_params'])
    data = pd.DataFrame({'W1': sampler.generate_samples(sampler_setup['n_samples'])})

    # --- Generate Plots ---
    estimator_name = results['estimator_setup']['estimator']
    
    # 1. Density plot
    density_plot_fpath = os.path.join(plot_dir, f'{fname_no_ext}_density_ci.png')
    try:
        plot_density_with_ci(results, data, sampler, density_plot_fpath)
    except Exception as e:
        logging.error(f"Error generating density with CI plot: {e}, trying without CI...")
        plot_density_without_ci(results, data, sampler, density_plot_fpath)

    # 2. Convergence plots
    try:
        log_fpath = os.path.join(base_dir, 'logs', fname_no_ext + '.log')
        if not os.path.exists(log_fpath):
            logging.error(f"Log file not found: {log_fpath}")
        logging.info(f"Parsing log file: {log_fpath}")
        log_df = parse_log_file(log_fpath)
        plot_convergence(log_df, plot_dir, fname_no_ext, estimator_name)
    except Exception as e:
        logging.error(f"Error generating convergence plots: {e}")

    logging.info("Visualization script finished.")

if __name__ == "__main__":
    main()
