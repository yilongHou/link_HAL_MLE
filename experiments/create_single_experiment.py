"""
Create single experiment configurations for all combinations of components.

This script reads combination files for data generation processes and estimators,
and generates experiment configuration files for each distribution with all 5 
estimator methods (with 3 basis orders each). It also creates the necessary
directory structure with logs, plots, and results folders.

Example usage:
    python experiments/create_single_experiment.py \
        --output_base_dir experiments/compare_knot_selection
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_combination_files() -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load the three combination files for data generation processes, estimators, and sample sizes.
    
    Returns:
        Tuple of (data_generation_processes, estimators, sample_sizes)
    """
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    combinations_dir = os.path.join(script_dir, "compare_knot_selection", "combinations")
    
    # Load data generation processes
    dgp_path = os.path.join(combinations_dir, "data_generation_processes.json")
    try:
        with open(dgp_path, 'r') as f:
            data_generation_processes = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data generation processes file not found: {dgp_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in data generation processes file: {dgp_path}")
    
    # Load estimators
    estimators_path = os.path.join(combinations_dir, "estimators.json")
    try:
        with open(estimators_path, 'r') as f:
            estimators = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Estimators file not found: {estimators_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in estimators file: {estimators_path}")
    
    # Load sample sizes
    sample_sizes_path = os.path.join(combinations_dir, "sample_sizes.json")
    try:
        with open(sample_sizes_path, 'r') as f:
            sample_sizes = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Sample sizes file not found: {sample_sizes_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in sample sizes file: {sample_sizes_path}")
    
    return data_generation_processes, estimators, sample_sizes


def get_estimator_params(estimator_name: str, basis_order: int) -> Dict[str, Any]:
    """
    Get the hyperparameters for each estimator type and basis order.
    
    Args:
        estimator_name: Name of the estimator
        basis_order: Basis order (0, 1, or 2)
    
    Returns:
        Dictionary of estimator parameters
    """
    if estimator_name == "CVXPYEstimator":
        return {
            "basis_order": basis_order,
            "norm_constraint": 8
        }
    elif estimator_name == "FISTAEstimator":
        return {
            "lam": 15,
            "L": 1000000.0,
            "n_iterations": 40001,
            "tol": 1e-06,
            "basis_order": basis_order,
            "log_frequency": 10,
            "n_grid_points": 200
        }
    elif estimator_name == "ProximalAdaGradEstimator":
        return {
            "lam": 12,
            "n_iterations": 50001,
            "tol": 1e-06,
            "basis_order": basis_order,
            "log_frequency": 10,
            "n_grid_points": 200,
            "alpha": 0.0001
        }
    elif estimator_name == "ProximalNewtonEstimator":
        return {
            "lam": 15,
            "n_iterations": 101,
            "tol": 0.0005,
            "basis_order": basis_order,
            "log_frequency": 1,
            "n_grid_points": 200
        }
    elif estimator_name == "ProximalNewtonLBFGSEstimator":
        return {
            "lam": 15,
            "n_iterations": 10001,
            "tol": 1e-06,
            "basis_order": basis_order,
            "log_frequency": 100,
            "n_grid_points": 200
        }
    else:
        raise ValueError(f"Unknown estimator: {estimator_name}")


def create_experiment_config(dgp: Dict[str, Any], estimator_name: str, 
                           sample_size: Dict[str, Any], basis_order: int) -> Dict[str, Any]:
    """
    Create a complete experiment configuration from individual components.
    
    Args:
        dgp: Data generation process configuration
        estimator_name: Name of the estimator
        sample_size: Sample size configuration
        basis_order: Basis order (0, 1, or 2)
    
    Returns:
        Complete experiment configuration dictionary
    """
    # Create sampler setup by combining DGP and sample size
    sampler_setup = dgp["sampler_setup"].copy()
    sampler_setup["n_samples"] = sample_size["n_samples"]
    
    # Get estimator parameters
    estimator_params = get_estimator_params(estimator_name, basis_order)
    
    # Create the complete configuration
    config = {
        "sampler_setup": sampler_setup,
        "estimator_setup": {
            "estimator": estimator_name,
            "estimator_params": estimator_params
        }
    }
    
    return config


def generate_filename(sampler_name: str, estimator_name: str, basis_order: int) -> str:
    """
    Generate a meaningful filename for the experiment configuration.
    
    Args:
        sampler_name: Name of the sampler/DGP
        estimator_name: Name of the estimator
        basis_order: Basis order (0, 1, or 2)
    
    Returns:
        Filename string in format: {sampler_name}_{estimator_name}_Order{basis_order}.json
    """
    return f"{sampler_name}_{estimator_name}_Order{basis_order}.json"


def create_directory_structure(base_dir: str, sampler_name: str) -> str:
    """
    Create the directory structure for a single experiment.
    
    Args:
        base_dir: Base directory for experiments
        sampler_name: Name of the sampler/DGP
    
    Returns:
        Path to the experiment directory
    """
    experiment_dir = os.path.join(base_dir, f"single_{sampler_name}")
    
    # Create subdirectories
    subdirs = ["logs", "plots", "results", "setups"]
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    
    return experiment_dir


def create_all_single_experiments(output_base_dir: str) -> None:
    """
    Create single experiment configurations for all combinations of components.
    
    Args:
        output_base_dir: Base directory to save the experiment configurations.
    """
    # Load combination files
    data_generation_processes, estimators, sample_sizes = load_combination_files()
    
    # Use the first (and only) sample size configuration
    sample_size = sample_sizes[0]
    
    # Define basis orders to generate
    basis_orders = [0, 1, 2]
    
    # Ensure base output directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    total_configs = 0
    
    # Generate configurations for each DGP
    for dgp in data_generation_processes:
        sampler_name = dgp["sampler_name"]
        print(f"Creating experiments for {sampler_name}...")
        
        # Create directory structure
        experiment_dir = create_directory_structure(output_base_dir, sampler_name)
        setups_dir = os.path.join(experiment_dir, "setups")
        
        # Generate configurations for each estimator and basis order
        for estimator in estimators:
            estimator_name = estimator["estimator_setup"]["estimator"]
            
            for basis_order in basis_orders:
                # Create experiment configuration
                config = create_experiment_config(dgp, estimator_name, sample_size, basis_order)
                
                # Generate filename
                filename = generate_filename(sampler_name, estimator_name, basis_order)
                filepath = os.path.join(setups_dir, filename)
                
                # Save configuration
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=4)
                
                total_configs += 1
                print(f"  Created: {filename}")
    
    print(f"\nSuccessfully created {total_configs} experiment configurations.")
    print(f"Experiments saved to: {output_base_dir}")
    print(f"Generated {len(data_generation_processes)} experiment directories with logs, plots, results, and setups folders.")


def main():
    """
    Parse command-line arguments and generate all single experiment configurations.
    """
    parser = argparse.ArgumentParser(
        description="Generate single experiment configurations for all combinations of data generation processes and estimators."
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="experiments/compare_knot_selection",
        help="Base directory to save the experiment configurations (default: experiments/compare_knot_selection)"
    )
    
    args = parser.parse_args()
    
    try:
        create_all_single_experiments(args.output_base_dir)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
