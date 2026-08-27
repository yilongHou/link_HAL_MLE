"""
Create bulk experiment configurations from all combinations of components.

This script reads three combination files (data generation processes, estimators, 
and sample sizes) and generates all possible combinations. Each combination creates
a complete experiment configuration file with sampler setup, estimator setup, and
multiple random seeds.

Example usage:
    python experiments/create_bulk_experiment.py \
        --output_dir experiments/uniform_convergence/setups \
        --num_seeds 1000
"""
import argparse
import json
import os
import sys
import random
from itertools import product
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
    combinations_dir = os.path.join(script_dir, "uniform_convergence", "combinations")
    
    # Load data generation processes
    dgp_path = os.path.join(combinations_dir, "data_generation_processes.json")
    try:
        with open(dgp_path, 'r') as f:
            data_generation_processes = json.load(f)
    except FileNotFoundError:
        print(f"Error: Data generation processes file not found at {dgp_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {dgp_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load estimators
    estimators_path = os.path.join(combinations_dir, "estimators.json")
    try:
        with open(estimators_path, 'r') as f:
            estimators = json.load(f)
    except FileNotFoundError:
        print(f"Error: Estimators file not found at {estimators_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {estimators_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load sample sizes
    sample_sizes_path = os.path.join(combinations_dir, "sample_sizes.json")
    try:
        with open(sample_sizes_path, 'r') as f:
            sample_sizes = json.load(f)
    except FileNotFoundError:
        print(f"Error: Sample sizes file not found at {sample_sizes_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {sample_sizes_path}", file=sys.stderr)
        sys.exit(1)
    
    return data_generation_processes, estimators, sample_sizes


def generate_random_seeds(num_seeds: int, seed_start: int = 42) -> List[int]:
    """
    Generate a list of random seeds for reproducible experiments.
    
    Args:
        num_seeds (int): The number of random seeds to generate.
        seed_start (int): The starting value for the random seed sequence.
    
    Returns:
        List[int]: List of random seed integers.
    """
    random.seed(seed_start)
    return [random.randint(0, int(1e8)) for _ in range(num_seeds)]


def create_experiment_config(dgp: Dict[str, Any], estimator: Dict[str, Any], 
                           sample_size: Dict[str, Any], random_seeds: List[int]) -> Dict[str, Any]:
    """
    Create a complete experiment configuration from individual components.
    
    Args:
        dgp: Data generation process configuration
        estimator: Estimator configuration  
        sample_size: Sample size configuration
        random_seeds: List of random seeds
    
    Returns:
        Complete experiment configuration dictionary
    """
    # Create sampler setup by combining DGP and sample size
    sampler_setup = dgp["sampler_setup"].copy()
    sampler_setup["n_samples"] = sample_size["n_samples"]
    
    # Create the complete configuration
    config = {
        "sampler_setup": sampler_setup,
        "estimator_setup": estimator["estimator_setup"],
        "random_seeds": random_seeds
    }
    
    return config


def generate_filename(dgp: Dict[str, Any], estimator: Dict[str, Any], 
                     sample_size: Dict[str, Any]) -> str:
    """
    Generate a meaningful filename for the experiment configuration.
    
    Args:
        dgp: Data generation process configuration
        estimator: Estimator configuration
        sample_size: Sample size configuration
    
    Returns:
        Filename string in format: {sampler_name}_{estimator_name}_N{n_samples}.json
    """
    sampler_name = dgp["sampler_name"]
    estimator_name = estimator["estimator_setup"]["estimator"]
    n_samples = sample_size["n_samples"]
    
    return f"{sampler_name}_{estimator_name}_N{n_samples}.json"


def create_all_bulk_experiments(output_dir: str, num_seeds: int = 1000, seed_start: int = 42) -> None:
    """
    Create bulk experiment configurations for all combinations of components.
    
    Args:
        output_dir (str): Directory to save the experiment configuration files.
        num_seeds (int): Number of random seeds to generate per experiment.
        seed_start (int): Starting value for the random seed sequence.
    """
    # Load combination files
    data_generation_processes, estimators, sample_sizes = load_combination_files()
    
    # Generate random seeds (same for all experiments for consistency)
    random_seeds = generate_random_seeds(num_seeds, seed_start)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all combinations
    total_combinations = len(data_generation_processes) * len(estimators) * len(sample_sizes)
    print(f"Generating {total_combinations} experiment configurations...")
    
    created_count = 0
    for dgp, estimator, sample_size in product(data_generation_processes, estimators, sample_sizes):
        # Create experiment configuration
        config = create_experiment_config(dgp, estimator, sample_size, random_seeds)
        
        # Generate filename
        filename = generate_filename(dgp, estimator, sample_size)
        output_path = os.path.join(output_dir, filename)
        
        # Save configuration file
        try:
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
            created_count += 1
        except IOError as e:
            print(f"Error writing to output file at {output_path}: {e}", file=sys.stderr)
            continue
    
    print(f"Successfully created {created_count} experiment configurations with {num_seeds} seeds each.")
    print(f"Saved to directory: {output_dir}")





def main():
    """
    Parse command-line arguments and generate all bulk experiment configurations.
    """
    parser = argparse.ArgumentParser(
        description="Generate bulk experiment configurations for all combinations of data generation processes, estimators, and sample sizes."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/uniform_convergence/setups",
        help="Directory to save the experiment configuration files (default: experiments/uniform_convergence/setups)."
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=1000,
        help="Number of random seeds to generate per experiment (default: 1000)."
    )
    parser.add_argument(
        "--seed_start",
        type=int,
        default=42,
        help="Starting integer for the seed sequence (default: 42)."
    )

    args = parser.parse_args()

    create_all_bulk_experiments(
        output_dir=args.output_dir,
        num_seeds=args.num_seeds,
        seed_start=args.seed_start
    )


if __name__ == "__main__":
    main()
