"""
Run bulk experiments with hyperparameter tuning in parallel.

This script takes a bulk experiment setup file (containing multiple random seeds)
and runs each seed as a separate experiment with hyperparameter tuning. Each
experiment includes:
1. Hyperparameter optimization using Optuna with cross-validation
2. Model fitting with optimal hyperparameters
3. Results saving (hyperparams, estimated density, HAL results)
4. Density plot generation

Example usage:
    python experiments/run_bulk_experiment.py \
        experiments/uniform_convergence/setups/TruncatedGMMFiveSpikes_CVXPYEstimator_N3200.json \
        --n-workers 7
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Suppress optuna logging
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import samplers
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

# Import estimators
from methods import (
    CVXPYEstimator,
    FISTAEstimator,
    ProximalNewtonEstimator,
    ProximalNewtonLBFGSFullEstimator,
    AutoDiffEstimator,
    KDEEstimator,
    TrendFilteringADMMEstimator,
    TrendFilteringCVXPYEstimator,
    TrendFilteringCVXPYPP,
    TrendFilteringCVXPYPPA2Layered,
    # LogSplinesEstimator,
)

ESTIMATORS = {
    "CVXPYEstimator": CVXPYEstimator,
    "FISTAEstimator": FISTAEstimator,
    "ProximalNewtonEstimator": ProximalNewtonEstimator,
    "ProximalNewtonLBFGSFullEstimator": ProximalNewtonLBFGSFullEstimator,
    "AutoDiffEstimator": AutoDiffEstimator,
    "KDEEstimator": KDEEstimator,
    "TrendFilteringADMMEstimator": TrendFilteringADMMEstimator,
    "TrendFilteringCVXPYEstimator": TrendFilteringCVXPYEstimator,
    "TrendFilteringCVXPYPP": TrendFilteringCVXPYPP,
    "TrendFilteringCVXPYPPA2Layered": TrendFilteringCVXPYPPA2Layered,
    # "LogSplinesEstimator": LogSplinesEstimator,
}

# Import hyperparameter tuner
from cross_validation.optuna_hyperparam_selector import OptunaHyperparameterTuner


def serialize_dict_to_json(data_dict: dict) -> dict:
    """Serialize dictionary with numpy arrays and pandas objects to JSON-compatible format."""
    serialized = {}
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        elif isinstance(value, (np.integer, np.int64)):
            serialized[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            serialized[key] = float(value)
        elif isinstance(value, pd.Series):
            serialized[key] = value.tolist()
        elif isinstance(value, pd.DataFrame):
            serialized[key] = value.to_dict(orient='list')
        else:
            serialized[key] = value
    return serialized


def create_density_plot(
    estimator_results: Dict[str, Any],
    data: pd.DataFrame,
    true_sampler: Any,
    seed: int,
    save_path: str,
    hal_grid: Optional[np.ndarray] = None,
    hal_density: Optional[np.ndarray] = None,
) -> None:
    """
    Create and save density plot comparing estimated vs true density.
    
    Args:
        estimator_results: Results from the fitted estimator
        data: Training data
        true_sampler: True data generating sampler
        seed: Random seed used for this experiment
        save_path: Path to save the plot
    """
    try:
        plt.figure(figsize=(12, 8))
        
        # Extract density estimation
        grid_points = estimator_results['grid_points']
        estimated_density = estimator_results['estimated_density']
        
        # Plot estimated density
        plt.plot(
            grid_points,
            estimated_density,
            color='#007acc',
            linewidth=2,
            label='Estimated Density'
        )

        # Plot HAL density if provided (for TFA2 comparisons)
        if hal_grid is not None and hal_density is not None:
            plt.plot(
                hal_grid,
                hal_density,
                color='#ff7f0e',
                linestyle='--',
                linewidth=2,
                label='HAL Density'
            )
        
        # Plot true density
        true_density = true_sampler.compute_density(np.array(grid_points))
        plt.plot(grid_points, true_density, 
                color='#d62728', linestyle='--', linewidth=2, label='True Density')
        
        # Plot data histogram
        plt.hist(data['W1'], bins=50, density=True, alpha=0.6, 
                label='Data Histogram', color='#2ca02c')
        
        # Formatting
        n_knots = estimator_results.get('n_selected_knots', 'N/A')
        plt.title(f'Density Estimation (Seed: {seed}, Knots: {n_knots})', fontsize=14)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        # If plotting fails, save an empty plot or skip
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, f'Plot generation failed: {str(e)}', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'Failed Plot (Seed: {seed})')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def _load_hal_density_for_seed(
    base_dir: str,
    dgp_name: str,
    n_samples: int,
    n_seeds: int,
    seed: int,
    basis_order: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load HAL density for a given seed if it exists."""
    hal_experiment = f"{dgp_name}_CVXPYEstimator_basis{basis_order}_N{n_samples}_s{n_seeds}_optuna"
    hal_path = os.path.join(base_dir, "results", hal_experiment, f"seed_{seed}.json")
    if not os.path.isfile(hal_path):
        return None, None
    try:
        with open(hal_path, "r") as f:
            obj = json.load(f)
        if not obj or "estimated_density" not in obj:
            return None, None
        grid = np.asarray(obj["estimated_density"]["gridpoints"], dtype=float)
        dens = np.asarray(obj["estimated_density"]["density"], dtype=float)
        if grid.size < 2 or grid.shape != dens.shape:
            return None, None
        return grid, dens
    except Exception:
        return None, None


def regenerate_density_plot_for_seed(args_tuple: Tuple[Dict[str, Any], int, str, str, str]) -> bool:
    """
    Regenerate the density plot for an existing per-seed result JSON, optionally overlaying HAL.

    Args:
        args_tuple: (setup, seed, results_dir, plots_dir, experiment_name)

    Returns:
        True if plot generation succeeded, False otherwise.
    """
    setup, seed, results_dir, plots_dir, experiment_name = args_tuple
    try:
        results_file = os.path.join(results_dir, f"seed_{seed}.json")
        if not os.path.isfile(results_file):
            return False
        with open(results_file, "r") as f:
            obj = json.load(f)
        if not obj:
            return False

        sampler_setup = setup["sampler_setup"]
        estimator_setup = setup["estimator_setup"]
        sampler_name = sampler_setup["sampler"]
        sampler_params = sampler_setup["sampler_params"]
        n_samples = int(sampler_setup["n_samples"])

        sampler_class = SAMPLERS[sampler_name]
        sampler = sampler_class(**sampler_params)

        # Prefer stored data points (if available), else regenerate deterministically from seed.
        data_points = None
        if isinstance(obj, dict):
            hal_results = obj.get("HAL_results")
            if isinstance(hal_results, dict):
                data_points = hal_results.get("data_points")
        if data_points is None:
            np.random.seed(int(seed))
            data_points = sampler.generate_samples(n_samples)
        data = pd.DataFrame({"W1": np.asarray(data_points)})

        estimator_results = obj.get("HAL_results") or {}

        # Optional HAL overlay when plotting TFA2
        hal_grid = None
        hal_density = None
        estimator_name = estimator_setup.get("estimator")
        if estimator_name == "TrendFilteringCVXPYPPA2Layered":
            basis_order = int(estimator_setup.get("comparison_hal_basis_order", 0))
            n_seeds = len(setup.get("random_seeds", []))
            # results_dir is <base_dir>/results/<experiment_name> so go up two levels.
            base_dir = os.path.dirname(os.path.dirname(results_dir))
            hal_grid, hal_density = _load_hal_density_for_seed(
                base_dir=base_dir,
                dgp_name=str(sampler_setup.get("dgp_name", sampler_name)),
                n_samples=int(n_samples),
                n_seeds=int(n_seeds),
                seed=int(seed),
                basis_order=basis_order,
            )

        plot_file = os.path.join(plots_dir, f"seed_{seed}_density.png")
        create_density_plot(
            estimator_results=estimator_results,
            data=data,
            true_sampler=sampler,
            seed=int(seed),
            save_path=plot_file,
            hal_grid=hal_grid,
            hal_density=hal_density,
        )
        return True
    except Exception:
        return False


def run_single_seed_experiment(args_tuple: Tuple[Dict[str, Any], int, str, str, str, int, int]) -> bool:
    """
    Run a single experiment for one random seed.
    
    Args:
        args_tuple: Tuple containing (setup, seed, results_dir, plots_dir, experiment_name, n_trials, cv_folds)
    
    Returns:
        True if successful, False if failed
    """
    setup, seed, results_dir, plots_dir, experiment_name, n_trials, cv_folds = args_tuple
    
    try:
        # Set random seed
        np.random.seed(seed)
        
        # Extract setup components
        sampler_setup = setup['sampler_setup']
        estimator_setup = setup['estimator_setup']
        
        # Generate data
        sampler_name = sampler_setup['sampler']
        sampler_params = sampler_setup['sampler_params']
        n_samples = sampler_setup['n_samples']
        
        sampler_class = SAMPLERS[sampler_name]
        sampler = sampler_class(**sampler_params)
        data = pd.DataFrame({'W1': sampler.generate_samples(n_samples)})
        
        # Hyperparameter tuning
        estimator_name = estimator_setup['estimator']
        fixed_params = (
            estimator_setup.get("fixed_params")
            or estimator_setup.get("estimator_params")
            or {}
        )
        tuner_n_grid_points = int(estimator_setup.get("n_grid_points", 200))
        
        tuner = OptunaHyperparameterTuner(
            estimator_name=estimator_name,
            data=data,
            sampler_setup=sampler_setup,
            cv_folds=cv_folds,
            metric="sll",
            random_state=seed,
            max_iter=50_000,
            n_grid_points=tuner_n_grid_points,
            fixed_params=fixed_params,
            silent=True  # Suppress all output
            # silent=False  # Enable output for debugging
        )
        
        # Run hyperparameter optimization
        tuner.optimize(
            n_trials=n_trials,
            timeout=None,
            show_progress=False
        )
        
        # Fit best model
        best_estimator = tuner.fit_best_model()
        
        # Get results
        estimator_results = best_estimator.get_results()
        grid_points, estimated_density = best_estimator.get_density()
        
        # Prepare results dictionary
        results_dict = {
            "hyperparams": tuner.best_params,
            "estimated_density": {
                "gridpoints": grid_points.tolist() if isinstance(grid_points, np.ndarray) else grid_points,
                "density": estimated_density.tolist() if isinstance(estimated_density, np.ndarray) else estimated_density
            },
            "HAL_results": serialize_dict_to_json(estimator_results)
        }
        
        # Save results JSON
        results_file = os.path.join(results_dir, f"seed_{seed}.json")
        os.makedirs(results_dir, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Create and save density plot
        plot_file = os.path.join(plots_dir, f"seed_{seed}_density.png")
        hal_grid = None
        hal_density = None
        if estimator_name == "TrendFilteringCVXPYPPA2Layered":
            basis_order = int(estimator_setup.get("comparison_hal_basis_order", 0))
            n_seeds = len(setup.get("random_seeds", []))
            # results_dir is <base_dir>/results/<experiment_name> so go up two levels.
            base_dir = os.path.dirname(os.path.dirname(results_dir))
            hal_grid, hal_density = _load_hal_density_for_seed(
                base_dir=base_dir,
                dgp_name=str(sampler_setup.get("dgp_name", sampler_name)),
                n_samples=int(n_samples),
                n_seeds=int(n_seeds),
                seed=int(seed),
                basis_order=basis_order,
            )
        create_density_plot(
            estimator_results,
            data,
            sampler,
            seed,
            plot_file,
            hal_grid=hal_grid,
            hal_density=hal_density,
        )
        
        return True
        
    except Exception as e:
        # For failed experiments, save empty JSON
        try:
            results_file = os.path.join(results_dir, f"seed_{seed}.json")
            os.makedirs(results_dir, exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump({}, f)
        except:
            pass
        
        return False


def run_bulk_experiment(
    setup_path: str,
    n_workers: int = 7,
    n_trials: int = 50,
    cv_folds: int = 5,
    force: bool = False,
    regen_plots: bool = False,
) -> None:
    """
    Run bulk experiment with multiple seeds in parallel.
    
    Args:
        setup_path: Path to the bulk experiment setup JSON file
        n_workers: Number of parallel workers
    """
    # Load setup
    with open(setup_path, 'r') as f:
        setup = json.load(f)
    
    if 'random_seeds' not in setup:
        raise ValueError("Setup file must contain 'random_seeds' list for bulk experiments")
    
    # Extract experiment name from filename
    experiment_name = os.path.splitext(os.path.basename(setup_path))[0]
    base_dir = os.path.dirname(os.path.dirname(setup_path))
    
    results_dir = os.path.join(base_dir, 'results', experiment_name)
    plots_dir = os.path.join(base_dir, 'plots', experiment_name)
    
    print(f"Running bulk experiment: {experiment_name}")
    print(f"Number of seeds: {len(setup['random_seeds'])}")
    print(f"Parallel workers: {n_workers}")
    print(f"Results directory: {results_dir}")
    print(f"Plots directory: {plots_dir}")
    
    # Prepare arguments for multiprocessing
    args_list = []
    plot_args_list: List[Tuple[Dict[str, Any], int, str, str, str]] = []
    for seed in setup["random_seeds"]:
        seed_result_path = os.path.join(results_dir, f"seed_{seed}.json")
        if (not force) and os.path.exists(seed_result_path):
            # Skip seed if results exist, unless forcing rerun
            if regen_plots:
                plot_args_list.append((setup, seed, results_dir, plots_dir, experiment_name))
            continue
        args_list.append((setup, seed, results_dir, plots_dir, experiment_name, n_trials, cv_folds))
        if regen_plots:
            plot_args_list.append((setup, seed, results_dir, plots_dir, experiment_name))
    
    # Run experiments in parallel with progress bar
    successful_experiments = 0
    failed_experiments = 0
    
    if args_list:
        with Pool(n_workers) as pool:
            with tqdm(total=len(args_list), desc="Running experiments") as pbar:
                for result in pool.imap(run_single_seed_experiment, args_list):
                    if result:
                        successful_experiments += 1
                    else:
                        failed_experiments += 1
                    pbar.update(1)
    else:
        print("No experiments to run (all per-seed results already exist).")

    # Optionally regenerate plots from existing result JSONs (no Optuna reruns)
    if regen_plots and plot_args_list:
        print(f"\nRegenerating {len(plot_args_list)} density plots from existing results...")
        regenerated = 0
        with Pool(n_workers) as pool:
            with tqdm(total=len(plot_args_list), desc="Regenerating plots") as pbar:
                for ok in pool.imap(regenerate_density_plot_for_seed, plot_args_list):
                    regenerated += int(bool(ok))
                    pbar.update(1)
        print(f"Regenerated plots: {regenerated}/{len(plot_args_list)}")
    
    print(f"\nBulk experiment completed!")
    print(f"Successful experiments: {successful_experiments}")
    print(f"Failed experiments: {failed_experiments}")
    if not successful_experiments:
        print("No successful experiments to evaluate.")
    else:
        print(f"Success rate: {successful_experiments/(successful_experiments + failed_experiments)*100:.1f}%")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run bulk experiments with hyperparameter tuning in parallel"
    )
    parser.add_argument(
        "setup_path",
        type=str,
        help="Path to the bulk experiment setup JSON file"
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for hyperparameter tuning (default: 50)"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun even if per-seed result JSON already exists (overwrite).",
    )
    parser.add_argument(
        "--regen-plots",
        action="store_true",
        help="Regenerate density plots from existing per-seed JSON results (no rerun).",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.setup_path):
        print(f"Error: Setup file not found: {args.setup_path}")
        sys.exit(1)
    
    run_bulk_experiment(
        args.setup_path,
        args.n_workers,
        args.n_trials,
        args.cv_folds,
        force=args.force,
        regen_plots=args.regen_plots,
    )


if __name__ == "__main__":
    main()
