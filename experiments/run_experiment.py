"""
Run a single experiment based on a JSON setup file.

This script takes a path to a JSON file describing the experiment setup.
It initializes a sampler and an estimator based on the setup, generates data,
fits the estimator, and saves the results and logs.

For bulk experiments, a --seed parameter must be provided to select which
seed to run from the list of seeds in the setup file.

Example usage for a single experiment:
    python experiments/run_experiment.py experiments/single_trunc_gmm/setups/TruncatedGMM_CVXPYEstimator_Order0.json

Example usage for a bulk experiment:
    python experiments/run_experiment.py experiments/bulk_trunc_gmm/setups/your_bulk_setup.json --seed 123
"""
import argparse
import json
import logging
import os
import sys
import time

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

# --- Import Samplers ---
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

# --- Import Estimators ---
from methods import (
    CVXPYEstimator,
    FISTAEstimator,
    ProjectedGDEstimator,
    ProximalGDEstimator,
    ProximalAdaGradEstimator,
    ProximalNewtonEstimator,
    ProximalNewtonLBFGSEstimator,
    ProximalNewtonScaledDiagonalCDEstimator,
    ProximalNewtonLBFGSFullEstimator,
    AutoDiffEstimator,
    KDEEstimator,
    TrendFilteringADMMEstimator,
    # LogSplinesEstimator,
)

ESTIMATORS = {
    "CVXPYEstimator": CVXPYEstimator,
    "FISTAEstimator": FISTAEstimator,
    "ProjectedGDEstimator": ProjectedGDEstimator,
    "ProximalGDEstimator": ProximalGDEstimator,
    "ProximalAdaGradEstimator": ProximalAdaGradEstimator,
    "ProximalNewtonEstimator": ProximalNewtonEstimator,
    "ProximalNewtonLBFGSEstimator": ProximalNewtonLBFGSEstimator,
    "ProximalNewtonScaledDiagonalCDEstimator": ProximalNewtonScaledDiagonalCDEstimator,
    "ProximalNewtonLBFGSFullEstimator": ProximalNewtonLBFGSFullEstimator,
    "AutoDiffEstimator": AutoDiffEstimator,
    "KDEEstimator": KDEEstimator,
    "TrendFilteringADMMEstimator": TrendFilteringADMMEstimator,
    # "LogSplinesEstimator": LogSplinesEstimator,
}


def serialize_dict_to_json(estimator_result_dict: dict) -> dict:
    """Serialize estimator results to a dictionary. (for things like numpy arrays and pandas objects)"""
    serialized_results = {}
    for key, value in estimator_result_dict.items():
        if isinstance(value, np.ndarray):
            serialized_results[key] = value.tolist()
        elif isinstance(value, (np.integer, np.int64)):
            serialized_results[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            serialized_results[key] = float(value)
        elif isinstance(value, pd.Series):
            serialized_results[key] = value.tolist()
        elif isinstance(value, pd.DataFrame):
            serialized_results[key] = value.to_dict(orient='list')
        else:
            serialized_results[key] = value
    return serialized_results


def run_experiment(setup_fpath: str, seed: int = None):
    """
    Main function to run a single experiment.

    Args:
        setup_fpath: Path to the JSON setup file.
        seed: The random seed to use. If provided, it overrides the seed in the file
              and is required for bulk experiments.
    """
    # --- Logging setup for this script (console only) ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Starting experiment for setup: {setup_fpath}")

    # --- Load setup ---
    with open(setup_fpath, 'r') as f:
        setup = json.load(f)

    # --- Path and Seed setup ---
    base_dir = os.path.dirname(os.path.dirname(setup_fpath))
    fname = os.path.basename(setup_fpath)
    
    if 'random_seeds' in setup:
        if seed is None:
            logging.error("Bulk experiment setup detected, but no --seed specified.")
            sys.exit(1)
        if seed not in setup['random_seeds']:
            logging.error(f"Seed {seed} not found in the list of seeds in the setup file.")
            sys.exit(1)
        
        np.random.seed(seed)
        logging.info(f"Using random seed from bulk config: {seed}")
        
        # Modify output paths to include the seed
        base_fname, ext = os.path.splitext(fname)
        run_fname = f"{base_fname}_seed_{seed}{ext}"
        
        result_dir = os.path.join(base_dir, 'results')
        log_dir = os.path.join(base_dir, 'logs')
        
        result_fpath = os.path.join(result_dir, run_fname)
        log_fpath = os.path.join(log_dir, os.path.splitext(run_fname)[0] + '.log')

    else: # Original behavior for single experiment
        result_dir = os.path.join(base_dir, 'results')
        log_dir = os.path.join(base_dir, 'logs')
        
        result_fpath = os.path.join(result_dir, fname)
        log_fpath = os.path.join(log_dir, os.path.splitext(fname)[0] + '.log')

        run_seed = setup.get('random_seed', 42)
        np.random.seed(run_seed)
        logging.info(f"Using random seed: {run_seed}")

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    logging.info(f"Results will be saved to: {result_fpath}")
    logging.info(f"Logs will be saved to: {log_fpath}")

    # --- Sampler and Data Generation ---
    sampler_setup = setup['sampler_setup']
    sampler_name = sampler_setup['sampler']
    sampler_params = sampler_setup['sampler_params']
    n_samples = sampler_setup['n_samples']
    
    if sampler_name not in SAMPLERS:
        logging.error(f"Sampler '{sampler_name}' not found.")
        sys.exit(1)
        
    sampler_cls = SAMPLERS[sampler_name]
    sampler = sampler_cls(**sampler_params)
    logging.info(f"Initialized sampler: {sampler_name} with params: {sampler_params}")

    data = pd.DataFrame({'W1': sampler.generate_samples(n_samples)})
    logging.info(f"Generated {n_samples} samples.")

    # --- Estimator ---
    estimator_setup = setup['estimator_setup']
    estimator_name = estimator_setup['estimator']
    estimator_params = estimator_setup['estimator_params']

    # Add the log_fpath to the estimator params. The BaseEstimator will handle it.
    estimator_params['log_dir'] = log_fpath

    if estimator_name not in ESTIMATORS:
        logging.error(f"Estimator '{estimator_name}' not found.")
        sys.exit(1)

    estimator_cls = ESTIMATORS[estimator_name]
    estimator = estimator_cls(**estimator_params)
    logging.info(f"Initialized estimator: {estimator_name} with params: {estimator_params}")

    # --- Fitting ---
    logging.info("Fitting estimator...")
    start_time = time.time()
    fit_params = {'data': data}
    estimator.fit(**fit_params)
    end_time = time.time()
    runtime = end_time - start_time
    logging.info(f"Fitting finished in {runtime:.2f} seconds.")

    # --- Results ---
    estimation_results = estimator.get_results()
    estimation_results['runtime'] = runtime

    # --- Save results ---
    results_payload = {
        "sampler_setup": sampler_setup,
        "estimator_setup": estimator_setup,
        "results": serialize_dict_to_json(estimation_results),
        "fit_params": {'data_shape': list(data.shape)},
    }
    if seed is not None:
        results_payload['random_seed'] = seed
    if setup.get('random_seed') is not None:
        results_payload['random_seed'] = setup['random_seed']


    with open(result_fpath, 'w') as f:
        json.dump(results_payload, f, indent=4)

    logging.info(f"Successfully saved results to {result_fpath}")


def main():
    """
    Parses command line arguments and runs the experiment.
    """
    parser = argparse.ArgumentParser(description="Run a density estimation experiment.")
    parser.add_argument("setup_fpath", type=str, help="Path to the experiment setup JSON file.")
    parser.add_argument("--seed", type=int, default=None, help="The random seed to use. Required for bulk experiments.")
    args = parser.parse_args()
    run_experiment(args.setup_fpath, args.seed)


if __name__ == "__main__":
    main()
