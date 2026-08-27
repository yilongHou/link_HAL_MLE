import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import json
import os

from utils import (
    TruncatedNormal,
    TruncatedGMM,
    Sinusoidal,
    StepFunction,
    plot_density,
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

# load results from json
fname = f"experiments/uniform_convergence/results/TruncatedGMMFiveSpikes_CVXPYEstimator_N200/seed_1362459.json"

path_parts = fname.split('/')
experiment_part = path_parts[-2]  # e.g., "TruncatedGMMFiveSpikes_CVXPYEstimator_N400"
seed_part = path_parts[-1]  # e.g., "seed_1362459.json"

# Parse experiment part
exp_split = experiment_part.split('_')
dgp_name = exp_split[0]  # e.g., "TruncatedGMMFiveSpikes"
estimator_name = exp_split[1]  # e.g., "CVXPYEstimator"
n_samples_str = exp_split[2][1:]  # remove 'N' from "N400"
n_samples = int(n_samples_str)

# Parse seed part
seed_str = seed_part.split('_')[1].split('.json')[0]
seed = int(seed_str)

print(f"Loading results from {fname} ...")
with open(fname, "r") as f:
    loaded_results = json.load(f)
estimation_results = loaded_results

# Extract data from the results (instead of regenerating)
data_points = estimation_results['HAL_results']['data_points']
data = pd.DataFrame({'W1': data_points})

print(f"Loaded {len(data_points)} data points for DGP {dgp_name}")
print(f"Data shape: {data.shape}")
print(f"Data summary: {data['W1'].describe()}")

# Get DGP configuration for true density computation (for plotting reference)
dgp_config = DGP_CONFIGS[dgp_name]
sampler_class = SAMPLERS[dgp_config["sampler"]]
sampler = sampler_class(**dgp_config["sampler_params"])

# Check what keys are available in estimation_results
print("Keys in estimation_results:")
for key in estimation_results.keys():
    print(f"  {key}: {type(estimation_results[key])}")
    if isinstance(estimation_results[key], (list, np.ndarray)):
        if hasattr(estimation_results[key], 'shape'):
            print(f"    Shape: {estimation_results[key].shape}")
        else:
            print(f"    Length: {len(estimation_results[key])}")

print(f"Basis order from hyperparams: {estimation_results['hyperparams']['basis_order']}")

# =============================================================================
# DENSITY VARIANCE COMPUTATION DEMONSTRATION
# =============================================================================

# Import the necessary functions from density_variance
from density_variance.density_variance import (
    estimate_covariance_beta,
    density_confidence_interval
)

# Adapt the JSON structure to what the density variance functions expect
# The functions expect estimation_results['results']['theta_hat'] but our JSON has HAL_results
# Also need to add estimator_setup with basis_order from hyperparams
adapted_results = {
    'results': estimation_results['HAL_results'],
    'estimator_setup': {
        'estimator_params': {
            'basis_order': estimation_results['hyperparams']['basis_order']
        }
    }
}

# Step 1: Compute the covariance matrix of beta
print("Step 1: Computing covariance matrix of beta...")
cov_beta = estimate_covariance_beta(data, adapted_results)
print(f"Covariance matrix shape: {cov_beta.shape}")
print(f"Covariance matrix computed successfully!")

# Step 2: Define evaluation points for density variance and confidence intervals
print("\nStep 2: Setting up evaluation points...")

x_values = np.linspace(0, 1, 501)
print(f"Evaluating density variance at {len(x_values)} equally spaced points from 0 to 1")

# Step 3: Compute confidence intervals for the density estimates
print("\nStep 3: Computing confidence intervals...")
ci_df = density_confidence_interval(x_values, adapted_results, cov_beta, alpha=0.05)
print(f"Confidence intervals computed for {len(ci_df)} points")
print(f"Average standard error: {ci_df['se'].mean():.4f}")
print(f"Average interval width: {(ci_df['upper'] - ci_df['lower']).mean():.4f}")

# Verify that the recomputed density matches the stored density
print("\nStep 4: Verifying density computation...")
stored_density = np.array(estimation_results['HAL_results']['estimated_density'])
stored_grid = np.array(estimation_results['HAL_results']['grid_points'])

# Interpolate stored density to our evaluation points for comparison
from scipy.interpolate import interp1d
stored_density_interp = interp1d(stored_grid, stored_density, 
                                kind='linear', bounds_error=False, fill_value=0)
stored_density_at_x = stored_density_interp(x_values)

# Compare with our recomputed density
density_diff = np.abs(ci_df['density'].values - stored_density_at_x)
print(f"Max absolute difference between recomputed and stored density: {np.max(density_diff):.6f}")
print(f"Mean absolute difference: {np.mean(density_diff):.6f}")



####### VISUALIZATION #######


# Main plot
plt.figure(figsize=(10,6))
plt.plot(ci_df['x'], ci_df['density'], label='Recomputed Density', color='blue', linewidth=2)
plt.plot(stored_grid, stored_density, label='Stored Density (JSON)', color='orange', linestyle='--', alpha=0.8)
plt.fill_between(ci_df['x'], ci_df['lower'], ci_df['upper'], color='dodgerblue', alpha=0.3,
                 label='95% Confidence Interval')
plt.hist(data['W1'], bins=50, density=True, alpha=0.3, label='Training Data Histogram', color='green')
plt.plot(x_values, sampler.compute_density(x_values), label='True Density', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Density Estimate with 95% Confidence Intervals')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
