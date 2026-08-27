#!/usr/bin/env python3
"""
Plot oracle coverage across evaluation points for TruncatedGMMFiveSpikes at N=400.

This script:
- Loads all experiments for TruncatedGMMFiveSpikes_CVXPYEstimator_N400
- Interpolates stored densities to a common 201-point grid on [0, 1]
- Masks positions outside each experiment's data domain [x_min, x_max]
- Computes oracle coverage at each grid point using oracle SE across experiments
- Plots coverage (%) vs x and saves figure locally
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def get_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def compute_true_density_five_spikes(eval_points: np.ndarray) -> np.ndarray:
    """
    Compute true density for TruncatedGMMFiveSpikes on eval_points.
    """
    # Ensure project root is on path to import utils
    project_root = get_project_root()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from utils import TruncatedGMM  # deferred import to avoid side effects

    components = [
        {"mean": 0.45, "std": 0.005, "lower": 0.0, "upper": 1.0},
        {"mean": 0.475, "std": 0.005, "lower": 0.0, "upper": 1.0},
        {"mean": 0.5, "std": 0.005, "lower": 0.0, "upper": 1.0},
        {"mean": 0.525, "std": 0.005, "lower": 0.0, "upper": 1.0},
        {"mean": 0.55, "std": 0.005, "lower": 0.0, "upper": 1.0},
        {"mean": 0.5, "std": 0.05, "lower": 0.0, "upper": 1.0},
    ]
    weights = [0.06666667, 0.06666667, 0.06666667, 0.06666667, 0.06666667, 0.66666667]
    sampler = TruncatedGMM(components=components, weights=weights)
    return sampler.compute_density(eval_points)


def load_densities_and_masks(results_dir: str, eval_points: np.ndarray):
    """
    Load all experiment files, interpolate densities to eval_points, and build valid masks.

    Returns:
        densities: np.ndarray of shape (n_experiments, n_eval_points)
        valid_masks: np.ndarray of shape (n_experiments, n_eval_points) with True where in [x_min, x_max]
    """
    json_files = sorted(glob.glob(os.path.join(results_dir, '*.json')))
    densities_list = []
    masks_list = []

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            if 'HAL_results' not in data:
                continue

            hal = data['HAL_results']
            stored_density = np.asarray(hal['estimated_density'], dtype=float)
            stored_grid = np.asarray(hal['grid_points'], dtype=float)
            data_points = np.asarray(hal['data_points'], dtype=float)

            if stored_density.size == 0 or stored_grid.size == 0 or data_points.size == 0:
                continue

            # Ensure grid is sorted for interpolation safety
            order = np.argsort(stored_grid)
            stored_grid_sorted = stored_grid[order]
            stored_density_sorted = stored_density[order]

            interp_fn = interp1d(
                stored_grid_sorted,
                stored_density_sorted,
                kind='linear',
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True,
            )
            density_at_eval = interp_fn(eval_points)

            x_min = float(np.min(data_points))
            x_max = float(np.max(data_points))
            valid_mask = (eval_points >= x_min) & (eval_points <= x_max)

            # Keep full-range densities; do not mask to [x_min, x_max]
            density_at_eval = density_at_eval.astype(float)

            densities_list.append(density_at_eval)
            masks_list.append(valid_mask)
        except Exception:
            # Skip unreadable or malformed files silently
            continue

    if not densities_list:
        return None, None

    densities = np.vstack(densities_list)
    valid_masks = np.vstack(masks_list)
    return densities, valid_masks


def main():
    project_root = get_project_root()
    results_dir = os.path.join(
        project_root,
        'experiments', 'uniform_convergence', 'results',
        'TruncatedGMMFiveSpikes_CVXPYEstimator_N400',
    )

    eval_points = np.linspace(0.0, 1.0, 201)

    true_density = compute_true_density_five_spikes(eval_points)
    densities, valid_masks = load_densities_and_masks(results_dir, eval_points)

    if densities is None or valid_masks is None:
        print('No valid experiment files found. Aborting.')
        return

    # Compute oracle SE at each eval point across full range
    oracle_se = np.nanstd(densities, axis=0, ddof=1)

    # Build coverage indicators per experiment at each eval point
    lower = densities - 1.96 * oracle_se
    upper = densities + 1.96 * oracle_se

    # Prepare indicator array with NaN where SE is NaN or density is NaN (e.g., outside interpolation support)
    invalid_positions = np.isnan(densities) | np.isnan(oracle_se)
    within = (true_density >= lower) & (true_density <= upper)
    indicators = np.where(invalid_positions, np.nan, within.astype(float))

    # Oracle coverage probability at each x (percentage)
    oracle_coverage_percent = np.nanmean(indicators, axis=0) * 100.0

    # Output directory
    out_dir = os.path.join(project_root, 'paper', 'resources', 'density_asymptotic_normality_and_var_est')
    os.makedirs(out_dir, exist_ok=True)

    # Save CSV with oracle coverage per x
    valid_fraction = np.mean(valid_masks, axis=0)
    df = pd.DataFrame({
        'x': eval_points,
        'oracle_coverage_percent': oracle_coverage_percent,
        'valid_fraction_percent': valid_fraction * 100.0,
        'oracle_se': oracle_se,
    })
    csv_path = os.path.join(out_dir, 'Oracle_Coverage_TruncatedGMMFiveSpikes_N400_per_x.csv')
    df.to_csv(csv_path, index=False)
    print(f'Saved CSV: {csv_path}')

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(eval_points, oracle_coverage_percent, color='green', linewidth=1.5, label='Oracle Coverage')
    plt.axhline(y=95, color='red', linestyle='--', linewidth=1.0, label='95% Target')

    # Visual cue for fraction of valid experiments per x (optional, as light fill)
    plt.fill_between(eval_points, 0, valid_fraction * 100.0, color='gray', alpha=0.1, label='Valid fraction (%)')

    mean_oracle = float(np.nanmean(oracle_coverage_percent))
    plt.title(f'TruncatedGMMFiveSpikes N=400 — Oracle coverage per x (mean={mean_oracle:.1f}%)')
    plt.xlabel('x')
    plt.ylabel('Coverage (%)')
    plt.ylim(0, 101)
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')

    out_path = os.path.join(out_dir, 'Oracle_Coverage_TruncatedGMMFiveSpikes_N400_per_x.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f'Saved plot: {out_path}')


if __name__ == '__main__':
    main()


