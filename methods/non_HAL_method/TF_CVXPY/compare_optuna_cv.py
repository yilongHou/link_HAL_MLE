"""
Compare original TF-CVXPY vs TF-CVXPY-PP using Optuna cross-validation.

Example:
    uv run methods/non_HAL_method/TF_CVXPY/compare_optuna_cv.py --n-samples 800 --n-trials 30
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cross_validation.optuna_hyperparam_selector import (
    OptunaHyperparameterTuner,
    THREEPEAKS_SAMPLER_CONFIG,
)
from utils.sampler.truncated_gmm import TruncatedGMM


def _make_sampler_setup() -> Dict[str, Any]:
    return {
        "sampler": "TruncatedGMM",
        "sampler_params": THREEPEAKS_SAMPLER_CONFIG,
    }


def _sample_data(n_samples: int, sampler_setup: Dict[str, Any]) -> pd.DataFrame:
    sampler = TruncatedGMM(**sampler_setup["sampler_params"])
    samples = sampler.generate_samples(n_samples)
    return pd.DataFrame({"W1": samples})


def run_optuna_comparison(
    data: pd.DataFrame,
    sampler_setup: Dict[str, Any],
    n_trials: int = 50,
    cv_folds: int = 3,
    metric: str = "sll",
    n_grid_points: int = 200,
    show_progress: bool = True,
    save_plot: bool = True,
    plot_dir: str = "local/optuna_plots",
) -> Dict[str, Any]:
    """
    Run Optuna CV tuning for original TF-CVXPY and TF-CVXPY-PP, and compare results.
    """
    estimator_names = ["TrendFilteringCVXPYEstimator", "TrendFilteringCVXPYPP"]
    results: Dict[str, Any] = {}

    for name in estimator_names:
        tuner = OptunaHyperparameterTuner(
            estimator_name=name,
            data=data,
            sampler_setup=sampler_setup,
            cv_folds=cv_folds,
            metric=metric,
            n_grid_points=n_grid_points,
            silent=False,
            show_progress=show_progress,
        )
        optuna_result = tuner.optimize(n_trials=n_trials, show_progress=show_progress)
        best_estimator = tuner.fit_best_model()
        eval_metrics = tuner.evaluate_best_model(
            best_estimator,
            save_plot=save_plot,
            plot_dir=plot_dir,
        )

        results[name] = {
            "tuner": tuner,
            "best_estimator": best_estimator,
            "optuna_result": optuna_result,
            "eval_metrics": eval_metrics,
        }

    # Comparison plot
    if save_plot:
        plt.figure(figsize=(12, 8))
        grid = None
        for name, payload in results.items():
            grid_points, density = payload["best_estimator"].get_density()
            grid = grid_points
            plt.plot(grid_points, density, linewidth=2, label=f"{name} (best)")

        # True density
        true_sampler = TruncatedGMM(**sampler_setup["sampler_params"])
        if grid is not None:
            true_density = true_sampler.compute_density(grid)
            plt.plot(grid, true_density, linestyle="--", linewidth=2, label="True Density")

        # Data histogram
        plt.hist(data["W1"], bins=50, density=True, alpha=0.4, label="Data Histogram")
        plt.title("Optuna-CV Comparison: TF-CVXPY vs TF-CVXPY-PP")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_plot:
            import os
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(f"{plot_dir}/tf_cvxpy_optuna_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna CV comparison for TF estimators")
    parser.add_argument("--n-samples", type=int, default=800)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--metric", type=str, default="sll", choices=["sll", "bic"])
    parser.add_argument("--n-grid-points", type=int, default=200)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    sampler_setup = _make_sampler_setup()
    data = _sample_data(args.n_samples, sampler_setup)

    run_optuna_comparison(
        data=data,
        sampler_setup=sampler_setup,
        n_trials=args.n_trials,
        cv_folds=args.cv_folds,
        metric=args.metric,
        n_grid_points=args.n_grid_points,
        show_progress=not args.no_progress,
        save_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()

