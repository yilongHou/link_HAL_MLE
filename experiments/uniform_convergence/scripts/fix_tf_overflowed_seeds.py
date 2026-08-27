#!/usr/bin/env python3
"""
Fix TF (TrendFiltering*Estimator) seed outputs whose saved densities contain NaNs/Infs.

This happens when theta_hat becomes large and naive exp(theta) overflows during density
postprocessing. We keep the existing optimization result (theta_hat/bin_widths/knots)
and recompute a numerically-stable piecewise density on the stored gridpoints, then
overwrite:
  - results/.../seed_<seed>.json
  - plots/.../seed_<seed>_density.png

Default behavior targets uniform-convergence N=800 TF folders named:
  experiments/uniform_convergence/results/*_TrendFilteringADMMEstimator_N800

Even if the estimator name says ADMM, in this repo it historically aliases to CVXPY TF.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class SamplerSpec:
    sampler_name: str
    sampler_params: Dict[str, Any]
    n_samples: int


def _stable_piecewise_density_from_theta(
    *, gridpoints: np.ndarray, theta_hat: np.ndarray, bin_widths: np.ndarray, knots: np.ndarray
) -> np.ndarray:
    gridpoints = np.asarray(gridpoints, dtype=float).ravel()
    theta = np.asarray(theta_hat, dtype=float).ravel()
    dx = np.asarray(bin_widths, dtype=float).ravel()
    kts = np.asarray(knots, dtype=float).ravel()

    if theta.size != dx.size:
        raise ValueError(f"theta_hat/bin_widths length mismatch: {theta.size} vs {dx.size}")
    if dx.size != (kts.size - 1):
        raise ValueError(f"bin_widths/knots length mismatch: {dx.size} vs {kts.size}")

    log_terms = theta + np.log(dx)
    m = float(np.max(log_terms))
    logZ = m + float(np.log(np.sum(np.exp(log_terms - m))))
    log_levels = theta - logZ

    idx = np.searchsorted(kts, gridpoints, side="right") - 1
    idx = np.clip(idx, 0, theta.size - 1)
    fhat = np.exp(log_levels[idx])
    return fhat


def _load_sampler_spec(setup_json_path: str) -> SamplerSpec:
    with open(setup_json_path, "r") as f:
        setup = json.load(f)
    ss = setup["sampler_setup"]
    return SamplerSpec(
        sampler_name=str(ss["sampler"]),
        sampler_params=dict(ss["sampler_params"]),
        n_samples=int(ss["n_samples"]),
    )


def _get_sampler_class(sampler_name: str):
    # Import locally so script can be imported without pulling deps.
    from utils.sampler.truncated_gmm import TruncatedGMM
    from utils.sampler.truncated_normal import TruncatedNormal
    from utils.sampler.sinusoidal import Sinusoidal
    from utils.sampler.step_function import StepFunction

    samplers = {
        "TruncatedGMM": TruncatedGMM,
        "TruncatedNormal": TruncatedNormal,
        "Sinusoidal": Sinusoidal,
        "StepFunction": StepFunction,
    }
    if sampler_name not in samplers:
        raise ValueError(f"Unknown sampler {sampler_name!r}. Available: {sorted(samplers.keys())}")
    return samplers[sampler_name]


def _plot_density(
    *,
    out_png: str,
    seed: int,
    sampler,
    n_samples: int,
    gridpoints: np.ndarray,
    density: np.ndarray,
    title_prefix: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gridpoints = np.asarray(gridpoints, dtype=float).ravel()
    density = np.asarray(density, dtype=float).ravel()

    # Re-generate the training sample for the histogram (matches experiment flow).
    np.random.seed(int(seed))
    data = sampler.generate_samples(int(n_samples))
    true_density = sampler.compute_density(gridpoints)

    plt.figure(figsize=(12, 8))
    plt.plot(gridpoints, density, color="#007acc", linewidth=2, label="Estimated Density (fixed)")
    plt.plot(gridpoints, true_density, color="#d62728", linestyle="--", linewidth=2, label="True Density")
    plt.hist(data, bins=50, density=True, alpha=0.6, label="Data Histogram", color="#2ca02c")
    plt.title(f"{title_prefix} (Seed: {seed})", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def _maybe_fix_seed_file(
    *,
    seed_json_path: str,
    plot_png_path: str,
    sampler,
    n_samples: int,
    title_prefix: str,
    dry_run: bool,
) -> bool:
    with open(seed_json_path, "r") as f:
        d = json.load(f)

    est = d.get("estimated_density") or {}
    x = np.asarray(est.get("gridpoints", []), dtype=float)
    fhat_saved = np.asarray(est.get("density", []), dtype=float)
    if x.size == 0 or fhat_saved.size == 0:
        return False

    if np.all(np.isfinite(x)) and np.all(np.isfinite(fhat_saved)):
        return False

    hr = d.get("HAL_results", {}) if isinstance(d, dict) else {}
    theta_hat = hr.get("theta_hat")
    bin_widths = hr.get("bin_widths")
    knots = hr.get("knots")
    if theta_hat is None or bin_widths is None or knots is None:
        return False

    fhat_fixed = _stable_piecewise_density_from_theta(
        gridpoints=x, theta_hat=np.asarray(theta_hat), bin_widths=np.asarray(bin_widths), knots=np.asarray(knots)
    )
    if (not np.all(np.isfinite(fhat_fixed))) or (not np.all(np.isfinite(x))):
        return False

    # Overwrite JSON fields in-place
    d["estimated_density"]["density"] = fhat_fixed.tolist()
    # Keep HAL_results consistent too (downstream may read either)
    if isinstance(hr, dict):
        hr["grid_points"] = x.tolist()
        hr["estimated_density"] = fhat_fixed.tolist()
        d["HAL_results"] = hr

    if dry_run:
        return True

    with open(seed_json_path, "w") as f:
        json.dump(d, f, indent=2)

    _plot_density(
        out_png=plot_png_path,
        seed=int(re.search(r"seed_(\\d+)\\.json$", os.path.basename(seed_json_path)).group(1)),
        sampler=sampler,
        n_samples=n_samples,
        gridpoints=x,
        density=fhat_fixed,
        title_prefix=title_prefix,
    )
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="experiments/uniform_convergence",
        help="Root uniform_convergence directory (contains results/, plots/, setups/).",
    )
    ap.add_argument("--sample-size", type=int, default=800)
    ap.add_argument("--dry-run", action="store_true", help="Only report what would be fixed; do not overwrite files.")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    results_root = os.path.join(root, "results")
    plots_root = os.path.join(root, "plots")
    setups_root = os.path.join(root, "setups")

    pattern = f"_TrendFilteringADMMEstimator_N{int(args.sample_size)}"
    result_dirs = sorted(
        d for d in os.listdir(results_root) if d.endswith(pattern) and os.path.isdir(os.path.join(results_root, d))
    )

    total_fixed = 0
    total_seen = 0
    per_dir = {}

    for dname in result_dirs:
        setup_path = os.path.join(setups_root, f"{dname}.json")
        if not os.path.exists(setup_path):
            continue

        spec = _load_sampler_spec(setup_path)
        sampler_cls = _get_sampler_class(spec.sampler_name)
        sampler = sampler_cls(**spec.sampler_params)

        res_dir = os.path.join(results_root, dname)
        plot_dir = os.path.join(plots_root, dname)
        seed_files = sorted(
            fn
            for fn in os.listdir(res_dir)
            if fn.startswith("seed_") and fn.endswith(".json") and ("_recalc" not in fn)
        )

        fixed_here = 0
        for fn in seed_files:
            total_seen += 1
            seed_path = os.path.join(res_dir, fn)
            seed = int(fn.split("_")[1].split(".")[0])
            png_path = os.path.join(plot_dir, f"seed_{seed}_density.png")
            ok = _maybe_fix_seed_file(
                seed_json_path=seed_path,
                plot_png_path=png_path,
                sampler=sampler,
                n_samples=spec.n_samples,
                title_prefix=dname,
                dry_run=bool(args.dry_run),
            )
            if ok:
                fixed_here += 1
                total_fixed += 1

        if fixed_here:
            per_dir[dname] = fixed_here

    print(f"Scanned {len(result_dirs)} TF result folders under {results_root}")
    print(f"Checked seed files: {total_seen}")
    print(f\"Fixed (overwrote) seed files: {total_fixed}{' (dry-run)' if args.dry_run else ''}\")
    if per_dir:
        print(\"Per-folder fixes:\")
        for k, v in sorted(per_dir.items()):
            print(f\"  - {k}: {v}\")


if __name__ == \"__main__\":
    main()


