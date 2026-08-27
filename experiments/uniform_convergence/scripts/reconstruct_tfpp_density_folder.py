#!/usr/bin/env python3
"""
Batch postprocess TFPP result JSONs to reconstruct a continuous density for k=1,2.

Why
---
The TFPP implementation stores a binned, piecewise-constant density (bin-lookup on theta).
For k>=1, we can postprocess the fitted theta values on the knot grid into a continuous
piecewise-polynomial log-density and evaluate it pointwise on the plotting grid, similar
to how HAL-MLE evaluates a basis expansion.

What this script does
---------------------
- Reads each `seed_*.json` under an input results directory
- Uses stored `theta_hat` + `knots` (no refit) to build a continuous order-k log-density
  via the **falling-factorial basis** (the natural TF basis), using alpha = H^{-1} theta
- Normalizes the resulting density by numerical integration on the same plotting grid
- Writes a *small* reconstructed JSON to an output directory (does NOT duplicate Hinv)
- Saves a per-seed comparison plot (original vs reconstructed) to an output plots directory

Example
-------
uv run python experiments/uniform_convergence/scripts/reconstruct_tfpp_density_folder.py \
  --in-dir  experiments/uniform_convergence/results/TruncatedNormal_TrendFilteringCVXPYPP_N800 \
  --out-dir experiments/uniform_convergence/results/TruncatedNormal_TrendFilteringCVXPYPP_N800_recon \
  --out-plots experiments/uniform_convergence/plots/TruncatedNormal_TrendFilteringCVXPYPP_N800_recon
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import numpy as np

# Ensure project root on path (this script is under experiments/uniform_convergence/scripts/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from methods.non_HAL_method.TF_CVXPY_PP.estimator import (  # noqa: E402
    build_extended_operator_ryan,
)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _normalize_density(x: np.ndarray, log_density: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    log_density = np.asarray(log_density, dtype=float).ravel()
    f = np.exp(log_density - np.max(log_density))  # stabilize
    Z = float(np.trapz(f, x))
    if not np.isfinite(Z) or Z <= 0:
        raise RuntimeError(f"Non-finite or non-positive normalization constant Z={Z}")
    return f / Z


def _falling_factorial_basis_matrix(
    *, x_eval: np.ndarray, x_nodes: np.ndarray, k: int
) -> np.ndarray:
    """
    Falling-factorial basis matrix H(x_eval) per Tibshirani (2014) Lemma 4.

    Given inputs x_1 < ... < x_n, the (generalized) falling-factorial basis is most
    naturally expressed in terms of *anchored products* (not raw monomials) on irregular
    grids, and matches the repo's `Hinv` convention (built via Ryan/Tibshirani recursion):

      h_1(x)=1
      h_2(x)=(x-x_1)
      h_3(x)=(x-x_1)(x-x_2)
      ...
      h_{k+1}(x)=∏_{ℓ=1}^k (x-x_ℓ)

      h_{k+1+j}(x)=k! · ∏_{ℓ=1}^k (x - x_{j+ℓ}) · 1{x >= x_{j+k}},
      for j=1,...,n-k-1.

    Then H_{ij} = h_j(x_i). We implement this directly for k in {0,1,2}.
    """
    x_eval = np.asarray(x_eval, dtype=float).ravel()
    x_nodes = np.asarray(x_nodes, dtype=float).ravel()
    n = int(x_nodes.size)
    if n < 2:
        raise ValueError("Need at least 2 nodes for falling-factorial basis.")
    if not np.all(np.diff(x_nodes) > 0):
        raise ValueError("x_nodes must be strictly increasing for falling-factorial basis.")
    if k < 0 or k > 2:
        raise ValueError(f"Expected k in {{0,1,2}}, got k={k}")
    if k > n - 2:
        raise ValueError(f"k must satisfy k <= n-2, got k={k} with n={n}")

    H = np.zeros((x_eval.size, n), dtype=float)
    H[:, 0] = 1.0

    if k >= 1:
        H[:, 1] = x_eval - x_nodes[0]
    if k >= 2:
        H[:, 2] = (x_eval - x_nodes[0]) * (x_eval - x_nodes[1])

    if k == 0:
        # h_{1+j}(x) = 1{x >= x_j}, j=1,...,n-1
        for j in range(1, n):
            H[:, j] = (x_eval >= x_nodes[j - 1]).astype(float)
        return H

    # h_{k+1+j}, j=1..n-k-1
    # 0-based column index: (k+1+j)-1 = k+j
    from math import factorial

    scale = float(factorial(int(k)))
    for j in range(1, n - k):
        col = k + j
        prod = np.ones_like(x_eval, dtype=float)
        for ell in range(1, k + 1):
            prod *= (x_eval - x_nodes[j + ell - 1])
        prod *= (x_eval >= x_nodes[j + k - 1]).astype(float)
        H[:, col] = scale * prod

    return H


def _log_density_via_falling_factorial(
    *, x_eval: np.ndarray, x_nodes: np.ndarray, alpha: np.ndarray, k: int
) -> np.ndarray:
    H_eval = _falling_factorial_basis_matrix(x_eval=x_eval, x_nodes=x_nodes, k=int(k))
    alpha = np.asarray(alpha, dtype=float).ravel()
    if H_eval.shape[1] != alpha.size:
        raise RuntimeError(
            f"Basis/alpha mismatch: H_eval.shape={H_eval.shape} alpha.size={alpha.size}"
        )
    return H_eval @ alpha


def _extract_seed(path: str) -> int:
    base = os.path.basename(path)
    if not base.startswith("seed_") or not base.endswith(".json"):
        raise ValueError(f"Unexpected seed filename: {base}")
    return int(base.replace("seed_", "").replace(".json", ""))


def _reconstruct_one(src: Dict[str, Any], *, seed: int) -> Dict[str, Any]:
    k = int(src["hyperparams"]["k"])
    norm_constraint = float(src["hyperparams"]["norm_constraint"])

    gridpoints = np.asarray(src["estimated_density"]["gridpoints"], dtype=float).ravel()
    density_orig = np.asarray(src["estimated_density"]["density"], dtype=float).ravel()

    # Stored TFPP fit is under "HAL_results" in these experiment JSONs
    theta = np.asarray(src["HAL_results"]["theta_hat"], dtype=float).ravel()
    knots = np.asarray(src["HAL_results"]["knots"], dtype=float).ravel()

    # This estimator builds Hinv on the *left endpoints* x_positions = knots[:-1].
    # theta_hat has length n_bins = len(knots)-1, aligned with these positions.
    t_nodes = knots[:-1]
    if len(t_nodes) != len(theta):
        raise RuntimeError(
            f"len(knots)={len(knots)} implies len(knots[:-1])={len(t_nodes)} but len(theta)={len(theta)}"
        )

    out: Dict[str, Any] = {
        "seed": int(seed),
        "hyperparams": {"k": int(k), "norm_constraint": float(norm_constraint)},
        "node_grid": t_nodes.tolist(),
        "theta_hat": theta.tolist(),
        "knots": knots.tolist(),
        "estimated_density_orig": {
            "gridpoints": gridpoints.tolist(),
            "density": density_orig.tolist(),
        },
    }

    if k == 0:
        # Already aligned with the cumulative-indicator basis (piecewise constant)
        out["estimated_density_reconstructed"] = {
            "gridpoints": gridpoints.tolist(),
            "density": density_orig.tolist(),
        }
        out["reconstruction"] = {
            "method": "noop_k0",
            "note": "k=0 is already piecewise-constant; reconstruction is identical to original.",
        }
        return out

    if k not in (1, 2):
        raise ValueError(f"This postprocessor is intended for k in {{0,1,2}}, got k={k}")

    # alpha = H^{-1} theta where H is the falling-factorial basis matrix on the node grid.
    # Prefer the stored Hinv (for auditability), but fall back to building via Ryan operator if absent.
    # Prefer the stored Hinv (for auditability), but fall back to building E if absent.
    alpha: np.ndarray
    if "Hinv" in src.get("HAL_results", {}):
        Hinv = np.asarray(src["HAL_results"]["Hinv"], dtype=float)
        alpha = Hinv @ theta
        alpha_source = "HAL_results.Hinv @ theta_hat"
    else:
        E = build_extended_operator_ryan(t_nodes, k=int(k), drop_intercept=False)
        tmp = E @ theta
        toarray_fn = getattr(tmp, "toarray", None)
        if callable(toarray_fn):
            alpha = np.asarray(toarray_fn()).ravel()
        else:
            alpha = np.asarray(tmp).ravel()
        alpha_source = "build_extended_operator_ryan(node_grid, drop_intercept=False) @ theta_hat"

    g_eval = _log_density_via_falling_factorial(
        x_eval=gridpoints, x_nodes=t_nodes, alpha=alpha, k=int(k)
    )
    density_recon = _normalize_density(gridpoints, g_eval)

    if density_recon.shape != density_orig.shape:
        raise RuntimeError("Original and reconstructed densities have different grid shapes.")
    l1 = float(np.trapz(np.abs(density_recon - density_orig), gridpoints))
    sup = float(np.max(np.abs(density_recon - density_orig)))

    out["reconstruction"] = {
        "method": "falling_factorial_basis",
        "alpha_source": alpha_source,
        "alpha": alpha.tolist(),
        "diff_vs_piecewise": {"l1": l1, "sup": sup},
    }
    out["estimated_density_reconstructed"] = {
        "gridpoints": gridpoints.tolist(),
        "density": density_recon.tolist(),
    }
    return out


def _plot_compare(
    *,
    seed: int,
    k: int,
    gridpoints: np.ndarray,
    density_orig: np.ndarray,
    density_recon: np.ndarray,
    out_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.figure(figsize=(8.5, 4.5), dpi=160)
    plt.plot(gridpoints, density_orig, lw=1.6, label="original (piecewise-constant)")
    plt.plot(gridpoints, density_recon, lw=1.6, label="reconstructed (continuous)")
    plt.title(f"TFPP density: original vs reconstructed (k={k}, seed={seed})")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--out-plots", type=str, required=True)
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process the first N seeds (sorted).")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)
    out_plots = os.path.abspath(args.out_plots)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_plots, exist_ok=True)

    files = sorted(
        [
            os.path.join(in_dir, f)
            for f in os.listdir(in_dir)
            if f.startswith("seed_") and f.endswith(".json")
        ]
    )
    if args.limit and args.limit > 0:
        files = files[: int(args.limit)]

    if not files:
        raise RuntimeError(f"No seed JSONs found under {in_dir}")

    for path in files:
        seed = _extract_seed(path)
        src = _load_json(path)
        recon = _reconstruct_one(src, seed=seed)

        out_json = os.path.join(out_dir, f"seed_{seed}.json")
        _save_json(out_json, recon)

        k = int(recon["hyperparams"]["k"])
        gridpoints = np.asarray(recon["estimated_density_orig"]["gridpoints"], dtype=float)
        density_orig = np.asarray(recon["estimated_density_orig"]["density"], dtype=float)
        density_recon = np.asarray(recon["estimated_density_reconstructed"]["density"], dtype=float)
        out_png = os.path.join(out_plots, f"seed_{seed}_compare_recon.png")
        _plot_compare(
            seed=seed,
            k=k,
            gridpoints=gridpoints,
            density_orig=density_orig,
            density_recon=density_recon,
            out_path=out_png,
        )


if __name__ == "__main__":
    main()


