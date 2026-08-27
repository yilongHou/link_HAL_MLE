"""
Visualize loss per FLOP across optimization algorithms.

This script reads per-iteration logs for several optimization algorithms and
plots the objective function loss versus cumulative FLOPs for each basis order.
It overlays a horizontal reference line for the final loss achieved
by the CVXPY estimator.

The FLOP calculations are based on the analysis in algorithm_flop_per_iter.md
and are implementation-aware, accounting for matrix operations, line search,
coordinate descent, and other algorithm-specific costs.

Example usage:
    uv run experiments/compare_knot_selection/visualize_loss_per_flop.py \
        --dgp all \
        --figure-dir paper/resources/optimization_algorithms/per_flop
"""

import argparse
import json
import logging
import os
import re
import sys
import math
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# Add project root to Python path (match style of other scripts in this repo)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.basis import create_basis_functions

with open('dpg-name-mapping.json', 'r') as f:
    DGP_NAME_MAPPING = json.load(f)

###############################################################################
# Configuration
###############################################################################

# Supported DGPs and their folder suffix under experiments/compare_knot_selection
SUPPORTED_DGPS: List[str] = [
    "TruncatedNormal",
    "TruncatedGMMSymmetricThree",
    "TruncatedGMMAsymmetricThree",
    "TruncatedGMMFiveSpikes",
    "StepFunction",
    "Sinusoidal",
]

# Algorithms to visualize (display name, identifier used in filenames, color, linestyle)
ALGORITHMS: List[Tuple[str, str, str, str]] = [
    ("FISTA", "FISTAEstimator", "darkorange", '--'),
    ("ProximalAdaGrad", "ProximalAdaGradEstimator", "mediumblue", '-'),
    ("ProximalNewton", "ProximalNewtonEstimator", "darkgreen", '-.'),
    ("ProximalNewtonLBFGS", "ProximalNewtonLBFGSEstimator", "crimson", ':'),
]

# Basis orders to plot
BASIS_ORDERS: List[int] = [0, 1, 2]

# Candidate column names in logs representing the objective function loss
LOSS_CANDIDATES: List[str] = [
    "obj",
    "objective",
    "loss",
    "neg_log_likelihood",
    "nll",
    "f",
    "f_val",
    "objective_value",
    "cost",
]


###############################################################################
# FLOP Calculation Functions (same as per_flop knot script)
###############################################################################

def calculate_fista_flops(n_samples: int, k_params: int, g_grid: int) -> float:
    """
    Calculate FLOPs for one FISTA iteration.
    Formula: 4·N·K + 6·G·K from algorithm_flop_per_iter.md
    """
    return 4.0 * n_samples * k_params + 6.0 * g_grid * k_params


def calculate_adagrad_flops(n_samples: int, k_params: int, g_grid: int) -> float:
    """
    Calculate FLOPs for one ProximalAdaGrad iteration.
    Formula: 2·N·K + 8·G·K (tight bound) from algorithm_flop_per_iter.md
    """
    return 2.0 * n_samples * k_params + 8.0 * g_grid * k_params


def calculate_newton_flops(n_samples: int, k_params: int, g_grid: int, 
                          s_sweeps: int, l_k: int) -> float:
    """
    Calculate FLOPs for one ProximalNewton iteration.
    Formula: [2·N·K + 2·G·K] (grad) + [2·G·K²] (Hessian) + [2·s·K²] (CD) + 
             L_k·[2·N·K + 2·G·K] + [2·G·K] (logZ)
    """
    grad_flops = 2.0 * n_samples * k_params + 2.0 * g_grid * k_params
    hessian_flops = 2.0 * g_grid * k_params * k_params
    cd_flops = 2.0 * s_sweeps * k_params * k_params
    line_search_flops = l_k * (2.0 * n_samples * k_params + 2.0 * g_grid * k_params)
    logz_flops = 2.0 * g_grid * k_params
    
    return grad_flops + hessian_flops + cd_flops + line_search_flops + logz_flops


def calculate_lbfgs_flops(n_samples: int, k_params: int, g_grid: int, l_k: int) -> float:
    """
    Calculate FLOPs for one ProximalNewtonLBFGS iteration.
    Formula: (1+L_k)·(2·N·K + 2·G·K) + 2·G·K from algorithm_flop_per_iter.md
    """
    base_flops = 2.0 * n_samples * k_params + 2.0 * g_grid * k_params
    return (1.0 + l_k) * base_flops + 2.0 * g_grid * k_params


def infer_line_search_steps(alpha: float, beta: float = 0.5) -> int:
    """
    Infer number of line search objective evaluations from final alpha.
    Formula: L_k = 1 + max(0, ceil(log(alpha) / log(beta))) when alpha starts at 1
    """
    # Guard invalid or missing alpha values
    if alpha is None or not np.isfinite(alpha):
        return 1
    # Clamp to sensible range (0, 1]; if <=0, fall back to 1 to avoid log domain errors
    if alpha >= 1.0:
        return 1
    if alpha <= 0.0:
        return 1
    # Compute number of backtracking steps
    denom = math.log(beta) if beta not in (0.0, 1.0) else math.log(0.5)
    backtracks = math.ceil(math.log(alpha) / denom)
    return 1 + max(0, backtracks)


###############################################################################
# Utilities (adapted from iteration-based script)
###############################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize loss per FLOP across optimization algorithms."
    )
    parser.add_argument(
        "--dgp",
        type=str,
        default="all",
        help=(
            "Which DGP to visualize. Use 'all' for all supported DGPs or provide "
            "a specific name (e.g., 'TruncatedNormal')."
        ),
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default="paper/resources/optimization_algorithms/per_flop",
        required=False,
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--save-legend",
        action="store_true",
        default=False,
        help="Save a standalone legend figure (no axes) to the figure directory.",
    )
    parser.add_argument(
        "--legend-filename",
        type=str,
        default="legend_loss_algorithms.png",
        help="Filename for the standalone legend image.",
    )
    parser.add_argument(
        "--legend-ncol",
        type=int,
        default=5,
        help="Number of columns in the legend layout.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_dgp_list(dgp_arg: str) -> List[str]:
    if dgp_arg.lower() == "all":
        return SUPPORTED_DGPS
    if dgp_arg not in SUPPORTED_DGPS:
        raise ValueError(
            f"Unsupported DGP '{dgp_arg}'. Supported values: {SUPPORTED_DGPS} or 'all'."
        )
    return [dgp_arg]


def parse_log_file(log_fpath: str) -> pd.DataFrame:
    """
    Parse an estimator log file to extract per-iteration metrics.
    Same as iteration-based script.
    """
    if not os.path.exists(log_fpath):
        logging.warning(f"Log file not found: {log_fpath}")
        return pd.DataFrame()

    log_entries: List[Dict[str, float]] = []
    log_pattern = re.compile(r"Iter\s+(\d+):\s+(.*)")
    kv_pattern = re.compile(r"([\w|‖\[\]:]+)=(-?[\d.]+(?:[eE][+-]?\d+)?)")

    with open(log_fpath, "r") as f:
        for line in f:
            match = log_pattern.search(line)
            if not match:
                continue
            iteration = int(match.group(1))
            metrics_str = match.group(2)

            entry: Dict[str, float] = {"iteration": float(iteration)}
            kv_pairs = kv_pattern.findall(metrics_str)
            for key, value in kv_pairs:
                clean_key = key.replace("‖", "").replace("‖₁", "_l1").strip()
                try:
                    entry[clean_key] = float(value)
                except ValueError:
                    continue
            log_entries.append(entry)

    if not log_entries:
        return pd.DataFrame()
    return pd.DataFrame(log_entries)


def parse_initialization_params(log_fpath: str, algo_id: str) -> Dict[str, Optional[int]]:
    """
    Parse initialization parameters (N, K, G) from the log file.
    
    Returns dictionary with keys: n_samples, k_params, g_grid, cd_sweeps
    """
    params = {
        "n_samples": None,
        "k_params": None, 
        "g_grid": None,
        "cd_sweeps": 2  # default from ProximalNewton
    }
    
    if not os.path.exists(log_fpath):
        return params
    
    # Patterns to match initialization log lines
    # Accept decimals and scientific notation for lam and other floats
    float_pat = r"[\deE+\-.]+"
    init_patterns = {
        "FISTAEstimator": re.compile(rf"FISTA: lam={float_pat}, L={float_pat}, n_grid=(\d+), basis_order=\d+, n_samples=(\d+), K=(\d+)"),
        "ProximalAdaGradEstimator": re.compile(rf"ProximalAdaGrad: lam={float_pat}, alpha={float_pat}, eps={float_pat}, rho={float_pat}, n_grid=(\d+), basis_order=\d+, n_samples=(\d+), K=(\d+)"),
        "ProximalNewtonEstimator": re.compile(rf"ProximalNewton: lam={float_pat}, n_grid=(\d+), cd_sweeps=(\d+), basis_order=\d+, n_samples=(\d+), K=(\d+)"),
        "ProximalNewtonLBFGSEstimator": re.compile(rf"ProximalNewtonLBFGS: lam={float_pat}, memory=\d+, n_grid=(\d+), basis_order=\d+, n_samples=(\d+), K=(\d+)"),
    }
    
    pattern = init_patterns.get(algo_id)
    if not pattern:
        return params
    
    with open(log_fpath, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                if algo_id == "FISTAEstimator":
                    params["g_grid"] = int(match.group(1)) - 1  # midpoints = n_grid_points - 1
                    params["n_samples"] = int(match.group(2))
                    params["k_params"] = int(match.group(3))
                elif algo_id == "ProximalNewtonEstimator":
                    params["g_grid"] = int(match.group(1)) - 1
                    params["cd_sweeps"] = int(match.group(2))
                    params["n_samples"] = int(match.group(3))
                    params["k_params"] = int(match.group(4))
                else:  # AdaGrad, LBFGS
                    params["g_grid"] = int(match.group(1)) - 1
                    params["n_samples"] = int(match.group(2))
                    params["k_params"] = int(match.group(3))
                break

    # Generic fallback: scan for 'K=#####' anywhere in the log
    if params["k_params"] is None and os.path.exists(log_fpath):
        try:
            with open(log_fpath, "r") as f:
                content = f.read()
            m = re.search(r"K=(\d+)", content)
            if m:
                params["k_params"] = int(m.group(1))
        except Exception:
            pass
    
    return params


def fallback_params_from_results(dgp: str, algo_id: str, order: int) -> Dict[str, Optional[int]]:
    """
    Fallback: read results JSON to infer n_samples, k_params, g_grid, cd_sweeps.
    """
    paths = build_paths_for_dgp(dgp)
    results_dir = paths["results"]
    result_fpath = os.path.join(results_dir, f"{dgp}_{algo_id}_Order{order}.json")
    params = {
        "n_samples": None,
        "k_params": None,
        "g_grid": None,
        "cd_sweeps": 2,
    }
    if not os.path.exists(result_fpath):
        return params
    try:
        with open(result_fpath, "r") as f:
            payload = json.load(f)
        # n_samples from sampler_setup
        n_samples = payload.get("sampler_setup", {}).get("n_samples")
        # G from estimator_setup.estimator_params.n_grid_points
        n_grid_points = payload.get("estimator_setup", {}).get("estimator_params", {}).get("n_grid_points")
        # K from results.theta_hat length
        theta_hat = payload.get("results", {}).get("theta_hat")
        basis_order = payload.get("estimator_setup", {}).get("estimator_params", {}).get("basis_order")
        cd_sweeps = payload.get("estimator_setup", {}).get("estimator_params", {}).get("cd_sweeps", 2)
        if isinstance(n_samples, int):
            params["n_samples"] = n_samples
        if isinstance(n_grid_points, int):
            params["g_grid"] = max(0, n_grid_points - 1)
        if isinstance(cd_sweeps, int):
            params["cd_sweeps"] = cd_sweeps
        if isinstance(theta_hat, list):
            params["k_params"] = int(len(theta_hat))
        elif isinstance(basis_order, int) and params["g_grid"] is not None:
            # Fallback: approximate K = (basis_order+1) + M, M≈#knots≈n_samples (over-approx)
            params["k_params"] = (basis_order + 1) + params["g_grid"]
    except Exception:
        pass
    return params


def detect_loss_column(df: pd.DataFrame) -> Optional[str]:
    """Same as iteration-based script."""
    if df is None or df.empty:
        return None
    available = set(df.columns)
    for candidate in LOSS_CANDIDATES:
        if candidate in available:
            return candidate
    # Heuristic: sometimes columns have slight naming variants
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in LOSS_CANDIDATES:
        for lc, orig in lower_map.items():
            if candidate.lower() in lc:
                return orig
    return None


def read_cvxpy_final_loss(results_fpath: str, prefer_penalized: bool = True) -> Optional[float]:
    """Recover CVXPY final loss from the results JSON.

    Fallback strategy when explicit loss is not stored:
    - Compute negative log-likelihood (NLL):
        NLL(θ) = -∑_i (φ(x_i)^T θ) + N * log(∑_j exp(φ(m_j)^T θ) * Δ_j)
      where m_j are midpoints saved as results["grid_points"], and Δ_j are interval
      widths reconstructed from those midpoints with edges at 0 and 1.
    - If prefer_penalized and results include lambda_val_lag, add λ*‖θ[1:]‖₁.
    """
    if not os.path.exists(results_fpath):
        logging.warning(f"CVXPY result file not found: {results_fpath}")
        return None
    try:
        with open(results_fpath, "r") as f:
            payload = json.load(f)

        # If result already has an explicit loss, use it
        if isinstance(payload, dict):
            results = payload.get("results", {})
            if isinstance(results, dict):
                for loss_key in ["final_loss", "objective", "loss", "neg_log_likelihood"]:
                    loss_val = results.get(loss_key)
                    if loss_val is not None:
                        return float(loss_val)

        # Otherwise, reconstruct from stored fields
        estimator_params = (
            payload.get("estimator_setup", {})
            .get("estimator_params", {})
            if isinstance(payload, dict)
            else {}
        )
        basis_order = int(estimator_params.get("basis_order", 0))

        results = payload.get("results", {}) if isinstance(payload, dict) else {}
        theta_hat = results.get("theta_hat")
        data_points = results.get("data_points")
        midpoints = results.get("grid_points")
        lambda_val = results.get("lambda_val_lag")
        n_samples = (
            int(payload.get("sampler_setup", {}).get("n_samples"))
            if isinstance(payload, dict)
            else None
        )

        if theta_hat is None or data_points is None or midpoints is None:
            logging.warning(
                f"Insufficient fields to reconstruct CVXPY loss in {results_fpath}"
            )
            return None

        theta = np.asarray(theta_hat, dtype=float)
        grid_points_hal = np.asarray(data_points, dtype=float)
        mid = np.asarray(midpoints, dtype=float)
        # Ensure sorted unique midpoints
        mid = np.sort(np.unique(mid))

        # Reconstruct edges and Δ_j from midpoints with boundaries [0,1]
        if mid.size < 1:
            logging.warning(f"Empty midpoints in {results_fpath}")
            return None
        edges = np.empty(mid.size + 1, dtype=float)
        edges[0] = 0.0
        edges[-1] = 1.0
        if mid.size > 1:
            edges[1:-1] = 0.5 * (mid[:-1] + mid[1:])
        else:
            edges[1:-1] = []  # no interior edges
        delta_j = edges[1:] - edges[:-1]

        # Basis at data points and midpoints
        df_data = pd.DataFrame({"W1": grid_points_hal})
        phi_data, _ = create_basis_functions(df_data, grid_points_hal, order=basis_order)
        phi_data_np = phi_data.numpy()

        df_mid = pd.DataFrame({"W1": mid})
        phi_mid, _ = create_basis_functions(df_mid, grid_points_hal, order=basis_order)
        phi_mid_np = phi_mid.numpy()

        # Compute stable logZ
        log_f_mid = phi_mid_np @ theta
        max_log = float(np.max(log_f_mid))
        z_val = np.sum(np.exp(log_f_mid - max_log) * delta_j)
        if not np.isfinite(z_val) or z_val <= 0.0:
            logging.warning(f"Invalid Z while reconstructing loss for {results_fpath}")
            return None
        logZ = max_log + np.log(z_val)

        # First term and N
        log_f_data = phi_data_np @ theta
        N = int(n_samples) if n_samples is not None else int(len(grid_points_hal))
        nll = -float(np.sum(log_f_data)) + float(N) * float(logZ)

        if prefer_penalized and lambda_val is not None:
            l1_norm = float(np.sum(np.abs(theta[1:])))
            return nll + float(lambda_val) * l1_norm
        return nll
    except Exception as e:
        logging.error(f"Failed to reconstruct CVXPY loss from {results_fpath}: {e}")
        return None


def build_paths_for_dgp(dgp: str) -> Dict[str, str]:
    """Same as iteration-based script."""
    base = os.path.join(
        project_root, "experiments", "compare_knot_selection", f"single_{dgp}"
    )
    return {
        "base": base,
        "logs": os.path.join(base, "logs"),
        "results": os.path.join(base, "results"),
    }


def compute_cumulative_flops(df: pd.DataFrame, algo_id: str, params: Dict[str, Optional[int]]) -> np.ndarray:
    """
    Compute cumulative FLOPs for each iteration based on algorithm and parameters.
    """
    if df.empty or any(v is None for v in [params["n_samples"], params["k_params"], params["g_grid"]]):
        return np.array([])
    
    # Extract typed params (after early return, they should be non-None)
    n_samples_v = params.get("n_samples")
    k_params_v = params.get("k_params")
    g_grid_v = params.get("g_grid")
    cd_sweeps_v = params.get("cd_sweeps", 2)

    # Static assurance for type checkers
    if n_samples_v is None or k_params_v is None or g_grid_v is None:
        return np.array([])
    n_samples = int(n_samples_v)
    k_params = int(k_params_v)
    g_grid = int(g_grid_v)
    cd_sweeps = int(cd_sweeps_v if cd_sweeps_v is not None else 2)
    
    flops_per_iter = []
    
    for _, row in df.iterrows():
        # Calculate FLOPs based on algorithm
        if algo_id == "FISTAEstimator":
            l_k = 1  # no line search in FISTA
            flops = calculate_fista_flops(n_samples, k_params, g_grid)
        elif algo_id == "ProximalAdaGradEstimator":
            l_k = 1  # no line search logged/used
            flops = calculate_adagrad_flops(n_samples, k_params, g_grid)
        elif algo_id == "ProximalNewtonEstimator":
            alpha_val = row.get("α", row.get("alpha", 1.0))
            try:
                alpha_num = float(alpha_val)  # type: ignore[arg-type]
            except Exception:
                alpha_num = 1.0
            l_k = infer_line_search_steps(alpha_num)
            flops = calculate_newton_flops(n_samples, k_params, g_grid, cd_sweeps, int(l_k))
        elif algo_id == "ProximalNewtonLBFGSEstimator":
            alpha_val = row.get("α", row.get("alpha", 1.0))
            try:
                alpha_num = float(alpha_val)  # type: ignore[arg-type]
            except Exception:
                alpha_num = 1.0
            l_k = infer_line_search_steps(alpha_num)
            flops = calculate_lbfgs_flops(n_samples, k_params, g_grid, int(l_k))
        else:
            flops = 0.0
        
        flops_per_iter.append(flops)
    
    # Return cumulative sum
    return np.cumsum(flops_per_iter)


def plot_for_dgp_and_order(
    dgp: str,
    order: int,
    figure_dir: str,
    cvxpy_loss: Optional[float],
    logs_dir: str,
) -> Optional[str]:
    """
    Generate a FLOP-based figure for a specific DGP and basis order.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    plotted_any = False
    x_max = 0.0
    y_max = -np.inf
    y_min = np.inf

    for display_name, algo_id, color, linestyle in ALGORITHMS:
        log_name = f"{dgp}_{algo_id}_Order{order}.log"
        log_fpath = os.path.join(logs_dir, log_name)
        df = parse_log_file(log_fpath)
        if df.empty:
            logging.info(f"No log data for {display_name} ({log_name}); skipping this curve.")
            continue

        # Parse initialization parameters (with fallback to results JSON)
        params = parse_initialization_params(log_fpath, algo_id)
        if any(v is None for v in [params["n_samples"], params["k_params"], params["g_grid"]]):
            fb = fallback_params_from_results(dgp, algo_id, order)
            # fill any missing
            for key in params:
                if params[key] is None and fb.get(key) is not None:
                    params[key] = fb[key]
            if any(v is None for v in [params["n_samples"], params["k_params"], params["g_grid"]]):
                logging.warning(f"Could not parse initialization parameters for {log_name}")
                continue

        loss_col = detect_loss_column(df)
        if not loss_col:
            logging.warning(
                f"No loss column detected in {log_name}. Available columns: {list(df.columns)}"
            )
            continue

        # Sort by iteration and compute cumulative FLOPs
        df = df.sort_values("iteration").copy()
        cumulative_flops = compute_cumulative_flops(df, algo_id, params)
        
        if len(cumulative_flops) == 0:
            logging.info(f"Could not compute FLOPs for {display_name} ({log_name}); skipping.")
            continue

        # Add starting point at small positive FLOPs with loss=0 (before optimization starts)
        flops_vals = cumulative_flops.astype(float)
        loss_vals = df[loss_col].astype(float).to_numpy()
        flops_with_start = np.concatenate([np.array([1.0], dtype=float), flops_vals])
        loss_with_start = np.concatenate([np.array([0.0], dtype=float), loss_vals])
        
        # Filter out any NaN values
        valid_mask = ~np.isnan(flops_with_start) & ~np.isnan(loss_with_start) & np.isfinite(loss_with_start)
        flops_plot = flops_with_start[valid_mask]
        loss_plot = loss_with_start[valid_mask]
        
        if len(flops_plot) == 0:
            continue

        ax.plot(
            flops_plot,
            loss_plot,
            label=f"{display_name} (final={loss_plot[-1]:.2e})",
            color=color,
            linewidth=1.0,
            linestyle=linestyle,
        )
        plotted_any = True
        x_max = max(x_max, float(flops_plot.max()))
        try:
            y_max = max(y_max, float(loss_plot.max()))
            y_min = min(y_min, float(loss_plot.min()))
        except Exception:
            pass

    if not plotted_any:
        plt.close(fig)
        logging.info(f"No curves plotted for {dgp}, order {order}; skipping figure.")
        return None

    # Reference line from CVXPY final loss
    if cvxpy_loss is not None and np.isfinite(cvxpy_loss):
        ax.axhline(
            y=cvxpy_loss,
            color="k",
            linestyle="-",
            linewidth=.75,
            label=f"CVXPY (final={cvxpy_loss:.2e})",            
        )
        y_max = max(y_max, cvxpy_loss)
        y_min = min(y_min, cvxpy_loss)

    # Formatting
    order_str = {
        0: "0th Order",
        1: "1st Order", 
        2: "2nd Order",
    }

    ax.set_title(f"{DGP_NAME_MAPPING[dgp]} ({order_str[order]})", fontsize=8)
    ax.set_xlabel("Cumulative FLOPs", fontsize=7)
    
    # Determine appropriate scaling for y-axis
    y_scale_factor = 1.0
    y_scale_label = ""
    if y_max >= 1e6:
        y_scale_factor = 1e6
        y_scale_label = " (×10⁶)"
    elif y_max >= 1e3:
        y_scale_factor = 1e3
        y_scale_label = " (×10³)"
    
    ax.set_ylabel(f"Loss{y_scale_label}", fontsize=7)
    
    # Format x-axis ticks for FLOPs (scientific notation for large numbers)
    xticks = ax.get_xticks()
    ax.set_xticks(ticks=xticks, labels=[f"{tick:.1e}" if tick >= 1e6 else f"{tick:.0f}" for tick in xticks], fontsize=6)
    
    # Scale y-ticks and format as rounded integers
    yticks = ax.get_yticks()
    scaled_yticks = yticks / y_scale_factor
    ax.set_yticks(ticks=yticks, labels=[f"{tick:.0f}" for tick in scaled_yticks], fontsize=7)
    
    ax.set_xscale("log")
    if y_min < y_max:
        margin = (y_max - y_min) * 0.05
        ax.set_ylim((y_min - margin, y_max + margin))
    ax.set_xlim(left=1.0, right=x_max * 1.05)
    
    ensure_dir(figure_dir)
    fname = f"{dgp}_order_{order}_loss_per_flop.png"
    fpath = os.path.join(figure_dir, fname)
    plt.grid(True, linestyle='--', alpha=0.25)
    plt.tight_layout()
    plt.savefig(fpath, dpi=300)
    plt.close(fig)
    logging.info(f"Saved figure: {fpath}")
    return fpath


def save_algorithm_legend(figure_dir: str, filename: str, ncol: int = 4, fontsize: int = 8) -> str:
    """Save a standalone legend image using ALGORITHMS styles.

    The legend uses the display names, colors, and linestyles defined in ALGORITHMS
    and renders without axes for inclusion in LaTeX as a shared legend.
    """
    ensure_dir(figure_dir)

    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            label=display_name,
        )
        for (display_name, _algo_id, color, linestyle) in ALGORITHMS
    ]

    # Add CVXPY reference style used in plots (black solid line)
    handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=1.5,
            label="CVXPY",
        )
    )

    # One-row legend if possible; make width flexible and keep height compact
    # Figsize width scaled by number of columns; height tuned for tight layout
    width_in = max(3.6, 1.2 * float(max(1, ncol)))
    fig = plt.figure(figsize=(width_in, 0.5))
    fig.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc="center",
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        handlelength=2.5,
        columnspacing=1.5,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    # No axes; export transparent background to blend in paper
    out_path = os.path.join(figure_dir, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    logging.info(f"Saved legend figure: {out_path}")
    return out_path


def run_for_dgp(dgp: str, figure_dir: str) -> List[str]:
    """Same structure as iteration-based script."""
    paths = build_paths_for_dgp(dgp)
    results_dir = paths["results"]
    logs_dir = paths["logs"]

    saved_paths: List[str] = []
    for order in BASIS_ORDERS:
        cvxpy_results_fpath = os.path.join(
            results_dir, f"{dgp}_CVXPYEstimator_Order{order}.json"
        )
        cvxpy_loss = read_cvxpy_final_loss(cvxpy_results_fpath)
        fig_path = plot_for_dgp_and_order(
            dgp=dgp,
            order=order,
            figure_dir=figure_dir,
            cvxpy_loss=cvxpy_loss,
            logs_dir=logs_dir,
        )
        if fig_path:
            saved_paths.append(fig_path)
    return saved_paths


def main() -> None:
    """Same as iteration-based script."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        dgp_list = get_dgp_list(args.dgp)
    except ValueError as e:
        logging.error(str(e))
        sys.exit(1)

    ensure_dir(args.figure_dir)

    all_saved: List[str] = []
    for dgp in dgp_list:
        logging.info(f"Processing DGP: {dgp}")
        saved = run_for_dgp(dgp, args.figure_dir)
        all_saved.extend(saved)

    if not all_saved:
        logging.warning("No figures were generated. Check logs and results availability.")

    # Optionally save a standalone legend that matches styles used in plots
    if args.save_legend:
        save_algorithm_legend(
            figure_dir=args.figure_dir,
            filename=args.legend_filename,
            ncol=args.legend_ncol,
        )


if __name__ == "__main__":
    main()
