"""
Visualize knot selection per FLOP across optimization algorithms.

This script reads per-iteration logs for several optimization algorithms and
plots the number of selected knots versus cumulative FLOPs for each basis order.
It overlays a horizontal reference line for the final number of knots selected
by the CVXPY estimator.

The FLOP calculations are based on the analysis in algorithm_flop_per_iter.md
and are implementation-aware, accounting for matrix operations, line search,
coordinate descent, and other algorithm-specific costs.

Example usage:
    uv run experiments/compare_knot_selection/visualize_knot_selection_per_flop.py \
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
import numpy as np
import pandas as pd


# Add project root to Python path (match style of other scripts in this repo)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

# Candidate column names in logs representing the number of selected knots
KNOT_COUNT_CANDIDATES: List[str] = [
    "n_selected_knots",
    "num_selected_knots",
    "n_knots",
    "num_knots",
    "selected_knots",
    "knot_count",
    "active_knots",
    "nnz_knots",
    "nnz",
    "support_size",
]


###############################################################################
# FLOP Calculation Functions
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
        description="Visualize knot selection per FLOP across optimization algorithms."
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


def detect_knot_count_column(df: pd.DataFrame) -> Optional[str]:
    """Same as iteration-based script."""
    if df is None or df.empty:
        return None
    available = set(df.columns)
    for candidate in KNOT_COUNT_CANDIDATES:
        if candidate in available:
            return candidate
    # Heuristic: sometimes columns have slight naming variants
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in KNOT_COUNT_CANDIDATES:
        for lc, orig in lower_map.items():
            if candidate.lower() in lc:
                return orig
    return None


def read_cvxpy_selected_knots(results_fpath: str) -> Optional[float]:
    """Same as iteration-based script."""
    if not os.path.exists(results_fpath):
        logging.warning(f"CVXPY result file not found: {results_fpath}")
        return None
    try:
        with open(results_fpath, "r") as f:
            payload = json.load(f)
        results = payload.get("results", {}) if isinstance(payload, dict) else {}
        n_knots = results.get("n_selected_knots")
        if n_knots is None:
            logging.warning(
                f"'n_selected_knots' not found in CVXPY results: {results_fpath}"
            )
            return None
        return float(n_knots)
    except Exception as e:
        logging.error(f"Failed to read CVXPY results from {results_fpath}: {e}")
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
    
    n_samples = params["n_samples"]
    k_params = params["k_params"]
    g_grid = params["g_grid"]
    cd_sweeps = params.get("cd_sweeps", 2)
    
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
                alpha_num = float(alpha_val)
            except Exception:
                alpha_num = 1.0
            l_k = infer_line_search_steps(alpha_num)
            flops = calculate_newton_flops(n_samples, k_params, g_grid, cd_sweeps, l_k)
        elif algo_id == "ProximalNewtonLBFGSEstimator":
            alpha_val = row.get("α", row.get("alpha", 1.0))
            try:
                alpha_num = float(alpha_val)
            except Exception:
                alpha_num = 1.0
            l_k = infer_line_search_steps(alpha_num)
            flops = calculate_lbfgs_flops(n_samples, k_params, g_grid, l_k)
        else:
            flops = 0.0
        
        flops_per_iter.append(flops)
    
    # Return cumulative sum
    return np.cumsum(flops_per_iter)


def plot_for_dgp_and_order(
    dgp: str,
    order: int,
    figure_dir: str,
    cvxpy_knots: Optional[float],
    logs_dir: str,
) -> Optional[str]:
    """
    Generate a FLOP-based figure for a specific DGP and basis order.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    plotted_any = False
    x_max = 0.0
    y_max = 0.0
    y_min = 3200.0

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

        knot_col = detect_knot_count_column(df)
        if not knot_col:
            logging.warning(
                f"No knot-count column detected in {log_name}. Available columns: {list(df.columns)}"
            )
            continue

        # Sort by iteration and compute cumulative FLOPs
        df = df.sort_values("iteration").copy()
        cumulative_flops = compute_cumulative_flops(df, algo_id, params)
        
        if len(cumulative_flops) == 0:
            logging.info(f"Could not compute FLOPs for {display_name} ({log_name}); skipping.")
            continue

        # Determine baseline active count = K-1 (intercept unpenalized).
        baseline_active = None
        try:
            baseline_active = int(params.get("k_params", 0)) - 1 if params.get("k_params") is not None else None
        except Exception:
            baseline_active = None

        # Add starting point at small positive FLOPs (log-scale safe) with baseline_active if available;
        # else use first observed value to avoid an artificial jump.
        init_knots = baseline_active if baseline_active is not None else float(df[knot_col].iloc[0])
        flops_with_start = np.concatenate([[1.0], cumulative_flops])
        knots_with_start = np.concatenate([[init_knots], df[knot_col].values])
        
        # Filter out any NaN values
        valid_mask = ~np.isnan(flops_with_start) & ~np.isnan(knots_with_start)
        flops_plot = flops_with_start[valid_mask]
        knots_plot = knots_with_start[valid_mask]
        
        if len(flops_plot) == 0:
            continue

        ax.plot(
            flops_plot,
            knots_plot,
            label=f"{display_name} (n={knots_plot[-1]:.0f})",
            color=color,
            linewidth=1.0,
            linestyle=linestyle,
        )
        plotted_any = True
        x_max = max(x_max, float(flops_plot.max()))
        try:
            y_max = max(y_max, float(knots_plot.max()))
            y_min = min(y_min, float(knots_plot.min()))
        except Exception:
            pass

    if not plotted_any:
        plt.close(fig)
        logging.info(f"No curves plotted for {dgp}, order {order}; skipping figure.")
        return None

    # Reference line from CVXPY selected knots
    if cvxpy_knots is not None:
        ax.axhline(
            y=cvxpy_knots,
            color="k",
            linestyle="-",
            linewidth=.75,
            label=f"CVXPY (n={cvxpy_knots:.0f})",            
        )
        y_max = max(y_max, cvxpy_knots)

    # Formatting
    order_str = {
        0: "0th Order",
        1: "1st Order", 
        2: "2nd Order",
    }

    ax.set_title(f"{DGP_NAME_MAPPING[dgp]} ({order_str[order]})", fontsize=8)
    ax.set_xlabel("Cumulative FLOPs", fontsize=7)
    ax.set_ylabel("N(knots)", fontsize=7)
    
    # Format x-axis ticks for FLOPs (scientific notation for large numbers)
    xticks = ax.get_xticks()
    ax.set_xticks(ticks=xticks, labels=[f"{tick:.1e}" if tick >= 1e6 else f"{tick:.0f}" for tick in xticks], fontsize=6)
    
    yticks = ax.get_yticks()
    ax.set_yticks(ticks=yticks, labels=[f"{tick:.0f}" for tick in yticks], fontsize=7)
    
    ax.set_xscale("log")
    ax.set_ylim((y_min - (y_max - y_min) * 0.025, y_max + (y_max - y_min) * 0.175))
    ax.set_xlim(left=1.0, right=x_max * 1.05)
    # ax.legend(loc="upper center", fontsize=7, ncol=3)
    
    ensure_dir(figure_dir)
    fname = f"{dgp}_order_{order}_knot_selection_flops.png"
    fpath = os.path.join(figure_dir, fname)
    plt.grid(True, linestyle='--', alpha=0.25)
    plt.tight_layout()
    plt.savefig(fpath, dpi=300)
    plt.close(fig)
    logging.info(f"Saved figure: {fpath}")
    return fpath


def get_final_knot_count_from_log(dgp: str, algo_id: str, order: int, logs_dir: str) -> Optional[float]:
    """
    Extract the final knot count from a log file.
    """
    log_name = f"{dgp}_{algo_id}_Order{order}.log"
    log_fpath = os.path.join(logs_dir, log_name)
    df = parse_log_file(log_fpath)
    if df.empty:
        return None
    
    knot_col = detect_knot_count_column(df)
    if not knot_col:
        return None
    
    # Get the final (last) knot count
    df = df.sort_values("iteration")
    return float(df[knot_col].iloc[-1])


def collect_final_knot_counts(dgp_list: List[str]) -> pd.DataFrame:
    """
    Collect final knot counts for all DGP-Algorithm-Order combinations.
    Returns a DataFrame with DGP-Algorithm pairs as rows and Orders as columns.
    """
    # All algorithms including CVXPY
    all_algorithms = [
        ("FISTA", "FISTAEstimator"),
        ("ProximalAdaGrad", "ProximalAdaGradEstimator"), 
        ("ProximalNewton", "ProximalNewtonEstimator"),
        ("ProximalNewtonLBFGS", "ProximalNewtonLBFGSEstimator"),
        ("CVXPY", "CVXPYEstimator"),
    ]
    
    results = []
    
    for dgp in dgp_list:
        paths = build_paths_for_dgp(dgp)
        results_dir = paths["results"]
        logs_dir = paths["logs"]
        
        for display_name, algo_id in all_algorithms:
            row_data = {"DGP": dgp, "Algorithm": display_name}
            
            for order in BASIS_ORDERS:
                if algo_id == "CVXPYEstimator":
                    # For CVXPY, read from results JSON
                    cvxpy_results_fpath = os.path.join(
                        results_dir, f"{dgp}_{algo_id}_Order{order}.json"
                    )
                    knot_count = read_cvxpy_selected_knots(cvxpy_results_fpath)
                else:
                    # For other algorithms, read from log files
                    knot_count = get_final_knot_count_from_log(dgp, algo_id, order, logs_dir)
                
                row_data[f"Order_{order}"] = knot_count
            
            results.append(row_data)
    
    return pd.DataFrame(results)


def save_knot_counts_csv(dgp_list: List[str], output_dir: str, filename: str) -> str:
    """
    Save final knot counts to CSV file.
    """
    df = collect_final_knot_counts(dgp_list)
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, filename)
    df.to_csv(csv_path, index=False)
    
    logging.info(f"Saved knot counts CSV: {csv_path}")
    return csv_path


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
        cvxpy_knots = read_cvxpy_selected_knots(cvxpy_results_fpath)
        fig_path = plot_for_dgp_and_order(
            dgp=dgp,
            order=order,
            figure_dir=figure_dir,
            cvxpy_knots=cvxpy_knots,
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
    
    # Generate CSV output with final knot counts
    csv_output_dir = os.path.join(project_root, "experiments", "compare_knot_selection")
    csv_filename = "final_knot_counts_per_flop.csv"
    save_knot_counts_csv(dgp_list, csv_output_dir, csv_filename)


if __name__ == "__main__":
    main()
