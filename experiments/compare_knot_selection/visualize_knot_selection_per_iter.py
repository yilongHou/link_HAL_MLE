"""
Visualize knot selection per iteration across optimization algorithms.

This script reads per-iteration logs for several optimization algorithms and
plots the number of selected knots versus iteration for each basis order.
It overlays a horizontal reference line for the final number of knots selected
by the CVXPY estimator.

Example usage:
    uv run experiments/compare_knot_selection/visualize_knot_selection_per_iter.py \
        --dgp all \
        --figure-dir paper/resources/optimization_algorithms/per_iter
"""

import argparse
import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

# Algorithms to visualize (display name, identifier used in filenames, color)
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
# Utilities
###############################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize knot selection per iteration across optimization algorithms."
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
        required=False,
        default="paper/resources/optimization_algorithms/per_iter",
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
        default="legend_knot_selection_algorithms.png",
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

    The expected line format is: "Iter <int>: key1=val1 key2=val2 ..."
    Keys and values are extracted with regex, matching the project's existing
    logging/visualization style.
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
                    # Skip values that cannot be parsed as float
                    continue
            log_entries.append(entry)

    if not log_entries:
        return pd.DataFrame()
    return pd.DataFrame(log_entries)


def detect_knot_count_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    available = set(df.columns)
    for candidate in KNOT_COUNT_CANDIDATES:
        if candidate in available:
            return candidate
    # Heuristic: sometimes columns have slight naming variants, try case-insensitive contains
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in KNOT_COUNT_CANDIDATES:
        for lc, orig in lower_map.items():
            if candidate.lower() in lc:
                return orig
    return None


def parse_initialization_params(log_fpath: str, algo_id: str) -> Dict[str, Optional[int]]:
    """
    Parse initialization parameters to recover K (parameter count) and n_grid.
    We mainly need K to set the baseline initial active-coefficient count K-1.
    """
    params = {
        "k_params": None,
        "g_grid": None,
    }
    if not os.path.exists(log_fpath):
        return params
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
                if algo_id == "ProximalNewtonEstimator":
                    params["g_grid"] = int(match.group(1)) - 1
                    params["k_params"] = int(match.group(4))
                else:
                    params["g_grid"] = int(match.group(1)) - 1
                    params["k_params"] = int(match.group(3))
                break
    return params


def fallback_params_from_results(dgp: str, algo_id: str, order: int) -> Dict[str, Optional[int]]:
    paths = build_paths_for_dgp(dgp)
    results_dir = paths["results"]
    result_fpath = os.path.join(results_dir, f"{dgp}_{algo_id}_Order{order}.json")
    params = {"k_params": None, "g_grid": None}
    if not os.path.exists(result_fpath):
        return params
    try:
        with open(result_fpath, "r") as f:
            payload = json.load(f)
        n_grid_points = payload.get("estimator_setup", {}).get("estimator_params", {}).get("n_grid_points")
        theta_hat = payload.get("results", {}).get("theta_hat")
        if isinstance(n_grid_points, int):
            params["g_grid"] = max(0, n_grid_points - 1)
        if isinstance(theta_hat, list):
            params["k_params"] = int(len(theta_hat))
    except Exception:
        pass
    return params


def read_cvxpy_selected_knots(results_fpath: str) -> Optional[float]:
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
    base = os.path.join(
        project_root, "experiments", "compare_knot_selection", f"single_{dgp}"
    )
    return {
        "base": base,
        "logs": os.path.join(base, "logs"),
        "results": os.path.join(base, "results"),
    }


def plot_for_dgp_and_order(
    dgp: str,
    order: int,
    figure_dir: str,
    cvxpy_knots: Optional[float],
    logs_dir: str,
) -> Optional[str]:
    """
    Generate a figure for a specific DGP and basis order.

    Returns the saved figure path, or None if nothing was plotted.
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

        knot_col = detect_knot_count_column(df)
        if not knot_col:
            logging.warning(
                f"No knot-count column detected in {log_name}. Available columns: {list(df.columns)}"
            )
            continue

        # Parse params for baseline K-1 (active coefficients excluding intercept)
        params = parse_initialization_params(log_fpath, algo_id)
        if params["k_params"] is None:
            fb = fallback_params_from_results(dgp, algo_id, order)
            for key in params:
                if params[key] is None and fb.get(key) is not None:
                    params[key] = fb[key]
        baseline_active = None
        if params["k_params"] is not None:
            try:
                baseline_active = int(params["k_params"]) - 1
            except Exception:
                baseline_active = None

        # Shift iteration by +1 for log-scale x-axis (avoid log(0))
        df = df.sort_values("iteration").copy()
        df["iter_plot"] = df["iteration"].astype(float) + 1.0
        # Insert baseline at iter_plot=1.0 with active count = K-1 (if available),
        # otherwise use the first observed value
        init_value = baseline_active if baseline_active is not None else float(df[knot_col].iloc[0])
        df_knot = pd.DataFrame({"iter_plot": [1.0], knot_col: [init_value]})
        
        # Append actual data, ensuring no duplicate iter_plot=1.0
        actual_data = df[["iter_plot", knot_col]].dropna()
        actual_data = actual_data[actual_data["iter_plot"] > 1.0]  # Skip iter=1 to avoid conflict
        
        df_knot = pd.concat([df_knot, actual_data], ignore_index=True)
        if df_knot.empty:
            logging.info(f"Knot-count series empty for {display_name} ({log_name}); skipping.")
            continue

        ax.plot(
            df_knot["iter_plot"],
            df_knot[knot_col],
            label=f"{display_name} (n={df_knot[knot_col].iloc[-1]:.0f})",
            color=color,
            linewidth=1.0,
            linestyle=linestyle,
        )
        plotted_any = True
        x_max = max(x_max, float(df_knot["iter_plot"].max()))
        try:
            y_max = max(y_max, float(df_knot[knot_col].max()))
            y_min = min(y_min, float(df_knot[knot_col].min()))
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
    ax.set_xlabel("Iteration", fontsize=7)
    ax.set_ylabel("N(knots)", fontsize=7)
    ax.set_xticks(ticks=ax.get_xticks(), labels=[f"{tick:.0f}" for tick in ax.get_xticks()], fontsize=6)
    yticks = ax.get_yticks()
    ax.set_yticks(ticks=yticks, labels=[f"{tick:.0f}" for tick in yticks], fontsize=7)
    ax.set_xscale("log")
    ax.set_ylim((y_min - (y_max - y_min) * 0.025, y_max + (y_max - y_min) * 0.175))
    ax.set_xlim(left=1.0, right=x_max * 1.05)
    # ax.legend(loc="upper center", fontsize=7, ncol=3)
    ensure_dir(figure_dir)
    fname = f"{dgp}_order_{order}_knot_selection.png"
    fpath = os.path.join(figure_dir, fname)
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)
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
    csv_filename = "final_knot_counts_per_iter.csv"
    save_knot_counts_csv(dgp_list, csv_output_dir, csv_filename)

    # Optionally save a standalone legend that matches styles used in plots
    if args.save_legend:
        save_algorithm_legend(
            figure_dir=args.figure_dir,
            filename=args.legend_filename,
            ncol=args.legend_ncol,
        )


if __name__ == "__main__":
    main()

