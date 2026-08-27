"""
Visualize loss per iteration across optimization algorithms.

This script reads per-iteration logs for several optimization algorithms and
plots the objective function loss versus iteration for each basis order.
It overlays a horizontal reference line for the final loss achieved
by the CVXPY estimator.

Example usage:
    uv run experiments/compare_knot_selection/visualize_loss_per_iter.py \
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

# Algorithms to visualize (display name, identifier used in filenames, color)
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
# Utilities
###############################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize loss per iteration across optimization algorithms."
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


def detect_loss_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    available = set(df.columns)
    for candidate in LOSS_CANDIDATES:
        if candidate in available:
            return candidate
    # Heuristic: sometimes columns have slight naming variants, try case-insensitive contains
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
    cvxpy_loss: Optional[float],
    logs_dir: str,
) -> Optional[str]:
    """
    Generate a figure for a specific DGP and basis order.

    Returns the saved figure path, or None if nothing was plotted.
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

        loss_col = detect_loss_column(df)
        if not loss_col:
            logging.warning(
                f"No loss column detected in {log_name}. Available columns: {list(df.columns)}"
            )
            continue

        # Shift iteration by +1 for log-scale x-axis (avoid log(0))
        df = df.sort_values("iteration").copy()
        df["iter_plot"] = df["iteration"].astype(float) + 1.0
        
        # Filter out any invalid loss values
        df_loss = df[["iter_plot", loss_col]].dropna()
        loss_vals = np.asarray(df_loss[loss_col])
        finite_mask = np.isfinite(loss_vals)
        df_loss = df_loss.loc[finite_mask]
        
        if df_loss.empty:
            logging.info(f"Loss series empty for {display_name} ({log_name}); skipping.")
            continue

        # Add starting point at iteration 0 with loss=0 (before optimization starts)
        df_loss_with_start = pd.DataFrame({"iter_plot": [1.0], loss_col: [0.0]})
        
        # Append actual data, ensuring no duplicate iter_plot=1.0
        actual_data = df_loss[df_loss["iter_plot"] > 1.0]  # Skip iter=1 to avoid conflict
        
        df_loss = pd.concat([df_loss_with_start, actual_data], ignore_index=True)
        if df_loss.empty:
            logging.info(f"Loss series empty after preprocessing for {display_name} ({log_name}); skipping.")
            continue

        ax.plot(
            df_loss["iter_plot"],
            df_loss[loss_col],
            label=f"{display_name} (final={df_loss[loss_col].iloc[-1]:.2e})",
            color=color,
            linewidth=1.0,
            linestyle=linestyle,
        )
        plotted_any = True
        x_max = max(x_max, float(df_loss["iter_plot"].max()))
        try:
            y_max = max(y_max, float(df_loss[loss_col].max()))
            y_min = min(y_min, float(df_loss[loss_col].min()))
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
    ax.set_xlabel("Iteration", fontsize=7)
    
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
    ax.set_xticks(ticks=ax.get_xticks(), labels=[f"{tick:.0f}" for tick in ax.get_xticks()], fontsize=6)
    
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
    fname = f"{dgp}_order_{order}_loss_per_iter.png"
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
