"""
Trend Filtering CVXPY with Parametric Penalty using TRUE Algorithm 2 approach.
VERSION 2: Fixed to include x=1.0 in nodes, making TFA2 equivalent to HAL for k=1.

Key Fix:
- OLD: x_nodes = [0, ξ₁, ..., ξₘ] (m+1 nodes, m-1 potential kinks)
- NEW: x_nodes = [0, ξ₁, ..., ξₘ, 1] (m+2 nodes, m potential kinks)

This allows TFA2 to have a kink at the last data point ξₘ, matching HAL's flexibility.

The key insight is that for piecewise linear functions (k=1), including the right
boundary as a node gives the same number of potential kinks as HAL's truncated
power basis with m spline knots.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from math import factorial
from typing import Dict, Optional, Tuple


class TrendFilteringCVXPYPPA2LayeredV2:
    """
    TFPP using truly layered constraints - no explicit H or H^{-1} matrix formation.
    
    Version 2: Includes x=1.0 in nodes for equivalence with HAL.
    """
    
    def __init__(
        self,
        k: int = 1,
        norm_constraint: float = 1.0,
        n_grid_points: int = 1000,
        solver: str = "MOSEK",
        tol: float = 1e-6,
        use_secondary_solver: bool = False,
        drop_intercept: bool = True,
    ):
        self.k = k
        self.norm_constraint = norm_constraint
        self.n_grid_points = n_grid_points
        self.solver = solver
        self.tol = tol
        self.use_secondary_solver = use_secondary_solver
        self.drop_intercept = drop_intercept
        self.is_fitted = False
        
        self.theta_hat: Optional[np.ndarray] = None
        self.knots: Optional[np.ndarray] = None
        self.bin_widths: Optional[np.ndarray] = None
        self.n_samples: Optional[int] = None
        self.optimization_status: Optional[str] = None
        self._u_values: Optional[Dict] = None
        self._poly_coeffs: Optional[np.ndarray] = None
        self._gaps: Optional[Dict] = None
        self._x_nodes: Optional[np.ndarray] = None  # Store for reconstruction
    
    def fit(self, data: pd.DataFrame) -> "TrendFilteringCVXPYPPA2LayeredV2":
        """
        Fit using layered constraints - NO explicit H matrix formation.
        
        KEY FIX: x_nodes now includes 1.0, giving the same number of potential
        kinks as HAL's truncated power basis.
        """
        y = data["W1"].values
        self.n_samples = len(y)
        
        # DATA-ADAPTIVE knots from actual data points
        sorted_y = np.sort(y)
        inner_knots = np.unique(sorted_y)
        self.knots = np.unique(np.concatenate([[0.0], inner_knots, [1.0]]))
        
        n_bins = len(self.knots) - 1
        self.bin_widths = np.diff(self.knots)
        
        # KEY FIX: Include 1.0 in x_nodes (was: knots[:-1], now: knots)
        # This gives us one more potential kink at the last data point
        x_nodes = self.knots.copy()  # [0, ξ₁, ..., ξₘ, 1]
        self._x_nodes = x_nodes
        n = len(x_nodes)  # Now n = n_bins + 1
        
        if self.k >= n - 1:
            raise ValueError(f"k={self.k} must be <= n - 2 = {n - 2}")
        
        # Count observations per bin (unchanged - bins are between knots)
        counts, _ = np.histogram(y, bins=self.knots)
        
        # Compute gap vectors for each level
        gaps = {}
        for m in range(1, self.k + 2):
            gaps[m] = x_nodes[m:] - x_nodes[:-m]
        
        # Check for zero gaps
        for m, g in gaps.items():
            if np.any(g <= 0):
                raise ValueError(f"Zero or negative gap at level {m}")
        
        # ==================================================================
        # Set up CVXPY variables - LAYERED approach
        # ==================================================================
        
        # Main parameter: theta (log-density at each node, including x=1)
        theta = cp.Variable(n)
        
        # Intermediate variables for divided differences
        u = {}
        for m in range(1, self.k + 2):
            u[m] = cp.Variable(n - m)
        
        # ==================================================================
        # Constraints
        # ==================================================================
        constraints = []
        
        # Identifiability: theta[0] = 0
        constraints.append(theta[0] == 0)
        
        # Level 1: First difference
        for j in range(n - 1):
            constraints.append(u[1][j] == theta[j + 1] - theta[j])
        
        # Levels 2 to k+1: Divided differences with scaling
        for m in range(1, self.k + 1):
            n_m = n - m
            for j in range(n_m - 1):
                constraints.append(
                    u[m + 1][j] == m * (u[m][j + 1] / gaps[m][j + 1] - u[m][j] / gaps[m][j])
                )
        
        # ==================================================================
        # Polynomial coefficients (C matrix rows)
        # ==================================================================
        poly_coeffs = []
        for i in range(1, self.k + 1):
            scale = 1.0 / factorial(i - 1) if i > 1 else 1.0
            poly_coeffs.append(scale * u[i][0] / gaps[i][0])
        
        # ==================================================================
        # Full sectional variation norm constraint
        # ==================================================================
        penalty_scale = 1.0 / factorial(self.k)
        
        if self.drop_intercept:
            if len(poly_coeffs) > 0:
                t_poly = cp.Variable(len(poly_coeffs))
                for i, c in enumerate(poly_coeffs):
                    constraints.append(t_poly[i] >= c)
                    constraints.append(t_poly[i] >= -c)
                total_penalty = cp.sum(t_poly) + penalty_scale * cp.norm1(u[self.k + 1])
            else:
                total_penalty = penalty_scale * cp.norm1(u[self.k + 1])
            constraints.append(total_penalty <= self.norm_constraint)
        else:
            if len(poly_coeffs) > 0:
                t_poly = cp.Variable(len(poly_coeffs))
                for i, c in enumerate(poly_coeffs):
                    constraints.append(t_poly[i] >= c)
                    constraints.append(t_poly[i] >= -c)
                total_penalty = cp.sum(t_poly) + penalty_scale * cp.norm1(u[self.k + 1])
            else:
                total_penalty = penalty_scale * cp.norm1(u[self.k + 1])
            constraints.append(total_penalty <= self.norm_constraint)
        
        # ==================================================================
        # Objective: Negative log-likelihood
        # 
        # KEY: The likelihood uses theta[:-1] (excluding the value at x=1)
        # because bins are [0,ξ₁), [ξ₁,ξ₂), ..., [ξₘ,1), and theta[i]
        # represents the log-density in bin i.
        # ==================================================================
        log_dx = np.log(self.bin_widths)
        
        # Use theta[:-1] for the likelihood (n_bins = n-1 theta values)
        theta_for_bins = theta[:-1]  # Exclude theta at x=1
        
        logZ = cp.log_sum_exp(theta_for_bins + log_dx)
        nll = -counts @ theta_for_bins + self.n_samples * logZ
        
        problem = cp.Problem(cp.Minimize(nll), constraints)
        
        # ==================================================================
        # Solve
        # ==================================================================
        try:
            problem.solve(solver=self.solver, verbose=False)
            self.optimization_status = problem.status
        except Exception as e:
            if not self.use_secondary_solver:
                raise RuntimeError(f"CVXPY optimization failed: {e}")
            
            print(f"{self.solver} failed: {e}")
            for backup_solver in ["CLARABEL", "ECOS", "SCS"]:
                try:
                    problem.solve(solver=backup_solver, verbose=False)
                    self.optimization_status = problem.status
                    print(f"{backup_solver} succeeded")
                    break
                except:
                    continue
            else:
                raise RuntimeError("All solvers failed")
        
        if theta.value is None or problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"Optimization failed: {problem.status}")
        
        self.theta_hat = theta.value.copy()
        self._u_values = {m: u[m].value.copy() for m in range(1, self.k + 2)}
        self._gaps = gaps
        
        # Compute and store polynomial coefficients
        poly_coeff_values = []
        for i in range(1, self.k + 1):
            scale = 1.0 / factorial(i - 1) if i > 1 else 1.0
            poly_coeff_values.append(scale * self._u_values[i][0] / gaps[i][0])
        self._poly_coeffs = np.array(poly_coeff_values) if poly_coeff_values else np.array([])
        
        self.is_fitted = True
        
        return self
    
    def get_density(self, mode: str = "piecewise") -> Tuple[np.ndarray, np.ndarray]:
        """
        Get estimated density.

        Args:
            mode:
                - "piecewise": piecewise-constant density implied by (theta_hat, bin_widths)
                - "reconstructed": for k=1, return a *continuous* (piecewise-linear in log-density)
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() first")

        mode = str(mode).lower().strip()
        if mode not in {"piecewise", "reconstructed"}:
            raise ValueError(f"Unknown mode={mode}. Expected 'piecewise' or 'reconstructed'.")

        grid_points = np.linspace(0.0, 1.0, self.n_grid_points)
        theta = self.theta_hat
        dx = self.bin_widths
        x_nodes = self._x_nodes

        # ------------------------------------------------------------------
        # Piecewise-constant density (uses theta[:-1] for bins)
        # ------------------------------------------------------------------
        if mode == "piecewise" or self.k == 0:
            theta_bins = theta[:-1]  # Exclude theta at x=1
            log_terms = theta_bins + np.log(dx)
            m = np.max(log_terms)
            logZ = m + np.log(np.sum(np.exp(log_terms - m)))
            log_density_levels = theta_bins - logZ

            bin_indices = np.searchsorted(self.knots, grid_points, side="right") - 1
            bin_indices = np.clip(bin_indices, 0, len(theta_bins) - 1)
            density = np.exp(log_density_levels[bin_indices])
            return grid_points, density

        # ------------------------------------------------------------------
        # k=1 reconstruction: falling-factorial basis
        # Now includes the kink at the last data point!
        # ------------------------------------------------------------------
        if self.k != 1:
            raise NotImplementedError("mode='reconstructed' is currently implemented only for k=1.")

        theta_arr = np.asarray(theta, dtype=float).ravel()
        n = int(theta_arr.size)
        if n < 2:
            raise ValueError("Need at least 2 nodes for k=1 reconstruction.")

        # Compute alpha = H^{-1} theta for k=1
        gaps1 = np.diff(x_nodes)
        if np.any(gaps1 <= 0):
            raise ValueError("Non-increasing x_nodes for reconstruction.")
        
        u1 = np.diff(theta_arr)
        alpha2 = float(u1[0] / gaps1[0])
        u2 = (u1[1:] / gaps1[1:]) - (u1[:-1] / gaps1[:-1])  # Now has n-2 = m elements!

        # Falling factorial basis eval for k=1:
        # f(x) = alpha2*(x-x1) + sum_j u2[j]*(x - x_{j+1})+
        x1 = float(x_nodes[0])
        f_grid = alpha2 * (grid_points - x1)
        if u2.size:
            for j in range(u2.size):
                f_grid += float(u2[j]) * np.maximum(0.0, grid_points - float(x_nodes[j + 1]))

        # Exact normalization for exp(piecewise-linear) on breakpoints
        breaks = x_nodes  # All nodes including x=1
        f_breaks = alpha2 * (breaks - x1)
        if u2.size:
            for j in range(u2.size):
                f_breaks += float(u2[j]) * np.maximum(0.0, breaks - float(x_nodes[j + 1]))

        # Numerically stable logZ computation
        m = float(np.max(f_breaks))
        g = f_breaks - m
        Z_scaled = 0.0
        for i in range(breaks.size - 1):
            x0 = float(breaks[i])
            x1b = float(breaks[i + 1])
            if x1b <= x0:
                continue
            g0 = float(g[i])
            g1 = float(g[i + 1])
            dx_i = x1b - x0
            a = (g1 - g0) / dx_i
            if abs(a) < 1e-12:
                Z_scaled += np.exp(g0) * dx_i
            else:
                Z_scaled += np.exp(g0) * np.expm1(a * dx_i) / a
        
        if not np.isfinite(Z_scaled) or Z_scaled <= 0:
            raise ValueError("Failed to normalize reconstructed density.")
        
        logZ = m + np.log(Z_scaled)
        density = np.exp(f_grid - logZ)
        return grid_points, density
    
    def get_results(self, density_mode: str = "piecewise") -> Dict:
        """Get estimation results."""
        if not self.is_fitted:
            raise ValueError("Must call fit() first")

        grid_points, density = self.get_density(mode=density_mode)
        
        # Compute penalty components
        penalty_scale = 1.0 / factorial(self.k)
        u_k1_penalty = penalty_scale * np.sum(np.abs(self._u_values[self.k + 1]))
        poly_penalty = np.sum(np.abs(self._poly_coeffs)) if len(self._poly_coeffs) > 0 else 0.0
        total_penalty = poly_penalty + u_k1_penalty
        
        return {
            "k": self.k,
            "norm_constraint": self.norm_constraint,
            "theta_hat": self.theta_hat,
            "knots": self.knots,
            "x_nodes": self._x_nodes,
            "bin_widths": self.bin_widths,
            "n_samples": self.n_samples,
            "n_bins": len(self.bin_widths),
            "n_nodes": len(self._x_nodes),
            "estimated_density": density,
            "grid_points": grid_points,
            "density_mode": str(density_mode),
            "optimization_status": self.optimization_status,
            "min_gap": np.min(self.bin_widths),
            "max_gap": np.max(self.bin_widths),
            # Debugging: penalty components
            "poly_coeffs": self._poly_coeffs,
            "poly_penalty": poly_penalty,
            "divided_diff_penalty": u_k1_penalty,
            "total_penalty": total_penalty,
            # Number of potential kinks (for comparison with HAL)
            "n_potential_kinks": len(self._u_values[self.k + 1]) if self.k >= 1 else 0,
        }

