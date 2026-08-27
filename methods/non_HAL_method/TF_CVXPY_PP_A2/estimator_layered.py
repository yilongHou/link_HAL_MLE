"""
Trend Filtering CVXPY with Parametric Penalty using TRUE Algorithm 2 approach.

This implementation TRULY avoids forming the full H or H^{-1} matrix by expressing
all constraints as layered simple constraints that MOSEK can handle efficiently.

Key Insight from the Paper:
- H^{-1} = [C; (1/k!) D^{(k+1)}] where D^{(k+1)} is the (k+1)-th divided difference
- D^{(m+1)} = D_1 @ (m * Delta^{-m} @ D^(m))
- Instead of forming these matrices, we express the constraint through intermediate
  variables at each level

The constraint ||H^{-1} @ theta||_1 <= C becomes:
- u_0 = theta
- u_1 = first_differences(u_0)
- u_2 = scaled_differences(u_1, gaps_1)
- ...
- u_{k+1} = final divided differences
- ||[polynomial_coeffs; (1/k!) * u_{k+1}]||_1 <= C

Each level uses simple element-wise constraints, avoiding large matrix multiplications.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from math import factorial
from typing import Dict, Optional, Tuple


class TrendFilteringCVXPYPPA2Layered:
    """
    TFPP using truly layered constraints - no explicit H or H^{-1} matrix formation.
    
    This expresses the divided difference constraint through intermediate variables,
    giving MOSEK a sequence of simple constraints rather than one large matrix.
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
        self._u_values: Optional[Dict] = None  # Store intermediate variables for debugging
        self._poly_coeffs: Optional[np.ndarray] = None  # Store polynomial coefficients for debugging
        self._gaps: Optional[Dict] = None  # Store gap vectors for debugging
    
    def fit(self, data: pd.DataFrame) -> "TrendFilteringCVXPYPPA2Layered":
        """
        Fit using layered constraints - NO explicit H matrix formation.
        """
        y = data["W1"].values
        self.n_samples = len(y)
        
        # DATA-ADAPTIVE knots from actual data points
        sorted_y = np.sort(y)
        inner_knots = np.unique(sorted_y)
        self.knots = np.unique(np.concatenate([[0.0], inner_knots, [1.0]]))
        
        n_bins = len(self.knots) - 1
        self.bin_widths = np.diff(self.knots)
        
        # Node positions (left endpoints of bins)
        x_nodes = self.knots[:-1]
        n = len(x_nodes)
        
        if self.k >= n - 1:
            raise ValueError(f"k={self.k} must be <= n - 2 = {n - 2}")
        
        # Count observations per bin
        counts, _ = np.histogram(y, bins=self.knots)
        
        # Compute gap vectors for each level (these are just scalars, not matrices!)
        gaps = {}
        for m in range(1, self.k + 2):
            gaps[m] = x_nodes[m:] - x_nodes[:-m]  # Vector of length n-m
        
        # Check for zero gaps
        for m, g in gaps.items():
            if np.any(g <= 0):
                raise ValueError(f"Zero or negative gap at level {m}")
        
        # ==================================================================
        # Set up CVXPY variables - LAYERED approach
        # ==================================================================
        
        # Main parameter: theta (log-density at each bin)
        theta = cp.Variable(n)
        
        # Intermediate variables for divided differences
        # u[m] has length n - m
        u = {}
        for m in range(1, self.k + 2):
            u[m] = cp.Variable(n - m)
        
        # ==================================================================
        # Constraints - expressed as SIMPLE element-wise operations
        # ==================================================================
        constraints = []
        
        # Identifiability: theta[0] = 0
        constraints.append(theta[0] == 0)
        
        # Level 1: First difference
        # u_1[j] = theta[j+1] - theta[j] for j = 0, ..., n-2
        # Expressed as: u_1 = D_1 @ theta (but we write it element-wise)
        for j in range(n - 1):
            constraints.append(u[1][j] == theta[j + 1] - theta[j])
        
        # Levels 2 to k+1: Divided differences with scaling
        # Following Algorithm 2 / Ryan's recursion: D^{(m+1)} = D_1 @ (m * Delta^{-m} @ D^{(m)})
        # This means: DIVIDE by gaps[m] FIRST, THEN take differences
        # u_{m+1}[j] = m * (u_m[j+1]/gaps[m][j+1] - u_m[j]/gaps[m][j])
        for m in range(1, self.k + 1):
            n_m = n - m  # length of u[m]
            for j in range(n_m - 1):  # u[m+1] has length n_m - 1 = n - m - 1
                # Correct: divide first, then difference
                constraints.append(
                    u[m + 1][j] == m * (u[m][j + 1] / gaps[m][j + 1] - u[m][j] / gaps[m][j])
                )
        
        # ==================================================================
        # Polynomial coefficients (C matrix rows)
        # These extract the polynomial part of H^{-1} @ theta
        # ==================================================================
        # Following Ryan Tibshirani's falling factorial basis:
        # C_0 = theta[0] (intercept - dropped for identifiability)
        # C_i = (1/(i-1)!) * first element of (Delta_i^{-1} @ D^i @ theta)
        #     = (1/(i-1)!) * u_i[0] / gaps[i][0]  for i >= 1
        #
        # For k=1: We need C_1 = u_1[0] / gaps[1][0] (linear slope)
        # For k=2: We need C_1 and C_2 = (1/1!) * u_2[0] / gaps[2][0]
        
        # Compute polynomial coefficients using intermediate u variables
        poly_coeffs = []
        for i in range(1, self.k + 1):
            # C_i = (1/(i-1)!) * u_i[0] / gaps[i][0]
            scale = 1.0 / factorial(i - 1) if i > 1 else 1.0
            poly_coeffs.append(scale * u[i][0] / gaps[i][0])
        
        # ==================================================================
        # Full sectional variation norm constraint
        # ||H^{-1} @ theta||_1 = ||[C; (1/k!) D^{k+1}] @ theta||_1 <= C
        # ==================================================================
        penalty_scale = 1.0 / factorial(self.k)
        
        # Create auxiliary variable for the full penalty (to handle absolute values)
        # Penalty = sum(|C_i|) + (1/k!) * ||u_{k+1}||_1
        if self.drop_intercept:
            # Build the full penalty including polynomial coefficients (except intercept)
            # Use auxiliary variable t for the sum of |poly_coeffs|
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
            # Include intercept (C_0 = theta[0]) - but we set theta[0]=0 anyway
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
        # ==================================================================
        log_dx = np.log(self.bin_widths)
        logZ = cp.log_sum_exp(theta + log_dx)
        nll = -counts @ theta + self.n_samples * logZ
        
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
        self._gaps = gaps  # Store gaps for debugging
        
        # Compute and store polynomial coefficients for debugging
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
                  reconstruction on the knot grid.
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() first")

        mode = str(mode).lower().strip()
        if mode not in {"piecewise", "reconstructed"}:
            raise ValueError(f"Unknown mode={mode}. Expected 'piecewise' or 'reconstructed'.")

        grid_points = np.linspace(0.0, 1.0, self.n_grid_points)
        theta = self.theta_hat
        dx = self.bin_widths

        # ------------------------------------------------------------------
        # Piecewise-constant density (default; consistent with bin likelihood)
        # ------------------------------------------------------------------
        if mode == "piecewise" or self.k == 0:
            log_terms = theta + np.log(dx)
            m = np.max(log_terms)
            logZ = m + np.log(np.sum(np.exp(log_terms - m)))
            log_density_levels = theta - logZ

            bin_indices = np.searchsorted(self.knots, grid_points, side="right") - 1
            bin_indices = np.clip(bin_indices, 0, len(theta) - 1)
            density = np.exp(log_density_levels[bin_indices])
            return grid_points, density

        # ------------------------------------------------------------------
        # Falling-factorial reconstruction (k>=1): evaluate basis at knots, solve H alpha = theta,
        # then evaluate f(x)=H(x) alpha on grid.
        # ------------------------------------------------------------------
        if self.k < 1:
            raise NotImplementedError("mode='reconstructed' requires k>=1.")

        from math import factorial

        knots = np.asarray(self.knots, dtype=float).ravel()
        x_nodes = knots[:-1]  # length n == len(theta_hat)
        theta_nodes = np.asarray(theta, dtype=float).ravel()
        n = int(theta_nodes.size)
        if x_nodes.size != n:
            raise ValueError("Expected len(theta_hat) == len(knots)-1 for reconstruction.")
        if n < (self.k + 1):
            raise ValueError("Not enough nodes for reconstruction.")

        def _ff_basis(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float).ravel()
            B = np.zeros((x.size, n), dtype=float)
            # Polynomial part
            for j in range(0, self.k + 1):
                if j == 0:
                    B[:, j] = 1.0
                    continue
                prod = np.ones_like(x)
                for l in range(j):
                    prod *= (x - float(x_nodes[l]))
                B[:, j] = prod / float(factorial(j))
            # Truncated part
            for j in range(self.k + 1, n):
                prod = np.ones_like(x)
                for l in range(j - self.k, j):
                    prod *= np.maximum(0.0, x - float(x_nodes[l]))
                B[:, j] = prod / float(factorial(self.k))
            return B

        H = _ff_basis(x_nodes)  # n x n lower-triangular
        alpha = np.linalg.solve(H, theta_nodes)
        f_grid = _ff_basis(grid_points) @ alpha

        # Normalize numerically on the grid
        m = float(np.max(f_grid))
        Z = float(np.trapz(np.exp(f_grid - m), grid_points))
        if not np.isfinite(Z) or Z <= 0:
            raise ValueError("Failed to normalize reconstructed density.")
        density = np.exp(f_grid - (m + np.log(Z)))
        return grid_points, density
    
    def get_results(self, density_mode: str = "piecewise") -> Dict:
        """Get estimation results."""
        if not self.is_fitted:
            raise ValueError("Must call fit() first")

        grid_points, density = self.get_density(mode=density_mode)
        
        # Compute penalty components for inspection
        penalty_scale = 1.0 / factorial(self.k)
        u_k1_penalty = penalty_scale * np.sum(np.abs(self._u_values[self.k + 1]))
        poly_penalty = np.sum(np.abs(self._poly_coeffs)) if len(self._poly_coeffs) > 0 else 0.0
        total_penalty = poly_penalty + u_k1_penalty
        
        return {
            "k": self.k,
            "norm_constraint": self.norm_constraint,
            "theta_hat": self.theta_hat,
            "knots": self.knots,
            "bin_widths": self.bin_widths,
            "n_samples": self.n_samples,
            "n_bins": len(self.bin_widths),
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
        }

