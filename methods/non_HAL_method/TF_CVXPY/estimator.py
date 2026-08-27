import numpy as np
import pandas as pd
import cvxpy as cp
import scipy.sparse as sp
from typing import Dict, Optional, Tuple


def build_difference_matrix(n: int, k: int):
    """
    Build k-th order difference matrix D^(k) of size (n-k) x n.
    """
    if k == 0:
        return sp.eye(n, format='csr')
    elif k == 1:
        # Create first-order difference matrix manually
        row = []
        col = []
        data = []
        for i in range(n-1):
            row.extend([i, i])
            col.extend([i, i+1])
            data.extend([-1, 1])
        return sp.csr_matrix((data, (row, col)), shape=(n-1, n))
    else:
        D_prev = build_difference_matrix(n, k-1)
        D1 = build_difference_matrix(D_prev.shape[0], 1)
        return D1 @ D_prev


class TrendFilteringCVXPYEstimator:
    """
    Trend Filtering density estimator using CVXPY for convex optimization.
    
    This class implements trend filtering with CVXPY, solving:
    
    min_θ  f(θ)
    s.t.   ||D^(k+1)θ||₁ ≤ C
    
    where f(θ) = -Σθᵢ + (n+1)log(Z(θ)) is the negative log-likelihood,
    D^(k+1) is the (k+1)-th order difference matrix, and C is the constraint bound.
    """
    
    def __init__(
        self,
        k: int = 1,
        norm_constraint: float = 1.0,
        n_grid_points: int = 1000,
        solver: str = "MOSEK",
        tol: float = 1e-6,
        use_secondary_solver: bool = False,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize Trend Filtering CVXPY estimator.
        
        Args:
            k: Order of trend filtering (k=0: piecewise constant, k=1: piecewise linear, etc.)
            norm_constraint: Upper bound C for the constraint ||D^(k+1)θ||₁ ≤ C
            n_grid_points: Number of grid points for density evaluation
            solver: CVXPY solver to use (default: "MOSEK")
            tol: Tolerance for optimization
            use_secondary_solver: Whether to try backup solvers if primary fails
        """
        self.k = k
        self.norm_constraint = norm_constraint
        self.n_grid_points = n_grid_points
        self.solver = solver
        self.tol = tol
        self.use_secondary_solver = use_secondary_solver
        self.is_fitted = False
        
        # Results storage
        self.theta_hat: Optional[np.ndarray] = None
        self.knots: Optional[np.ndarray] = None
        self.bin_widths: Optional[np.ndarray] = None
        self.n_samples: Optional[int] = None
        
        # Internal attributes for inspection
        self.optimized_theta_raw: Optional[np.ndarray] = None
        self.optimization_status: Optional[str] = None
        
    def fit(self, data: pd.DataFrame) -> 'TrendFilteringCVXPYEstimator':
        """
        Fit the Trend Filtering density estimator.
        
        Args:
            data: DataFrame with column 'W1' containing the observations
            
        Returns:
            Self for method chaining
        """
        y = data['W1'].values
        self.n_samples = len(y)

        # Grid construction: EQUALLY SPACED grid based on sample size
        # This matches TFPP and avoids numerical pathologies from extremely small gaps
        # in data-adaptive grids when n is large.
        #
        # Use n_bins = n_samples + 1 to keep parameter dimension comparable to
        # the legacy data-adaptive choice.
        n_bins = self.n_samples + 1
        self.knots = np.linspace(0.0, 1.0, n_bins + 1)  # n_bins+1 knots for n_bins bins
        self.bin_widths = np.diff(self.knots)  # all equal

        # Bin counts for binned likelihood
        counts, _ = np.histogram(y, bins=self.knots)
        
        # Create CVXPY variables
        # theta_full represents [θ₀, θ₁, ..., θₙ] but we'll constrain θ₀ = 0
        theta_full = cp.Variable(n_bins)
        
        # Constraint: θ₀ = 0 (identifiability)
        constraints = [theta_full[0] == 0]
        
        # Build difference matrix D^(k+1) for penalty
        # Note: penalty is on (k+1)-th differences as per methodology
        penalty_order = self.k + 1
        if penalty_order >= n_bins:
            # If penalty order is too high, reduce it
            penalty_order = max(1, n_bins - 1)
            
        D_matrix = build_difference_matrix(n_bins, penalty_order)
        # Convert to dense matrix for CVXPY compatibility
        D_matrix_dense = D_matrix.toarray()
        
        # Objective function components (binned likelihood)
        # nll(θ) = - Σ_j c_j θ_j + n log(Z(θ))
        # where Z(θ) = Σ_j exp(θ_j) Δx_j and c_j are histogram counts in each bin.
        first_term = -cp.sum(cp.multiply(counts, theta_full))
        
        # 2. Normalization term: n log(Z(θ))
        # Z(θ) = Σᵢ₌₀ⁿ exp(θᵢ) * Δxᵢ = exp(0)*Δx₀ + Σᵢ₌₁ⁿ exp(θᵢ)*Δxᵢ
        # Since θ₀ = 0, exp(θ₀) = 1, we can rewrite as:
        # log(Z) = log(Δx₀ + Σᵢ₌₁ⁿ exp(θᵢ)*Δxᵢ)
        # Use log_sum_exp for DCP compliance
        log_terms = cp.hstack([
            cp.log(self.bin_widths[0]),  # log(Δx₀) for the first bin where θ₀=0
            theta_full[1:] + cp.log(self.bin_widths[1:])  # θᵢ + log(Δxᵢ) for i=1,...,n
        ])
        log_Z = cp.log_sum_exp(log_terms)
        second_term = self.n_samples * log_Z
        
        # Complete objective (only the loss function, no penalty)
        loss = first_term + second_term
        objective = cp.Minimize(loss)
        
        # Constraints: θ₀ = 0 and ||D^(k+1)θ||₁ ≤ C
        constraints = [
            theta_full[0] == 0,  # Identifiability constraint
            cp.norm1(D_matrix_dense @ theta_full) <= self.norm_constraint  # Smoothness constraint
        ]
        
        # Create and solve problem
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=self.solver, verbose=False)
            self.optimization_status = problem.status
        except Exception as e:
            if not self.use_secondary_solver:
                raise RuntimeError(f"CVXPY optimization failed: {e}")
            
            print(f"{self.solver} solver failed with k={self.k}, constraint={self.norm_constraint}")
            print("Trying CLARABEL as secondary solver...")
            try:
                problem.solve(solver="CLARABEL", verbose=False)
                self.optimization_status = problem.status
                print("CLARABEL succeeded as secondary solver")
            except Exception as e2:
                print(f"CLARABEL also failed: {e2}")
                print("Falling back to ECOS...")
                try:
                    problem.solve(solver="ECOS", verbose=False)
                    self.optimization_status = problem.status
                    print("ECOS succeeded as tertiary solver")
                except Exception as e3:
                    print(f"ECOS also failed: {e3}")
                    print("Falling back to SCS...")
                    try:
                        problem.solve(solver="SCS", verbose=False)
                        self.optimization_status = problem.status
                        print("SCS succeeded as final solver")
                    except Exception as e4:
                        raise RuntimeError(f"All solvers failed. Last error: {e4}")
        
        # Check if optimization was successful
        if theta_full.value is None or problem.status not in ['optimal', 'optimal_inaccurate']:
            raise RuntimeError(f"CVXPY optimization failed with status: {problem.status}")
        
        # Store results
        self.optimized_theta_raw = theta_full.value.copy()
        self.theta_hat = theta_full.value.copy()
        
        self.is_fitted = True
        return self
    
    def get_density(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the estimated density on the evaluation grid.
        
        Returns:
            Tuple of (grid_points, density_values)
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting density. Call fit() first.")
        
        if self.theta_hat is None or self.knots is None or self.bin_widths is None:
            raise ValueError("Estimator not properly fitted - missing required attributes.")
        
        # Create evaluation grid
        grid_points = np.linspace(0.0, 1.0, self.n_grid_points)

        # Numerically-stable normalization:
        #   Z = sum_j exp(theta_j) * dx_j
        # Work in log-space to avoid overflow when theta is large.
        theta = np.asarray(self.theta_hat, dtype=float).ravel()
        dx = np.asarray(self.bin_widths, dtype=float).ravel()
        if theta.shape[0] != dx.shape[0]:
            raise ValueError(f"theta_hat/bin_widths length mismatch: {theta.shape[0]} vs {dx.shape[0]}")

        log_terms = theta + np.log(dx)
        m = float(np.max(log_terms))
        # logZ = m + log(sum(exp(log_terms - m)))
        logZ = m + float(np.log(np.sum(np.exp(log_terms - m))))
        log_density_levels = theta - logZ  # log f(x) within each bin

        # Find which bin each evaluation point belongs to
        bin_indices = np.searchsorted(self.knots, grid_points, side="right") - 1
        bin_indices = np.clip(bin_indices, 0, len(theta) - 1)

        # Assign density values (piecewise-constant)
        density = np.exp(log_density_levels[bin_indices])
        
        return grid_points, density
    
    def get_results(self) -> Dict:
        """
        Get comprehensive results from the fitting process.
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting results. Call fit() first.")
        
        if self.bin_widths is None:
            raise ValueError("Estimator not properly fitted - missing bin_widths.")
        
        grid_points, density = self.get_density()
        
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
            "optimization_status": self.optimization_status,
            "optimized_theta_raw": self.optimized_theta_raw,
        }