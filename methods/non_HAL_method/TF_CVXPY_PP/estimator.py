import numpy as np
import pandas as pd
import cvxpy as cp
from math import factorial
from typing import Dict, Optional, Tuple


def _first_difference_matrix(n: int) -> np.ndarray:
    """
    Build first-order difference matrix of size (n-1) x n.
    """
    if n <= 1:
        raise ValueError("n must be >= 2 to build a first difference matrix.")
    D1 = np.zeros((n - 1, n))
    for i in range(n - 1):
        D1[i, i] = -1.0
        D1[i, i + 1] = 1.0
    return D1


def _build_divided_difference_operators(
    x_positions: np.ndarray, order: int
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Build divided-difference operators for irregular grids.

    Returns:
        D_ops: dict with D_ops[i] = D^(i) (size (m-i) x m)
        Delta_ops: dict with Delta_ops[i] = diag entries for Delta^(i)
    """
    if order < 1:
        raise ValueError("order must be >= 1.")
    m = len(x_positions)
    if order >= m:
        raise ValueError("order must be <= m-1 to build divided differences.")

    D_ops: Dict[int, np.ndarray] = {}
    Delta_ops: Dict[int, np.ndarray] = {}

    D_ops[1] = _first_difference_matrix(m)
    for i in range(1, order):
        delta_i = x_positions[i:] - x_positions[:-i]
        if np.any(delta_i <= 0):
            raise ValueError("x_positions must be strictly increasing.")
        Delta_ops[i] = delta_i
        D1_i = _first_difference_matrix(m - i)
        D_ops[i + 1] = D1_i @ (D_ops[i] / delta_i[:, None])

    return D_ops, Delta_ops


def _falling_factorial_basis_matrix_parametric(
    *, x_eval: np.ndarray, x_nodes: np.ndarray, k: int
) -> np.ndarray:
    """
    Falling-factorial basis matrix H(x_eval) aligned with the Hinv built in
    TrendFilteringCVXPYParametricPenaltyEstimator.

    This matches the internal divided-difference / scaling conventions used to build Hinv
    (see construction of C and (1/k!) D^(k+1)). Numerically, it satisfies:
        H(x_nodes) @ (Hinv @ theta) ~= theta
    for k in {0,1,2}.
    """
    x_eval = np.asarray(x_eval, dtype=float).ravel()
    x_nodes = np.asarray(x_nodes, dtype=float).ravel()
    n = int(x_nodes.size)
    if n < 2:
        raise ValueError("Need at least 2 nodes for falling-factorial basis.")
    if not np.all(np.diff(x_nodes) > 0):
        raise ValueError("x_nodes must be strictly increasing for falling-factorial basis.")
    if k < 0:
        raise ValueError("k must be >= 0.")
    if k > n - 2:
        raise ValueError(f"k must satisfy k <= n-2, got k={k} with n={n}")

    H = np.zeros((x_eval.size, n), dtype=float)
    H[:, 0] = 1.0

    if k == 0:
        # h_{1+j}(x) = 1{x >= x_j}, j=1..n-1
        for j in range(1, n):
            H[:, j] = (x_eval >= x_nodes[j - 1]).astype(float)
        return H

    # Polynomial part: anchored products
    # Identifiability constraint theta[0]=0 removes the intercept direction in practice,
    # but Hinv is built with the full basis, so we keep the full polynomial block here.
    H[:, 1] = x_eval - x_nodes[0]
    if k >= 2:
        H[:, 2] = (x_eval - x_nodes[0]) * (x_eval - x_nodes[1])
    if k > 2:
        # Not expected in our experiments; implement if/when needed.
        raise ValueError("Falling-factorial evaluation currently implemented for k<=2.")

    # Truncated part: scaled by k! to match the (1/k!) in Hinv construction.
    scale = float(factorial(int(k)))
    for j in range(1, n - k):
        col = k + j
        prod = np.ones_like(x_eval, dtype=float)
        for ell in range(1, k + 1):
            prod *= (x_eval - x_nodes[j + ell - 1])
        prod *= (x_eval >= x_nodes[j + k - 1]).astype(float)
        H[:, col] = scale * prod

    return H


def _normalize_density_on_grid(grid_points: np.ndarray, log_density: np.ndarray) -> np.ndarray:
    grid_points = np.asarray(grid_points, dtype=float).ravel()
    log_density = np.asarray(log_density, dtype=float).ravel()
    if grid_points.size != log_density.size:
        raise ValueError("grid_points and log_density must have same length.")
    f = np.exp(log_density - np.max(log_density))
    Z = float(np.trapz(f, grid_points))
    if not np.isfinite(Z) or Z <= 0:
        raise RuntimeError(f"Non-finite or non-positive normalization constant Z={Z}")
    return f / Z


class TrendFilteringCVXPYParametricPenaltyEstimator:
    """
    Trend filtering density estimator using CVXPY with parametric-penalty TF.

    This solves:
        min_theta  -c^T theta + n logsumexp(theta + log(dx))
                    + lam * ||H_inv^(k) theta||_1
        s.t. theta[0] = 0
    """

    def __init__(
        self,
        k: int = 1,
        lam: Optional[float] = 1.0,
        norm_constraint: Optional[float] = None,
        n_grid_points: int = 1000,
        solver: str = "MOSEK",
        tol: float = 1e-6,
        use_secondary_solver: bool = False,
    ):
        self.k = k
        self.lam = lam
        self.norm_constraint = norm_constraint
        self.n_grid_points = n_grid_points
        self.solver = solver
        self.tol = tol
        self.use_secondary_solver = use_secondary_solver
        self.is_fitted = False

        self.theta_hat: Optional[np.ndarray] = None
        self.knots: Optional[np.ndarray] = None
        self.bin_widths: Optional[np.ndarray] = None
        self.n_samples: Optional[int] = None
        self.optimized_theta_raw: Optional[np.ndarray] = None
        self.optimization_status: Optional[str] = None
        self.hinv: Optional[np.ndarray] = None

    def fit(self, data: pd.DataFrame) -> "TrendFilteringCVXPYParametricPenaltyEstimator":
        y = data["W1"].values
        self.n_samples = len(y)

        # Grid construction: EQUALLY SPACED grid based on sample size
        # This ensures numerical stability of Ryan's divided difference operators
        m = self.n_samples + 1  # Number of bins (same count as data-adaptive)
        self.knots = np.linspace(0, 1, m + 1)  # m+1 knots for m bins
        self.bin_widths = np.diff(self.knots)  # All equal: 1/m

        if self.k >= m - 1:
            raise ValueError("k must be <= n_bins - 2 for parametric-penalty TF.")

        # Count observations per bin using histogram
        counts, _ = np.histogram(y, bins=self.knots)

        # Build divided-difference operators for density reconstruction
        # On equally-spaced grid, these are numerically stable!
        x_positions = self.knots[:-1]
        D_ops, Delta_ops = _build_divided_difference_operators(x_positions, self.k + 1)
        
        # Build C matrix using divided differences (for reconstruction)
        C_recon = np.zeros((self.k + 1, m))
        C_recon[0, 0] = 1.0
        for i in range(1, self.k + 1):
            delta_i = Delta_ops[i]
            Di_scaled = D_ops[i] / delta_i[:, None]
            C_recon[i, :] = (1.0 / factorial(i - 1)) * Di_scaled[0, :]
        
        # H_inv for density reconstruction (uses divided differences)
        self.hinv = np.vstack(
            [C_recon, (1.0 / factorial(self.k)) * D_ops[self.k + 1]]
        )
        
        # Build penalty operator using Ryan's divided differences
        # Now numerically stable on equally-spaced grid!
        penalty_op = self.hinv

        theta = cp.Variable(m)
        constraints = [theta[0] == 0]

        log_dx = np.log(self.bin_widths)
        logZ = cp.log_sum_exp(theta + log_dx)
        nll = -counts @ theta + self.n_samples * logZ
        # Use simple penalty operator for constraint (numerically stable)
        if self.norm_constraint is not None:
            constraints.append(cp.norm1(penalty_op @ theta) <= self.norm_constraint)
            problem = cp.Problem(cp.Minimize(nll), constraints)
        else:
            if self.lam is None:
                raise ValueError("Either norm_constraint or lam must be provided.")
            pen = self.lam * cp.norm1(penalty_op @ theta)
            problem = cp.Problem(cp.Minimize(nll + pen), constraints)

        try:
            if self.solver.upper() == "MOSEK":
                problem.solve(solver=self.solver, verbose=False)
            else:
                problem.solve(solver=self.solver, verbose=False, abstol=self.tol, reltol=self.tol)
            self.optimization_status = problem.status
        except Exception as e:
            if not self.use_secondary_solver:
                raise RuntimeError(f"CVXPY optimization failed: {e}")
            try:
                problem.solve(solver="CLARABEL", verbose=False)
                self.optimization_status = problem.status
            except Exception as e2:
                try:
                    problem.solve(solver="ECOS", verbose=False)
                    self.optimization_status = problem.status
                except Exception as e3:
                    try:
                        problem.solve(solver="SCS", verbose=False)
                        self.optimization_status = problem.status
                    except Exception as e4:
                        raise RuntimeError(f"All solvers failed. Last error: {e4}")

        if theta.value is None or problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"CVXPY optimization failed with status: {problem.status}")

        self.optimized_theta_raw = theta.value.copy()
        self.theta_hat = theta.value.copy()
        self.is_fitted = True
        return self

    def get_density(self, density_mode: str = "auto") -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting density. Call fit() first.")
        if self.theta_hat is None or self.knots is None or self.bin_widths is None:
            raise ValueError("Estimator not properly fitted - missing required attributes.")

        grid_points = np.linspace(0, 1, self.n_grid_points)
        if density_mode == "auto":
            # Use piecewise mode by default for numerical stability on irregular
            # data-based grids. The continuous/falling-factorial reconstruction
            # can be unstable when divided differences are computed on non-uniform
            # grids with small gaps.
            density_mode = "piecewise"

        if density_mode == "piecewise" or self.k == 0:
            exp_theta = np.exp(self.theta_hat)
            Z = np.sum(exp_theta * self.bin_widths)
            bin_indices = np.searchsorted(self.knots, grid_points, side="right") - 1
            bin_indices = np.clip(bin_indices, 0, len(self.theta_hat) - 1)
            density = exp_theta[bin_indices] / Z
            return grid_points, density

        if density_mode != "continuous":
            raise ValueError(f"Unknown density_mode={density_mode!r} (expected 'piecewise' or 'continuous').")
        if self.hinv is None:
            raise ValueError("Missing Hinv; cannot compute continuous TFPP density.")

        # Node grid aligned with Hinv construction: left endpoints x_0,...,x_n
        x_nodes = self.knots[:-1]
        alpha = self.hinv @ self.theta_hat
        H_eval = _falling_factorial_basis_matrix_parametric(
            x_eval=grid_points, x_nodes=x_nodes, k=int(self.k)
        )
        log_f = H_eval @ alpha
        density_cont = _normalize_density_on_grid(grid_points, log_f)
        return grid_points, density_cont

    def get_results(self) -> Dict:
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting results. Call fit() first.")
        grid_points, density_piecewise = self.get_density("piecewise")
        _, density_cont_default = self.get_density("auto")
        out = {
            "k": self.k,
            "lam": self.lam,
            "norm_constraint": self.norm_constraint,
            "theta_hat": self.theta_hat,
            "knots": self.knots,
            "bin_widths": self.bin_widths,
            "n_samples": self.n_samples,
            "n_bins": len(self.bin_widths) if self.bin_widths is not None else None,
            # Methodology update: `estimated_density` follows the default density mode
            # (piecewise for k=0; continuous for k>=1).
            "estimated_density": density_cont_default,
            "grid_points": grid_points,
            "density_mode_default": "piecewise" if self.k == 0 else "continuous",
            "optimization_status": self.optimization_status,
            "optimized_theta_raw": self.optimized_theta_raw,
            "Hinv": self.hinv,
            "estimated_density_piecewise": density_piecewise,
            "grid_points_piecewise": grid_points,
        }
        # Always also provide the continuous evaluation for k>=1.
        if self.k != 0:
            _, density_cont = self.get_density("continuous")
            out["estimated_density_continuous"] = density_cont
            out["grid_points_continuous"] = grid_points
        return out
import numpy as np
import pandas as pd
import cvxpy as cp
import scipy.sparse as sp
from math import factorial
from typing import Dict, Optional, Tuple


def build_difference_matrix(n: int, k: int):
    """
    Build k-th order (unweighted) difference matrix D^(k) of size (n-k) x n.
    This matches the legacy TF_CVXPY implementation for backward compatibility.
    """
    if k == 0:
        return sp.eye(n, format="csr")
    if k == 1:
        row = []
        col = []
        data = []
        for i in range(n - 1):
            row.extend([i, i])
            col.extend([i, i + 1])
            data.extend([-1.0, 1.0])
        return sp.csr_matrix((data, (row, col)), shape=(n - 1, n))
    D_prev = build_difference_matrix(n, k - 1)
    D1 = build_difference_matrix(D_prev.shape[0], 1)
    return D1 @ D_prev


def _build_first_diff(n_rows: int, n_cols: int):
    """Unweighted first-difference operator with shape (n_rows, n_cols)."""
    if n_cols != n_rows + 1:
        raise ValueError("First-difference matrix requires n_cols = n_rows + 1.")
    data = np.array([-1.0, 1.0] * n_rows, dtype=float)
    rows = np.repeat(np.arange(n_rows), 2)
    # Interleave columns as [0,1, 1,2, 2,3, ...] to match rows.
    cols = np.column_stack([np.arange(n_rows), np.arange(1, n_rows + 1)]).ravel()
    return sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))


def build_D_list_ryan(x: np.ndarray, k: int):
    """
    Build D^(m) for m=1..k+1 using Ryan Tibshirani's recursion (fallfact Lemma 2).
    D^(m+1) = D1_shrunk * m * (Delta^(m))^{-1} * D^(m)
    """
    x = np.asarray(x, dtype=float)
    n = int(len(x))
    if n < 2:
        raise ValueError("Grid must have at least 2 points.")

    D_list = [sp.csr_matrix((0, 0)) for _ in range(k + 2)]
    D_list[1] = _build_first_diff(n - 1, n)

    for m in range(1, k + 1):
        gaps = x[m:] - x[:-m]
        if np.any(gaps <= 0):
            raise ValueError("Grid must be strictly increasing for Ryan recursion.")
        Delta_inv = sp.diags(1.0 / gaps, format="csr")
        D1_shrunk = _build_first_diff(n - m - 1, n - m)
        D_list[m + 1] = D1_shrunk @ (m * (Delta_inv @ D_list[m]))

    return D_list


def build_C_block_ryan(x: np.ndarray, k: int, D_list):
    """
    Build C per fallfact Lemma 2:
      C_1 = e_1^T
      C_{i+1} = first row of (1/(i-1)! * (Delta^(i))^{-1} * D^(i)).
    """
    x = np.asarray(x, dtype=float)
    n = int(len(x))

    rows = []
    e1 = sp.csr_matrix(([1.0], ([0], [0])), shape=(1, n))
    rows.append(e1)

    for i in range(1, k + 1):
        gaps = x[i:] - x[:-i]
        if np.any(gaps <= 0):
            raise ValueError("Grid must be strictly increasing for Ryan recursion.")
        Delta_inv = sp.diags(1.0 / gaps, format="csr")
        Gi = (1.0 / factorial(i - 1)) * (Delta_inv @ D_list[i])
        rows.append(Gi[0, :])

    return sp.vstack(rows, format="csr").tocsr()


def _build_simple_C_block(n: int, k: int):
    """
    Build polynomial coefficient extraction matrix using simple unweighted differences.
    
    C = [e_1^T; D^1[0,:]; D^2[0,:]; ...; D^k[0,:]]
    
    This extracts polynomial coefficients without the numerical instability
    of divided differences on irregular grids.
    """
    rows = []
    rows.append(sp.csr_matrix(([1.0], ([0], [0])), shape=(1, n)))  # Intercept
    for i in range(1, k + 1):
        D_i = build_difference_matrix(n, i)
        rows.append(D_i[0, :])  # First row extracts i-th difference at start
    return sp.vstack(rows, format="csr").tocsr()


def build_extended_operator_ryan(
    x: np.ndarray,
    k: int,
    *,
    drop_intercept: bool = True,
    use_simple_operators: bool = True,
):
    """
    Build E = [C; D^(k+1)] for sectional variation norm penalty.
    
    Args:
        x: Grid points (typically data-based knots)
        k: Polynomial order
        drop_intercept: Whether to drop the intercept row from C
        use_simple_operators: If True (default), use simple unweighted difference
            operators which are numerically stable on irregular data-based grids.
            If False, use Ryan's divided difference operators (may be unstable
            on irregular grids).
    
    The simple operators avoid the numerical instability that occurs when using
    divided differences on irregular data-based grids, ensuring the sectional
    variation norm converges as n -> infinity for smooth densities.
    """
    x = np.asarray(x, dtype=float)
    n = int(len(x))
    if k < 0 or k > n - 2:
        raise ValueError("k must satisfy 0 <= k <= n-2 for Ryan operator.")

    if use_simple_operators:
        # Use simple unweighted differences - numerically stable on irregular grids
        C = _build_simple_C_block(n, k)
        Dk1 = build_difference_matrix(n, k + 1)
    else:
        # Original Ryan approach - may be numerically unstable on irregular grids
        D_list = build_D_list_ryan(x, k)
        Dk1 = D_list[k + 1]
        C = build_C_block_ryan(x, k, D_list)
        Dk1 = (1.0 / factorial(k)) * Dk1

    if drop_intercept:
        C = C.tocsr()[1:, :]

    E = sp.vstack([C, Dk1], format="csr").tocsr()
    return E


class TrendFilteringCVXPYPP:
    """
    Trend Filtering density estimator using CVXPY with an optional
    polynomial-part penalty (Ryan Tibshirani's extended discrete derivative).
    """

    def __init__(
        self,
        k: int = 1,
        norm_constraint: float = 1.0,
        n_grid_points: int = 1000,
        solver: str = "MOSEK",
        tol: float = 1e-6,
        use_secondary_solver: bool = False,
        parametric_penalty: bool = True,
        drop_intercept: bool = True,
        log_dir: Optional[str] = None,
    ):
        self.k = k
        self.norm_constraint = norm_constraint
        self.n_grid_points = n_grid_points
        self.solver = solver
        self.tol = tol
        self.use_secondary_solver = use_secondary_solver
        self.parametric_penalty = parametric_penalty
        self.drop_intercept = drop_intercept
        self.is_fitted = False

        self.theta_hat: Optional[np.ndarray] = None
        self.knots: Optional[np.ndarray] = None
        self.bin_widths: Optional[np.ndarray] = None
        self.n_samples: Optional[int] = None

        self.optimized_theta_raw: Optional[np.ndarray] = None
        self.optimization_status: Optional[str] = None

    def fit(self, data: pd.DataFrame) -> "TrendFilteringCVXPYPP":
        y = data["W1"].values
        self.n_samples = len(y)

        # Grid construction: EQUALLY SPACED grid based on sample size
        # This ensures numerical stability of Ryan's divided difference operators
        # Number of bins = n_samples + 1 (same as data-adaptive grid)
        n_bins = self.n_samples + 1
        self.knots = np.linspace(0, 1, n_bins + 1)  # n_bins + 1 knots for n_bins bins
        self.bin_widths = np.diff(self.knots)  # All equal: 1/n_bins

        # Count observations per bin using histogram
        counts, _ = np.histogram(y, bins=self.knots)
        
        theta_full = cp.Variable(n_bins)

        # Identifiability constraint
        constraints = [theta_full[0] == 0]

        penalty_order = self.k + 1
        if penalty_order >= n_bins:
            penalty_order = max(1, n_bins - 1)

        if not self.parametric_penalty:
            # Legacy penalty operator to preserve old behavior
            D_old = build_difference_matrix(n_bins, penalty_order)
            penalty_op = D_old
        else:
            # Ryan's operator on EQUALLY SPACED grid - now numerically stable!
            # Use right bin endpoints to define the grid for theta.
            x_penalty = self.knots[1:]
            penalty_op = build_extended_operator_ryan(
                x_penalty,
                k=int(self.k),
                drop_intercept=bool(self.drop_intercept),
                use_simple_operators=False,  # Use Ryan's original operators on regular grid
            )

        # Objective: negative log-likelihood using histogram counts
        # -sum_j c_j * theta_j + n * log(sum_j exp(theta_j) * dx_j)
        log_dx = np.log(self.bin_widths)
        first_term = -counts @ theta_full
        log_Z = cp.log_sum_exp(theta_full + log_dx)
        second_term = self.n_samples * log_Z
        loss = first_term + second_term
        objective = cp.Minimize(loss)

        constraints = [
            theta_full[0] == 0,
            cp.norm1(penalty_op @ theta_full) <= self.norm_constraint,
        ]

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

        if theta_full.value is None or problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"CVXPY optimization failed with status: {problem.status}")

        self.optimized_theta_raw = theta_full.value.copy()
        self.theta_hat = theta_full.value.copy()
        self.is_fitted = True
        return self

    def get_density(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting density. Call fit() first.")

        if self.theta_hat is None or self.knots is None or self.bin_widths is None:
            raise ValueError("Estimator not properly fitted - missing required attributes.")

        grid_points = np.linspace(0.0, 1.0, self.n_grid_points)

        # Numerically-stable piecewise density computation in log-space:
        #   f(x) = exp(theta_bin) / Z,  Z = sum_j exp(theta_j) * dx_j
        theta = np.asarray(self.theta_hat, dtype=float).ravel()
        dx = np.asarray(self.bin_widths, dtype=float).ravel()
        if theta.shape[0] != dx.shape[0]:
            raise ValueError(f"theta_hat/bin_widths length mismatch: {theta.shape[0]} vs {dx.shape[0]}")

        log_terms = theta + np.log(dx)
        m = float(np.max(log_terms))
        logZ = m + float(np.log(np.sum(np.exp(log_terms - m))))
        log_density_levels = theta - logZ

        bin_indices = np.searchsorted(self.knots, grid_points, side="right") - 1
        bin_indices = np.clip(bin_indices, 0, len(theta) - 1)
        density = np.exp(log_density_levels[bin_indices])

        return grid_points, density

    def get_results(self) -> Dict:
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting results. Call fit() first.")

        if self.bin_widths is None:
            raise ValueError("Estimator not properly fitted - missing bin_widths.")

        grid_points, density = self.get_density()
        return {
            "k": self.k,
            "norm_constraint": self.norm_constraint,
            "parametric_penalty": self.parametric_penalty,
            "drop_intercept": self.drop_intercept,
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

