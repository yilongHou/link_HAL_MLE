import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from sklearn.isotonic import isotonic_regression
from typing import Dict, Tuple, Union
import scipy.sparse as sp

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

def soft_threshold(x, threshold):
    """
    Soft thresholding operator.
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def negative_log_likelihood(theta, counts, bin_widths, n_samples: int):
    """
    Negative log-likelihood on a binned grid.

    We parameterize a piecewise-constant log density on m bins with parameters
    theta_full = [0, theta_1, ..., theta_{m-1}] (theta_0 fixed for identifiability).

    With counts c_j in each bin and widths dx_j, the (negative) log-likelihood is:
        nll(theta) = - sum_{j=1}^{m-1} c_j * theta_j + n * log( sum_{j=0}^{m-1} exp(theta_j) * dx_j )
    """
    theta = np.asarray(theta, dtype=float).ravel()
    counts = np.asarray(counts, dtype=float).ravel()
    bin_widths = np.asarray(bin_widths, dtype=float).ravel()

    if counts.size != bin_widths.size:
        raise ValueError("counts and bin_widths must have same length.")
    if counts.size != theta.size + 1:
        raise ValueError("theta must have length n_bins-1 (theta_0 is fixed at 0).")

    # Full parameter vector is [0, theta_1, ..., theta_{m-1}]
    exp_theta_full = np.concatenate([[1.0], np.exp(theta)])
    Z = float(np.sum(exp_theta_full * bin_widths))
    if not np.isfinite(Z) or Z <= 0:
        raise RuntimeError(f"Invalid normalization constant Z={Z}")

    return -float(counts[1:] @ theta) + float(n_samples) * float(np.log(Z))

def nll_gradient(theta, counts, bin_widths, n_samples: int):
    """
    Gradient of negative log-likelihood.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    counts = np.asarray(counts, dtype=float).ravel()
    bin_widths = np.asarray(bin_widths, dtype=float).ravel()

    if counts.size != bin_widths.size:
        raise ValueError("counts and bin_widths must have same length.")
    if counts.size != theta.size + 1:
        raise ValueError("theta must have length n_bins-1 (theta_0 is fixed at 0).")

    exp_theta_full = np.concatenate([[1.0], np.exp(theta)])
    Z = float(np.sum(exp_theta_full * bin_widths))
    if not np.isfinite(Z) or Z <= 0:
        raise RuntimeError(f"Invalid normalization constant Z={Z}")
    
    # Gradient w.r.t. theta_1..theta_{m-1}
    grad = -counts[1:] + float(n_samples) * (np.exp(theta) * bin_widths[1:]) / Z
    return grad

class TrendFilteringADMMEstimator:
    """
    Trend Filtering density estimation using ADMM.
    """
    
    def __init__(self, k=1, lam=1.0, rho=None, max_iter=1000, 
                 eps_pri=1e-4, eps_dual=1e-4, verbose=False, n_grid_points=1000,
                 grid: str = "equal",
                 log_dir=None  # Not used in this class, but can be added for consistency
        ):
        self.k = k
        self.lam = lam
        self.rho = rho if rho is not None else lam
        self.max_iter = max_iter
        self.eps_pri = eps_pri
        self.eps_dual = eps_dual
        self.verbose = verbose
        self.n_grid_points = n_grid_points
        if grid not in {"equal", "data"}:
            raise ValueError("grid must be one of {'equal','data'}")
        self.grid = grid
        self.is_fitted = False
        
    def fit(self, data: pd.DataFrame) -> 'TrendFilteringADMMEstimator':
        """
        Fit the trend filtering density estimator.
        """
        y = data['W1'].values
        self.n_samples = int(len(y))

        if self.grid == "equal":
            # Equal-spaced knots (match TFPP methodology / improves numerical stability)
            # Use m = n_samples + 1 bins to keep parameter dimension comparable to data-knot TF.
            self.n_bins = self.n_samples + 1
            self.knots = np.linspace(0.0, 1.0, self.n_bins + 1)
            self.bin_widths = np.diff(self.knots)
            # Count observations per bin
            self.counts, _ = np.histogram(y, bins=self.knots)
        else:
            # Legacy behavior: data-adaptive knots at each sample
        y_sorted = np.sort(np.array(y))
            self.knots = np.concatenate([[0.0], y_sorted, [1.0]])
            self.bin_widths = np.diff(self.knots)
            # Under data-knots, each interior bin has exactly one observation by construction
            self.n_bins = self.n_samples + 1
            # counts length = n_bins, with counts[0] = 0 (since first knot is 0), counts[1:] = 1
            self.counts = np.ones(self.n_bins, dtype=int)
            self.counts[0] = 0

        # We optimize over (n_bins-1) parameters θ_1, ..., θ_{n_bins-1} with θ_0 = 0 fixed
        # The difference matrix operates on the full n_bins parameter vector [0, θ_1, ..., θ_{n_bins-1}]
        self.Dk = build_difference_matrix(self.n_bins, self.k)
        m = self.Dk.shape[0]
        
        theta = np.zeros(self.n_bins - 1)  # Only the free parameters θ_1, ..., θ_{m-1}
        alpha = np.zeros(m)
        u = np.zeros(m)
        
        eps_pri = self.eps_pri * np.sqrt(m)
        eps_dual = self.eps_dual * np.sqrt(self.n_bins - 1)
        
        for iteration in range(self.max_iter):
            alpha_old = alpha.copy()
            
            theta = self._theta_update(theta, alpha, u)
            
            # Create full parameter vector [0, θ_1, ..., θ_n] for difference operation
            theta_full = np.concatenate([[0], theta])
            v = self.Dk @ theta_full - u
            alpha = self._alpha_update(v)
            u = u + alpha - self.Dk @ theta_full
            
            r_norm, s_norm = self._compute_residuals(theta_full, alpha, alpha_old)
            
            if r_norm <= eps_pri and s_norm <= eps_dual:
                break
                
        self.theta_final = theta
        self.is_fitted = True
        return self
    
    def _theta_update(self, theta, alpha, u):
        def objective(theta_var):
            nll = negative_log_likelihood(theta_var, self.counts, self.bin_widths, self.n_samples)
            # Create full parameter vector [0, θ_1, ..., θ_n] for quadratic term
            theta_full = np.concatenate([[0], theta_var])
            residual = alpha - self.Dk @ theta_full + u
            quadratic = 0.5 * self.rho * np.sum(residual**2)
            return nll + quadratic
            
        def gradient(theta_var):
            nll_grad = nll_gradient(theta_var, self.counts, self.bin_widths, self.n_samples)
            # Create full parameter vector [0, θ_1, ..., θ_n] for quadratic term
            theta_full = np.concatenate([[0], theta_var])
            residual = alpha - self.Dk @ theta_full + u
            quad_grad = -self.rho * (self.Dk.T @ residual)[1:]  # Exclude gradient w.r.t. θ_0
            return nll_grad + quad_grad
        
        result = minimize(objective, theta, method='L-BFGS-B', jac=gradient,
                         options={'maxiter': 10, 'gtol': 1e-6})
        
        return result.x
    
    def _alpha_update(self, v):
        m = len(v)
        if m <= 1:
            return soft_threshold(v, self.lam / self.rho)

        cs = np.cumsum(np.concatenate(([0.0], v)))
        lower = isotonic_regression(cs - 1.0, y_min=None, y_max=None, increasing=True)
        upper = isotonic_regression(cs + 1.0, y_min=None, y_max=None, increasing=True)
        proj = 0.5 * (np.array(lower) + np.array(upper))
        alpha = np.diff(proj)
        return alpha
    
    def _compute_residuals(self, theta_full, alpha, alpha_old):
        r = alpha - self.Dk @ theta_full
        r_norm = np.linalg.norm(r)
        s = self.rho * (self.Dk.T @ (alpha - alpha_old))
        s_norm = np.linalg.norm(s)
        return r_norm, s_norm

    def get_density(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting density. Call fit() first.")
        
        # Create evaluation grid
        grid_points = np.linspace(0, 1, self.n_grid_points)
        
        # The theta_final contains log-density levels for the first n bins
        # The log-level for the first bin [0, y_1) is 0, and for bins [y_i, y_{i+1}) is theta_i
        exp_theta_full = np.concatenate([[1.0], np.exp(self.theta_final)])  # [exp(0), exp(theta_1), ..., exp(theta_n)]
        
        # Normalization constant: Z = ∫ exp(θ(x)) dx = Σᵢ₌₀^{n} exp(θᵢ) * Δxᵢ
        Z = np.sum(exp_theta_full * self.bin_widths)
        
        # Find which bin each evaluation point belongs to
        # Bins are: [x_0,x_1), [x_1,x_2), ..., [x_n,x_{n+1})
        # Use knots[1:] = [x_1, x_2, ..., x_{n+1}] as right endpoints
        bin_indices = np.searchsorted(self.knots[1:], grid_points, side='right')
        
        # Clip bin indices to valid range [0, n] (we have n+1 bins indexed 0 to n)
        bin_indices = np.clip(bin_indices, 0, len(exp_theta_full) - 1)
        
        # Assign densities for all points
        density = exp_theta_full[bin_indices] / Z
        
        # Verify normalization over the full support [0, 1]
        if self.verbose:
            integral = np.sum((exp_theta_full / Z) * self.bin_widths)
            print(f"Density defined over [0, 1]. Integral = {integral:.5f}")
            print(f"Number of bins: {len(self.bin_widths)}")
            print(f"exp_theta_full shape: {exp_theta_full.shape}")
            print(f"bin_indices range: [{bin_indices.min()}, {bin_indices.max()}]")
        
        return grid_points, density

    def get_results(self) -> Dict:
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting results. Call fit() first.")
        
        grid_points, density = self.get_density()
        
        return {
            "k": self.k,
            "lam": self.lam,
            "rho": self.rho,
            "theta_final": self.theta_final.tolist(),
            "grid_points": grid_points.tolist(),
            "estimated_density": density.tolist(),
            "n_selected_knots": len(self.knots),  # Total number of knots (n+2)
        }
