import numpy as np
import pandas as pd
import torch
from typing import Union, Tuple


def create_cumulative_indicators_ge(
    series: pd.Series, 
    grid_points: np.ndarray
) -> pd.DataFrame:
    """
    Create cumulative greater-or-equal indicator columns for a pandas Series.
    This is for order=0 case only: {1, I(x >= ξ₁), I(x >= ξ₂), ..., I(x >= ξₘ)}
    Args:
        series: Input pandas Series
        grid_points: Grid points for which to create indicators
        
    Returns:
        DataFrame with indicator columns
    """
    # Use vectorized operations for efficiency
    x_values = series.values[:, np.newaxis]  # Shape: (n, 1)
    grid_values = grid_points[np.newaxis, :]  # Shape: (1, m)
    
    # Vectorized comparison: (n, m) boolean array
    indicators_array = (x_values >= grid_values).astype(np.float32)
    
    # Create column names
    col_names = [f"ge_{value:.6f}" for value in grid_points]
    
    return pd.DataFrame(indicators_array, columns=col_names, index=series.index)


def create_basis_functions(
    data_long_train: pd.DataFrame, 
    grid_points: np.ndarray, 
    order: int = 1,
    include_intercept: bool = True
) -> torch.Tensor:
    """
    Create truncated power basis functions for the data.
    
    For order=0 (Schumaker convention):
        {1, I(x >= ξ₁), I(x >= ξ₂), ..., I(x >= ξₘ)}
    
    For order≥1:
        {1, x, x², ..., x^k, (x-ξ₁)₊^k, (x-ξ₂)₊^k, ..., (x-ξₘ)₊^k}
        where (x-ξⱼ)₊^k = max(x-ξⱼ, 0)^k
    
    Args:
        data_long_train: DataFrame containing the data (uses 'W1' column)
        grid_points: Knot points ξ₁, ξ₂, ..., ξₘ for truncated power functions
        order: Polynomial and spline order (k)
        
    Returns:
        Torch tensor of shape (n, p) where p = (k+1) + m for order≥1, or 1+m for order=0
    """
    x = data_long_train['W1'].values  # Shape: (n,)
    n = len(x)
    m = len(grid_points)
    
    if order == 0:
        # Special case: use indicators with >= comparison
        # Basis: {1, I(x >= ξ₁), I(x >= ξ₂), ..., I(x >= ξₘ)}
        basis_list = []
        basis_names = []
        
        # Intercept term
        if include_intercept:
            # Add intercept term (constant 1)
            basis_list.append(np.ones(n, dtype=np.float32))
            basis_names.append("Intercept")
        
        # Indicator functions using vectorized operations
        x_broadcast = x[:, np.newaxis]  # Shape: (n, 1)
        grid_broadcast = grid_points[np.newaxis, :]  # Shape: (1, m)
        indicators = (x_broadcast >= grid_broadcast).astype(np.float32)  # Shape: (n, m)
        
        for j in range(m):
            basis_list.append(indicators[:, j])
            basis_names.append(f"I(x >= {grid_points[j]:.6f})")
        
        basis_array = np.column_stack(basis_list)  # Shape: (n, 1+m)
        if include_intercept:
            assert basis_array.shape[1] == (1 + m), f"Basis shape mismatch for order=0: {basis_array.shape[1]} != {1 + m}"
        else:
            assert basis_array.shape[1] == m, f"Basis shape mismatch for order=0: {basis_array.shape[1]} != {m}"
        assert basis_array.shape[0] == n, f"Basis shape mismatch for order=0: {basis_array.shape[0]} != {n}"
        assert len(basis_names) == basis_array.shape[1], "Basis names count does not match basis array columns"

    else:
        # Order k ≥ 1: polynomial terms + truncated power terms
        # Basis: {1, x, x², ..., x^k, (x-ξ₁)₊^k, (x-ξ₂)₊^k, ..., (x-ξₘ)₊^k}
        basis_list = []
        basis_names = []
        
        # Polynomial terms: 1, x, x², ..., x^k
        k = order
        for power in range(order + 1):
            if power == 0:
                # Intercept term (constant 1)
                if include_intercept:
                    basis_list.append(np.ones(n, dtype=np.float32))
                    basis_names.append("Intercept")
                else:
                    pass
            else:
                basis_list.append((x ** power).astype(np.float32))
                basis_names.append(f"x^{power}")
        
        # Truncated power terms: (x-ξⱼ)₊^k for each knot ξⱼ
        for knot in grid_points:
            truncated_power = np.maximum(x - knot, 0) ** order
            basis_list.append(truncated_power.astype(np.float32))
            basis_names.append(f"(x - {knot:.6f})_+^{order}")

        basis_array = np.column_stack(basis_list)  # Shape: (n, (k+1)+m)
        if include_intercept:
            assert basis_array.shape[1] == (order + 1 + m), f"Basis shape mismatch for order={order}: {basis_array.shape[1]} != {order + 1 + m}"
        else:
            assert basis_array.shape[1] == (order + m), f"Basis shape mismatch for order={order}: {basis_array.shape[1]} != {order + m}"
        assert basis_array.shape[0] == n, f"Basis shape mismatch for order={order}: {basis_array.shape[0]} != {n}"
        assert len(basis_names) == basis_array.shape[1], "Basis names count does not match basis array columns"
    
    # Convert to torch tensor
    basis_tensor = torch.tensor(basis_array, dtype=torch.float32)
    return basis_tensor, basis_names


def project_onto_l1_ball(v: torch.Tensor, z: float = 8.0) -> torch.Tensor:
    """
    Project a torch tensor v onto the L1 ball of radius z.
    
    Args:
        v: Input tensor to project
        z: L1 ball radius
        
    Returns:
        Projected tensor
    """
    if v.abs().sum() <= z:
        return v
    # Sort the absolute values in descending order
    u, _ = torch.sort(v.abs(), descending=True)
    sv = torch.cumsum(u, dim=0)
    # Create tensor for indices 1,...,n
    rho = torch.nonzero(u - (sv - z) / torch.arange(1, len(u)+1, device=v.device, dtype=v.dtype))
    if len(rho) == 0:
        tau = 0.0
    else:
        rho = (rho[-1, 0] + 1).float()  # last index (1-indexed)
        tau = (sv[int(rho.item()) - 1] - z) / rho
    return torch.sign(v) * torch.clamp(v.abs() - tau, min=0)


def hessian(phi: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Compute the Hessian of the log-likelihood at beta.

    Args:
        phi: n×p array, each row phi[i]=phi(X_i)
        beta: length-p vector
        
    Returns:
        p×p Hessian matrix
    """
    # 1) compute linear predictors and weights
    eta = phi.dot(beta)                      # shape (n,)
    w_unnorm = np.exp(eta)
    w = w_unnorm / w_unnorm.sum()            # shape (n,)

    # 2) weighted mean of phi
    E_phi = w @ phi                        # shape (p,)

    # 3) compute weighted covariance
    #    Cov = sum_i w_i * (phi[i] - E_phi) ⊗ (phi[i] - E_phi)
    centered = phi - E_phi                 # shape (n,p)
    cov = (centered * w[:,None]).T @ centered  # shape (p,p)

    # 4) the Hessian of the log-lik is -n * cov
    n = phi.shape[0]
    H = -n * cov
    return H


def kappa(M: np.ndarray, tol: float = 0.0) -> float:
    """
    Compute κ(M) = λ_max(M^T M) / λ_min(M^T M),
    but return np.inf if λ_min < tol (e.g. < 0 due to numerical noise).

    Args:
        M: array, shape (n, p)
        tol: minimum allowed eigenvalue (default 0.0)
        
    Returns:
        cond: the condition number or np.inf
    """
    # 1) form the symmetric matrix
    MtM = M.T @ M

    # 2) get its eigenvalues (fast for symmetric)
    eigs = np.linalg.eigvalsh(MtM)
    lam_min, lam_max = eigs[0], eigs[-1]

    # 3) guard against small/negative λ_min
    if lam_min < tol:
        return np.inf

    return lam_max / lam_min
