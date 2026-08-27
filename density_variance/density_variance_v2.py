import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
from typing import Dict, Tuple, Optional
import warnings

from utils.basis import create_basis_functions


def compute_fisher_information_matrix(expected_phi: np.ndarray, 
                                    phi_integration_selected: np.ndarray,
                                    density_integration: np.ndarray,
                                    dx: float) -> np.ndarray:
    """
    Compute the theoretical Fisher information matrix I_{f_n}(φ) = E[φ(X)φ(X)^T] - E[φ(X)]E[φ(X)]^T
    
    Args:
        expected_phi: E[φ(X)] of shape (K,)
        phi_integration_selected: Basis functions on integration grid of shape (n_grid, K)
        density_integration: Normalized density values on integration grid of shape (n_grid,)
        dx: Grid spacing for numerical integration
        
    Returns:
        Fisher information matrix of shape (K, K)
    """
    # Compute E[φ(X)φ(X)^T] = ∫ φ(x)φ(x)^T f(x; β̂) dx
    K = phi_integration_selected.shape[1]
    second_moment = np.zeros((K, K))
    
    for i in range(K):
        for j in range(K):
            # E[φ_i(X) φ_j(X)] = ∫ φ_i(x) φ_j(x) f(x; β̂) dx
            integrand = phi_integration_selected[:, i] * phi_integration_selected[:, j] * density_integration
            second_moment[i, j] = np.trapz(integrand, dx=dx)
    
    # Compute Fisher information: I = E[φ(X)φ(X)^T] - E[φ(X)]E[φ(X)]^T
    first_moment_outer = np.outer(expected_phi, expected_phi)
    fisher_information = second_moment - first_moment_outer
    # Enforce symmetry to reduce numerical asymmetry
    fisher_information = 0.5 * (fisher_information + fisher_information.T)
    
    return fisher_information


def estimate_variance(data: pd.DataFrame, estimation_results: Dict, 
                        ridge_param: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate variance components using V2 methodology.
    
    Following methodology.mmd:
    σ_n²(x) = (1/n) (p_{f_n}(x))² φ(x)^T I_{f_n}^{-1} P_n{S_{f_n}(φ)S_{f_n}(φ)^T} I_{f_n}^{-1} φ(x)
    
    Where:
    - I_{f_n} = E[φ(X)φ(X)^T] - E[φ(X)]E[φ(X)]^T (theoretical Fisher information)
    - P_n{S_{f_n}(φ)S_{f_n}(φ)^T} = (1/n) Σᵢ S(xᵢ) S(xᵢ)^T (observed Fisher information)
    - S_{f_n}(φ) = φ(x) - E[φ(X)] (score function)
    
    Args:
        data: Original training data DataFrame with 'W1' column
        estimation_results: Dictionary containing fitted model results
        ridge_param: Ridge regularization parameter for numerical stability
        
    Returns:
        Tuple of (fisher_information_inv, observed_information, expected_phi)
    """
    # Extract fitted parameters
    theta_hat_full = np.array(estimation_results['results']['theta_hat'])
    
    # Detect basis order from the data structure
    basis_order = detect_basis_order(estimation_results)
    
    # Get the original grid points used for basis construction
    original_data_points = np.array(estimation_results['results']['data_points'])
    grid_points_hal_original = np.unique(original_data_points)
    
    # Select the coefficients corresponding to selected basis functions
    theta_selected_ind = np.concatenate([
        np.arange(0, basis_order + 1),
        np.array(theta_hat_full[basis_order+1:]).nonzero()[0] + basis_order + 1
    ])
    theta_hat = theta_hat_full[theta_selected_ind]
    
    n_samples = len(data)
    
    # Create basis matrix for the data using the original grid points
    basis_matrix, _ = create_basis_functions(
        data, 
        grid_points_hal_original, 
        order=basis_order,
        include_intercept=True
    )
    basis_matrix = basis_matrix.numpy()
    basis_matrix_selected = basis_matrix[:, theta_selected_ind]
    
    # Compute E[φ(X)] by numerical integration
    n_grid_points = len(estimation_results.get('grid_points', np.linspace(0, 1, 200)))
    integration_grid = np.linspace(0, 1, n_grid_points)
    integration_df = pd.DataFrame({'W1': integration_grid})
    
    # Create basis matrix for integration grid
    phi_integration, _ = create_basis_functions(
        integration_df,
        grid_points_hal_original,
        order=basis_order,
        include_intercept=True
    )
    phi_integration = phi_integration.numpy()
    phi_integration_selected = phi_integration[:, theta_selected_ind]
    
    # Compute density values at integration points
    log_density_integration = phi_integration_selected @ theta_hat
    max_log_density = np.max(log_density_integration)
    density_unnorm = np.exp(log_density_integration - max_log_density)
    
    # Normalize using trapezoidal rule
    dx = integration_grid[1] - integration_grid[0]
    normalizing_constant = np.trapz(density_unnorm, dx=dx)
    density_integration = density_unnorm / normalizing_constant
    
    # Compute E[φ(X)] = ∫ φ(x) f(x; β̂) dx using trapezoidal rule
    expected_phi = np.trapz(
        phi_integration_selected * density_integration[:, np.newaxis], 
        dx=dx, 
        axis=0
    )
    
    # Compute theoretical Fisher information matrix I_{f_n}
    fisher_information = compute_fisher_information_matrix(
        expected_phi, phi_integration_selected, density_integration, dx
    )
    
    # Add ridge regularization to Fisher information matrix
    ridge_matrix = ridge_param * np.eye(fisher_information.shape[0])
    regularized_fisher_information = fisher_information + ridge_matrix
    # Also enforce symmetry on the regularized matrix
    regularized_fisher_information = 0.5 * (regularized_fisher_information + regularized_fisher_information.T)
    
    # Compute inverse of regularized Fisher information
    try:
        fisher_information_inv = np.linalg.inv(regularized_fisher_information)
    except np.linalg.LinAlgError:
        warnings.warn("Fisher information matrix is singular. Using pseudo-inverse.")
        fisher_information_inv = np.linalg.pinv(regularized_fisher_information)
    
    # Compute score for each observation: S(xᵢ) = φ(xᵢ) - E[φ(X)]
    scores = basis_matrix_selected - expected_phi[np.newaxis, :]
    
    # Compute observed Fisher information: P_n{S(φ)S(φ)^T} = (1/n) Σᵢ S(xᵢ) S(xᵢ)^T
    observed_information = (scores.T @ scores) / n_samples
    # Enforce symmetry
    observed_information = 0.5 * (observed_information + observed_information.T)
    
    return fisher_information_inv, observed_information, expected_phi


def density_variance(x_values: np.ndarray, estimation_results: Dict, 
                       fisher_information_inv: np.ndarray, observed_information: np.ndarray,
                       expected_phi: np.ndarray) -> np.ndarray:
    """
    Compute density variance using V2 methodology.
    
    σ_n²(x) = (1/n) (p_{f_n}(x))² φ(x)^T I_{f_n}^{-1} P_n{S_{f_n}(φ)S_{f_n}(φ)^T} I_{f_n}^{-1} φ(x)
    
    Args:
        x_values: Points where to evaluate variance
        estimation_results: Dictionary containing fitted model results
        fisher_information_inv: Inverse of theoretical Fisher information matrix
        observed_information: Observed Fisher information matrix
        expected_phi: E[φ(X)]
        
    Returns:
        Variance estimates at x_values
    """
    # Extract fitted parameters
    theta_hat_full = np.array(estimation_results['results']['theta_hat'])
    basis_order = detect_basis_order(estimation_results)
    original_data_points = np.array(estimation_results['results']['data_points'])
    grid_points_hal_original = np.unique(original_data_points)
    
    # Select coefficients
    theta_selected_ind = np.concatenate([
        np.arange(0, basis_order + 1),
        np.array(theta_hat_full[basis_order+1:]).nonzero()[0] + basis_order + 1
    ])
    theta_hat = theta_hat_full[theta_selected_ind]
    
    # Ensure x_values is within [0, 1]
    x_values = np.clip(x_values, 0, 1)
    
    # Create basis matrix for evaluation points
    eval_df = pd.DataFrame({'W1': x_values})
    phi_eval, _ = create_basis_functions(
        eval_df,
        grid_points_hal_original, 
        order=basis_order,
        include_intercept=True
    )
    phi_eval = phi_eval.numpy()
    phi_eval_selected = phi_eval[:, theta_selected_ind]
    
    # Compute density values at evaluation points
    log_density_eval = phi_eval_selected @ theta_hat
    
    # Normalize density (same normalization as in integration)
    n_grid_points = len(estimation_results.get('grid_points', np.linspace(0, 1, 200)))
    integration_grid = np.linspace(0, 1, n_grid_points)
    integration_df = pd.DataFrame({'W1': integration_grid})
    
    phi_integration, _ = create_basis_functions(
        integration_df,
        grid_points_hal_original,
        order=basis_order,
        include_intercept=True
    )
    phi_integration = phi_integration.numpy()
    phi_integration_selected = phi_integration[:, theta_selected_ind]
    
    log_density_integration = phi_integration_selected @ theta_hat
    max_log_density = np.max(log_density_integration)
    density_unnorm = np.exp(log_density_integration - max_log_density)
    
    dx = integration_grid[1] - integration_grid[0]
    normalizing_constant = np.trapz(density_unnorm, dx=dx)
    log_Z = np.log(normalizing_constant) + max_log_density
    
    density_eval = np.exp(log_density_eval - log_Z)
    
    # Compute variance for each evaluation point
    n_eval = len(x_values)
    variances = np.zeros(n_eval)
    # Precompute the sandwich matrix once: I_{f_n}^{-1} P_n{SS^T} I_{f_n}^{-1}
    temp = fisher_information_inv @ observed_information @ fisher_information_inv
    
    for i in range(n_eval):
        # Get φ(x) for this evaluation point (per methodology_v2, not centered)
        phi_x = phi_eval_selected[i]
        
        # Compute quadratic form: φ(x)^T I_{f_n}^{-1} P_n{S(φ)S(φ)^T} I_{f_n}^{-1} φ(x)
        quadratic_form = phi_x.T @ temp @ phi_x
        
        # Final variance: σ_n²(x) = (1/n) (p_{f_n}(x))² × quadratic_form
        variances[i] = (density_eval[i] ** 2) * quadratic_form / len(estimation_results['results']['data_points'])
    
    return variances


def density_confidence_interval(x_values: np.ndarray, estimation_results: Dict, 
                               cov_beta: Optional[np.ndarray] = None, alpha: float = 0.05, 
                               ridge_param: float = 1e-6) -> pd.DataFrame:
    """
    Compute confidence intervals for density estimates using V2 methodology.
    
    Args:
        x_values: Points where to evaluate confidence intervals
        estimation_results: Dictionary containing fitted model results  
        cov_beta: Covariance matrix (for V1 compatibility - ignored in V2)
        alpha: Significance level (default 0.05 for 95% CI)
        ridge_param: Ridge regularization parameter
        
    Returns:
        DataFrame with columns: x, density, se, lower, upper
    """
    # Compute variance components using V2 methodology
    fisher_information_inv, observed_information, expected_phi = estimate_variance(
        pd.DataFrame({'W1': estimation_results['results']['data_points']}), 
        estimation_results, 
        ridge_param
    )
    
    # Extract fitted parameters for density evaluation
    theta_hat_full = np.array(estimation_results['results']['theta_hat'])
    basis_order = detect_basis_order(estimation_results)
    original_data_points = np.array(estimation_results['results']['data_points'])
    grid_points_hal_original = np.unique(original_data_points)
    
    theta_selected_ind = np.concatenate([
        np.arange(0, basis_order + 1),
        np.array(theta_hat_full[basis_order+1:]).nonzero()[0] + basis_order + 1
    ])
    theta_hat = theta_hat_full[theta_selected_ind]
    
    # Ensure x_values is within [0, 1]
    x_values = np.clip(x_values, 0, 1)
    
    # Create basis matrix for evaluation points
    eval_df = pd.DataFrame({'W1': x_values})
    phi_eval, _ = create_basis_functions(
        eval_df,
        grid_points_hal_original, 
        order=basis_order,
        include_intercept=True
    )
    phi_eval = phi_eval.numpy()
    phi_eval_selected = phi_eval[:, theta_selected_ind]
    
    # Compute normalized density values
    log_density_eval = phi_eval_selected @ theta_hat
    
    # Get normalization constant
    n_grid_points = len(estimation_results.get('grid_points', np.linspace(0, 1, 200)))
    integration_grid = np.linspace(0, 1, n_grid_points)
    integration_df = pd.DataFrame({'W1': integration_grid})
    
    phi_integration, _ = create_basis_functions(
        integration_df,
        grid_points_hal_original,
        order=basis_order,
        include_intercept=True
    )
    phi_integration = phi_integration.numpy()
    phi_integration_selected = phi_integration[:, theta_selected_ind]
    
    log_density_integration = phi_integration_selected @ theta_hat
    max_log_density = np.max(log_density_integration)
    density_unnorm = np.exp(log_density_integration - max_log_density)
    
    dx = integration_grid[1] - integration_grid[0]
    normalizing_constant = np.trapz(density_unnorm, dx=dx)
    log_Z = np.log(normalizing_constant) + max_log_density
    
    density_eval = np.exp(log_density_eval - log_Z)
    
    # Compute variances using V2 methodology
    variances = density_variance(x_values, estimation_results, 
                                   fisher_information_inv, observed_information, expected_phi)
    
    # Handle potential negative variance due to numerical errors
    variances = np.maximum(variances, 0)
    standard_errors = np.sqrt(variances)
    # Apply standard error correction by sample size
    n_samples = len(estimation_results['results']['data_points'])
    standard_errors = standard_errors / np.sqrt(n_samples)
    
    # Critical value for confidence interval
    z_critical = norm.ppf(1 - alpha/2)
    
    # Compute confidence intervals
    margin_of_error = z_critical * standard_errors
    lower_bounds = np.maximum(density_eval - margin_of_error, 0)  # Truncate at 0
    upper_bounds = density_eval + margin_of_error
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'x': x_values,
        'density': density_eval,
        'se': standard_errors,
        'lower': lower_bounds,
        'upper': upper_bounds
    })
    
    return result_df


def detect_basis_order(estimation_results: Dict) -> int:
    """
    Detect the basis order used in the estimation.
    
    Args:
        estimation_results: Dictionary containing fitted model results
        
    Returns:
        Detected basis order
    """
    # First try to extract from estimator_setup (preferred method)
    if 'estimator_setup' in estimation_results:
        estimator_params = estimation_results['estimator_setup'].get('estimator_params', {})
        if 'basis_order' in estimator_params:
            return estimator_params['basis_order']
    
    # If not found in estimator_setup, try to extract from results structure
    if 'results' in estimation_results and 'estimator_setup' in estimation_results:
        estimator_params = estimation_results['estimator_setup'].get('estimator_params', {})
        if 'basis_order' in estimator_params:
            return estimator_params['basis_order']
    
    # Last resort: try to infer from structure (this should rarely be needed)
    if 'results' in estimation_results:
        results = estimation_results['results']
        if 'theta_hat' in results and 'grid_points_hal_selected' in results:
            theta_hat = np.array(results['theta_hat'])
            grid_points_hal_selected = np.array(results['grid_points_hal_selected'])
            
            K = len(theta_hat)
            m = len(grid_points_hal_selected)
            
            # For order 0: K = 1 + m (intercept + indicators)
            # For order k≥1: K = (k+1) + m (intercept + polynomials + truncated powers)
            if K == 1 + m:
                return 0
            else:
                # For higher order, solve: K = (k+1) + m
                order = K - m - 1
                if order >= 1:
                    return order
    
    # Final fallback
    return 0


def compute_density_variance_summary(data: pd.DataFrame, estimation_results: Dict,
                                       x_values: Optional[np.ndarray] = None,
                                       alpha: float = 0.05,
                                       ridge_param: float = 1e-6) -> Dict:
    """
    Compute a comprehensive summary of density variance analysis using V2 methodology.
    
    Args:
        data: Original training data
        estimation_results: Dictionary containing fitted model results
        x_values: Points for evaluation (default: 100 points from 0 to 1)
        alpha: Significance level for confidence intervals
        ridge_param: Ridge regularization parameter
        
    Returns:
        Dictionary containing analysis summary and confidence intervals
    """
    if x_values is None:
        x_values = np.linspace(0.01, 0.99, 100)
    
    # Step 1: Estimate variance components using V2 methodology
    fisher_information_inv, observed_information, expected_phi = estimate_variance(
        data, estimation_results, ridge_param
    )
    
    # Step 2: Compute confidence intervals (ensure correct argument order via keywords)
    ci_df = density_confidence_interval(
        x_values=x_values,
        estimation_results=estimation_results,
        cov_beta=None,
        alpha=alpha,
        ridge_param=ridge_param,
    )
    
    # Step 3: Compute summary statistics
    basis_order = detect_basis_order(estimation_results)
    n_selected_coeffs = len(estimation_results['results']['theta_hat'])
    
    # Condition numbers for both matrices (cond(inv(A)) == cond(A) for 2-norm, but keep as a simple diagnostic)
    fisher_condition_number = np.linalg.cond(fisher_information_inv)
    observed_condition_number = np.linalg.cond(observed_information)
    
    avg_std_error = ci_df['se'].mean()
    avg_interval_width = (ci_df['upper'] - ci_df['lower']).mean()
    
    summary = {
        'basis_order': basis_order,
        'n_selected_coeffs': n_selected_coeffs,
        'fisher_condition_number': fisher_condition_number,
        'observed_condition_number': observed_condition_number,
        'avg_std_error': avg_std_error,
        'avg_interval_width': avg_interval_width,
        'confidence_intervals': ci_df,
        'fisher_information_inv': fisher_information_inv,
        'observed_information': observed_information,
        'expected_phi': expected_phi
    }
    
    return summary


# =============================================================================
# COMPATIBILITY FUNCTIONS - Maintain same interface as V1 for existing notebooks
# =============================================================================

def estimate_covariance_beta(data: pd.DataFrame, estimation_results: Dict, 
                           ridge_param: float = 1e-6) -> np.ndarray:
    """
    Compatibility wrapper for estimate_variance_v2.
    Returns a covariance matrix that can be used with the V2 variance formula.
    
    Note: In V2 methodology, this doesn't directly return Cov(β) like in V1.
    Instead, it returns the sandwich covariance matrix component that's used 
    in the V2 variance formula: I_{f_n}^{-1} P_n{S(φ)S(φ)^T} I_{f_n}^{-1}
    
    Args:
        data: Original training data DataFrame with 'W1' column
        estimation_results: Dictionary containing fitted model results
        ridge_param: Ridge regularization parameter for numerical stability
        
    Returns:
        Sandwich covariance matrix component for V2 variance calculation
    """
    fisher_information_inv, observed_information, expected_phi = estimate_variance(
        data, estimation_results, ridge_param
    )
    
    # Compute the sandwich covariance matrix: I_{f_n}^{-1} P_n{S(φ)S(φ)^T} I_{f_n}^{-1}
    sandwich_cov = fisher_information_inv @ observed_information @ fisher_information_inv
    
    return sandwich_cov



