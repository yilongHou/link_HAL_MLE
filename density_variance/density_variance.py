import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
from typing import Dict, Tuple, Optional
import warnings

from utils.basis import create_basis_functions


def estimate_covariance_beta(data: pd.DataFrame, estimation_results: Dict, 
                           ridge_param: float = 1e-6) -> np.ndarray:
    """
    Estimate the covariance matrix of beta using the observed Fisher information matrix.
    
    Following methodology.md:
    The observed information is estimated as I(β̂) = (1/n) Σᵢ S(Oᵢ;β̂) S(Oᵢ;β̂)ᵀ
    With ridge regularization: I_reg(β̂) = I(β̂) + λ I
    Then Cov(β̂) = (1/n) I_reg(β̂)⁻¹
    
    Args:
        data: Original training data DataFrame with 'W1' column
        estimation_results: Dictionary containing fitted model results
        ridge_param: Ridge regularization parameter for numerical stability
        
    Returns:
        Covariance matrix of shape (K, K) where K is number of coefficients
    """
    # Extract fitted parameters
    theta_hat_full = np.array(estimation_results['results']['theta_hat'])
    grid_points_hal_selected = np.array(estimation_results['results']['grid_points_hal_selected'])
    
    # Detect basis order from the data structure
    basis_order = detect_basis_order(estimation_results)
    
    # Get the original grid points used for basis construction
    # These should be the unique data points used during training
    original_data_points = np.array(estimation_results['results']['data_points'])
    grid_points_hal_original = np.unique(original_data_points)
    
    # Select the coefficients corresponding to selected basis functions
    # For basis order k: first k+1 coefficients are polynomial terms (always selected)
    # Then select non-zero coefficients from the truncated power terms
    theta_selected_ind = np.concatenate([
        np.arange(0, basis_order + 1),
        np.array(theta_hat_full[basis_order+1:]).nonzero()[0] + basis_order + 1
    ])
    theta_hat = theta_hat_full[theta_selected_ind]
    
    # print(f"  Selected {len(theta_hat)} coefficients out of {len(theta_hat_full)} total")
    # print(f"  Basis order: {basis_order}, Original grid points: {len(grid_points_hal_original)}")
    # print(f"  Selected knots: {len(grid_points_hal_selected)}")
    
    n_samples = len(data)
    
    # Create basis matrix for the data using the original grid points (not selected)
    basis_matrix, _ = create_basis_functions(
        data, 
        grid_points_hal_original, 
        order=basis_order,
        include_intercept=True
    )
    basis_matrix = basis_matrix.numpy()  # Shape: (n, K_full)

    # Select only the columns corresponding to selected coefficients
    basis_matrix_selected = basis_matrix[:, theta_selected_ind]  # Shape: (n, K_selected)
    
    # Compute score contributions for each observation
    # Score is the gradient of log-likelihood: ∇ᵦ log f(xᵢ; β)
    
    # For HAL density model: log f(x; β) = φ(x)ᵀβ - log Z(β)
    # Score = φ(xᵢ) - E[φ(X)] where expectation is under current density
    
    # Compute E[φ(X)] by numerical integration
    # Create integration grid (same as used in estimation)
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
    phi_integration = phi_integration.numpy()  # Shape: (n_grid, K_full)

    # Select only the columns corresponding to selected coefficients
    phi_integration_selected = phi_integration[:, theta_selected_ind]  # Shape: (n_grid, K_selected)
    
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
    )  # Shape: (K,)
    
    # Compute score for each observation: S(xᵢ) = φ(xᵢ) - E[φ(X)]
    scores = basis_matrix_selected - expected_phi[np.newaxis, :]  # Shape: (n, K)
    
    # Compute observed Fisher information: I = (1/n) Σᵢ S(xᵢ) S(xᵢ)ᵀ
    observed_information = (scores.T @ scores) / n_samples  # Shape: (K, K)
    
    # Add ridge regularization for numerical stability
    ridge_matrix = ridge_param * np.eye(observed_information.shape[0])
    regularized_information = observed_information + ridge_matrix
    
    # Compute covariance matrix: Cov(β̂) = (1/n) I_reg⁻¹
    try:
        covariance_matrix = np.linalg.inv(regularized_information) / n_samples
    except np.linalg.LinAlgError:
        warnings.warn("Information matrix is singular. Using pseudo-inverse.")
        covariance_matrix = np.linalg.pinv(regularized_information) / n_samples
    
    return covariance_matrix


def density_confidence_interval(x_values: np.ndarray, estimation_results: Dict, 
                               cov_beta: np.ndarray, alpha: float = 0.05) -> pd.DataFrame:
    """
    Compute confidence intervals for density estimates at specified points using delta method.
    
    Following methodology.md:
    Var{f(x₀;β̂)} = f(x₀;β̂)² [φ(x₀) - E{φ(X)}]ᵀ Cov(β̂) [φ(x₀) - E{φ(X)}]
    
    Args:
        x_values: Points where to evaluate confidence intervals
        estimation_results: Dictionary containing fitted model results  
        cov_beta: Covariance matrix of coefficients
        alpha: Significance level (default 0.05 for 95% CI)
        
    Returns:
        DataFrame with columns: x, density, se, lower, upper
    """
    # Extract fitted parameters
    theta_hat_full = np.array(estimation_results['results']['theta_hat'])
    
    # Detect basis order
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
    
    # Ensure x_values is within [0, 1] and sorted
    x_values = np.clip(x_values, 0, 1)
    
    # Create DataFrame for evaluation points
    eval_df = pd.DataFrame({'W1': x_values})
    
    # Create basis matrix for evaluation points
    phi_eval, _ = create_basis_functions(
        eval_df,
        grid_points_hal_original, 
        order=basis_order,
        include_intercept=True
    )

    phi_eval = phi_eval.numpy()  # Shape: (n_eval, K_full)

    # Select only the columns corresponding to selected coefficients
    phi_eval_selected = phi_eval[:, theta_selected_ind]  # Shape: (n_eval, K_selected)
    
    # Compute E[φ(X)] (same as in covariance estimation)
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

    # Select only the columns corresponding to selected coefficients
    phi_integration_selected = phi_integration[:, theta_selected_ind]
    
    log_density_integration = phi_integration_selected @ theta_hat
    max_log_density = np.max(log_density_integration)
    density_unnorm = np.exp(log_density_integration - max_log_density)
    
    dx = integration_grid[1] - integration_grid[0]
    normalizing_constant = np.trapz(density_unnorm, dx=dx)
    density_integration = density_unnorm / normalizing_constant
    
    # Now compute properly normalized density values at evaluation points
    log_density_eval = phi_eval_selected @ theta_hat
    # Normalize by subtracting log of the normalizing constant
    # log(Z) = log(normalizing_constant) + max_log_density
    log_Z = np.log(normalizing_constant) + max_log_density
    density_eval = np.exp(log_density_eval - log_Z)
    
    expected_phi = np.trapz(
        phi_integration_selected * density_integration[:, np.newaxis],
        dx=dx,
        axis=0
    )
    
    # Compute gradient of density: ∇_β f(x₀) = f(x₀) [φ(x₀) - E[φ(X)]]
    n_eval = len(x_values)
    standard_errors = np.zeros(n_eval)
    
    for i in range(n_eval):
        # Gradient of density w.r.t. β at point x_values[i]
        grad_density = density_eval[i] * (phi_eval_selected[i] - expected_phi)
        
        # Variance using delta method: Var(f) = ∇f^T Cov(β) ∇f
        variance_f = grad_density.T @ cov_beta @ grad_density
        
        # Handle potential negative variance due to numerical errors
        if variance_f < 0:
            warnings.warn(f"Negative variance detected at x={x_values[i]:.3f}, setting to 0")
            variance_f = 0
            
        standard_errors[i] = np.sqrt(variance_f)
    
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
    Compute a comprehensive summary of density variance analysis.
    
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
    
    # Step 1: Estimate covariance matrix
    # print("  Estimating covariance matrix...")
    cov_beta = estimate_covariance_beta(data, estimation_results, ridge_param)
    
    # Step 2: Compute confidence intervals
    # print("  Computing confidence intervals...")
    ci_df = density_confidence_interval(x_values, estimation_results, cov_beta, alpha)
    
    # Step 3: Compute summary statistics
    basis_order = detect_basis_order(estimation_results)
    n_selected_coeffs = len(estimation_results['results']['theta_hat'])
    condition_number = np.linalg.cond(cov_beta)
    avg_std_error = ci_df['se'].mean()
    avg_interval_width = (ci_df['upper'] - ci_df['lower']).mean()
    
    summary = {
        'basis_order': basis_order,
        'n_selected_coeffs': n_selected_coeffs,
        'condition_number': condition_number,
        'avg_std_error': avg_std_error,
        'avg_interval_width': avg_interval_width,
        'confidence_intervals': ci_df,
        'covariance_matrix': cov_beta
    }
    
    return summary