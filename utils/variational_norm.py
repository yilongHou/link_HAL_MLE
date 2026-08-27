import numpy as np
from scipy.integrate import quad
from typing import Callable, Union
import warnings

# ## Summary of Variational Norms

# | Distribution      | Order 0    | Order 1    | Order 2     | Order 3           |
# |-------------------|------------|------------|-------------|-------------------|
# | Truncated Normal  | 7.978835   | 96.787658  | 1491.001449 | 823397257.370666  |
# | Sinusoidal        | 1.785077   | 6.060486   | 292.109509  | 554838838.538698  |
# | Truncated GMM     | 14.087051  | 311.287937 | 8917.195009 | 932544602.073146  |
# | Step Function     | 1.333333   | N/A        | N/A         | N/A               |


def numerical_derivative(f: Callable, x: float, k: int = 1, h: float = 1e-6) -> float:
    """
    Recursively approximate the k-th derivative of f at x via finite differences.
    
    Args:
        f: Function to differentiate
        x: Point at which to compute the derivative
        k: Order of derivative (0 returns f(x))
        h: Step size for finite differences
        
    Returns:
        Approximation of f^{(k)}(x)
    """
    if k == 0:
        return f(x)
    else:
        # Forward finite difference: f'(x) ≈ (f(x+h) - f(x)) / h
        return (numerical_derivative(f, x + h, k - 1, h) -
                numerical_derivative(f, x, k - 1, h)) / h


def total_variation_of_derivative(f: Callable, k: int, a: float = 0, b: float = 1, 
                                 epsabs: float = 1e-8, epsrel: float = 1e-8) -> float:
    """
    Compute ∫_a^b | f^{(k)}(x) | dx via numerical integration.
    
    Args:
        f: Function whose k-th derivative's total variation to compute
        k: Order of derivative
        a: Lower bound of integration
        b: Upper bound of integration
        epsabs: Absolute error tolerance for integration
        epsrel: Relative error tolerance for integration
        
    Returns:
        Total variation of the k-th derivative
    """
    def integrand(x):
        try:
            return np.abs(numerical_derivative(f, x, k))
        except (OverflowError, ZeroDivisionError):
            # Handle numerical issues gracefully
            return 0.0
    
    try:
        variation, error = quad(integrand, a, b, epsabs=epsabs, epsrel=epsrel)
        return variation
    except Exception as e:
        warnings.warn(f"Integration failed: {e}. Returning 0.")
        return 0.0


def variational_norm(f: Callable, order: int = 0, a: float = 0, b: float = 1,
                    h: float = 1e-6, epsabs: float = 1e-8, epsrel: float = 1e-8) -> float:
    """
    Compute the k-th order variational norm:
        VN_k(f) = SUM_{0}^{k}[| f^{(k)}(a) |] + ∫_a^b | f^{(k+1)}(x) | dx.
    
    Args:
        f: Function to compute variational norm for
        order: Order k of the variational norm
        a: Lower bound of domain
        b: Upper bound of domain
        h: Step size for numerical differentiation
        epsabs: Absolute error tolerance for integration
        epsrel: Relative error tolerance for integration
        
    Returns:
        k-th order variational norm of f
    """
    if order < 0:
        raise ValueError("Order must be non-negative")
    
    try:
        # Boundary term: Sum |f^{(k)}(a)| from 0 to k
        boundary_term = np.sum([
            np.abs(numerical_derivative(f, a, order, h)) for order in range(order + 1)
        ])
        
        # Total variation term: ∫_a^b |f^{(k+1)}(x)| dx
        tv_term = total_variation_of_derivative(f, order + 1, a, b, epsabs, epsrel)
        
        return boundary_term + tv_term
    
    except Exception as e:
        warnings.warn(f"Variational norm computation failed: {e}. Returning inf.")
        return np.inf


class VariationalNormComputer:
    """
    A class for computing variational norms of probability density functions.
    
    This class provides methods to compute variational norms for different orders
    and handles the specific cases for different types of distributions.
    """
    
    def __init__(self, h: float = 1e-6, epsabs: float = 1e-8, epsrel: float = 1e-8):
        """
        Initialize the variational norm computer.
        
        Args:
            h: Step size for numerical differentiation
            epsabs: Absolute error tolerance for integration
            epsrel: Relative error tolerance for integration
        """
        self.h = h
        self.epsabs = epsabs
        self.epsrel = epsrel
    
    def compute_single_norm(self, f: Callable, order: int, a: float = 0, b: float = 1) -> float:
        """
        Compute a single variational norm for a given function and order.
        
        Args:
            f: Function to compute variational norm for
            order: Order of the variational norm
            a: Lower bound of domain
            b: Upper bound of domain
            
        Returns:
            Variational norm value
        """
        return variational_norm(f, order, a, b, self.h, self.epsabs, self.epsrel)
    
    def compute_multiple_norms(self, f: Callable, max_order: int = 3, 
                              a: float = 0, b: float = 1) -> dict:
        """
        Compute variational norms for multiple orders.
        
        Args:
            f: Function to compute variational norms for
            max_order: Maximum order to compute (inclusive)
            a: Lower bound of domain
            b: Upper bound of domain
            
        Returns:
            Dictionary mapping order to variational norm value
        """
        norms = {}
        for order in range(max_order + 1):
            norms[order] = self.compute_single_norm(f, order, a, b)
        return norms
    
    def compute_for_sampler_distributions(self, sampler_classes: list, 
                                        max_order: int = 3) -> dict:
        """
        Compute variational norms for multiple sampler distribution classes.
        
        Args:
            sampler_classes: List of tuples (name, instance) where instance has compute_density method
            max_order: Maximum order to compute (inclusive)
            
        Returns:
            Nested dictionary: {distribution_name: {order: norm_value}}
        """
        results = {}
        
        for name, sampler_instance in sampler_classes:
            print(f"\nComputing variational norms for {name}...")
            
            # Create a wrapper function for the PDF
            def pdf_func(x):
                return sampler_instance.compute_density(np.array([x]))[0]
            
            # Special handling for step function (only order 0)
            if "step" in name.lower():
                results[name] = {0: self.compute_single_norm(pdf_func, 0)}
                print(f"  VN (order=0): {results[name][0]:.6f}")
            else:
                results[name] = self.compute_multiple_norms(pdf_func, max_order)
                for order, norm_value in results[name].items():
                    print(f"  VN (order={order}): {norm_value:.6f}")
        
        return results
    
    def compute_for_function_dict(self, pdf_dict: dict, max_order: int = 3) -> dict:
        """
        Compute variational norms for a dictionary of PDF functions.
        
        Args:
            pdf_dict: Dictionary mapping names to PDF functions
            max_order: Maximum order to compute (inclusive)
            
        Returns:
            Nested dictionary: {distribution_name: {order: norm_value}}
        """
        results = {}
        
        for name, pdf_func in pdf_dict.items():
            print(f"\nComputing variational norms for {name}...")
            
            # Special handling for step function (only order 0)
            if "step" in name.lower():
                results[name] = {0: self.compute_single_norm(pdf_func, 0)}
                print(f"  VN (order=0): {results[name][0]:.6f}")
            else:
                results[name] = self.compute_multiple_norms(pdf_func, max_order)
                for order, norm_value in results[name].items():
                    print(f"  VN (order={order}): {norm_value:.6f}")
        
        return results


def analytical_step_function_vn(level1: float = 1.0, level2: float = 0.5, 
                               breakpoint: float = 0.5) -> float:
    """
    Compute the analytical variational norm (order 0) for a step function.
    
    For a step function with f(x) = level1 for x < breakpoint, level2 for x >= breakpoint,
    the 0-order VN is |f(0)| + total_variation(f) = level1 + |level1 - level2|.
    
    Args:
        level1: Function value for x < breakpoint
        level2: Function value for x >= breakpoint  
        breakpoint: Point where function changes (not used in VN computation)
        
    Returns:
        Analytical variational norm (order 0)
    """
    return level1 + abs(level1 - level2)


# Convenience function for quick computation
def compute_vn_for_pdf(pdf_func: Callable, orders: list = [0, 1, 2, 3], 
                      skip_higher_orders: bool = False) -> dict:
    """
    Convenience function to compute variational norms for a single PDF.
    
    Args:
        pdf_func: PDF function to analyze
        orders: List of orders to compute
        skip_higher_orders: If True, only compute order 0 (useful for step functions)
        
    Returns:
        Dictionary mapping order to variational norm value
    """
    computer = VariationalNormComputer()
    
    if skip_higher_orders:
        orders = [0]
    
    results = {}
    for order in orders:
        results[order] = computer.compute_single_norm(pdf_func, order)
    
    return results