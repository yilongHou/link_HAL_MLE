import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from typing import Dict, Tuple

class KDEEstimator:
    """
    Density estimator using Kernel Density Estimation (KDE).
    
    This class is a wrapper around scikit-learn's KernelDensity to make it
    compatible with the project's estimator structure.
    """
    
    def __init__(
        self,
        kernel: str = 'gaussian',
        bandwidth: float = 0.0125,
        n_grid_points: int = 1000,
        log_dir: str = None # Not used in this class, but can be added for consistency
    ):
        """
        Initialize KDE estimator.
        
        Args:
            kernel: The kernel to use in the KDE.
            bandwidth: The bandwidth of the kernel.
            n_grid_points: Number of points for the evaluation grid.
        """
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.n_grid_points = n_grid_points
        self.is_fitted = False
        self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        self.grid_points: np.ndarray = np.linspace(0, 1, self.n_grid_points).reshape(-1, 1)
        self.estimated_density: np.ndarray = None

    def fit(self, data: pd.DataFrame) -> 'KDEEstimator':
        """
        Fit the KDE estimator to data.
        
        Args:
            data: DataFrame with column 'W1' containing the observations.
            
        Returns:
            Self for method chaining.
        """
        self.kde.fit(data[['W1']])
        self.is_fitted = True
        
        # Pre-compute density for get_density and get_results
        log_dens = self.kde.score_samples(self.grid_points)
        self.estimated_density = np.exp(log_dens)
        
        return self
        
    def get_density(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the estimated density on the evaluation grid.
        
        Returns:
            Tuple of (grid_points, density_values).
        
        Raises:
            ValueError: If the estimator hasn't been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting density. Call fit() first.")
            
        return self.grid_points.ravel(), self.estimated_density

    def get_results(self) -> Dict:
        """
        Get results from the fitting process.
        
        Returns:
            Dictionary containing all relevant results.
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting results. Call fit() first.")
            
        return {
            "kernel": self.kernel,
            "bandwidth": self.bandwidth,
            "grid_points": self.grid_points.ravel().tolist(),
            "estimated_density": self.estimated_density.tolist(),
            "n_selected_knots": None,  # Not applicable for KDE
        }
