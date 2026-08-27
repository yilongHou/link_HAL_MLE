import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings
import os
import subprocess

# Check if rpy2 is available
try:
    from rpy2.robjects import pandas2ri, r
    from rpy2.robjects.packages import importr
    RPY2_AVAILABLE = True
except ImportError as e:
    RPY2_AVAILABLE = False
    warnings.warn(f"rpy2 not available. LogSplinesEstimator will not work. Error: {e}")

def check_r_installation():
    """Check if R is properly installed and accessible."""
    try:
        result = subprocess.run(['R', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            return True, result.stdout.split('\n')[0]
        else:
            return False, "R command not found"
    except FileNotFoundError:
        return False, "R not installed"

def get_r_architecture():
    """Get R architecture information."""
    try:
        result = subprocess.run(['R', '--slave', '--vanilla', '-e', 'cat(R.version$arch)'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return "unknown"


class LogSplinesEstimator:
    """
    Density estimator using R's logspline package.
    
    This class uses R's logspline package through rpy2 to perform 
    density estimation using log-spline methods.
    """
    
    def __init__(
        self,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        n_grid_points: int = 1000,
        log_dir: Optional[str] = None,
        install_r_package: bool = True
    ):
        """
        Initialize LogSplines estimator.
        
        Args:
            lower_bound: Lower bound for the support of the density.
            upper_bound: Upper bound for the support of the density.
            n_grid_points: Number of points for the evaluation grid.
            log_dir: Directory for logging (not used but kept for consistency).
            install_r_package: Whether to install the R logspline package if not available.
        """
        if not RPY2_AVAILABLE:
            raise ImportError(
                "rpy2 is not available. Please install it with: pip install rpy2"
            )
        
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_grid_points = n_grid_points
        self.log_dir = log_dir
        self.basis_order = None  # logspline automatically selects basis order
        self.is_fitted = False
        self.estimated_density: np.ndarray = np.array([])
        self.grid_points: np.ndarray = np.linspace(
            self.lower_bound, self.upper_bound, self.n_grid_points
        )
        
        # Initialize R interface
        self._setup_r_environment(install_r_package)
        
    def _setup_r_environment(self, install_package: bool = True):
        """Setup R environment and import logspline package."""
        try:
            # Check R installation first
            r_available, r_info = check_r_installation()
            if not r_available:
                raise RuntimeError(f"R is not properly installed: {r_info}")
            
            r_arch = get_r_architecture()
            print(f"R installation found: {r_info}")
            print(f"R architecture: {r_arch}")
            
            # Activate pandas <-> R conversion (updated for newer rpy2)
            from rpy2.robjects.conversion import localconverter
            from rpy2.robjects import pandas2ri
            
            # Store the converter for later use
            self.converter = localconverter(pandas2ri.converter)
            
            # Try to import logspline package
            try:
                self.logspline = importr("logspline")
            except Exception:
                if install_package:
                    # Install logspline package if not available
                    print("Installing R logspline package...")
                    r('if (!require("logspline")) install.packages("logspline", repos="https://cloud.r-project.org")')
                    self.logspline = importr("logspline")
                else:
                    raise ImportError("R logspline package not available and installation disabled")
                    
        except Exception as e:
            raise RuntimeError(f"Failed to setup R environment: {e}")
    
    def fit(self, data: pd.DataFrame) -> 'LogSplinesEstimator':
        """
        Fit the logspline estimator to data.
        
        Args:
            data: DataFrame with column 'W1' containing the observations.
            
        Returns:
            Self for method chaining.
        """
        if not RPY2_AVAILABLE:
            raise RuntimeError("rpy2 not available")
            
        # Extract observations
        observations = data['W1'].values
        
        # Convert to R object using the converter context
        with self.converter:
            obs_r = pandas2ri.py2rpy(pd.Series(observations))
        
        # Fit logspline model
        try:
            self.logspline_fit = self.logspline.logspline(x=obs_r)
        except Exception as e:
            raise RuntimeError(f"Failed to fit logspline model: {e}")
        
        # Evaluate density on grid
        with self.converter:
            grid_r = pandas2ri.py2rpy(pd.Series(self.grid_points))
        
        try:
            # Get density values from R using the correct function reference
            dens_r = r.dlogspline(grid_r, self.logspline_fit)
            self.estimated_density = np.array(dens_r)
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate density: {e}")
        
        self.is_fitted = True
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
            
        return self.grid_points, self.estimated_density

    def get_results(self) -> Dict:
        """
        Get results from the fitting process.
        
        Returns:
            Dictionary containing all relevant results.
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting results. Call fit() first.")
            
        return {
            "method": "logsplines",
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "grid_points": self.grid_points.tolist(),
            "estimated_density": self.estimated_density.tolist(),
            "n_selected_knots": None,  # logspline automatically selects knots
        }
    
    def evaluate_density(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the fitted density at arbitrary points.
        
        Args:
            x: Points at which to evaluate the density.
            
        Returns:
            Density values at the given points.
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before evaluating density. Call fit() first.")
        
        # Convert points to R object
        with self.converter:
            x_r = pandas2ri.py2rpy(pd.Series(x))
        
        try:
            # Evaluate density using the correct function reference
            dens_r = r.dlogspline(x_r, self.logspline_fit)
            return np.array(dens_r)
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate density at given points: {e}")