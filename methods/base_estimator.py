import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
import os
from abc import ABC, abstractmethod
from utils.basis import create_basis_functions
from utils.plotting import plot_density
from scipy.interpolate import interp1d

EXPLOSION_THRESHOLD = 1e4  # Threshold for detecting numerical explosion in optimization

class BaseEstimator(ABC):
    """
    Base class for all density estimators.
    
    Provides common functionality for:
    - Logging setup
    - Density evaluation at specific points
    - Log-likelihood computation
    - Common initialization parameters
    """
    
    def __init__(
        self,
        lam: float = 3.0,
        n_iterations: int = 100,
        tol: float = 1e-18,
        basis_order: int = 0,
        log_dir: str = "./local/logs/experiment.log",
        log_frequency: int = 10,
        **kwargs  # Allow subclasses to pass additional parameters
    ):
        """
        Initialize base estimator.
        
        Args:
            lam: L1 regularization parameter
            n_iterations: Maximum number of iterations
            tol: Convergence tolerance
            basis_order: Order of the truncated power basis
            log_dir: Directory for logging
            log_frequency: Frequency of logging (-1 means no logging)
            **kwargs: Additional parameters for subclasses
        """
        self.lam = lam
        self.n_iterations = n_iterations
        self.tol = tol
        self.basis_order = basis_order
        self.is_fitted = False

        self.explosion_threshold = EXPLOSION_THRESHOLD

        self.basis_names: Optional[list] = None
        self.fitted_theta_dict: Optional[Dict[str, float]] = None
        
        # Will be set during fitting
        self.theta_hat: Optional[np.ndarray] = None
        self.grid_points: Optional[np.ndarray] = None
        self._grid_points_hal: Optional[np.ndarray] = None
        self.grid_midpoints: Optional[np.ndarray] = None
        self.delta_j: Optional[np.ndarray] = None
        self.grid_points_hal_selected: Optional[np.ndarray] = None
        
        # Setup logging
        self.log_dir = log_dir
        self.do_log = (log_frequency > 0) and (log_dir is not None)
        self.log_frequency = log_frequency
        if self.do_log:
            self._setup_logging()
    
    def _setup_logging(self):
        """Setup a dedicated logger for the estimator instance."""
        # Ensure the log directory exists.
        os.makedirs(os.path.dirname(self.log_dir), exist_ok=True)

        # Create a unique logger name for this instance to avoid conflicts.
        logger_name = f"{self.__class__.__name__}-{id(self)}"
        self.logger = logging.getLogger(logger_name)
        
        # Prevent messages from propagating to the root logger.
        self.logger.propagate = False
        
        # Set the logging level.
        self.logger.setLevel(logging.INFO)

        # If handlers are already present, clear them to avoid duplicate logs.
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Create a file handler to write to the specified log file (overwriting previous logs).
        file_handler = logging.FileHandler(self.log_dir, mode='w')
        file_handler.setLevel(logging.INFO)

        # Create a formatter and set it for the handler.
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the handler to the logger.
        self.logger.addHandler(file_handler)
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> 'BaseEstimator':
        """
        Fit the estimator to data.
        
        Args:
            data: DataFrame with column 'W1' containing the observations
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    def get_density(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the estimated density on the evaluation grid.
        
        Returns:
            Tuple of (grid_points, density_values)
        
        Raises:
            ValueError: If the estimator hasn't been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting density. Call fit() first.")

        if self.grid_points is None:
            raise ValueError("Evaluation grid points (self.grid_points) are not set. Call fit() first.")
        
        density = self.get_density_at_points(self.grid_points)
        
        return self.grid_points, density
    
    @abstractmethod
    def get_results(self) -> Dict:
        """
        Get comprehensive results from the fitting process.
        
        Returns:
            Dictionary containing all relevant results
        """
        pass

    def _get_common_results(self) -> Dict:
        """
        Helper to get results common to all estimators inheriting from BaseEstimator.
        
        Returns:
            A dictionary with standardized result fields.
        """
        if not self.is_fitted or self.theta_hat is None:
            raise ValueError("Estimator must be fitted before getting results.")

        grid_points, density = self.get_density()
        
        # Count selected knots (non-zero coefficients for HAL basis, excluding intercept and polynomials)
        hal_coeffs = self.theta_hat[self.basis_order + 1:]
        selected_knots_count = np.sum(np.abs(hal_coeffs) > self.tol)

        print(f"Results - number of selected knots: {selected_knots_count}")
        
        return {
            "fitted_theta_dict": self.fitted_theta_dict,
            "theta_hat": self.theta_hat,
            "data_points": getattr(self, '_grid_points_hal', None),
            "grid_points_hal_selected": getattr(self, 'grid_points_hal_selected', None),
            "n_selected_knots": selected_knots_count,
            "estimated_density": density,
            "grid_points": grid_points,
            "intercept": self.theta_hat[0],
            "hal_coeffs": hal_coeffs,
        }
    
    def get_density_at_points(self, points: np.ndarray) -> np.ndarray:
        """
        Get the estimated density at specific points using the fitted model.
        
        Args:
            points: Array of points where to evaluate the density
            
        Returns:
            Array of density values at the given points
            
        Raises:
            ValueError: If the estimator hasn't been fitted yet
        """
        if not self.is_fitted or self.theta_hat is None or self._grid_points_hal is None or self.grid_midpoints is None or self.delta_j is None:
            raise ValueError("Estimator must be fitted before getting density. Call fit() first.")
        
        return BaseEstimator.calculate_density_at_points(
            points=points,
            theta_hat=self.theta_hat,
            basis_grid_points=self._grid_points_hal,
            basis_order=self.basis_order,
            norm_grid_midpoints=self.grid_midpoints,
            norm_delta_j=self.delta_j
        )
    
    @staticmethod
    def calculate_density_at_points(
        points: np.ndarray,
        theta_hat: np.ndarray,
        basis_grid_points: np.ndarray,
        basis_order: int,
        norm_grid_midpoints: Optional[np.ndarray] = None,
        norm_delta_j: Optional[np.ndarray] = None,
        n_norm_grid_points: int = 1000
    ) -> np.ndarray:
        """
        Calculates the estimated density at specific points.

        This is a static method that contains the core logic for density calculation,
        allowing it to be used independently of a fitted estimator instance.

        Args:
            points: Array of points where to evaluate the density.
            theta_hat: The coefficient vector for the basis functions.
            basis_grid_points: The grid points used to define the HAL basis functions.
            basis_order: The order of the truncated power basis.
            norm_grid_midpoints: Optional. The midpoints for numerical integration. If None, a grid
                               will be generated based on `basis_grid_points`. It is highly
                               recommended to provide this to ensure consistent normalization.
            norm_delta_j: Optional. The interval widths for numerical integration. If None, will be
                          generated.
            n_norm_grid_points: Number of points for the auto-generated normalization grid.

        Returns:
            Array of density values at the given points.
        """
        # Ensure points is a numpy array and flatten
        pts = np.asarray(points).ravel()
                        
        # --- Normalization Calculation ---
        if norm_grid_midpoints is None or norm_delta_j is None:
            # WARNING: Auto-generating the normalization grid can lead to inconsistent
            # density values. The normalization constant should be fixed for a given model.
            # A new grid is created over the range of the basis functions.
            norm_grid = np.linspace(0, 1+1/n_norm_grid_points, n_norm_grid_points)
            _norm_delta_j = np.diff(norm_grid)
            _norm_grid_midpoints = norm_grid[:-1] + _norm_delta_j / 2
        else:
            _norm_grid_midpoints = norm_grid_midpoints
            _norm_delta_j = norm_delta_j
        
        # Use the calculated or provided normalization grid
        df_mid = pd.DataFrame({'W1': _norm_grid_midpoints})
        b_mid, _ = create_basis_functions(df_mid, basis_grid_points, order=basis_order)
        phi_mid = b_mid.numpy()
        
        log_f_mid = phi_mid @ theta_hat
        f_u = np.exp(log_f_mid)
        Z = np.sum(f_u * _norm_delta_j)
        density_hat = f_u / Z

        # Make sure density sums to 1
        if not np.isclose(np.sum(density_hat * _norm_delta_j), 1.0):
            logging.warning("Density normalization failed to sum to 1. Adjusting density values.")
            density_hat *= Z / np.sum(density_hat * _norm_delta_j)
        # Interpolate density values at the requested points
        density_interp = interp1d(
            _norm_grid_midpoints, 
            density_hat, 
            kind="linear",
            bounds_error=False, 
            fill_value=(density_hat[0], density_hat[-1])
        )
        # Evaluate the density at the requested points
        density = density_interp(pts)
        
        return density

    def get_log_likelihood_for_points(self, points: np.ndarray) -> np.ndarray:
        """
        Get the log-likelihood for specific points.
        
        Args:
            points: Array of points where to evaluate the log-likelihood
            
        Returns:
            Array of log-likelihood values at the given points
            
        Raises:
            ValueError: If the estimator hasn't been fitted yet
        """        
        if self.theta_hat is None:
            raise ValueError("theta_hat is None, fitting may have failed.")
        
        if self._grid_points_hal is None:
            raise ValueError("_grid_points_hal is None, fitting may have failed.")
        
        if self.grid_midpoints is None or self.delta_j is None:
            raise ValueError("grid_midpoints or delta_j is None, fitting may have failed.")
        
        # Ensure points is a numpy array and flatten
        pts = np.asarray(points).ravel()
        
        # Create DataFrame for evaluation points
        df_eval = pd.DataFrame({'W1': pts})
        
        # Create basis functions at evaluation points using the same grid points as training
        b_eval, _ = create_basis_functions(df_eval, self._grid_points_hal, order=self.basis_order)
        phi_eval = b_eval.numpy()  # (n_pts, K)
        
        # Compute un-normalized log density
        log_density = phi_eval @ self.theta_hat
        
        # Use the same normalization approach as in training (midpoints)
        df_mid = pd.DataFrame({'W1': self.grid_midpoints})
        b_mid, _ = create_basis_functions(df_mid, self._grid_points_hal, order=self.basis_order)
        phi_mid = b_mid.numpy()
        
        log_f_mid = phi_mid @ self.theta_hat
        max_log_f = np.max(log_f_mid)
        Z = np.sum(np.exp(log_f_mid - max_log_f) * self.delta_j)
        log_Z = np.log(Z)
        
        # Return log-likelihood: log_density - max_log_f - log_Z
        log_likelihood = log_density - max_log_f - log_Z
        
        return log_likelihood


    def get_avg_log_likelihood_for_points(self, points: np.ndarray) -> float:
        """
        Get the sum of log-likelihood for specific points.
        
        Args:
            points: Array of points where to evaluate the log-likelihood
            
        Returns:
            Sum of log-likelihood values at the given points
            
        Raises:
            ValueError: If the estimator hasn't been fitted yet
        """
        # Use interpolation to get density values at the points
        grid_points, estimated_density = self.get_density()

        density_interp = interp1d(
            grid_points, 
            estimated_density, 
            kind="linear",
            bounds_error=False, 
            fill_value=(estimated_density[0], estimated_density[-1])
        )

        interpolated_density = density_interp(points)
        log_density = np.log(interpolated_density)
        # sum_log_likelihood = np.sum(log_density)
        # return float(sum_log_likelihood)
        return np.mean(log_density)
    
    def get_sum_log_likelihood_for_points(self, points: np.ndarray) -> float:
        """
        Get the sum of log-likelihood for specific points.
        
        Args:
            points: Array of points where to evaluate the log-likelihood
            
        Returns:
            Sum of log-likelihood values at the given points
            
        Raises:
            ValueError: If the estimator hasn't been fitted yet
        """
        # Use interpolation to get density values at the points
        grid_points, estimated_density = self.get_density()

        density_interp = interp1d(
            grid_points, 
            estimated_density, 
            kind="linear",
            bounds_error=False, 
            fill_value=(estimated_density[0], estimated_density[-1])
        )

        interpolated_density = density_interp(points)
        log_density = np.log(interpolated_density)
        sum_log_likelihood = np.sum(log_density)
        return float(sum_log_likelihood)
    
    def compute_bic(self, data: pd.DataFrame) -> float:
        """
        Compute the Bayesian Information Criterion (BIC) for the fitted model.
        
        BIC = -2 * log_likelihood + k * log(n)
        
        where:
        - log_likelihood is the log-likelihood of the model on the data
        - k is the number of non-zero parameters (effective model complexity)
        - n is the number of data points
        
        Args:
            data: DataFrame with column 'W1' containing the observations used for fitting
            
        Returns:
            BIC value (lower is better)
            
        Raises:
            ValueError: If the estimator hasn't been fitted yet
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before computing BIC. Call fit() first.")
        
        if self.theta_hat is None:
            raise ValueError("theta_hat is None, fitting may have failed.")
        
        # Number of data points
        n = len(data)
        
        # Compute log-likelihood on the training data
        points = np.asarray(data['W1'].values)
        sum_log_likelihood = self.get_sum_log_likelihood_for_points(points)
        
        # Count non-zero parameters (effective model complexity)
        # Include intercept (always counted) + non-zero penalized coefficients
        k = 1 + np.sum(np.abs(self.theta_hat[1:]) > self.tol)
        
        # Compute BIC
        bic = -2 * sum_log_likelihood + k * np.log(n)
        
        return float(bic)

    def plot_estimator_results(self, data: pd.DataFrame, title: str = "Estimator Results") -> plt.Figure:
        grid_points, estimated_density = self.get_density()
        estimation_results = self.get_results()

        density_plot = plot_density(
            grid_points=grid_points,
            true_density=None,
            # true_density=sampler.compute_density(grid_points),
            estimated_density=estimated_density,
            title=title,
            method_label=f"{self.__class__.__name__} Density \n(Num Knots: {estimation_results['n_selected_knots']})",
            show=False
        )
        
        # add data["W1"].hist(bins=100, density=True, alpha=0.5) to the plot
        ax = density_plot.gca()
        ax.hist(
            data["W1"], 
            bins=100, 
            density=True, 
            alpha=0.5, 
            label="Sampled Data"
        )
        ax.legend()
        return density_plot
    
    def _check_objective_explosion(self, obj: float, iteration: int) -> bool:
        """
        Check if objective function has exploded or become non-finite.
        
        This method provides a centralized way to check for numerical issues
        in the objective function during optimization.
        
        Args:
            obj: Current objective function value
            iteration: Current iteration number
            
        Returns:
            True if optimization should stop (explosion detected), False otherwise
            
        Side Effects:
            Prints warning messages when explosion is detected
        """
        if not np.isfinite(obj):
            print(f"Warning: Objective function is non-finite at iteration {iteration}, stopping optimization")
            return True
        elif obj > self.explosion_threshold or obj < -self.explosion_threshold:
            print(f"Warning: Objective function exploded to {obj:.2e} at iteration {iteration}, stopping optimization")
            return True
        return False
