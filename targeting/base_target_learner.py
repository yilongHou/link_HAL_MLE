import time
import numpy as np
import pandas as pd
import cvxpy as cp
from abc import ABC, abstractmethod
from typing import Union, Dict, Any
from utils.basis import create_basis_functions


class BaseTargetLearner(ABC):
    """
    Base class for targeting learners in maximum likelihood estimation.
    
    This class encapsulates all shared logic for the M-step update, where only
    the targeting basis construction varies between different targeting strategies.
    """
    
    def __init__(self, norm_constraint: int = 20, basis_order: int = 0):
        """
        Initialize the base target learner.
        
        Parameters
        ----------
        norm_constraint : int, optional
            L1-norm constraint for the targeting coefficient(s) (default is 20).
        basis_order : int, optional
            Order of the basis functions (default is 0).
        """
        self.norm_constraint = norm_constraint
        self.basis_order = basis_order
    
    @abstractmethod
    def get_b_ik_targeting(self, uncensored_data: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Create targeting basis design matrix for uncensored data.
        
        Parameters
        ----------
        uncensored_data : pd.DataFrame
            The uncensored data points.
        **kwargs
            Additional method-specific parameters.
            
        Returns
        -------
        np.ndarray
            Targeting basis matrix for data points, shape (n_data, n_targeting_basis).
        """
        pass
    
    @abstractmethod
    def get_b_jk_targeting(self, data_grid: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Create targeting basis design matrix for integration grid.
        
        Parameters
        ----------
        data_grid : pd.DataFrame
            The integration grid points.
        **kwargs
            Additional method-specific parameters.
            
        Returns
        -------
        np.ndarray
            Targeting basis matrix for grid points, shape (n_grid, n_targeting_basis).
        """
        pass
    
    @abstractmethod
    def get_b_jk_targeting_full(self, data_grid_full: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Create targeting basis design matrix for fine evaluation grid.
        
        Parameters
        ----------
        data_grid_full : pd.DataFrame
            The fine evaluation grid points.
        **kwargs
            Additional method-specific parameters.
            
        Returns
        -------
        np.ndarray
            Targeting basis matrix for fine grid, shape (n_grid_full, n_targeting_basis).
        """
        pass
    
    def run_m_step(
        self,
        uncensored_augmented: pd.DataFrame,
        grid_points_hal_selected: Union[list, np.ndarray],
        old_theta: Union[list, np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform an M-step update for a density model.
        
        This function performs an M-step update for a density model that has the form
           f(x) = exp(linear_predictor) / Z,
        where the linear predictor is the sum of two parts:
          (i) a "hal‐basis" part with an intercept and coefficients given by old_theta,
         (ii) a new "targeting‐basis" part with free coefficients (α) that are optimized.

        Only the targeting basis coefficients are updated; the intercept and the hal‐basis
        coefficients (old_theta) are kept fixed.
        
        Parameters
        ----------
        uncensored_augmented : pd.DataFrame
            A pandas DataFrame containing the uncensored data points.
        grid_points_hal_selected : Union[list, np.ndarray]
            The hal design grid points.
        old_theta : Union[list, np.ndarray]
            The current (fixed) parameter vector for the intercept and hal basis functions.
            Here, old_theta[0] is the intercept and old_theta[1:] are the hal coefficients.
        **kwargs
            Additional method-specific parameters passed to targeting basis methods.
            
        Returns
        -------
        Dict[str, Any]
            A dictionary with updated parameters and estimated density on a fine grid.
        """
        # ----------------------------
        # 1. Design matrix construction
        # ----------------------------
        start_time_m_design = time.time()

        # Use the hal grid from the previous iteration
        grid_points_hal = grid_points_hal_selected  # for hal-basis
        uncensored_data = uncensored_augmented.copy()

        # Create hal basis design matrix for uncensored data
        basis_tensor, _ = create_basis_functions(uncensored_data, grid_points_hal,
                                              order=self.basis_order, include_intercept=False)
        b_ik = basis_tensor.detach().numpy()  # shape: (n_data, n_hal_basis)

        # Create targeting basis design matrix for uncensored data
        b_ik_targeting = self.get_b_ik_targeting(uncensored_data, **kwargs)

        # Create an evaluation grid (here 200 points) for computing the normalization constant.
        grid_eval = np.linspace(0, 1, 2001)
        combined_grid = np.concatenate((grid_eval, grid_points_hal_selected))
        filtered_grid = combined_grid[combined_grid >= 0]
        filtered_grid = np.unique(filtered_grid)
        grid_eval = np.sort(filtered_grid)

        # Compute midpoints for integration
        grid_midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2
        data_grid = pd.DataFrame({'W1': grid_midpoints})

        # Evaluate hal basis on the integration grid
        basis_grid_tensor, _ = create_basis_functions(data_grid, grid_points_hal,
                                                   order=self.basis_order, include_intercept=False)
        b_jk = basis_grid_tensor.numpy()  # shape: (n_grid, n_hal_basis)

        # Evaluate targeting basis on the integration grid
        b_jk_targeting = self.get_b_jk_targeting(data_grid, **kwargs)

        end_time_m_design = time.time()
        m_step_time_design = end_time_m_design - start_time_m_design
        # print(f"M-step Design Matrix Calculation time (sec): {m_step_time_design:.4f}")

        # ----------------------------
        # 2. Set up the optimization problem
        # ----------------------------
        start_time_m_optimize = time.time()

        # Compute the fixed (hal-basis) part of the linear predictor:
        #   fixed = intercept + (hal_basis dot old_hal_coefficients)
        # For uncensored data:
        fixed_data = old_theta[0] + np.dot(b_ik, old_theta[1:])
        # For the integration grid:
        fixed_grid = old_theta[0] + np.dot(b_jk, old_theta[1:])
        # --- Set weights to 1 by default ---
        weights_M = np.ones(len(uncensored_data))
        n_samples_weighted = np.sum(weights_M)

        # Compute the widths (delta) for integration and their logarithms
        delta_j = grid_eval[1:] - grid_eval[:-1]
        log_delta_j = np.log(delta_j)

        # The only free parameter is the targeting basis coefficient vector.
        # Its dimension is determined by the number of targeting basis functions.
        r = b_ik_targeting.shape[1]
        alpha = cp.Variable(r)

        # Define the full linear predictor by adding the targeting part:
        # For data points:
        L_data = fixed_data + b_ik_targeting @ alpha  # shape: (n_data,)
        # For grid points:
        L_grid = fixed_grid + b_jk_targeting @ alpha    # shape: (n_grid,)

        # Formulate the (negative) log-likelihood.
        # Data term:  -sum_i weights[i] * (fixed + targeting_effect)
        data_term = -cp.sum(cp.multiply(weights_M, L_data))
        # Normalization term:  n_samples_weighted * log_sum_exp( log(delta_j) + L_grid )
        norm_term = n_samples_weighted * cp.log_sum_exp(log_delta_j + L_grid)
        loss = data_term + norm_term

        # Optionally, we include a norm constraint on the targeting coefficients.
        constraints = [cp.norm1(alpha) <= self.norm_constraint]

        objective = cp.Minimize(loss)
        problem = cp.Problem(objective, constraints)

        # Warm start: initialize alpha (you may change this if you have a previous guess)
        alpha.value = np.zeros(r)

        # Solve the optimization problem (here using the SCS solver)
        try:
            problem.solve(solver="MOSEK", warm_start=True)
        except:
            try:
                problem.solve(solver="ECOS", warm_start=True)
            except:
                problem.solve(solver="ECOS", warm_start=True)


        end_time_m_optimize = time.time()
        m_step_time_optimize = end_time_m_optimize - start_time_m_optimize
        # print(f"M-step CVXPY optimize time (sec): {m_step_time_optimize:.4f}")

        # ----------------------------
        # 3. Compute the estimated density on a fine grid
        # ----------------------------
        # Retrieve the optimized targeting coefficients
        theta_targeting = alpha.value

        # The fixed hal-basis parameters remain the same:
        theta_intercept = old_theta[0]
        theta_selected = old_theta[1:]

        # Now, evaluate the density on a fine grid (here 2000 points)
        grid_eval_fine = np.linspace(0, 1, 2001)
        combined_grid_fine = np.concatenate((grid_eval_fine, grid_points_hal_selected))
        filtered_grid_fine = combined_grid_fine[combined_grid_fine >= 0]
        filtered_grid_fine = np.unique(filtered_grid_fine)
        grid_eval_full = np.sort(filtered_grid_fine)

        grid_midpoints_full = (grid_eval_full[:-1] + grid_eval_full[1:]) / 2
        data_grid_full = pd.DataFrame({'W1': grid_midpoints_full})

        # Evaluate hal basis on the fine grid
        basis_grid_tensor_full, _ = create_basis_functions(data_grid_full, grid_points_hal,
                                                    order=self.basis_order, include_intercept=False)
        b_jk_full = basis_grid_tensor_full.numpy()

        # Evaluate targeting basis on the fine grid
        b_jk_targeting_full = self.get_b_jk_targeting_full(data_grid_full, **kwargs)

        delta_j_full = grid_eval_full[1:] - grid_eval_full[:-1]

        # The estimated log-density is the sum of the fixed (hal) part and the optimized targeting part.
        estimated_log_density = theta_intercept + b_jk_full @ theta_selected + b_jk_targeting_full @ theta_targeting
        estimated_density = np.exp(estimated_log_density)
        estimated_density /= np.sum(estimated_density * delta_j_full)  # Normalize so it integrates to 1

        # ----------------------------
        # 4. Package and return results
        # ----------------------------
        updated_results = {
            "old_theta": old_theta,               # fixed intercept and hal-basis coefficients
            "theta_targeting": theta_targeting,     # newly optimized targeting coefficients
            "theta_intercept": theta_intercept,
            "theta_selected": theta_selected,
            "estimated_density": estimated_density,
            "grid_midpoints": grid_midpoints_full,
            "grid_points_hal_selected": grid_points_hal_selected,
            "b_jk": b_jk_full,
            "b_jk_targeting": b_jk_targeting_full,
            "delta_j": delta_j_full,
            "grid_eval": grid_eval_full,
        }

        return updated_results
    
    @abstractmethod
    def get_estimand_variance(self, targeted_fit: Dict[str, Any], uncensored_data: pd.DataFrame, **kwargs) -> Union[float, np.ndarray]:
        """
        Compute the variance of the estimand based on the efficient influence curve (EIC).
        
        This method should be implemented in subclasses.
        
        Parameters
        ----------
        targeted_fit : Dict[str, Any]
            The results dictionary from the `run_m_step` method.
        uncensored_data : pd.DataFrame
            The uncensored data points, used to construct the EIC for each observation.
        **kwargs
            Additional method-specific parameters (e.g., `targeting_points`).
            
        Returns
        -------
        Union[float, np.ndarray]
            The estimated variance(s) of the estimand(s).
        """
        raise NotImplementedError("This method should be implemented in subclasses.")