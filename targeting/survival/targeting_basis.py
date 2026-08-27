import pandas as pd
import numpy as np
from typing import Union, Tuple
import torch

# Helper function to create targeting basis /clever covariates
def create_cumulative_indicators_g(
        series: pd.Series,
        grid_points: Union[list, np.ndarray],
        prefix: str = 'W1'
) -> pd.DataFrame:
    """
    Create cumulative greater-or-equal indicators based on unequally spaced grid.

    Parameters:
    - series: pd.Series
    - grid_points: array-like, shape (K+1,)
    - prefix: str, prefix for indicator column names

    Returns:
    - indicators: pd.DataFrame
    """
    indicators_dict = {}
    id = 1
    for value in grid_points:
        id += 1
        indicator_col = f"{prefix}_g_{value:.4f}_{id}"
        indicators_dict[indicator_col] = (series > value).astype(int)
    indicators = pd.DataFrame(indicators_dict, index=series.index)

    return indicators

# Generate basis functions for data and evaluation grid
def create_targeting_basis_functions(
        data_long_train: pd.DataFrame,
        grid_points: Union[list, np.ndarray],
        prefix: str = 'W1'
) -> torch.Tensor:
    """    Create targeting basis functions for the data based on cumulative indicators.
    Parameters:
    - data_long_train: pd.DataFrame containing the training data with 'W1' column
    - grid_points: array-like, shape (K+1,), the grid points for cumulative indicators
    - prefix: str, prefix for indicator column names    
    Returns:
    - basis_tensor: torch.Tensor of shape (n, p) where n is the number of observations and p is the number of basis functions
    """
    indicators = create_cumulative_indicators_g(data_long_train['W1'], grid_points, prefix=prefix)
    basis_df = pd.concat([indicators], axis=1)
    basis_tensor = torch.tensor(basis_df.values, dtype=torch.float32)
    return basis_tensor