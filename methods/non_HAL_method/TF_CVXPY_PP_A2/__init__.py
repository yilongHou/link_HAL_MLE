"""
Trend Filtering CVXPY with Parametric Penalty using Algorithm 2 (layered constraints).

This module implements TFPP using a reformulation that avoids explicit H^{-1} computation
by expressing the falling factorial constraints as layered auxiliary variables.
This can improve numerical stability on data-adaptive (irregular) grids.

Two implementations:
1. TrendFilteringCVXPYPPA2 - Dual formulation (still uses H matrix)
2. TrendFilteringCVXPYPPA2Layered - TRUE layered constraints (no matrix formation)
"""

from .estimator import TrendFilteringCVXPYPPA2
from .estimator_layered import TrendFilteringCVXPYPPA2Layered

__all__ = ["TrendFilteringCVXPYPPA2", "TrendFilteringCVXPYPPA2Layered"]

