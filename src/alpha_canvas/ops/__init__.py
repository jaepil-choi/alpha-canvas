"""Operators module for alpha-canvas.

This module contains all operator implementations organized by category:
- timeseries: Time-series operators (polymorphic: work on DataPanel and future DataTensor)
- crosssection: Cross-sectional operators (Panel-specific)
- classification: Classification operators for creating axes (cs_quantile, cs_cut)
- transform: Transformation operators (group_neutralize, etc.)
"""

from .timeseries import TsMean

__all__ = ['TsMean']


