"""Operators module for alpha-canvas.

This module contains all operator implementations organized by category:
- timeseries: Time-series operators (polymorphic: work on DataPanel and future DataTensor)
- crosssection: Cross-sectional operators (Panel-specific)
- logical: Boolean Expression operators (comparisons and logical operations)
- classification: Classification operators for creating axes (cs_quantile, cs_cut)
- transform: Transformation operators (group_neutralize, etc.)
"""

from .timeseries import TsMean, TsAny
from .crosssection import Rank
from .logical import Equals, NotEquals, GreaterThan, LessThan, GreaterOrEqual, LessOrEqual, And, Or, Not

__all__ = [
    'TsMean', 'TsAny', 'Rank',
    'Equals', 'NotEquals', 'GreaterThan', 'LessThan', 'GreaterOrEqual', 'LessOrEqual',
    'And', 'Or', 'Not'
]


