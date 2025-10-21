"""Cross-sectional operators for alpha-canvas."""

from dataclasses import dataclass
from alpha_canvas.core.expression import Expression
import numpy as np
import xarray as xr
from scipy.stats import rankdata


@dataclass(eq=False)  # Disable dataclass __eq__ to use Expression operators
class Rank(Expression):
    """Cross-sectional percentile ranking operator.
    
    Ranks values across assets at each time point using percentile method.
    Produces values in [0.0, 1.0] where 0.0 = smallest, 1.0 = largest.
    
    Args:
        child: Expression to rank
    
    Returns:
        DataArray with percentile ranks (0.0 to 1.0).
        NaN values remain NaN.
        
    Example:
        >>> # Rank market cap (small → 0.0, large → 1.0)
        >>> expr = Rank(child=Field('market_cap'))
        >>> rc.add_data('mcap_rank', expr)
        
    Notes:
        - Operates on asset dimension only (cross-sectional)
        - Each time step ranked independently
        - Ascending: smallest value gets 0.0
        - NaN values preserved automatically via scipy
        - Uses ordinal ranking (distinct ranks)
        - Panel-only operator
    """
    child: Expression
    
    def accept(self, visitor):
        """Accept a visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, child_result: 'xr.DataArray') -> 'xr.DataArray':
        """Core computation logic for cross-sectional percentile ranking.
        
        Uses scipy.stats.rankdata with:
            - method='ordinal': Each value gets distinct rank
            - nan_policy='omit': NaNs preserved in output automatically
        
        Then converts to percentiles: (rank - 1) / (n - 1)
        
        Args:
            child_result: Input DataArray from child expression
        
        Returns:
            DataArray with percentile ranks along asset dimension
        """
        result = child_result.copy().astype(float)  # Ensure float dtype
        
        for t in range(child_result.shape[0]):
            row = child_result.values[t, :]
            
            # Rank with scipy (NaN handling automatic)
            ranks = rankdata(row, method='ordinal', nan_policy='omit')
            
            # Convert to percentiles [0.0, 1.0]
            valid_count = np.sum(~np.isnan(row))
            
            if valid_count > 1:
                # Percentile = (rank - 1) / (n - 1)
                # np.where preserves NaN where ranks are NaN
                result.values[t, :] = np.where(
                    np.isnan(ranks),
                    np.nan,
                    (ranks - 1) / (valid_count - 1)
                )
            elif valid_count == 1:
                # Single valid value → 0.5 (middle)
                result.values[t, :] = np.where(np.isnan(row), np.nan, 0.5)
            else:
                # All NaN → keep as NaN
                result.values[t, :] = np.nan
        
        return result

