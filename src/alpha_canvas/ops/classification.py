"""Classification operators for bucketing and categorization."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
import xarray as xr

from alpha_canvas.core.expression import Expression


@dataclass(eq=False)
class CsQuantile(Expression):
    """Cross-sectional quantile bucketing - returns categorical labels.
    
    Preserves input (T, N) shape. Each timestep is independently bucketed.
    Supports both independent sort (whole universe) and dependent sort
    (within groups via group_by parameter).
    
    Args:
        child: Expression to bucket (e.g., Field('market_cap'))
        bins: Number of quantile buckets
        labels: List of string labels (must have length == bins)
        group_by: Optional field name for dependent sort
    
    Returns:
        Categorical DataArray with same (T, N) shape as input
    
    Example:
        # Independent sort (whole universe)
        size = CsQuantile(
            child=Field('market_cap'),
            bins=2,
            labels=['small', 'big']
        )
        
        # Dependent sort (within size groups)
        value = CsQuantile(
            child=Field('book_to_market'),
            bins=3,
            labels=['low', 'mid', 'high'],
            group_by='size'
        )
    """
    
    child: Expression
    bins: int
    labels: List[str]
    group_by: Optional[str] = None
    
    def __post_init__(self):
        """Validate parameters."""
        if len(self.labels) != self.bins:
            raise ValueError(
                f"labels length ({len(self.labels)}) must equal bins ({self.bins})"
            )
    
    def accept(self, visitor):
        """Visitor interface."""
        return visitor.visit_operator(self)
    
    def compute(
        self, 
        child_result: xr.DataArray, 
        group_labels: Optional[xr.DataArray] = None
    ) -> xr.DataArray:
        """Apply quantile bucketing.
        
        Args:
            child_result: Input data to bucket
            group_labels: Optional group labels for dependent sort
        
        Returns:
            Categorical DataArray with same (T, N) shape as input
        """
        if group_labels is None:
            return self._quantile_independent(child_result)
        else:
            return self._quantile_grouped(child_result, group_labels)
    
    def _quantile_independent(self, data: xr.DataArray) -> xr.DataArray:
        """Independent sort - qcut at each timestep across all assets.
        
        Args:
            data: Input DataArray with (time, asset) dimensions
        
        Returns:
            Categorical DataArray with same shape
        """
        def qcut_at_timestep(data_slice):
            """Apply pd.qcut to a single timestep's cross-section."""
            try:
                # Flatten to 1D for pd.qcut
                values_1d = data_slice.values.flatten()
                result = pd.qcut(
                    values_1d, 
                    q=self.bins, 
                    labels=self.labels, 
                    duplicates='drop'
                )
                # Reshape back to original shape
                result_array = np.array(result).reshape(data_slice.shape)
                return xr.DataArray(
                    result_array, 
                    dims=data_slice.dims, 
                    coords=data_slice.coords
                )
            except Exception:
                # Edge case: all same values, all NaN, etc.
                # Return NaN array with same shape
                return xr.DataArray(
                    np.full_like(data_slice.values, np.nan, dtype=object),
                    dims=data_slice.dims, 
                    coords=data_slice.coords
                )
        
        # Apply qcut at each timestep, xarray concatenates back
        result = data.groupby('time').map(qcut_at_timestep)
        return result
    
    def _quantile_grouped(
        self, 
        data: xr.DataArray, 
        groups: xr.DataArray
    ) -> xr.DataArray:
        """Dependent sort - qcut within each group at each timestep.
        
        This implements the nested groupby pattern:
        data.groupby(groups) → for each group → groupby('time') → qcut
        
        Args:
            data: Input DataArray with (time, asset) dimensions
            groups: Group labels DataArray with same shape
        
        Returns:
            Categorical DataArray with same shape
        """
        def apply_qcut_within_group(group_data: xr.DataArray) -> xr.DataArray:
            """Apply qcut at each timestep within this group."""
            return self._quantile_independent(group_data)
        
        # Nested groupby: groups → time → qcut
        # xarray automatically concatenates results back to (T, N) shape
        result = data.groupby(groups).map(apply_qcut_within_group)
        return result

