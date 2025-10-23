"""
Stateless data loader for alpha-database.

This module provides transformation utilities for converting DataFrames to xarray.DataArray.
Unlike alpha-canvas's DataLoader, this is stateless (no dates in constructor).
"""

import pandas as pd
import xarray as xr


class DataLoader:
    """Stateless data loader - transforms DataFrames to xarray without storing state.
    
    This is a pure transformation utility. It does not store config, dates, or any state.
    All inputs are explicitly passed as method parameters.
    
    Example:
        >>> loader = DataLoader()
        >>> data_array = loader.pivot_to_xarray(
        ...     df=long_df,
        ...     time_col='date',
        ...     asset_col='ticker',
        ...     value_col='price'
        ... )
    """
    
    def __init__(self):
        """Initialize stateless data loader.
        
        No parameters needed - this is a stateless utility.
        """
        pass
    
    def pivot_to_xarray(
        self,
        df: pd.DataFrame,
        time_col: str,
        asset_col: str,
        value_col: str
    ) -> xr.DataArray:
        """Pivot long-format DataFrame to (T, N) xarray.DataArray.
        
        Transforms a "long" DataFrame (one row per observation) into a wide xarray
        with time as rows and assets as columns.
        
        Args:
            df: Long-format DataFrame with columns for time, asset, and value
            time_col: Name of the time/date column
            asset_col: Name of the asset/security identifier column
            value_col: Name of the value column
        
        Returns:
            xarray.DataArray with dims=['time', 'asset'] and shape (T, N)
        
        Example:
            Long format:
                | date       | ticker | price  |
                |------------|--------|--------|
                | 2024-01-01 | AAPL   | 100.0  |
                | 2024-01-01 | MSFT   | 200.0  |
                | 2024-01-02 | AAPL   | 101.0  |
                | 2024-01-02 | MSFT   | 201.0  |
            
            Wide format (xarray):
                dims: (time: 2, asset: 2)
                coords:
                    time: [2024-01-01, 2024-01-02]
                    asset: ['AAPL', 'MSFT']
                data:
                    [[100.0, 200.0],
                     [101.0, 201.0]]
        """
        # Pivot DataFrame: index=time, columns=asset, values=value
        wide_df = df.pivot(index=time_col, columns=asset_col, values=value_col)
        
        # Convert to xarray.DataArray
        data_array = xr.DataArray(
            wide_df.values,
            dims=['time', 'asset'],
            coords={
                'time': wide_df.index.values,
                'asset': wide_df.columns.values
            }
        )
        
        return data_array


