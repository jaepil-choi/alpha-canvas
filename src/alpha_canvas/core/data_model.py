"""
Data model for alpha-canvas.

This module provides the DataPanel class, a thin wrapper around xarray.Dataset
for (T, N) panel data operations.
"""

import xarray as xr


class DataPanel:
    """Wrapper around xarray.Dataset for (T, N) panel data.
    
    DataPanel provides a consistent interface for managing time-series cross-sectional
    data with (time, asset) dimensions. It supports the "Open Toolkit" philosophy,
    allowing seamless data ejection to pure xarray and injection of external data.
    
    Attributes:
        _dataset: Internal xarray.Dataset storing all data variables
    
    Example:
        >>> import pandas as pd
        >>> time_idx = pd.date_range('2020-01-01', periods=100)
        >>> asset_idx = ['AAPL', 'GOOGL', 'MSFT']
        >>> panel = DataPanel(time_idx, asset_idx)
        >>> 
        >>> # Add data
        >>> returns = xr.DataArray(data, dims=['time', 'asset'], coords=...)
        >>> panel.add_data('returns', returns)
        >>> 
        >>> # Eject for external manipulation
        >>> pure_ds = panel.db
        >>> 
        >>> # Inject external data
        >>> beta = compute_with_scipy(pure_ds['returns'])
        >>> panel.add_data('beta', beta)
    """
    
    def __init__(self, time_index, asset_index):
        """Initialize DataPanel with time and asset indices.
        
        Args:
            time_index: Array-like time index (e.g., pd.DatetimeIndex)
            asset_index: Array-like asset identifiers (e.g., list of tickers)
        
        Example:
            >>> time_idx = pd.date_range('2020-01-01', periods=252)
            >>> assets = ['AAPL', 'GOOGL', 'MSFT']
            >>> panel = DataPanel(time_idx, assets)
        """
        self._dataset = xr.Dataset(
            coords={
                'time': time_index,
                'asset': asset_index
            }
        )
    
    def add_data(self, name: str, data: xr.DataArray):
        """Add data variable to the dataset.
        
        This method adds a new data variable to the internal Dataset using the
        assign pattern. It validates that the provided DataArray has the correct
        dimensions (time, asset).
        
        Args:
            name: Name for the data variable
            data: xarray.DataArray with dims=['time', 'asset']
        
        Raises:
            ValueError: If data does not have dims ['time', 'asset']
        
        Example:
            >>> returns = xr.DataArray(
            ...     np.random.randn(252, 3),
            ...     dims=['time', 'asset'],
            ...     coords={'time': time_idx, 'asset': assets}
            ... )
            >>> panel.add_data('returns', returns)
        """
        # Validate dimensions match
        if not set(data.dims) == {'time', 'asset'}:
            raise ValueError(
                f"DataArray must have dims ['time', 'asset'], got {list(data.dims)}"
            )
        
        # Use assign to add data_var (immutable pattern)
        self._dataset = self._dataset.assign({name: data})
    
    @property
    def db(self) -> xr.Dataset:
        """Eject: Return pure xarray.Dataset.
        
        This property implements the "Eject" part of the Open Toolkit philosophy.
        It returns the internal Dataset without any wrapping, allowing users to
        manipulate it with standard xarray operations or external libraries like
        scipy, statsmodels, etc.
        
        Returns:
            Pure xarray.Dataset (not wrapped)
        
        Example:
            >>> # Eject for external manipulation
            >>> pure_ds = panel.db
            >>> # Use with scipy, statsmodels, etc.
            >>> betas = run_regression(pure_ds['returns'], pure_ds['market'])
            >>> # Inject results back
            >>> panel.add_data('beta', betas)
        """
        return self._dataset


