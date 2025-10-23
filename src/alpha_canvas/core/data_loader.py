"""
DataLoader for loading data from Parquet files using DuckDB.

This module provides the DataLoader class which executes SQL queries
on Parquet files and converts results to (T, N) xarray.DataArray.
"""

import duckdb
import pandas as pd
import xarray as xr
from typing import Dict


class DataLoader:
    """Load data from Parquet files using DuckDB queries.
    
    DataLoader executes SQL queries from config on Parquet files,
    converts long format results to wide format, and wraps them
    in xarray.DataArray with (time, asset) dimensions.
    
    Attributes:
        _config: ConfigLoader instance containing field definitions
        start_date: Start date for data queries (YYYY-MM-DD format)
        end_date: End date for data queries (YYYY-MM-DD format)
    
    Example:
        >>> config = ConfigLoader('config')
        >>> loader = DataLoader(config, start_date='2024-01-01', end_date='2024-12-31')
        >>> adj_close = loader.load_field('adj_close')
        >>> print(adj_close.shape)  # (T, N) shape
        (252, 50)
    """
    
    def __init__(self, config_loader, start_date: str, end_date: str):
        """Initialize DataLoader with config and date range.
        
        Args:
            config_loader: ConfigLoader instance
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Example:
            >>> loader = DataLoader(config, '2024-01-01', '2024-12-31')
        """
        self._config = config_loader
        self.start_date = start_date
        self.end_date = end_date
    
    def load_field(self, field_name: str) -> xr.DataArray:
        """Load field from Parquet and return as (T, N) DataArray.
        
        This method:
        1. Gets field definition from config
        2. Substitutes date parameters in query
        3. Executes query with DuckDB
        4. Pivots long format to wide format
        5. Converts to xarray.DataArray
        
        Args:
            field_name: Name of field to load (e.g., 'adj_close', 'volume')
        
        Returns:
            xarray.DataArray with dims=['time', 'asset']
        
        Raises:
            KeyError: If field_name not found in config
        
        Example:
            >>> adj_close = loader.load_field('adj_close')
            >>> print(adj_close.dims)
            ('time', 'asset')
        """
        # 1. Get field config
        field_def = self._config.get_field(field_name)
        
        # 2. Get query and substitute date parameters
        query = field_def['query']
        query = query.replace(':start_date', f"'{self.start_date}'")
        query = query.replace(':end_date', f"'{self.end_date}'")
        
        # 3. Execute query with DuckDB
        df = duckdb.query(query).to_df()
        
        # 4. Pivot to wide format and convert to xarray
        return self._pivot_to_xarray(df, field_def)
    
    def _pivot_to_xarray(self, df: pd.DataFrame, field_def: Dict) -> xr.DataArray:
        """Pivot long DataFrame to (T, N) xarray.DataArray.
        
        Converts long format DataFrame (rows per security-date pair)
        to wide format (dates as rows, securities as columns) and
        wraps in xarray.DataArray.
        
        Args:
            df: Long format DataFrame with columns [index_col, security_col, value_col]
            field_def: Field definition dict with column name mappings
        
        Returns:
            xarray.DataArray with dims=['time', 'asset']
        
        Example:
            >>> df_long = pd.DataFrame({
            ...     'date': ['2024-01-01', '2024-01-01'],
            ...     'security_id': ['AAPL', 'GOOGL'],
            ...     'adj_close': [150.0, 2800.0]
            ... })
            >>> result = loader._pivot_to_xarray(df_long, field_def)
            >>> print(result.shape)
            (1, 2)
        """
        # Use field_def to get column names
        index_col = field_def['index_col']  # e.g., 'date'
        security_col = field_def['security_col']  # e.g., 'security_id'
        value_col = field_def['value_col']  # e.g., 'adj_close'
        
        # Pivot: rows=date, columns=security_id, values=adj_close
        wide_df = df.pivot(index=index_col, columns=security_col, values=value_col)
        
        # Convert to xarray with proper dimension names
        # Use .values to convert pandas Index to numpy array (avoids xarray coordinate naming issues)
        data_array = xr.DataArray(
            wide_df.values,
            dims=['time', 'asset'],
            coords={
                'time': wide_df.index.values,
                'asset': wide_df.columns.values
            }
        )
        
        return data_array

