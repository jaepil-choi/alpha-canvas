"""
DataSource facade for alpha-database.

This module provides the main user-facing API for loading data from various sources.
"""

import xarray as xr
from typing import Dict
from .config import ConfigLoader
from .data_loader import DataLoader
from ..readers.base import BaseReader
from ..readers.parquet import ParquetReader


class DataSource:
    """Facade for loading data from multiple sources.
    
    DataSource is the main entry point for alpha-database. It coordinates
    configuration, readers, and data transformation to provide a simple API
    for loading data fields.
    
    Key Features:
    - Config-driven: Field definitions loaded from YAML
    - Multi-backend: Supports multiple data sources (Parquet, CSV, PostgreSQL, etc.)
    - Stateless: Date ranges passed per call, not stored
    - Extensible: Custom readers can be registered via plugin mechanism
    
    Example:
        >>> # Basic usage
        >>> ds = DataSource('config/data.yaml')
        >>> data = ds.load_field('adj_close', '2024-01-01', '2024-12-31')
        
        >>> # Plugin registration
        >>> ds.register_reader('custom', CustomReader())
        >>> custom_data = ds.load_field('custom_field', '2024-01-01', '2024-12-31')
    """
    
    def __init__(self, config_path: str = 'config/data.yaml'):
        """Initialize DataSource with config path.
        
        Args:
            config_path: Path to data configuration YAML file or directory
                         If directory, looks for 'data.yaml' inside
                         Default: 'config/data.yaml'
        
        Example:
            >>> # Using default config
            >>> ds = DataSource()
            
            >>> # Using custom config path
            >>> ds = DataSource('custom_config/data.yaml')
        """
        # Determine if config_path is a file or directory
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            # Extract directory path
            config_dir = '/'.join(config_path.split('/')[:-1]) or 'config'
        else:
            # Assume it's a directory
            config_dir = config_path
        
        # Initialize components
        self._config = ConfigLoader(config_dir)
        self._data_loader = DataLoader()
        
        # Register core readers
        self._readers: Dict[str, BaseReader] = {
            'parquet': ParquetReader(),
        }
    
    def register_reader(self, reader_type: str, reader: BaseReader):
        """Register a custom reader for a specific data source type.
        
        This enables the plugin architecture - users or additional packages
        can provide specialized readers (e.g., FnGuideReader, BloombergReader).
        
        Args:
            reader_type: Type identifier (e.g., 'postgres', 'fnguide')
            reader: Reader instance implementing BaseReader interface
        
        Example:
            >>> # Register PostgreSQL reader
            >>> ds.register_reader('postgres', PostgresReader(connection_string))
            
            >>> # Register custom FnGuide reader
            >>> ds.register_reader('fnguide', FnGuideExcelReader())
        """
        self._readers[reader_type] = reader
    
    def load_field(
        self,
        field_name: str,
        start_date: str,
        end_date: str
    ) -> xr.DataArray:
        """Load a data field with specified date range.
        
        This is the main method for loading data. It:
        1. Looks up field configuration
        2. Selects appropriate reader based on db_type
        3. Executes query with date parameters
        4. Transforms result to xarray.DataArray
        
        Args:
            field_name: Name of field to load (e.g., 'adj_close', 'volume')
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
        
        Returns:
            xarray.DataArray with dims=['time', 'asset'] and shape (T, N)
        
        Raises:
            KeyError: If field_name not found in configuration
            ValueError: If db_type not supported (no registered reader)
        
        Example:
            >>> ds = DataSource()
            >>> adj_close = ds.load_field('adj_close', '2024-01-01', '2024-12-31')
            >>> print(adj_close.shape)  # (252, 100) - 252 trading days, 100 assets
            
            >>> # Load multiple fields with same instance
            >>> volume = ds.load_field('volume', '2024-01-01', '2024-12-31')
            >>> market_cap = ds.load_field('market_cap', '2024-01-01', '2024-12-31')
        """
        # Step 1: Get field configuration
        field_config = self._config.get_field(field_name)
        
        # Step 2: Select reader based on db_type
        reader_type = field_config.get('db_type', 'parquet')  # Default to parquet
        if reader_type not in self._readers:
            raise ValueError(
                f"No reader registered for db_type '{reader_type}'. "
                f"Available readers: {list(self._readers.keys())}"
            )
        reader = self._readers[reader_type]
        
        # Step 3: Execute query with parameters
        params = {
            'start_date': start_date,
            'end_date': end_date
        }
        df_long = reader.read(field_config['query'], params)
        
        # Step 4: Pivot to xarray
        data_array = self._data_loader.pivot_to_xarray(
            df=df_long,
            time_col=field_config['index_col'],
            asset_col=field_config['security_col'],
            value_col=field_config['value_col']
        )
        
        return data_array
    
    def list_fields(self):
        """List all available field names.
        
        Returns:
            List of field names configured in data.yaml
        
        Example:
            >>> ds = DataSource()
            >>> fields = ds.list_fields()
            >>> print(fields)  # ['adj_close', 'volume', 'market_cap', ...]
        """
        return self._config.list_fields()

