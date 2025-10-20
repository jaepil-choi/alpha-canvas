"""
Tests for DataLoader class.

DataLoader is responsible for loading data from Parquet files using
DuckDB queries and converting them to (T, N) xarray.DataArray.
"""

import pytest
import xarray as xr
import pandas as pd
from pathlib import Path
from alpha_canvas.core.config import ConfigLoader
from alpha_canvas.core.data_loader import DataLoader


class TestDataLoaderInit:
    """Test DataLoader initialization."""
    
    def test_init_with_config_and_dates(self):
        """DataLoader should initialize with config and date range."""
        config = ConfigLoader('config')
        loader = DataLoader(config, start_date='2024-01-05', end_date='2024-01-15')
        
        assert loader._config is config
        assert loader.start_date == '2024-01-05'
        assert loader.end_date == '2024-01-15'


class TestLoadField:
    """Test loading fields from Parquet files."""
    
    def test_load_parquet_field(self):
        """Test loading a field from Parquet file."""
        # Setup
        config = ConfigLoader('config')
        loader = DataLoader(config, start_date='2024-01-05', end_date='2024-01-15')
        
        # Execute
        result = loader.load_field('adj_close')
        
        # Verify
        assert isinstance(result, xr.DataArray)
        assert result.dims == ('time', 'asset')
        assert result.shape[0] == 7  # 7 trading days
        assert result.shape[1] == 6  # 6 securities
    
    def test_load_field_returns_correct_data_type(self):
        """Test that loaded field has correct dtype."""
        config = ConfigLoader('config')
        loader = DataLoader(config, start_date='2024-01-05', end_date='2024-01-15')
        
        result = loader.load_field('adj_close')
        
        assert result.dtype == 'float64'
    
    def test_load_field_has_correct_coordinates(self):
        """Test that loaded field has properly labeled coordinates."""
        config = ConfigLoader('config')
        loader = DataLoader(config, start_date='2024-01-05', end_date='2024-01-15')
        
        result = loader.load_field('adj_close')
        
        assert 'time' in result.coords
        assert 'asset' in result.coords
        assert len(result.coords['time']) == 7
        assert len(result.coords['asset']) == 6
    
    def test_load_volume_field(self):
        """Test loading volume field (integer data)."""
        config = ConfigLoader('config')
        loader = DataLoader(config, start_date='2024-01-05', end_date='2024-01-15')
        
        result = loader.load_field('volume')
        
        assert isinstance(result, xr.DataArray)
        assert result.dims == ('time', 'asset')
        assert result.dtype == 'int64'
    
    def test_load_nonexistent_field_raises_error(self):
        """Test that loading non-existent field raises KeyError."""
        config = ConfigLoader('config')
        loader = DataLoader(config, start_date='2024-01-05', end_date='2024-01-15')
        
        with pytest.raises(KeyError):
            loader.load_field('nonexistent_field')


class TestQueryWithDateParameters:
    """Test date parameter substitution in queries."""
    
    def test_query_with_date_parameters(self):
        """Test that date parameters are substituted correctly."""
        config = ConfigLoader('config')
        loader = DataLoader(config, start_date='2024-01-05', end_date='2024-01-10')
        
        result = loader.load_field('adj_close')
        
        # Verify date range is respected
        dates = result.coords['time'].values
        assert str(dates[0]) >= '2024-01-05'
        assert str(dates[-1]) <= '2024-01-10'
    
    def test_different_date_ranges(self):
        """Test loading with different date ranges."""
        config = ConfigLoader('config')
        
        # Load wide range
        loader1 = DataLoader(config, start_date='2024-01-05', end_date='2024-01-15')
        result1 = loader1.load_field('adj_close')
        
        # Load narrow range
        loader2 = DataLoader(config, start_date='2024-01-08', end_date='2024-01-10')
        result2 = loader2.load_field('adj_close')
        
        # Narrow range should have fewer time points
        assert result2.shape[0] < result1.shape[0]


class TestPivotToXarray:
    """Test internal pivot operation."""
    
    def test_pivot_long_to_wide(self):
        """Test that long format DataFrame is correctly pivoted to wide format."""
        config = ConfigLoader('config')
        loader = DataLoader(config, start_date='2024-01-05', end_date='2024-01-15')
        
        # Load field (internally uses pivot)
        result = loader.load_field('adj_close')
        
        # Verify shape is (T, N), not long format
        assert len(result.shape) == 2
        assert result.shape[0] > 0  # Time dimension
        assert result.shape[1] > 0  # Asset dimension
    
    def test_pivot_preserves_data_integrity(self):
        """Test that pivoting doesn't corrupt data values."""
        config = ConfigLoader('config')
        loader = DataLoader(config, start_date='2024-01-05', end_date='2024-01-15')
        
        result = loader.load_field('adj_close')
        
        # All values should be positive (prices)
        assert (result > 0).all()
        
        # No NaN values (all securities traded every day)
        assert not result.isnull().any()


class TestMultipleFields:
    """Test loading multiple fields."""
    
    def test_load_multiple_fields_same_shape(self):
        """Test that multiple fields have consistent shapes."""
        config = ConfigLoader('config')
        loader = DataLoader(config, start_date='2024-01-05', end_date='2024-01-15')
        
        adj_close = loader.load_field('adj_close')
        volume = loader.load_field('volume')
        
        # Same shape
        assert adj_close.shape == volume.shape
        
        # Same coordinates
        assert all(adj_close.coords['time'] == volume.coords['time'])
        assert all(adj_close.coords['asset'] == volume.coords['asset'])

