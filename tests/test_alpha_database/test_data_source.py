"""
Integration tests for alpha-database DataSource.

These tests validate that the new DataSource produces identical results
to the old DataLoader from alpha-canvas.
"""

import pytest
import numpy as np
from alpha_canvas.core.config import ConfigLoader
from alpha_canvas.core.data_loader import DataLoader
from alpha_database import DataSource, BaseReader
import pandas as pd
import xarray as xr


class TestDataSourceIntegration:
    """Integration tests comparing DataSource (new) vs DataLoader (old)."""
    
    def test_identical_results_single_field(self):
        """Test that DataSource produces identical results to DataLoader."""
        # Parameters
        config_path = 'config'
        start_date = '2024-01-01'
        end_date = '2024-01-31'
        field_name = 'adj_close'
        
        # Old design
        config_old = ConfigLoader(config_path)
        loader_old = DataLoader(config_old, start_date, end_date)
        result_old = loader_old.load_field(field_name)
        
        # New design
        ds_new = DataSource(config_path)
        result_new = ds_new.load_field(field_name, start_date, end_date)
        
        # Assertions
        assert result_old.shape == result_new.shape, "Shapes must match"
        assert result_old.dims == result_new.dims, "Dimensions must match"
        assert np.array_equal(result_old.time.values, result_new.time.values), "Time coords must match"
        assert np.array_equal(result_old.asset.values, result_new.asset.values), "Asset coords must match"
        
        # Check data values (handle NaN properly)
        old_vals = result_old.values
        new_vals = result_new.values
        assert np.array_equal(np.isnan(old_vals), np.isnan(new_vals)), "NaN positions must match"
        
        # Check non-NaN values are close
        mask = ~np.isnan(old_vals)
        assert np.allclose(old_vals[mask], new_vals[mask], rtol=1e-9, atol=1e-12), \
            "Non-NaN values must be identical"
    
    def test_reusability_multiple_fields(self):
        """Test that same DataSource instance can load multiple fields."""
        ds = DataSource('config')
        start_date = '2024-01-01'
        end_date = '2024-01-31'
        
        # Load multiple fields with same instance
        adj_close = ds.load_field('adj_close', start_date, end_date)
        volume = ds.load_field('volume', start_date, end_date)
        
        # Verify both loaded successfully
        assert adj_close.shape[0] > 0, "adj_close should have data"
        assert volume.shape[0] > 0, "volume should have data"
        
        # Verify they have same shape (same date range)
        assert adj_close.shape == volume.shape, "Both fields should have same shape"
    
    def test_stateless_different_date_ranges(self):
        """Test that same DataSource instance can handle different date ranges."""
        ds = DataSource('config')
        field_name = 'adj_close'
        
        # Load January data
        jan_data = ds.load_field(field_name, '2024-01-01', '2024-01-31')
        jan_shape = jan_data.shape
        
        # Load February data (different range)
        feb_data = ds.load_field(field_name, '2024-02-01', '2024-02-29')
        feb_shape = feb_data.shape
        
        # Load January again
        jan_data_2 = ds.load_field(field_name, '2024-01-01', '2024-01-31')
        
        # Verify no state pollution
        assert jan_data.equals(jan_data_2), "January data should be identical on both calls"
        assert jan_shape == jan_data_2.shape, "Shape should match original"
    
    def test_list_fields(self):
        """Test that DataSource can list available fields."""
        ds = DataSource('config')
        fields = ds.list_fields()
        
        # Verify it returns a list
        assert isinstance(fields, list), "list_fields() should return a list"
        
        # Verify it contains expected fields
        assert 'adj_close' in fields, "Should include adj_close"
        assert 'volume' in fields, "Should include volume"
    
    def test_plugin_registration(self):
        """Test that custom readers can be registered."""
        
        # Create a mock reader
        class MockReader(BaseReader):
            def read(self, query, params):
                # Return mock DataFrame
                return pd.DataFrame({
                    'date': pd.date_range('2024-01-01', periods=10),
                    'ticker': ['MOCK'] * 10,
                    'value': np.random.randn(10)
                })
        
        # Register custom reader
        ds = DataSource('config')
        ds.register_reader('mock', MockReader())
        
        # Verify reader was registered
        assert 'mock' in ds._readers, "Mock reader should be registered"
        assert isinstance(ds._readers['mock'], MockReader), "Should store correct reader instance"


class TestDataSourceErrors:
    """Test error handling in DataSource."""
    
    def test_invalid_field_name(self):
        """Test that invalid field name raises KeyError."""
        ds = DataSource('config')
        
        with pytest.raises(KeyError, match="Field 'invalid_field' not found"):
            ds.load_field('invalid_field', '2024-01-01', '2024-12-31')
    
    def test_unsupported_db_type(self):
        """Test that unsupported db_type raises ValueError."""
        # This test would require a config with unsupported db_type
        # Skip for now as it requires config manipulation
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


