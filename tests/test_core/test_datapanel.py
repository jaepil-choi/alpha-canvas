"""
Tests for DataPanel class.

These tests follow TDD methodology - they define expected behavior before implementation.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from alpha_canvas.core.data_model import DataPanel


class TestDataPanel:
    """Test suite for DataPanel class."""
    
    @pytest.fixture
    def time_index(self):
        """Fixture providing time index."""
        return pd.date_range('2020-01-01', periods=100)
    
    @pytest.fixture
    def asset_index(self):
        """Fixture providing asset index."""
        return [f'ASSET_{i}' for i in range(50)]
    
    def test_datapanel_creation(self, time_index, asset_index):
        """Test creating DataPanel with time and asset indices."""
        panel = DataPanel(time_index, asset_index)
        assert panel is not None
        assert panel.db.sizes['time'] == 100
        assert panel.db.sizes['asset'] == 50
    
    def test_add_float_data(self, time_index, asset_index):
        """Test adding float DataArray."""
        panel = DataPanel(time_index, asset_index)
        returns_data = xr.DataArray(
            np.random.randn(100, 50),
            dims=['time', 'asset'],
            coords={'time': time_index, 'asset': asset_index}
        )
        panel.add_data('returns', returns_data)
        
        assert 'returns' in panel.db.data_vars
        assert panel.db['returns'].dtype == np.float64
        assert panel.db['returns'].shape == (100, 50)
    
    def test_add_categorical_data(self, time_index, asset_index):
        """Test adding string/categorical DataArray."""
        panel = DataPanel(time_index, asset_index)
        size_labels = np.random.choice(['small', 'big'], size=(100, 50))
        size_data = xr.DataArray(
            size_labels,
            dims=['time', 'asset'],
            coords={'time': time_index, 'asset': asset_index}
        )
        panel.add_data('size', size_data)
        
        assert 'size' in panel.db.data_vars
        assert panel.db['size'].shape == (100, 50)
        # Check that we can access unique values
        unique_vals = set(np.unique(panel.db['size'].values))
        assert unique_vals.issubset({'small', 'big'})
    
    def test_eject_pure_dataset(self, time_index, asset_index):
        """Test that db property returns pure xarray.Dataset."""
        panel = DataPanel(time_index, asset_index)
        ds = panel.db
        
        assert isinstance(ds, xr.Dataset)
        assert not isinstance(ds, DataPanel)  # Not wrapped
        assert type(ds) == xr.Dataset  # Exact type match
    
    def test_boolean_indexing(self, time_index, asset_index):
        """Test creating boolean masks from categorical data."""
        panel = DataPanel(time_index, asset_index)
        size_labels = np.random.choice(['small', 'big'], size=(100, 50))
        size_data = xr.DataArray(
            size_labels,
            dims=['time', 'asset'],
            coords={'time': time_index, 'asset': asset_index}
        )
        panel.add_data('size', size_data)
        
        # Create boolean mask
        mask = (panel.db['size'] == 'small')
        
        assert isinstance(mask, xr.DataArray)
        assert mask.dtype == bool
        assert mask.shape == (100, 50)
        # Check that we have both True and False values
        assert mask.any().item()  # At least one True
        assert (~mask).any().item()  # At least one False
    
    def test_dimension_validation(self, time_index, asset_index):
        """Test that add_data validates dimensions."""
        panel = DataPanel(time_index, asset_index)
        
        # Create DataArray with wrong dimensions
        wrong_dims_data = xr.DataArray(
            np.random.randn(100, 50),
            dims=['wrong_dim1', 'wrong_dim2']
        )
        
        with pytest.raises(ValueError) as exc_info:
            panel.add_data('wrong_data', wrong_dims_data)
        
        assert 'dims' in str(exc_info.value).lower()
    
    def test_inject_external_dataarray(self, time_index, asset_index):
        """Test injecting externally created DataArray."""
        panel = DataPanel(time_index, asset_index)
        
        # Create external DataArray (simulating scipy/statsmodels output)
        external_data = xr.DataArray(
            np.random.randn(100, 50),
            dims=['time', 'asset'],
            coords={'time': time_index, 'asset': asset_index}
        )
        
        # Inject should work seamlessly
        panel.add_data('external_beta', external_data)
        
        assert 'external_beta' in panel.db.data_vars
        assert panel.db['external_beta'].shape == (100, 50)
    
    def test_multiple_data_vars(self, time_index, asset_index):
        """Test adding multiple data variables."""
        panel = DataPanel(time_index, asset_index)
        
        # Add multiple vars
        for i in range(3):
            data = xr.DataArray(
                np.random.randn(100, 50),
                dims=['time', 'asset'],
                coords={'time': time_index, 'asset': asset_index}
            )
            panel.add_data(f'var_{i}', data)
        
        assert len(panel.db.data_vars) == 3
        assert 'var_0' in panel.db.data_vars
        assert 'var_1' in panel.db.data_vars
        assert 'var_2' in panel.db.data_vars


