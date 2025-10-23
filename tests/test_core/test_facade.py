"""
Tests for AlphaCanvas facade.

These tests follow TDD methodology - they define expected behavior before implementation.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from alpha_canvas.core.facade import AlphaCanvas
from alpha_canvas.core.expression import Field


class TestAlphaCanvas:
    """Test suite for AlphaCanvas facade."""
    
    def test_alpha_canvas_initialization(self):
        """Test creating AlphaCanvas instance."""
        rc = AlphaCanvas(config_dir='config')
        
        assert rc is not None
        assert rc.db is not None
        assert isinstance(rc.db, xr.Dataset)
    
    def test_default_time_and_asset_indices(self):
        """Test AlphaCanvas creates default indices."""
        rc = AlphaCanvas(config_dir='config')
        
        # Should have default time and asset dimensions
        assert 'time' in rc.db.sizes
        assert 'asset' in rc.db.sizes
        assert rc.db.sizes['time'] > 0
        assert rc.db.sizes['asset'] > 0
    
    def test_custom_time_and_asset_indices(self):
        """Test AlphaCanvas with custom indices."""
        time_idx = pd.date_range('2021-01-01', periods=200)
        asset_idx = ['AAPL', 'GOOGL', 'MSFT']
        
        rc = AlphaCanvas(
            config_dir='config',
            time_index=time_idx,
            asset_index=asset_idx
        )
        
        assert rc.db.sizes['time'] == 200
        assert rc.db.sizes['asset'] == 3
    
    def test_add_data_with_dataarray(self):
        """Test adding DataArray directly (inject pattern)."""
        rc = AlphaCanvas(config_dir='config')
        
        # Create test data
        time_idx = list(rc.db.coords['time'].values)
        asset_idx = list(rc.db.coords['asset'].values)
        
        data = xr.DataArray(
            np.random.randn(len(time_idx), len(asset_idx)),
            dims=['time', 'asset'],
            coords={'time': time_idx, 'asset': asset_idx}
        )
        
        rc.add_data('test_field', data)
        
        assert 'test_field' in rc.db.data_vars
        assert rc.db['test_field'].shape == data.shape
    
    def test_add_data_with_expression(self):
        """Test adding data via Expression evaluation."""
        rc = AlphaCanvas(config_dir='config')
        
        # First add some data
        time_idx = list(rc.db.coords['time'].values)
        asset_idx = list(rc.db.coords['asset'].values)
        
        returns_data = xr.DataArray(
            np.random.randn(len(time_idx), len(asset_idx)),
            dims=['time', 'asset'],
            coords={'time': time_idx, 'asset': asset_idx}
        )
        rc.add_data('returns', returns_data)
        
        # Now add via Expression
        field_expr = Field('returns')
        rc.add_data('returns_copy', field_expr)
        
        assert 'returns_copy' in rc.db.data_vars
        assert 'returns_copy' in rc.rules
        assert rc.rules['returns_copy'] == field_expr
    
    def test_rules_dict_stores_expressions(self):
        """Test that expressions are stored in rules dict."""
        rc = AlphaCanvas(config_dir='config')
        
        # Add data first
        time_idx = list(rc.db.coords['time'].values)
        asset_idx = list(rc.db.coords['asset'].values)
        
        data = xr.DataArray(
            np.random.randn(len(time_idx), len(asset_idx)),
            dims=['time', 'asset'],
            coords={'time': time_idx, 'asset': asset_idx}
        )
        rc.add_data('base_data', data)
        
        # Add expression
        expr = Field('base_data')
        rc.add_data('derived', expr)
        
        assert 'derived' in rc.rules
        assert isinstance(rc.rules['derived'], Field)
    
    def test_db_property_returns_pure_dataset(self):
        """Test that db property returns pure xarray.Dataset."""
        rc = AlphaCanvas(config_dir='config')
        ds = rc.db
        
        assert type(ds) == xr.Dataset  # Exact type, not subclass
        assert not isinstance(ds, AlphaCanvas)
    
    def test_config_loader_accessible(self):
        """Test that ConfigLoader is accessible."""
        rc = AlphaCanvas(config_dir='config')
        
        # Should have config
        assert hasattr(rc, '_config')
        fields = rc._config.list_fields()
        assert len(fields) > 0
    
    def test_evaluator_syncs_with_dataset(self):
        """Test that evaluator stays synced with dataset changes."""
        rc = AlphaCanvas(config_dir='config')
        
        # Add data
        time_idx = list(rc.db.coords['time'].values)
        asset_idx = list(rc.db.coords['asset'].values)
        
        data1 = xr.DataArray(
            np.random.randn(len(time_idx), len(asset_idx)),
            dims=['time', 'asset'],
            coords={'time': time_idx, 'asset': asset_idx}
        )
        rc.add_data('data1', data1)
        
        # Add expression that references data1
        field_expr = Field('data1')
        rc.add_data('data1_copy', field_expr)
        
        # Should work without error (evaluator has updated dataset)
        assert 'data1_copy' in rc.db.data_vars
    
    def test_multiple_data_additions(self):
        """Test adding multiple data variables sequentially."""
        rc = AlphaCanvas(config_dir='config')
        
        time_idx = list(rc.db.coords['time'].values)
        asset_idx = list(rc.db.coords['asset'].values)
        
        # Add multiple data vars
        for i in range(5):
            data = xr.DataArray(
                np.random.randn(len(time_idx), len(asset_idx)),
                dims=['time', 'asset'],
                coords={'time': time_idx, 'asset': asset_idx}
            )
            rc.add_data(f'var_{i}', data)
        
        assert len(rc.db.data_vars) == 5
        for i in range(5):
            assert f'var_{i}' in rc.db.data_vars


