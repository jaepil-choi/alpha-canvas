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
from alpha_database import DataSource


class TestAlphaCanvas:
    """Test suite for AlphaCanvas facade."""
    
    def test_alpha_canvas_initialization(self):
        """Test creating AlphaCanvas instance."""
        ds = DataSource('config')
        rc = AlphaCanvas(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        assert rc is not None
        assert rc.db is not None
        assert isinstance(rc.db, xr.Dataset)
    
    # REMOVED: test_default_time_and_asset_indices
    # Reason: time_index/asset_index parameters removed (data_source is mandatory)
    
    # REMOVED: test_custom_time_and_asset_indices
    # Reason: time_index/asset_index parameters removed (data_source is mandatory)
    
    def test_add_data_with_dataarray(self):
        """Test adding DataArray directly (inject pattern)."""
        ds = DataSource('config')
        rc = AlphaCanvas(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        # Create test data matching loaded data dimensions
        # Load a field first to establish dimensions
        sample_field = ds.load_field('adj_close', '2024-01-01', '2024-01-31')
        time_idx = list(sample_field.coords['time'].values)
        asset_idx = list(sample_field.coords['asset'].values)
        
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
        ds = DataSource('config')
        rc = AlphaCanvas(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        # Get dimensions from a sample field
        sample_field = ds.load_field('adj_close', '2024-01-01', '2024-01-31')
        time_idx = list(sample_field.coords['time'].values)
        asset_idx = list(sample_field.coords['asset'].values)
        
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
        ds = DataSource('config')
        rc = AlphaCanvas(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        # Get dimensions
        sample_field = ds.load_field('adj_close', '2024-01-01', '2024-01-31')
        time_idx = list(sample_field.coords['time'].values)
        asset_idx = list(sample_field.coords['asset'].values)
        
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
        ds = DataSource('config')
        rc = AlphaCanvas(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        dataset = rc.db
        
        assert type(dataset) == xr.Dataset  # Exact type, not subclass
        assert not isinstance(dataset, AlphaCanvas)
    
    def test_evaluator_syncs_with_dataset(self):
        """Test that evaluator stays synced with dataset changes."""
        ds = DataSource('config')
        rc = AlphaCanvas(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        # Get dimensions
        sample_field = ds.load_field('adj_close', '2024-01-01', '2024-01-31')
        time_idx = list(sample_field.coords['time'].values)
        asset_idx = list(sample_field.coords['asset'].values)
        
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
        ds = DataSource('config')
        rc = AlphaCanvas(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        # Get dimensions
        sample_field = ds.load_field('adj_close', '2024-01-01', '2024-01-31')
        time_idx = list(sample_field.coords['time'].values)
        asset_idx = list(sample_field.coords['asset'].values)
        
        # Add multiple data vars
        for i in range(5):
            data = xr.DataArray(
                np.random.randn(len(time_idx), len(asset_idx)),
                dims=['time', 'asset'],
                coords={'time': time_idx, 'asset': asset_idx}
            )
            rc.add_data(f'var_{i}', data)
        
        # Check that all 5 test variables exist (may have additional auto-loaded vars like returns)
        assert len(rc.db.data_vars) >= 5
        for i in range(5):
            assert f'var_{i}' in rc.db.data_vars


