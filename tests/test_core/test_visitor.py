"""
Tests for EvaluateVisitor class.

These tests follow TDD methodology - they define expected behavior before implementation.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from alpha_canvas.core.visitor import EvaluateVisitor
from alpha_canvas.core.expression import Field


class TestEvaluateVisitor:
    """Test suite for EvaluateVisitor class."""
    
    @pytest.fixture
    def test_dataset(self):
        """Fixture providing test dataset."""
        time_idx = pd.date_range('2020-01-01', periods=100)
        asset_idx = [f'ASSET_{i}' for i in range(50)]
        
        ds = xr.Dataset(
            coords={'time': time_idx, 'asset': asset_idx}
        )
        
        # Add test data
        returns_data = xr.DataArray(
            np.random.randn(100, 50),
            dims=['time', 'asset'],
            coords={'time': time_idx, 'asset': asset_idx}
        )
        mcap_data = xr.DataArray(
            np.random.randn(100, 50) * 1000 + 5000,
            dims=['time', 'asset'],
            coords={'time': time_idx, 'asset': asset_idx}
        )
        
        ds = ds.assign({'returns': returns_data, 'mcap': mcap_data})
        return ds
    
    def test_visitor_initialization(self, test_dataset):
        """Test creating EvaluateVisitor with Dataset."""
        visitor = EvaluateVisitor(test_dataset)
        
        assert visitor is not None
        assert visitor._step_counter == 0
        assert len(visitor._signal_cache) == 0
    
    def test_visit_field(self, test_dataset):
        """Test visiting Field expression."""
        visitor = EvaluateVisitor(test_dataset)
        field = Field('returns')
        
        result = visitor.evaluate(field)
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == (100, 50)
        assert visitor._step_counter == 1  # Incremented after caching
        assert 0 in visitor._signal_cache  # Step 0 cached
    
    def test_cache_structure(self, test_dataset):
        """Test cache stores (name, DataArray) tuples."""
        visitor = EvaluateVisitor(test_dataset)
        field = Field('returns')
        
        _ = visitor.evaluate(field)
        
        cached_entry = visitor._signal_cache[0]
        assert isinstance(cached_entry, tuple)
        assert len(cached_entry) == 2
        
        name, data = cached_entry
        assert isinstance(name, str)
        assert 'Field' in name or 'returns' in name
        assert isinstance(data, xr.DataArray)
    
    def test_get_cached(self, test_dataset):
        """Test retrieving cached results."""
        visitor = EvaluateVisitor(test_dataset)
        field = Field('returns')
        
        original_result = visitor.evaluate(field)
        cached_name, cached_data = visitor.get_cached(0)
        
        assert isinstance(cached_name, str)
        assert isinstance(cached_data, xr.DataArray)
        assert cached_data.shape == original_result.shape
    
    def test_step_counter_increments(self, test_dataset):
        """Test step counter increments correctly."""
        visitor = EvaluateVisitor(test_dataset)
        
        assert visitor._step_counter == 0
        
        # Evaluate first field
        _ = visitor.evaluate(Field('returns'))
        assert visitor._step_counter == 1
        
        # Evaluate resets counter
        _ = visitor.evaluate(Field('mcap'))
        assert visitor._step_counter == 1  # Reset by evaluate()
    
    def test_evaluate_resets_state(self, test_dataset):
        """Test that evaluate() resets cache and counter."""
        visitor = EvaluateVisitor(test_dataset)
        
        # First evaluation
        _ = visitor.evaluate(Field('returns'))
        assert visitor._step_counter == 1
        assert len(visitor._signal_cache) == 1
        
        # Second evaluation should reset
        _ = visitor.evaluate(Field('mcap'))
        assert visitor._step_counter == 1  # Reset to 0, then incremented
        assert len(visitor._signal_cache) == 1  # Only one entry from latest eval
        assert 0 in visitor._signal_cache
    
    def test_multiple_fields_sequential(self, test_dataset):
        """Test evaluating multiple fields sequentially (without reset)."""
        visitor = EvaluateVisitor(test_dataset)
        
        # Reset cache manually to test sequential evaluation
        visitor._step_counter = 0
        visitor._signal_cache = {}
        
        # Visit fields directly (bypass evaluate reset)
        field1 = Field('returns')
        field2 = Field('mcap')
        
        _ = field1.accept(visitor)
        _ = field2.accept(visitor)
        
        assert visitor._step_counter == 2
        assert len(visitor._signal_cache) == 2
        assert 0 in visitor._signal_cache
        assert 1 in visitor._signal_cache
        
        # Check both are cached correctly
        name0, data0 = visitor._signal_cache[0]
        name1, data1 = visitor._signal_cache[1]
        
        assert 'returns' in name0
        assert 'mcap' in name1
    
    def test_visitor_with_missing_field(self, test_dataset):
        """Test visiting field that doesn't exist in dataset."""
        visitor = EvaluateVisitor(test_dataset)
        field = Field('nonexistent_field')
        
        with pytest.raises(RuntimeError):
            visitor.evaluate(field)
    
    def test_cache_preserves_data(self, test_dataset):
        """Test that cached data is identical to original result."""
        visitor = EvaluateVisitor(test_dataset)
        field = Field('returns')
        
        result = visitor.evaluate(field)
        cached_name, cached_data = visitor.get_cached(0)
        
        # Check data is identical (same object reference)
        assert cached_data is result
        np.testing.assert_array_equal(cached_data.values, result.values)


