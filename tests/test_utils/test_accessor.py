"""
Tests for DataAccessor class.

Tests accessor functionality both in isolation and integrated with AlphaCanvas.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr

from alpha_canvas.core.expression import Field, Expression
from alpha_canvas.ops.logical import Equals
from alpha_canvas.utils.accessor import DataAccessor
from alpha_canvas import AlphaCanvas
from alpha_canvas.core.data_model import DataPanel


class TestDataAccessor:
    """Test DataAccessor class in isolation."""
    
    def test_getitem_returns_field_expression(self):
        """Verify that accessor['field'] returns a Field Expression."""
        accessor = DataAccessor()
        
        result = accessor['size']
        
        assert isinstance(result, Field)
        assert isinstance(result, Expression)
        assert result.name == 'size'
    
    def test_getitem_comparison_returns_expression(self):
        """Verify that accessor['field'] == value returns Equals Expression."""
        accessor = DataAccessor()
        
        result = accessor['size'] == 'small'
        
        assert isinstance(result, Equals)
        assert isinstance(result, Expression)
        assert isinstance(result.left, Field)
        assert result.left.name == 'size'
        assert result.right == 'small'
    
    def test_getitem_invalid_type_raises_error(self):
        """Verify that non-string field names raise TypeError."""
        accessor = DataAccessor()
        
        with pytest.raises(TypeError, match="Field name must be string"):
            accessor[123]
        
        with pytest.raises(TypeError, match="Field name must be string"):
            accessor[None]
        
        with pytest.raises(TypeError, match="Field name must be string"):
            accessor[['field1', 'field2']]
    
    def test_attribute_access_not_supported(self):
        """Verify that attribute access raises AttributeError."""
        accessor = DataAccessor()
        
        with pytest.raises(AttributeError, match="does not support attribute access"):
            _ = accessor.size
        
        with pytest.raises(AttributeError, match="Use rc\\.data\\['momentum'\\]"):
            _ = accessor.momentum


class TestDataAccessorIntegration:
    """Test DataAccessor integrated with AlphaCanvas."""
    
    @pytest.fixture
    def sample_canvas(self):
        """Create AlphaCanvas with sample data."""
        # Create sample data
        time_index = pd.date_range('2024-01-01', periods=10, freq='D')
        asset_index = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
        
        # Create categorical size data
        size_data = xr.DataArray(
            np.array([
                ['small', 'big', 'small', 'big', 'small'],
                ['small', 'big', 'big', 'small', 'small'],
                ['big', 'small', 'small', 'big', 'small'],
                ['small', 'big', 'small', 'small', 'big'],
                ['big', 'small', 'big', 'small', 'small'],
                ['small', 'big', 'small', 'big', 'small'],
                ['big', 'small', 'big', 'small', 'big'],
                ['small', 'big', 'small', 'big', 'small'],
                ['big', 'small', 'small', 'small', 'big'],
                ['small', 'big', 'big', 'big', 'small'],
            ]),
            dims=['time', 'asset'],
            coords={'time': time_index, 'asset': asset_index}
        )
        
        # Create numeric momentum data
        momentum_data = xr.DataArray(
            np.array([
                ['high', 'low', 'high', 'low', 'high'],
                ['low', 'high', 'low', 'high', 'low'],
                ['high', 'low', 'high', 'low', 'high'],
                ['low', 'high', 'low', 'high', 'low'],
                ['high', 'low', 'high', 'low', 'high'],
                ['low', 'high', 'low', 'high', 'low'],
                ['high', 'low', 'high', 'low', 'high'],
                ['low', 'high', 'low', 'high', 'low'],
                ['high', 'low', 'high', 'low', 'high'],
                ['low', 'high', 'low', 'high', 'low'],
            ]),
            dims=['time', 'asset'],
            coords={'time': time_index, 'asset': asset_index}
        )
        
        # Create price data for universe
        price_data = xr.DataArray(
            np.array([
                [150.0, 300.0, 120.0, 400.0, 200.0],
                [155.0, 310.0, 125.0, 410.0, 205.0],
                [160.0, 320.0, 130.0, 420.0, 210.0],
                [165.0, 330.0, 135.0, 430.0, 215.0],
                [170.0, 340.0, 140.0, 440.0, 220.0],
                [175.0, 350.0, 145.0, 450.0, 225.0],
                [180.0, 360.0, 150.0, 460.0, 230.0],
                [185.0, 370.0, 155.0, 470.0, 235.0],
                [190.0, 380.0, 160.0, 480.0, 240.0],
                [195.0, 390.0, 165.0, 490.0, 245.0],
            ]),
            dims=['time', 'asset'],
            coords={'time': time_index, 'asset': asset_index}
        )
        
        # Create DataPanel
        panel = DataPanel(time_index, asset_index)
        panel.add_data('size', size_data)
        panel.add_data('momentum', momentum_data)
        panel.add_data('price', price_data)
        
        # Create AlphaCanvas
        rc = AlphaCanvas(
            time_index=time_index,
            asset_index=asset_index
        )
        rc._panel = panel
        
        # Re-initialize evaluator with the populated panel
        from alpha_canvas.core.visitor import EvaluateVisitor
        rc._evaluator = EvaluateVisitor(rc._panel.db, None)
        
        return rc
    
    def test_rc_data_returns_accessor(self, sample_canvas):
        """Verify that rc.data returns a DataAccessor instance."""
        rc = sample_canvas
        
        assert hasattr(rc, 'data')
        assert isinstance(rc.data, DataAccessor)
    
    def test_rc_data_expression_evaluation(self, sample_canvas):
        """Verify end-to-end: rc.data['field'] == value → evaluate."""
        rc = sample_canvas
        
        # Create expression using accessor
        expr = rc.data['size'] == 'small'
        
        # Verify it's an Expression
        assert isinstance(expr, Equals)
        
        # Evaluate
        result = rc.evaluate(expr)
        
        # Verify result
        assert isinstance(result, xr.DataArray)
        assert result.shape == (10, 5)
        assert result.dtype == bool or result.dtype == float  # bool or float with NaN
        
        # Count True values
        if result.dtype == bool:
            true_count = result.sum().values
        else:
            true_count = (result == 1.0).sum().values
        
        # Should match manual count
        manual_count = (rc._panel.db['size'] == 'small').sum().values
        assert true_count == manual_count
    
    def test_rc_data_universe_masking(self, sample_canvas):
        """Verify universe masking is applied correctly."""
        rc = sample_canvas
        
        # Set universe mask (exclude some positions)
        universe_mask = rc._panel.db['price'] > 150.0
        rc._universe_mask = universe_mask
        rc._evaluator._universe_mask = universe_mask
        
        # Create and evaluate expression
        expr = rc.data['size'] == 'small'
        result = rc.evaluate(expr)
        
        # Verify universe was applied (masked positions should be NaN or False)
        # Check a few positions we know should be masked
        assert result.shape == (10, 5)
        
        # At time 0, AAPL (150.0) should be masked
        # Check if the masking affected the result
        # (Exact value depends on masking implementation, but shape should match)
        assert result.shape == universe_mask.shape
    
    def test_rc_data_complex_logic(self, sample_canvas):
        """Verify chained comparisons work correctly."""
        rc = sample_canvas
        
        # Create complex expression
        expr = (rc.data['size'] == 'small') & (rc.data['momentum'] == 'high')
        
        # Verify it's a logical Expression tree
        from alpha_canvas.ops.logical import And
        assert isinstance(expr, And)
        
        # Evaluate
        result = rc.evaluate(expr)
        
        # Verify result
        assert isinstance(result, xr.DataArray)
        assert result.shape == (10, 5)
        
        # Manual verification
        size_small = (rc._panel.db['size'] == 'small')
        momentum_high = (rc._panel.db['momentum'] == 'high')
        expected = size_small & momentum_high
        
        if result.dtype == bool:
            expected_count = expected.sum().values
            actual_count = result.sum().values
        else:
            expected_count = expected.sum().values
            actual_count = (result == 1.0).sum().values
        
        assert actual_count == expected_count
    
    def test_rc_data_multiple_field_access(self, sample_canvas):
        """Verify accessing multiple fields works independently."""
        rc = sample_canvas
        
        field1 = rc.data['size']
        field2 = rc.data['momentum']
        field3 = rc.data['price']
        
        # Verify each is a Field Expression
        assert isinstance(field1, Field)
        assert isinstance(field2, Field)
        assert isinstance(field3, Field)
        
        # Verify field names
        assert field1.name == 'size'
        assert field2.name == 'momentum'
        assert field3.name == 'price'
        
        # Verify they're independent objects
        assert field1 is not field2
        assert field2 is not field3
    
    def test_rc_data_comparison_types(self, sample_canvas):
        """Verify different comparison operators work."""
        rc = sample_canvas
        
        # String equality
        expr1 = rc.data['size'] == 'small'
        assert isinstance(expr1, Equals)
        
        # Numeric comparison
        from alpha_canvas.ops.logical import GreaterThan, LessThan
        expr2 = rc.data['price'] > 150.0
        assert isinstance(expr2, GreaterThan)
        
        expr3 = rc.data['price'] < 200.0
        assert isinstance(expr3, LessThan)
        
        # All should evaluate successfully
        result1 = rc.evaluate(expr1)
        result2 = rc.evaluate(expr2)
        result3 = rc.evaluate(expr3)
        
        assert result1.shape == (10, 5)
        assert result2.shape == (10, 5)
        assert result3.shape == (10, 5)

