"""
Tests for time-series operators.

These tests validate TsMean and future time-series operators (ts_sum, ts_std, etc.)
"""

import pytest
import xarray as xr
import numpy as np
from alpha_canvas.core.expression import Field
from alpha_canvas.ops.timeseries import TsMean
from alpha_canvas.core.visitor import EvaluateVisitor


class TestTsMeanCompute:
    """Test TsMean.compute() method directly (no Visitor).
    
    These tests validate that the operator owns its computation logic
    and can be tested independently without Visitor integration.
    """
    
    def test_compute_basic(self):
        """Test compute() can be called directly without Visitor."""
        # Create test data
        data = xr.DataArray(
            [[1, 2], [3, 4], [5, 6]],
            dims=['time', 'asset'],
            coords={'time': range(3), 'asset': ['A', 'B']}
        )
        
        # Create operator (child is dummy - not used in direct compute)
        operator = TsMean(child=Field('dummy'), window=2)
        
        # Call compute() directly
        result = operator.compute(data)
        
        # Verify computation
        assert result.shape == data.shape
        assert np.isnan(result.values[0, 0])  # First row NaN (incomplete window)
        assert result.values[1, 0] == 2.0     # mean([1, 3]) = 2.0
        assert result.values[1, 1] == 3.0     # mean([2, 4]) = 3.0
        assert result.values[2, 0] == 4.0     # mean([3, 5]) = 4.0
    
    def test_compute_is_pure_function(self):
        """Test compute() is a pure function with no side effects."""
        data = xr.DataArray(
            [[10, 20], [30, 40]],
            dims=['time', 'asset']
        )
        
        operator = TsMean(child=Field('dummy'), window=2)
        
        # Call twice
        result1 = operator.compute(data)
        result2 = operator.compute(data)
        
        # Results should be identical
        assert np.allclose(result1.values, result2.values, equal_nan=True)
        
        # Original data unchanged
        assert data.values[0, 0] == 10
        assert data.values[1, 1] == 40
    
    def test_compute_window_one(self):
        """Test compute() with window=1 returns original data."""
        data = xr.DataArray(
            [[1, 2], [3, 4], [5, 6]],
            dims=['time', 'asset']
        )
        
        operator = TsMean(child=Field('dummy'), window=1)
        result = operator.compute(data)
        
        # window=1 should return original (as float)
        assert np.allclose(result.values, data.values.astype(float))
    
    def test_compute_window_larger_than_length(self):
        """Test compute() with window > length returns all NaN."""
        data = xr.DataArray(
            [[1, 2], [3, 4], [5, 6]],
            dims=['time', 'asset']
        )
        
        operator = TsMean(child=Field('dummy'), window=10)  # window=10, length=3
        result = operator.compute(data)
        
        # All values should be NaN (cannot form any complete window)
        assert np.all(np.isnan(result.values))
    
    def test_compute_with_existing_nans(self):
        """Test compute() handles existing NaN values correctly."""
        data = xr.DataArray(
            [[1, 2], [np.nan, 4], [3, 6]],
            dims=['time', 'asset']
        )
        
        operator = TsMean(child=Field('dummy'), window=2)
        result = operator.compute(data)
        
        # Row 1: Asset A has NaN, so result is NaN
        assert np.isnan(result.values[1, 0])
        
        # Row 1: Asset B is mean([2, 4]) = 3.0
        assert result.values[1, 1] == 3.0


class TestTsMeanExpression:
    """Tests for TsMean Expression creation."""
    
    def test_ts_mean_expression_creation(self):
        """Test TsMean Expression can be created with correct attributes."""
        expr = TsMean(child=Field('returns'), window=3)
        
        assert expr.window == 3
        assert isinstance(expr.child, Field)
        assert expr.child.name == 'returns'
    
    def test_ts_mean_window_validation(self):
        """Test TsMean with different window sizes."""
        # window=1 (minimum)
        expr1 = TsMean(child=Field('returns'), window=1)
        assert expr1.window == 1
        
        # window=252 (typical trading year)
        expr252 = TsMean(child=Field('returns'), window=252)
        assert expr252.window == 252
    
    def test_ts_mean_has_accept_method(self):
        """Test TsMean implements Visitor pattern."""
        expr = TsMean(child=Field('returns'), window=5)
        assert hasattr(expr, 'accept')
        assert callable(expr.accept)


class TestTsMeanComputation:
    """Tests for TsMean computation logic."""
    
    def test_ts_mean_basic_computation(self):
        """Test ts_mean calculates rolling mean correctly."""
        # Create sample (5, 3) data
        data = xr.DataArray(
            [[1, 2, 3],
             [2, 3, 4],
             [3, 4, 5],
             [4, 5, 6],
             [5, 6, 7]],
            dims=['time', 'asset'],
            coords={'time': range(5), 'asset': ['A', 'B', 'C']}
        )
        
        ds = xr.Dataset({'returns': data})
        visitor = EvaluateVisitor(ds)
        
        expr = TsMean(child=Field('returns'), window=3)
        result = visitor.evaluate(expr)
        
        # Check shape preserved
        assert result.shape == (5, 3)
        
        # Check first 2 rows are NaN (window=3, min_periods=3)
        assert np.isnan(result.values[0, 0])
        assert np.isnan(result.values[1, 0])
        
        # Check third row is mean of [1, 2, 3]
        assert result.values[2, 0] == 2.0
        assert result.values[2, 1] == 3.0
        assert result.values[2, 2] == 4.0
        
        # Check fourth row is mean of [2, 3, 4]
        assert result.values[3, 0] == 3.0
        assert result.values[3, 1] == 4.0
        assert result.values[3, 2] == 5.0
        
        # Check fifth row is mean of [3, 4, 5]
        assert result.values[4, 0] == 4.0
        assert result.values[4, 1] == 5.0
        assert result.values[4, 2] == 6.0
    
    def test_ts_mean_cross_sectional_independence(self):
        """Test that each asset is computed independently."""
        # Different values per asset
        data = xr.DataArray(
            [[10, 100, 1000],
             [20, 200, 2000],
             [30, 300, 3000]],
            dims=['time', 'asset'],
            coords={'time': range(3), 'asset': ['A', 'B', 'C']}
        )
        
        ds = xr.Dataset({'values': data})
        visitor = EvaluateVisitor(ds)
        
        expr = TsMean(child=Field('values'), window=2)
        result = visitor.evaluate(expr)
        
        # Check each asset independently
        # Asset A: mean([10, 20]) = 15
        assert result.values[1, 0] == 15.0
        
        # Asset B: mean([100, 200]) = 150
        assert result.values[1, 1] == 150.0
        
        # Asset C: mean([1000, 2000]) = 1500
        assert result.values[1, 2] == 1500.0
        
        # No cross-contamination
        assert result.values[1, 0] != result.values[1, 1]
        assert result.values[1, 1] != result.values[1, 2]
    
    def test_ts_mean_preserves_dimensions(self):
        """Test that output has same dimensions and coords as input."""
        data = xr.DataArray(
            np.random.randn(10, 5),
            dims=['time', 'asset'],
            coords={
                'time': pd.date_range('2024-01-01', periods=10),
                'asset': ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']
            }
        )
        
        ds = xr.Dataset({'returns': data})
        visitor = EvaluateVisitor(ds)
        
        expr = TsMean(child=Field('returns'), window=5)
        result = visitor.evaluate(expr)
        
        # Check dimensions
        assert result.dims == data.dims
        assert result.shape == data.shape
        
        # Check coordinates preserved
        assert list(result.coords['time'].values) == list(data.coords['time'].values)
        assert list(result.coords['asset'].values) == list(data.coords['asset'].values)


class TestTsMeanEdgeCases:
    """Tests for edge cases."""
    
    def test_ts_mean_window_equals_one(self):
        """Test window=1 returns original data."""
        data = xr.DataArray(
            [[1, 2], [3, 4], [5, 6]],
            dims=['time', 'asset'],
            coords={'time': range(3), 'asset': ['A', 'B']}
        )
        
        ds = xr.Dataset({'values': data})
        visitor = EvaluateVisitor(ds)
        
        expr = TsMean(child=Field('values'), window=1)
        result = visitor.evaluate(expr)
        
        # window=1 should return original (as float)
        assert np.allclose(result.values, data.values.astype(float))
    
    def test_ts_mean_window_greater_than_length(self):
        """Test window > T returns all NaN except possibly last row."""
        data = xr.DataArray(
            [[1, 2], [3, 4], [5, 6]],
            dims=['time', 'asset'],
            coords={'time': range(3), 'asset': ['A', 'B']}
        )
        
        ds = xr.Dataset({'values': data})
        visitor = EvaluateVisitor(ds)
        
        expr = TsMean(child=Field('values'), window=10)  # window=10, but T=3
        result = visitor.evaluate(expr)
        
        # All values should be NaN (cannot form any complete window)
        assert np.all(np.isnan(result.values))
    
    def test_ts_mean_with_existing_nans(self):
        """Test rolling mean handles existing NaN values."""
        data = xr.DataArray(
            [[1, 2], [np.nan, 4], [3, 6]],
            dims=['time', 'asset'],
            coords={'time': range(3), 'asset': ['A', 'B']}
        )
        
        ds = xr.Dataset({'values': data})
        visitor = EvaluateVisitor(ds)
        
        expr = TsMean(child=Field('values'), window=2)
        result = visitor.evaluate(expr)
        
        # Row 1: Asset A has NaN (1 + NaN), so result is NaN
        assert np.isnan(result.values[1, 0])
        
        # Row 1: Asset B is mean([2, 4]) = 3.0
        assert result.values[1, 1] == 3.0


class TestTsMeanCaching:
    """Tests for Expression caching behavior."""
    
    def test_ts_mean_caching_creates_steps(self):
        """Test that ts_mean results are cached with correct steps."""
        data = xr.DataArray(
            [[1, 2], [3, 4]],
            dims=['time', 'asset'],
            coords={'time': range(2), 'asset': ['A', 'B']}
        )
        
        ds = xr.Dataset({'returns': data})
        visitor = EvaluateVisitor(ds)
        
        expr = TsMean(child=Field('returns'), window=2)
        visitor.evaluate(expr)
        
        # Should have 2 steps cached: step 0 (Field), step 1 (TsMean)
        assert len(visitor._signal_cache) == 2
        
        # Check step 0 is Field
        assert 'Field' in visitor._signal_cache[0][0]
        
        # Check step 1 is TsMean
        assert 'TsMean' in visitor._signal_cache[1][0]
    
    def test_ts_mean_cached_data_matches_result(self):
        """Test that cached data matches the final result."""
        data = xr.DataArray(
            [[1, 2], [3, 4], [5, 6]],
            dims=['time', 'asset'],
            coords={'time': range(3), 'asset': ['A', 'B']}
        )
        
        ds = xr.Dataset({'returns': data})
        visitor = EvaluateVisitor(ds)
        
        expr = TsMean(child=Field('returns'), window=2)
        result = visitor.evaluate(expr)
        
        # Get cached TsMean result (step 1)
        cached_result = visitor._signal_cache[1][1]
        
        # Should be identical
        assert np.allclose(result.values, cached_result.values, equal_nan=True)


class TestTsMeanIntegration:
    """Integration tests with complex scenarios."""
    
    def test_ts_mean_nested_expression(self):
        """Test ts_mean of ts_mean (nested operators)."""
        data = xr.DataArray(
            [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
            dims=['time', 'asset'],
            coords={'time': range(5), 'asset': ['A', 'B']}
        )
        
        ds = xr.Dataset({'returns': data})
        visitor = EvaluateVisitor(ds)
        
        # ts_mean of ts_mean: first smooth, then smooth again
        inner_expr = TsMean(child=Field('returns'), window=2)
        outer_expr = TsMean(child=inner_expr, window=2)
        
        result = visitor.evaluate(outer_expr)
        
        # Should have 3 steps: Field, inner TsMean, outer TsMean
        assert len(visitor._signal_cache) == 3
        
        # Result shape preserved
        assert result.shape == (5, 2)
        
        # First 2 rows should be NaN (compounding window effects)
        assert np.isnan(result.values[0, 0])
        assert np.isnan(result.values[1, 0])


# Import pandas for date range
import pandas as pd


# ============================================================
# TsAny Operator Tests
# ============================================================

class TestTsAnyCompute:
    """Test TsAny.compute() method directly (unit tests)."""
    
    def test_compute_basic(self):
        """Test basic rolling any functionality."""
        from alpha_canvas.ops.timeseries import TsAny
        from alpha_canvas.core.expression import Field
        
        dates = pd.date_range('2025-01-01', periods=7, freq='D')
        assets = ['A']
        
        # Boolean data: [False, True, False, False, False, True, False]
        # Window=3, any in window:
        #   idx 0-1: NaN (not enough data)
        #   idx 2: any([False, True, False]) = True
        #   idx 3: any([True, False, False]) = True
        #   idx 4: any([False, False, False]) = False
        #   idx 5: any([False, False, True]) = True
        #   idx 6: any([False, True, False]) = True
        data = xr.DataArray(
            [[False], [True], [False], [False], [False], [True], [False]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        op = TsAny(child=Field('dummy'), window=3)
        result = op.compute(data)
        
        # Check shape
        assert result.shape == (7, 1)
        
        # NaNs at start
        assert np.isnan(result.values[0, 0])
        assert np.isnan(result.values[1, 0])
        
        # Rolling any results
        assert result.values[2, 0] == True   # Had True at idx 1
        assert result.values[3, 0] == True   # Had True at idx 1
        assert result.values[4, 0] == False  # No True in window
        assert result.values[5, 0] == True   # Has True at idx 5
        assert result.values[6, 0] == True   # Had True at idx 5
    
    def test_compute_is_pure_function(self):
        """Test that compute doesn't modify input."""
        from alpha_canvas.ops.timeseries import TsAny
        from alpha_canvas.core.expression import Field
        
        dates = pd.date_range('2025-01-01', periods=5, freq='D')
        assets = ['A']
        
        data = xr.DataArray(
            [[True], [False], [True], [False], [False]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        original = data.copy()
        
        op = TsAny(child=Field('dummy'), window=3)
        result = op.compute(data)
        
        # Input unchanged
        assert (data.values == original.values).all()
        
        # Result is different object
        assert result is not data
    
    def test_compute_window_one(self):
        """Test window=1 (identity for boolean)."""
        from alpha_canvas.ops.timeseries import TsAny
        from alpha_canvas.core.expression import Field
        
        dates = pd.date_range('2025-01-01', periods=4, freq='D')
        assets = ['A']
        
        data = xr.DataArray(
            [[True], [False], [True], [False]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        op = TsAny(child=Field('dummy'), window=1)
        result = op.compute(data)
        
        # Window=1 with min_periods=1 should be identity for boolean
        assert result.values[0, 0] == True
        assert result.values[1, 0] == False
        assert result.values[2, 0] == True
        assert result.values[3, 0] == False
    
    def test_compute_multiple_assets(self):
        """Test with multiple assets (cross-sectional independence)."""
        from alpha_canvas.ops.timeseries import TsAny
        from alpha_canvas.core.expression import Field
        
        dates = pd.date_range('2025-01-01', periods=5, freq='D')
        assets = ['A', 'B']
        
        # A: [T, F, F, F, F] → any in window=3: [NaN, NaN, T, F, F]
        #     idx 2: window=[0,1,2] → [T,F,F] → T
        #     idx 3: window=[1,2,3] → [F,F,F] → F
        #     idx 4: window=[2,3,4] → [F,F,F] → F
        # B: [F, F, T, F, F] → any in window=3: [NaN, NaN, T, T, T]
        #     idx 2: window=[0,1,2] → [F,F,T] → T
        #     idx 3: window=[1,2,3] → [F,T,F] → T
        #     idx 4: window=[2,3,4] → [T,F,F] → T
        data = xr.DataArray(
            [[True, False], [False, False], [False, True], [False, False], [False, False]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        op = TsAny(child=Field('dummy'), window=3)
        result = op.compute(data)
        
        # Asset A
        assert np.isnan(result.values[0, 0])
        assert np.isnan(result.values[1, 0])
        assert result.values[2, 0] == True   # Window [0,1,2]: has True at idx 0
        assert result.values[3, 0] == False  # Window [1,2,3]: no True
        assert result.values[4, 0] == False  # Window [2,3,4]: no True
        
        # Asset B
        assert np.isnan(result.values[0, 1])
        assert np.isnan(result.values[1, 1])
        assert result.values[2, 1] == True   # Has True at idx 2
        assert result.values[3, 1] == True   # Had True at idx 2
        assert result.values[4, 1] == True   # Had True at idx 2
    
    def test_compute_all_false(self):
        """Test window with all False values."""
        from alpha_canvas.ops.timeseries import TsAny
        from alpha_canvas.core.expression import Field
        
        dates = pd.date_range('2025-01-01', periods=5, freq='D')
        assets = ['A']
        
        data = xr.DataArray(
            [[False], [False], [False], [False], [False]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        op = TsAny(child=Field('dummy'), window=3)
        result = op.compute(data)
        
        # Should be False after window is full
        assert np.isnan(result.values[0, 0])
        assert np.isnan(result.values[1, 0])
        assert result.values[2, 0] == False
        assert result.values[3, 0] == False
        assert result.values[4, 0] == False
    
    def test_compute_all_true(self):
        """Test window with all True values."""
        from alpha_canvas.ops.timeseries import TsAny
        from alpha_canvas.core.expression import Field
        
        dates = pd.date_range('2025-01-01', periods=5, freq='D')
        assets = ['A']
        
        data = xr.DataArray(
            [[True], [True], [True], [True], [True]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        op = TsAny(child=Field('dummy'), window=3)
        result = op.compute(data)
        
        # Should be True after window is full
        assert np.isnan(result.values[0, 0])
        assert np.isnan(result.values[1, 0])
        assert result.values[2, 0] == True
        assert result.values[3, 0] == True
        assert result.values[4, 0] == True


class TestTsAnyIntegration:
    """Test TsAny operator with Visitor (integration tests)."""
    
    def test_ts_any_with_visitor(self):
        """Test TsAny evaluation through visitor."""
        from alpha_canvas.ops.timeseries import TsAny
        from alpha_canvas.core.expression import Field
        from alpha_canvas.core.visitor import EvaluateVisitor
        
        dates = pd.date_range('2025-01-01', periods=5, freq='D')
        assets = ['A', 'B']
        
        # Create boolean field
        bool_data = xr.DataArray(
            [[True, False], [False, False], [False, True], [True, False], [False, False]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        ds = xr.Dataset({'condition': bool_data})
        
        # Create expression: ts_any(Field('condition'), 3)
        expr = TsAny(child=Field('condition'), window=3)
        
        # Evaluate
        visitor = EvaluateVisitor(ds)
        result = visitor.evaluate(expr)
        
        # Check result
        assert isinstance(result, xr.DataArray)
        assert result.shape == (5, 2)
        
        # Check some values
        assert np.isnan(result.values[0, 0])  # Not enough data
        assert result.values[2, 0] == True    # Had True in window

