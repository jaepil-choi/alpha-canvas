"""
Tests for cross-sectional operators.

Following TDD principle: Tests written before implementation,
based on Experiment 12 findings.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpha_canvas.ops.crosssection import Rank
from alpha_canvas.core.expression import Field
from alpha_canvas.core.visitor import EvaluateVisitor


class TestRankCompute:
    """Unit tests for Rank.compute() method (operator owns computation)."""
    
    def test_compute_basic(self):
        """Test basic percentile ranking with simple values."""
        # Arrange: Create data with clear ordering
        dates = pd.date_range('2024-01-01', periods=1, freq='D')
        assets = ['A', 'B', 'C']
        
        data = xr.DataArray(
            [[10.0, 50.0, 30.0]],  # A=smallest, B=largest, C=middle
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        rank_op = Rank(child=None)  # child not needed for compute()
        
        # Act: Call compute directly
        result = rank_op.compute(data)
        
        # Assert: Verify percentile values
        assert result.values[0, 0] == 0.0, "Smallest value should be 0.0"
        assert result.values[0, 1] == 1.0, "Largest value should be 1.0"
        assert result.values[0, 2] == 0.5, "Middle value should be 0.5"
        assert result.dims == data.dims, "Dimensions should be preserved"
    
    def test_compute_ascending(self):
        """Test ascending order: smallest → 0.0, largest → 1.0."""
        dates = pd.date_range('2024-01-01', periods=1, freq='D')
        assets = ['W', 'X', 'Y', 'Z']
        
        # Descending values in data: [100, 75, 50, 25]
        data = xr.DataArray(
            [[100.0, 75.0, 50.0, 25.0]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        rank_op = Rank(child=None)
        result = rank_op.compute(data)
        
        # Find indices of min and max
        min_idx = 3  # Asset 'Z' with value 25
        max_idx = 0  # Asset 'W' with value 100
        
        assert result.values[0, min_idx] == 0.0, "Min value (25) → rank 0.0"
        assert result.values[0, max_idx] == 1.0, "Max value (100) → rank 1.0"
        
        # Verify intermediate values
        np.testing.assert_array_almost_equal(
            result.values[0, :],
            [1.0, 2/3, 1/3, 0.0],
            decimal=6
        )
    
    def test_compute_is_pure_function(self):
        """Test compute() has no side effects (pure function)."""
        dates = pd.date_range('2024-01-01', periods=1, freq='D')
        assets = ['A', 'B', 'C']
        
        data = xr.DataArray(
            [[10.0, 20.0, 30.0]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        original_values = data.values.copy()
        rank_op = Rank(child=None)
        
        # Call compute twice
        result1 = rank_op.compute(data)
        result2 = rank_op.compute(data)
        
        # Assert: Input unchanged, outputs identical
        np.testing.assert_array_equal(data.values, original_values)
        np.testing.assert_array_equal(result1.values, result2.values)
    
    def test_compute_with_nans(self):
        """Test NaN preservation (scipy nan_policy='omit')."""
        dates = pd.date_range('2024-01-01', periods=2, freq='D')
        assets = ['A', 'B', 'C']
        
        data = xr.DataArray(
            [[10.0, np.nan, 30.0],   # Time 0: A=0.0, B=NaN, C=1.0
             [100.0, 200.0, np.nan]], # Time 1: A=0.0, B=1.0, C=NaN
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        rank_op = Rank(child=None)
        result = rank_op.compute(data)
        
        # Assert: NaN preserved in output
        assert result.values[0, 0] == 0.0
        assert np.isnan(result.values[0, 1]), "NaN should remain NaN"
        assert result.values[0, 2] == 1.0
        
        assert result.values[1, 0] == 0.0
        assert result.values[1, 1] == 1.0
        assert np.isnan(result.values[1, 2]), "NaN should remain NaN"
    
    def test_compute_all_nans(self):
        """Test edge case: all NaN row."""
        dates = pd.date_range('2024-01-01', periods=1, freq='D')
        assets = ['A', 'B', 'C']
        
        data = xr.DataArray(
            [[np.nan, np.nan, np.nan]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        rank_op = Rank(child=None)
        result = rank_op.compute(data)
        
        # Assert: All NaN output
        assert np.all(np.isnan(result.values)), "All-NaN input → All-NaN output"
    
    def test_compute_single_asset(self):
        """Test edge case: N=1 (single asset)."""
        dates = pd.date_range('2024-01-01', periods=1, freq='D')
        assets = ['A']
        
        data = xr.DataArray(
            [[42.0]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        rank_op = Rank(child=None)
        result = rank_op.compute(data)
        
        # Assert: Single value → 0.5 (middle of [0, 1])
        assert result.values[0, 0] == 0.5, "Single value should map to 0.5"
    
    def test_compute_time_independence(self):
        """Test each time step is ranked independently."""
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        assets = ['A', 'B', 'C']
        
        data = xr.DataArray(
            [[10, 50, 30],   # Time 0: [0.0, 1.0, 0.5]
             [100, 200, 150], # Time 1: [0.0, 1.0, 0.5]
             [5, 15, 10]],    # Time 2: [0.0, 1.0, 0.5]
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        rank_op = Rank(child=None)
        result = rank_op.compute(data.astype(float))
        
        # Each time step should have same rank pattern
        expected_row = [0.0, 1.0, 0.5]
        for t in range(3):
            np.testing.assert_array_equal(result.values[t, :], expected_row)
    
    def test_compute_with_ties(self):
        """Test ordinal ranking handles ties (no tied ranks)."""
        dates = pd.date_range('2024-01-01', periods=1, freq='D')
        assets = ['A', 'B', 'C', 'D']
        
        # Two equal values: B and C both have 50
        data = xr.DataArray(
            [[10.0, 50.0, 50.0, 90.0]],  # Ordinal: [1, 2, 3, 4]
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        rank_op = Rank(child=None)
        result = rank_op.compute(data)
        
        # Verify no exact ties in output (ordinal method)
        assert result.values[0, 0] == 0.0, "Smallest (10) → 0.0"
        assert result.values[0, 3] == 1.0, "Largest (90) → 1.0"
        
        # Two 50s get consecutive ranks (2, 3) → (1/3, 2/3)
        # scipy assigns ordinal ranks in order of appearance
        assert result.values[0, 1] != result.values[0, 2], "Ordinal: no ties"
        assert 0.0 < result.values[0, 1] < 1.0
        assert 0.0 < result.values[0, 2] < 1.0


class TestRankIntegration:
    """Integration tests for Rank with Visitor pattern."""
    
    def test_rank_with_visitor(self):
        """Test Rank operator with EvaluateVisitor (full expression tree)."""
        # Arrange: Create dataset
        dates = pd.date_range('2024-01-01', periods=2, freq='D')
        assets = ['A', 'B', 'C']
        
        ds = xr.Dataset(
            {
                'price': xr.DataArray(
                    [[100.0, 200.0, 150.0],
                     [110.0, 210.0, 160.0]],
                    dims=['time', 'asset'],
                    coords={'time': dates, 'asset': assets}
                )
            }
        )
        
        # Create expression tree: Rank(Field('price'))
        expr = Rank(child=Field('price'))
        
        # Create visitor
        visitor = EvaluateVisitor(ds)
        
        # Act: Evaluate expression
        result = expr.accept(visitor)
        
        # Assert: Result shape and values
        assert result.shape == (2, 3), "Output shape should match input"
        assert result.dims == ('time', 'asset')
        
        # Time 0: [100, 200, 150] → [0.0, 1.0, 0.5]
        np.testing.assert_array_equal(result.values[0, :], [0.0, 1.0, 0.5])
        
        # Time 1: [110, 210, 160] → [0.0, 1.0, 0.5]
        np.testing.assert_array_equal(result.values[1, :], [0.0, 1.0, 0.5])
    
    def test_rank_caching(self):
        """Test visitor caching for Rank operator."""
        dates = pd.date_range('2024-01-01', periods=1, freq='D')
        assets = ['A', 'B']
        
        ds = xr.Dataset(
            {
                'val': xr.DataArray(
                    [[10.0, 20.0]],
                    dims=['time', 'asset'],
                    coords={'time': dates, 'asset': assets}
                )
            }
        )
        
        expr = Rank(child=Field('val'))
        visitor = EvaluateVisitor(ds)
        
        # Act: Evaluate
        result = expr.accept(visitor)
        
        # Assert: Check cached steps
        assert len(visitor._signal_cache) >= 2, "Should have at least 2 steps (Field + Rank)"
        
        # Verify Field was cached
        assert 'Field' in visitor._signal_cache[0][0], "Step 0 should be Field"
        
        # Verify Rank was cached
        assert 'Rank' in visitor._signal_cache[1][0], "Step 1 should be Rank"
        
        # Verify cached result
        cached_result = visitor._signal_cache[1][1]
        np.testing.assert_array_equal(cached_result.values, result.values)

