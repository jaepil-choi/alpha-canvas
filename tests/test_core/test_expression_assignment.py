"""
Tests for Expression assignment functionality (lazy evaluation pattern).

Tests cover:
- Expression.__setitem__ storage behavior
- Multiple assignments
- Visitor integration with assignments
- Overlapping mask behavior
- Universe masking with assignments
- Separate caching of base and final results
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpha_canvas.core.expression import Expression, Field
from alpha_canvas.core.visitor import EvaluateVisitor
from alpha_canvas.ops.timeseries import TsMean
from alpha_canvas.ops.logical import Equals


# ============================================================================
# UNIT TESTS: Expression.__setitem__
# ============================================================================

class TestExpressionAssignmentStorage:
    """Test that assignments are stored correctly in Expression."""
    
    def test_expression_assignment_stores_tuple(self):
        """Assignment stores (mask, value) tuple in list."""
        expr = Field('returns')
        mask = Field('size') == 'small'
        
        expr[mask] = 1.0
        
        assert hasattr(expr, '_assignments'), "Expression should have _assignments attribute"
        assert len(expr._assignments) == 1
        assert expr._assignments[0] == (mask, 1.0)
    
    def test_multiple_assignments_sequential(self):
        """Multiple assignments stored in order."""
        expr = Field('returns')
        mask1 = Field('size') == 'small'
        mask2 = Field('size') == 'big'
        
        expr[mask1] = 1.0
        expr[mask2] = -1.0
        
        assert len(expr._assignments) == 2
        assert expr._assignments[0] == (mask1, 1.0)
        assert expr._assignments[1] == (mask2, -1.0)
    
    def test_assignment_returns_none(self):
        """__setitem__ returns None (standard Python behavior)."""
        expr = Field('returns')
        mask = Field('size') == 'small'
        
        result = expr.__setitem__(mask, 1.0)
        
        assert result is None, "__setitem__ should return None"
    
    def test_assignment_with_literal_mask(self):
        """Can assign with literal boolean array (not Expression)."""
        expr = Field('returns')
        # Create a literal boolean array
        mask = np.array([True, False, True, False])
        
        expr[mask] = 1.0
        
        assert len(expr._assignments) == 1
        assert expr._assignments[0][1] == 1.0
        assert isinstance(expr._assignments[0][0], np.ndarray)
    
    def test_empty_assignments_list_initially(self):
        """New Expression has no _assignments until first assignment (lazy init)."""
        expr = Field('returns')
        
        # Lazy initialization - _assignments doesn't exist until first use
        assert not hasattr(expr, '_assignments')
        
        # After first assignment, it exists
        expr[Field('size') == 'small'] = 1.0
        assert hasattr(expr, '_assignments')
        assert len(expr._assignments) == 1


# ============================================================================
# INTEGRATION TESTS: Visitor with Assignments
# ============================================================================

def create_sample_dataset():
    """Create sample dataset for testing."""
    times = pd.date_range('2024-01-01', periods=5, freq='D')
    assets = ['A', 'B', 'C', 'D']
    
    # Returns data
    returns = xr.DataArray(
        [[0.01, 0.02, -0.01, 0.03],
         [0.02, -0.01, 0.01, 0.02],
         [-0.01, 0.03, 0.02, -0.02],
         [0.03, 0.01, -0.01, 0.01],
         [0.01, 0.02, 0.03, -0.01]],
        dims=['time', 'asset'],
        coords={'time': times, 'asset': assets}
    )
    
    # Size labels
    size = xr.DataArray(
        [['small', 'small', 'big', 'big'],
         ['small', 'small', 'big', 'big'],
         ['small', 'small', 'big', 'big'],
         ['small', 'small', 'big', 'big'],
         ['small', 'small', 'big', 'big']],
        dims=['time', 'asset'],
        coords={'time': times, 'asset': assets}
    )
    
    # Momentum labels
    momentum = xr.DataArray(
        [['high', 'low', 'high', 'low'],
         ['high', 'low', 'high', 'low'],
         ['high', 'low', 'high', 'low'],
         ['high', 'low', 'high', 'low'],
         ['high', 'low', 'high', 'low']],
        dims=['time', 'asset'],
        coords={'time': times, 'asset': assets}
    )
    
    ds = xr.Dataset({
        'returns': returns,
        'size': size,
        'momentum': momentum
    })
    
    return ds


def create_universe_mask(ds):
    """Create universe mask excluding last asset."""
    universe = xr.DataArray(
        np.array([[True, True, True, False]] * len(ds.coords['time'])),
        dims=['time', 'asset'],
        coords={'time': ds.coords['time'], 'asset': ds.coords['asset']}
    )
    return universe


class TestVisitorAppliesAssignments:
    """Test that Visitor correctly applies assignments."""
    
    def test_visitor_applies_assignments(self):
        """Visitor evaluates base Expression, then applies assignments."""
        # Setup
        ds = create_sample_dataset()
        visitor = EvaluateVisitor(ds, None)
        
        # Create Expression with assignments
        expr = Field('returns')
        mask = Equals(Field('size'), 'small')
        expr[mask] = 1.0
        
        # Evaluate
        result = visitor.evaluate(expr)
        
        # Validate: small positions should be 1.0
        size_data = ds['size']
        small_mask = (size_data == 'small')
        assert np.all(result.values[small_mask.values] == 1.0)
        
        # Non-small positions should keep original values
        big_mask = (size_data == 'big')
        assert np.allclose(result.values[big_mask.values], ds['returns'].values[big_mask.values])
    
    def test_overlapping_masks_later_wins(self):
        """Later assignment overwrites earlier for overlapping positions."""
        ds = create_sample_dataset()
        visitor = EvaluateVisitor(ds, None)
        
        expr = Field('returns')
        mask1 = Equals(Field('size'), 'small')
        mask2 = Equals(Field('momentum'), 'high')
        # Both masks may overlap (small & high)
        expr[mask1] = 1.0
        expr[mask2] = -1.0  # This wins for overlap
        
        result = visitor.evaluate(expr)
        
        # Validate: high momentum should be -1.0 regardless of size
        high_mom = (ds['momentum'] == 'high')
        assert np.all(result.values[high_mom.values] == -1.0)
    
    def test_assignment_with_universe_masking(self):
        """Assignments outside universe become NaN."""
        # Setup with universe mask
        ds = create_sample_dataset()
        universe = create_universe_mask(ds)
        visitor = EvaluateVisitor(ds, None)
        visitor._universe_mask = universe
        
        expr = Field('returns')
        # Assign to all positions (including outside universe)
        expr[Equals(Field('size'), 'small')] = 1.0
        
        result = visitor.evaluate(expr)
        
        # Positions outside universe should be NaN
        outside_universe = ~universe
        assert np.all(np.isnan(result.values[outside_universe.values]))
        
        # Positions inside universe should have assigned value (if small)
        inside_and_small = universe.values & (ds['size'].values == 'small')
        assert np.all(result.values[inside_and_small] == 1.0)
    
    def test_assignment_caching_separate_steps(self):
        """Visitor caches base result and final result separately."""
        ds = create_sample_dataset()
        visitor = EvaluateVisitor(ds, None)
        
        expr = TsMean(Field('returns'), 3)
        expr[Equals(Field('size'), 'small')] = 1.0
        
        result = visitor.evaluate(expr)
        
        # Should have multiple steps cached:
        # Step 0: Field('returns')
        # Step 1: TsMean result (base)
        # Step 2+: After assignments (final)
        assert len(visitor._signal_cache) >= 3, "Should cache at least 3 steps (Field, TsMean base, final)"
    
    def test_no_assignments_skips_special_handling(self):
        """Expression without assignments evaluates normally."""
        ds = create_sample_dataset()
        visitor = EvaluateVisitor(ds, None)
        
        expr = Field('returns')
        # No assignments
        
        result = visitor.evaluate(expr)
        
        # Should just return field data
        assert np.allclose(result.values, ds['returns'].values)
    
    def test_multiple_assignments_all_applied(self):
        """All assignments in list are applied sequentially."""
        ds = create_sample_dataset()
        visitor = EvaluateVisitor(ds, None)
        
        expr = Field('returns')
        expr[Equals(Field('size'), 'small')] = 1.0
        expr[Equals(Field('size'), 'big')] = -1.0
        expr[Equals(Field('momentum'), 'high')] = 0.5
        
        result = visitor.evaluate(expr)
        
        # High momentum should be 0.5 (latest assignment)
        high_mom = (ds['momentum'] == 'high')
        assert np.all(result.values[high_mom.values] == 0.5)


# ============================================================================
# EDGE CASES
# ============================================================================

class TestAssignmentEdgeCases:
    """Test edge cases for assignment functionality."""
    
    def test_assignment_to_all_nan_base(self):
        """Can assign to Expression that evaluates to all NaN."""
        times = pd.date_range('2024-01-01', periods=3, freq='D')
        assets = ['A', 'B']
        
        # All NaN data
        data = xr.DataArray(
            np.full((3, 2), np.nan),
            dims=['time', 'asset'],
            coords={'time': times, 'asset': assets}
        )
        size = xr.DataArray(
            [['small', 'big'], ['small', 'big'], ['small', 'big']],
            dims=['time', 'asset'],
            coords={'time': times, 'asset': assets}
        )
        
        ds = xr.Dataset({'data': data, 'size': size})
        visitor = EvaluateVisitor(ds, None)
        
        expr = Field('data')
        expr[Equals(Field('size'), 'small')] = 1.0
        
        result = visitor.evaluate(expr)
        
        # Small positions should be 1.0, big should be NaN
        small_mask = (size == 'small')
        assert np.all(result.values[small_mask.values] == 1.0)
        assert np.all(np.isnan(result.values[~small_mask.values]))
    
    def test_assignment_with_all_false_mask(self):
        """Assignment with mask that's all False does nothing."""
        ds = create_sample_dataset()
        visitor = EvaluateVisitor(ds, None)
        
        expr = Field('returns')
        # Mask that's all False
        false_mask = Equals(Field('size'), 'nonexistent')
        expr[false_mask] = 999.0
        
        result = visitor.evaluate(expr)
        
        # Should keep original values
        assert np.allclose(result.values, ds['returns'].values)
    
    def test_assignment_with_numeric_value_types(self):
        """Can assign int, float, or numpy types."""
        ds = create_sample_dataset()
        visitor = EvaluateVisitor(ds, None)
        
        expr1 = Field('returns')
        expr1[Equals(Field('size'), 'small')] = 1  # int
        result1 = visitor.evaluate(expr1)
        
        expr2 = Field('returns')
        expr2[Equals(Field('size'), 'small')] = 1.5  # float
        result2 = visitor.evaluate(expr2)
        
        expr3 = Field('returns')
        expr3[Equals(Field('size'), 'small')] = np.float64(2.0)  # numpy type
        result3 = visitor.evaluate(expr3)
        
        # All should work
        small_mask = (ds['size'] == 'small')
        assert np.all(result1.values[small_mask.values] == 1.0)
        assert np.all(result2.values[small_mask.values] == 1.5)
        assert np.all(result3.values[small_mask.values] == 2.0)

