"""Tests for classification operators (cs_quantile)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpha_canvas.core.data_model import DataPanel
from alpha_canvas.core.expression import Field
from alpha_canvas.core.visitor import EvaluateVisitor
from alpha_canvas.ops.classification import CsQuantile


class TestCsQuantileUnit:
    """Unit tests for CsQuantile.compute() method."""
    
    def test_compute_independent_sort_basic(self):
        """Test basic independent sort returns categorical labels."""
        # Create test data
        data = xr.DataArray(
            [[10, 50, 30], [20, 60, 40]],
            dims=['time', 'asset'],
            coords={
                'time': pd.date_range('2024-01-01', periods=2, freq='D'),
                'asset': ['A', 'B', 'C']
            }
        )
        
        # Create operator
        op = CsQuantile(
            child=Field('dummy'),
            bins=2,
            labels=['small', 'big']
        )
        
        # Call compute directly
        result = op.compute(data)
        
        # Verify shape preservation
        assert result.shape == data.shape, "Shape should be preserved"
        
        # Verify categorical output
        assert result.dtype == object, "Should return object dtype"
        
        # Verify labels at first timestep
        # 10 < 30 < 50 → quantile split at 30 → [small, big, small]
        assert result.values[0, 0] == 'small'
        assert result.values[0, 1] == 'big'
        assert result.values[0, 2] == 'small'
    
    def test_compute_dependent_sort_basic(self):
        """Test dependent sort produces different cutoffs within groups."""
        # Create test data
        data = xr.DataArray(
            [[10, 50, 30, 15, 55, 35]],  # 2 groups of 3
            dims=['time', 'asset'],
            coords={
                'time': pd.date_range('2024-01-01', periods=1, freq='D'),
                'asset': [f'A{i}' for i in range(6)]
            }
        )
        
        # Create group labels (first 3 are 'small', last 3 are 'big')
        groups = xr.DataArray(
            [['small', 'small', 'small', 'big', 'big', 'big']],
            dims=['time', 'asset'],
            coords=data.coords
        )
        
        # Create operator
        op = CsQuantile(
            child=Field('dummy'),
            bins=2,
            labels=['low', 'high'],
            group_by='size'
        )
        
        # Call compute with group labels
        result = op.compute(data, group_labels=groups)
        
        # Verify shape preservation
        assert result.shape == data.shape
        
        # Verify dependent sort produces different cutoffs
        # Group 'small': [10, 50, 30] → split at 30 → [low, high, low]
        # Group 'big': [15, 55, 35] → split at 35 → [low, high, low]
        assert result.values[0, 0] == 'low'   # 10 in small group
        assert result.values[0, 1] == 'high'  # 50 in small group
        assert result.values[0, 2] == 'low'   # 30 in small group
        assert result.values[0, 3] == 'low'   # 15 in big group
        assert result.values[0, 4] == 'high'  # 55 in big group
        assert result.values[0, 5] == 'low'   # 35 in big group
    
    def test_compute_preserves_shape(self):
        """Test that output shape exactly matches input shape."""
        # Various shapes
        shapes = [(10, 6), (5, 20), (100, 3)]
        
        for T, N in shapes:
            data = xr.DataArray(
                np.random.rand(T, N),
                dims=['time', 'asset'],
                coords={
                    'time': pd.date_range('2024-01-01', periods=T, freq='D'),
                    'asset': [f'A{i}' for i in range(N)]
                }
            )
            
            op = CsQuantile(
                child=Field('dummy'),
                bins=3,
                labels=['low', 'mid', 'high']
            )
            
            result = op.compute(data)
            
            assert result.shape == (T, N), f"Shape mismatch for {(T, N)}"
    
    def test_compute_returns_categorical_labels(self):
        """Test that output contains specified labels."""
        data = xr.DataArray(
            [[10, 20, 30, 40, 50, 60]],
            dims=['time', 'asset'],
            coords={
                'time': pd.date_range('2024-01-01', periods=1, freq='D'),
                'asset': [f'A{i}' for i in range(6)]
            }
        )
        
        op = CsQuantile(
            child=Field('dummy'),
            bins=3,
            labels=['small', 'mid', 'big']
        )
        
        result = op.compute(data)
        
        # Get unique labels (excluding NaN)
        unique_labels = set(result.values.flatten()) - {np.nan}
        expected_labels = {'small', 'mid', 'big'}
        
        assert unique_labels == expected_labels, f"Got {unique_labels}, expected {expected_labels}"
    
    def test_compute_handles_nan(self):
        """Test that NaN values remain NaN in output."""
        data = xr.DataArray(
            [[np.nan, 20, 30], [10, np.nan, 50]],
            dims=['time', 'asset'],
            coords={
                'time': pd.date_range('2024-01-01', periods=2, freq='D'),
                'asset': ['A', 'B', 'C']
            }
        )
        
        op = CsQuantile(
            child=Field('dummy'),
            bins=2,
            labels=['small', 'big']
        )
        
        result = op.compute(data)
        
        # Verify NaN positions preserved
        assert pd.isna(result.values[0, 0]), "NaN at [0,0] should be preserved"
        assert pd.isna(result.values[1, 1]), "NaN at [1,1] should be preserved"
        
        # Verify non-NaN positions have labels
        assert result.values[0, 1] in ['small', 'big']
        assert result.values[0, 2] in ['small', 'big']
    
    def test_compute_validates_bins_labels_match(self):
        """Test that bins and labels length must match."""
        with pytest.raises(ValueError, match="labels length.*must equal bins"):
            CsQuantile(
                child=Field('dummy'),
                bins=3,
                labels=['small', 'big']  # Only 2 labels for 3 bins!
            )
    
    def test_compute_different_cutoffs_independent_vs_dependent(self):
        """Test that independent and dependent sorts produce different results."""
        # Create data where independent/dependent should differ
        data = xr.DataArray(
            [[10, 50, 20, 15, 55, 25]],  # 2 groups
            dims=['time', 'asset'],
            coords={
                'time': pd.date_range('2024-01-01', periods=1, freq='D'),
                'asset': [f'A{i}' for i in range(6)]
            }
        )
        
        # Groups for dependent sort
        groups = xr.DataArray(
            [['group_a', 'group_a', 'group_a', 'group_b', 'group_b', 'group_b']],
            dims=['time', 'asset'],
            coords=data.coords
        )
        
        op = CsQuantile(
            child=Field('dummy'),
            bins=2,
            labels=['low', 'high']
        )
        
        # Independent sort (whole universe)
        result_independent = op.compute(data)
        
        # Dependent sort (within groups)
        result_dependent = op.compute(data, group_labels=groups)
        
        # They should produce different labelings
        differences = result_independent.values != result_dependent.values
        valid_positions = ~(pd.isna(result_independent.values) | pd.isna(result_dependent.values))
        
        diff_count = np.sum(differences & valid_positions)
        assert diff_count > 0, "Independent and dependent sort should produce different results"


class TestCsQuantileIntegration:
    """Integration tests with Visitor and Expression tree."""
    
    def test_independent_sort_with_visitor(self):
        """Test independent sort through Visitor."""
        # Create dataset
        data = xr.DataArray(
            [[10, 50, 30], [20, 60, 40]],
            dims=['time', 'asset'],
            coords={
                'time': pd.date_range('2024-01-01', periods=2, freq='D'),
                'asset': ['A', 'B', 'C']
            }
        )
        
        ds = xr.Dataset({'market_cap': data})
        visitor = EvaluateVisitor(ds, None)
        
        # Create expression
        expr = CsQuantile(
            child=Field('market_cap'),
            bins=2,
            labels=['small', 'big']
        )
        
        # Evaluate through visitor
        result = visitor.evaluate(expr)
        
        # Verify result
        assert result.shape == data.shape
        assert result.dtype == object
        assert result.values[0, 0] == 'small'
    
    def test_dependent_sort_with_visitor(self):
        """Test dependent sort through Visitor with group_by lookup."""
        # Create dataset with data and size groups
        data = xr.DataArray(
            [[10, 50, 30, 15, 55, 35]],
            dims=['time', 'asset'],
            coords={
                'time': pd.date_range('2024-01-01', periods=1, freq='D'),
                'asset': [f'A{i}' for i in range(6)]
            }
        )
        
        size_labels = xr.DataArray(
            [['small', 'small', 'small', 'big', 'big', 'big']],
            dims=['time', 'asset'],
            coords=data.coords
        )
        
        ds = xr.Dataset({
            'btm': data,
            'size': size_labels
        })
        
        visitor = EvaluateVisitor(ds, None)
        
        # Create expression with group_by
        expr = CsQuantile(
            child=Field('btm'),
            bins=2,
            labels=['low', 'high'],
            group_by='size'
        )
        
        # Evaluate through visitor
        result = visitor.evaluate(expr)
        
        # Verify result
        assert result.shape == data.shape
        assert result.dtype == object
        
        # Verify dependent sort worked (different cutoffs per group)
        assert result.values[0, 0] == 'low'   # 10 in small group
        assert result.values[0, 1] == 'high'  # 50 in small group
    
    def test_group_by_field_lookup_from_dataset(self):
        """Test that Visitor correctly looks up group_by field from dataset."""
        data = xr.DataArray(
            [[10, 50]],
            dims=['time', 'asset'],
            coords={
                'time': pd.date_range('2024-01-01', periods=1, freq='D'),
                'asset': ['A', 'B']
            }
        )
        
        groups = xr.DataArray(
            [['group_a', 'group_b']],
            dims=['time', 'asset'],
            coords=data.coords
        )
        
        ds = xr.Dataset({'returns': data, 'sector': groups})
        visitor = EvaluateVisitor(ds, None)
        
        # Expression references 'sector' by name
        expr = CsQuantile(
            child=Field('returns'),
            bins=2,
            labels=['low', 'high'],
            group_by='sector'
        )
        
        # Should successfully look up 'sector' from dataset
        result = visitor.evaluate(expr)
        assert result is not None
    
    def test_universe_masking_applied(self):
        """Test that universe masking is applied to cs_quantile output."""
        data = xr.DataArray(
            [[10, 50, 30]],
            dims=['time', 'asset'],
            coords={
                'time': pd.date_range('2024-01-01', periods=1, freq='D'),
                'asset': ['A', 'B', 'C']
            }
        )
        
        # Universe excludes asset B
        universe = xr.DataArray(
            [[True, False, True]],
            dims=['time', 'asset'],
            coords=data.coords
        )
        
        ds = xr.Dataset({'market_cap': data})
        visitor = EvaluateVisitor(ds, None)
        visitor._universe_mask = universe
        
        expr = CsQuantile(
            child=Field('market_cap'),
            bins=2,
            labels=['small', 'big']
        )
        
        result = visitor.evaluate(expr)
        
        # Asset B should be NaN (excluded by universe)
        assert pd.isna(result.values[0, 1]), "Universe-excluded asset should be NaN"
        
        # Assets A and C should have labels
        assert result.values[0, 0] in ['small', 'big']
        assert result.values[0, 2] in ['small', 'big']
    
    def test_caching_works(self):
        """Test that Visitor caches cs_quantile results."""
        data = xr.DataArray(
            [[10, 50, 30]],
            dims=['time', 'asset'],
            coords={
                'time': pd.date_range('2024-01-01', periods=1, freq='D'),
                'asset': ['A', 'B', 'C']
            }
        )
        
        ds = xr.Dataset({'market_cap': data})
        visitor = EvaluateVisitor(ds, None)
        
        expr = CsQuantile(
            child=Field('market_cap'),
            bins=2,
            labels=['small', 'big']
        )
        
        # Evaluate
        result = visitor.evaluate(expr)
        
        # Check cache contains both Field and CsQuantile steps
        assert len(visitor._signal_cache) == 2
        assert 'Field_market_cap' in visitor._signal_cache[0][0]
        assert 'CsQuantile' in visitor._signal_cache[1][0]
    
    def test_group_by_field_not_found_raises_error(self):
        """Test that missing group_by field raises clear error."""
        data = xr.DataArray(
            [[10, 50]],
            dims=['time', 'asset'],
            coords={
                'time': pd.date_range('2024-01-01', periods=1, freq='D'),
                'asset': ['A', 'B']
            }
        )
        
        ds = xr.Dataset({'returns': data})
        visitor = EvaluateVisitor(ds, None)
        
        # Reference non-existent field
        expr = CsQuantile(
            child=Field('returns'),
            bins=2,
            labels=['low', 'high'],
            group_by='sector'  # Doesn't exist!
        )
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="group_by field 'sector' not found"):
            visitor.evaluate(expr)

