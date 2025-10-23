"""
Tests for universe masking functionality.

Following TDD principle: Tests written before implementation,
based on Experiment 13 findings.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpha_canvas import AlphaCanvas
from alpha_canvas.core.expression import Field
from alpha_canvas.core.visitor import EvaluateVisitor
from alpha_canvas.ops.timeseries import TsMean
from alpha_canvas.ops.crosssection import Rank
from alpha_database import DataSource


class TestAlphaCanvasUniverseInit:
    """Test universe parameter in AlphaCanvas initialization."""
    
    def test_alphacanvas_init_without_universe(self):
        """Test AlphaCanvas initialization without universe (default behavior)."""
        ds = DataSource('config')
        rc = AlphaCanvas(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        # Universe should be None
        assert rc.universe is None
        assert rc._evaluator._universe_mask is None
    
    def test_alphacanvas_init_with_universe_dataarray(self):
        """Test AlphaCanvas initialization with boolean DataArray universe."""
        ds = DataSource('config')
        
        # Load sample data to get actual dimensions
        sample = ds.load_field('adj_close', '2024-01-01', '2024-01-31')
        dates = sample.coords['time'].values
        assets = sample.coords['asset'].values
        
        # Create universe mask matching actual data dimensions
        universe = xr.DataArray(
            np.random.choice([True, False], size=(len(dates), len(assets))),
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        rc = AlphaCanvas(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31',
            universe=universe
        )
        
        # Universe should be set
        assert rc.universe is not None
        assert rc._evaluator._universe_mask is not None
        np.testing.assert_array_equal(rc.universe.values, universe.values)
    
    def test_invalid_universe_shape(self):
        """Test error on universe with wrong shape."""
        ds = DataSource('config')
        
        # Load sample to get actual dimensions
        sample = ds.load_field('adj_close', '2024-01-01', '2024-01-31')
        dates = sample.coords['time'].values
        assets = sample.coords['asset'].values
        
        # Wrong shape universe (fewer time steps)
        bad_universe = xr.DataArray(
            [[True, False], [True, True]],  # Only 2 time steps instead of actual length
            dims=['time', 'asset']
        )
        
        # Initialize with bad universe
        rc = AlphaCanvas(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31',
            universe=bad_universe
        )
        
        # Shape validation should fail when data is first loaded
        with pytest.raises(ValueError, match="Universe mask shape"):
            # Trigger panel initialization by loading data
            rc.add_data('test', ds.load_field('adj_close', '2024-01-01', '2024-01-31'))
    
    def test_invalid_universe_dtype(self):
        """Test error on universe with non-boolean dtype."""
        ds = DataSource('config')
        
        # Load sample to get actual dimensions
        sample = ds.load_field('adj_close', '2024-01-01', '2024-01-31')
        dates = sample.coords['time'].values
        assets = sample.coords['asset'].values
        
        # Float universe instead of boolean
        bad_universe = xr.DataArray(
            np.ones((len(dates), len(assets)), dtype=float),
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        with pytest.raises(TypeError, match="Universe must be boolean"):
            rc = AlphaCanvas(
                data_source=ds,
                start_date='2024-01-01',
                end_date='2024-01-31',
                universe=bad_universe
            )


class TestFieldUniverseMasking:
    """Test universe masking at Field retrieval (input masking)."""
    
    def test_field_applies_universe_mask(self):
        """Test that Field retrieval auto-applies universe mask."""
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        assets = ['A', 'B']
        
        # Create test data
        test_data = xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        # Universe: exclude asset B
        universe = xr.DataArray(
            [[True, False], [True, False], [True, False]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        # Create dataset with test data
        ds = xr.Dataset({'test_field': test_data})
        
        # Create visitor with universe
        visitor = EvaluateVisitor(ds)
        visitor._universe_mask = universe
        
        # Retrieve field
        field_expr = Field('test_field')
        result = visitor.evaluate(field_expr)
        
        # Assert: Asset B should be NaN
        assert result.values[0, 0] == 1.0
        assert np.isnan(result.values[0, 1])
        assert result.values[1, 0] == 3.0
        assert np.isnan(result.values[1, 1])
    
    def test_field_without_universe_no_masking(self):
        """Test that Field retrieval works normally without universe."""
        dates = pd.date_range('2024-01-01', periods=2, freq='D')
        assets = ['A', 'B']
        
        test_data = xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        ds = xr.Dataset({'test_field': test_data})
        visitor = EvaluateVisitor(ds)
        # No universe set (None)
        
        field_expr = Field('test_field')
        result = visitor.evaluate(field_expr)
        
        # No masking should occur
        np.testing.assert_array_equal(result.values, test_data.values)


class TestOperatorUniverseMasking:
    """Test universe masking at operator output (output masking)."""
    
    def test_operator_applies_universe_mask(self):
        """Test that operators auto-apply universe mask to output."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        assets = ['A', 'B']
        
        # Create test data
        test_data = xr.DataArray(
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        # Universe: exclude asset B
        universe = xr.DataArray(
            [[True, False]] * 5,
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        ds = xr.Dataset({'returns': test_data})
        visitor = EvaluateVisitor(ds)
        visitor._universe_mask = universe
        
        # Apply ts_mean
        expr = TsMean(child=Field('returns'), window=3)
        result = visitor.evaluate(expr)
        
        # Assert: Asset B should be all NaN in output
        assert np.all(np.isnan(result.values[:, 1]))
        
        # Assert: Asset A should have valid values where window is complete
        assert not np.isnan(result.values[2, 0])  # First complete window
    
    def test_double_masking_idempotent(self):
        """Test that double masking (Field + Operator) doesn't change result."""
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        assets = ['A', 'B']
        
        test_data = xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        universe = xr.DataArray(
            [[True, False], [True, True], [False, True]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        # Manual double masking
        masked_once = test_data.where(universe, np.nan)
        masked_twice = masked_once.where(universe, np.nan)
        
        # Should be identical (including NaN positions)
        assert np.allclose(
            masked_once.values,
            masked_twice.values,
            equal_nan=True
        )


class TestUniverseWithOperatorChains:
    """Test universe masking through operator chains."""
    
    def test_universe_with_ts_mean_chain(self):
        """Test universe masking with ts_mean operator."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        assets = ['A', 'B', 'C']
        
        test_data = xr.DataArray(
            [[1.0, 2.0, 3.0],
             [2.0, 3.0, 4.0],
             [3.0, 4.0, 5.0],
             [4.0, 5.0, 6.0],
             [5.0, 6.0, 7.0]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        # Exclude asset B from universe
        universe = xr.DataArray(
            [[True, False, True]] * 5,
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        ds = xr.Dataset({'data': test_data})
        visitor = EvaluateVisitor(ds)
        visitor._universe_mask = universe
        
        # Evaluate ts_mean
        expr = TsMean(child=Field('data'), window=3)
        result = visitor.evaluate(expr)
        
        # Asset B should be all NaN
        assert np.all(np.isnan(result.values[:, 1]))
        
        # Assets A and C should have values where window is complete
        assert not np.isnan(result.values[2, 0])
        assert not np.isnan(result.values[2, 2])
    
    def test_universe_with_rank(self):
        """Test universe masking with rank() operator."""
        dates = pd.date_range('2024-01-01', periods=1, freq='D')
        assets = ['A', 'B', 'C', 'D']
        
        test_data = xr.DataArray(
            [[10.0, 50.0, 30.0, 20.0]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        # Exclude asset B
        universe = xr.DataArray(
            [[True, False, True, True]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        ds = xr.Dataset({'values': test_data})
        visitor = EvaluateVisitor(ds)
        visitor._universe_mask = universe
        
        # Evaluate rank
        expr = Rank(child=Field('values'))
        result = visitor.evaluate(expr)
        
        # Asset B should be NaN
        assert np.isnan(result.values[0, 1])
        
        # Other assets ranked among themselves (A=10, C=30, D=20)
        assert result.values[0, 0] == 0.0  # A smallest
        assert result.values[0, 2] == 1.0  # C largest
        assert result.values[0, 3] == 0.5  # D middle


class TestUniverseEdgeCases:
    """Test edge cases for universe masking."""
    
    def test_all_false_universe_returns_all_nan(self):
        """Test that all-False universe produces all-NaN output."""
        dates = pd.date_range('2024-01-01', periods=2, freq='D')
        assets = ['A', 'B']
        
        test_data = xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        # Empty universe (all False)
        empty_universe = xr.DataArray(
            [[False, False], [False, False]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        ds = xr.Dataset({'data': test_data})
        visitor = EvaluateVisitor(ds)
        visitor._universe_mask = empty_universe
        
        # Retrieve field
        field_expr = Field('data')
        result = visitor.evaluate(field_expr)
        
        # All values should be NaN
        assert np.all(np.isnan(result.values))
    
    def test_time_varying_universe(self):
        """Test time-varying universe (e.g., delisting scenario)."""
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        assets = ['AAPL', 'DELIST', 'NVDA']
        
        test_data = xr.DataArray(
            [[10.0, 20.0, 30.0],
             [11.0, 21.0, 31.0],
             [12.0, 22.0, 32.0]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        # DELIST removed from universe on day 2
        time_varying_universe = xr.DataArray(
            [[True, True, True],   # Day 1: All in universe
             [True, False, True],  # Day 2: DELIST excluded
             [True, False, True]], # Day 3: DELIST still excluded
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        ds = xr.Dataset({'price': test_data})
        visitor = EvaluateVisitor(ds)
        visitor._universe_mask = time_varying_universe
        
        field_expr = Field('price')
        result = visitor.evaluate(field_expr)
        
        # DELIST has value on day 1, NaN on days 2-3
        assert result.values[0, 1] == 20.0
        assert np.isnan(result.values[1, 1])
        assert np.isnan(result.values[2, 1])
        
        # Other stocks unaffected
        assert result.values[1, 0] == 11.0
        assert result.values[1, 2] == 31.0


class TestUniverseInjectedData:
    """Test universe masking with injected DataArray (Open Toolkit pattern)."""
    
    def test_injected_data_respects_universe(self):
        """Test that add_data with DataArray applies universe mask."""
        ds_source = DataSource('config')
        
        # Load sample to get actual dimensions
        sample = ds_source.load_field('adj_close', '2024-01-01', '2024-01-31')
        dates = sample.coords['time'].values[:3]  # Use first 3 days
        assets = sample.coords['asset'].values[:2]  # Use first 2 assets
        
        # External data (calculated in Jupyter)
        external_data = xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        # Universe excludes asset 1 (second asset)
        universe = xr.DataArray(
            [[True, False]] * 3,
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        rc = AlphaCanvas(
            data_source=ds_source,
            start_date='2024-01-01',
            end_date='2024-01-31',
            universe=universe
        )
        
        # Inject external data
        rc.add_data('external', external_data)
        
        # Data should be masked
        result = rc.db['external']
        assert result.values[0, 0] == 1.0
        assert np.isnan(result.values[0, 1])
        assert result.values[1, 0] == 3.0
        assert np.isnan(result.values[1, 1])

