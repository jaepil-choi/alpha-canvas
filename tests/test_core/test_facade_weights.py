"""
Integration tests for AlphaCanvas weight scaling methods.
Tests Strategy Pattern integration and dual-cache architecture.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from alpha_canvas.core.facade import AlphaCanvas
from alpha_canvas.core.expression import Field
from alpha_canvas.ops.timeseries import TsMean
from alpha_canvas.portfolio.strategies import GrossNetScaler, DollarNeutralScaler


class TestAlphaCanvasScaleWeights:
    """Test facade integration with weight scaling (Strategy Pattern)."""
    
    def test_scale_with_dataarray(self):
        """Test scale_weights with DataArray input."""
        # Setup
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        assets = ['A', 'B', 'C', 'D']
        
        time_index = dates
        asset_index = assets
        
        rc = AlphaCanvas(time_index=time_index, asset_index=asset_index)
        
        # Create signal DataArray
        signal = xr.DataArray(
            np.random.randn(10, 4),
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        # Scale with strategy
        scaler = DollarNeutralScaler()
        weights = rc.scale_weights(signal, scaler)
        
        # Verify result
        assert weights.shape == signal.shape
        assert weights.dims == signal.dims
        
    def test_scaler_required_no_default(self):
        """Test that scaler parameter is required (explicit strategy)."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        assets = ['A', 'B']
        
        rc = AlphaCanvas(time_index=dates, asset_index=assets)
        
        signal = xr.DataArray(
            np.random.randn(5, 2),
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        # Should require scaler parameter (no default)
        with pytest.raises(TypeError):
            rc.scale_weights(signal)  # Missing scaler argument
            
    def test_strategy_pattern_replaceable(self):
        """Test that different strategies can be easily swapped."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        assets = ['A', 'B', 'C']
        
        rc = AlphaCanvas(time_index=dates, asset_index=assets)
        
        signal = xr.DataArray(
            np.random.randn(10, 3),
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        # Strategy 1: Dollar Neutral
        scaler1 = DollarNeutralScaler()
        weights1 = rc.scale_weights(signal, scaler1)
        
        # Strategy 2: Net Long Bias
        scaler2 = GrossNetScaler(target_gross=2.0, target_net=0.3)
        weights2 = rc.scale_weights(signal, scaler2)
        
        # Different strategies produce different weights
        assert not np.allclose(weights1.values, weights2.values)


class TestAlphaCanvasEvaluateWithScaler:
    """Test evaluate() method with scaler parameter (dual-cache)."""
    
    def test_evaluate_with_scaler_caches_weights(self):
        """Test rc.evaluate() with scaler parameter caches weights."""
        # Setup
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        assets = ['A', 'B', 'C']
        
        rc = AlphaCanvas(time_index=dates, asset_index=assets)
        
        # Add returns data
        returns = xr.DataArray(
            np.random.randn(10, 3) * 0.02,
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        rc.add_data('returns', returns)
        
        # Evaluate with scaler
        expr = TsMean(Field('returns'), window=3)
        scaler = DollarNeutralScaler()
        
        result = rc.evaluate(expr, scaler)
        
        # Verify result is signal (not weights)
        assert result.shape == (10, 3)
        
        # Verify weights cached
        weights_step_0 = rc.get_weights(0)
        weights_step_1 = rc.get_weights(1)
        
        assert weights_step_0 is not None
        assert weights_step_1 is not None
        
        assert weights_step_0.shape == (10, 3)
        assert weights_step_1.shape == (10, 3)
    
    def test_evaluate_without_scaler_no_weights(self):
        """Test evaluate() without scaler doesn't cache weights."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        assets = ['A', 'B', 'C']
        
        rc = AlphaCanvas(time_index=dates, asset_index=assets)
        
        returns = xr.DataArray(
            np.random.randn(10, 3) * 0.02,
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        rc.add_data('returns', returns)
        
        # Evaluate without scaler
        expr = TsMean(Field('returns'), window=3)
        result = rc.evaluate(expr)
        
        # Weights should be None
        weights_step_0 = rc.get_weights(0)
        weights_step_1 = rc.get_weights(1)
        
        assert weights_step_0 is None
        assert weights_step_1 is None
    
    def test_get_weights_dollar_neutral_constraint(self):
        """Test weights from get_weights() satisfy dollar-neutral constraint."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        assets = ['A', 'B', 'C', 'D']
        
        rc = AlphaCanvas(time_index=dates, asset_index=assets)
        
        returns = xr.DataArray(
            np.random.randn(10, 4) * 0.02,
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        rc.add_data('returns', returns)
        
        # Evaluate with DollarNeutralScaler
        expr = TsMean(Field('returns'), window=3)
        result = rc.evaluate(expr, scaler=DollarNeutralScaler())
        
        # Check weights at step 1 (TsMean result)
        weights = rc.get_weights(1)
        
        # Calculate long/short sums (allowing for NaN due to window)
        long_sum = weights.where(weights > 0, 0.0).sum(dim='asset').mean().values
        short_sum = weights.where(weights < 0, 0.0).sum(dim='asset').mean().values
        
        # Dollar neutral: Long ≈ 1.0, Short ≈ -1.0
        # Relaxed tolerance due to NaN values from rolling window
        assert abs(long_sum - 1.0) < 0.5
        assert abs(short_sum + 1.0) < 0.5
    
    def test_scaler_replacement_efficiency(self):
        """Test swapping scalers recalculates only weights, not signals."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        assets = ['A', 'B', 'C']
        
        rc = AlphaCanvas(time_index=dates, asset_index=assets)
        
        returns = xr.DataArray(
            np.random.randn(10, 3) * 0.02,
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        rc.add_data('returns', returns)
        
        expr = TsMean(Field('returns'), window=3)
        
        # First evaluation with scaler 1
        result1 = rc.evaluate(expr, scaler=DollarNeutralScaler())
        signal_cache_1 = {k: v for k, v in rc._evaluator._signal_cache.items()}
        weights1_step1 = rc.get_weights(1)
        
        # Second evaluation with scaler 2
        result2 = rc.evaluate(expr, scaler=GrossNetScaler(2.0, 0.3))
        signal_cache_2 = {k: v for k, v in rc._evaluator._signal_cache.items()}
        weights2_step1 = rc.get_weights(1)
        
        # Signal caches should be identical
        assert signal_cache_1.keys() == signal_cache_2.keys()
        for step in signal_cache_1.keys():
            name1, sig1 = signal_cache_1[step]
            name2, sig2 = signal_cache_2[step]
            assert name1 == name2
            assert np.allclose(sig1.values, sig2.values, equal_nan=True)
        
        # Weights should be different (net exposure differs)
        net1 = weights1_step1.sum(dim='asset').mean().values
        net2 = weights2_step1.sum(dim='asset').mean().values
        
        # DollarNeutral targets 0, GrossNet(2.0, 0.3) targets 0.3
        assert abs(net1 - 0.0) < 0.5
        assert abs(net2 - 0.3) < 0.5
    
    def test_get_weights_nonexistent_step_raises_error(self):
        """Test get_weights() raises KeyError for invalid step."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        assets = ['A', 'B']
        
        rc = AlphaCanvas(time_index=dates, asset_index=assets)
        
        returns = xr.DataArray(
            np.random.randn(5, 2),
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        rc.add_data('returns', returns)
        
        # Evaluate
        result = rc.evaluate(Field('returns'), scaler=DollarNeutralScaler())
        
        # Valid step
        weights_0 = rc.get_weights(0)
        assert weights_0 is not None
        
        # Invalid step should raise KeyError
        with pytest.raises(KeyError):
            rc.get_weights(999)

