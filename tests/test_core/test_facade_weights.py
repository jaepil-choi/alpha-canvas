"""
Integration tests for AlphaCanvas.scale_weights() method.
Tests Strategy Pattern integration.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from alpha_canvas.core.facade import AlphaCanvas
from alpha_canvas.core.expression import Field
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

