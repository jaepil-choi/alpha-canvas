"""
Tests for WeightScaler abstract base class.
Following TDD: Tests written first based on Experiment 18.
"""
import pytest
import xarray as xr
import numpy as np
import pandas as pd
from alpha_canvas.portfolio.base import WeightScaler


class MockScaler(WeightScaler):
    """Mock scaler for testing abstract base class."""
    def scale(self, signal):
        return signal  # Just return signal unchanged


class TestWeightScalerValidation:
    """Test _validate_signal method."""
    
    def test_validate_correct_dims(self):
        """Test validation passes for correct dimensions."""
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        assets = ['A', 'B', 'C']
        signal = xr.DataArray(
            np.random.randn(3, 3),
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        scaler = MockScaler()
        # Should not raise
        scaler._validate_signal(signal)
        
    def test_validate_wrong_dims(self):
        """Test validation fails for wrong dimensions."""
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        securities = ['A', 'B', 'C']
        signal = xr.DataArray(
            np.random.randn(3, 3),
            dims=['time', 'security'],  # Wrong dimension name
            coords={'time': dates, 'security': securities}
        )
        
        scaler = MockScaler()
        with pytest.raises(ValueError, match="must have dims"):
            scaler._validate_signal(signal)
        
    def test_validate_all_nan(self):
        """Test validation fails when all values are NaN."""
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        assets = ['A', 'B', 'C']
        signal = xr.DataArray(
            np.full((3, 3), np.nan),
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        scaler = MockScaler()
        with pytest.raises(ValueError, match="All signal values are NaN"):
            scaler._validate_signal(signal)

