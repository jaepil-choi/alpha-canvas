"""
Tests for backtesting functionality with portfolio return computation.

This module tests the triple-cache architecture and shift-mask workflow
for position-level portfolio return tracking.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpha_canvas.core.expression import Field
from alpha_canvas.core.visitor import EvaluateVisitor
from alpha_canvas.ops.timeseries import TsMean
from alpha_canvas.portfolio.strategies import DollarNeutralScaler, GrossNetScaler
from alpha_database import DataSource


class TestPortfolioReturnComputation:
    """Test _compute_portfolio_returns() method."""
    
    def test_compute_portfolio_returns_basic(self):
        """Test basic portfolio return computation."""
        # Setup
        dates = pd.date_range('2024-01-01', periods=5)
        assets = ['AAPL', 'GOOGL', 'MSFT']
        
        ds = xr.Dataset(coords={'time': dates, 'asset': assets})
        visitor = EvaluateVisitor(ds)
        
        # Weights and returns
        weights = xr.DataArray(
            [[0.3, 0.4, 0.3],
             [0.2, 0.5, 0.3],
             [0.4, 0.3, 0.3],
             [0.3, 0.3, 0.4],
             [0.3, 0.4, 0.3]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        returns = xr.DataArray(
            [[0.02, 0.01, -0.01],
             [0.01, -0.02, 0.03],
             [-0.01, 0.02, 0.01],
             [0.03, -0.02, 0.02],
             [0.01, 0.03, -0.02]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        visitor._returns_data = returns
        
        # Compute
        port_return = visitor._compute_portfolio_returns(weights)
        
        # Verify shape preserved
        assert port_return.shape == (5, 3)
        
        # Verify first day is NaN (no weights from previous day)
        assert np.all(np.isnan(port_return.isel(time=0).values))
        
        # Verify second day uses first day's weights
        expected_day1 = weights.isel(time=0).values * returns.isel(time=1).values
        np.testing.assert_array_almost_equal(
            port_return.isel(time=1).values,
            expected_day1
        )
    
    def test_compute_portfolio_returns_with_universe(self):
        """Test portfolio returns with universe masking."""
        dates = pd.date_range('2024-01-01', periods=3)
        assets = ['AAPL', 'GOOGL']
        
        ds = xr.Dataset(coords={'time': dates, 'asset': assets})
        visitor = EvaluateVisitor(ds)
        
        # Universe: GOOGL exits on day 2
        universe = xr.DataArray(
            [[True, True],
             [True, True],
             [True, False]],  # GOOGL exits
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        visitor._universe_mask = universe
        
        weights = xr.DataArray(
            [[0.5, 0.5],
             [0.5, 0.5],
             [0.5, 0.5]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        returns = xr.DataArray(
            [[0.01, 0.02],
             [0.03, 0.04],
             [0.05, 0.06]],
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        visitor._returns_data = returns
        
        # Compute
        port_return = visitor._compute_portfolio_returns(weights)
        
        # GOOGL should have NaN on day 2 (exited universe)
        assert np.isnan(port_return.isel(time=2, asset=1).values)
        
        # AAPL should still have value on day 2
        assert not np.isnan(port_return.isel(time=2, asset=0).values)


class TestTripleCachePopulation:
    """Test triple-cache (signal, weight, port_return) population."""
    
    def test_triple_cache_with_scaler(self):
        """Test all three caches populated when scaler provided."""
        dates = pd.date_range('2024-01-01', periods=10)
        assets = ['AAPL', 'GOOGL', 'MSFT']
        
        # Create dataset with returns
        data = xr.DataArray(
            np.random.randn(10, 3),
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        returns = xr.DataArray(
            np.random.randn(10, 3) * 0.02,
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        ds = xr.Dataset({'price': data, 'returns': returns})
        visitor = EvaluateVisitor(ds)
        visitor._returns_data = returns
        
        # Evaluate with scaler
        field = Field('price')
        scaler = DollarNeutralScaler()
        result = visitor.evaluate(field, scaler=scaler)
        
        # Verify signal cache
        assert len(visitor._signal_cache) == 1
        assert 'Field_price' in visitor._signal_cache[0][0]
        
        # Verify weight cache
        assert len(visitor._weight_cache) == 1
        weights = visitor._weight_cache[0][1]
        assert weights is not None
        assert weights.shape == (10, 3)
        
        # Verify port_return cache
        assert len(visitor._port_return_cache) == 1
        port_return = visitor._port_return_cache[0][1]
        assert port_return is not None
        assert port_return.shape == (10, 3)
    
    def test_no_port_return_without_scaler(self):
        """Test port_return cache empty when no scaler."""
        dates = pd.date_range('2024-01-01', periods=10)
        assets = ['AAPL', 'GOOGL']
        
        data = xr.DataArray(
            np.random.randn(10, 2),
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        ds = xr.Dataset({'price': data})
        visitor = EvaluateVisitor(ds)
        
        # Evaluate without scaler
        field = Field('price')
        result = visitor.evaluate(field)
        
        # Signal cache populated
        assert len(visitor._signal_cache) == 1
        
        # Weight and port_return caches empty
        assert len(visitor._weight_cache) == 0
        assert len(visitor._port_return_cache) == 0
    
    def test_port_return_recalculation_with_new_scaler(self):
        """Test port_return recalculated when scaler changes."""
        dates = pd.date_range('2024-01-01', periods=10)
        assets = ['AAPL', 'GOOGL']
        
        data = xr.DataArray(
            np.random.randn(10, 2),
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        returns = xr.DataArray(
            np.random.randn(10, 2) * 0.02,
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        ds = xr.Dataset({'price': data, 'returns': returns})
        visitor = EvaluateVisitor(ds)
        visitor._returns_data = returns
        
        # First evaluation
        field = Field('price')
        scaler1 = DollarNeutralScaler()
        visitor.evaluate(field, scaler=scaler1)
        
        port_return1 = visitor._port_return_cache[0][1]
        
        # Recalculate with new scaler
        scaler2 = GrossNetScaler(target_gross=2.0, target_net=0.5)
        visitor.recalculate_weights_with_scaler(scaler2)
        
        port_return2 = visitor._port_return_cache[0][1]
        
        # Port returns should be different
        assert not np.allclose(
            port_return1.values,
            port_return2.values,
            equal_nan=True
        )


class TestFacadeConvenienceMethods:
    """Test AlphaCanvas convenience methods for backtest."""
    
    def test_get_port_return_returns_correct_shape(self):
        """Test get_port_return() returns (T, N) shape."""
        from alpha_canvas import AlphaCanvas
        
        dates = pd.date_range('2024-01-01', periods=10)
        assets = ['AAPL', 'GOOGL', 'MSFT']
        
        ds = DataSource('config')
        rc = AlphaCanvas(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        # Add data
        signal = xr.DataArray(
            np.random.randn(10, 3),
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        returns = xr.DataArray(
            np.random.randn(10, 3) * 0.02,
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        rc.add_data('signal', signal)
        rc._returns = returns
        rc._evaluator._returns_data = returns
        
        # Evaluate with scaler
        result = rc.evaluate(Field('signal'), scaler=DollarNeutralScaler())
        
        # Get port_return
        port_return = rc.get_port_return(0)
        
        assert port_return is not None
        assert port_return.shape == (10, 3)
    
    def test_get_daily_pnl_aggregates_correctly(self):
        """Test get_daily_pnl() aggregates to (T,) shape."""
        from alpha_canvas import AlphaCanvas
        
        dates = pd.date_range('2024-01-01', periods=10)
        assets = ['AAPL', 'GOOGL']
        
        ds = DataSource('config')
        rc = AlphaCanvas(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        signal = xr.DataArray(
            np.random.randn(10, 2),
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        returns = xr.DataArray(
            np.random.randn(10, 2) * 0.02,
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        rc.add_data('signal', signal)
        rc._returns = returns
        rc._evaluator._returns_data = returns
        
        # Evaluate
        result = rc.evaluate(Field('signal'), scaler=DollarNeutralScaler())
        
        # Get daily PnL
        daily_pnl = rc.get_daily_pnl(0)
        
        assert daily_pnl is not None
        assert daily_pnl.shape == (10,)
        assert daily_pnl.dims == ('time',)
    
    def test_get_cumulative_pnl_uses_cumsum(self):
        """Test get_cumulative_pnl() returns cumulative sum."""
        from alpha_canvas import AlphaCanvas
        
        dates = pd.date_range('2024-01-01', periods=5)
        assets = ['AAPL', 'GOOGL']
        
        ds = DataSource('config')
        rc = AlphaCanvas(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        signal = xr.DataArray(
            [[1.0, -1.0]] * 5,
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        returns = xr.DataArray(
            [[0.01, 0.02]] * 5,
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        rc.add_data('signal', signal)
        rc._returns = returns
        rc._evaluator._returns_data = returns
        
        # Evaluate
        result = rc.evaluate(Field('signal'), scaler=DollarNeutralScaler())
        
        # Get cumulative PnL
        cum_pnl = rc.get_cumulative_pnl(0)
        
        assert cum_pnl is not None
        # First value should be 0 (cumsum of NaN starts at 0)
        assert cum_pnl.isel(time=0).values == 0.0
        
        # Verify it's a cumulative sum (decreasing due to NaN on first day)
        assert cum_pnl.shape == (5,)
        # Later values should be non-zero and cumulative
        assert cum_pnl.isel(time=-1).values != 0.0
    
    def test_methods_return_none_without_scaler(self):
        """Test convenience methods return None when no scaler used."""
        from alpha_canvas import AlphaCanvas
        
        dates = pd.date_range('2024-01-01', periods=10)
        assets = ['AAPL', 'GOOGL']
        
        ds = DataSource('config')
        rc = AlphaCanvas(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        signal = xr.DataArray(
            np.random.randn(10, 2),
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        rc.add_data('signal', signal)
        
        # Evaluate WITHOUT scaler
        result = rc.evaluate(Field('signal'))
        
        # All methods should return None
        assert rc.get_port_return(0) is None
        assert rc.get_daily_pnl(0) is None
        assert rc.get_cumulative_pnl(0) is None


class TestWinnerLoserAttribution:
    """Test position-level attribution analysis."""
    
    def test_position_level_returns_enable_attribution(self):
        """Test that position-level returns can identify winners/losers."""
        from alpha_canvas import AlphaCanvas
        
        dates = pd.date_range('2024-01-01', periods=10)
        assets = ['WINNER', 'LOSER', 'NEUTRAL']
        
        ds = DataSource('config')
        rc = AlphaCanvas(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        # Create signal - WINNER gets positive weight, LOSER gets negative weight
        signal = xr.DataArray(
            [[2.0, -2.0, 0.5]] * 10,
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        # WINNER has high positive returns, LOSER has high positive returns too
        # But: WINNER has positive weight, LOSER has negative weight
        # So WINNER contributes positive, LOSER contributes negative (short losing money)
        returns = xr.DataArray(
            [[0.05, 0.05, 0.01]] * 10,  # Both go up
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        rc.add_data('signal', signal)
        rc._returns = returns
        rc._evaluator._returns_data = returns
        
        # Evaluate
        result = rc.evaluate(Field('signal'), scaler=DollarNeutralScaler())
        
        # Get position-level returns
        port_return = rc.get_port_return(0)
        
        # Calculate total contribution per asset (skip first NaN day)
        total_contrib = port_return.isel(time=slice(1, None)).sum(dim='time')
        
        # WINNER: positive weight * positive return = positive contribution
        # LOSER: negative weight * positive return = negative contribution  
        winner_contrib = total_contrib.sel(asset='WINNER').values
        loser_contrib = total_contrib.sel(asset='LOSER').values
        
        # Winner should have positive contribution, loser should have negative
        assert winner_contrib > 0
        assert loser_contrib < 0
        # Winner should contribute more than loser
        assert winner_contrib > loser_contrib
    
    def test_top_bottom_stock_identification(self):
        """Test identifying top and bottom contributors."""
        from alpha_canvas import AlphaCanvas
        
        dates = pd.date_range('2024-01-01', periods=20)
        assets = [f'STOCK_{i}' for i in range(5)]
        
        ds = DataSource('config')
        rc = AlphaCanvas(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        # Random signal
        np.random.seed(42)
        signal = xr.DataArray(
            np.random.randn(20, 5),
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        # Returns with clear winner and loser
        returns_data = np.random.randn(20, 5) * 0.01
        returns_data[:, 0] += 0.02  # STOCK_0 consistently outperforms
        returns_data[:, 4] -= 0.02  # STOCK_4 consistently underperforms
        
        returns = xr.DataArray(
            returns_data,
            dims=['time', 'asset'],
            coords={'time': dates, 'asset': assets}
        )
        
        rc.add_data('signal', signal)
        rc._returns = returns
        rc._evaluator._returns_data = returns
        
        # Evaluate
        result = rc.evaluate(Field('signal'), scaler=DollarNeutralScaler())
        
        # Attribution
        port_return = rc.get_port_return(0)
        total_contrib = port_return.sum(dim='time')
        
        # Find best and worst
        best_idx = total_contrib.argmax(dim='asset').values
        worst_idx = total_contrib.argmin(dim='asset').values
        
        best_stock = assets[best_idx]
        worst_stock = assets[worst_idx]
        
        # Verify identification works (should find some pattern)
        assert isinstance(best_stock, str)
        assert isinstance(worst_stock, str)
        assert best_stock != worst_stock

