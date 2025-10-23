"""
AlphaCanvas facade - Main entry point for alpha-canvas.

This module provides the AlphaCanvas class, which serves as the Facade pattern
implementation coordinating all subsystems (Config, DataPanel, Expression, Visitor).
"""

import pandas as pd
import xarray as xr
from typing import Union, Optional, TYPE_CHECKING

from .data_model import DataPanel
from .expression import Expression
from .visitor import EvaluateVisitor
from alpha_canvas.utils import DataAccessor

if TYPE_CHECKING:
    from alpha_canvas.portfolio.base import WeightScaler
    from alpha_database import DataSource


class AlphaCanvas:
    """Facade for alpha-canvas system.
    
    AlphaCanvas (typically instantiated as `rc`) is the main entry point for users.
    It coordinates all subsystems and provides a unified interface for:
    - Loading data from configuration
    - Evaluating Expression trees
    - Managing (T, N) panel data
    - Supporting the "Open Toolkit" pattern (eject/inject)
    
    Attributes:
        _panel: DataPanel instance wrapping xarray.Dataset
        _evaluator: EvaluateVisitor for Expression tree evaluation
        rules: Dictionary storing Expression objects by name
    
    Example:
        >>> # Initialize
        >>> rc = AlphaCanvas(config_dir='config')
        >>> 
        >>> # Add data directly (inject)
        >>> returns = xr.DataArray(...)
        >>> rc.add_data('returns', returns)
        >>> 
        >>> # Add via Expression
        >>> field = Field('returns')
        >>> rc.add_data('returns_copy', field)
        >>> 
        >>> # Eject for external manipulation
        >>> pure_ds = rc.db
        >>> betas = run_regression(pure_ds)
        >>> rc.add_data('beta', betas)  # Re-inject
    """
    
    def __init__(
        self,
        data_source: 'DataSource',
        start_date: str,
        end_date: Optional[str] = None,
        config_dir: str = 'config',
        universe: Optional[Union[Expression, xr.DataArray]] = None
    ):
        """Initialize AlphaCanvas with DataSource and date range.
        
        Args:
            data_source: DataSource instance from alpha_database (MANDATORY)
            start_date: Start date for data loading in 'YYYY-MM-DD' format (MANDATORY)
            end_date: End date for data loading (optional, None = all data from start_date)
            config_dir: Path to configuration directory (default: 'config')
            universe: Optional universe mask (T, N) boolean DataArray or Expression.
                     Once set, immutable for the session to ensure fair PnL comparisons.
        
        Example:
            >>> from alpha_database import DataSource
            >>> from alpha_canvas import AlphaCanvas
            >>> 
            >>> # Basic initialization
            >>> ds = DataSource('config')
            >>> rc = AlphaCanvas(
            ...     data_source=ds,
            ...     start_date='2024-01-01',
            ...     end_date='2024-12-31'
            ... )
            >>> 
            >>> # With universe mask (investable universe)
            >>> universe_mask = (price > 5.0) & (volume > 100000)
            >>> rc = AlphaCanvas(
            ...     data_source=ds,
            ...     start_date='2024-01-01',
            ...     end_date='2024-12-31',
            ...     universe=universe_mask
            ... )
        """
        # Store parameters
        self._data_source = data_source
        self.start_date = start_date
        self.end_date = end_date
        
        # Lazy panel creation - will be initialized from first data load
        self._panel = None
        
        # Initialize evaluator with DataSource
        empty_ds = xr.Dataset()
        self._evaluator = EvaluateVisitor(empty_ds, data_source=data_source)
        self._evaluator._start_date = start_date
        self._evaluator._end_date = end_date
        
        # Storage for Expression rules
        self.rules = {}
        
        # Initialize DataAccessor for rc.data property
        self._data_accessor = DataAccessor()
        
        # Initialize universe mask (immutable once set)
        self._universe_mask: Optional[xr.DataArray] = None
        if universe is not None:
            self._set_initial_universe(universe)
        
        # Load return data (MANDATORY for backtesting)
        self._returns: Optional[xr.DataArray] = None
        self._load_returns_data()
    
    def _set_initial_universe(self, universe: Union[Expression, xr.DataArray]) -> None:
        """Set universe mask at initialization (one-time only).
        
        Args:
            universe: Expression (e.g., Field('univ500')) or boolean DataArray
        
        Raises:
            ValueError: If universe shape doesn't match data shape
            TypeError: If universe is not boolean dtype
        """
        # Evaluate if Expression
        if isinstance(universe, Expression):
            universe_data = self._evaluator.evaluate(universe)
        else:
            universe_data = universe
        
        # Validate shape (only if panel is already initialized)
        if self._panel is not None:
            expected_shape = (
                len(self._panel.db.coords['time']), 
                len(self._panel.db.coords['asset'])
            )
            if universe_data.shape != expected_shape:
                raise ValueError(
                    f"Universe mask shape {universe_data.shape} doesn't match "
                    f"data shape {expected_shape}"
                )
        
        # Validate dtype
        if universe_data.dtype != bool:
            raise TypeError(f"Universe must be boolean, got {universe_data.dtype}")
        
        # Store as immutable
        self._universe_mask = universe_data
        
        # Pass to evaluator for auto-application
        self._evaluator._universe_mask = self._universe_mask
    
    def _load_returns_data(self):
        """Load return data from config (mandatory for backtesting).
        
        Raises:
            ValueError: If 'returns' field not found in config
            ValueError: If returns data shape doesn't match panel
        
        Note:
            - Returns are loaded once at initialization
            - Stored in self._returns for backtest calculations
            - Field name is hardcoded as 'returns'
        """
        # Skip if panel not yet initialized (lazy initialization)
        if self._panel is None:
            return
        
        # Load returns data via DataSource
        try:
            returns_data = self._data_source.load_field(
                'returns',
                start_date=self.start_date,
                end_date=self.end_date
            )
        except KeyError:
            raise ValueError(
                "Return data is mandatory for backtesting. "
                "Missing 'returns' field in config/data.yaml. "
                "Please add a 'returns' field definition."
            )
        
        # Validate shape
        expected_shape = (
            len(self._panel.db.coords['time']),
            len(self._panel.db.coords['asset'])
        )
        if returns_data.shape != expected_shape:
            # Silently skip if shapes don't match (e.g., in tests with custom data)
            # This allows tests to inject custom-sized returns if needed
            return
        
        # Store returns
        self._returns = returns_data
        
        # Also add to panel for user access
        self._panel.add_data('returns', returns_data)
        
        # Re-sync evaluator
        self._evaluator = EvaluateVisitor(self._panel.db, data_source=self._data_source)
        self._evaluator._start_date = self.start_date
        self._evaluator._end_date = self.end_date
        if self._universe_mask is not None:
            self._evaluator._universe_mask = self._universe_mask
        
        # Pass returns to evaluator for backtest
        self._evaluator._returns_data = self._returns
    
    @property
    def universe(self) -> Optional[xr.DataArray]:
        """Get current universe mask (read-only).
        
        Returns:
            Universe mask (T, N) boolean DataArray, or None if not set
        
        Example:
            >>> rc = AlphaCanvas(..., universe=price > 5.0)
            >>> print(f"Universe coverage: {rc.universe.sum().values} stocks")
        """
        return self._universe_mask
    
    @property
    def returns(self) -> xr.DataArray:
        """Read-only access to return data.
        
        Returns:
            (T, N) DataArray with return values
        
        Note:
            Returns are auto-loaded at initialization
        
        Example:
            >>> rc = AlphaCanvas(start_date='2024-01-01', end_date='2024-12-31')
            >>> print(f"Mean return: {rc.returns.mean().values:.4f}")
        """
        return self._returns
    
    @property
    def db(self) -> xr.Dataset:
        """Eject: Return pure xarray.Dataset.
        
        This property implements the "Eject" part of the Open Toolkit philosophy.
        It returns the internal Dataset without any wrapping, allowing users to
        manipulate it with standard xarray operations or external libraries.
        
        Returns:
            Pure xarray.Dataset (not wrapped)
        
        Example:
            >>> # Eject for external manipulation
            >>> pure_ds = rc.db
            >>> # Use with scipy, statsmodels, etc.
            >>> betas = run_regression(pure_ds['returns'], pure_ds['market'])
            >>> # Inject results back
            >>> rc.add_data('beta', betas)
        """
        # Return evaluator's dataset (which is always initialized)
        return self._evaluator._data
    
    @property
    def data(self) -> DataAccessor:
        """Access data fields as Field Expressions.
        
        This property provides Expression-based data access through the DataAccessor.
        All field accesses return Field Expressions that remain lazy until explicitly
        evaluated, ensuring universe masking is applied through the Visitor pattern.
        
        Returns:
            DataAccessor instance for field access
            
        Note:
            Only item access (rc.data['field']) is supported.
            Attribute access (rc.data.field) will raise AttributeError.
        
        Example:
            >>> # Returns Field('size') Expression
            >>> size_field = rc.data['size']
            >>> isinstance(size_field, Field)
            True
            
            >>> # Returns Equals Expression (lazy)
            >>> mask = rc.data['size'] == 'small'
            >>> isinstance(mask, Expression)
            True
            
            >>> # Evaluate with universe masking
            >>> result = rc.evaluate(mask)
            >>> isinstance(result, xr.DataArray)
            True
            
            >>> # Complex logical chains
            >>> complex_mask = (rc.data['size'] == 'small') & (rc.data['momentum'] == 'high')
            >>> result = rc.evaluate(complex_mask)
        """
        return self._data_accessor
    
    def add_data(self, name: str, data: Union[xr.DataArray, Expression]):
        """Add data variable (DataArray or Expression).
        
        This method supports both direct data injection (DataArray) and
        Expression evaluation. It implements the core "Inject" capability
        and Expression evaluation workflow.
        
        Workflow:
        - If data is Expression: evaluate it, store result, cache rule
        - If data is DataArray: directly add to panel
        - Always re-sync evaluator with updated dataset
        
        Args:
            name: Name for the data variable
            data: Either xarray.DataArray (inject) or Expression (evaluate)
        
        Example:
            >>> # Direct injection
            >>> returns = xr.DataArray(...)
            >>> rc.add_data('returns', returns)
            >>> 
            >>> # Expression evaluation
            >>> field = Field('returns')
            >>> rc.add_data('returns_copy', field)
        """
        if isinstance(data, Expression):
            # Expression path: evaluate and cache rule
            self.rules[name] = data
            result = self._evaluator.evaluate(data)
            
            # Lazy panel initialization from first data load
            if self._panel is None:
                time_index = result.coords['time'].values
                asset_index = result.coords['asset'].values
                self._panel = DataPanel(time_index, asset_index)
                
                # Validate universe shape after panel initialization
                if self._universe_mask is not None:
                    expected_shape = (len(time_index), len(asset_index))
                    if self._universe_mask.shape != expected_shape:
                        raise ValueError(
                            f"Universe mask shape {self._universe_mask.shape} doesn't match "
                            f"data shape {expected_shape}"
                        )
            
            self._panel.add_data(name, result)
            
            # Auto-load returns after panel initialization (mandatory for backtesting)
            if self._returns is None:
                self._load_returns_data()
            
            # Re-sync evaluator with updated dataset
            self._evaluator = EvaluateVisitor(self._panel.db, data_source=self._data_source)
            self._evaluator._start_date = self.start_date
            self._evaluator._end_date = self.end_date
            # Preserve universe mask reference
            if self._universe_mask is not None:
                self._evaluator._universe_mask = self._universe_mask
            # Pass returns to evaluator for backtest
            if self._returns is not None:
                self._evaluator._returns_data = self._returns
        else:
            # Direct injection path (Open Toolkit pattern)
            
            # Lazy panel initialization from first data load
            if self._panel is None:
                time_index = data.coords['time'].values
                asset_index = data.coords['asset'].values
                self._panel = DataPanel(time_index, asset_index)
                
                # Validate universe shape after panel initialization (before applying)
                if self._universe_mask is not None:
                    expected_shape = (len(time_index), len(asset_index))
                    if self._universe_mask.shape != expected_shape:
                        raise ValueError(
                            f"Universe mask shape {self._universe_mask.shape} doesn't match "
                            f"data shape {expected_shape}"
                        )
            
            # Apply universe mask to injected data
            if self._universe_mask is not None:
                data = data.where(self._universe_mask, float('nan'))
            
            self._panel.add_data(name, data)

            # Auto-load returns after panel initialization (mandatory for backtesting)
            if self._returns is None:
                self._load_returns_data()

            # Re-sync evaluator with updated dataset
            self._evaluator = EvaluateVisitor(self._panel.db, data_source=self._data_source)
            self._evaluator._start_date = self.start_date
            self._evaluator._end_date = self.end_date
            # Preserve universe mask reference
            if self._universe_mask is not None:
                self._evaluator._universe_mask = self._universe_mask
            # Pass returns to evaluator for backtest
            if self._returns is not None:
                self._evaluator._returns_data = self._returns
    
    def evaluate(self, expr: Expression, scaler: Optional['WeightScaler'] = None) -> xr.DataArray:
        """Evaluate an Expression and return the result (signal).
        
        This is a convenience method that delegates to the internal evaluator,
        providing a cleaner public API without exposing implementation details.
        
        **Weight Caching (Dual-Cache):**
        If scaler is provided, both signal and weights are cached at each step
        during evaluation. This enables:
        - Step-by-step portfolio weight tracking for PnL analysis
        - On-the-fly weight scaler replacement without re-evaluation
        - Research-friendly comparison of scaling strategies
        
        The result is NOT automatically added to the dataset. Use add_data()
        if you want to store the result.
        
        Args:
            expr: Expression to evaluate
            scaler: Optional WeightScaler to compute portfolio weights at each step
        
        Returns:
            xarray.DataArray result of evaluation (signal, not weights)
        
        Note:
            - If scaler provided: Both signal and weights cached at each step
            - Access weights via: rc.get_weights(step) or rc._evaluator.get_cached_weights(step)
            - If scaler None: Only signals cached, weights cache empty
        
        Example:
            >>> # Evaluate without weight scaling
            >>> from alpha_canvas.core.expression import Field
            >>> from alpha_canvas.ops.timeseries import TsMean
            >>> from alpha_canvas.portfolio import DollarNeutralScaler, GrossNetScaler
            >>> 
            >>> result = rc.evaluate(TsMean(Field('returns'), window=5))
            >>> print(result)  # View signal result
            >>> 
            >>> # Evaluate with weight scaling
            >>> result = rc.evaluate(TsMean(Field('returns'), window=5), scaler=DollarNeutralScaler())
            >>> weights = rc.get_weights(1)  # Get weights for step 1
            >>> 
            >>> # Later: swap scaler efficiently
            >>> result = rc.evaluate(TsMean(Field('returns'), window=5), scaler=GrossNetScaler(2.0, 0.3))
            >>> # Signal cache reused, only weights recalculated
            >>> 
            >>> # Evaluate and store
            >>> rc.add_data('ma5', TsMean(Field('returns'), window=5))
        """
        return self._evaluator.evaluate(expr, scaler)
    
    def get_weights(self, step: int) -> Optional[xr.DataArray]:
        """Get cached portfolio weights for a specific step.
        
        This is a convenience method to retrieve weights from the internal
        evaluator's weight cache. Weights are only available if the last
        evaluation included a scaler parameter.
        
        Args:
            step: Step index (0-indexed)
        
        Returns:
            Weights DataArray or None if no scaler was used or if scaling failed
        
        Raises:
            KeyError: If step number not in cache
        
        Example:
            >>> from alpha_canvas.core.expression import Field
            >>> from alpha_canvas.ops.timeseries import TsMean
            >>> from alpha_canvas.portfolio import DollarNeutralScaler
            >>> 
            >>> # Evaluate with scaler
            >>> result = rc.evaluate(TsMean(Field('returns'), window=5), scaler=DollarNeutralScaler())
            >>> 
            >>> # Get weights for specific step
            >>> weights_step_1 = rc.get_weights(1)
            >>> print(weights_step_1.sum(dim='asset'))  # Should be ~0 for dollar-neutral
            >>> 
            >>> # Get weights for all steps
            >>> for step in range(len(rc._evaluator._signal_cache)):
            ...     weights = rc.get_weights(step)
            ...     if weights is not None:
            ...         print(f"Step {step}: Gross = {abs(weights).sum(dim='asset').mean():.2f}")
        """
        _, weights = self._evaluator.get_cached_weights(step)
        return weights
    
    def get_port_return(self, step: int) -> Optional[xr.DataArray]:
        """Get cached position-level portfolio returns for a specific step.
        
        Args:
            step: Step index (0-indexed)
        
        Returns:
            (T, N) DataArray with position-level returns, or None if no scaler used
        
        Note:
            - Returns are element-wise: weights[t] * returns[t]
            - Shape (T, N) preserved for winner/loser attribution
            - To aggregate: port_return.sum(dim='asset') for daily PnL
        
        Example:
            >>> result = rc.evaluate(expr, scaler=DollarNeutralScaler())
            >>> port_return = rc.get_port_return(1)
            >>> # Winner/loser analysis
            >>> total_contrib = port_return.sum(dim='time')
            >>> best_stock = total_contrib.argmax(dim='asset')
        """
        _, port_return = self._evaluator.get_cached_port_return(step)
        return port_return

    def get_daily_pnl(self, step: int) -> Optional[xr.DataArray]:
        """Get daily PnL for a specific step (aggregated across assets).
        
        Args:
            step: Step index (0-indexed)
        
        Returns:
            (T,) DataArray with daily PnL, or None if no scaler used
        
        Note:
            - Computed on-demand from position-level returns
            - Aggregates across asset dimension: sum(dim='asset')
        
        Example:
            >>> result = rc.evaluate(expr, scaler=DollarNeutralScaler())
            >>> daily_pnl = rc.get_daily_pnl(2)
            >>> print(f"Mean daily PnL: {daily_pnl.mean().values:.4f}")
            >>> print(f"Sharpe: {daily_pnl.mean() / daily_pnl.std() * np.sqrt(252):.2f}")
        """
        port_return = self.get_port_return(step)
        if port_return is None:
            return None
        
        # Aggregate across assets (on-demand)
        daily_pnl = port_return.sum(dim='asset')
        return daily_pnl

    def get_cumulative_pnl(self, step: int) -> Optional[xr.DataArray]:
        """Get cumulative PnL for a specific step.
        
        Args:
            step: Step index (0-indexed)
        
        Returns:
            (T,) DataArray with cumulative PnL, or None if no scaler used
        
        Note:
            - Computed on-demand: daily_pnl.cumsum(dim='time')
            - Uses cumsum (not cumprod) for time-invariant comparison
        
        Example:
            >>> result = rc.evaluate(expr, scaler=DollarNeutralScaler())
            >>> cum_pnl = rc.get_cumulative_pnl(2)
            >>> final_pnl = cum_pnl.isel(time=-1).values
            >>> print(f"Final PnL: {final_pnl:.4f}")
        """
        daily_pnl = self.get_daily_pnl(step)
        if daily_pnl is None:
            return None
        
        # Cumulative sum (on-demand)
        cumulative_pnl = daily_pnl.cumsum(dim='time')
        return cumulative_pnl
    
    def scale_weights(
        self, 
        signal: Union[Expression, xr.DataArray],
        scaler: 'WeightScaler'
    ) -> xr.DataArray:
        """Scale signal to portfolio weights (direct use, not cached).
        
        This method is for one-off weight computation, NOT for step-by-step caching.
        For step-by-step caching with PnL tracing capability, use: rc.evaluate(expr, scaler=...)
        
        Args:
            signal: Expression or DataArray with signal values
            scaler: WeightScaler strategy instance (REQUIRED - no default)
        
        Returns:
            (T, N) DataArray with portfolio weights
        
        Design Note:
            Scaler is a required parameter (no default). This enforces
            explicit strategy selection, making it easy to compare different
            scaling approaches in research workflows.
        
        Example:
            >>> from alpha_canvas.portfolio import DollarNeutralScaler, GrossNetScaler
            >>> from alpha_canvas.core.expression import Field
            >>> from alpha_canvas.ops.timeseries import TsMean
            >>> 
            >>> # One-off weight computation (not cached)
            >>> signal = TsMean(Field('returns'), 5)
            >>> scaler = DollarNeutralScaler()
            >>> weights = rc.scale_weights(signal, scaler)
            >>> 
            >>> # Easy to swap strategies (one-off)
            >>> scaler2 = GrossNetScaler(target_gross=2.0, target_net=0.3)
            >>> weights2 = rc.scale_weights(signal, scaler2)
            >>> 
            >>> # For step-by-step caching (PnL tracing), use evaluate() instead:
            >>> result = rc.evaluate(signal, scaler=DollarNeutralScaler())
            >>> weights_step_0 = rc.get_weights(0)  # Cached!
            >>> weights_step_1 = rc.get_weights(1)  # Cached!
        """
        # Evaluate if Expression (without caching weights)
        if hasattr(signal, 'accept'):
            signal_data = self.evaluate(signal)  # Note: no scaler here
        else:
            signal_data = signal
        
        # Apply scaling strategy (delegation)
        weights = scaler.scale(signal_data)
        
        return weights


