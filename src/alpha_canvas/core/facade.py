"""
AlphaCanvas facade - Main entry point for alpha-canvas.

This module provides the AlphaCanvas class, which serves as the Facade pattern
implementation coordinating all subsystems (Config, DataPanel, Expression, Visitor).
"""

import pandas as pd
import xarray as xr
from typing import Union, Optional

from .config import ConfigLoader
from .data_model import DataPanel
from .expression import Expression
from .visitor import EvaluateVisitor
from .data_loader import DataLoader
from alpha_canvas.utils import DataAccessor


class AlphaCanvas:
    """Facade for alpha-canvas system.
    
    AlphaCanvas (typically instantiated as `rc`) is the main entry point for users.
    It coordinates all subsystems and provides a unified interface for:
    - Loading data from configuration
    - Evaluating Expression trees
    - Managing (T, N) panel data
    - Supporting the "Open Toolkit" pattern (eject/inject)
    
    Attributes:
        _config: ConfigLoader instance for YAML configuration
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
        config_dir: str = 'config',
        start_date: str = None,
        end_date: str = None,
        time_index=None,
        asset_index=None,
        universe: Optional[Union[Expression, xr.DataArray]] = None
    ):
        """Initialize AlphaCanvas with configuration and data indices.
        
        Args:
            config_dir: Path to configuration directory (default: 'config')
            start_date: Start date for data loading (YYYY-MM-DD format)
            end_date: End date for data loading (YYYY-MM-DD format)
            time_index: Time index for panel data (used if start_date/end_date not provided)
            asset_index: Asset identifiers (used if start_date/end_date not provided)
            universe: Optional universe mask (T, N) boolean DataArray or Expression.
                     Once set, immutable for the session to ensure fair PnL comparisons.
        
        Example:
            >>> # With date range (loads from Parquet)
            >>> rc = AlphaCanvas(
            ...     start_date='2024-01-01',
            ...     end_date='2024-12-31'
            ... )
            >>> 
            >>> # With custom indices (manual setup)
            >>> time_idx = pd.date_range('2021-01-01', periods=252)
            >>> assets = ['AAPL', 'GOOGL', 'MSFT']
            >>> rc = AlphaCanvas(
            ...     config_dir='custom_config',
            ...     time_index=time_idx,
            ...     asset_index=assets
            ... )
            >>> 
            >>> # With universe mask (investable universe)
            >>> universe_mask = (price > 5.0) & (volume > 100000)
            >>> rc = AlphaCanvas(
            ...     start_date='2024-01-01',
            ...     end_date='2024-12-31',
            ...     universe=universe_mask
            ... )
        """
        # Load configurations
        self._config = ConfigLoader(config_dir)
        
        # Initialize DataLoader if date range provided
        if start_date and end_date:
            self._data_loader = DataLoader(self._config, start_date, end_date)
            # Lazy panel creation - will be initialized from first data load
            self._panel = None
        else:
            self._data_loader = None
            # Initialize data panel with default or custom indices
            if time_index is None:
                time_index = pd.date_range('2020-01-01', periods=100)
            if asset_index is None:
                asset_index = [f'ASSET_{i}' for i in range(50)]
            
            self._panel = DataPanel(time_index, asset_index)
        
        # Initialize evaluator (will be properly set after first data load if lazy)
        if self._panel is not None:
            self._evaluator = EvaluateVisitor(self._panel.db, self._data_loader)
        else:
            # Create empty dataset for lazy initialization
            import xarray as xr
            empty_ds = xr.Dataset()
            self._evaluator = EvaluateVisitor(empty_ds, self._data_loader)
        
        # Storage for Expression rules
        self.rules = {}
        
        # Initialize DataAccessor for rc.data property
        self._data_accessor = DataAccessor()
        
        # Initialize universe mask (immutable once set)
        self._universe_mask: Optional[xr.DataArray] = None
        if universe is not None:
            self._set_initial_universe(universe)
    
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
        
        # Validate shape
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
        return self._panel.db
    
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
            
            self._panel.add_data(name, result)
            
            # Re-sync evaluator with updated dataset
            self._evaluator = EvaluateVisitor(self._panel.db, self._data_loader)
            # Preserve universe mask reference
            if self._universe_mask is not None:
                self._evaluator._universe_mask = self._universe_mask
        else:
            # Direct injection path (Open Toolkit pattern)
            # Apply universe mask to injected data
            if self._universe_mask is not None:
                data = data.where(self._universe_mask, float('nan'))
            
            # Lazy panel initialization from first data load
            if self._panel is None:
                time_index = data.coords['time'].values
                asset_index = data.coords['asset'].values
                self._panel = DataPanel(time_index, asset_index)
            
            self._panel.add_data(name, data)

            # Re-sync evaluator with updated dataset
            self._evaluator = EvaluateVisitor(self._panel.db, self._data_loader)
            # Preserve universe mask reference
            if self._universe_mask is not None:
                self._evaluator._universe_mask = self._universe_mask
    
    def evaluate(self, expr: Expression) -> xr.DataArray:
        """Evaluate an Expression and return the result.
        
        This is a convenience method that delegates to the internal evaluator,
        providing a cleaner public API without exposing implementation details.
        
        The result is NOT automatically added to the dataset. Use add_data()
        if you want to store the result.
        
        Args:
            expr: Expression to evaluate
        
        Returns:
            xarray.DataArray result of evaluation (with universe masking applied)
        
        Example:
            >>> # Evaluate without storing
            >>> from alpha_canvas.core.expression import Field
            >>> from alpha_canvas.ops.timeseries import TsMean
            >>> 
            >>> result = rc.evaluate(TsMean(Field('returns'), window=5))
            >>> print(result)  # View result
            >>> 
            >>> # Evaluate and store
            >>> rc.add_data('ma5', TsMean(Field('returns'), window=5))
            >>> 
            >>> # Compare to direct evaluator access (not recommended)
            >>> # result = rc._evaluator.evaluate(expr)  # Don't do this!
            >>> result = rc.evaluate(expr)  # Do this instead!
        """
        return self._evaluator.evaluate(expr)


