"""
Visitor pattern for evaluating Expression trees.

This module provides the EvaluateVisitor which traverses Expression trees
in depth-first order, caching intermediate results with integer step indices.
"""

import numpy as np
import xarray as xr
from typing import Dict, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from alpha_canvas.portfolio.base import WeightScaler


class EvaluateVisitor:
    """Evaluates Expression tree with depth-first traversal and caching.
    
    EvaluateVisitor implements the Visitor pattern to execute Expression trees.
    It maintains state (cache, step counter) and provides traceability by caching
    every intermediate computation result.
    
    Attributes:
        _data: Source xarray.Dataset containing all data variables
        _cache: Dictionary mapping step numbers to (name, DataArray) tuples
        _step_counter: Current step number (increments after each node visit)
    
    Cache Structure:
        {
            0: ('Field_returns', <DataArray>),
            1: ('TsMean', <DataArray>),
            2: ('Rank', <DataArray>),
            ...
        }
    
    Example:
        >>> ds = xr.Dataset(...)  # Dataset with 'returns' data
        >>> visitor = EvaluateVisitor(ds)
        >>> 
        >>> field = Field('returns')
        >>> result = visitor.evaluate(field)
        >>> 
        >>> # Access cached intermediate results
        >>> name, data = visitor.get_cached(0)
    """
    
    def __init__(self, data_source: xr.Dataset, data_loader=None):
        """Initialize EvaluateVisitor with data source and optional data loader.
        
        Args:
            data_source: xarray.Dataset containing all data variables
            data_loader: Optional DataLoader for loading fields from Parquet
        
        Example:
            >>> ds = xr.Dataset(coords={'time': [...], 'asset': [...]})
            >>> visitor = EvaluateVisitor(ds)
            >>> 
            >>> # With data loader
            >>> visitor = EvaluateVisitor(ds, data_loader)
        """
        self._data = data_source
        self._data_loader = data_loader
        self._universe_mask: Optional[xr.DataArray] = None  # Set by AlphaCanvas
        
        # Dual-cache architecture for PnL tracing
        self._signal_cache: Dict[int, Tuple[str, xr.DataArray]] = {}  # Persistent
        self._weight_cache: Dict[int, Tuple[str, Optional[xr.DataArray]]] = {}  # Renewable
        
        self._step_counter = 0
        self._scaler: Optional['WeightScaler'] = None  # Current scaler
    
    def evaluate(self, expr, scaler: Optional['WeightScaler'] = None) -> xr.DataArray:
        """Evaluate expression and cache both signal and weights at each step.
        
        This is the main entry point for evaluating an Expression tree.
        It resets the cache and step counter before evaluation to ensure
        each evaluation starts fresh.
        
        **Assignment Handling (Lazy Evaluation):**
        If the expression has assignments (stored via `expr[mask] = value`),
        this method:
        1. Evaluates the base expression (tree traversal)
        2. Caches the base result as a separate step for traceability
        3. Applies assignments sequentially using `_apply_assignments`
        4. Applies universe masking to the final result
        5. Caches the final result
        
        **Weight Caching (Dual-Cache):**
        If scaler is provided:
        - Both signal and weights cached at each step
        - Weights computed by calling scaler.scale(signal) at each step
        If scaler is None:
        - Only signals cached, weight cache remains empty
        Changing scaler:
        - Signal cache repopulated (always fresh evaluation)
        - Weight cache recalculated with new scaler
        
        Args:
            expr: Expression to evaluate
            scaler: Optional WeightScaler to compute portfolio weights at each step
        
        Returns:
            xarray.DataArray result of evaluation (signal, not weights)
        
        Example:
            >>> # Without assignments and without scaler
            >>> field = Field('returns')
            >>> result = visitor.evaluate(field)
            >>> 
            >>> # With weight scaling
            >>> result = visitor.evaluate(field, scaler=DollarNeutralScaler())
            >>> 
            >>> # With assignments (lazy evaluation)
            >>> signal = Field('returns')
            >>> signal[Field('size') == 'small'] = 1.0
            >>> signal[Field('size') == 'big'] = -1.0
            >>> result = visitor.evaluate(signal, scaler=DollarNeutralScaler())
        """
        # Reset state for new evaluation
        self._step_counter = 0
        self._signal_cache = {}
        
        # Check if scaler changed
        scaler_changed = (scaler is not None and scaler is not self._scaler)
        
        if scaler_changed:
            # Scaler changed: reset weight cache
            self._weight_cache = {}
            self._scaler = scaler
        elif scaler is None:
            # No scaler: clear weight cache
            self._weight_cache = {}
            self._scaler = None
        
        # Evaluate base expression (tree traversal, populates signal_cache and weight_cache)
        base_result = expr.accept(self)
        
        # Check if expression has assignments (lazy initialization)
        assignments = getattr(expr, '_assignments', None)
        if assignments:
            # Cache base result for traceability
            base_name = f"{expr.__class__.__name__}_base"
            self._cache_signal_and_weights(base_name, base_result)
            
            # Apply assignments sequentially
            final_result = self._apply_assignments(base_result, assignments)
            
            # Apply universe masking to final result
            if self._universe_mask is not None:
                final_result = final_result.where(self._universe_mask)
            
            # Cache final result
            final_name = f"{expr.__class__.__name__}_with_assignments"
            self._cache_signal_and_weights(final_name, final_result)
            
            return final_result
        
        # No assignments, return base result as-is
        return base_result
    
    def _apply_assignments(self, base_result: xr.DataArray, assignments: list) -> xr.DataArray:
        """Apply assignments sequentially to base result.
        
        Assignments are applied in the order they were added to the Expression.
        For overlapping masks, later assignments overwrite earlier ones
        (sequential application semantics).
        
        Args:
            base_result: Base DataArray to modify (result of base expression)
            assignments: List of (mask, value) tuples
        
        Returns:
            Modified DataArray with assignments applied
        
        Note:
            - Each mask can be an Expression (evaluated lazily here) or DataArray
            - Values are scalars that replace base_result where mask is True
            - NaN values outside the universe are preserved
        
        Example:
            >>> base = xr.DataArray(...)  # All zeros
            >>> assignments = [
            ...     (Field('size') == 'small', 1.0),
            ...     (Field('size') == 'big', -1.0)
            ... ]
            >>> result = visitor._apply_assignments(base, assignments)
            >>> # result: 1.0 where size=='small', -1.0 where size=='big'
        """
        # Start with a copy to avoid mutating the base result
        result = base_result.copy(deep=True)
        
        for mask_expr, value in assignments:
            # If mask is an Expression, evaluate it
            if hasattr(mask_expr, 'accept'):
                mask_data = mask_expr.accept(self)
            else:
                # Already a DataArray or numpy array
                mask_data = mask_expr
            
            # Ensure mask is boolean (required for ~ operator)
            # This handles cases where mask_data might be float or int
            mask_bool = mask_data.astype(bool)
            
            # Apply assignment: replace values where mask is True
            result = result.where(~mask_bool, value)
        
        return result
    
    def visit_field(self, node) -> xr.DataArray:
        """Visit Field node: retrieve from dataset or load from DB with INPUT MASKING.
        
        This method first checks if the field exists in the dataset.
        If not, and a data_loader is available, it loads the field from
        the database (Parquet file) using the data_loader.
        
        Universe masking is applied at this stage (input masking) to ensure
        all data entering the computation pipeline respects the investable universe.
        
        Args:
            node: Field expression node
        
        Returns:
            xarray.DataArray from dataset or loaded from DB (with universe applied)
        
        Raises:
            KeyError: If field name not found in dataset or config
            RuntimeError: If field not in dataset and no data_loader available
        
        Example:
            >>> field = Field('returns')
            >>> result = field.accept(visitor)  # Calls visit_field
        """
        # Check if already in dataset
        if node.name in self._data:
            result = self._data[node.name]
        else:
            # Load from DB using DataLoader
            if self._data_loader is None:
                raise RuntimeError(
                    f"Field '{node.name}' not found in dataset and no DataLoader available. "
                    f"Initialize AlphaCanvas with start_date and end_date to enable data loading."
                )
            result = self._data_loader.load_field(node.name)
            # Add to dataset for caching
            self._data = self._data.assign({node.name: result})
        
        # INPUT MASKING: Apply universe at field retrieval
        if self._universe_mask is not None:
            result = result.where(self._universe_mask, np.nan)
        
        self._cache_signal_and_weights(f"Field_{node.name}", result)
        return result
    
    def visit_constant(self, node) -> xr.DataArray:
        """Visit Constant node: create constant-valued DataArray with panel shape.
        
        Creates a (T, N) DataArray filled with the constant value.
        The shape is determined by the existing dataset's time and asset coordinates.
        Universe masking is applied by the Visitor after evaluation.
        
        Args:
            node: Constant expression node with 'value' attribute
        
        Returns:
            xr.DataArray filled with constant value, panel-shaped (T, N)
        
        Example:
            >>> constant = Constant(0.0)
            >>> result = constant.accept(visitor)  # Creates zeros array
        """
        # Get shape from dataset coordinates
        time_coord = self._data.coords['time']
        asset_coord = self._data.coords['asset']
        
        # Create constant-valued array
        result = xr.DataArray(
            np.full((len(time_coord), len(asset_coord)), node.value),
            dims=['time', 'asset'],
            coords={'time': time_coord, 'asset': asset_coord}
        )
        
        # Universe masking is applied by OUTPUT MASKING in visit_operator
        # or by evaluate() after assignments
        
        self._cache_signal_and_weights(f"Constant_{node.value}", result)
        return result
    
    def visit_operator(self, node) -> xr.DataArray:
        """Generic visitor for operators with OUTPUT MASKING.
        
        This method implements the standard pattern for all operators:
        1. Traverse tree (evaluate child/children) - child already masked
        2. Delegate computation to operator's compute()
        3. Apply universe mask to output (output masking)
        4. Cache result
        
        All operators (TsMean, TsAny, Rank, Boolean, etc.) use this single method,
        eliminating code duplication entirely.
        
        Handles multiple operator patterns:
        - Single child: TsMean, Rank, Not (has 'child' attribute)
        - Binary: And, Or (has 'left' and 'right' attributes)
        - Comparison: Equals, GreaterThan (has 'left' and 'right', right may be literal)
        - CsQuantile: Special case with optional group_by parameter
        
        The double masking strategy (Field input + Operator output) creates
        a trust chain where operators trust their input is masked and ensure
        their output is also masked.
        
        Args:
            node: Expression node with compute() method and child/children attributes
        
        Returns:
            DataArray result from operator's compute() (with universe applied)
        
        Example:
            >>> # Single child operator
            >>> expr = TsMean(child=Field('returns'), window=5)
            >>> result = expr.accept(visitor)
            >>> 
            >>> # Binary operator
            >>> small = Field('size') == 'small'
            >>> high = Field('value') == 'high'
            >>> mask = small & high  # And(small, high)
            >>> result = mask.accept(visitor)
        """
        from alpha_canvas.core.expression import Expression
        from alpha_canvas.ops.classification import CsQuantile
        
        # Special handling for CsQuantile (needs group_by lookup)
        if isinstance(node, CsQuantile):
            # 1. Evaluate child
            child_result = node.child.accept(self)
            
            # 2. Look up group_by field if specified
            group_labels = None
            if node.group_by is not None:
                if node.group_by not in self._data:
                    raise ValueError(
                        f"group_by field '{node.group_by}' not found in dataset"
                    )
                group_labels = self._data[node.group_by]
            
            # 3. Delegate to compute()
            result = node.compute(child_result, group_labels)
            
            # 4. Apply universe masking
            if self._universe_mask is not None:
                result = result.where(self._universe_mask, np.nan)
            
            # 5. Cache
            self._cache_signal_and_weights("CsQuantile", result)
            return result
        
        # 1. Traversal: evaluate child/children expressions
        if hasattr(node, 'child'):
            # Single child operator (TsMean, Rank, Not)
            child_result = node.child.accept(self)
            result = node.compute(child_result)
        
        elif hasattr(node, 'left') and hasattr(node, 'right'):
            # Binary operator (And, Or, Equals, GreaterThan, etc.)
            left_result = node.left.accept(self)
            
            # Check if right is Expression or literal
            if isinstance(node.right, Expression):
                right_result = node.right.accept(self)
                result = node.compute(left_result, right_result)
            else:
                # Right is literal (e.g., 'small', 5.0)
                result = node.compute(left_result)
        
        else:
            # Fallback: assume single child
            # This shouldn't happen if operators are properly defined
            child_result = node.child.accept(self)
            result = node.compute(child_result)
        
        # 3. OUTPUT MASKING: Apply universe to operator result
        if self._universe_mask is not None:
            result = result.where(self._universe_mask, np.nan)
        
        # 4. State collection: cache result with step counter
        operator_name = node.__class__.__name__
        self._cache_signal_and_weights(operator_name, result)
        
        return result
    
    def _cache_signal_and_weights(self, name: str, signal: xr.DataArray):
        """Cache signal and compute/cache weights if scaler is present.
        
        Stores the signal in _signal_cache. If a scaler is configured,
        also computes and caches weights in _weight_cache.
        
        Args:
            name: Descriptive name for this step (e.g., 'Field_returns', 'TsMean')
            signal: Signal DataArray to cache
        
        Note:
            - Signal always cached in _signal_cache
            - If scaler present: weights computed and cached in _weight_cache
            - If scaler None: _weight_cache not updated for this step
            - If scaling fails (e.g., categorical data), None cached for weights
            - Step counter is incremented AFTER caching
        """
        # Always cache signal
        self._signal_cache[self._step_counter] = (name, signal)
        
        # Cache weights if scaler present
        if self._scaler is not None:
            try:
                weights = self._scaler.scale(signal)
                self._weight_cache[self._step_counter] = (name, weights)
            except Exception as e:
                # If scaling fails (e.g., categorical data), cache None
                # This allows mixed step types (some scalable, some not)
                self._weight_cache[self._step_counter] = (name, None)
        
        self._step_counter += 1
    
    def recalculate_weights_with_scaler(self, scaler: 'WeightScaler'):
        """Recalculate all weights from existing signal cache with new scaler.
        
        This enables efficient strategy comparison without re-evaluation.
        The signal cache remains unchanged (efficient!), only weights
        are recalculated.
        
        Args:
            scaler: New WeightScaler to apply
        
        Note:
            - Signal cache unchanged (efficient!)
            - Weight cache completely recalculated
            - Used when user wants to compare different scaling strategies
        
        Example:
            >>> # Initial evaluation
            >>> visitor.evaluate(expr, scaler=DollarNeutralScaler())
            >>> 
            >>> # Later: swap scaler without re-evaluation
            >>> visitor.recalculate_weights_with_scaler(GrossNetScaler(2.0, 0.3))
        """
        self._scaler = scaler
        self._weight_cache = {}
        
        for step_idx in sorted(self._signal_cache.keys()):
            name, signal = self._signal_cache[step_idx]
            
            try:
                weights = scaler.scale(signal)
                self._weight_cache[step_idx] = (name, weights)
            except Exception:
                # Scaling failed (e.g., categorical data)
                self._weight_cache[step_idx] = (name, None)
    
    def get_cached_signal(self, step: int) -> Tuple[str, xr.DataArray]:
        """Retrieve cached signal by step number.
        
        Args:
            step: Step number (0-indexed)
        
        Returns:
            Tuple of (name, signal_DataArray) for the requested step
        
        Raises:
            KeyError: If step number not in cache
        
        Example:
            >>> result = visitor.evaluate(field, scaler=DollarNeutralScaler())
            >>> name, signal = visitor.get_cached_signal(0)
            >>> print(name)  # 'Field_returns'
        """
        return self._signal_cache[step]
    
    def get_cached_weights(self, step: int) -> Tuple[str, Optional[xr.DataArray]]:
        """Retrieve cached weights by step number.
        
        Args:
            step: Step number (0-indexed)
        
        Returns:
            Tuple of (name, weights_DataArray or None)
        
        Raises:
            KeyError: If step number not in cache
        
        Note:
            weights can be None if:
            - No scaler was used during evaluation
            - Scaling failed for this step (e.g., categorical data)
        
        Example:
            >>> result = visitor.evaluate(field, scaler=DollarNeutralScaler())
            >>> name, weights = visitor.get_cached_weights(0)
            >>> if weights is not None:
            ...     print(f"Gross: {abs(weights).sum(dim='asset').mean()}")
        """
        if step not in self._weight_cache:
            return (self._signal_cache[step][0], None)
        return self._weight_cache[step]
    
    def get_cached(self, step: int) -> Tuple[str, xr.DataArray]:
        """Retrieve cached signal by step number (backward compatibility).
        
        This method is maintained for backward compatibility.
        New code should use get_cached_signal() instead.
        
        Args:
            step: Step number (0-indexed)
        
        Returns:
            Tuple of (name, DataArray) for the requested step
        
        Raises:
            KeyError: If step number not in cache
        
        Example:
            >>> result = visitor.evaluate(field)
            >>> name, data = visitor.get_cached(0)
            >>> print(name)  # 'Field_returns'
        """
        return self._signal_cache[step]


