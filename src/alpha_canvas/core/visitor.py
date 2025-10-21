"""
Visitor pattern for evaluating Expression trees.

This module provides the EvaluateVisitor which traverses Expression trees
in depth-first order, caching intermediate results with integer step indices.
"""

import xarray as xr
from typing import Dict, Tuple


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
        self._cache: Dict[int, Tuple[str, xr.DataArray]] = {}
        self._step_counter = 0
    
    def evaluate(self, expr) -> xr.DataArray:
        """Evaluate expression and return result.
        
        This is the main entry point for evaluating an Expression tree.
        It resets the cache and step counter before evaluation to ensure
        each evaluation starts fresh.
        
        Args:
            expr: Expression to evaluate
        
        Returns:
            xarray.DataArray result of evaluation
        
        Example:
            >>> field = Field('returns')
            >>> result = visitor.evaluate(field)
            >>> assert result.shape == (100, 50)
        """
        # Reset state for new evaluation
        self._step_counter = 0
        self._cache = {}
        
        # Start depth-first traversal
        return expr.accept(self)
    
    def visit_field(self, node) -> xr.DataArray:
        """Visit Field node: retrieve from dataset or load from DB.
        
        This method first checks if the field exists in the dataset.
        If not, and a data_loader is available, it loads the field from
        the database (Parquet file) using the data_loader.
        
        Args:
            node: Field expression node
        
        Returns:
            xarray.DataArray from dataset or loaded from DB
        
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
        
        self._cache_result(f"Field_{node.name}", result)
        return result
    
    def visit_operator(self, node) -> xr.DataArray:
        """Generic visitor for operators following compute() pattern.
        
        This method implements the standard 3-step pattern for all operators:
        1. Traverse tree (evaluate child/children)
        2. Delegate computation to operator's compute()
        3. Cache result
        
        All operators (TsMean, TsAny, Rank, etc.) use this single method,
        eliminating code duplication entirely.
        
        Args:
            node: Expression node with compute() method and child attribute
        
        Returns:
            DataArray result from operator's compute()
        
        Example:
            >>> # All operators call this via accept(visitor)
            >>> expr = TsMean(child=Field('returns'), window=5)
            >>> result = expr.accept(visitor)  # Calls visit_operator
        """
        # 1. Traversal: evaluate child expression
        child_result = node.child.accept(self)
        
        # 2. Delegation: operator does its own computation
        result = node.compute(child_result)
        
        # 3. State collection: cache result with step counter
        operator_name = node.__class__.__name__
        self._cache_result(operator_name, result)
        
        return result
    
    def _cache_result(self, name: str, result: xr.DataArray):
        """Cache result with current step number.
        
        Stores the result in the cache with the current step number as key,
        then increments the step counter.
        
        Args:
            name: Descriptive name for this step (e.g., 'Field_returns', 'TsMean')
            result: DataArray result to cache
        
        Note:
            Step counter is incremented AFTER caching, so step numbers are
            0-indexed and match the order of cache insertion.
        """
        self._cache[self._step_counter] = (name, result)
        self._step_counter += 1
    
    def get_cached(self, step: int) -> Tuple[str, xr.DataArray]:
        """Retrieve cached result by step number.
        
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
        return self._cache[step]


