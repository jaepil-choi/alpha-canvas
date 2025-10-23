"""
Expression tree for alpha-canvas.

This module provides the Composite pattern implementation for computation recipes.
Expression objects represent "how to compute" without holding actual data.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List


class Expression(ABC):
    """Base class for all expression nodes.
    
    Expression uses the Composite pattern to represent computation trees.
    Each node defines "what to compute" without holding actual (T, N) data.
    
    The Visitor pattern (via accept()) is used to execute the computation:
    - Expression: Defines the structure (what)
    - Visitor: Performs the execution (how)
    
    Boolean Expression Support:
    - Comparison operators (==, !=, <, >, <=, >=) create Boolean Expressions
    - Logical operators (&, |, ~) combine Boolean Expressions
    - All operations remain lazy until evaluated through Visitor
    
    Signal Assignment Support (Lazy Evaluation):
    - __setitem__ stores assignment operations as (mask, value) tuples
    - Assignments are applied during evaluation by Visitor
    - Later assignments overwrite earlier ones for overlapping positions
    
    Example:
        >>> # Leaf node
        >>> field = Field('returns')
        >>> 
        >>> # Composite node
        >>> smoothed = TsMean(field, window=10)
        >>> 
        >>> # Boolean Expression (lazy)
        >>> size = Field('size')
        >>> mask = size == 'small'  # Creates Equals Expression
        >>> 
        >>> # Signal assignment (lazy)
        >>> signal = Field('returns')
        >>> signal[mask] = 1.0  # Stored, not evaluated
        >>> 
        >>> # Evaluate with visitor
        >>> result = visitor.evaluate(signal)
    """
    
    def __setitem__(self, mask, value):
        """Store assignment for lazy evaluation.
        
        Args:
            mask: Boolean Expression or DataArray indicating where to assign
            value: Scalar value to assign where mask is True
        
        Note:
            Assignments are stored as (mask, value) tuples and applied sequentially
            during evaluation. Later assignments overwrite earlier ones for overlapping
            positions.
            
            Uses lazy initialization - _assignments list is created on first use.
        
        Example:
            >>> signal = Field('returns')
            >>> signal[Field('size') == 'small'] = 1.0
            >>> signal[Field('size') == 'big'] = -1.0
            >>> # Assignments stored, not evaluated
        """
        # Lazy initialization - create _assignments if it doesn't exist
        if not hasattr(self, '_assignments'):
            self._assignments = []
        
        self._assignments.append((mask, value))
    
    @abstractmethod
    def accept(self, visitor):
        """Accept a visitor (Visitor pattern).
        
        This method delegates execution to the visitor, which knows how to
        process each specific expression type.
        
        Args:
            visitor: Visitor instance (e.g., EvaluateVisitor)
        
        Returns:
            Result of visiting this node (typically xr.DataArray)
        """
        pass
    
    # Comparison operators (create Boolean Expressions)
    def __eq__(self, other):
        """Equality comparison → Equals Expression.
        
        Args:
            other: Value or Expression to compare with
        
        Returns:
            Equals Expression (lazy, not evaluated)
        
        Example:
            >>> size = Field('size')
            >>> mask = size == 'small'  # Equals(Field('size'), 'small')
        """
        from alpha_canvas.ops.logical import Equals
        return Equals(self, other)
    
    def __ne__(self, other):
        """Not-equal comparison → NotEquals Expression."""
        from alpha_canvas.ops.logical import NotEquals
        return NotEquals(self, other)
    
    def __gt__(self, other):
        """Greater-than comparison → GreaterThan Expression."""
        from alpha_canvas.ops.logical import GreaterThan
        return GreaterThan(self, other)
    
    def __lt__(self, other):
        """Less-than comparison → LessThan Expression."""
        from alpha_canvas.ops.logical import LessThan
        return LessThan(self, other)
    
    def __ge__(self, other):
        """Greater-or-equal comparison → GreaterOrEqual Expression."""
        from alpha_canvas.ops.logical import GreaterOrEqual
        return GreaterOrEqual(self, other)
    
    def __le__(self, other):
        """Less-or-equal comparison → LessOrEqual Expression."""
        from alpha_canvas.ops.logical import LessOrEqual
        return LessOrEqual(self, other)
    
    # Logical operators (combine Boolean Expressions)
    def __and__(self, other):
        """Logical AND → And Expression.
        
        Args:
            other: Another Boolean Expression
        
        Returns:
            And Expression (lazy)
        
        Example:
            >>> small = Field('size') == 'small'
            >>> high = Field('value') == 'high'
            >>> mask = small & high  # And(small, high)
        """
        from alpha_canvas.ops.logical import And
        return And(self, other)
    
    def __or__(self, other):
        """Logical OR → Or Expression."""
        from alpha_canvas.ops.logical import Or
        return Or(self, other)
    
    def __invert__(self):
        """Logical NOT → Not Expression.
        
        Returns:
            Not Expression (lazy)
        
        Example:
            >>> small = Field('size') == 'small'
            >>> not_small = ~small  # Not(small)
        """
        from alpha_canvas.ops.logical import Not
        return Not(self)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize Expression to dict (convenience wrapper).
        
        Returns:
            JSON-serializable dict representation
            
        Example:
            >>> expr = Rank(TsMean(Field('returns'), window=5))
            >>> expr_dict = expr.to_dict()
            >>> # {'type': 'Rank', 'child': {'type': 'TsMean', ...}}
        """
        from .serialization import SerializationVisitor
        return self.accept(SerializationVisitor())
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Expression':
        """Deserialize dict to Expression (convenience wrapper).
        
        Args:
            data: Serialized Expression dict
        
        Returns:
            Reconstructed Expression object
            
        Example:
            >>> expr_dict = {...}
            >>> expr = Expression.from_dict(expr_dict)
        """
        from .serialization import DeserializationVisitor
        return DeserializationVisitor.from_dict(data)
    
    def get_field_dependencies(self) -> List[str]:
        """Extract Field dependencies (convenience wrapper).
        
        Returns:
            List of unique field names this Expression depends on
            
        Example:
            >>> expr = Rank(TsMean(Field('returns'), window=5))
            >>> deps = expr.get_field_dependencies()
            >>> # ['returns']
        """
        from .serialization import DependencyExtractor
        return DependencyExtractor.extract(self)


@dataclass(eq=False)  # Disable dataclass __eq__ to use Expression operators
class Field(Expression):
    """Leaf node: Reference to a data field.
    
    Field represents a reference to data that exists in the dataset or
    should be loaded from the database (via config).
    
    Attributes:
        name: Name of the field to retrieve (e.g., 'returns', 'market_cap')
    
    Example:
        >>> field = Field('returns')
        >>> result = visitor.evaluate(field)
        >>> # result is xr.DataArray with (T, N) returns data
    """
    name: str
    
    def accept(self, visitor):
        """Accept visitor and delegate to visit_field().
        
        Args:
            visitor: Visitor instance with visit_field() method
        
        Returns:
            Result from visitor.visit_field(self)
        """
        return visitor.visit_field(self)

