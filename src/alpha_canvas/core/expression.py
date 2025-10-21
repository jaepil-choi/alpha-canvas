"""
Expression tree for alpha-canvas.

This module provides the Composite pattern implementation for computation recipes.
Expression objects represent "how to compute" without holding actual data.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


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
        >>> # Evaluate with visitor
        >>> result = visitor.evaluate(mask)
    """
    
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

