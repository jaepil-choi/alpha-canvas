"""Logical and comparison operators for Boolean Expressions.

These operators enable Expression-based comparisons and logical operations:
- Comparison: Equals, NotEquals, GreaterThan, LessThan, GreaterOrEqual, LessOrEqual
- Logical: And, Or, Not

All Boolean Expressions remain lazy until evaluated through Visitor.
This ensures universe masking is always applied through the Expression system.
"""

from dataclasses import dataclass
from typing import Union, Any
from alpha_canvas.core.expression import Expression


# ==============================================================================
# Comparison Operators
# ==============================================================================

@dataclass(eq=False)  # Disable dataclass __eq__ to use Expression operators
class Equals(Expression):
    """Equality comparison Expression.
    
    Returns boolean DataArray where left == right.
    
    Args:
        left: Left-hand Expression
        right: Right-hand value (literal) or Expression
    
    Returns:
        Boolean DataArray (same shape as input)
        
    Example:
        >>> # String comparison
        >>> size = Field('size')
        >>> small_mask = size == 'small'  # Equals(Field('size'), 'small')
        >>> 
        >>> # Numeric comparison
        >>> price = Field('price')
        >>> cheap_mask = price == 5.0  # Equals(Field('price'), 5.0)
        >>> 
        >>> # Expression comparison
        >>> threshold = Field('threshold')
        >>> mask = price == threshold  # Equals(Field('price'), Field('threshold'))
    
    Notes:
        - NaN == value → False (standard xarray behavior)
        - Works with any data type (string, float, int, bool)
        - Both left and right go through Visitor (universe-masked)
    """
    left: Expression
    right: Union[Expression, Any]
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, left_result: 'xr.DataArray', right_result: Any = None) -> 'xr.DataArray':
        """Core computation logic for equality comparison.
        
        Args:
            left_result: Evaluated left Expression result
            right_result: Evaluated right result (if Expression), or None (if literal)
        
        Returns:
            Boolean DataArray
        """
        if right_result is not None:
            # Right is Expression (was evaluated)
            return left_result == right_result
        else:
            # Right is literal
            return left_result == self.right


@dataclass(eq=False)
class NotEquals(Expression):
    """Not-equal comparison Expression."""
    left: Expression
    right: Union[Expression, Any]
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, left_result: 'xr.DataArray', right_result: Any = None) -> 'xr.DataArray':
        """Core computation logic for not-equal comparison."""
        if right_result is not None:
            return left_result != right_result
        else:
            return left_result != self.right


@dataclass(eq=False)
class GreaterThan(Expression):
    """Greater-than comparison Expression."""
    left: Expression
    right: Union[Expression, Any]
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, left_result: 'xr.DataArray', right_result: Any = None) -> 'xr.DataArray':
        """Core computation logic for greater-than comparison."""
        if right_result is not None:
            return left_result > right_result
        else:
            return left_result > self.right


@dataclass(eq=False)
class LessThan(Expression):
    """Less-than comparison Expression."""
    left: Expression
    right: Union[Expression, Any]
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, left_result: 'xr.DataArray', right_result: Any = None) -> 'xr.DataArray':
        """Core computation logic for less-than comparison."""
        if right_result is not None:
            return left_result < right_result
        else:
            return left_result < self.right


@dataclass(eq=False)
class GreaterOrEqual(Expression):
    """Greater-or-equal comparison Expression."""
    left: Expression
    right: Union[Expression, Any]
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, left_result: 'xr.DataArray', right_result: Any = None) -> 'xr.DataArray':
        """Core computation logic for greater-or-equal comparison."""
        if right_result is not None:
            return left_result >= right_result
        else:
            return left_result >= self.right


@dataclass(eq=False)
class LessOrEqual(Expression):
    """Less-or-equal comparison Expression."""
    left: Expression
    right: Union[Expression, Any]
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, left_result: 'xr.DataArray', right_result: Any = None) -> 'xr.DataArray':
        """Core computation logic for less-or-equal comparison."""
        if right_result is not None:
            return left_result <= right_result
        else:
            return left_result <= self.right


# ==============================================================================
# Logical Operators
# ==============================================================================

@dataclass(eq=False)
class And(Expression):
    """Logical AND Expression.
    
    Combines two boolean Expressions with logical AND.
    
    Args:
        left: Left boolean Expression
        right: Right boolean Expression
    
    Returns:
        Boolean DataArray where both are True
        
    Example:
        >>> small = Field('size') == 'small'
        >>> high = Field('value') == 'high'
        >>> mask = small & high  # And(small, high)
        >>> 
        >>> # Equivalent to:
        >>> mask = (Field('size') == 'small') & (Field('value') == 'high')
    
    Notes:
        - Uses bitwise & operator (not 'and' keyword)
        - NaN & True → False (standard xarray behavior)
        - Short-circuits are not possible (all Expressions evaluated)
    """
    left: Expression
    right: Expression
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, left_result: 'xr.DataArray', right_result: 'xr.DataArray') -> 'xr.DataArray':
        """Core computation logic for logical AND.
        
        Args:
            left_result: Evaluated left boolean Expression
            right_result: Evaluated right boolean Expression
        
        Returns:
            Boolean DataArray (True where both are True)
        """
        return left_result & right_result


@dataclass(eq=False)
class Or(Expression):
    """Logical OR Expression.
    
    Combines two boolean Expressions with logical OR.
    
    Args:
        left: Left boolean Expression
        right: Right boolean Expression
    
    Returns:
        Boolean DataArray where either is True
        
    Example:
        >>> small = Field('size') == 'small'
        >>> big = Field('size') == 'big'
        >>> extremes = small | big  # Or(small, big)
    
    Notes:
        - Uses bitwise | operator (not 'or' keyword)
        - NaN | True → True (standard xarray behavior)
    """
    left: Expression
    right: Expression
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, left_result: 'xr.DataArray', right_result: 'xr.DataArray') -> 'xr.DataArray':
        """Core computation logic for logical OR.
        
        Args:
            left_result: Evaluated left boolean Expression
            right_result: Evaluated right boolean Expression
        
        Returns:
            Boolean DataArray (True where either is True)
        """
        return left_result | right_result


@dataclass(eq=False)
class Not(Expression):
    """Logical NOT Expression.
    
    Inverts a boolean Expression.
    
    Args:
        child: Boolean Expression to invert
    
    Returns:
        Boolean DataArray with inverted values
        
    Example:
        >>> small = Field('size') == 'small'
        >>> not_small = ~small  # Not(small)
        >>> 
        >>> # Equivalent to:
        >>> not_small = ~(Field('size') == 'small')
    
    Notes:
        - Uses bitwise ~ operator (not 'not' keyword)
        - ~NaN → True in some numpy versions (behavior may vary)
    """
    child: Expression
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, child_result: 'xr.DataArray') -> 'xr.DataArray':
        """Core computation logic for logical NOT.
        
        Args:
            child_result: Evaluated child boolean Expression
        
        Returns:
            Boolean DataArray with inverted values
        """
        return ~child_result

