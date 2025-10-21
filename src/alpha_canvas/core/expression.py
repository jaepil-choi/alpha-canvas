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
    
    Example:
        >>> # Leaf node
        >>> field = Field('returns')
        >>> 
        >>> # Composite node (future)
        >>> smoothed = TsMean(field, window=10)
        >>> 
        >>> # Evaluate with visitor
        >>> result = visitor.evaluate(smoothed)
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


@dataclass
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

