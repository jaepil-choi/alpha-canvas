"""
Expression serialization visitors for alpha-canvas.

This module provides visitor-based serialization, deserialization, and dependency
extraction for Expression trees. This enables saving and loading alpha signals
for persistence and reproducibility.

Design:
- SerializationVisitor: Expression → dict (JSON-compatible)
- DeserializationVisitor: dict → Expression (reconstruction)
- DependencyExtractor: Expression → List[str] (field dependencies)

All visitors follow the same accept() pattern as EvaluateVisitor for consistency.
"""

from typing import Dict, Any, List
from alpha_canvas.core.expression import Expression, Field


class SerializationVisitor:
    """Serialize Expression tree to JSON-compatible dict.
    
    This visitor traverses an Expression tree and converts it to a structured
    dictionary that can be saved as JSON. Each Expression type is serialized
    with its type name and parameters.
    
    Example:
        >>> expr = Rank(TsMean(Field('returns'), window=5))
        >>> serializer = SerializationVisitor()
        >>> expr_dict = expr.accept(serializer)
        >>> # {'type': 'Rank', 'child': {'type': 'TsMean', ...}}
    """
    
    def visit_field(self, node: Field) -> Dict[str, Any]:
        """Serialize Field node.
        
        Args:
            node: Field Expression
        
        Returns:
            Dict with type and field name
        """
        return {
            'type': 'Field',
            'name': node.name
        }
    
    def visit_constant(self, node) -> Dict[str, Any]:
        """Serialize Constant node.
        
        Args:
            node: Constant Expression
        
        Returns:
            Dict with type and constant value
        """
        return {
            'type': 'Constant',
            'value': node.value
        }
    
    def visit_operator(self, node: Expression) -> Dict[str, Any]:
        """Serialize operator node (handles all operator types).
        
        This method dispatches to the appropriate serialization logic based
        on the operator type. It handles:
        - Time-series operators (TsMean, TsAny)
        - Cross-section operators (Rank)
        - Classification operators (CsQuantile)
        - Comparison operators (Equals, NotEquals, GreaterThan, etc.)
        - Logical operators (And, Or, Not)
        
        Args:
            node: Operator Expression
        
        Returns:
            Dict with type, parameters, and serialized children
        """
        node_type = type(node).__name__
        
        # Time-series operators
        if node_type == 'TsMean':
            return {
                'type': 'TsMean',
                'child': node.child.accept(self),
                'window': node.window
            }
        
        elif node_type == 'TsAny':
            return {
                'type': 'TsAny',
                'child': node.child.accept(self),
                'window': node.window
            }
        
        # Cross-section operators
        elif node_type == 'Rank':
            return {
                'type': 'Rank',
                'child': node.child.accept(self)
            }
        
        # Classification operators
        elif node_type == 'CsQuantile':
            return {
                'type': 'CsQuantile',
                'child': node.child.accept(self),
                'bins': node.bins,
                'labels': node.labels,
                'group_by': node.group_by
            }
        
        # Comparison operators (binary with special right-hand side handling)
        elif node_type in ['Equals', 'NotEquals', 'GreaterThan', 'LessThan', 
                           'GreaterOrEqual', 'LessOrEqual']:
            # Check if right is an Expression or a literal
            right_is_expr = isinstance(node.right, Expression)
            return {
                'type': node_type,
                'left': node.left.accept(self),
                'right': node.right.accept(self) if right_is_expr else node.right,
                'right_is_expr': right_is_expr
            }
        
        # Logical operators
        elif node_type == 'And':
            return {
                'type': 'And',
                'left': node.left.accept(self),
                'right': node.right.accept(self)
            }
        
        elif node_type == 'Or':
            return {
                'type': 'Or',
                'left': node.left.accept(self),
                'right': node.right.accept(self)
            }
        
        elif node_type == 'Not':
            return {
                'type': 'Not',
                'child': node.child.accept(self)
            }
        
        else:
            raise ValueError(f"Unknown operator type: {node_type}")


class DeserializationVisitor:
    """Deserialize dict to Expression tree.
    
    This visitor reconstructs Expression objects from serialized dictionaries.
    It uses type dispatch to route to the appropriate Expression constructor.
    
    Example:
        >>> expr_dict = {'type': 'Field', 'name': 'returns'}
        >>> expr = DeserializationVisitor.from_dict(expr_dict)
        >>> # Field('returns')
    """
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Expression:
        """Reconstruct Expression from serialized dict.
        
        Args:
            data: Serialized Expression dict with 'type' key
        
        Returns:
            Reconstructed Expression object
        
        Raises:
            ValueError: If expression type is unknown
        """
        expr_type = data['type']
        
        # Leaf nodes
        if expr_type == 'Field':
            return Field(data['name'])
        
        elif expr_type == 'Constant':
            from alpha_canvas.ops.constants import Constant
            return Constant(data['value'])
        
        # Time-series operators
        elif expr_type == 'TsMean':
            from alpha_canvas.ops.timeseries import TsMean
            child = DeserializationVisitor.from_dict(data['child'])
            return TsMean(child=child, window=data['window'])
        
        elif expr_type == 'TsAny':
            from alpha_canvas.ops.timeseries import TsAny
            child = DeserializationVisitor.from_dict(data['child'])
            return TsAny(child=child, window=data['window'])
        
        # Cross-section operators
        elif expr_type == 'Rank':
            from alpha_canvas.ops.crosssection import Rank
            child = DeserializationVisitor.from_dict(data['child'])
            return Rank(child=child)
        
        # Classification operators
        elif expr_type == 'CsQuantile':
            from alpha_canvas.ops.classification import CsQuantile
            child = DeserializationVisitor.from_dict(data['child'])
            return CsQuantile(
                child=child,
                bins=data['bins'],
                labels=data['labels'],
                group_by=data.get('group_by')  # Optional
            )
        
        # Comparison operators
        elif expr_type == 'Equals':
            from alpha_canvas.ops.logical import Equals
            left = DeserializationVisitor.from_dict(data['left'])
            right = (DeserializationVisitor.from_dict(data['right']) 
                    if data['right_is_expr'] else data['right'])
            return Equals(left=left, right=right)
        
        elif expr_type == 'NotEquals':
            from alpha_canvas.ops.logical import NotEquals
            left = DeserializationVisitor.from_dict(data['left'])
            right = (DeserializationVisitor.from_dict(data['right']) 
                    if data['right_is_expr'] else data['right'])
            return NotEquals(left=left, right=right)
        
        elif expr_type == 'GreaterThan':
            from alpha_canvas.ops.logical import GreaterThan
            left = DeserializationVisitor.from_dict(data['left'])
            right = (DeserializationVisitor.from_dict(data['right']) 
                    if data['right_is_expr'] else data['right'])
            return GreaterThan(left=left, right=right)
        
        elif expr_type == 'LessThan':
            from alpha_canvas.ops.logical import LessThan
            left = DeserializationVisitor.from_dict(data['left'])
            right = (DeserializationVisitor.from_dict(data['right']) 
                    if data['right_is_expr'] else data['right'])
            return LessThan(left=left, right=right)
        
        elif expr_type == 'GreaterOrEqual':
            from alpha_canvas.ops.logical import GreaterOrEqual
            left = DeserializationVisitor.from_dict(data['left'])
            right = (DeserializationVisitor.from_dict(data['right']) 
                    if data['right_is_expr'] else data['right'])
            return GreaterOrEqual(left=left, right=right)
        
        elif expr_type == 'LessOrEqual':
            from alpha_canvas.ops.logical import LessOrEqual
            left = DeserializationVisitor.from_dict(data['left'])
            right = (DeserializationVisitor.from_dict(data['right']) 
                    if data['right_is_expr'] else data['right'])
            return LessOrEqual(left=left, right=right)
        
        # Logical operators
        elif expr_type == 'And':
            from alpha_canvas.ops.logical import And
            left = DeserializationVisitor.from_dict(data['left'])
            right = DeserializationVisitor.from_dict(data['right'])
            return And(left=left, right=right)
        
        elif expr_type == 'Or':
            from alpha_canvas.ops.logical import Or
            left = DeserializationVisitor.from_dict(data['left'])
            right = DeserializationVisitor.from_dict(data['right'])
            return Or(left=left, right=right)
        
        elif expr_type == 'Not':
            from alpha_canvas.ops.logical import Not
            child = DeserializationVisitor.from_dict(data['child'])
            return Not(child=child)
        
        else:
            raise ValueError(f"Unknown expression type: {expr_type}")


class DependencyExtractor:
    """Extract Field dependencies from Expression tree.
    
    This visitor traverses an Expression tree and collects all Field names
    that the expression depends on. This is used for data lineage tracking.
    
    Example:
        >>> expr = Rank(TsMean(Field('returns'), window=5))
        >>> deps = DependencyExtractor.extract(expr)
        >>> # ['returns']
    """
    
    def __init__(self):
        """Initialize dependency list."""
        self.dependencies: List[str] = []
    
    def visit_field(self, node: Field) -> None:
        """Visit Field node and record dependency.
        
        Args:
            node: Field Expression
        """
        self.dependencies.append(node.name)
    
    def visit_constant(self, node) -> None:
        """Visit Constant node (no dependencies).
        
        Args:
            node: Constant Expression
        """
        pass  # Constants have no dependencies
    
    def visit_operator(self, node: Expression) -> None:
        """Visit operator node and recursively extract dependencies.
        
        This method handles all operator types by recursively visiting
        their child expressions.
        
        Args:
            node: Operator Expression
        """
        node_type = type(node).__name__
        
        # Single-child operators
        if node_type in ['TsMean', 'TsAny', 'Rank', 'CsQuantile', 'Not']:
            node.child.accept(self)
        
        # Binary operators (comparison and logical)
        elif node_type in ['Equals', 'NotEquals', 'GreaterThan', 'LessThan',
                          'GreaterOrEqual', 'LessOrEqual', 'And', 'Or']:
            node.left.accept(self)
            # Right can be literal or Expression
            if isinstance(node.right, Expression):
                node.right.accept(self)
        
        else:
            raise ValueError(f"Unknown operator type: {node_type}")
    
    @staticmethod
    def extract(expr: Expression) -> List[str]:
        """Extract unique field dependencies from Expression.
        
        Args:
            expr: Expression tree to analyze
        
        Returns:
            List of unique field names (deduplicated)
        
        Example:
            >>> expr = Field('returns') & Field('returns')
            >>> deps = DependencyExtractor.extract(expr)
            >>> # ['returns'] (deduplicated)
        """
        extractor = DependencyExtractor()
        expr.accept(extractor)
        return list(set(extractor.dependencies))  # Deduplicate

