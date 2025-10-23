"""
Tests for Expression serialization visitors.

This module tests serialization, deserialization, and dependency extraction
for all 14 Expression types in alpha-canvas.
"""

import pytest
from alpha_canvas.core.expression import Expression, Field
from alpha_canvas.core.serialization import (
    SerializationVisitor,
    DeserializationVisitor,
    DependencyExtractor
)
from alpha_canvas.ops.constants import Constant
from alpha_canvas.ops.timeseries import TsMean, TsAny
from alpha_canvas.ops.crosssection import Rank
from alpha_canvas.ops.classification import CsQuantile
from alpha_canvas.ops.logical import (
    Equals, NotEquals, GreaterThan, LessThan, GreaterOrEqual, LessOrEqual,
    And, Or, Not
)


class TestSerializationVisitor:
    """Test serialization of Expression trees to dicts."""
    
    def test_serialize_field(self):
        """Test Field serialization."""
        field = Field('returns')
        serializer = SerializationVisitor()
        result = field.accept(serializer)
        
        assert result == {'type': 'Field', 'name': 'returns'}
    
    def test_serialize_constant(self):
        """Test Constant serialization."""
        const = Constant(0.0)
        serializer = SerializationVisitor()
        result = const.accept(serializer)
        
        assert result == {'type': 'Constant', 'value': 0.0}
    
    def test_serialize_ts_mean(self):
        """Test TsMean serialization."""
        expr = TsMean(Field('returns'), window=5)
        serializer = SerializationVisitor()
        result = expr.accept(serializer)
        
        assert result['type'] == 'TsMean'
        assert result['window'] == 5
        assert result['child'] == {'type': 'Field', 'name': 'returns'}
    
    def test_serialize_ts_any(self):
        """Test TsAny serialization."""
        expr = TsAny(Field('surge'), window=3)
        serializer = SerializationVisitor()
        result = expr.accept(serializer)
        
        assert result['type'] == 'TsAny'
        assert result['window'] == 3
        assert result['child'] == {'type': 'Field', 'name': 'surge'}
    
    def test_serialize_rank(self):
        """Test Rank serialization."""
        expr = Rank(Field('returns'))
        serializer = SerializationVisitor()
        result = expr.accept(serializer)
        
        assert result['type'] == 'Rank'
        assert result['child'] == {'type': 'Field', 'name': 'returns'}
    
    def test_serialize_cs_quantile(self):
        """Test CsQuantile serialization with all parameters."""
        expr = CsQuantile(
            Field('market_cap'),
            bins=2,
            labels=['small', 'big'],
            group_by='sector'
        )
        serializer = SerializationVisitor()
        result = expr.accept(serializer)
        
        assert result['type'] == 'CsQuantile'
        assert result['bins'] == 2
        assert result['labels'] == ['small', 'big']
        assert result['group_by'] == 'sector'
        assert result['child'] == {'type': 'Field', 'name': 'market_cap'}
    
    def test_serialize_comparison_operators(self):
        """Test all 6 comparison operators."""
        operators = [
            (Equals, 'Equals'),
            (NotEquals, 'NotEquals'),
            (GreaterThan, 'GreaterThan'),
            (LessThan, 'LessThan'),
            (GreaterOrEqual, 'GreaterOrEqual'),
            (LessOrEqual, 'LessOrEqual')
        ]
        
        serializer = SerializationVisitor()
        
        for OpClass, op_name in operators:
            expr = OpClass(Field('price'), 100.0)
            result = expr.accept(serializer)
            
            assert result['type'] == op_name
            assert result['left'] == {'type': 'Field', 'name': 'price'}
            assert result['right'] == 100.0
            assert result['right_is_expr'] == False
    
    def test_serialize_logical_operators(self):
        """Test And, Or, Not logical operators."""
        # And
        and_expr = And(Field('x'), Field('y'))
        serializer = SerializationVisitor()
        result = and_expr.accept(serializer)
        
        assert result['type'] == 'And'
        assert result['left'] == {'type': 'Field', 'name': 'x'}
        assert result['right'] == {'type': 'Field', 'name': 'y'}
        
        # Or
        or_expr = Or(Field('a'), Field('b'))
        result = or_expr.accept(serializer)
        
        assert result['type'] == 'Or'
        assert result['left'] == {'type': 'Field', 'name': 'a'}
        assert result['right'] == {'type': 'Field', 'name': 'b'}
        
        # Not
        not_expr = Not(Field('condition'))
        result = not_expr.accept(serializer)
        
        assert result['type'] == 'Not'
        assert result['child'] == {'type': 'Field', 'name': 'condition'}
    
    def test_serialize_nested_expression(self):
        """Test serialization of complex nested expression."""
        # Rank(TsMean(Field('returns'), window=5))
        expr = Rank(TsMean(Field('returns'), window=5))
        serializer = SerializationVisitor()
        result = expr.accept(serializer)
        
        assert result['type'] == 'Rank'
        assert result['child']['type'] == 'TsMean'
        assert result['child']['window'] == 5
        assert result['child']['child']['type'] == 'Field'
        assert result['child']['child']['name'] == 'returns'
    
    def test_serialize_comparison_with_literal(self):
        """Test comparison operator with literal right-hand side."""
        expr = Equals(Field('price'), 5.0)
        serializer = SerializationVisitor()
        result = expr.accept(serializer)
        
        assert result['type'] == 'Equals'
        assert result['left'] == {'type': 'Field', 'name': 'price'}
        assert result['right'] == 5.0
        assert result['right_is_expr'] == False
    
    def test_serialize_comparison_with_expression(self):
        """Test comparison operator with Expression right-hand side."""
        expr = Equals(Field('x'), Field('y'))
        serializer = SerializationVisitor()
        result = expr.accept(serializer)
        
        assert result['type'] == 'Equals'
        assert result['left'] == {'type': 'Field', 'name': 'x'}
        assert result['right'] == {'type': 'Field', 'name': 'y'}
        assert result['right_is_expr'] == True


class TestDeserializationVisitor:
    """Test deserialization of dicts to Expression trees."""
    
    def test_deserialize_field(self):
        """Test Field deserialization."""
        data = {'type': 'Field', 'name': 'returns'}
        expr = DeserializationVisitor.from_dict(data)
        
        assert isinstance(expr, Field)
        assert expr.name == 'returns'
    
    def test_deserialize_constant(self):
        """Test Constant deserialization."""
        data = {'type': 'Constant', 'value': 0.0}
        expr = DeserializationVisitor.from_dict(data)
        
        assert isinstance(expr, Constant)
        assert expr.value == 0.0
    
    def test_deserialize_all_operator_types(self):
        """Test deserialization of all 14 Expression types."""
        test_cases = [
            # Time-series
            ({
                'type': 'TsMean',
                'child': {'type': 'Field', 'name': 'returns'},
                'window': 5
            }, TsMean),
            
            ({
                'type': 'TsAny',
                'child': {'type': 'Field', 'name': 'surge'},
                'window': 3
            }, TsAny),
            
            # Cross-section
            ({
                'type': 'Rank',
                'child': {'type': 'Field', 'name': 'returns'}
            }, Rank),
            
            # Classification
            ({
                'type': 'CsQuantile',
                'child': {'type': 'Field', 'name': 'market_cap'},
                'bins': 2,
                'labels': ['small', 'big'],
                'group_by': None
            }, CsQuantile),
            
            # Comparison
            ({
                'type': 'Equals',
                'left': {'type': 'Field', 'name': 'x'},
                'right': 5.0,
                'right_is_expr': False
            }, Equals),
            
            ({
                'type': 'NotEquals',
                'left': {'type': 'Field', 'name': 'x'},
                'right': 5.0,
                'right_is_expr': False
            }, NotEquals),
            
            ({
                'type': 'GreaterThan',
                'left': {'type': 'Field', 'name': 'x'},
                'right': 5.0,
                'right_is_expr': False
            }, GreaterThan),
            
            ({
                'type': 'LessThan',
                'left': {'type': 'Field', 'name': 'x'},
                'right': 5.0,
                'right_is_expr': False
            }, LessThan),
            
            ({
                'type': 'GreaterOrEqual',
                'left': {'type': 'Field', 'name': 'x'},
                'right': 5.0,
                'right_is_expr': False
            }, GreaterOrEqual),
            
            ({
                'type': 'LessOrEqual',
                'left': {'type': 'Field', 'name': 'x'},
                'right': 5.0,
                'right_is_expr': False
            }, LessOrEqual),
            
            # Logical
            ({
                'type': 'And',
                'left': {'type': 'Field', 'name': 'a'},
                'right': {'type': 'Field', 'name': 'b'}
            }, And),
            
            ({
                'type': 'Or',
                'left': {'type': 'Field', 'name': 'a'},
                'right': {'type': 'Field', 'name': 'b'}
            }, Or),
            
            ({
                'type': 'Not',
                'child': {'type': 'Field', 'name': 'condition'}
            }, Not),
        ]
        
        for data, expected_type in test_cases:
            expr = DeserializationVisitor.from_dict(data)
            assert isinstance(expr, expected_type), f"Failed for {data['type']}"
    
    def test_deserialize_nested_expression(self):
        """Test deserialization of nested expression."""
        data = {
            'type': 'Rank',
            'child': {
                'type': 'TsMean',
                'child': {'type': 'Field', 'name': 'returns'},
                'window': 5
            }
        }
        
        expr = DeserializationVisitor.from_dict(data)
        
        assert isinstance(expr, Rank)
        assert isinstance(expr.child, TsMean)
        assert expr.child.window == 5
        assert isinstance(expr.child.child, Field)
        assert expr.child.child.name == 'returns'
    
    def test_round_trip_preserves_structure(self):
        """Test that serialize → deserialize preserves Expression structure."""
        original = Rank(TsMean(Field('returns'), window=5))
        
        # Serialize
        serializer = SerializationVisitor()
        data = original.accept(serializer)
        
        # Deserialize
        reconstructed = DeserializationVisitor.from_dict(data)
        
        # Verify structure
        assert isinstance(reconstructed, Rank)
        assert isinstance(reconstructed.child, TsMean)
        assert reconstructed.child.window == 5
        assert isinstance(reconstructed.child.child, Field)
        assert reconstructed.child.child.name == 'returns'


class TestDependencyExtractor:
    """Test dependency extraction from Expression trees."""
    
    def test_extract_single_field(self):
        """Test extraction from single Field."""
        expr = Field('returns')
        deps = DependencyExtractor.extract(expr)
        
        assert deps == ['returns']
    
    def test_extract_nested_fields(self):
        """Test extraction from nested expression."""
        expr = TsMean(Field('returns'), window=5)
        deps = DependencyExtractor.extract(expr)
        
        assert deps == ['returns']
    
    def test_extract_multiple_fields(self):
        """Test extraction from expression with multiple fields."""
        expr = Equals(Field('x'), Field('y'))
        deps = DependencyExtractor.extract(expr)
        
        assert sorted(deps) == ['x', 'y']
    
    def test_extract_deduplicate(self):
        """Test that duplicate fields are deduplicated."""
        expr = And(Field('x'), Field('x'))
        deps = DependencyExtractor.extract(expr)
        
        assert deps == ['x']
        assert len(deps) == 1
    
    def test_extract_from_complex_tree(self):
        """Test extraction from complex nested tree with multiple fields."""
        # Rank(TsMean(Field('returns'), window=5)) & (Field('size') == 'small')
        left = Rank(TsMean(Field('returns'), window=5))
        right = Equals(Field('size'), 'small')
        expr = And(left, right)
        
        deps = DependencyExtractor.extract(expr)
        
        assert sorted(deps) == ['returns', 'size']
    
    def test_extract_constant_no_dependencies(self):
        """Test that Constant has no dependencies."""
        expr = Constant(0.0)
        deps = DependencyExtractor.extract(expr)
        
        assert deps == []
    
    def test_extract_comparison_with_literal(self):
        """Test extraction from comparison with literal (only left field)."""
        expr = Equals(Field('price'), 100.0)
        deps = DependencyExtractor.extract(expr)
        
        assert deps == ['price']


class TestConvenienceWrappers:
    """Test Expression convenience wrapper methods."""
    
    def test_to_dict_wrapper(self):
        """Test Expression.to_dict() convenience method."""
        expr = Field('returns')
        result = expr.to_dict()
        
        assert result == {'type': 'Field', 'name': 'returns'}
    
    def test_from_dict_wrapper(self):
        """Test Expression.from_dict() convenience method."""
        data = {'type': 'Field', 'name': 'returns'}
        expr = Expression.from_dict(data)
        
        assert isinstance(expr, Field)
        assert expr.name == 'returns'
    
    def test_get_field_dependencies_wrapper(self):
        """Test Expression.get_field_dependencies() convenience method."""
        expr = Rank(TsMean(Field('returns'), window=5))
        deps = expr.get_field_dependencies()
        
        assert deps == ['returns']
    
    def test_wrappers_with_complex_expression(self):
        """Test all wrappers work with complex expressions."""
        original = Rank(TsMean(Field('returns'), window=5))
        
        # to_dict
        data = original.to_dict()
        assert data['type'] == 'Rank'
        
        # from_dict
        reconstructed = Expression.from_dict(data)
        assert isinstance(reconstructed, Rank)
        
        # get_field_dependencies
        deps = original.get_field_dependencies()
        assert deps == ['returns']


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_serialize_assignment(self):
        """Test that _assignments attribute is NOT serialized."""
        expr = Field('returns')
        expr[Field('size') == 'small'] = 1.0  # Add assignment
        
        serializer = SerializationVisitor()
        result = expr.accept(serializer)
        
        # Should only serialize the Field itself, not assignments
        assert result == {'type': 'Field', 'name': 'returns'}
        assert '_assignments' not in result
    
    def test_cs_quantile_with_optional_group_by(self):
        """Test CsQuantile with and without group_by."""
        # Without group_by
        expr1 = CsQuantile(
            Field('market_cap'),
            bins=2,
            labels=['small', 'big']
        )
        serializer = SerializationVisitor()
        result1 = expr1.accept(serializer)
        
        assert result1['group_by'] is None
        
        # With group_by
        expr2 = CsQuantile(
            Field('market_cap'),
            bins=2,
            labels=['small', 'big'],
            group_by='sector'
        )
        result2 = expr2.accept(serializer)
        
        assert result2['group_by'] == 'sector'
        
        # Round-trip both
        reconstructed1 = DeserializationVisitor.from_dict(result1)
        assert reconstructed1.group_by is None
        
        reconstructed2 = DeserializationVisitor.from_dict(result2)
        assert reconstructed2.group_by == 'sector'
    
    def test_empty_constant(self):
        """Test Constant with 0.0 value."""
        expr = Constant(0.0)
        serializer = SerializationVisitor()
        result = expr.accept(serializer)
        
        assert result == {'type': 'Constant', 'value': 0.0}
        
        # Round-trip
        reconstructed = DeserializationVisitor.from_dict(result)
        assert reconstructed.value == 0.0
    
    def test_comparison_with_expression_right_hand_side(self):
        """Test comparison where right side is an Expression."""
        expr = Equals(Field('x'), Field('y'))
        
        # Serialize
        serializer = SerializationVisitor()
        data = expr.accept(serializer)
        
        assert data['right_is_expr'] == True
        assert data['right'] == {'type': 'Field', 'name': 'y'}
        
        # Deserialize
        reconstructed = DeserializationVisitor.from_dict(data)
        
        assert isinstance(reconstructed.left, Field)
        assert isinstance(reconstructed.right, Field)
        assert reconstructed.left.name == 'x'
        assert reconstructed.right.name == 'y'
    
    def test_unknown_type_raises_error(self):
        """Test that unknown type raises ValueError."""
        data = {'type': 'UnknownOperator'}
        
        with pytest.raises(ValueError, match="Unknown expression type"):
            DeserializationVisitor.from_dict(data)
    
    def test_deeply_nested_expression(self):
        """Test serialization of deeply nested expression."""
        # Not(And(Or(Field('a'), Field('b')), Field('c')))
        expr = Not(And(Or(Field('a'), Field('b')), Field('c')))
        
        # Serialize
        data = expr.to_dict()
        
        assert data['type'] == 'Not'
        assert data['child']['type'] == 'And'
        assert data['child']['left']['type'] == 'Or'
        
        # Deserialize
        reconstructed = Expression.from_dict(data)
        
        assert isinstance(reconstructed, Not)
        assert isinstance(reconstructed.child, And)
        assert isinstance(reconstructed.child.left, Or)
        
        # Dependencies
        deps = reconstructed.get_field_dependencies()
        assert sorted(deps) == ['a', 'b', 'c']

