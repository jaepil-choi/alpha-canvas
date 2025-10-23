"""
Tests for Expression classes.

These tests follow TDD methodology - they define expected behavior before implementation.
"""

import pytest
from alpha_canvas.core.expression import Expression, Field


class TestExpression:
    """Test suite for Expression base class."""
    
    def test_expression_is_abstract(self):
        """Test that Expression cannot be instantiated directly."""
        with pytest.raises(TypeError):
            # Should raise TypeError because Expression is abstract
            Expression()


class TestField:
    """Test suite for Field leaf expression."""
    
    def test_field_creation(self):
        """Test creating Field leaf expression."""
        field = Field('returns')
        assert field.name == 'returns'
    
    def test_field_with_different_names(self):
        """Test Field can be created with various names."""
        field1 = Field('market_cap')
        field2 = Field('adj_close')
        field3 = Field('volume')
        
        assert field1.name == 'market_cap'
        assert field2.name == 'adj_close'
        assert field3.name == 'volume'
    
    def test_field_is_expression(self):
        """Test that Field is an instance of Expression."""
        field = Field('returns')
        assert isinstance(field, Expression)
    
    def test_field_has_accept_method(self):
        """Test that Field has accept() method."""
        field = Field('returns')
        assert hasattr(field, 'accept')
        assert callable(field.accept)
    
    def test_field_accept_calls_visitor(self):
        """Test Field.accept() calls visitor.visit_field()."""
        field = Field('returns')
        
        # Mock visitor
        class MockVisitor:
            def __init__(self):
                self.visited_field = None
            
            def visit_field(self, node):
                self.visited_field = node
                return "mock_result"
        
        visitor = MockVisitor()
        result = field.accept(visitor)
        
        assert visitor.visited_field is field
        assert result == "mock_result"
    
    def test_field_equality(self):
        """Test Field equality based on name."""
        field1 = Field('returns')
        field2 = Field('returns')
        field3 = Field('market_cap')
        
        assert field1 == field2  # Same name
        assert field1 != field3  # Different name
    
    def test_field_repr(self):
        """Test Field has useful string representation."""
        field = Field('returns')
        repr_str = repr(field)
        assert 'Field' in repr_str
        assert 'returns' in repr_str

