"""Property accessor classes for AlphaCanvas facade."""

from alpha_canvas.core.expression import Field


class DataAccessor:
    """Accessor for rc.data that returns Field Expressions.
    
    This accessor enables Expression-based data access:
        rc.data['field_name'] → Field('field_name')
        rc.data['size'] == 'small' → Equals(Field('size'), 'small')
    
    The Field Expressions remain lazy until explicitly evaluated,
    ensuring universe masking is consistently applied through the Visitor.
    
    Note:
        Only item access (rc.data['field']) is supported.
        Attribute access (rc.data.field) is NOT supported.
    
    Example:
        >>> accessor = DataAccessor()
        >>> size_field = accessor['size']  # Returns Field('size')
        >>> mask = size_field == 'small'   # Returns Equals Expression
        >>> result = rc.evaluate(mask)     # Evaluates with universe masking
    """
    
    def __getitem__(self, field_name: str) -> Field:
        """Return Field Expression for the given field name.
        
        Args:
            field_name: Name of the field to access
            
        Returns:
            Field Expression wrapping the field name
            
        Raises:
            TypeError: If field_name is not a string
            
        Example:
            >>> accessor = DataAccessor()
            >>> field = accessor['size']
            >>> isinstance(field, Field)
            True
            >>> field.name
            'size'
        """
        if not isinstance(field_name, str):
            raise TypeError(
                f"Field name must be string, got {type(field_name).__name__}"
            )
        
        return Field(field_name)
    
    def __getattr__(self, name: str):
        """Prevent attribute access - only item access is allowed.
        
        This ensures a single, consistent access pattern.
        
        Args:
            name: Attribute name being accessed
            
        Raises:
            AttributeError: Always raised to prevent attribute access
            
        Example:
            >>> accessor = DataAccessor()
            >>> accessor.size  # Raises AttributeError
            Traceback (most recent call last):
                ...
            AttributeError: DataAccessor does not support attribute access. 
                Use rc.data['size'] instead of rc.data.size
        """
        raise AttributeError(
            f"DataAccessor does not support attribute access. "
            f"Use rc.data['{name}'] instead of rc.data.{name}"
        )

