"""
Constant Expression for creating constant-valued DataArrays.

This module provides the Constant Expression which creates a universe-shaped
DataArray filled with a constant value. It's useful as a blank canvas for
signal construction via assignments.
"""

from dataclasses import dataclass
import numpy as np
import xarray as xr
from alpha_canvas.core.expression import Expression


@dataclass(eq=False)
class Constant(Expression):
    """Expression that produces a constant-valued DataArray.
    
    Creates a universe-shaped (T, N) DataArray filled with the specified
    constant value. This serves as a "blank canvas" for signal construction
    where assignments can be applied via `signal[mask] = value`.
    
    The Constant operator is not truly a constant in the mathematical sense
    (it doesn't evaluate to a scalar). Instead, it broadcasts a scalar value
    to the full panel shape, respecting universe masking.
    
    Attributes:
        value: Scalar value to fill the array with
    
    Example:
        >>> # Create a blank canvas of zeros
        >>> signal = Constant(0.0)
        >>> 
        >>> # Apply assignments
        >>> signal[Field('size') == 'small'] = 1.0
        >>> signal[Field('size') == 'big'] = -1.0
        >>> 
        >>> # Evaluate to get final signal
        >>> result = visitor.evaluate(signal)
    
    Note:
        - The Constant is evaluated by the Visitor, which provides shape/coords
        - Universe masking is applied by the Visitor after evaluation
        - Common use case: `Constant(0.0)` for blank signal canvas
        - Alternative: `Constant(np.nan)` for NaN-filled canvas
    """
    value: float
    
    def accept(self, visitor):
        """Accept visitor and delegate to visit_constant().
        
        Args:
            visitor: Visitor instance with visit_constant() method
        
        Returns:
            xr.DataArray: Constant-valued array with shape (T, N)
        """
        return visitor.visit_constant(self)


# Note: The compute() method is not needed for Constant because the Visitor
# handles the creation of the constant array using the panel shape/coords.
# This is different from operators like TsMean which transform existing data.

