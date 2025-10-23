"""Abstract base class for weight scaling strategies."""
from abc import ABC, abstractmethod
import xarray as xr


class WeightScaler(ABC):
    """Abstract base class for portfolio weight scaling strategies.
    
    Strategy Pattern: Defines interface for weight scaling algorithms.
    Each strategy converts arbitrary signal values to portfolio weights
    by applying specific constraints.
    
    Design Principle:
        - Stateless: Scalers don't store state between calls
        - Cross-sectional: Each time period processed independently
        - NaN-aware: Respects universe masking (preserves NaN positions)
        - Replaceable: Easy to swap strategies without code changes
    
    Example Usage:
        >>> scaler = DollarNeutralScaler()  # Choose strategy
        >>> weights = scaler.scale(signal)  # Apply strategy
        
    To implement a new strategy:
        1. Inherit from WeightScaler
        2. Implement scale(signal) method
        3. Call _validate_signal(signal) at start
        4. Return (T, N) DataArray with weights
    """
    
    @abstractmethod
    def scale(self, signal: xr.DataArray) -> xr.DataArray:
        """Scale signal to portfolio weights.
        
        Args:
            signal: (T, N) DataArray with arbitrary signal values
            
        Returns:
            (T, N) DataArray with portfolio weights
            
        Note:
            - Must preserve NaN positions (universe masking)
            - Must preserve input shape
            - Should process each time period independently
        """
        pass
    
    def _validate_signal(self, signal: xr.DataArray):
        """Validate signal dimensions and values.
        
        Raises:
            ValueError: If dimensions incorrect or all values NaN
        """
        if signal.dims != ('time', 'asset'):
            raise ValueError(
                f"Signal must have dims ('time', 'asset'), got {signal.dims}"
            )
        
        # Check if any timestep has non-NaN values
        non_nan_counts = (~signal.isnull()).sum(dim='asset')
        if (non_nan_counts == 0).all():
            raise ValueError(
                "All signal values are NaN across all time periods"
            )

