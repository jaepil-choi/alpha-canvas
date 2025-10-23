"""Weight scaling strategy implementations."""
import numpy as np
import xarray as xr
from alpha_canvas.portfolio.base import WeightScaler


class GrossNetScaler(WeightScaler):
    """Unified weight scaler based on gross and net exposure targets.
    
    Uses the unified framework:
        L_target = (G + N) / 2
        S_target = (G - N) / 2
    
    Where:
        G = target_gross_exposure = sum(abs(weights))
        N = target_net_exposure = sum(weights)
        L = sum of positive weights (long)
        S = sum of negative weights (short, negative value)
    
    Implementation:
        - Fully vectorized (NO Python loops)
        - Always meets gross target via final scaling step
        - Net target achievable only for mixed signals
        - One-sided signals: Net = ±Gross (unavoidable)
        - Performance: 7-40ms for typical datasets
    
    Args:
        target_gross: Target gross exposure (default: 2.0 for 200% gross)
        target_net: Target net exposure (default: 0.0 for dollar-neutral)
    
    Examples:
        >>> # Dollar neutral: L=1.0, S=-1.0
        >>> scaler = GrossNetScaler(target_gross=2.0, target_net=0.0)
        >>> 
        >>> # Net long 10%: L=1.1, S=-0.9
        >>> scaler = GrossNetScaler(target_gross=2.0, target_net=0.2)
        >>> 
        >>> # Crypto futures: L=0.5, S=-0.5
        >>> scaler = GrossNetScaler(target_gross=1.0, target_net=0.0)
        >>> 
        >>> # Long-only-like: Use large net bias
        >>> scaler = GrossNetScaler(target_gross=1.0, target_net=1.0)
        >>> # With all-positive signals: L=1.0, S=0.0
    """
    
    def __init__(self, target_gross: float = 2.0, target_net: float = 0.0):
        self.target_gross = target_gross
        self.target_net = target_net
        
        # Validate constraints
        if target_gross < 0:
            raise ValueError("target_gross must be non-negative")
        if abs(target_net) > target_gross:
            raise ValueError(
                "Absolute net exposure cannot exceed gross exposure"
            )
        
        # Calculate target long and short books
        self.L_target = (target_gross + target_net) / 2.0
        self.S_target = (target_net - target_gross) / 2.0  # Negative value
    
    def scale(self, signal: xr.DataArray) -> xr.DataArray:
        """Scale signal using fully vectorized gross/net exposure constraints.
        
        Algorithm (validated in Experiment 18):
            1. Separate positive/negative signals
            2. Normalize separately (handle 0/0 → NaN → 0)
            3. Apply L_target and S_target
            4. Calculate actual gross per row
            5. Scale to meet target gross (always achievable)
            6. Convert computational NaN to 0
            7. Apply universe mask (preserve signal NaN)
        
        Key Innovation: NO ITERATION - pure vectorized operations.
        """
        self._validate_signal(signal)
        
        # Step 1: Separate positive/negative (vectorized)
        s_pos = signal.where(signal > 0, 0.0)
        s_neg = signal.where(signal < 0, 0.0)
        
        # Step 2: Sum along asset dimension (vectorized)
        sum_pos = s_pos.sum(dim='asset', skipna=True)  # Shape: (time,)
        sum_neg = s_neg.sum(dim='asset', skipna=True)  # Shape: (time,)
        
        # Step 3: Normalize (vectorized, handles 0/0 → nan → 0)
        norm_pos = (s_pos / sum_pos).fillna(0.0)
        norm_neg_abs = (np.abs(s_neg) / np.abs(sum_neg)).fillna(0.0)
        
        # Step 4: Apply L/S targets (vectorized)
        weights_long = norm_pos * self.L_target
        weights_short_mag = norm_neg_abs * np.abs(self.S_target)
        
        # Step 5: Combine (subtract to make short side negative)
        weights = weights_long - weights_short_mag
        
        # Step 6: Calculate actual gross per row (vectorized)
        actual_gross = np.abs(weights).sum(dim='asset', skipna=True)  # Shape: (time,)
        
        # Step 7: Scale to meet target gross (vectorized)
        # Use xr.where to avoid inf from 0/0
        scale_factor = xr.where(actual_gross > 0, self.target_gross / actual_gross, 1.0)
        final_weights = weights * scale_factor
        
        # Step 8: Convert computational NaN to 0 (BEFORE universe mask)
        final_weights = final_weights.fillna(0.0)
        
        # Step 9: Apply universe mask (preserves NaN where signal was NaN)
        final_weights = final_weights.where(~signal.isnull())
        
        return final_weights


class DollarNeutralScaler(GrossNetScaler):
    """Dollar neutral: sum(long) = 1.0, sum(short) = -1.0.
    
    Convenience wrapper for GrossNetScaler(2.0, 0.0).
    This is the most common scaler for market-neutral strategies.
    
    Equivalent to:
        >>> GrossNetScaler(target_gross=2.0, target_net=0.0)
    """
    def __init__(self):
        super().__init__(target_gross=2.0, target_net=0.0)

