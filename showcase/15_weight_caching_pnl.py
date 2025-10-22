"""
Showcase 15: Weight Caching for PnL Analysis

Demonstrates:
1. Dual-cache architecture (signals + weights)
2. Step-by-step weight tracking
3. Efficient scaler replacement
4. Preparing for future PnL tracing
"""
import numpy as np
import pandas as pd
import xarray as xr
from alpha_canvas import AlphaCanvas
from alpha_canvas.core.expression import Field
from alpha_canvas.ops.timeseries import TsMean
from alpha_canvas.ops.crosssection import Rank
from alpha_canvas.portfolio.strategies import DollarNeutralScaler, GrossNetScaler

print("="*70)
print("SHOWCASE 15: Weight Caching for PnL Analysis")
print("="*70)

# Setup
dates = pd.date_range('2024-01-01', periods=20, freq='D')
assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
rc = AlphaCanvas(time_index=dates, asset_index=assets)

# Add mock returns data
np.random.seed(42)
returns_data = xr.DataArray(
    np.random.randn(20, 5) * 0.02,  # 2% daily volatility
    dims=['time', 'asset'],
    coords={'time': dates, 'asset': assets}
)
rc.add_data('returns', returns_data)

print("\nStep 1: Define multi-step alpha signal")
print("-" * 70)
# Multi-step expression: smooth returns -> rank
alpha_expr = Rank(TsMean(Field('returns'), window=5))
print("Expression: Rank(TsMean(Field('returns'), window=5))")
print("  Step 0: Field('returns') - raw returns")
print("  Step 1: TsMean(..., 5) - 5-day moving average")
print("  Step 2: Rank(...) - cross-sectional rank")

print("\nStep 2: Evaluate with DollarNeutralScaler")
print("-" * 70)
scaler1 = DollarNeutralScaler()
result = rc.evaluate(alpha_expr, scaler=scaler1)

print(f"\nCached steps: {len(rc._evaluator._signal_cache)}")
for step in range(len(rc._evaluator._signal_cache)):
    name, signal = rc._evaluator.get_cached_signal(step)
    _, weights = rc._evaluator.get_cached_weights(step)
    
    print(f"\nStep {step}: {name}")
    print(f"  Signal - mean: {signal.mean().values:.4f}, std: {signal.std().values:.4f}")
    
    if weights is not None:
        long_sum = weights.where(weights > 0, 0.0).sum(dim='asset').mean().values
        short_sum = weights.where(weights < 0, 0.0).sum(dim='asset').mean().values
        gross = abs(weights).sum(dim='asset').mean().values
        net = weights.sum(dim='asset').mean().values
        
        print(f"  Weights - Long: {long_sum:.4f}, Short: {short_sum:.4f}")
        print(f"           Gross: {gross:.4f}, Net: {net:.4f}")

print("\nStep 3: Swap to GrossNetScaler (efficient!)")
print("-" * 70)
scaler2 = GrossNetScaler(target_gross=2.0, target_net=0.3)
result = rc.evaluate(alpha_expr, scaler=scaler2)

print("\nNew weights with net-long bias:")
for step in range(len(rc._evaluator._weight_cache)):
    name, weights = rc._evaluator.get_cached_weights(step)
    
    if weights is not None:
        long_sum = weights.where(weights > 0, 0.0).sum(dim='asset').mean().values
        short_sum = weights.where(weights < 0, 0.0).sum(dim='asset').mean().values
        gross = abs(weights).sum(dim='asset').mean().values
        net = weights.sum(dim='asset').mean().values
        
        print(f"\nStep {step}: {name}")
        print(f"  Weights - Long: {long_sum:.4f}, Short: {short_sum:.4f}")
        print(f"           Gross: {gross:.4f}, Net: {net:.4f}")

print("\n" + "="*70)
print("KEY INSIGHT: Signal cache unchanged, only weights recalculated!")
print("This enables efficient strategy comparison for research.")
print("="*70)

print("\nStep 4: Accessing specific step weights")
print("-" * 70)
# Access specific step weights using the convenience method
weights_step_0 = rc.get_weights(0)  # Field('returns')
weights_step_1 = rc.get_weights(1)  # TsMean result
weights_step_2 = rc.get_weights(2)  # Rank result

print(f"\nStep 0 (Field) weights shape: {weights_step_0.shape}")
print(f"Step 1 (TsMean) weights shape: {weights_step_1.shape}")
print(f"Step 2 (Rank) weights shape: {weights_step_2.shape}")

# Show sample weights for first date at final step
print(f"\nSample weights at final step (Rank) for {dates[10]}:")
for asset in assets:
    weight = weights_step_2.sel(time=dates[10], asset=asset).values
    print(f"  {asset}: {weight:+.4f}")

print("\n" + "="*70)
print("FUTURE USE CASE: PnL Tracing")
print("="*70)
print("""
With dual-cache architecture, we can now implement:

1. **Step-by-step PnL decomposition:**
   - Which step (operator) contributes most to final PnL?
   - How do intermediate signals perform vs. final signal?

2. **Weight scaler comparison:**
   - Compare PnL of DollarNeutral vs. Net-Long strategies
   - Analyze how different gross/net targets affect returns
   - All without re-evaluating signals (efficient!)

3. **Attribution analysis:**
   - Decompose PnL into signal generation vs. weight scaling
   - Identify where value is created (or lost) in the pipeline

Example future API:
  >>> pnl_tracer = PnLTracer(rc._evaluator, returns_data)
  >>> step_pnls = pnl_tracer.decompose_by_step()
  >>> print(f"TsMean PnL: {step_pnls[1]:.2%}")
  >>> print(f"Rank PnL: {step_pnls[2]:.2%}")
""")

print("="*70)
print("END OF SHOWCASE 15")
print("="*70)

