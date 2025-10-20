"""
Showcase 07: ts_any() Operator - Surge Event Detection

This showcase demonstrates the ts_any() operator for detecting events
within a rolling time window. This is particularly useful for:

1. Surge detection (large price movements)
2. High volume events  
3. Stop-loss triggers
4. Threshold crossings

The operator checks if ANY value in a rolling window is True.
"""

import pandas as pd
import xarray as xr
import numpy as np
from alpha_canvas import AlphaCanvas
from alpha_canvas.core.expression import Field
from alpha_canvas.ops.timeseries import TsAny

print("="*70)
print("SHOWCASE 07: ts_any() - Surge Event Detection")
print("="*70)

# ============================================================================
# Part 1: Setup AlphaCanvas with Mock Data
# ============================================================================
print("\n[Part 1] Setting up AlphaCanvas...")

# Initialize
rc = AlphaCanvas(
    config_dir='config',
    start_date='2024-01-01',
    end_date='2024-01-15'
)

print(f"  Config loaded: {rc._config.list_fields()}")

# Load price data
rc.add_data('close', Field('adj_close'))

print(f"  Data loaded: close")
print(f"  Data shape: {rc.db['close'].shape}")
print(f"  Time range: {rc.db['close'].time.values[0]} to {rc.db['close'].time.values[-1]}")
print(f"  Assets: {list(rc.db['close'].asset.values)}")

# ============================================================================
# Part 2: Calculate Returns
# ============================================================================
print("\n[Part 2] Calculating returns...")

# Get price data
prices = rc.db['close']

# Calculate daily returns: (price_t / price_{t-1}) - 1
# Note: This will have NaN for first row
returns = (prices / prices.shift(time=1)) - 1

# Add to dataset
rc.add_data('returns', returns)

print("  Returns calculated")
print("\n  Sample returns (first 5 days, first 2 assets):")
print(returns.sel(asset=['AAPL', 'NVDA']).isel(time=slice(0, 5)).to_pandas())

# Show return distribution
print(f"\n  Return statistics:")
print(f"    Mean: {float(returns.mean()):.4f}")
print(f"    Std: {float(returns.std()):.4f}")
print(f"    Min: {float(returns.min()):.4f}")
print(f"    Max: {float(returns.max()):.4f}")

# ============================================================================
# Part 3: Detect Surge Events (returns > 3%)
# ============================================================================
print("\n[Part 3] Detecting surge events (>3% daily returns)...")

# Create boolean mask for surge events
surge_mask = rc.db['returns'] > 0.03

# Count surge events
surge_count = int(surge_mask.sum())
total_observations = int((~np.isnan(rc.db['returns'])).sum())

print(f"  Surge events found: {surge_count} / {total_observations} observations")
print(f"  Surge rate: {surge_count/total_observations*100:.2f}%")

# Show which assets had surges
print("\n  Assets with surge events:")
for asset in surge_mask.asset.values:
    asset_surges = int(surge_mask.sel(asset=asset).sum())
    if asset_surges > 0:
        # Find dates
        surge_dates = surge_mask.sel(asset=asset).time.values[surge_mask.sel(asset=asset).values]
        print(f"    {asset}: {asset_surges} surges on {[str(d)[:10] for d in surge_dates]}")

# ============================================================================
# Part 4: Apply ts_any() - Had Surge in Last 5 Days
# ============================================================================
print("\n[Part 4] Applying ts_any() operator...")

# First, add the surge mask to the dataset
# (Boolean mask where returns > 3%)
rc.add_data('surge_mask', surge_mask)

# Create expression: ts_any(Field('surge_mask'), window=5)
# This asks: "Did this asset have a >3% return in the last 5 days?"
surge_event_expr = TsAny(
    child=Field('surge_mask'),
    window=5
)

# Evaluate through AlphaCanvas
rc.add_data('surge_event', surge_event_expr)

print(f"  ts_any() expression evaluated")
print(f"  Result shape: {rc.db['surge_event'].shape}")
print(f"  Result dtype: {rc.db['surge_event'].dtype}")

# ============================================================================
# Part 5: Analyze Window Persistence
# ============================================================================
print("\n[Part 5] Analyzing event persistence in rolling window...")

# Look at AAPL specifically
aapl_returns = rc.db['returns'].sel(asset='AAPL')
aapl_surge = surge_mask.sel(asset='AAPL')
aapl_had_surge = rc.db['surge_event'].sel(asset='AAPL')

print("\n  AAPL Event Timeline:")
print("  " + "-"*60)
print(f"  {'Date':<12} | {'Return':<8} | {'Surge':<6} | {'Had Surge (5d)'}")
print("  " + "-"*60)

for i, date in enumerate(aapl_returns.time.values):
    ret_val = float(aapl_returns.isel(time=i).values)
    surge_val = bool(aapl_surge.isel(time=i).values) if not np.isnan(ret_val) else False
    had_surge_val = aapl_had_surge.isel(time=i).values
    
    # Format
    date_str = str(date)[:10]
    ret_str = f"{ret_val:>+6.2%}" if not np.isnan(ret_val) else "  NaN  "
    surge_str = "[YES]" if surge_val else " no  "
    had_str = "[YES]" if had_surge_val == True else (" nan " if np.isnan(had_surge_val) else " no  ")
    
    print(f"  {date_str} | {ret_str} | {surge_str} | {had_str}")

print("  " + "-"*60)

# ============================================================================
# Part 6: Cross-Sectional Analysis
# ============================================================================
print("\n[Part 6] Cross-sectional analysis - Assets with active events...")

# Count assets with active surge events by date
events_by_date = rc.db['surge_event'].sum(dim='asset')

print("\n  Number of assets with active surge events by date:")
for i, date in enumerate(events_by_date.time.values):
    count = float(events_by_date.isel(time=i).values)
    if not np.isnan(count):
        date_str = str(date)[:10]
        print(f"    {date_str}: {int(count)} assets")

# ============================================================================
# Part 7: Compare with Direct Rolling Sum
# ============================================================================
print("\n[Part 7] Validating against direct implementation...")

# Direct calculation for AAPL (to validate)
# Issue: NaN > 0 returns False, not NaN
direct_calc_naive = (aapl_surge.rolling(time=5, min_periods=5).sum() > 0)

# Our implementation preserves NaN where window is incomplete
# Let's replicate that behavior for comparison
count = aapl_surge.rolling(time=5, min_periods=5).sum()
direct_calc = count > 0
# Preserve NaN where count is NaN
direct_calc = direct_calc.where(~np.isnan(count), np.nan)

# Compare
matches = (direct_calc.values == aapl_had_surge.values) | \
          (np.isnan(direct_calc.values) & np.isnan(aapl_had_surge.values))

if matches.all():
    print("  [OK] ts_any() matches direct rolling().sum() > 0 calculation")
else:
    print("  [WARNING] Mismatch detected!")
    print(f"    Differences: {(~matches).sum()} positions")
    
    # Show the differences
    print("\n  Debugging mismatches:")
    print(f"  {'Date':<12} | {'Count':<6} | {'Direct':<8} | {'TsAny':<8}")
    print("  " + "-"*50)
    for i in range(len(aapl_had_surge)):
        if not matches[i]:
            date_str = str(aapl_had_surge.time.values[i])[:10]
            cnt = float(count.isel(time=i).values) if i < len(count) else np.nan
            direct_val = direct_calc.isel(time=i).values
            tsany_val = aapl_had_surge.isel(time=i).values
            print(f"  {date_str} | {cnt:>6} | {str(direct_val):<8} | {str(tsany_val):<8}")

# ============================================================================
# Part 8: Use Case - Filtering High-Momentum Assets
# ============================================================================
print("\n[Part 8] Use case: Filter for high-momentum assets...")

# Get latest values (last valid date)
latest_date = rc.db['surge_event'].time.values[-1]
latest_events = rc.db['surge_event'].sel(time=latest_date)

# Find assets with active surge events
active_surge_assets = []
for asset in latest_events.asset.values:
    if latest_events.sel(asset=asset).item() == True:
        active_surge_assets.append(asset)

print(f"\n  As of {str(latest_date)[:10]}:")
print(f"    Assets with surge in last 5 days: {active_surge_assets}")

if active_surge_assets:
    print("\n  Recent returns for these assets:")
    for asset in active_surge_assets:
        recent_returns = rc.db['returns'].sel(asset=asset).isel(time=slice(-5, None))
        print(f"\n    {asset} (last 5 days):")
        for i, date in enumerate(recent_returns.time.values):
            ret = float(recent_returns.isel(time=i).values)
            surge_marker = " <- SURGE!" if ret > 0.03 else ""
            print(f"      {str(date)[:10]}: {ret:>+6.2%}{surge_marker}")

# ============================================================================
# Part 9: Expression Tree Inspection
# ============================================================================
print("\n[Part 9] Expression tree inspection...")

# The expression was evaluated during add_data()
print("""
  The ts_any() expression was evaluated and cached during add_data().
  The visitor traversed the expression tree:
  
    Step 1: Field('surge_mask') → Load boolean mask  
    Step 2: TsAny(window=5) → Apply rolling any operation
    
  Result is stored in rc.db['surge_event'].
""")

# ============================================================================
# Part 10: Performance Note
# ============================================================================
print("\n[Part 10] Performance characteristics...")

print("""
  ts_any() uses rolling().sum() > 0 pattern:
  
  1. Counts True values in window (sum)
  2. Checks if count > 0 (any)
  3. Preserves NaN for incomplete windows
  
  This approach is:
  - 3.92x faster than reduce(np.any) 
  - Semantically clear ("count > 0 means any")
  - Compatible with xarray rolling window mechanics
  
  Validated in Experiment 11.
""")

print("\n" + "="*70)
print("SHOWCASE COMPLETE")
print("="*70)

print("\nKey Takeaways:")
print("1. ts_any() detects if ANY value in rolling window is True")
print("2. Useful for event detection (surges, volume spikes, etc.)")
print("3. Events persist for entire window duration (5 days here)")
print("4. Cross-sectionally independent (each asset tracked separately)")
print("5. NaN handling: incomplete windows return NaN")
print("6. Integrates seamlessly with AlphaCanvas expression system")

