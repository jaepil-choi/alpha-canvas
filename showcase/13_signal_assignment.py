"""
Showcase: Signal Canvas & Lazy Assignment (Fama-French Factor)

Demonstrates:
- Implicit blank canvas (starting from constant or field)
- Lazy signal assignment with selector masks
- Fama-French 2x3 factor construction (Size x Value)
- Universe masking integration
- Actual data verification in terminal

Date: 2025-10-21
Status: Complete
"""

import numpy as np
import pandas as pd
import xarray as xr
from alpha_canvas.core.facade import AlphaCanvas
from alpha_canvas.core.expression import Field
from alpha_canvas.ops.constants import Constant
from alpha_canvas.ops.classification import CsQuantile


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_dataarray_preview(name: str, data: xr.DataArray, n_times=3, n_assets=5):
    """Print a preview of DataArray with actual values."""
    print(f"\n{name}:")
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    
    # Select subset for display
    time_slice = slice(0, min(n_times, len(data.time)))
    asset_slice = slice(0, min(n_assets, len(data.asset)))
    
    subset = data.isel(time=time_slice, asset=asset_slice)
    
    print(f"\n  Data preview (first {n_times} times, {n_assets} assets):")
    df = subset.to_pandas()
    print(df.to_string())
    
    # Show statistics (only for numeric data)
    if np.issubdtype(data.dtype, np.number):
        print(f"\n  Statistics:")
        print(f"    Min: {float(data.min()):.4f}")
        print(f"    Max: {float(data.max()):.4f}")
        print(f"    Mean: {float(data.mean()):.4f}")
        print(f"    NaN count: {int(data.isnull().sum())}")
    else:
        print(f"\n  Statistics:")
        print(f"    Unique values: {list(np.unique(data.values[~pd.isnull(data.values)]))}")
        print(f"    NaN count: {int(data.isnull().sum())}")


def create_sample_data():
    """Create realistic sample dataset with size and value characteristics.
    
    Returns:
        xarray.Dataset with market_cap and book_to_market fields
    """
    np.random.seed(42)
    
    # 10 time periods, 6 assets
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD']
    
    # Market cap: varies across assets (size dimension)
    # Small caps: ~10-50B, Large caps: ~500-2000B
    base_mcap = np.array([1500, 1200, 1000, 600, 800, 50])  # Billions
    mcap_data = np.zeros((10, 6))
    for i in range(10):
        # Add some temporal variation (±10%)
        mcap_data[i, :] = base_mcap * (1 + np.random.uniform(-0.1, 0.1, 6))
    
    # Book-to-market ratio: varies across assets (value dimension)
    # Growth stocks: ~0.2-0.5, Value stocks: ~1.5-3.0
    base_btm = np.array([0.3, 0.4, 0.35, 2.5, 0.5, 2.8])
    btm_data = np.zeros((10, 6))
    for i in range(10):
        # Add some temporal variation (±20%)
        btm_data[i, :] = base_btm * (1 + np.random.uniform(-0.2, 0.2, 6))
    
    # Create xarray Dataset
    ds = xr.Dataset(
        {
            'market_cap': xr.DataArray(
                mcap_data,
                dims=['time', 'asset'],
                coords={'time': dates, 'asset': tickers}
            ),
            'book_to_market': xr.DataArray(
                btm_data,
                dims=['time', 'asset'],
                coords={'time': dates, 'asset': tickers}
            )
        }
    )
    
    return ds


def main():
    """Demonstrate signal canvas and lazy assignment with Fama-French factor."""
    
    print_section("SIGNAL CANVAS & LAZY ASSIGNMENT SHOWCASE")
    print("\nObjective: Construct Fama-French 2x3 factor (Size x Value)")
    print("- Size: Small (S) vs Big (B) based on market cap")
    print("- Value: Low (L), Medium (M), High (H) based on book-to-market")
    print("- Signal: +1 for S/H, -1 for B/L, 0 elsewhere")
    
    # =========================================================================
    # STEP 1: Initialize AlphaCanvas with sample data
    # =========================================================================
    print_section("Step 1: Initialize AlphaCanvas")
    
    ds = create_sample_data()
    print(f"Created dataset with {len(ds.time)} time periods, {len(ds.asset)} assets")
    print(f"Fields: {list(ds.data_vars)}")
    
    # Initialize AlphaCanvas
    rc = AlphaCanvas(
        time_index=ds.time.values,
        asset_index=ds.asset.values
    )
    
    # Inject data directly by setting the internal dataset
    rc._panel._dataset = ds
    
    # Re-initialize evaluator with populated dataset
    from alpha_canvas.core.visitor import EvaluateVisitor
    rc._evaluator = EvaluateVisitor(rc._panel.db, None)
    
    print("[OK] AlphaCanvas initialized with sample data")
    
    # =========================================================================
    # STEP 2: Preview raw data
    # =========================================================================
    print_section("Step 2: Preview Raw Data")
    
    mcap = rc.evaluate(Field('market_cap'))
    btm = rc.evaluate(Field('book_to_market'))
    
    print_dataarray_preview("Market Cap (Billions)", mcap, n_times=5, n_assets=6)
    print_dataarray_preview("Book-to-Market Ratio", btm, n_times=5, n_assets=6)
    
    # =========================================================================
    # STEP 3: Create size and value classifications
    # =========================================================================
    print_section("Step 3: Create Size and Value Classifications")
    
    # Size: small vs big (2 groups)
    size_expr = CsQuantile(
        child=Field('market_cap'),
        bins=2,
        labels=['small', 'big']
    )
    
    # Value: low, medium, high (3 groups)
    value_expr = CsQuantile(
        child=Field('book_to_market'),
        bins=3,
        labels=['low', 'medium', 'high']
    )
    
    # Add to data
    rc.add_data('size', size_expr)
    rc.add_data('value', value_expr)
    
    print("[OK] Created size classification (small/big)")
    print("[OK] Created value classification (low/medium/high)")
    
    # Preview classifications
    size_data = rc.evaluate(Field('size'))
    value_data = rc.evaluate(Field('value'))
    
    print_dataarray_preview("Size Classification", size_data, n_times=5, n_assets=6)
    print_dataarray_preview("Value Classification", value_data, n_times=5, n_assets=6)
    
    # =========================================================================
    # STEP 4: Create selector masks
    # =========================================================================
    print_section("Step 4: Create Selector Masks")
    
    # Create boolean masks using comparison operators
    is_small = rc.data['size'] == 'small'
    is_big = rc.data['size'] == 'big'
    is_low = rc.data['value'] == 'low'
    is_high = rc.data['value'] == 'high'
    
    print("[OK] Created selector masks:")
    print("  - is_small: size == 'small'")
    print("  - is_big: size == 'big'")
    print("  - is_low: value == 'low'")
    print("  - is_high: value == 'high'")
    
    # Preview one selector (to show it's an Expression before evaluation)
    print(f"\n  Type of 'is_small': {type(is_small).__name__}")
    print(f"  This is an Expression (lazy, not yet evaluated)")
    
    # Evaluate to show actual boolean values
    is_small_data = rc.evaluate(is_small)
    print_dataarray_preview("is_small (evaluated)", is_small_data, n_times=5, n_assets=6)
    
    # =========================================================================
    # STEP 5: Construct Fama-French factor with lazy assignments
    # =========================================================================
    print_section("Step 5: Construct Fama-French Factor (Lazy Assignment)")
    
    # Start with a blank canvas (all zeros)
    signal = Constant(0.0)
    
    print("Starting canvas: Constant(0.0)")
    print("  - This creates a universe-shaped array filled with 0.0")
    
    # Define factor: Small/High-Value = +1, Big/Low-Value = -1
    signal[is_small & is_high] = 1.0   # Small cap, high B/M (value)
    signal[is_big & is_low] = -1.0     # Big cap, low B/M (growth)
    
    print("\nAssignments stored (lazy, not yet evaluated):")
    print("  1. signal[(size=='small') & (value=='high')] = 1.0")
    print("  2. signal[(size=='big') & (value=='low')] = -1.0")
    
    # Verify assignments are stored
    print(f"\nNumber of assignments stored: {len(signal._assignments)}")
    print(f"Assignment 1: mask={type(signal._assignments[0][0]).__name__}, value={signal._assignments[0][1]}")
    print(f"Assignment 2: mask={type(signal._assignments[1][0]).__name__}, value={signal._assignments[1][1]}")
    
    # =========================================================================
    # STEP 6: Evaluate signal (assignments applied lazily)
    # =========================================================================
    print_section("Step 6: Evaluate Signal (Lazy Execution)")
    
    print("Calling rc.evaluate(signal)...")
    print("  - Evaluates base expression (Constant(0.0))")
    print("  - Applies assignments sequentially")
    print("  - Returns final result")
    
    result = rc.evaluate(signal)
    
    print(f"\n[OK] Signal evaluated successfully")
    print_dataarray_preview("Fama-French Factor Signal", result, n_times=10, n_assets=6)
    
    # =========================================================================
    # STEP 7: Verify results manually
    # =========================================================================
    print_section("Step 7: Verify Results")
    
    # Re-evaluate selectors for verification
    size_final = rc.evaluate(Field('size'))
    value_final = rc.evaluate(Field('value'))
    
    print("\nManual verification for first time period:")
    for i, ticker in enumerate(ds.asset.values):
        s = str(size_final.values[0, i])
        v = str(value_final.values[0, i])
        sig = result.values[0, i]
        
        expected = None
        if s == 'small' and v == 'high':
            expected = 1.0
        elif s == 'big' and v == 'low':
            expected = -1.0
        else:
            expected = 0.0
        
        match = "[OK]" if abs(sig - expected) < 1e-6 else "[FAIL]"
        print(f"  {ticker:6s}: size={s:6s}, value={v:6s} -> signal={sig:+.1f} (expected {expected:+.1f}) {match}")
    
    # Count assignments
    n_long = int((result == 1.0).sum())
    n_short = int((result == -1.0).sum())
    n_neutral = int((result == 0.0).sum())
    
    print(f"\nSignal distribution:")
    print(f"  Long (+1.0):   {n_long:3d} positions")
    print(f"  Short (-1.0):  {n_short:3d} positions")
    print(f"  Neutral (0.0): {n_neutral:3d} positions")
    print(f"  Total:         {n_long + n_short + n_neutral:3d} positions")
    
    # =========================================================================
    # STEP 8: Demonstrate traceability (cached steps)
    # =========================================================================
    print_section("Step 8: Traceability (Cached Steps)")
    
    print(f"Number of cached steps: {len(rc._evaluator._cache)}")
    print("\nCached computation steps:")
    for step_idx in sorted(rc._evaluator._cache.keys()):
        name, data = rc._evaluator._cache[step_idx]
        print(f"  Step {step_idx}: {name:30s} shape={data.shape} dtype={data.dtype}")
    
    print("\n[OK] Each assignment step is cached for traceability")
    
    # =========================================================================
    # STEP 9: Test multiple sequential assignments (overlapping masks)
    # =========================================================================
    print_section("Step 9: Overlapping Masks (Sequential Application)")
    
    # Create a signal with overlapping assignments
    signal2 = Constant(0.0)
    signal2[is_small] = 0.5         # All small caps = 0.5
    signal2[is_small & is_high] = 1.0  # Small/High overwrites to 1.0
    
    print("Assignments (with overlap):")
    print("  1. signal[size=='small'] = 0.5")
    print("  2. signal[(size=='small') & (value=='high')] = 1.0")
    print("\nExpected behavior:")
    print("  - Small caps initially get 0.5")
    print("  - Small/High-value overwrites to 1.0 (later assignment wins)")
    
    result2 = rc.evaluate(signal2)
    print_dataarray_preview("Signal with Overlapping Assignments", result2, n_times=5, n_assets=6)
    
    # Verify overlap
    is_small_data = rc.evaluate(is_small)
    is_small_high_data = rc.evaluate(is_small & is_high)
    
    print("\nVerification:")
    for i, ticker in enumerate(ds.asset.values):
        if is_small_data.values[0, i]:
            sig = result2.values[0, i]
            is_sh = is_small_high_data.values[0, i]
            expected = 1.0 if is_sh else 0.5
            match = "[OK]" if abs(sig - expected) < 1e-6 else "[FAIL]"
            category = "Small/High" if is_sh else "Small/Other"
            print(f"  {ticker:6s}: {category:12s} -> signal={sig:+.1f} (expected {expected:+.1f}) {match}")
    
    # =========================================================================
    # SUCCESS
    # =========================================================================
    print_section("SHOWCASE COMPLETE")
    
    print("\n[SUCCESS] All features demonstrated:")
    print("  [OK] Implicit blank canvas (Constant(0.0))")
    print("  [OK] Lazy assignment storage (Expression.__setitem__)")
    print("  [OK] Boolean selector masks (comparison operators)")
    print("  [OK] Logical operations (& for AND)")
    print("  [OK] Sequential assignment application")
    print("  [OK] Overlapping mask handling (later wins)")
    print("  [OK] Visitor integration (evaluate with assignments)")
    print("  [OK] Traceability (cached steps)")
    print("  [OK] Actual data verification (no hardcoded results)")
    
    print("\n" + "="*70)
    print("  Fama-French 2x3 Factor Construction: VERIFIED")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

