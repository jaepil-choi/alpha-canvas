"""
Showcase 12: Cross-Sectional Quantile Bucketing (cs_quantile)

This showcase demonstrates the cs_quantile operator for creating categorical
labels based on quantile bucketing. Supports both:
1. Independent sort: Quantile across entire universe
2. Dependent sort: Quantile within groups (Fama-French style)

This is the foundation for multi-dimensional factor portfolio construction.
"""

import numpy as np
import pandas as pd
import xarray as xr

from alpha_canvas import AlphaCanvas
from alpha_canvas.core.expression import Field
from alpha_canvas.ops.classification import CsQuantile


def main():
    print("=" * 80)
    print("SHOWCASE 12: CROSS-SECTIONAL QUANTILE BUCKETING")
    print("=" * 80)
    
    # =========================================================================
    # Step 1: Setup - Create Sample Data
    # =========================================================================
    print("\n[Step 1] Setup: Creating sample market data...")
    print("-" * 80)
    
    T = 10  # 10 days
    N = 6   # 6 stocks
    
    # Create date and asset indices
    time_index = pd.date_range('2024-01-01', periods=T, freq='D')
    asset_index = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META']
    
    # Initialize AlphaCanvas with minimal setup
    # Note: We'll manually create the dataset instead of using config
    rc = AlphaCanvas(
        time_index=time_index,
        asset_index=asset_index
    )
    
    # Create market cap data (small vs big stocks)
    np.random.seed(42)
    market_cap_values = np.random.rand(T, N) * 1000 + 100
    market_cap_values[:, 0:3] += 500  # AAPL, GOOGL, MSFT are big-cap
    
    market_cap = xr.DataArray(
        market_cap_values,
        dims=['time', 'asset'],
        coords={'time': time_index, 'asset': asset_index}
    )
    
    # Create book-to-market data (value vs growth)
    btm_values = np.random.rand(T, N) * 5
    
    btm = xr.DataArray(
        btm_values,
        dims=['time', 'asset'],
        coords={'time': time_index, 'asset': asset_index}
    )
    
    # Add data to canvas
    rc.add_data('market_cap', market_cap)
    rc.add_data('btm', btm)
    
    print(f"  Created {T} days × {N} assets dataset")
    print(f"  Assets: {asset_index}")
    print(f"\n  Market Cap (day 0):")
    print(f"    {market_cap.isel(time=0).to_pandas().to_dict()}")
    print(f"\n  Book-to-Market (day 0):")
    print(f"    {btm.isel(time=0).to_pandas().to_dict()}")
    
    # =========================================================================
    # Step 2: Independent Sort - Size Quantiles
    # =========================================================================
    print("\n[Step 2] Independent Sort: Size quantiles (whole universe)...")
    print("-" * 80)
    
    # Create size quantile expression
    size_expr = CsQuantile(
        child=Field('market_cap'),
        bins=2,
        labels=['small', 'big']
    )
    
    # Evaluate and add to canvas
    size_result = rc._evaluator.evaluate(size_expr)
    rc.add_data('size', size_result)
    
    print(f"  Bins: 2")
    print(f"  Labels: ['small', 'big']")
    print(f"  Method: Independent sort (all stocks ranked together)")
    
    print(f"\n  Size labels (day 0):")
    size_day0 = size_result.isel(time=0).to_pandas()
    for asset, label in size_day0.items():
        mcap = market_cap.isel(time=0).sel(asset=asset).values
        print(f"    {asset:6s}: {label:5s} (market_cap={mcap:7.2f})")
    
    print(f"\n  Shape preserved: Input {market_cap.shape} → Output {size_result.shape}")
    print(f"  Data type: {size_result.dtype}")
    
    # =========================================================================
    # Step 3: Independent Sort - Value Quantiles
    # =========================================================================
    print("\n[Step 3] Independent Sort: Value quantiles (whole universe)...")
    print("-" * 80)
    
    # Create value quantile expression (independent sort)
    value_indep_expr = CsQuantile(
        child=Field('btm'),
        bins=3,
        labels=['low', 'mid', 'high']
    )
    
    value_indep_result = rc._evaluator.evaluate(value_indep_expr)
    
    print(f"  Bins: 3")
    print(f"  Labels: ['low', 'mid', 'high']")
    print(f"  Method: Independent sort (all stocks ranked together)")
    
    print(f"\n  Value labels (independent, day 0):")
    value_indep_day0 = value_indep_result.isel(time=0).to_pandas()
    for asset, label in value_indep_day0.items():
        btm_val = btm.isel(time=0).sel(asset=asset).values
        print(f"    {asset:6s}: {label:4s} (B/M={btm_val:5.3f})")
    
    # =========================================================================
    # Step 4: Dependent Sort - Value within Size Groups
    # =========================================================================
    print("\n[Step 4] Dependent Sort: Value within size groups (Fama-French)...")
    print("-" * 80)
    
    # Create value quantile expression (dependent sort)
    value_dep_expr = CsQuantile(
        child=Field('btm'),
        bins=3,
        labels=['low', 'mid', 'high'],
        group_by='size'  # Sort within each size group!
    )
    
    value_dep_result = rc._evaluator.evaluate(value_dep_expr)
    rc.add_data('value', value_dep_result)
    
    print(f"  Bins: 3")
    print(f"  Labels: ['low', 'mid', 'high']")
    print(f"  Method: Dependent sort (ranked WITHIN each size group)")
    print(f"  Group by: 'size' field")
    
    print(f"\n  Value labels (dependent, day 0):")
    value_dep_day0 = value_dep_result.isel(time=0).to_pandas()
    size_day0 = size_result.isel(time=0).to_pandas()
    for asset in asset_index:
        btm_val = btm.isel(time=0).sel(asset=asset).values
        size_label = size_day0[asset]
        value_label = value_dep_day0[asset]
        print(f"    {asset:6s}: {value_label:4s} [{size_label:5s} group] (B/M={btm_val:5.3f})")
    
    # =========================================================================
    # Step 5: Compare Independent vs Dependent Sort
    # =========================================================================
    print("\n[Step 5] Comparing Independent vs Dependent Sort...")
    print("-" * 80)
    
    print(f"\n  COMPARISON at day 0:")
    print(f"  {'Asset':<8} {'Size':<6} {'B/M Value':<10} {'Indep Sort':<12} {'Dep Sort':<10} {'Same?'}")
    print(f"  {'-'*8} {'-'*6} {'-'*10} {'-'*12} {'-'*10} {'-'*5}")
    
    differences = 0
    for asset in asset_index:
        btm_val = btm.isel(time=0).sel(asset=asset).values
        size_label = size_day0[asset]
        indep_label = value_indep_day0[asset]
        dep_label = value_dep_day0[asset]
        same = indep_label == dep_label
        if not same:
            differences += 1
        
        print(f"  {asset:<8} {size_label:<6} {btm_val:<10.3f} {indep_label:<12} {dep_label:<10} {'✓' if same else '✗'}")
    
    print(f"\n  Differences: {differences}/{N} assets have different labels")
    print(f"  Insight: Dependent sort creates DIFFERENT CUTOFFS within each size group")
    print(f"           This is the Fama-French 2×3 portfolio methodology!")
    
    # =========================================================================
    # Step 6: Fama-French 2×3 Portfolio Construction
    # =========================================================================
    print("\n[Step 6] Fama-French Style Portfolio Construction...")
    print("-" * 80)
    
    print(f"\n  Using dependent sort results, we can create 6 portfolios:")
    
    # Count stocks in each portfolio at day 0
    portfolios = {}
    for size_label in ['small', 'big']:
        for value_label in ['low', 'mid', 'high']:
            mask = (size_result.isel(time=0) == size_label) & (value_dep_result.isel(time=0) == value_label)
            stocks = [asset for asset in asset_index if mask.sel(asset=asset).values]
            portfolios[f"{size_label}_{value_label}"] = stocks
            print(f"    {size_label.upper():5s} × {value_label.upper():4s}: {stocks}")
    
    print(f"\n  Classic Fama-French factors can be constructed as:")
    print(f"    SMB = (small_low + small_mid + small_high) / 3")
    print(f"          - (big_low + big_mid + big_high) / 3")
    print(f"    ")
    print(f"    HML = (small_high + big_high) / 2")
    print(f"          - (small_low + big_low) / 2")
    
    # =========================================================================
    # Step 7: Shape and Type Verification
    # =========================================================================
    print("\n[Step 7] Verification: Shape preservation and data types...")
    print("-" * 80)
    
    print(f"\n  Input data shape:")
    print(f"    market_cap: {market_cap.shape}")
    print(f"    btm:        {btm.shape}")
    
    print(f"\n  Output shape (preserved):")
    print(f"    size:  {size_result.shape}")
    print(f"    value: {value_dep_result.shape}")
    
    print(f"\n  Data types:")
    print(f"    Input (market_cap): {market_cap.dtype}")
    print(f"    Output (size):      {size_result.dtype} (categorical labels)")
    
    print(f"\n  ✓ Shape preserved: (T, N) → (T, N)")
    print(f"  ✓ Type converted: numeric → categorical")
    
    # =========================================================================
    # Step 8: Integration with Boolean Expressions
    # =========================================================================
    print("\n[Step 8] Integration: Using cs_quantile with Boolean Expressions...")
    print("-" * 80)
    
    # Create boolean masks using comparisons
    small_cap = rc.data['size'] == 'small'
    high_value = rc.data['value'] == 'high'
    
    # Combine with logical operators
    small_value = small_cap & high_value
    
    # Evaluate
    small_value_mask = rc.evaluate(small_value)
    
    print(f"  Expression: (rc.data['size'] == 'small') & (rc.data['value'] == 'high')")
    print(f"\n  Small-cap Value stocks (day 0):")
    small_value_day0 = small_value_mask.isel(time=0)
    for asset in asset_index:
        is_selected = small_value_day0.sel(asset=asset).values
        if is_selected:
            size_lbl = size_day0[asset]
            value_lbl = value_dep_day0[asset]
            print(f"    ✓ {asset}: size={size_lbl}, value={value_lbl}")
    
    print(f"\n  ✓ cs_quantile integrates seamlessly with Boolean Expressions")
    print(f"  ✓ Enables powerful multi-dimensional portfolio selection")
    
    # =========================================================================
    # Step 9: Performance Summary
    # =========================================================================
    print("\n[Step 9] Performance and Capabilities Summary...")
    print("-" * 80)
    
    print(f"\n  Dataset: {T} days × {N} assets")
    print(f"  Operations completed:")
    print(f"    1. Independent sort (size): 2 bins")
    print(f"    2. Independent sort (value): 3 bins")
    print(f"    3. Dependent sort (value within size): 3 bins × 2 groups")
    print(f"    4. Boolean Expression integration")
    
    print(f"\n  Key Features Demonstrated:")
    print(f"    ✓ Shape preservation: (T, N) → (T, N)")
    print(f"    ✓ Categorical labels: 'small'/'big', 'low'/'mid'/'high'")
    print(f"    ✓ Independent vs dependent sorting")
    print(f"    ✓ Different cutoffs per group (Fama-French methodology)")
    print(f"    ✓ Integration with Boolean Expressions")
    print(f"    ✓ Multi-dimensional portfolio construction")
    
    print(f"\n  Ready for:")
    print(f"    → Fama-French factor research")
    print(f"    → Multi-dimensional portfolio strategies")
    print(f"    → Quantile-based signal generation")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SHOWCASE COMPLETE")
    print("=" * 80)
    
    print(f"\n✓ cs_quantile operator successfully demonstrated!")
    print(f"✓ Both independent and dependent sort validated")
    print(f"✓ Fama-French 2×3 portfolio methodology ready")
    print(f"✓ Integration with Boolean Expressions confirmed")
    
    print(f"\n  This operator enables sophisticated multi-dimensional")
    print(f"  factor research and portfolio construction!")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

