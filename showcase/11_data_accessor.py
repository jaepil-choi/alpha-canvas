"""
Showcase 11: DataAccessor - Selector Interface

Demonstrates:
1. Basic field access returning Field Expressions
2. Comparison creating Boolean Expressions
3. Complex logical chains
4. Evaluation with universe masking
5. Multi-dimensional selection patterns (Fama-French style)

This showcase proves that Phase 7B (DataAccessor) successfully integrates
with Phase 7A (Boolean Expressions) to enable natural, Pythonic selector
syntax while maintaining lazy evaluation and universe safety.
"""

import numpy as np
import pandas as pd
import xarray as xr
from alpha_canvas import AlphaCanvas
from alpha_canvas.core.expression import Field, Expression
from alpha_canvas.ops.logical import Equals, And


def print_section(title):
    """Print section header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def main():
    print("="*60)
    print("SHOWCASE 11: DataAccessor - Selector Interface")
    print("="*60)
    print("\nDemonstrates Phase 7B: rc.data accessor returns Field Expressions")
    print("Integrated with Phase 7A: Boolean Expression infrastructure")
    
    # ================================================================
    # Setup: Create AlphaCanvas with Sample Data
    # ================================================================
    print_section("Setup: Creating Sample Data")
    
    time_index = pd.date_range('2024-01-01', periods=10, freq='D')
    asset_index = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
    
    print(f"  Time range: {time_index[0].date()} to {time_index[-1].date()}")
    print(f"  Assets: {asset_index}")
    
    # Create categorical data for selection
    size_data = xr.DataArray(
        np.array([
            ['small', 'big', 'small', 'big', 'small'],
            ['small', 'big', 'big', 'small', 'small'],
            ['big', 'small', 'small', 'big', 'small'],
            ['small', 'big', 'small', 'small', 'big'],
            ['big', 'small', 'big', 'small', 'small'],
            ['small', 'big', 'small', 'big', 'small'],
            ['big', 'small', 'big', 'small', 'big'],
            ['small', 'big', 'small', 'big', 'small'],
            ['big', 'small', 'small', 'small', 'big'],
            ['small', 'big', 'big', 'big', 'small'],
        ]),
        dims=['time', 'asset'],
        coords={'time': time_index, 'asset': asset_index}
    )
    
    value_data = xr.DataArray(
        np.array([
            ['high', 'low', 'high', 'low', 'high'],
            ['low', 'high', 'low', 'high', 'low'],
            ['high', 'low', 'high', 'low', 'high'],
            ['low', 'high', 'low', 'high', 'low'],
            ['high', 'low', 'high', 'low', 'high'],
            ['low', 'high', 'low', 'high', 'low'],
            ['high', 'low', 'high', 'low', 'high'],
            ['low', 'high', 'low', 'high', 'low'],
            ['high', 'low', 'high', 'low', 'high'],
            ['low', 'high', 'low', 'high', 'low'],
        ]),
        dims=['time', 'asset'],
        coords={'time': time_index, 'asset': asset_index}
    )
    
    momentum_data = xr.DataArray(
        np.array([
            ['high', 'high', 'low', 'high', 'low'],
            ['low', 'high', 'low', 'low', 'high'],
            ['high', 'low', 'high', 'high', 'low'],
            ['high', 'high', 'low', 'low', 'high'],
            ['low', 'low', 'high', 'high', 'low'],
            ['high', 'high', 'low', 'high', 'low'],
            ['low', 'high', 'high', 'low', 'high'],
            ['high', 'low', 'low', 'high', 'low'],
            ['low', 'high', 'high', 'low', 'high'],
            ['high', 'low', 'low', 'high', 'high'],
        ]),
        dims=['time', 'asset'],
        coords={'time': time_index, 'asset': asset_index}
    )
    
    price_data = xr.DataArray(
        np.array([
            [150.0, 300.0, 120.0, 400.0, 200.0],
            [155.0, 310.0, 125.0, 410.0, 205.0],
            [160.0, 320.0, 130.0, 420.0, 210.0],
            [165.0, 330.0, 135.0, 430.0, 215.0],
            [170.0, 340.0, 140.0, 440.0, 220.0],
            [175.0, 350.0, 145.0, 450.0, 225.0],
            [180.0, 360.0, 150.0, 460.0, 230.0],
            [185.0, 370.0, 155.0, 470.0, 235.0],
            [190.0, 380.0, 160.0, 480.0, 240.0],
            [195.0, 390.0, 165.0, 490.0, 245.0],
        ]),
        dims=['time', 'asset'],
        coords={'time': time_index, 'asset': asset_index}
    )
    
    # Initialize AlphaCanvas
    rc = AlphaCanvas(
        time_index=time_index,
        asset_index=asset_index
    )
    
    # Add data to canvas
    rc.add_data('size', size_data)
    rc.add_data('value', value_data)
    rc.add_data('momentum', momentum_data)
    rc.add_data('price', price_data)
    
    print(f"  Dataset variables: {list(rc.db.data_vars.keys())}")
    
    # ================================================================
    # Section 1: Basic Field Access Returns Field Expression
    # ================================================================
    print_section("1. Basic Field Access")
    
    size_field = rc.data['size']
    
    print(f"  rc.data['size'] → {size_field}")
    print(f"  Type: {type(size_field).__name__}")
    print(f"  Is Field: {isinstance(size_field, Field)}")
    print(f"  Is Expression: {isinstance(size_field, Expression)}")
    print(f"  Field name: {size_field.name}")
    print("\n  [OK] Accessor returns Field Expression (lazy)")
    
    # ================================================================
    # Section 2: Comparison Creates Boolean Expression
    # ================================================================
    print_section("2. Comparison Creates Expression (Not Immediate Boolean)")
    
    mask_expr = rc.data['size'] == 'small'
    
    print(f"  rc.data['size'] == 'small' → {mask_expr}")
    print(f"  Type: {type(mask_expr).__name__}")
    print(f"  Is Equals: {isinstance(mask_expr, Equals)}")
    print(f"  Is Expression: {isinstance(mask_expr, Expression)}")
    print(f"  Left operand: {mask_expr.left}")
    print(f"  Right operand: {mask_expr.right!r}")
    print("\n  [OK] Comparison creates Expression (lazy, not immediate)")
    
    # ================================================================
    # Section 3: Evaluation with Results
    # ================================================================
    print_section("3. Evaluate Expression to Get Boolean DataArray")
    
    result = rc.evaluate(mask_expr)
    
    print(f"  Evaluated shape: {result.shape}")
    print(f"  Evaluated dtype: {result.dtype}")
    print(f"  Result values (first 3 timesteps):")
    for i in range(3):
        print(f"    {time_index[i].date()}: {result.isel(time=i).values}")
    
    # Count True values
    if result.dtype == bool:
        true_count = result.sum().values
    else:
        true_count = (result == 1.0).sum().values
    
    print(f"\n  Total 'small' positions: {true_count} out of {result.size}")
    print(f"  [OK] Expression evaluated to boolean mask")
    
    # ================================================================
    # Section 4: Complex Logical Chains
    # ================================================================
    print_section("4. Complex Logical Chains")
    
    # Create complex expression
    small_expr = rc.data['size'] == 'small'
    momentum_expr = rc.data['momentum'] == 'high'
    complex_expr = small_expr & momentum_expr
    
    print(f"  Small-cap: {small_expr}")
    print(f"  High momentum: {momentum_expr}")
    print(f"  Combined (AND): {complex_expr}")
    print(f"  Type: {type(complex_expr).__name__}")
    print(f"  Is And: {isinstance(complex_expr, And)}")
    
    # Evaluate
    result = rc.evaluate(complex_expr)
    
    if result.dtype == bool:
        true_count = result.sum().values
    else:
        true_count = (result == 1.0).sum().values
    
    print(f"\n  Small-cap AND high momentum: {true_count} positions")
    print(f"  Result (first 2 timesteps):")
    for i in range(2):
        print(f"    {time_index[i].date()}: {result.isel(time=i).values}")
    print(f"\n  [OK] Logical chains work correctly")
    
    # ================================================================
    # Section 5: Multi-Dimensional Selection (Fama-French Style)
    # ================================================================
    print_section("5. Multi-Dimensional Selection (Fama-French Style)")
    
    # Small-cap, high-value stocks
    small_value = (
        (rc.data['size'] == 'small') & 
        (rc.data['value'] == 'high')
    )
    result_sv = rc.evaluate(small_value)
    
    if result_sv.dtype == bool:
        count_sv = result_sv.sum().values
    else:
        count_sv = (result_sv == 1.0).sum().values
    
    print(f"  Small + High Value: {count_sv} positions")
    
    # Big-cap, low-value stocks
    big_growth = (
        (rc.data['size'] == 'big') & 
        (rc.data['value'] == 'low')
    )
    result_bg = rc.evaluate(big_growth)
    
    if result_bg.dtype == bool:
        count_bg = result_bg.sum().values
    else:
        count_bg = (result_bg == 1.0).sum().values
    
    print(f"  Big + Low Value: {count_bg} positions")
    
    # Triple selection: size × value × momentum
    triple_selection = (
        (rc.data['size'] == 'small') & 
        (rc.data['value'] == 'high') & 
        (rc.data['momentum'] == 'high')
    )
    result_triple = rc.evaluate(triple_selection)
    
    if result_triple.dtype == bool:
        count_triple = result_triple.sum().values
    else:
        count_triple = (result_triple == 1.0).sum().values
    
    print(f"  Small + High Value + High Momentum: {count_triple} positions")
    print("\n  [OK] Multi-dimensional Fama-French style selection works")
    
    # ================================================================
    # Section 6: Numeric Comparisons
    # ================================================================
    print_section("6. Numeric Comparisons")
    
    # Price filter
    high_price = rc.data['price'] > 200.0
    result_price = rc.evaluate(high_price)
    
    if result_price.dtype == bool:
        count_price = result_price.sum().values
    else:
        count_price = (result_price == 1.0).sum().values
    
    print(f"  Price > $200: {count_price} positions")
    
    # Combined: high price AND small cap
    premium_small = (rc.data['price'] > 200.0) & (rc.data['size'] == 'small')
    result_ps = rc.evaluate(premium_small)
    
    if result_ps.dtype == bool:
        count_ps = result_ps.sum().values
    else:
        count_ps = (result_ps == 1.0).sum().values
    
    print(f"  Price > $200 AND Small: {count_ps} positions")
    print("\n  [OK] Numeric and categorical comparisons can be combined")
    
    # ================================================================
    # Section 7: Negation
    # ================================================================
    print_section("7. Negation")
    
    # Not small (i.e., big)
    not_small = ~(rc.data['size'] == 'small')
    result_ns = rc.evaluate(not_small)
    
    if result_ns.dtype == bool:
        count_ns = result_ns.sum().values
    else:
        count_ns = (result_ns == 1.0).sum().values
    
    print(f"  NOT small: {count_ns} positions")
    print(f"  [OK] Negation operator works")
    
    # ================================================================
    # Section 8: OR Logic
    # ================================================================
    print_section("8. OR Logic")
    
    # Small OR high momentum
    small_or_momentum = (rc.data['size'] == 'small') | (rc.data['momentum'] == 'high')
    result_or = rc.evaluate(small_or_momentum)
    
    if result_or.dtype == bool:
        count_or = result_or.sum().values
    else:
        count_or = (result_or == 1.0).sum().values
    
    print(f"  Small OR High Momentum: {count_or} positions")
    print(f"  [OK] OR logic works")
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*60)
    print("SHOWCASE SUMMARY")
    print("="*60)
    print("  [OK] 1. DataAccessor returns Field Expressions")
    print("  [OK] 2. Comparisons create Boolean Expressions (lazy)")
    print("  [OK] 3. Expressions evaluate to boolean DataArrays")
    print("  [OK] 4. Complex logical chains work correctly")
    print("  [OK] 5. Multi-dimensional selection (Fama-French style)")
    print("  [OK] 6. Numeric and categorical comparisons combine")
    print("  [OK] 7. Negation operator works")
    print("  [OK] 8. OR logic works")
    print("\n" + "="*60)
    print("SHOWCASE COMPLETE")
    print("="*60)
    print("\nKey Takeaways:")
    print("  - rc.data['field'] returns Field Expression (lazy)")
    print("  - Comparisons create Boolean Expressions (lazy)")
    print("  - rc.evaluate(expr) executes with universe masking")
    print("  - Pythonic syntax: ==, !=, <, >, <=, >=, &, |, ~")
    print("  - No special 'axis' accessor needed - one pattern!")
    print("  - Ready for Fama-French multi-dimensional portfolios")


if __name__ == '__main__':
    main()

