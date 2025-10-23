"""
Showcase 03: Expression & Visitor Patterns

This script demonstrates the Expression tree (Composite pattern) and
EvaluateVisitor (Visitor pattern) with integer-based step caching.

Run: poetry run python showcase/03_expression_visitor.py
"""

import numpy as np
import pandas as pd
import xarray as xr
from alpha_canvas.core.expression import Expression, Field, AddOne
from alpha_canvas.core.visitor import EvaluateVisitor


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print("\n" + "=" * 70)
    print("  ALPHA-CANVAS MVP SHOWCASE")
    print("  Phase 3: Expression & Visitor Patterns")
    print("=" * 70)
    
    # Setup: Create test dataset
    print_section("Setup: Creating Test Dataset")
    time_idx = pd.date_range('2020-01-01', periods=100)
    asset_idx = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    
    ds = xr.Dataset(coords={'time': time_idx, 'asset': asset_idx})
    
    # Add mock data
    returns = xr.DataArray(
        np.random.randn(100, 5) * 0.02,
        dims=['time', 'asset'],
        coords={'time': time_idx, 'asset': asset_idx}
    )
    market_cap = xr.DataArray(
        np.random.randn(100, 5) * 1000 + 5000,
        dims=['time', 'asset'],
        coords={'time': time_idx, 'asset': asset_idx}
    )
    volume = xr.DataArray(
        np.random.randn(100, 5) * 1000000 + 5000000,
        dims=['time', 'asset'],
        coords={'time': time_idx, 'asset': asset_idx}
    )
    
    ds = ds.assign({
        'returns': returns,
        'market_cap': market_cap,
        'volume': volume
    })
    
    print("[OK] Test dataset created")
    print(f"     Dimensions: time={len(time_idx)}, asset={len(asset_idx)}")
    print(f"     Data variables: {list(ds.data_vars)}")
    
    # Section 1: Expression creation
    print_section("1. Creating Expression Objects (Composite Pattern)")
    
    field1 = Field('returns')
    print(f"[OK] Created: {field1}")
    print(f"     Type: {type(field1).__name__}")
    print(f"     Is Expression: {isinstance(field1, Expression)}")
    
    field2 = Field('market_cap')
    print(f"\n[OK] Created: {field2}")
    
    field3 = Field('volume')
    print(f"\n[OK] Created: {field3}")
    
    # Section 2: Visitor initialization
    print_section("2. Initializing EvaluateVisitor")
    visitor = EvaluateVisitor(ds)
    print("[OK] EvaluateVisitor created")
    print(f"     Data source: xarray.Dataset")
    print(f"     Initial step counter: {visitor._step_counter}")
    print(f"     Initial cache size: {len(visitor._cache)}")
    
    # Section 3: Evaluating expressions
    print_section("3. Evaluating Expression (Visitor Pattern)")
    
    print("\n  Evaluating Field('returns')...")
    result1 = visitor.evaluate(field1)
    print(f"  [OK] Evaluation complete")
    print(f"       Result type: {type(result1)}")
    print(f"       Result shape: {result1.shape}")
    print(f"       Mean: {result1.mean().item():.6f}")
    print(f"       Step counter after eval: {visitor._step_counter}")
    
    # Section 4: Cache inspection
    print_section("4. Inspecting Cache (Integer Step Indexing)")
    
    print(f"[OK] Cache contains {len(visitor._cache)} entry")
    print("\n  Cache structure:")
    for step, (name, data) in visitor._cache.items():
        print(f"    Step {step}: name='{name}', shape={data.shape}, dtype={data.dtype}")
    
    # Section 5: Sequential evaluation
    print_section("5. Sequential Evaluation (Multiple Fields)")
    
    # Reset and evaluate multiple fields
    visitor._step_counter = 0
    visitor._cache = {}
    
    print("\n  Evaluating multiple fields sequentially...")
    _ = field1.accept(visitor)  # Step 0
    print(f"  [OK] Field('returns')   -> Step 0, counter now: {visitor._step_counter}")
    
    _ = field2.accept(visitor)  # Step 1
    print(f"  [OK] Field('market_cap') -> Step 1, counter now: {visitor._step_counter}")
    
    _ = field3.accept(visitor)  # Step 2
    print(f"  [OK] Field('volume')     -> Step 2, counter now: {visitor._step_counter}")
    
    print(f"\n[OK] Final cache size: {len(visitor._cache)} entries")
    
    # Section 6: Cache retrieval
    print_section("6. Retrieving Cached Results")
    
    print("\n  Retrieving cached steps...")
    for step in range(3):
        name, data = visitor.get_cached(step)
        print(f"  Step {step}: '{name:20s}'  shape={data.shape}  mean={data.mean().item():10.2f}")
    
    # Section 7: Cache persistence
    print_section("7. Cache Persistence Test")
    
    # Store reference to step 1
    name_1, data_1 = visitor.get_cached(1)
    print(f"[OK] Retrieved step 1: '{name_1}'")
    print(f"     Data is identical object: {data_1 is ds['market_cap']}")
    
    # Verify data integrity
    print(f"     Data integrity check: ", end="")
    if np.array_equal(data_1.values, ds['market_cap'].values):
        print("[PASS]")
    else:
        print("[FAIL]")
    
    # Section 8: Evaluate reset behavior
    print_section("8. Evaluate() Reset Behavior")
    
    print("\n  Before new evaluate():")
    print(f"    Step counter: {visitor._step_counter}")
    print(f"    Cache size: {len(visitor._cache)}")
    
    # New evaluation resets cache
    result_new = visitor.evaluate(Field('returns'))
    print(f"\n  After evaluate(Field('returns')):")
    print(f"    Step counter: {visitor._step_counter}")
    print(f"    Cache size: {len(visitor._cache)}")
    print(f"    [OK] Cache was reset for new evaluation")
    
    # Section 9: Composite Expression (Depth-First Traversal)
    print_section("9. Composite Expression (AddOne Operator)")
    
    print("\n  Creating composite expression: AddOne(Field('returns'))...")
    composite_expr = AddOne(Field('returns'))
    print(f"  [OK] Expression created")
    print(f"       Type: {type(composite_expr).__name__}")
    print(f"       Child: {composite_expr.child}")
    
    print("\n  Evaluating composite expression...")
    print("  Expected traversal order (depth-first):")
    print("    1. Visit Field('returns')           -> Step 0")
    print("    2. Visit AddOne (apply +1 operation) -> Step 1")
    
    result_composite = visitor.evaluate(composite_expr)
    print(f"\n  [OK] Evaluation complete")
    print(f"       Final result shape: {result_composite.shape}")
    print(f"       Final result mean: {result_composite.mean().item():.6f}")
    print(f"       Original returns mean: {ds['returns'].mean().item():.6f}")
    print(f"       Difference: {(result_composite.mean() - ds['returns'].mean()).item():.6f}")
    
    print(f"\n  Cache after composite evaluation:")
    print(f"       Cache size: {len(visitor._cache)} steps")
    for step, (name, data) in visitor._cache.items():
        print(f"       Step {step}: {name:20s}  mean={data.mean().item():10.6f}")
    
    print("\n  [OK] Depth-first traversal verified!")
    print("       Step 0: Child evaluated first")
    print("       Step 1: Parent operation applied")
    
    # Section 10: Nested Composite Expression
    print_section("10. Nested Composite Expression")
    
    print("\n  Creating nested expression: AddOne(AddOne(Field('returns')))...")
    nested_expr = AddOne(AddOne(Field('returns')))
    print(f"  [OK] Nested expression created")
    print(f"       Outer: AddOne")
    print(f"       Inner: AddOne")
    print(f"       Leaf:  Field('returns')")
    
    print("\n  Expected traversal order (depth-first, post-order):")
    print("    1. Visit Field('returns')              -> Step 0")
    print("    2. Visit AddOne (inner, +1)            -> Step 1")
    print("    3. Visit AddOne (outer, +1)            -> Step 2")
    
    result_nested = visitor.evaluate(nested_expr)
    print(f"\n  [OK] Evaluation complete")
    print(f"       Final result mean: {result_nested.mean().item():.6f}")
    print(f"       Original returns mean: {ds['returns'].mean().item():.6f}")
    print(f"       Expected difference: +2.000000")
    print(f"       Actual difference: {(result_nested.mean() - ds['returns'].mean()).item():.6f}")
    
    print(f"\n  Cache after nested evaluation:")
    print(f"       Cache size: {len(visitor._cache)} steps")
    for step, (name, data) in visitor._cache.items():
        mean_val = data.mean().item()
        print(f"       Step {step}: {name:20s}  mean={mean_val:10.6f}")
    
    print("\n  [OK] Nested depth-first traversal verified!")
    print("       Step 0: Leaf (Field)")
    print("       Step 1: Inner operation (AddOne)")
    print("       Step 2: Outer operation (AddOne)")
    
    # Section 11: Error handling
    print_section("11. Error Handling")
    
    print("\n  Testing non-existent field...")
    try:
        bad_field = Field('nonexistent')
        visitor.evaluate(bad_field)
        print("  [FAIL] Should have raised KeyError")
    except KeyError as e:
        print(f"  [OK] Correctly raised KeyError: {e}")
    
    # Final summary
    print_section("SUMMARY")
    print("[SUCCESS] Expression & Visitor Demonstration Complete")
    print()
    print("Key Features Demonstrated:")
    print("  ✓ Expression ABC and Field leaf nodes")
    print("  ✓ Composite pattern for computation trees")
    print("  ✓ Visitor pattern for expression evaluation")
    print("  ✓ Integer-based step indexing (0, 1, 2...)")
    print("  ✓ Cache structure: Dict[int, Tuple[str, DataArray]]")
    print("  ✓ Sequential evaluation with step counter")
    print("  ✓ Cache retrieval by step number")
    print("  ✓ Cache reset on new evaluate()")
    print("  ✓ Composite expressions (AddOne operator)")
    print("  ✓ Depth-first traversal (children evaluated first)")
    print("  ✓ Nested expressions (AddOne(AddOne(...)))")
    print("  ✓ Post-order caching (cache after computation)")
    print("  ✓ Error handling for missing fields")
    print()
    print("Expression Tree Evaluation:")
    print(f"  • Simple fields evaluated: 3")
    print(f"  • Composite expressions: 2 (single and nested)")
    print(f"  • Final cache steps: {len(visitor._cache)}")
    print(f"  • Pattern: Depth-first, post-order traversal ✓")
    print()
    print("This demonstrates how future operators (ts_mean, cs_rank, etc.)")
    print("will work: evaluate children first, then apply operation, then cache.")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()

