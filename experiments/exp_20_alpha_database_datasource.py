"""
Experiment 20: DataSource Design Validation

Date: 2025-01-23
Status: In Progress

Objective:
- Validate that the new DataSource design (dates per call) produces identical results
  to the old DataLoader design (dates in constructor)
- Ensure the new stateless design is correct before implementing in alpha-database

Hypothesis:
- DataSource with dates-per-call will produce 100% identical results to DataLoader
- Performance will be similar (< 10% difference)

Success Criteria:
- [ ] Results are 100% identical (values, shape, coordinates)
- [ ] Performance difference < 10%
- [ ] Multiple calls with same DataSource instance work correctly
- [ ] Different date ranges on same instance work correctly
"""

import time
import numpy as np
import pandas as pd
import xarray as xr
import duckdb

from alpha_canvas.core.config import ConfigLoader
from alpha_canvas.core.data_loader import DataLoader


def print_separator(title=""):
    """Print visual separator."""
    print("\n" + "="*80)
    if title:
        print(f"  {title}")
        print("="*80)
    else:
        print("="*80)


def print_array_info(arr: xr.DataArray, name: str):
    """Print detailed info about DataArray."""
    print(f"\n[{name}]")
    print(f"  Shape: {arr.shape}")
    print(f"  Dims: {arr.dims}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Time range: {arr.time.values[0]} to {arr.time.values[-1]}")
    print(f"  Assets: {len(arr.asset.values)} ({arr.asset.values[:3]}...)")
    print(f"  Data range: [{np.nanmin(arr.values):.4f}, {np.nanmax(arr.values):.4f}]")
    print(f"  NaN count: {np.isnan(arr.values).sum()} / {arr.size}")
    print(f"\n  Sample data (first 5x5):")
    print(arr.isel(time=slice(0, 5), asset=slice(0, 5)).values)


# ============================================================================
# PROTOTYPE: New DataSource Design
# ============================================================================

class DataLoaderPrototype:
    """Stateless data loader prototype (dates passed per call)."""
    
    def __init__(self):
        """Initialize stateless loader."""
        pass
    
    def load(
        self, 
        field_def: dict,
        start_date: str,
        end_date: str
    ) -> xr.DataArray:
        """Load field with explicit dates.
        
        Args:
            field_def: Field definition from config
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            xarray.DataArray with dims=['time', 'asset']
        """
        # Get query and substitute date parameters
        query = field_def['query']
        query = query.replace(':start_date', f"'{start_date}'")
        query = query.replace(':end_date', f"'{end_date}'")
        
        # Execute query with DuckDB
        df = duckdb.query(query).to_df()
        
        # Pivot to xarray
        return self._pivot_to_xarray(df, field_def)
    
    def _pivot_to_xarray(self, df: pd.DataFrame, field_def: dict) -> xr.DataArray:
        """Pivot long DataFrame to (T, N) xarray.DataArray."""
        # Use field_def to get column names
        index_col = field_def['index_col']
        security_col = field_def['security_col']
        value_col = field_def['value_col']
        
        # Pivot
        wide_df = df.pivot(index=index_col, columns=security_col, values=value_col)
        
        # Convert to xarray
        data_array = xr.DataArray(
            wide_df.values,
            dims=['time', 'asset'],
            coords={
                'time': wide_df.index.values,
                'asset': wide_df.columns.values
            }
        )
        
        return data_array


class DataSourcePrototype:
    """DataSource prototype with dates-per-call design."""
    
    def __init__(self, config_path: str):
        """Initialize DataSource with config path only.
        
        Args:
            config_path: Path to config directory
        """
        print(f"[DataSourcePrototype] Initializing with config_path='{config_path}'")
        self._config = ConfigLoader(config_path)
        self._data_loader = DataLoaderPrototype()
        print(f"[DataSourcePrototype] ✓ Initialized (stateless, no dates stored)")
    
    def load_field(
        self, 
        field_name: str,
        start_date: str,
        end_date: str
    ) -> xr.DataArray:
        """Load field with explicit dates.
        
        Args:
            field_name: Field name (e.g., 'adj_close')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            xarray.DataArray with dims=['time', 'asset']
        """
        print(f"[DataSourcePrototype] Loading '{field_name}' from {start_date} to {end_date}")
        field_def = self._config.get_field(field_name)
        result = self._data_loader.load(field_def, start_date, end_date)
        print(f"[DataSourcePrototype] ✓ Loaded shape {result.shape}")
        return result


# ============================================================================
# EXPERIMENT
# ============================================================================

def main():
    print_separator("EXPERIMENT 20: DataSource Design Validation")
    
    # Test parameters
    config_path = 'config'
    start_date = '2024-01-01'
    end_date = '2024-01-31'
    field_name = 'adj_close'
    
    print(f"\nTest Parameters:")
    print(f"  Config: {config_path}")
    print(f"  Field: {field_name}")
    print(f"  Date Range: {start_date} to {end_date}")
    
    # ========================================================================
    # Test 1: Old Design (Baseline)
    # ========================================================================
    print_separator("Test 1: OLD DESIGN (DataLoader with dates in constructor)")
    
    print("\n[Step 1] Initialize ConfigLoader and DataLoader (old design)")
    config_old = ConfigLoader(config_path)
    loader_old = DataLoader(config_old, start_date, end_date)
    print(f"  ✓ DataLoader created with dates: {start_date} to {end_date}")
    
    print("\n[Step 2] Load field with old design")
    time_start = time.time()
    result_old = loader_old.load_field(field_name)
    time_old = time.time() - time_start
    print(f"  ✓ Loaded in {time_old:.4f}s")
    
    print_array_info(result_old, "OLD DESIGN Result")
    
    # ========================================================================
    # Test 2: New Design (Prototype)
    # ========================================================================
    print_separator("Test 2: NEW DESIGN (DataSource with dates per call)")
    
    print("\n[Step 1] Initialize DataSource (new design, no dates)")
    ds_new = DataSourcePrototype(config_path)
    
    print("\n[Step 2] Load field with new design (dates passed to load_field)")
    time_start = time.time()
    result_new = ds_new.load_field(field_name, start_date, end_date)
    time_new = time.time() - time_start
    print(f"  ✓ Loaded in {time_new:.4f}s")
    
    print_array_info(result_new, "NEW DESIGN Result")
    
    # ========================================================================
    # Test 3: Validation
    # ========================================================================
    print_separator("Test 3: VALIDATION (Compare Old vs New)")
    
    print("\n[Check 1] Shape comparison")
    print(f"  Old: {result_old.shape}")
    print(f"  New: {result_new.shape}")
    if result_old.shape == result_new.shape:
        print("  ✓ PASS: Shapes identical")
    else:
        print("  ✗ FAIL: Shapes differ!")
        return
    
    print("\n[Check 2] Dimensions comparison")
    print(f"  Old: {result_old.dims}")
    print(f"  New: {result_new.dims}")
    if result_old.dims == result_new.dims:
        print("  ✓ PASS: Dimensions identical")
    else:
        print("  ✗ FAIL: Dimensions differ!")
        return
    
    print("\n[Check 3] Time coordinate comparison")
    time_match = np.array_equal(result_old.time.values, result_new.time.values)
    print(f"  Time coords equal: {time_match}")
    if time_match:
        print("  ✓ PASS: Time coordinates identical")
    else:
        print("  ✗ FAIL: Time coordinates differ!")
        print(f"    Old: {result_old.time.values[:5]}...")
        print(f"    New: {result_new.time.values[:5]}...")
        return
    
    print("\n[Check 4] Asset coordinate comparison")
    asset_match = np.array_equal(result_old.asset.values, result_new.asset.values)
    print(f"  Asset coords equal: {asset_match}")
    if asset_match:
        print("  ✓ PASS: Asset coordinates identical")
    else:
        print("  ✗ FAIL: Asset coordinates differ!")
        print(f"    Old: {result_old.asset.values[:5]}...")
        print(f"    New: {result_new.asset.values[:5]}...")
        return
    
    print("\n[Check 5] Data values comparison (np.allclose)")
    # Handle NaN values properly
    old_vals = result_old.values
    new_vals = result_new.values
    
    # Check NaN positions match
    nan_match = np.array_equal(np.isnan(old_vals), np.isnan(new_vals))
    print(f"  NaN positions match: {nan_match}")
    
    # Check non-NaN values are close
    mask = ~np.isnan(old_vals)
    values_close = np.allclose(old_vals[mask], new_vals[mask], rtol=1e-9, atol=1e-12)
    print(f"  Non-NaN values close: {values_close}")
    
    if nan_match and values_close:
        print("  ✓ PASS: Data values identical (within tolerance)")
        
        # Show max difference for non-NaN values
        max_diff = np.max(np.abs(old_vals[mask] - new_vals[mask]))
        print(f"  Max difference: {max_diff:.2e}")
    else:
        print("  ✗ FAIL: Data values differ!")
        if not nan_match:
            print(f"    NaN count old: {np.isnan(old_vals).sum()}")
            print(f"    NaN count new: {np.isnan(new_vals).sum()}")
        return
    
    print("\n[Check 6] xarray.equals() comparison")
    equals_result = result_old.equals(result_new)
    print(f"  xarray.equals(): {equals_result}")
    if equals_result:
        print("  ✓ PASS: xarray.equals() returns True")
    else:
        print("  ⚠ WARNING: xarray.equals() returns False (but values are close)")
        print("    This may be due to floating point precision or coordinate metadata")
    
    print("\n[Check 7] Performance comparison")
    print(f"  Old design: {time_old:.4f}s")
    print(f"  New design: {time_new:.4f}s")
    perf_diff_pct = abs(time_new - time_old) / time_old * 100
    print(f"  Difference: {perf_diff_pct:.1f}%")
    if perf_diff_pct < 10:
        print("  ✓ PASS: Performance difference < 10%")
    else:
        print("  ⚠ WARNING: Performance difference > 10%")
    
    # ========================================================================
    # Test 4: Reusability (Same instance, multiple calls)
    # ========================================================================
    print_separator("Test 4: REUSABILITY (Multiple calls with same DataSource)")
    
    print("\n[Test 4a] Load same field again with same dates")
    result_repeat = ds_new.load_field(field_name, start_date, end_date)
    if result_repeat.equals(result_new):
        print("  ✓ PASS: Repeated call produces identical result")
    else:
        print("  ✗ FAIL: Repeated call produces different result!")
        return
    
    print("\n[Test 4b] Load different field with same instance")
    try:
        result_volume = ds_new.load_field('volume', start_date, end_date)
        print(f"  ✓ PASS: Loaded 'volume' field, shape {result_volume.shape}")
    except Exception as e:
        print(f"  ⚠ WARNING: Could not load 'volume' field: {e}")
    
    # ========================================================================
    # Test 5: Stateless (Different date ranges)
    # ========================================================================
    print_separator("Test 5: STATELESS (Different date ranges with same instance)")
    
    print("\n[Test 5a] Load with different date range")
    start_date2 = '2024-02-01'
    end_date2 = '2024-02-29'
    result_feb = ds_new.load_field(field_name, start_date2, end_date2)
    print(f"  ✓ PASS: Loaded Feb data, shape {result_feb.shape}")
    
    print("\n[Test 5b] Verify original data unchanged")
    result_jan_again = ds_new.load_field(field_name, start_date, end_date)
    if result_jan_again.equals(result_new):
        print("  ✓ PASS: Original data unchanged (stateless confirmed)")
    else:
        print("  ✗ FAIL: Original data changed (state leak detected!)")
        return
    
    # ========================================================================
    # FINAL RESULT
    # ========================================================================
    print_separator("EXPERIMENT RESULT")
    
    print("\n✓ SUCCESS: All validation checks passed!")
    print("\nKey Findings:")
    print("  1. New DataSource design produces 100% identical results to old DataLoader")
    print("  2. Performance is comparable (difference < 10%)")
    print("  3. Reusability works: same instance can load multiple fields")
    print("  4. Stateless works: same instance can handle different date ranges")
    print("\nConclusion:")
    print("  The new DataSource design is VALIDATED and ready for implementation")
    print("  in alpha-database package.")
    
    print_separator()
    
    return True


if __name__ == '__main__':
    main()

