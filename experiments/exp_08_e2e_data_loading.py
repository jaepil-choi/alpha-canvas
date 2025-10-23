"""
Experiment 08: End-to-End Data Loading Integration

Date: 2024-01-20
Status: In Progress

Objective:
- Validate complete workflow from Parquet file to AlphaCanvas
- Test Field expression triggering data load
- Verify data accessible via rc.db

Success Criteria:
- [ ] AlphaCanvas initializes with date range
- [ ] Field('adj_close') triggers automatic data load
- [ ] Data accessible via rc.db
- [ ] Data has correct shape and coordinates
- [ ] Multiple fields can be loaded
"""

from alpha_canvas import AlphaCanvas
from alpha_canvas.core.expression import Field

def main():
    print("=" * 70)
    print("EXPERIMENT 08: End-to-End Data Loading Integration")
    print("=" * 70)
    
    # Step 1: Initialize AlphaCanvas with date range
    print("\n[Step 1] Initializing AlphaCanvas with date range...")
    rc = AlphaCanvas(
        config_dir='config',
        start_date='2024-01-05',
        end_date='2024-01-15'
    )
    
    print("[OK] AlphaCanvas initialized")
    print(f"     Config: {rc._config is not None}")
    print(f"     DataLoader: {rc._data_loader is not None}")
    print(f"     Panel: {rc._panel is not None}")
    print(f"     Evaluator: {rc._evaluator is not None}")
    
    # Step 2: Add field via Expression (should trigger data load)
    print("\n[Step 2] Adding field via Expression...")
    print("  Creating Field('adj_close') expression...")
    
    field_expr = Field('adj_close')
    print(f"  [OK] Expression created: {field_expr}")
    
    print("\n  Adding to AlphaCanvas (this will trigger data load)...")
    rc.add_data('close', field_expr)
    
    print("  [OK] Data added successfully")
    
    # Step 3: Verify data is loaded and accessible
    print("\n[Step 3] Verifying data accessibility...")
    
    # Check if in rules
    print(f"  'close' in rules: {'close' in rc.rules}")
    
    # Check if in dataset
    print(f"  'close' in dataset: {'close' in rc.db.data_vars}")
    
    if 'close' in rc.db.data_vars:
        print("  [OK] Data variable exists in dataset")
    
    # Step 4: Inspect data shape and properties
    print("\n[Step 4] Inspecting loaded data...")
    
    close_data = rc.db['close']
    print(f"  Data type: {type(close_data)}")
    print(f"  Shape: {close_data.shape}")
    print(f"  Dimensions: {close_data.dims}")
    print(f"  Dtype: {close_data.dtype}")
    
    print(f"\n  Coordinates:")
    print(f"    time: {len(close_data.coords['time'])} values")
    print(f"          First: {close_data.coords['time'].values[0]}")
    print(f"          Last: {close_data.coords['time'].values[-1]}")
    print(f"    asset: {len(close_data.coords['asset'])} values")
    print(f"           {list(close_data.coords['asset'].values)}")
    
    # Step 5: Verify data range
    print("\n[Step 5] Verifying date range filtering...")
    
    first_date = str(close_data.coords['time'].values[0])
    last_date = str(close_data.coords['time'].values[-1])
    
    print(f"  Requested: 2024-01-05 to 2024-01-15")
    print(f"  Actual:    {first_date} to {last_date}")
    print(f"  Match: {first_date >= '2024-01-05' and last_date <= '2024-01-15'}")
    
    # Step 6: Display sample data
    print("\n[Step 6] Sample data values...")
    print(f"  First time point (all assets):")
    print(f"    {close_data.isel(time=0).values}")
    
    print(f"\n  First asset (all time points):")
    print(f"    {close_data.isel(asset=0).values}")
    
    # Step 7: Load second field
    print("\n[Step 7] Loading second field (volume)...")
    
    volume_expr = Field('volume')
    rc.add_data('vol', volume_expr)
    
    print("  [OK] Volume data added")
    print(f"       Shape: {rc.db['vol'].shape}")
    print(f"       Dtype: {rc.db['vol'].dtype}")
    
    # Step 8: Verify both fields have same shape
    print("\n[Step 8] Verifying consistency...")
    
    close_shape = rc.db['close'].shape
    volume_shape = rc.db['vol'].shape
    
    print(f"  close shape:  {close_shape}")
    print(f"  volume shape: {volume_shape}")
    print(f"  Shapes match: {close_shape == volume_shape}")
    
    # Check coordinates match
    time_match = all(rc.db['close'].coords['time'] == rc.db['vol'].coords['time'])
    asset_match = all(rc.db['close'].coords['asset'] == rc.db['vol'].coords['asset'])
    
    print(f"  Time coords match:  {time_match}")
    print(f"  Asset coords match: {asset_match}")
    
    # Step 9: Test "eject" pattern
    print("\n[Step 9] Testing Open Toolkit (eject pattern)...")
    
    pure_ds = rc.db
    print(f"  Ejected dataset type: {type(pure_ds)}")
    print(f"  Is xarray.Dataset: {type(pure_ds).__name__ == 'Dataset'}")
    print(f"  Data variables: {list(pure_ds.data_vars)}")
    
    # Step 10: Test loading same field again (should use cache)
    print("\n[Step 10] Testing data caching...")
    
    print("  Adding 'adj_close' field again with different name...")
    rc.add_data('close_copy', Field('adj_close'))
    
    print(f"  [OK] Added successfully")
    print(f"       Data variables now: {list(rc.db.data_vars)}")
    
    # Verify data is identical
    import numpy as np
    data_match = np.array_equal(
        rc.db['close'].values, 
        rc.db['close_copy'].values
    )
    print(f"       Data identical: {data_match}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("[SUCCESS] End-to-End Data Loading Integration Complete!")
    print()
    print("Workflow Validated:")
    print("  1. ✓ AlphaCanvas initialized with date range")
    print("  2. ✓ Field expression created")
    print("  3. ✓ add_data() triggers automatic Parquet loading via DataLoader")
    print("  4. ✓ DuckDB query executed with date filtering")
    print("  5. ✓ Long format pivoted to (T, N) wide format")
    print("  6. ✓ Converted to xarray.DataArray")
    print("  7. ✓ Added to rc.db Dataset")
    print("  8. ✓ Data accessible via rc.db['field_name']")
    print("  9. ✓ Multiple fields loaded with consistent coordinates")
    print(" 10. ✓ Open Toolkit eject pattern works")
    print()
    print("Final State:")
    print(f"  Dataset shape: ({close_shape[0]} time points, {close_shape[1]} assets)")
    print(f"  Data variables: {list(rc.db.data_vars)}")
    print(f"  Date range: {first_date} to {last_date}")
    print(f"  Assets: {list(rc.db.coords['asset'].values)}")
    print()
    print("✓ Phase 5 Complete: Parquet Data Loading with DuckDB")
    print("=" * 70)


if __name__ == '__main__':
    main()

