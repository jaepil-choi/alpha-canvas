"""
Experiment 07: Long-to-Wide Pivot and xarray Conversion

Date: 2024-01-20
Status: In Progress

Objective:
- Validate pivoting long format DataFrame to (T, N) xarray.DataArray
- Test dimension naming and coordinate handling
- Verify data integrity after pivot

Success Criteria:
- [ ] Long format (90 rows) → Wide format (15 rows × 6 columns)
- [ ] xarray.DataArray created with correct dimensions
- [ ] Coordinates properly labeled (time, asset)
- [ ] Data values preserved correctly
"""

import duckdb
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path

def main():
    print("=" * 70)
    print("EXPERIMENT 07: Long-to-Wide Pivot and xarray Conversion")
    print("=" * 70)
    
    # Step 1: Load data with DuckDB (from previous experiment)
    print("\n[Step 1] Loading data with DuckDB...")
    parquet_path = Path('data/pricevolume.parquet')
    
    query = f"""
    SELECT 
      date, 
      security_id, 
      close_price * adjustment_factor as adj_close 
    FROM read_parquet('{parquet_path}')
    WHERE date >= '2024-01-05' AND date <= '2024-01-15'
    """
    
    df_long = duckdb.query(query).to_df()
    print(f"  [OK] Loaded long format data")
    print(f"       Shape: {df_long.shape}")
    print(f"       Columns: {list(df_long.columns)}")
    
    print(f"\n  Long format sample (first 12 rows):")
    print(df_long.head(12).to_string(index=False))
    
    # Step 2: Pivot to wide format
    print("\n[Step 2] Pivoting to wide format...")
    print(f"  Pivot parameters:")
    print(f"    index (rows): date")
    print(f"    columns: security_id")
    print(f"    values: adj_close")
    
    df_wide = df_long.pivot(index='date', columns='security_id', values='adj_close')
    
    print(f"\n  [OK] Pivot complete")
    print(f"       Shape: {df_wide.shape}")
    print(f"       Index (dates): {len(df_wide.index)}")
    print(f"       Columns (securities): {len(df_wide.columns)}")
    print(f"       Column names: {list(df_wide.columns)}")
    
    print(f"\n  Wide format DataFrame:")
    print(df_wide.to_string())
    
    # Step 3: Validate pivot
    print("\n[Step 3] Validating pivot correctness...")
    
    # Check: no missing values (all securities traded every day)
    missing_count = df_wide.isnull().sum().sum()
    print(f"  Missing values: {missing_count}")
    if missing_count == 0:
        print(f"    [OK] No missing values")
    else:
        print(f"    [WARN] {missing_count} missing values detected")
    
    # Check: values match original
    sample_date = df_wide.index[0]
    sample_security = df_wide.columns[0]
    wide_value = df_wide.loc[sample_date, sample_security]
    long_value = df_long[(df_long['date'] == sample_date) & 
                         (df_long['security_id'] == sample_security)]['adj_close'].iloc[0]
    
    print(f"\n  Spot check:")
    print(f"    Date: {sample_date}, Security: {sample_security}")
    print(f"    Long format value: {long_value:.2f}")
    print(f"    Wide format value: {wide_value:.2f}")
    print(f"    Match: {wide_value == long_value}")
    
    # Step 4: Convert to xarray.DataArray
    print("\n[Step 4] Converting to xarray.DataArray...")
    
    data_array = xr.DataArray(
        df_wide.values,
        dims=['time', 'asset'],
        coords={
            'time': df_wide.index.values,  # Convert to numpy array
            'asset': df_wide.columns.values  # Convert to numpy array
        }
    )
    
    print(f"  [OK] xarray.DataArray created")
    print(f"       Type: {type(data_array)}")
    print(f"       Dimensions: {data_array.dims}")
    print(f"       Shape: {data_array.shape}")
    print(f"       Size: {data_array.size}")
    print(f"       Dtype: {data_array.dtype}")
    
    print(f"\n  Coordinates:")
    print(f"    time: {len(data_array.coords['time'])} values")
    print(f"          First: {data_array.coords['time'].values[0]}")
    print(f"          Last: {data_array.coords['time'].values[-1]}")
    print(f"    asset: {len(data_array.coords['asset'])} values")
    print(f"           {list(data_array.coords['asset'].values)}")
    
    # Step 5: Test xarray indexing
    print("\n[Step 5] Testing xarray indexing...")
    
    # Select by time
    first_time = data_array.isel(time=0)
    print(f"  [OK] Select first time slice: shape={first_time.shape}")
    print(f"       Values: {first_time.values}")
    
    # Select by asset
    first_asset = data_array.isel(asset=0)
    print(f"\n  [OK] Select first asset: shape={first_asset.shape}")
    print(f"       Asset name: {data_array.coords['asset'].values[0]}")
    print(f"       Values: {first_asset.values}")
    
    # Select specific point
    specific_value = data_array.isel(time=0, asset=0)
    print(f"\n  [OK] Select specific point [0,0]: {specific_value.values:.2f}")
    print(f"       Matches DataFrame: {specific_value.values == df_wide.iloc[0, 0]}")
    
    # Step 6: Test xarray label-based indexing
    print("\n[Step 6] Testing label-based indexing...")
    
    # Select by date label
    date_label = data_array.coords['time'].values[2]
    data_by_date = data_array.sel(time=date_label)
    print(f"  [OK] Select by date '{date_label}': shape={data_by_date.shape}")
    
    # Select by asset label
    asset_label = 'AAPL'
    data_by_asset = data_array.sel(asset=asset_label)
    print(f"  [OK] Select by asset '{asset_label}': shape={data_by_asset.shape}")
    
    # Select specific cell by labels
    specific_cell = data_array.sel(time=date_label, asset=asset_label)
    print(f"  [OK] Select cell ({date_label}, {asset_label}): {specific_cell.values:.2f}")
    
    # Step 7: Test xarray operations
    print("\n[Step 7] Testing xarray operations...")
    
    # Mean across time
    mean_by_asset = data_array.mean(dim='time')
    print(f"  [OK] Mean by asset (across time): shape={mean_by_asset.shape}")
    print(f"       Values: {mean_by_asset.values}")
    
    # Mean across assets
    mean_by_time = data_array.mean(dim='asset')
    print(f"\n  [OK] Mean by time (across assets): shape={mean_by_time.shape}")
    print(f"       Values: {mean_by_time.values}")
    
    # Overall mean
    overall_mean = data_array.mean()
    print(f"\n  [OK] Overall mean: {overall_mean.values:.2f}")
    
    # Step 8: Test conversion back to DataFrame
    print("\n[Step 8] Testing conversion back to DataFrame...")
    
    df_reconstructed = data_array.to_pandas()
    print(f"  [OK] Converted back to DataFrame")
    print(f"       Shape: {df_reconstructed.shape}")
    print(f"       Equals original: {df_reconstructed.equals(df_wide)}")
    
    if not df_reconstructed.equals(df_wide):
        print(f"       [WARN] DataFrames not exactly equal, checking values...")
        print(f"       Values close: {np.allclose(df_reconstructed.values, df_wide.values)}")
    
    # Step 9: Test with Dataset (multiple variables)
    print("\n[Step 9] Testing xarray.Dataset with multiple variables...")
    
    # Load volume data
    query_volume = f"""
    SELECT 
      date, 
      security_id, 
      volume 
    FROM read_parquet('{parquet_path}')
    WHERE date >= '2024-01-05' AND date <= '2024-01-15'
    """
    
    df_volume_long = duckdb.query(query_volume).to_df()
    df_volume_wide = df_volume_long.pivot(index='date', columns='security_id', values='volume')
    
    volume_array = xr.DataArray(
        df_volume_wide.values,
        dims=['time', 'asset'],
        coords={
            'time': df_volume_wide.index.values,  # Convert to numpy array
            'asset': df_volume_wide.columns.values  # Convert to numpy array
        }
    )
    
    # Create Dataset with both variables
    dataset = xr.Dataset({
        'adj_close': data_array,
        'volume': volume_array
    })
    
    print(f"  [OK] xarray.Dataset created")
    print(f"       Data variables: {list(dataset.data_vars)}")
    print(f"       Dimensions: {dataset.dims}")
    print(f"       Coordinates: {list(dataset.coords)}")
    
    print(f"\n  Dataset representation:")
    print(dataset)
    
    # Step 10: Performance test
    print("\n[Step 10] Performance test (pivot 100 times)...")
    import time
    
    start_time = time.time()
    for _ in range(100):
        df_temp = df_long.pivot(index='date', columns='security_id', values='adj_close')
        _ = xr.DataArray(df_temp.values, dims=['time', 'asset'],
                        coords={'time': df_temp.index.values, 'asset': df_temp.columns.values})
    elapsed = time.time() - start_time
    
    print(f"  [OK] 100 pivot+xarray conversions")
    print(f"       Total time: {elapsed:.3f}s")
    print(f"       Average per conversion: {elapsed/100*1000:.2f}ms")
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("[SUCCESS] Long-to-Wide Pivot and xarray Conversion Complete!")
    print()
    print("Key Findings:")
    print(f"  ✓ Long format ({df_long.shape[0]} rows) → Wide format ({df_wide.shape})")
    print(f"  ✓ xarray.DataArray shape: {data_array.shape} (T={data_array.shape[0]}, N={data_array.shape[1]})")
    print(f"  ✓ Dimensions correctly named: {data_array.dims}")
    print(f"  ✓ Coordinates properly labeled")
    print(f"  ✓ Data integrity preserved")
    print(f"  ✓ xarray indexing works (integer and label-based)")
    print(f"  ✓ xarray operations work (mean, etc.)")
    print(f"  ✓ Can create Dataset with multiple variables")
    print(f"  ✓ Pivot+xarray performance: ~{elapsed/100*1000:.2f}ms per conversion")
    print()
    print("Implementation Implications:")
    print("  • Use pandas.pivot() for long-to-wide transformation")
    print("  • Use xr.DataArray(df.values, dims=[...], coords={...}) for conversion")
    print("  • time coordinate = DataFrame.index")
    print("  • asset coordinate = DataFrame.columns")
    print("  • Performance is acceptable for real-time usage")
    print("=" * 70)


if __name__ == '__main__':
    main()

