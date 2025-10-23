"""
Experiment 02: xarray.Dataset as DataPanel

Date: 2025-01-20
Status: In Progress

Objective:
- Validate that xarray.Dataset can serve as our (T, N) data container

Hypothesis:
- Dataset can store heterogeneous types and support eject/inject pattern

Success Criteria:
- [ ] Create Dataset with (time, asset) coordinates
- [ ] Add float data_var (e.g., 'returns')
- [ ] Add string data_var (e.g., 'size' labels)
- [ ] Eject: Extract pure Dataset and verify type
- [ ] Inject: Add external DataArray back
- [ ] Boolean indexing produces correct mask
"""

import numpy as np
import pandas as pd
import xarray as xr


def main():
    print("=" * 60)
    print("EXPERIMENT 02: DataPanel Model Validation")
    print("=" * 60)
    
    # Step 1: Create Dataset with coords
    print("\n[Step 1] Creating Dataset with coords...")
    time_idx = pd.date_range('2020-01-01', periods=100)
    asset_idx = [f'ASSET_{i}' for i in range(50)]
    
    ds = xr.Dataset(
        coords={
            'time': time_idx,
            'asset': asset_idx
        }
    )
    
    print(f"  [OK] Dataset created with dims: {dict(ds.dims)}")
    print(f"  [OK] Coords: time ({len(ds.coords['time'])},), asset ({len(ds.coords['asset'])},)")
    
    # Step 2: Add heterogeneous data_vars
    print("\n[Step 2] Adding heterogeneous data_vars...")
    
    # Add float data
    returns_data = xr.DataArray(
        np.random.randn(100, 50),
        dims=['time', 'asset'],
        coords={'time': time_idx, 'asset': asset_idx}
    )
    ds = ds.assign({'returns': returns_data})
    print(f"  [OK] Added 'returns' ({ds['returns'].dtype}) shape {ds['returns'].shape}")
    
    # Add string/categorical data
    size_labels = np.random.choice(['small', 'big'], size=(100, 50))
    size_data = xr.DataArray(
        size_labels,
        dims=['time', 'asset'],
        coords={'time': time_idx, 'asset': asset_idx}
    )
    ds = ds.assign({'size': size_data})
    unique_labels = np.unique(size_labels)
    print(f"  [OK] Added 'size' ({ds['size'].dtype}) shape {ds['size'].shape} with labels {list(unique_labels)}")
    
    # Step 3: Test Eject pattern
    print("\n[Step 3] Testing Eject pattern...")
    ejected_ds = ds
    print(f"  [OK] Ejected ds type: {type(ejected_ds)}")
    
    # Verify we can access with standard xarray
    try:
        _ = ejected_ds['returns']
        print(f"  [OK] Can access with standard xarray: ds['returns']")
    except Exception as e:
        print(f"  [FAIL] Cannot access: {e}")
        return
    
    # Step 4: Test boolean indexing
    print("\n[Step 4] Testing boolean indexing...")
    mask = (ds['size'] == 'small')
    print(f"  [OK] (ds['size'] == 'small') -> shape {mask.shape}, dtype: {mask.dtype}")
    
    true_count = mask.sum().item()
    total_count = mask.size
    true_pct = (true_count / total_count) * 100
    print(f"  [OK] Mask has {true_pct:.1f}% True values ({true_count}/{total_count})")
    
    # Step 5: Test Inject pattern
    print("\n[Step 5] Testing Inject pattern...")
    
    # Create external DataArray with numpy
    beta_values = np.random.randn(100, 50) * 0.5 + 1.0
    beta_array = xr.DataArray(
        beta_values,
        dims=['time', 'asset'],
        coords={'time': time_idx, 'asset': asset_idx}
    )
    print(f"  [OK] Created external DataArray 'beta' with numpy")
    
    # Inject via assign
    ds = ds.assign({'beta': beta_array})
    print(f"  [OK] Injected via ds.assign({{'beta': beta_array}})")
    
    # Verify accessible
    try:
        _ = ds['beta']
        print(f"  [OK] ds['beta'] accessible, shape: {ds['beta'].shape}")
    except Exception as e:
        print(f"  [FAIL] Cannot access beta: {e}")
        return
    
    # Step 6: Validate dimensions consistency
    print("\n[Step 6] Validating dimensions consistency...")
    all_vars_consistent = True
    for var_name in ds.data_vars:
        if set(ds[var_name].dims) != {'time', 'asset'}:
            print(f"  [FAIL] Variable '{var_name}' has inconsistent dims: {ds[var_name].dims}")
            all_vars_consistent = False
    
    if all_vars_consistent:
        print(f"  [OK] All data_vars have consistent dims: ['time', 'asset']")
    
    # Verdict
    print("\n" + "=" * 60)
    if all_vars_consistent:
        print("VERDICT: [SUCCESS] - xarray.Dataset is viable DataPanel model")
    else:
        print("VERDICT: [FAILURE] - Dimension consistency issues")
    print("=" * 60)


if __name__ == '__main__':
    main()


