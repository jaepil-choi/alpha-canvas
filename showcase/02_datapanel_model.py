"""
Showcase 02: DataPanel Model

This script demonstrates the DataPanel wrapper around xarray.Dataset,
including the Open Toolkit pattern (eject/inject).

Run: poetry run python showcase/02_datapanel_model.py
"""

import numpy as np
import pandas as pd
import xarray as xr
from alpha_canvas.core.data_model import DataPanel


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print("\n" + "=" * 70)
    print("  ALPHA-CANVAS MVP SHOWCASE")
    print("  Phase 2: DataPanel Model")
    print("=" * 70)
    
    # Section 1: Create DataPanel
    print_section("1. Creating DataPanel with (T, N) Coordinates")
    time_idx = pd.date_range('2020-01-01', periods=100, freq='D')
    asset_idx = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    
    panel = DataPanel(time_idx, asset_idx)
    print(f"[OK] DataPanel created")
    print(f"     Time dimension:  {panel.db.sizes['time']} days")
    print(f"     Asset dimension: {panel.db.sizes['asset']} assets")
    print(f"     Assets: {', '.join(asset_idx)}")
    
    # Section 2: Add float data
    print_section("2. Adding Float Data (Returns)")
    returns_data = xr.DataArray(
        np.random.randn(100, 5) * 0.02,  # 2% daily volatility
        dims=['time', 'asset'],
        coords={'time': time_idx, 'asset': asset_idx}
    )
    panel.add_data('returns', returns_data)
    print("[OK] Added 'returns' (float64)")
    print(f"     Shape: {panel.db['returns'].shape}")
    print(f"     Dtype: {panel.db['returns'].dtype}")
    print(f"     Mean:  {panel.db['returns'].mean().item():.4f}")
    print(f"     Std:   {panel.db['returns'].std().item():.4f}")
    
    # Section 3: Add categorical data
    print_section("3. Adding Categorical Data (Size Labels)")
    # Assign size categories based on market cap
    size_labels = np.array(['large', 'large', 'mid', 'large', 'large'])
    size_data = xr.DataArray(
        np.tile(size_labels, (100, 1)),
        dims=['time', 'asset'],
        coords={'time': time_idx, 'asset': asset_idx}
    )
    panel.add_data('size', size_data)
    print("[OK] Added 'size' (categorical)")
    print(f"     Shape: {panel.db['size'].shape}")
    print(f"     Unique values: {list(np.unique(panel.db['size'].values))}")
    
    # Section 4: Boolean indexing
    print_section("4. Boolean Indexing (Selector Pattern)")
    large_mask = (panel.db['size'] == 'large')
    print("[OK] Created boolean mask: (panel.db['size'] == 'large')")
    print(f"     Mask shape: {large_mask.shape}")
    print(f"     Mask dtype: {large_mask.dtype}")
    print(f"     True count: {large_mask.sum().item()}")
    
    # Get returns for large caps only
    large_returns = panel.db['returns'].where(large_mask)
    print("\n[OK] Applied mask to returns")
    print(f"     Large-cap mean return: {large_returns.mean().item():.4f}")
    
    # Section 5: Eject pattern
    print_section("5. Eject Pattern (Open Toolkit)")
    pure_ds = panel.db
    print("[OK] Ejected pure xarray.Dataset")
    print(f"     Type: {type(pure_ds)}")
    print(f"     Is pure Dataset: {type(pure_ds) == xr.Dataset}")
    print(f"     Data variables: {list(pure_ds.data_vars)}")
    print("\n  Can now use with scipy, statsmodels, etc...")
    
    # Simulate external calculation
    print("\n  Simulating external calculation with numpy:")
    external_calc = np.exp(pure_ds['returns'].values)  # Exponential returns
    print(f"  [OK] Calculated exp(returns) with numpy")
    print(f"       Result shape: {external_calc.shape}")
    
    # Section 6: Inject pattern
    print_section("6. Inject Pattern (Open Toolkit)")
    # Create external DataArray (simulating scipy/statsmodels output)
    beta_values = np.random.randn(100, 5) * 0.5 + 1.0  # Beta ~ 1.0
    beta_array = xr.DataArray(
        beta_values,
        dims=['time', 'asset'],
        coords={'time': time_idx, 'asset': asset_idx}
    )
    print("[OK] Created external DataArray (simulating scipy output)")
    print(f"     Shape: {beta_array.shape}")
    print(f"     Mean beta: {beta_array.mean().item():.4f}")
    
    panel.add_data('beta', beta_array)
    print("\n[OK] Injected external data back into DataPanel")
    print(f"     'beta' now in panel: {'beta' in panel.db.data_vars}")
    
    # Section 7: Multiple data types
    print_section("7. Heterogeneous Data Types")
    print("\nDataPanel now contains:")
    for var_name in panel.db.data_vars:
        var = panel.db[var_name]
        print(f"  • {var_name:12s}  dtype: {str(var.dtype):10s}  shape: {var.shape}")
    
    # Section 8: Dimension validation
    print_section("8. Dimension Validation")
    print("\n  Testing invalid dimensions...")
    try:
        wrong_dims = xr.DataArray(
            np.random.randn(100, 5),
            dims=['wrong1', 'wrong2']
        )
        panel.add_data('wrong', wrong_dims)
        print("  [FAIL] Should have raised ValueError")
    except ValueError as e:
        print(f"  [OK] Correctly raised ValueError")
        print(f"       Message: {str(e)[:60]}...")
    
    # Final summary
    print_section("SUMMARY")
    print("[SUCCESS] DataPanel Model Demonstration Complete")
    print()
    print("Key Features Demonstrated:")
    print("  ✓ DataPanel creation with (time, asset) coordinates")
    print("  ✓ Adding float data (returns)")
    print("  ✓ Adding categorical data (size labels)")
    print("  ✓ Boolean indexing for selection")
    print("  ✓ Eject pattern (pure xarray.Dataset)")
    print("  ✓ Inject pattern (external DataArray)")
    print("  ✓ Heterogeneous data types")
    print("  ✓ Dimension validation")
    print()
    print(f"Final DataPanel Status: {len(panel.db.data_vars)} data variables")
    print(f"  {list(panel.db.data_vars)}")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()

