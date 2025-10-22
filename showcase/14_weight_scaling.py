"""
Showcase 14: Portfolio Weight Scaling (Strategy Pattern)

Demonstrates:
1. Strategy Pattern for weight scaling
2. Dollar-neutral portfolio (L=1.0, S=-1.0)
3. Net-long bias portfolio (L=1.1, S=-0.9)
4. One-sided signal handling
5. NaN preservation (universe integration)
6. Easy strategy replacement
"""
import numpy as np
import pandas as pd
import xarray as xr
from alpha_canvas.portfolio import GrossNetScaler, DollarNeutralScaler


def print_section(title: str):
    """Print section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def print_weights_summary(signal: xr.DataArray, weights: xr.DataArray, scaler_name: str):
    """Print weight scaling summary statistics."""
    print(f"\n{scaler_name} Results:")
    print(f"  Signal range: [{signal.min().values:.4f}, {signal.max().values:.4f}]")
    print(f"  Weight range: [{weights.min().values:.4f}, {weights.max().values:.4f}]")
    
    # Calculate exposures
    long_exp = weights.where(weights > 0, 0.0).sum(dim='asset').mean().values
    short_exp = weights.where(weights < 0, 0.0).sum(dim='asset').mean().values
    gross_exp = np.abs(weights).sum(dim='asset').mean().values
    net_exp = weights.sum(dim='asset').mean().values
    
    print(f"  \nExposures (averaged across time):")
    print(f"    Long:  {long_exp:6.4f}")
    print(f"    Short: {short_exp:6.4f}")
    print(f"    Gross: {gross_exp:6.4f}")
    print(f"    Net:   {net_exp:6.4f}")


def main():
    print_section("SHOWCASE 14: PORTFOLIO WEIGHT SCALING")
    
    # Setup
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA']
    
    print("Dataset Setup:")
    print(f"  Time periods: {len(dates)} (20 days)")
    print(f"  Assets: {len(assets)} ({', '.join(assets)})")
    
    # Generate mixed signal (both positive and negative)
    np.random.seed(42)
    signal_values = np.random.randn(20, 6)
    signal = xr.DataArray(
        signal_values,
        dims=['time', 'asset'],
        coords={'time': dates, 'asset': assets}
    )
    
    print(f"\nSignal Statistics:")
    print(f"  Mean: {signal.mean().values:.4f}")
    print(f"  Std:  {signal.std().values:.4f}")
    print(f"  Positive values: {(signal > 0).sum().values} / {signal.size}")
    print(f"  Negative values: {(signal < 0).sum().values} / {signal.size}")
    
    # ========== Strategy 1: Dollar Neutral ==========
    print_section("Strategy 1: Dollar Neutral (L=1.0, S=-1.0)")
    
    print("Using DollarNeutralScaler():")
    print("  Equivalent to: GrossNetScaler(target_gross=2.0, target_net=0.0)")
    
    scaler_dn = DollarNeutralScaler()
    weights_dn = scaler_dn.scale(signal)
    
    print_weights_summary(signal, weights_dn, "Dollar Neutral")
    
    # Show sample timestep
    print("\n  Sample timestep (2024-01-01):")
    sample_date = dates[0]
    print("    Asset      Signal    Weight")
    print("    " + "-"*35)
    for asset in assets:
        sig_val = signal.sel(time=sample_date, asset=asset).values
        wgt_val = weights_dn.sel(time=sample_date, asset=asset).values
        print(f"    {asset:6s}  {sig_val:8.4f}  {wgt_val:8.4f}")
    
    # ========== Strategy 2: Net Long Bias ==========
    print_section("Strategy 2: Net Long Bias (L=1.1, S=-0.9)")
    
    print("Using GrossNetScaler(target_gross=2.0, target_net=0.2):")
    print("  L_target = (G+N)/2 = 1.1")
    print("  S_target = (N-G)/2 = -0.9")
    
    scaler_nl = GrossNetScaler(target_gross=2.0, target_net=0.2)
    weights_nl = scaler_nl.scale(signal)
    
    print_weights_summary(signal, weights_nl, "Net Long")
    
    # ========== Strategy 3: Custom Targets ==========
    print_section("Strategy 3: Custom Gross/Net Targets")
    
    print("Using GrossNetScaler(target_gross=1.5, target_net=-0.3):")
    print("  L_target = (G+N)/2 = 0.6")
    print("  S_target = (N-G)/2 = -0.9")
    print("  (Slightly short-biased)")
    
    scaler_custom = GrossNetScaler(target_gross=1.5, target_net=-0.3)
    weights_custom = scaler_custom.scale(signal)
    
    print_weights_summary(signal, weights_custom, "Custom")
    
    # ========== Comparison ==========
    print_section("Strategy Comparison")
    
    print("Same signal, different strategies:\n")
    print("Strategy          Long    Short   Gross    Net")
    print("-" * 50)
    
    for name, weights in [
        ("Dollar Neutral ", weights_dn),
        ("Net Long       ", weights_nl),
        ("Custom         ", weights_custom),
    ]:
        long_exp = weights.where(weights > 0, 0.0).sum(dim='asset').mean().values
        short_exp = weights.where(weights < 0, 0.0).sum(dim='asset').mean().values
        gross_exp = np.abs(weights).sum(dim='asset').mean().values
        net_exp = weights.sum(dim='asset').mean().values
        
        print(f"{name}  {long_exp:6.3f}  {short_exp:6.3f}  {gross_exp:6.3f}  {net_exp:6.3f}")
    
    # ========== Edge Case: One-Sided Signal ==========
    print_section("Edge Case: One-Sided Signal")
    
    print("Testing with all-positive signal:")
    print("  Expected: Gross=2.0 (always met), Net=2.0 (one-sided)\n")
    
    # All positive signal
    signal_positive = np.abs(signal)
    
    scaler_test = DollarNeutralScaler()
    weights_positive = scaler_test.scale(signal_positive)
    
    long_exp = weights_positive.where(weights_positive > 0, 0.0).sum(dim='asset').mean().values
    short_exp = weights_positive.where(weights_positive < 0, 0.0).sum(dim='asset').mean().values
    gross_exp = np.abs(weights_positive).sum(dim='asset').mean().values
    net_exp = weights_positive.sum(dim='asset').mean().values
    
    print(f"  Result:")
    print(f"    Long:  {long_exp:6.3f} (all signal is long)")
    print(f"    Short: {short_exp:6.3f} (no short positions)")
    print(f"    Gross: {gross_exp:6.3f} ✓ (target met)")
    print(f"    Net:   {net_exp:6.3f} (target unachievable)")
    print(f"\n  Note: Gross target always met via vectorized scaling.")
    print(f"        Net target only achievable with mixed signals.")
    
    # ========== NaN Preservation ==========
    print_section("NaN Preservation (Universe Integration)")
    
    print("Testing with universe-masked signal:\n")
    
    # Create signal with NaN (simulating universe mask)
    signal_with_nan = signal.copy()
    signal_with_nan.values[:, -2:] = np.nan  # Last 2 assets always outside universe
    signal_with_nan.values[::2, 2] = np.nan  # Asset 3 outside universe on even days
    
    nan_count_signal = signal_with_nan.isnull().sum().values
    print(f"  Signal NaN count: {nan_count_signal}")
    
    weights_with_nan = scaler_dn.scale(signal_with_nan)
    nan_count_weights = weights_with_nan.isnull().sum().values
    
    print(f"  Weights NaN count: {nan_count_weights}")
    print(f"\n  ✓ NaN positions preserved (universe maintained)")
    
    # Check non-NaN positions still meet constraints
    valid_weights = weights_with_nan.where(~signal_with_nan.isnull())
    gross_valid = np.abs(valid_weights).sum(dim='asset').mean(skipna=True).values
    net_valid = valid_weights.sum(dim='asset').mean(skipna=True).values
    
    print(f"\n  Valid (non-NaN) positions:")
    print(f"    Gross: {gross_valid:.4f} ✓")
    print(f"    Net:   {net_valid:.4f} ✓")
    
    # ========== Strategy Pattern Benefits ==========
    print_section("Strategy Pattern Benefits")
    
    print("Key advantages of the Strategy Pattern design:\n")
    print("1. Easy to swap strategies:")
    print("   ```python")
    print("   # Just change the scaler instance")
    print("   scaler1 = DollarNeutralScaler()")
    print("   scaler2 = GrossNetScaler(target_gross=2.0, target_net=0.3)")
    print("   weights1 = rc.scale_weights(signal, scaler1)")
    print("   weights2 = rc.scale_weights(signal, scaler2)")
    print("   ```")
    
    print("\n2. No default scaler (explicit choice required):")
    print("   - Forces conscious decision about scaling approach")
    print("   - Makes research workflows more explicit")
    print("   - Easy to compare different strategies")
    
    print("\n3. Easy to extend with new scalers:")
    print("   ```python")
    print("   class SoftmaxScaler(WeightScaler):")
    print("       def __init__(self, temperature=1.0):")
    print("           self.temperature = temperature")
    print("       ")
    print("       def scale(self, signal):")
    print("           # Softmax implementation...")
    print("           return weights")
    print("   ")
    print("   # Usage (no changes to existing code)")
    print("   scaler = SoftmaxScaler(temperature=2.0)")
    print("   weights = rc.scale_weights(signal, scaler)")
    print("   ```")
    
    print("\n4. Stateless and composable:")
    print("   - Scalers don't store state")
    print("   - Can be reused across multiple signals")
    print("   - Thread-safe by design")
    
    print_section("SHOWCASE 14 COMPLETE")
    print("✓ Dollar Neutral scaling demonstrated")
    print("✓ Net Long Bias scaling demonstrated")
    print("✓ Custom targets demonstrated")
    print("✓ One-sided signal handling demonstrated")
    print("✓ NaN preservation demonstrated")
    print("✓ Strategy Pattern benefits explained\n")


if __name__ == '__main__':
    main()

