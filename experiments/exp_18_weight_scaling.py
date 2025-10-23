"""
Experiment 18: Portfolio Weight Scaling

Date: 2025-01-22
Status: In Progress

Objective:
- Validate GrossNetScaler mathematics: L = (G+N)/2, S = (G-N)/2
- Confirm cross-sectional independence with groupby('time').map()
- Verify NaN handling preserves universe masking
- Test edge cases (all positive, all negative, zeros, single asset)
- Measure performance characteristics

Hypothesis:
- Separate positive/negative scaling maintains constraints
- xarray.groupby('time').map() preserves (T, N) shape
- NaN values pass through scaling unchanged
- Performance overhead should be <10ms for typical sizes

Success Criteria:
- [ ] Dollar neutral: sum(long)=1.0, sum(short)=-1.0
- [ ] Net long bias: asymmetric allocation verified
- [ ] Long only: negatives become 0, sum=target
- [ ] NaN preservation: universe masking maintained
- [ ] Edge cases handled gracefully
"""

import time
import numpy as np
import xarray as xr


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_signal_preview(signal, name="Signal", show_all=False):
    """Print signal preview with first 3 timesteps or all data."""
    print(f"\n{name}:")
    print(f"  Shape: {signal.shape}")
    print(f"  Dtype: {signal.dtype}")
    
    if show_all or len(signal.time) <= 10:
        # Show all data if requested or if small dataset
        print(f"\n  Complete Data:")
        df = signal.to_pandas()
        print(df.to_string())
    else:
        # Show first 3 timesteps
        preview = signal.isel(time=slice(0, min(3, len(signal.time))))
        print(f"\n  First 3 timesteps:")
        df = preview.to_pandas()
        print(df.to_string())
    
    # Statistics
    print(f"\n  Statistics:")
    print(f"    Min: {float(signal.min()):.4f}")
    print(f"    Max: {float(signal.max()):.4f}")
    print(f"    Mean: {float(signal.mean()):.4f}")
    print(f"    NaN count: {int(signal.isnull().sum())}")


def validate_constraints(weights, target_gross=None, target_net=None, 
                        target_long=None, target_short=None, tolerance=1e-6):
    """Validate portfolio constraints with detailed output."""
    print(f"\n  Constraint Validation:")
    
    # Calculate actual values (per timestep, then average)
    long_weights = weights.where(weights > 0, 0.0)
    short_weights = weights.where(weights < 0, 0.0)
    
    actual_long = long_weights.sum(dim='asset', skipna=True)
    actual_short = short_weights.sum(dim='asset', skipna=True)
    actual_gross = np.abs(weights).sum(dim='asset', skipna=True)
    actual_net = weights.sum(dim='asset', skipna=True)
    
    # Average across time
    avg_long = float(actual_long.mean())
    avg_short = float(actual_short.mean())
    avg_gross = float(actual_gross.mean())
    avg_net = float(actual_net.mean())
    
    print(f"    Actual Long:  {avg_long:.6f}")
    print(f"    Actual Short: {avg_short:.6f}")
    print(f"    Actual Gross: {avg_gross:.6f}")
    print(f"    Actual Net:   {avg_net:.6f}")
    
    # Check targets
    success = True
    if target_long is not None:
        diff = abs(avg_long - target_long)
        status = "✓" if diff < tolerance else "✗"
        print(f"    {status} Target Long: {target_long:.6f} (diff: {diff:.2e})")
        success = success and (diff < tolerance)
    
    if target_short is not None:
        diff = abs(avg_short - target_short)
        status = "✓" if diff < tolerance else "✗"
        print(f"    {status} Target Short: {target_short:.6f} (diff: {diff:.2e})")
        success = success and (diff < tolerance)
    
    if target_gross is not None:
        diff = abs(avg_gross - target_gross)
        status = "✓" if diff < tolerance else "✗"
        print(f"    {status} Target Gross: {target_gross:.6f} (diff: {diff:.2e})")
        success = success and (diff < tolerance)
    
    if target_net is not None:
        diff = abs(avg_net - target_net)
        status = "✓" if diff < tolerance else "✗"
        print(f"    {status} Target Net: {target_net:.6f} (diff: {diff:.2e})")
        success = success and (diff < tolerance)
    
    return success


def scale_grossnet(signal: xr.DataArray, target_gross: float, target_net: float) -> xr.DataArray:
    """
    Fully vectorized GrossNetScaler - NO ITERATION!
    
    Uses unified framework:
        L_target = (G + N) / 2
        S_target = (G - N) / 2
    
    Key innovation: Always meet gross target via scaling, even for one-sided signals.
    Net target may be unachievable for all-positive or all-negative rows.
    
    TODO: In production, emit WARNING when net target is unachievable (one-sided rows).
    """
    # Calculate targets
    L_target = (target_gross + target_net) / 2.0
    S_target = (target_net - target_gross) / 2.0  # Negative value
    
    print(f"\n  Target Calculation:")
    print(f"    Gross (G) = {target_gross}")
    print(f"    Net (N)   = {target_net}")
    print(f"    L_target  = (G+N)/2 = {L_target:.6f}")
    print(f"    S_target  = (N-G)/2 = {S_target:.6f}")
    
    # Step 1: Separate positive and negative (vectorized)
    s_pos = signal.where(signal > 0, 0.0)
    s_neg = signal.where(signal < 0, 0.0)
    
    # Step 2: Sum along asset dimension (vectorized)
    sum_pos = s_pos.sum(dim='asset', skipna=True)  # Shape: (time,)
    sum_neg = s_neg.sum(dim='asset', skipna=True)  # Shape: (time,), negative values
    
    # Step 3: Normalize (vectorized, handles 0/0 → nan → 0)
    norm_pos = (s_pos / sum_pos).fillna(0.0)
    # For negatives: use absolute value of sum to avoid sign issues
    norm_neg_abs = (np.abs(s_neg) / np.abs(sum_neg)).fillna(0.0)
    
    # Step 4: Apply L/S targets (vectorized)
    weights_long = norm_pos * L_target
    weights_short_mag = norm_neg_abs * np.abs(S_target)  # Positive magnitudes
    
    # Step 5: Combine (subtract to make short side negative)
    weights = weights_long - weights_short_mag
    
    # Step 6: Calculate actual gross per row (vectorized)
    actual_gross = np.abs(weights).sum(dim='asset', skipna=True)  # Shape: (time,)
    
    # Step 7: Scale to meet target gross (vectorized)
    # Use xr.where to avoid inf from division by zero
    scale_factor = xr.where(actual_gross > 0, target_gross / actual_gross, 1.0)
    
    final_weights = weights * scale_factor
    
    # Step 8: Convert computational NaN to 0 (all-zero signals) BEFORE universe mask
    final_weights = final_weights.fillna(0.0)
    
    # Step 9: Apply universe mask (preserve NaN where signal was NaN)
    final_weights = final_weights.where(~signal.isnull())
    
    return final_weights


def scale_longonly(signal: xr.DataArray, target_long: float) -> xr.DataArray:
    """Long-only scaler implementation for experiment."""
    
    def scale_single_period(signal_slice: xr.DataArray) -> xr.DataArray:
        # Only keep positive values
        s_pos = signal_slice.where(signal_slice > 0, 0.0)
        sum_pos = s_pos.sum(skipna=True)
        
        if sum_pos > 0:
            weights = (s_pos / sum_pos) * target_long
        else:
            weights = xr.zeros_like(signal_slice)
        
        # Preserve NaN
        return weights.where(~signal_slice.isnull())
    
    result = signal.groupby('time').map(scale_single_period)
    return result


def scenario_1_dollar_neutral():
    """Scenario 1: Dollar neutral (G=2.0, N=0.0 → L=1.0, S=-1.0)."""
    print_section("Scenario 1: Dollar Neutral Portfolio")
    
    print("\nObjective: G=2.0, N=0.0 → sum(long)=1.0, sum(short)=-1.0")
    
    # Create mixed positive/negative signals
    np.random.seed(42)
    T, N = 10, 6
    times = pd.date_range('2024-01-01', periods=T, freq='D')
    assets = [f'ASSET_{i}' for i in range(N)]
    
    # Mixed signals: some positive, some negative
    signal_values = np.random.randn(T, N) * 0.5  # Mean 0, mix of +/-
    signal = xr.DataArray(
        signal_values,
        dims=['time', 'asset'],
        coords={'time': times, 'asset': assets}
    )
    
    print_signal_preview(signal, "Input Signal")
    
    # Apply GrossNetScaler
    print("\nApplying GrossNetScaler(target_gross=2.0, target_net=0.0)...")
    start_time = time.time()
    weights = scale_grossnet(signal, target_gross=2.0, target_net=0.0)
    elapsed = time.time() - start_time
    
    print(f"\n  ✓ Scaling completed in {elapsed*1000:.3f}ms")
    
    print_signal_preview(weights, "Output Weights", show_all=True)
    
    # Validate
    success = validate_constraints(
        weights, 
        target_gross=2.0, 
        target_net=0.0,
        target_long=1.0,
        target_short=-1.0
    )
    
    if success:
        print("\n  ✓✓✓ SCENARIO 1: SUCCESS ✓✓✓")
    else:
        print("\n  ✗✗✗ SCENARIO 1: FAILURE ✗✗✗")
    
    return success


def scenario_2_net_long_bias():
    """Scenario 2: Net long bias (G=2.0, N=0.2 → L=1.1, S=-0.9)."""
    print_section("Scenario 2: Net Long Bias Portfolio")
    
    print("\nObjective: G=2.0, N=0.2 → sum(long)=1.1, sum(short)=-0.9")
    
    # Create mixed signals
    np.random.seed(43)
    T, N = 10, 6
    times = pd.date_range('2024-01-01', periods=T, freq='D')
    assets = [f'ASSET_{i}' for i in range(N)]
    
    signal_values = np.random.randn(T, N) * 0.5
    signal = xr.DataArray(
        signal_values,
        dims=['time', 'asset'],
        coords={'time': times, 'asset': assets}
    )
    
    print_signal_preview(signal, "Input Signal")
    
    # Apply GrossNetScaler with net bias
    print("\nApplying GrossNetScaler(target_gross=2.0, target_net=0.2)...")
    start_time = time.time()
    weights = scale_grossnet(signal, target_gross=2.0, target_net=0.2)
    elapsed = time.time() - start_time
    
    print(f"\n  ✓ Scaling completed in {elapsed*1000:.3f}ms")
    
    print_signal_preview(weights, "Output Weights", show_all=True)
    
    # Validate asymmetric allocation
    success = validate_constraints(
        weights,
        target_gross=2.0,
        target_net=0.2,
        target_long=1.1,
        target_short=-0.9
    )
    
    if success:
        print("\n  ✓✓✓ SCENARIO 2: SUCCESS ✓✓✓")
    else:
        print("\n  ✗✗✗ SCENARIO 2: FAILURE ✗✗✗")
    
    return success


def scenario_3_long_only():
    """Scenario 3: Long only portfolio."""
    print_section("Scenario 3: Long Only Portfolio")
    
    print("\nObjective: Ignore negatives, sum(weights) = 1.0 (per timestep with positives)")
    
    # Create mixed signals (some negative will be ignored)
    np.random.seed(44)
    T, N = 10, 6
    times = pd.date_range('2024-01-01', periods=T, freq='D')
    assets = [f'ASSET_{i}' for i in range(N)]
    
    signal_values = np.random.randn(T, N) * 0.5
    signal = xr.DataArray(
        signal_values,
        dims=['time', 'asset'],
        coords={'time': times, 'asset': assets}
    )
    
    print_signal_preview(signal, "Input Signal")
    
    # Apply LongOnlyScaler
    print("\nApplying LongOnlyScaler(target_long=1.0)...")
    start_time = time.time()
    weights = scale_longonly(signal, target_long=1.0)
    elapsed = time.time() - start_time
    
    print(f"\n  ✓ Scaling completed in {elapsed*1000:.3f}ms")
    
    print_signal_preview(weights, "Output Weights", show_all=True)
    
    # Check all weights >= 0
    min_weight = float(weights.min())
    all_non_negative = min_weight >= 0
    
    print(f"\n  Additional Checks:")
    status = "✓" if all_non_negative else "✗"
    print(f"    {status} All weights >= 0 (min: {min_weight:.6f})")
    
    # Check per-timestep sums for timesteps with positive signals
    per_timestep_long = weights.sum(dim='asset', skipna=True)
    timesteps_with_positives = per_timestep_long > 0
    valid_timestep_sums = per_timestep_long.where(timesteps_with_positives, drop=True)
    
    if len(valid_timestep_sums) > 0:
        all_sum_to_one = np.allclose(valid_timestep_sums.values, 1.0, atol=1e-6)
        status = "✓" if all_sum_to_one else "✗"
        print(f"    {status} Each timestep with positives sums to 1.0")
        print(f"    Timesteps with positive signals: {int(timesteps_with_positives.sum())}/{T}")
    else:
        all_sum_to_one = False
        print(f"    ✗ No timesteps with positive signals")
    
    success = all_non_negative and all_sum_to_one
    
    if success:
        print("\n  ✓✓✓ SCENARIO 3: SUCCESS ✓✓✓")
    else:
        print("\n  ✗✗✗ SCENARIO 3: FAILURE ✗✗✗")
    
    return success


def scenario_4_nan_preservation():
    """Scenario 4: NaN preservation (universe masking)."""
    print_section("Scenario 4: NaN Preservation (Universe Masking)")
    
    print("\nObjective: NaN in signal → NaN in weights (universe maintained)")
    
    # Create signal with NaN values (simulating universe mask)
    np.random.seed(45)
    T, N = 10, 6
    times = pd.date_range('2024-01-01', periods=T, freq='D')
    assets = [f'ASSET_{i}' for i in range(N)]
    
    signal_values = np.random.randn(T, N) * 0.5
    
    # Mask out some positions (universe exclusion)
    # Assets 0, 1: always in universe
    # Assets 2, 3: intermittent (50% of time)
    # Assets 4, 5: mostly out (80% NaN)
    mask = np.random.rand(T, N) > 0.3
    mask[:, 0:2] = True  # Always in
    mask[:, 4:6] = np.random.rand(T, 2) > 0.8  # Mostly out
    
    signal_values = np.where(mask, signal_values, np.nan)
    
    signal = xr.DataArray(
        signal_values,
        dims=['time', 'asset'],
        coords={'time': times, 'asset': assets}
    )
    
    print_signal_preview(signal, "Input Signal (with NaN)")
    
    # Apply GrossNetScaler
    print("\nApplying GrossNetScaler(target_gross=2.0, target_net=0.0)...")
    weights = scale_grossnet(signal, target_gross=2.0, target_net=0.0)
    
    print_signal_preview(weights, "Output Weights (with NaN)", show_all=True)
    
    # Validate NaN preservation
    signal_nan_mask = signal.isnull()
    weights_nan_mask = weights.isnull()
    
    nan_preserved = (signal_nan_mask == weights_nan_mask).all()
    
    print(f"\n  NaN Preservation Check:")
    status = "✓" if nan_preserved else "✗"
    print(f"    {status} NaN positions match: {bool(nan_preserved)}")
    print(f"    Signal NaN count: {int(signal_nan_mask.sum())}")
    print(f"    Weights NaN count: {int(weights_nan_mask.sum())}")
    
    # For universe-masked data, check per-timestep constraints
    # (aggregates won't match due to varying data availability)
    print(f"\n  Per-Timestep Constraint Validation:")
    per_timestep_success = True
    for t in range(min(3, len(weights.time))):  # Check first 3 timesteps
        time_weights = weights.isel(time=t)
        valid_mask = ~time_weights.isnull()
        
        if valid_mask.sum() == 0:
            print(f"    Time {t}: No valid data (all NaN)")
            continue
        
        long_sum = float(time_weights.where(time_weights > 0, 0.0).sum())
        short_sum = float(time_weights.where(time_weights < 0, 0.0).sum())
        gross_sum = float(np.abs(time_weights).sum())
        net_sum = float(time_weights.sum())
        
        # Check if constraints are met (allowing for partial data)
        # When data is sparse, we still scale to targets
        has_positives = long_sum > 0
        has_negatives = short_sum < 0
        
        if has_positives and has_negatives:
            # Full dollar-neutral expected
            long_ok = abs(long_sum - 1.0) < 1e-6
            short_ok = abs(short_sum + 1.0) < 1e-6
            print(f"    Time {t}: L={long_sum:.3f} S={short_sum:.3f} " +
                  f"({'✓' if long_ok and short_ok else '✗'})")
            per_timestep_success = per_timestep_success and long_ok and short_ok
        else:
            # Partial data - just report
            print(f"    Time {t}: L={long_sum:.3f} S={short_sum:.3f} (partial data)")
    
    success = bool(nan_preserved) and per_timestep_success
    
    if success:
        print("\n  ✓✓✓ SCENARIO 4: SUCCESS ✓✓✓")
    else:
        print("\n  ✗✗✗ SCENARIO 4: FAILURE ✗✗✗")
    
    return success


def scenario_5_edge_cases():
    """Scenario 5: Edge cases."""
    print_section("Scenario 5: Edge Cases")
    
    all_success = True
    
    # Edge Case 5a: All positive signals (fallback to unit normalization)
    print("\n[5a] All Positive Signals (Fallback Behavior):")
    np.random.seed(46)
    T, N = 5, 4
    times = pd.date_range('2024-01-01', periods=T, freq='D')
    assets = [f'ASSET_{i}' for i in range(N)]
    
    signal_values = np.abs(np.random.randn(T, N)) + 0.1  # All positive
    signal = xr.DataArray(
        signal_values,
        dims=['time', 'asset'],
        coords={'time': times, 'asset': assets}
    )
    
    print(f"  Signal range: [{float(signal.min()):.4f}, {float(signal.max()):.4f}]")
    print(f"  Expected: Gross=2.0 (always met), Net=2.0 (one-sided signal)")
    
    weights = scale_grossnet(signal, target_gross=2.0, target_net=0.0)
    
    print(f"\n  Weights:")
    print(weights.to_pandas().to_string())
    
    # Check that gross target is met
    avg_gross = float(np.abs(weights).sum(dim='asset').mean())
    avg_net = float(weights.sum(dim='asset').mean())
    
    print(f"\n  Result: Gross={avg_gross:.6f}, Net={avg_net:.6f}")
    print(f"  Note: Gross target always met, Net target unachievable (one-sided)")
    
    case_5a_success = abs(avg_gross - 2.0) < 1e-6
    status = "✓" if case_5a_success else "✗"
    print(f"  {status} Case 5a: {'PASS' if case_5a_success else 'FAIL'}")
    
    all_success = all_success and case_5a_success
    
    # Edge Case 5b: All negative signals (fallback to unit normalization)
    print("\n[5b] All Negative Signals (Fallback Behavior):")
    signal_values = -np.abs(np.random.randn(T, N)) - 0.1  # All negative
    signal = xr.DataArray(
        signal_values,
        dims=['time', 'asset'],
        coords={'time': times, 'asset': assets}
    )
    
    print(f"  Signal range: [{float(signal.min()):.4f}, {float(signal.max()):.4f}]")
    print(f"  Expected: Gross=2.0 (always met), Net=-2.0 (one-sided signal)")
    
    weights = scale_grossnet(signal, target_gross=2.0, target_net=0.0)
    
    print(f"\n  Weights:")
    print(weights.to_pandas().to_string())
    
    # Check that gross target is met
    avg_gross = float(np.abs(weights).sum(dim='asset').mean())
    avg_net = float(weights.sum(dim='asset').mean())
    
    print(f"\n  Result: Gross={avg_gross:.6f}, Net={avg_net:.6f}")
    print(f"  Note: Gross target always met, Net target unachievable (one-sided)")
    
    case_5b_success = abs(avg_gross - 2.0) < 1e-6
    status = "✓" if case_5b_success else "✗"
    print(f"  {status} Case 5b: {'PASS' if case_5b_success else 'FAIL'}")
    
    all_success = all_success and case_5b_success
    
    # Edge Case 5c: Single valid asset per timestep
    print("\n[5c] Single Valid Asset Per Timestep:")
    signal_values = np.full((T, N), np.nan)
    for t in range(T):
        signal_values[t, t % N] = np.random.randn() * 0.5  # One valid per time
    
    signal = xr.DataArray(
        signal_values,
        dims=['time', 'asset'],
        coords={'time': times, 'asset': assets}
    )
    
    print(f"  Valid assets per timestep: 1")
    print(f"  Total valid: {int((~signal.isnull()).sum())}")
    
    weights = scale_grossnet(signal, target_gross=2.0, target_net=0.0)
    
    print(f"\n  Weights:")
    print(weights.to_pandas().to_string())
    
    # Each timestep should have single position = +1 or -1
    case_5c_success = True
    for t in range(T):
        time_weights = weights.isel(time=t)
        valid_count = int((~time_weights.isnull()).sum())
        if valid_count != 1:
            case_5c_success = False
            print(f"  ✗ Time {t}: Expected 1 valid, got {valid_count}")
    
    status = "✓" if case_5c_success else "✗"
    print(f"  {status} Case 5c: {'PASS' if case_5c_success else 'FAIL'}")
    
    all_success = all_success and case_5c_success
    
    # Edge Case 5d: All zeros
    print("\n[5d] All Zero Signals:")
    signal_values = np.zeros((T, N))
    signal = xr.DataArray(
        signal_values,
        dims=['time', 'asset'],
        coords={'time': times, 'asset': assets}
    )
    
    weights = scale_grossnet(signal, target_gross=2.0, target_net=0.0)
    
    print(f"\n  Weights:")
    print(weights.to_pandas().to_string())
    
    # Check that all weights are exactly 0 (not NaN)
    all_zeros = (weights == 0.0).all()
    no_nans = (~weights.isnull()).all()
    
    print(f"\n  Result:")
    print(f"    All weights zero: {bool(all_zeros)}")
    print(f"    No NaN values: {bool(no_nans)}")
    
    case_5d_success = bool(all_zeros) and bool(no_nans)
    status = "✓" if case_5d_success else "✗"
    print(f"  {status} Case 5d: {'PASS' if case_5d_success else 'FAIL'}")
    
    all_success = all_success and case_5d_success
    
    if all_success:
        print("\n  ✓✓✓ SCENARIO 5: ALL EDGE CASES PASSED ✓✓✓")
    else:
        print("\n  ✗✗✗ SCENARIO 5: SOME EDGE CASES FAILED ✗✗✗")
    
    return all_success


def scenario_6_vectorized_edge_cases():
    """Scenario 6: Specific edge cases with vectorized approach."""
    print_section("Scenario 6: Vectorized Edge Cases (5 Specific Rows)")
    
    print("\nObjective: Validate vectorized gross-scaling with mixed edge cases")
    print("  Row 0: [3, 5, 7, 6] - All positive")
    print("  Row 1: [3, -6, 9, 0] - Mixed")
    print("  Row 2: [3, -6, 9, -4] - Mixed") 
    print("  Row 3: [-2, -5, -1, -9] - All negative")
    print("  Row 4: [0, 0, 0, 0] - All zeros")
    
    # Create specific test data
    np.random.seed(50)
    signal_values = np.array([
        [3.0,  5.0,  7.0,  6.0],
        [3.0, -6.0,  9.0,  0.0],
        [3.0, -6.0,  9.0, -4.0],
        [-2.0, -5.0, -1.0, -9.0],
        [0.0,  0.0,  0.0,  0.0],
    ])
    
    times = pd.date_range('2024-01-01', periods=5, freq='D')
    assets = ['ASSET_A', 'ASSET_B', 'ASSET_C', 'ASSET_D']
    
    signal = xr.DataArray(
        signal_values,
        dims=['time', 'asset'],
        coords={'time': times, 'asset': assets}
    )
    
    print_signal_preview(signal, "Input Signal")
    
    # Apply GrossNetScaler with challenging targets
    print("\nApplying GrossNetScaler(target_gross=2.2, target_net=-0.5)...")
    print("Expected:")
    print("  - All rows: Gross = 2.2 ✓")
    print("  - Mixed rows (1, 2): Net = -0.5 ✓")
    print("  - One-sided rows (0, 3): Net ≠ -0.5 (unavoidable)")
    print("  - All-zeros row (4): Weights = 0")
    
    start_time = time.time()
    weights = scale_grossnet(signal, target_gross=2.2, target_net=-0.5)
    elapsed = time.time() - start_time
    
    print(f"\n  ✓ Scaling completed in {elapsed*1000:.3f}ms (vectorized!)")
    
    print("\nFinal Weights (All 5 Rows):")
    print(weights.to_pandas().to_string())
    
    # Validate per-row constraints
    print(f"\n  Per-Row Validation:")
    success = True
    
    for t in range(len(signal.time)):
        row_weights = weights.isel(time=t)
        
        long_sum = float(row_weights.where(row_weights > 0, 0.0).sum())
        short_sum = float(row_weights.where(row_weights < 0, 0.0).sum())
        net_sum = float(row_weights.sum())
        gross_sum = float(np.abs(row_weights).sum())
        
        # Determine if row is mixed
        has_positive = long_sum > 1e-6
        has_negative = short_sum < -1e-6
        is_mixed = has_positive and has_negative
        is_zeros = gross_sum < 1e-6
        
        # Check constraints
        if not is_zeros:
            gross_ok = abs(gross_sum - 2.2) < 1e-6
            success = success and gross_ok
            gross_status = "✓" if gross_ok else "✗"
        else:
            gross_ok = True
            gross_status = "✓"
        
        if is_mixed:
            net_ok = abs(net_sum + 0.5) < 1e-6
            success = success and net_ok
            net_status = "✓" if net_ok else "✗"
        else:
            net_status = "n/a"
        
        row_type = "mixed" if is_mixed else ("zeros" if is_zeros else "one-sided")
        
        print(f"    Row {t} ({row_type:10s}): "
              f"L={long_sum:6.3f}, S={short_sum:6.3f}, "
              f"Net={net_sum:6.3f} {net_status:>3s}, "
              f"Gross={gross_sum:6.3f} {gross_status:>3s}")
    
    if success:
        print("\n  ✓✓✓ SCENARIO 6: SUCCESS ✓✓✓")
    else:
        print("\n  ✗✗✗ SCENARIO 6: FAILURE ✗✗✗")
    
    return success


def performance_benchmark():
    """Performance benchmark for typical dataset sizes."""
    print_section("Performance Benchmark")
    
    print("\nTesting scaling performance for various dataset sizes:")
    
    sizes = [
        (10, 6, "Small (10×6)"),
        (100, 50, "Medium (100×50)"),
        (252, 100, "1Y Daily (252×100)"),
        (1000, 500, "Large (1000×500)")
    ]
    
    for T, N, description in sizes:
        np.random.seed(42)
        times = pd.date_range('2024-01-01', periods=T, freq='D')
        assets = [f'ASSET_{i}' for i in range(N)]
        
        signal_values = np.random.randn(T, N) * 0.5
        signal = xr.DataArray(
            signal_values,
            dims=['time', 'asset'],
            coords={'time': times, 'asset': assets}
        )
        
        # Warm-up
        _ = scale_grossnet(signal, target_gross=2.0, target_net=0.0)
        
        # Benchmark
        iterations = 10
        start_time = time.time()
        for _ in range(iterations):
            _ = scale_grossnet(signal, target_gross=2.0, target_net=0.0)
        elapsed = (time.time() - start_time) / iterations
        
        print(f"  {description:20s}: {elapsed*1000:7.3f}ms per scale")
    
    print("\n  ✓ Performance benchmark complete")


def main():
    """Main experiment execution."""
    print("="*70)
    print("  EXPERIMENT 18: PORTFOLIO WEIGHT SCALING")
    print("="*70)
    print("\nValidating GrossNetScaler mathematics and implementation patterns")
    
    results = []
    
    # Run all scenarios
    results.append(("Scenario 1: Dollar Neutral", scenario_1_dollar_neutral()))
    results.append(("Scenario 2: Net Long Bias", scenario_2_net_long_bias()))
    results.append(("Scenario 3: Long Only", scenario_3_long_only()))
    results.append(("Scenario 4: NaN Preservation", scenario_4_nan_preservation()))
    results.append(("Scenario 5: Edge Cases", scenario_5_edge_cases()))
    results.append(("Scenario 6: Vectorized Edge Cases", scenario_6_vectorized_edge_cases()))
    
    # Performance
    performance_benchmark()
    
    # Summary
    print_section("EXPERIMENT SUMMARY")
    
    print("\nResults:")
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status:8s} - {name}")
    
    all_passed = all(success for _, success in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("  ✓✓✓ ALL SCENARIOS PASSED ✓✓✓")
        print("  Experiment validated: Ready for TDD implementation")
    else:
        print("  ✗✗✗ SOME SCENARIOS FAILED ✗✗✗")
        print("  Review failures before proceeding")
    print("="*70)
    
    return all_passed


if __name__ == '__main__':
    import pandas as pd
    success = main()
    exit(0 if success else 1)

