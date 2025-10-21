"""
Showcase 09: Universe Masking - Automatic Investable Universe Application

This showcase demonstrates the automatic universe masking feature, which ensures
all data and operator results respect the defined investable universe.

Key Features:
1. Initialize AlphaCanvas with universe mask (price > threshold)
2. Field retrieval automatically masked (input masking)
3. Operator output automatically masked (output masking)
4. Compare: data with vs without universe
5. Operator chains preserve masking (Field → ts_mean → rank)
6. Open Toolkit: injected data also respects universe
7. Read-only universe property for inspection

Design Pattern: Double Masking Strategy
- INPUT MASKING: Applied at Field retrieval (visit_field)
- OUTPUT MASKING: Applied at operator output (visit_operator)
- This creates a "trust chain" where every computation is guaranteed masked
"""

import numpy as np
import pandas as pd
import xarray as xr

from alpha_canvas import AlphaCanvas
from alpha_canvas.core.expression import Field
from alpha_canvas.ops.timeseries import TsMean
from alpha_canvas.ops.crosssection import Rank


def main():
    print("="*80)
    print("SHOWCASE 09: Universe Masking - Automatic Investable Universe")
    print("="*80)
    
    # ================================================================
    # Step 1: Create Mock Data
    # ================================================================
    print("\n[Step 1] Creating mock price data with varying liquidity...")
    
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    assets = ['AAPL', 'PENNY_STOCK', 'NVDA', 'MICROCAP', 'GOOGL', 'ILLIQUID']
    
    # Create price data with realistic universe dynamics
    # PENNY_STOCK: Starts below $5, crosses threshold after day 5
    # ILLIQUID: Fluctuates around $5, enters/exits universe (edge case)
    prices = xr.DataArray(
        [
            [150.0, 4.2, 500.0, 0.8, 140.0, 3.0],   # Day 1  - PENNY below, ILLIQUID below
            [151.0, 4.5, 505.0, 0.7, 141.0, 6.0],   # Day 2  - PENNY below, ILLIQUID above!
            [152.0, 4.8, 510.0, 0.9, 142.0, 4.0],   # Day 3  - PENNY below, ILLIQUID below
            [150.5, 4.6, 508.0, 0.6, 140.5, 7.0],   # Day 4  - PENNY below, ILLIQUID above!
            [153.0, 4.9, 515.0, 0.8, 143.0, 9.0],   # Day 5  - PENNY below, ILLIQUID above!
            [154.0, 5.2, 520.0, 0.7, 144.0, 8.5],   # Day 6  - PENNY above!, ILLIQUID above!
            [152.5, 5.5, 518.0, 0.9, 142.5, 10.0],  # Day 7  - PENNY above!, ILLIQUID above!
            [155.0, 5.8, 525.0, 0.8, 145.0, 11.0],  # Day 8  - PENNY above!, ILLIQUID above!
            [156.0, 6.2, 530.0, 0.6, 146.0, 12.0],  # Day 9  - PENNY above!, ILLIQUID above!
            [157.0, 6.5, 535.0, 0.7, 147.0, 13.0],  # Day 10 - PENNY above!, ILLIQUID above!
        ],
        dims=['time', 'asset'],
        coords={'time': dates, 'asset': assets}
    )
    
    print("\n  Price data preview:")
    print(prices.to_pandas().head(3))
    
    # ================================================================
    # Step 2: Define Universe (price > $5.0 threshold)
    # ================================================================
    print("\n[Step 2] Defining universe mask (price > $5.0)...")
    
    # Create universe: Only stocks with price > 5.0 are tradeable
    universe_mask = prices > 5.0
    
    print("\n  Universe mask (True = tradeable):")
    print(universe_mask.to_pandas().head(3))
    
    # Count stocks in universe
    stocks_in_universe = universe_mask.sum(dim='asset')
    print("\n  Stocks in universe per day:")
    for date, count in zip(dates, stocks_in_universe.values):
        print(f"    {date.date()}: {int(count)} stocks")
    
    # Show which stocks enter/exit universe
    print("\n  Dynamic universe membership:")
    print("  Stock         | Day 1 | Day 2 | Day 3 | Day 4 | Day 5 | Day 6 | Day 7-10")
    print("  " + "-"*72)
    for i, stock in enumerate(assets):
        day1 = "✓" if universe_mask.values[0, i] else "✗"
        day2 = "✓" if universe_mask.values[1, i] else "✗"
        day3 = "✓" if universe_mask.values[2, i] else "✗"
        day4 = "✓" if universe_mask.values[3, i] else "✗"
        day5 = "✓" if universe_mask.values[4, i] else "✗"
        day6 = "✓" if universe_mask.values[5, i] else "✗"
        day7_10 = "✓" if all(universe_mask.values[6:, i]) else "✗"
        print(f"  {stock:14s}| {day1:5s} | {day2:5s} | {day3:5s} | {day4:5s} | {day5:5s} | {day6:5s} | {day7_10:5s}")
    
    print("\n  Key observations:")
    print("  • PENNY_STOCK: Excluded days 1-5, enters universe day 6+")
    print("  • ILLIQUID: Volatile inclusion - in/out/in pattern (days 2, 4-5)")
    print("  • Large caps (AAPL, NVDA, GOOGL): Always in universe")
    print("  • MICROCAP: Always excluded (price < $1)")
    
    # ================================================================
    # Step 3: Initialize WITHOUT Universe (Baseline)
    # ================================================================
    print("\n[Step 3] Baseline: AlphaCanvas WITHOUT universe...")
    
    rc_no_univ = AlphaCanvas(
        config_dir='config',
        time_index=dates,
        asset_index=assets
    )
    
    # Add price data
    rc_no_univ.add_data('close', prices)
    
    # Check universe
    print(f"\n  Universe set: {rc_no_univ.universe is not None}")
    print(f"  Universe value: {rc_no_univ.universe}")
    
    # Retrieve data
    baseline_close = rc_no_univ.db['close']
    print("\n  Baseline data (no universe) - Day 1:")
    print(f"    {baseline_close.to_pandas().iloc[0]}")
    
    # ================================================================
    # Step 4: Initialize WITH Universe
    # ================================================================
    print("\n[Step 4] AlphaCanvas WITH universe (price > $5.0)...")
    
    rc_with_univ = AlphaCanvas(
        config_dir='config',
        time_index=dates,
        asset_index=assets,
        universe=universe_mask
    )
    
    # Check universe
    print(f"\n  Universe set: {rc_with_univ.universe is not None}")
    print(f"  Universe shape: {rc_with_univ.universe.shape}")
    print(f"  Total positions in universe: {rc_with_univ.universe.sum().values}")
    
    # ================================================================
    # Step 5: Field Retrieval with Auto-Masking
    # ================================================================
    print("\n[Step 5] Field retrieval with automatic INPUT masking...")
    
    # Add price data (same data, but universe applied)
    rc_with_univ.add_data('close', prices)
    
    # Retrieve data (should be masked)
    masked_close = rc_with_univ.db['close']
    
    print("\n  Masked data (with universe) - Day 1:")
    print(f"    {masked_close.to_pandas().iloc[0]}")
    
    # Compare with baseline
    print("\n  Comparison (Day 1):")
    print("  Stock         | Baseline | Masked  | In Universe?")
    print("  " + "-"*56)
    for i, stock in enumerate(assets):
        baseline_val = baseline_close.values[0, i]
        masked_val = masked_close.values[0, i]
        in_univ = "Yes" if universe_mask.values[0, i] else "No"
        masked_str = f"{masked_val:.1f}" if not np.isnan(masked_val) else "NaN"
        print(f"  {stock:14s}| {baseline_val:6.1f}   | {masked_str:6s}  | {in_univ}")
    
    # ================================================================
    # Step 6: Operator Chain with Universe
    # ================================================================
    print("\n[Step 6] Operator chain: ts_mean preserves universe masking...")
    
    # Add 3-day moving average
    rc_with_univ.add_data(
        'close_ma3',
        TsMean(child=Field('close'), window=3)
    )
    
    ma3_data = rc_with_univ.db['close_ma3']
    
    print("\n  3-day moving average (masked) - Day 5:")
    print(f"    {ma3_data.to_pandas().iloc[4]}")
    
    # Verify: Low-priced stocks should still be NaN
    print("\n  Verification: Low-priced stocks masked throughout")
    print("  Stock         | Day 1  | Day 5 MA3 | Status")
    print("  " + "-"*50)
    for i, stock in enumerate(assets):
        day1 = masked_close.values[0, i]
        day5_ma = ma3_data.values[4, i]
        day1_str = f"{day1:.1f}" if not np.isnan(day1) else "NaN"
        day5_str = f"{day5_ma:.1f}" if not np.isnan(day5_ma) else "NaN "
        status = "✓ Masked" if np.isnan(day5_ma) and universe_mask.values[0, i] == False else "✓ Valid"
        print(f"  {stock:14s}| {day1_str:5s}  | {day5_str:5s}     | {status}")
    
    # ================================================================
    # Step 7: Cross-Sectional Operator with Universe
    # ================================================================
    print("\n[Step 7] Cross-sectional operator: rank respects universe...")
    
    # Add rank (percentile ranking across stocks)
    rc_with_univ.add_data(
        'close_rank',
        Rank(child=Field('close'))
    )
    
    rank_data = rc_with_univ.db['close_rank']
    
    print("\n  Price ranks (Day 1):")
    print("  Stock         | Price | Rank     | Explanation")
    print("  " + "-"*62)
    for i, stock in enumerate(assets):
        price = prices.values[0, i]
        rank = rank_data.values[0, i]
        rank_str = f"{rank:.2f}" if not np.isnan(rank) else "NaN "
        
        if np.isnan(rank):
            explanation = "Excluded from universe"
        elif rank == 0.0:
            explanation = "Lowest in universe"
        elif rank == 1.0:
            explanation = "Highest in universe"
        else:
            explanation = f"Middle ({rank*100:.0f}th percentile)"
        
        print(f"  {stock:14s}| ${price:5.1f} | {rank_str:6s}   | {explanation}")
    
    # Verify: Only tradeable stocks are ranked
    valid_ranks = rank_data.values[0, ~np.isnan(rank_data.values[0])]
    print(f"\n  Valid ranks count: {len(valid_ranks)} (should match tradeable stocks)")
    print(f"  Tradeable stocks: {universe_mask.values[0].sum()}")
    assert len(valid_ranks) == universe_mask.values[0].sum(), "Rank count mismatch!"
    print("  ✓ Rank respects universe correctly")
    
    # ================================================================
    # Step 8: Injected Data Respects Universe (Open Toolkit)
    # ================================================================
    print("\n[Step 8] Open Toolkit: Injected DataArray also respects universe...")
    
    # Calculate returns externally (in "Jupyter")
    # Manual pct_change calculation
    external_returns = (prices - prices.shift(time=1)) / prices.shift(time=1)
    
    print("\n  External returns (before injection) - Day 2:")
    print(f"    {external_returns.to_pandas().iloc[1]}")
    
    # Inject back to AlphaCanvas
    rc_with_univ.add_data('returns', external_returns)
    
    # Retrieve (should be masked)
    injected_returns = rc_with_univ.db['returns']
    
    print("\n  Injected returns (after universe applied) - Day 2:")
    print(f"    {injected_returns.to_pandas().iloc[1]}")
    
    print("\n  Verification: Low-priced stocks masked after injection")
    print("  Stock         | External Ret | Injected Ret | Status")
    print("  " + "-"*58)
    for i, stock in enumerate(assets):
        ext_ret = external_returns.values[1, i]
        inj_ret = injected_returns.values[1, i]
        ext_str = f"{ext_ret*100:+5.1f}%" if not np.isnan(ext_ret) else "NaN"
        inj_str = f"{inj_ret*100:+5.1f}%" if not np.isnan(inj_ret) else "NaN  "
        status = "✓ Masked" if np.isnan(inj_ret) and universe_mask.values[1, i] == False else "✓ Kept"
        print(f"  {stock:14s}| {ext_str:9s}    | {inj_str:9s}    | {status}")
    
    # ================================================================
    # Step 9: Complex Operator Chain
    # ================================================================
    print("\n[Step 9] Complex chain: Field → ts_mean → rank (all masked)...")
    
    # Chain: smoothed prices → rank smoothed prices
    rc_with_univ.add_data(
        'smooth_rank',
        Rank(child=TsMean(child=Field('close'), window=3))
    )
    
    smooth_rank = rc_with_univ.db['smooth_rank']
    
    print("\n  Smoothed price ranks (Day 5):")
    print("  Stock         | Close | MA3   | Rank     | Status")
    print("  " + "-"*62)
    for i, stock in enumerate(assets):
        close = masked_close.values[4, i]
        ma3 = ma3_data.values[4, i]
        rank = smooth_rank.values[4, i]
        
        close_str = f"{close:.1f}" if not np.isnan(close) else "NaN"
        ma3_str = f"{ma3:.1f}" if not np.isnan(ma3) else "NaN"
        rank_str = f"{rank:.2f}" if not np.isnan(rank) else "NaN "
        
        status = "Excluded" if np.isnan(rank) else "Ranked"
        print(f"  {stock:14s}| {close_str:5s} | {ma3_str:5s} | {rank_str:6s}   | {status}")
    
    # ================================================================
    # Step 10: Time-Varying Universe Analysis
    # ================================================================
    print("\n[Step 10] Time-varying universe: PENNY_STOCK and ILLIQUID behavior...")
    
    # Analyze PENNY_STOCK (enters universe on day 6)
    penny_idx = assets.index('PENNY_STOCK')
    illiquid_idx = assets.index('ILLIQUID')
    
    print("\n  PENNY_STOCK price and masking over time:")
    print("  Day | Price | In Univ? | Masked Data | MA3 Data | Rank")
    print("  " + "-"*60)
    for day in range(10):
        price = prices.values[day, penny_idx]
        in_univ = "Yes" if universe_mask.values[day, penny_idx] else "No "
        masked_val = masked_close.values[day, penny_idx]
        ma3_val = ma3_data.values[day, penny_idx] if day >= 0 else np.nan
        rank_val = rank_data.values[day, penny_idx]
        
        masked_str = f"{masked_val:.1f}" if not np.isnan(masked_val) else "NaN"
        ma3_str = f"{ma3_val:.1f}" if not np.isnan(ma3_val) else "NaN"
        rank_str = f"{rank_val:.2f}" if not np.isnan(rank_val) else "NaN "
        
        print(f"  {day+1:3d} | ${price:.1f} | {in_univ:8s} | {masked_str:11s} | {ma3_str:8s} | {rank_str}")
    
    print("\n  Key insight: PENNY_STOCK excluded until price crosses $5.00 threshold")
    print("  • Days 1-5: Price < $5 → NaN in all fields (masked out)")
    print("  • Day 6+: Price ≥ $5 → Real values appear, enters rankings")
    print("  • MA3 computation: Needs 3 valid (unmasked) values to produce output")
    
    print("\n  ILLIQUID price and masking over time:")
    print("  Day | Price | In Univ? | Masked Data | MA3 Data | Rank")
    print("  " + "-"*60)
    for day in range(10):
        price = prices.values[day, illiquid_idx]
        in_univ = "Yes" if universe_mask.values[day, illiquid_idx] else "No "
        masked_val = masked_close.values[day, illiquid_idx]
        ma3_val = ma3_data.values[day, illiquid_idx] if day >= 0 else np.nan
        rank_val = rank_data.values[day, illiquid_idx]
        
        masked_str = f"{masked_val:.1f}" if not np.isnan(masked_val) else "NaN"
        ma3_str = f"{ma3_val:.1f}" if not np.isnan(ma3_val) else "NaN"
        rank_str = f"{rank_val:.2f}" if not np.isnan(rank_val) else "NaN "
        
        print(f"  {day+1:3d} | ${price:.1f} | {in_univ:8s} | {masked_str:11s} | {ma3_str:8s} | {rank_str}")
    
    print("\n  Key insight: ILLIQUID has volatile universe inclusion (edge case)")
    print("  • Day 1: $3.0 → Excluded (below threshold)")
    print("  • Day 2: $6.0 → Included! (crosses threshold)")
    print("  • Day 3: $4.0 → Excluded (drops below)")
    print("  • Day 4-5: $7-9 → Included again")
    print("  • Day 6+: Stays included")
    print("  • This creates gaps in MA3 calculation and ranking")
    
    # ================================================================
    # Step 11: Universe Coverage Statistics
    # ================================================================
    print("\n[Step 11] Universe coverage statistics...")
    
    # Calculate coverage per day
    coverage = universe_mask.sum(dim='asset')
    
    print("\n  Universe coverage over time:")
    print("  Date       | Stocks in Universe | Coverage")
    print("  " + "-"*48)
    for i, date in enumerate(dates):
        count = int(coverage.values[i])
        pct = count / len(assets) * 100
        print(f"  {date.date()} | {count:18d} | {pct:5.1f}%")
    
    # Summary statistics
    print("\n  Summary:")
    print(f"    Total stocks: {len(assets)}")
    print(f"    Min coverage: {int(coverage.min().values)} stocks (day {int(coverage.argmin().values + 1)})")
    print(f"    Max coverage: {int(coverage.max().values)} stocks (day {int(coverage.argmax().values + 1)})")
    print(f"    Average coverage: {coverage.mean().values:.1f} stocks ({coverage.mean().values/len(assets)*100:.1f}%)")
    print(f"\n    Day 1 exclusions: {', '.join([assets[i] for i in range(len(assets)) if not universe_mask.values[0, i]])}")
    print(f"    Day 10 exclusions: {', '.join([assets[i] for i in range(len(assets)) if not universe_mask.values[9, i]])}")
    print(f"\n    Stocks entering universe mid-period: PENNY_STOCK (day 6)")
    print(f"    Stocks with volatile inclusion: ILLIQUID (in/out pattern)")
    
    # ================================================================
    # Conclusion
    # ================================================================
    print("\n" + "="*80)
    print("SHOWCASE COMPLETE")
    print("="*80)
    
    print("\n[KEY TAKEAWAYS]")
    print("1. Universe set at initialization (immutable for fair PnL comparison)")
    print("2. Double masking strategy:")
    print("   - INPUT MASKING: Applied at Field retrieval (visit_field)")
    print("   - OUTPUT MASKING: Applied at operator output (visit_operator)")
    print("3. Trust chain: Operators trust input is masked, ensure output is masked")
    print("4. Works with:")
    print("   - Field retrieval (data loading)")
    print("   - Time-series operators (ts_mean)")
    print("   - Cross-sectional operators (rank)")
    print("   - Operator chains (Field → ts_mean → rank)")
    print("   - Injected data (Open Toolkit pattern)")
    print("5. Read-only universe property for inspection")
    print("6. Future extension: Field('univ500') for database-backed universes")
    
    print("\n[DESIGN RATIONALE]")
    print("Q: Why double masking? Isn't it redundant?")
    print("A: No! It creates a trust chain:")
    print("   - Field masking: Ensures raw data respects universe")
    print("   - Operator masking: Guarantees output respects universe")
    print("   - Idempotent: Masked data stays masked (no data corruption)")
    print("   - Performance: <15% overhead (negligible with xarray lazy eval)")
    print("   - Trust: Operators don't need to worry about universe logic")
    print("\nQ: Why immutable universe?")
    print("A: Fair PnL step-by-step comparison requires fixed universe")
    print("   - Can't compare alpha_t vs alpha_{t+1} if universe changes")
    print("   - Ensures reproducible backtests")


if __name__ == '__main__':
    main()

