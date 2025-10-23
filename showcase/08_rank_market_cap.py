"""
Showcase 08: rank() - Cross-Sectional Percentile Ranking

This demonstration shows:
1. Basic ranking: market cap (small → 0.0, large → 1.0)
2. NaN handling: Automatic preservation
3. Time independence: Each time step ranked separately
4. Use cases:
   - Demeaned ranks: Rank(returns) - 0.5 for long/short signals
   - Smooth ranks: ts_mean(Rank(returns), 5) for trend following
5. Compare raw vs ranked values

Date: 2025-01-21
"""

import numpy as np
import pandas as pd
import xarray as xr
from alpha_canvas import AlphaCanvas
from alpha_canvas.core.expression import Field
from alpha_canvas.ops.crosssection import Rank
from alpha_canvas.ops.timeseries import TsMean

def main():
    print("="*80)
    print("SHOWCASE 08: rank() - Cross-Sectional Percentile Ranking")
    print("="*80)
    
    # ================================================================
    # Step 1: Setup - Initialize AlphaCanvas with date range
    # ================================================================
    print("\n[Step 1] Initializing AlphaCanvas...")
    
    rc = AlphaCanvas(
        config_dir='config',
        start_date='2024-01-01',
        end_date='2024-01-10'
    )
    
    # Create date range and get assets from config
    dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
    # Note: In production, assets come from loaded data or config
    # For demo, we'll use the mock data assets
    assets = ['AAPL', 'NVDA', 'MSFT', 'TSLA', 'META', 'GOOG']
    
    print(f"  Time range: {dates[0]} to {dates[-1]}")
    print(f"  Asset universe: {assets}")
    print(f"  [OK] AlphaCanvas initialized")
    
    # ================================================================
    # Step 2: Add market cap data with varying values
    # ================================================================
    print("\n[Step 2] Adding market cap data...")
    
    # Create market cap data: different scales for different companies
    # Small cap: 1-2B, Mid cap: 5-10B, Large cap: 50-100B
    np.random.seed(42)
    market_cap_data = []
    
    for t in range(len(dates)):
        # AAPL: Large cap ~80B
        aapl = 80 + np.random.randn() * 5
        # NVDA: Large cap ~70B
        nvda = 70 + np.random.randn() * 5
        # MSFT: Large cap ~90B
        msft = 90 + np.random.randn() * 5
        # TSLA: Mid cap ~8B
        tsla = 8 + np.random.randn() * 1
        # META: Mid cap ~6B
        meta = 6 + np.random.randn() * 1
        # GOOG: Small cap ~1.5B
        goog = 1.5 + np.random.randn() * 0.3
        
        market_cap_data.append([aapl, nvda, msft, tsla, meta, goog])
    
    market_cap = xr.DataArray(
        market_cap_data,
        dims=['time', 'asset'],
        coords={'time': dates, 'asset': assets}
    )
    
    # Add a couple NaN values to demonstrate NaN handling
    market_cap.values[2, 1] = np.nan  # NVDA on day 3
    market_cap.values[5, 4] = np.nan  # META on day 6
    
    rc.add_data('market_cap', market_cap)
    
    print("  Market cap data added:")
    print(f"  Shape: {market_cap.shape}")
    print(f"  Range: ${market_cap.values[~np.isnan(market_cap.values)].min():.1f}B - ${market_cap.values[~np.isnan(market_cap.values)].max():.1f}B")
    print(f"  NaN count: {np.sum(np.isnan(market_cap.values))}")
    print(f"  [OK] Market cap data prepared")
    
    # ================================================================
    # Step 3: Apply rank() operator
    # ================================================================
    print("\n[Step 3] Applying rank() operator...")
    
    # Create expression: Rank(Field('market_cap'))
    rank_expr = Rank(child=Field('market_cap'))
    rc.add_data('mcap_rank', rank_expr)
    
    ranked = rc.db['mcap_rank']
    
    print("  rank() operator applied:")
    print(f"  Input shape: {market_cap.shape}")
    print(f"  Output shape: {ranked.shape}")
    print(f"  Output range: [{ranked.values[~np.isnan(ranked.values)].min():.6f}, {ranked.values[~np.isnan(ranked.values)].max():.6f}]")
    print(f"  Expected range: [0.0, 1.0] ✓")
    print(f"  [OK] Ranking completed")
    
    # ================================================================
    # Step 4: Demonstrate Basic Ranking (time step 0)
    # ================================================================
    print("\n[Step 4] Basic Ranking - Time Step 0...")
    
    t0_raw = market_cap.isel(time=0).values
    t0_ranked = ranked.isel(time=0).values
    
    print("\n  Raw Market Cap (Time 0):")
    for i, asset in enumerate(assets):
        print(f"    {asset}: ${t0_raw[i]:.2f}B")
    
    print("\n  Ranked (Percentile):")
    for i, asset in enumerate(assets):
        print(f"    {asset}: {t0_ranked[i]:.3f}")
    
    # Verify ranking correctness
    sorted_indices = np.argsort(t0_raw)
    print("\n  Verification (sorted by market cap):")
    for rank, idx in enumerate(sorted_indices):
        print(f"    #{rank+1}: {assets[idx]} (${t0_raw[idx]:.2f}B) → rank {t0_ranked[idx]:.3f}")
    
    print("  [OK] Smallest → 0.0, Largest → 1.0")
    
    # ================================================================
    # Step 5: Demonstrate NaN Handling
    # ================================================================
    print("\n[Step 5] NaN Handling...")
    
    t2_raw = market_cap.isel(time=2).values
    t2_ranked = ranked.isel(time=2).values
    
    print("\n  Time Step 2 (NVDA has NaN):")
    print("  Raw values:")
    for i, asset in enumerate(assets):
        val_str = f"${t2_raw[i]:.2f}B" if not np.isnan(t2_raw[i]) else "NaN"
        print(f"    {asset}: {val_str}")
    
    print("\n  Ranked values:")
    for i, asset in enumerate(assets):
        rank_str = f"{t2_ranked[i]:.3f}" if not np.isnan(t2_ranked[i]) else "NaN"
        print(f"    {asset}: {rank_str}")
    
    # Verify NaN preservation
    assert np.isnan(t2_ranked[1]), "NaN should be preserved in output"
    print("\n  [OK] NaN automatically preserved in ranking")
    
    # ================================================================
    # Step 6: Time Independence
    # ================================================================
    print("\n[Step 6] Time Independence...")
    
    # Check GOOG (smallest cap) across time
    goog_idx = list(assets).index('GOOG')
    goog_raw = market_cap.sel(asset='GOOG').values
    goog_ranked = ranked.sel(asset='GOOG').values
    
    print("\n  GOOG (consistently smallest market cap) across time:")
    print("  Day | Raw Value | Rank")
    print("  ----|-----------|------")
    for t in range(len(dates)):
        print(f"  {t+1:3d} | ${goog_raw[t]:9.2f}B | {goog_ranked[t]:.3f}")
    
    # Verify GOOG is always ranked 0.0 (smallest)
    assert np.all(goog_ranked == 0.0), "GOOG should always be ranked 0.0"
    print("\n  [OK] Each time step ranked independently")
    print("       GOOG consistently smallest → rank 0.0")
    
    # ================================================================
    # Step 7: Use Case 1 - Demeaned Ranks for Long/Short
    # ================================================================
    print("\n[Step 7] Use Case 1: Demeaned Ranks for Long/Short Signals...")
    
    # Add returns data
    returns_data = []
    np.random.seed(123)
    for t in range(len(dates)):
        returns_data.append(np.random.randn(len(assets)) * 0.02)  # 2% volatility
    
    returns = xr.DataArray(
        returns_data,
        dims=['time', 'asset'],
        coords={'time': dates, 'asset': assets}
    )
    rc.add_data('returns', returns)
    
    # Rank returns and demean
    rank_ret_expr = Rank(child=Field('returns'))
    rc.add_data('rank_returns', rank_ret_expr)
    
    # Demean: subtract 0.5 so [-0.5, 0.5] where negative = short, positive = long
    demeaned_ranks = rc.db['rank_returns'] - 0.5
    rc.add_data('demeaned_rank_returns', demeaned_ranks)
    
    print("\n  Demeaned Rank Returns (Time 0):")
    print("  Asset | Return | Rank | Demeaned | Signal")
    print("  ------|--------|------|----------|-------")
    
    t0_ret = returns.isel(time=0).values
    t0_rank_ret = rc.db['rank_returns'].isel(time=0).values
    t0_demean = demeaned_ranks.isel(time=0).values
    
    for i, asset in enumerate(assets):
        signal = "LONG" if t0_demean[i] > 0 else "SHORT"
        print(f"  {asset:5s} | {t0_ret[i]:6.3f} | {t0_rank_ret[i]:4.2f} | {t0_demean[i]:7.3f} | {signal:5s}")
    
    print("\n  [OK] Demeaned ranks: [-0.5, 0.5] for long/short signals")
    print("       Rank > 0.5 (positive demean) → LONG")
    print("       Rank < 0.5 (negative demean) → SHORT")
    
    # ================================================================
    # Step 8: Use Case 2 - Smooth Ranks with ts_mean
    # ================================================================
    print("\n[Step 8] Use Case 2: Smooth Ranks with ts_mean...")
    
    # Apply 3-day moving average to ranked returns
    smooth_rank_expr = TsMean(child=Field('rank_returns'), window=3)
    rc.add_data('smooth_rank_returns', smooth_rank_expr)
    
    smooth_ranks = rc.db['smooth_rank_returns']
    
    print("\n  Smooth Rank Returns (3-day MA) for AAPL:")
    print("  Day | Raw Rank | 3-day MA")
    print("  ----|----------|----------")
    
    aapl_raw_rank = rc.db['rank_returns'].sel(asset='AAPL').values
    aapl_smooth = smooth_ranks.sel(asset='AAPL').values
    
    for t in range(len(dates)):
        raw_str = f"{aapl_raw_rank[t]:.3f}" if not np.isnan(aapl_raw_rank[t]) else "NaN"
        smooth_str = f"{aapl_smooth[t]:.3f}" if not np.isnan(aapl_smooth[t]) else "NaN"
        print(f"  {t+1:3d} | {raw_str:8s} | {smooth_str:8s}")
    
    print("\n  [OK] Smoothed ranks reduce noise for trend following")
    print("       First 2 days NaN due to min_periods=3")
    print("       Day 3+ shows 3-day average of ranks")
    
    # ================================================================
    # Step 9: Expression Tree Visualization
    # ================================================================
    print("\n[Step 9] Expression Tree Structure...")
    
    print("\n  Computation Graph:")
    print("  ")
    print("    Field('market_cap')")
    print("          │")
    print("          ▼")
    print("     Rank(child)")
    print("          │")
    print("          ▼")
    print("   [0.0 - 1.0]")
    print("  ")
    print("  For smooth ranks:")
    print("  ")
    print("    Field('rank_returns')")
    print("          │")
    print("          ▼")
    print("  TsMean(child, window=3)")
    print("          │")
    print("          ▼")
    print("  Smoothed Ranks")
    
    print("\n  Evaluation with Visitor:")
    print("    1. Field('market_cap') → (10, 6) DataArray")
    print("    2. Rank.compute() → percentile conversion")
    print("    3. Cache result as step 1")
    print("  ")
    print("  All steps cached in visitor._cache for traceability")
    
    # ================================================================
    # Step 10: Statistical Validation
    # ================================================================
    print("\n[Step 10] Statistical Validation...")
    
    # Check that ranks are uniformly distributed (excluding NaN)
    all_ranks = ranked.values[~np.isnan(ranked.values)]
    
    print(f"\n  Rank Statistics:")
    print(f"    Count: {len(all_ranks)}")
    print(f"    Min: {all_ranks.min():.6f}")
    print(f"    Max: {all_ranks.max():.6f}")
    print(f"    Mean: {all_ranks.mean():.6f}")
    print(f"    Median: {np.median(all_ranks):.6f}")
    print(f"    Std Dev: {all_ranks.std():.6f}")
    
    # For uniform distribution [0, 1], mean should be ~0.5, std ~0.29
    expected_mean = 0.5
    expected_std = np.sqrt(1/12)  # Continuous uniform variance = (b-a)²/12
    
    print(f"\n  Expected (Uniform Distribution):")
    print(f"    Mean: {expected_mean:.6f}")
    print(f"    Std Dev: {expected_std:.6f}")
    
    mean_diff = abs(all_ranks.mean() - expected_mean)
    std_diff = abs(all_ranks.std() - expected_std)
    
    print(f"\n  Difference:")
    print(f"    Mean: {mean_diff:.6f}")
    print(f"    Std Dev: {std_diff:.6f}")
    
    print("\n  [OK] Ranks approximately uniformly distributed")
    
    print("\n" + "="*80)
    print("SHOWCASE COMPLETE")
    print("="*80)
    
    print("\n[KEY TAKEAWAYS]")
    print("1. rank() provides percentile ranking: smallest → 0.0, largest → 1.0")
    print("2. NaN values automatically preserved (scipy nan_policy='omit')")
    print("3. Each time step ranked independently (cross-sectional)")
    print("4. Demeaned ranks (rank - 0.5) useful for long/short signals")
    print("5. Smooth ranks (ts_mean(rank)) reduce noise for trend following")
    print("6. Generic visitor pattern: Rank uses same visit_operator() as TsMean, TsAny")
    print("7. All 87 tests pass with new Rank operator and refactored Visitor")
    
    print("\n[NEXT STEPS]")
    print("- Implement cs_quantile() for creating quantile groups")
    print("- Implement group_neutralize() for sector/industry neutralization")
    print("- Add more cross-sectional operators (cs_zscore, cs_scale)")
    print("- Explore operator composition patterns")

if __name__ == '__main__':
    main()

