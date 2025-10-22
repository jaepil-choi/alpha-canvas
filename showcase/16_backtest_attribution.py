"""
Showcase 16: Backtesting with Portfolio Return Attribution

This showcase demonstrates the triple-cache backtesting system that automatically
computes and caches position-level portfolio returns for step-by-step PnL tracing
and winner/loser attribution analysis.

Features Demonstrated:
1. Automatic backtest execution when scaler provided
2. Position-level returns (T, N) for attribution
3. On-demand daily and cumulative PnL aggregation
4. Winner/loser stock identification
5. Shift-mask workflow preventing forward-looking bias
6. Re-masking preventing NaN pollution
7. Scaler comparison (dollar-neutral vs net-long)
8. Cumsum-based fair strategy comparison
"""

import numpy as np
import pandas as pd
import xarray as xr

from alpha_canvas import AlphaCanvas
from alpha_canvas.core.expression import Field
from alpha_canvas.ops.timeseries import TsMean
from alpha_canvas.ops.crosssection import Rank
from alpha_canvas.portfolio.strategies import DollarNeutralScaler, GrossNetScaler


def main():
    print("="*80)
    print("SHOWCASE 16: Backtesting with Portfolio Return Attribution")
    print("="*80)
    
    # ========================================================================
    # SETUP: Load data from fake Parquet database
    # ========================================================================
    print("\n" + "="*80)
    print("SETUP: Load Price and Return Data from Config")
    print("="*80)
    
    # Initialize with date range to trigger data loading from config
    rc = AlphaCanvas(
        start_date='2024-01-01',
        end_date='2024-01-20'
    )
    
    # Load price field (this triggers panel initialization AND auto-loads returns)
    print(f"  Loading data from: data/fake/pricevolume.parquet")
    rc.add_data('price', Field('adj_close'))
    
    print(f"  [OK] Data loaded successfully")
    print(f"       Time periods: {len(rc.db.coords['time'])}")
    print(f"       Assets: {list(rc.db.coords['asset'].values[:5])}... ({len(rc.db.coords['asset'])} total)")
    print(f"       Price field shape: {rc.db['price'].shape}")
    print(f"       Returns auto-loaded: {rc._returns is not None}")
    print(f"       Returns shape: {rc._returns.shape}")
    print(f"       → Returns loaded automatically (mandatory for backtesting)!")
    
    # ========================================================================
    # DEMO 1: Automatic Backtest with Dollar-Neutral Strategy
    # ========================================================================
    print("\n" + "="*80)
    print("DEMO 1: Automatic Backtest Execution")
    print("="*80)
    
    # Create multi-step expression
    expr = Rank(TsMean(Field('price'), window=5))
    
    print("\n[Step 1] Evaluate expression WITH scaler...")
    print(f"  Expression: Rank(TsMean(Field('price'), window=5))")
    
    scaler = DollarNeutralScaler()
    result = rc.evaluate(expr, scaler=scaler)
    
    print(f"  [OK] Evaluation complete")
    print(f"       Steps cached: {len(rc._evaluator._signal_cache)}")
    print(f"       Backtest auto-run: YES (scaler provided)")
    
    # Show cache structure
    print("\n[Step 2] Triple-cache populated automatically:")
    for step_idx in range(len(rc._evaluator._signal_cache)):
        signal_name, signal = rc._evaluator.get_cached_signal(step_idx)
        _, weights = rc._evaluator.get_cached_weights(step_idx)
        _, port_return = rc._evaluator.get_cached_port_return(step_idx)
        
        print(f"\n  Step {step_idx}: {signal_name}")
        print(f"    Signal shape: {signal.shape}")
        print(f"    Weights: {'Yes' if weights is not None else 'None'} " + 
              (f"(shape {weights.shape})" if weights is not None else ""))
        print(f"    Port Return: {'Yes' if port_return is not None else 'None'} " +
              (f"(shape {port_return.shape})" if port_return is not None else ""))
    
    # ========================================================================
    # DEMO 2: Position-Level Returns for Attribution
    # ========================================================================
    print("\n" + "="*80)
    print("DEMO 2: Position-Level Returns (Winner/Loser Attribution)")
    print("="*80)
    
    # Get position-level returns for final step
    final_step = len(rc._evaluator._signal_cache) - 1
    port_return = rc.get_port_return(final_step)
    
    print(f"\n[Position-Level Returns] Step {final_step}")
    print(f"  Shape: {port_return.shape} - (T, N) preserved for attribution")
    print(f"  First day (shifted): All NaN (no positions from yesterday)")
    print(f"  First few days: NaN due to ts_mean(window=5) warmup period")
    print(f"\n  Sample data (days 6-9 with actual values, all assets):")
    print(port_return.isel(time=slice(5, 9)).to_pandas().to_string())
    
    # Calculate total contribution per asset
    print("\n[Winner/Loser Analysis]")
    total_contrib = port_return.sum(dim='time')
    
    # Sort by contribution
    sorted_contrib = total_contrib.sortby(total_contrib, ascending=False)
    
    # Get number of assets from data
    num_assets = len(sorted_contrib.asset)
    
    print("  Total PnL contribution by asset:")
    for i, asset in enumerate(sorted_contrib.asset.values):
        contrib = sorted_contrib.sel(asset=asset).values
        rank = i + 1
        emoji = "🏆" if rank == 1 else ("📉" if rank == num_assets else "  ")
        print(f"    {emoji} {rank}. {asset:6s}: {contrib:+.6f}")
    
    best_stock = sorted_contrib.asset.values[0]
    worst_stock = sorted_contrib.asset.values[-1]
    print(f"\n  Best performer: {best_stock}")
    print(f"  Worst performer: {worst_stock}")
    
    # ========================================================================
    # DEMO 3: On-Demand Daily and Cumulative PnL
    # ========================================================================
    print("\n" + "="*80)
    print("DEMO 3: On-Demand PnL Aggregation")
    print("="*80)
    
    # Daily PnL
    daily_pnl = rc.get_daily_pnl(final_step)
    
    print(f"\n[Daily PnL] Aggregated across assets")
    print(f"  Shape: {daily_pnl.shape} - (T,) time series")
    print(f"  Mean daily PnL: {daily_pnl.mean().values:.6f}")
    print(f"  Std daily PnL: {daily_pnl.std().values:.6f}")
    sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
    print(f"  Annualized Sharpe: {sharpe.values:.2f}")
    
    print(f"\n  Daily PnL time series (first 10 days):")
    for i in range(min(10, len(daily_pnl))):
        pnl_value = daily_pnl.isel(time=i).values
        date = daily_pnl.time.values[i]
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        bar = "█" * max(0, int(abs(pnl_value) * 1000))
        sign = "+" if pnl_value >= 0 else "-"
        print(f"    {date_str}: {sign}{abs(pnl_value):.6f} {bar}")
    
    # Cumulative PnL
    cum_pnl = rc.get_cumulative_pnl(final_step)
    
    print(f"\n[Cumulative PnL] Using cumsum (time-invariant)")
    print(f"  Shape: {cum_pnl.shape}")
    print(f"  Final PnL: {cum_pnl.isel(time=-1).values:.6f}")
    
    print(f"\n  Why cumsum over cumprod?")
    print(f"    - Cumsum: Time-invariant (fair comparison)")
    print(f"    - Cumprod: Compound effect favors longer strategies (unfair)")
    
    # ========================================================================
    # DEMO 4: Shift-Mask Workflow (Forward-Bias Prevention)
    # ========================================================================
    print("\n" + "="*80)
    print("DEMO 4: Shift-Mask Workflow (No Forward-Looking Bias)")
    print("="*80)
    
    print("\n[Workflow Explanation]")
    print("  1. Generate weights from signal at time t-1")
    print("  2. Shift weights: weights_shifted = weights.shift(time=1)")
    print("  3. Re-mask with universe at time t (critical!)")
    print("  4. Calculate returns: port_return = final_weights * returns[t]")
    
    print("\n[Why Re-Masking Matters]")
    print("  - Stock exits universe → weight becomes NaN")
    print("  - Without re-mask: NaN * return = NaN → PnL is NaN (pollution)")
    print("  - With re-mask: Liquidate exited positions → PnL valid")
    
    # Demonstrate with actual data (use day 7 which has valid weights)
    weights_step = rc.get_weights(final_step)
    weights_shifted = weights_step.shift(time=1)
    
    print(f"\n[Example: Day 7 Weights (after warmup period)]")
    day_idx = 6  # Day 7 (0-indexed)
    print(f"  Original weights (day 7): {weights_step.isel(time=day_idx).values}")
    print(f"  Shifted weights (day 7):  {weights_shifted.isel(time=day_idx).values}")
    print(f"  → We trade using DAY 6's signal on DAY 7's returns")
    print(f"  → This prevents forward-looking bias!")
    print(f"  → Note: Shifted weights come from previous day's signal")
    
    # ========================================================================
    # DEMO 5: Scaler Comparison (Efficient Strategy Testing)
    # ========================================================================
    print("\n" + "="*80)
    print("DEMO 5: Efficient Scaler Comparison")
    print("="*80)
    
    print("\n[Strategy 1: Dollar-Neutral (already evaluated)]")
    pnl1 = rc.get_cumulative_pnl(final_step).isel(time=-1).values
    print(f"  Target: Gross=2.0, Net=0.0")
    print(f"  Final PnL: {pnl1:.6f}")
    
    print("\n[Strategy 2: Net-Long Bias]")
    print("  Recalculating weights with new scaler...")
    scaler2 = GrossNetScaler(target_gross=2.0, target_net=0.5)
    rc._evaluator.recalculate_weights_with_scaler(scaler2)
    
    pnl2 = rc.get_cumulative_pnl(final_step).isel(time=-1).values
    print(f"  Target: Gross=2.0, Net=0.5")
    print(f"  Final PnL: {pnl2:.6f}")
    
    print(f"\n[Efficiency Note]")
    print(f"  Signal cache: REUSED (no re-evaluation)")
    print(f"  Weight cache: RECALCULATED (new scaler)")
    print(f"  Port return cache: RECALCULATED (new weights)")
    print(f"  → Fast strategy comparison without re-running expressions!")
    
    # ========================================================================
    # DEMO 6: Step-by-Step PnL Tracing
    # ========================================================================
    print("\n" + "="*80)
    print("DEMO 6: Step-by-Step PnL Decomposition")
    print("="*80)
    
    print("\n[PnL at Each Expression Step]")
    print("  (Using dollar-neutral scaler)")
    
    # Re-evaluate with original scaler for comparison
    rc.evaluate(expr, scaler=DollarNeutralScaler())
    
    for step_idx in range(len(rc._evaluator._signal_cache)):
        signal_name, _ = rc._evaluator.get_cached_signal(step_idx)
        cum_pnl_step = rc.get_cumulative_pnl(step_idx)
        
        if cum_pnl_step is not None:
            final_pnl = cum_pnl_step.isel(time=-1).values
            print(f"  Step {step_idx} ({signal_name:30s}): PnL = {final_pnl:+.6f}")
        else:
            print(f"  Step {step_idx} ({signal_name:30s}): No backtest (no scaler)")
    
    print(f"\n[Use Case]")
    print(f"  - Debug which operator improves/degrades PnL")
    print(f"  - Compare PnL before and after signal transformations")
    print(f"  - Identify optimal point in expression tree")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY: Key Benefits of Triple-Cache Backtesting")
    print("="*80)
    
    print("\n1. Automatic Execution")
    print("   - Just provide scaler to evaluate()")
    print("   - Portfolio returns computed automatically")
    print("   - No separate backtest API needed")
    
    print("\n2. Position-Level Attribution")
    print("   - (T, N) shape preserved for winner/loser analysis")
    print("   - Identify which stocks drove PnL")
    print("   - Debug factor performance at stock level")
    
    print("\n3. Forward-Bias Prevention")
    print("   - Shift-mask workflow ensures realistic backtest")
    print("   - Trade on yesterday's signal, today's returns")
    print("   - Re-masking prevents NaN pollution")
    
    print("\n4. Efficient Strategy Comparison")
    print("   - Signal cache reused across scalers")
    print("   - Only weights and returns recalculated")
    print("   - Fast iteration for strategy optimization")
    
    print("\n5. Step-by-Step Traceability")
    print("   - PnL cached at every expression step")
    print("   - Compare PnL before/after each operator")
    print("   - Foundation for PnL attribution (future)")
    
    print("\n6. Fair Comparison (Cumsum)")
    print("   - Time-invariant metric")
    print("   - Doesn't favor longer strategies")
    print("   - Better for research than cumprod")
    
    print("\n" + "="*80)
    print("SHOWCASE COMPLETE - Backtest & Attribution Ready!")
    print("="*80)
    print()


if __name__ == '__main__':
    main()

