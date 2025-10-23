"""
Run All Showcases

This script runs all showcase demonstrations in sequence.

Run: poetry run python showcase/run_all.py
"""

import subprocess
import sys
from pathlib import Path


def run_showcase(script_name):
    """Run a showcase script and return success status."""
    print(f"\n{'=' * 80}")
    print(f"Running: {script_name}")
    print('=' * 80)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        print(f"\n[ERROR] {script_name} failed!")
        return False


def main():
    """Run all showcase scripts in order."""
    print("\n" + "=" * 80)
    print("  ALPHA-CANVAS MVP - COMPLETE SHOWCASE SUITE")
    print("  Running all demonstrations...")
    print("=" * 80)
    
    showcase_dir = Path(__file__).parent
    
    # Define showcases in order
    showcases = [
        '01_config_module.py',
        '02_datapanel_model.py',
        '03_expression_visitor.py',
        '04_facade_complete.py',
        '05_parquet_data_loading.py',
        '06_ts_mean_operator.py',
        '07_ts_any_surge_detection.py',
        '08_rank_market_cap.py',
        '09_universe_masking.py',
        '10_boolean_expressions.py',
        '11_data_accessor.py',
        '12_cs_quantile.py',
        '13_signal_assignment.py',
        '14_weight_scaling.py',
        '15_weight_caching_pnl.py',
        '16_backtest_attribution.py',
    '17_alpha_database_datasource.py',
    '18_alpha_canvas_datasource_integration.py'
]
    
    results = {}
    
    # Run each showcase
    for showcase in showcases:
        script_path = showcase_dir / showcase
        success = run_showcase(script_path)
        results[showcase] = success
        
        if not success:
            print("\n[ABORT] Stopping due to failure")
            break
    
    # Print summary
    print("\n" + "=" * 80)
    print("  SHOWCASE SUITE SUMMARY")
    print("=" * 80)
    
    for showcase, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {showcase}")
    
    # Overall result
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "=" * 80)
        print("  ✓ ALL SHOWCASES PASSED!")
        print("  Alpha-Canvas MVP Foundation: Production Ready")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("  ✗ SOME SHOWCASES FAILED")
        print("=" * 80)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

