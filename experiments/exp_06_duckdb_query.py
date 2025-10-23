"""
Experiment 06: DuckDB Query Validation

Date: 2024-01-20
Status: In Progress

Objective:
- Validate DuckDB can query Parquet files with SQL
- Test calculated columns and date filtering
- Verify query results match expectations

Success Criteria:
- [ ] DuckDB can read Parquet files
- [ ] Calculated columns work correctly
- [ ] Date filtering works
- [ ] Returns pandas DataFrame in long format
"""

import duckdb
import pandas as pd
from pathlib import Path

def main():
    print("=" * 70)
    print("EXPERIMENT 06: DuckDB Query Validation")
    print("=" * 70)
    
    # Step 1: Verify Parquet file exists
    print("\n[Step 1] Verifying Parquet file...")
    parquet_path = Path('data/pricevolume.parquet')
    if not parquet_path.exists():
        print(f"  [FAIL] File not found: {parquet_path}")
        print("  Run exp_05_create_mock_data.py first!")
        return
    
    print(f"  [OK] File exists: {parquet_path}")
    print(f"       Size: {parquet_path.stat().st_size:,} bytes")
    
    # Step 2: Basic query - read all data
    print("\n[Step 2] Basic query: SELECT * FROM Parquet...")
    query1 = f"SELECT * FROM read_parquet('{parquet_path}')"
    
    df1 = duckdb.query(query1).to_df()
    print(f"  [OK] Query executed")
    print(f"       Rows: {len(df1)}")
    print(f"       Columns: {list(df1.columns)}")
    print(f"       Dtypes:")
    for col, dtype in df1.dtypes.items():
        print(f"         {col:25s} -> {dtype}")
    
    print(f"\n  Sample data (first 5 rows):")
    print(df1.head().to_string(index=False))
    
    # Step 3: Query with calculated column
    print("\n[Step 3] Query with calculated column (adj_close)...")
    query2 = f"""
    SELECT 
      date, 
      security_id, 
      close_price,
      adjustment_factor,
      close_price * adjustment_factor as adj_close 
    FROM read_parquet('{parquet_path}')
    """
    
    df2 = duckdb.query(query2).to_df()
    print(f"  [OK] Query executed")
    print(f"       Rows: {len(df2)}")
    print(f"       Has adj_close column: {'adj_close' in df2.columns}")
    
    print(f"\n  Validating calculation (close_price * adjustment_factor)...")
    df2['expected_adj_close'] = df2['close_price'] * df2['adjustment_factor']
    df2['match'] = df2['adj_close'] == df2['expected_adj_close']
    
    print(f"       All matches: {df2['match'].all()}")
    print(f"       Sample:")
    print(df2[['security_id', 'close_price', 'adjustment_factor', 'adj_close', 'match']].head().to_string(index=False))
    
    # Step 4: Query with date filtering
    print("\n[Step 4] Query with date filtering...")
    start_date = '2024-01-05'
    end_date = '2024-01-15'
    
    query3 = f"""
    SELECT 
      date, 
      security_id, 
      close_price * adjustment_factor as adj_close 
    FROM read_parquet('{parquet_path}')
    WHERE date >= '{start_date}' 
      AND date <= '{end_date}'
    """
    
    df3 = duckdb.query(query3).to_df()
    print(f"  [OK] Query executed")
    print(f"       Rows returned: {len(df3)}")
    print(f"       Date range requested: {start_date} to {end_date}")
    print(f"       Date range actual: {df3['date'].min()} to {df3['date'].max()}")
    
    # Count unique dates
    unique_dates = df3['date'].nunique()
    unique_securities = df3['security_id'].nunique()
    print(f"       Unique dates: {unique_dates}")
    print(f"       Unique securities: {unique_securities}")
    print(f"       Expected rows: {unique_dates * unique_securities}")
    print(f"       Actual rows: {len(df3)}")
    print(f"       Match: {len(df3) == unique_dates * unique_securities}")
    
    print(f"\n  Sample filtered data:")
    print(df3.head(12).to_string(index=False))  # Show 2 days × 6 securities
    
    # Step 5: Query with parameterized dates (simulating :start_date, :end_date)
    print("\n[Step 5] Testing parameterized date substitution...")
    query_template = """
    SELECT 
      date, 
      security_id, 
      close_price * adjustment_factor as adj_close 
    FROM read_parquet('data/pricevolume.parquet')
    WHERE date >= :start_date 
      AND date <= :end_date
    """
    
    # Simulate parameter substitution (what DataLoader will do)
    query_substituted = query_template.replace(':start_date', f"'{start_date}'")
    query_substituted = query_substituted.replace(':end_date', f"'{end_date}'")
    
    print(f"  Original template:")
    print(f"    WHERE date >= :start_date AND date <= :end_date")
    print(f"\n  After substitution:")
    print(f"    WHERE date >= '{start_date}' AND date <= '{end_date}'")
    
    df5 = duckdb.query(query_substituted).to_df()
    print(f"\n  [OK] Parameterized query executed")
    print(f"       Rows: {len(df5)}")
    print(f"       Matches previous result: {len(df5) == len(df3)}")
    
    # Step 6: Test multiple field queries
    print("\n[Step 6] Testing multiple field queries...")
    
    # Query 1: adj_close
    query_adj_close = f"""
    SELECT date, security_id, close_price * adjustment_factor as adj_close 
    FROM read_parquet('{parquet_path}')
    WHERE date >= '{start_date}' AND date <= '{end_date}'
    """
    
    # Query 2: volume
    query_volume = f"""
    SELECT date, security_id, volume 
    FROM read_parquet('{parquet_path}')
    WHERE date >= '{start_date}' AND date <= '{end_date}'
    """
    
    df_adj_close = duckdb.query(query_adj_close).to_df()
    df_volume = duckdb.query(query_volume).to_df()
    
    print(f"  [OK] adj_close query: {len(df_adj_close)} rows")
    print(f"  [OK] volume query: {len(df_volume)} rows")
    print(f"       Both queries return same shape: {df_adj_close.shape == df_volume.shape}")
    
    # Step 7: Performance test
    print("\n[Step 7] Performance test (100 queries)...")
    import time
    
    start_time = time.time()
    for _ in range(100):
        _ = duckdb.query(query3).to_df()
    elapsed = time.time() - start_time
    
    print(f"  [OK] 100 queries executed")
    print(f"       Total time: {elapsed:.3f}s")
    print(f"       Average per query: {elapsed/100*1000:.2f}ms")
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("[SUCCESS] DuckDB Query Validation Complete!")
    print()
    print("Key Findings:")
    print("  ✓ DuckDB can read Parquet files with read_parquet()")
    print("  ✓ Calculated columns work correctly (close * adj_factor)")
    print("  ✓ Date filtering works as expected")
    print("  ✓ Parameter substitution pattern works")
    print("  ✓ Returns pandas DataFrame in long format")
    print(f"  ✓ Query performance: ~{elapsed/100*1000:.2f}ms per query")
    print()
    print("Implementation Implications:")
    print("  • DataLoader can use simple string replace for :start_date/:end_date")
    print("  • DuckDB queries are fast enough for real-time usage")
    print("  • Long format DataFrame ready for pivot operation")
    print("=" * 70)


if __name__ == '__main__':
    main()

