"""
Experiment 05: Create Mock Parquet Data

Date: 2024-01-20
Status: In Progress

Objective:
- Create synthetic Parquet file with realistic trading data structure
- Validate Parquet file creation and basic properties

Success Criteria:
- [ ] Parquet file created at data/pricevolume.parquet
- [ ] 90 rows (15 days × 6 securities)
- [ ] All required columns present
- [ ] Data types are correct
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def main():
    print("=" * 70)
    print("EXPERIMENT 05: Create Mock Parquet Data")
    print("=" * 70)
    
    # Step 1: Generate date range (15 trading days, skip weekends)
    print("\n[Step 1] Generating trading dates...")
    start_date = datetime(2024, 1, 2)  # Tuesday
    dates = []
    current_date = start_date
    while len(dates) < 15:
        if current_date.weekday() < 5:  # Monday=0, Friday=4
            dates.append(current_date)
        current_date += timedelta(days=1)
    
    print(f"  [OK] Generated {len(dates)} trading days")
    print(f"       First date: {dates[0].strftime('%Y-%m-%d')}")
    print(f"       Last date: {dates[-1].strftime('%Y-%m-%d')}")
    
    # Step 2: Define securities
    print("\n[Step 2] Defining securities...")
    securities = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    print(f"  [OK] {len(securities)} securities: {', '.join(securities)}")
    
    # Step 3: Generate price data
    print("\n[Step 3] Simulating price data...")
    np.random.seed(42)  # For reproducibility
    
    records = []
    for security in securities:
        price = 100.0  # Starting price
        for date in dates:
            # Random daily return between -5% and +5%
            daily_return = np.random.uniform(-0.05, 0.05)
            
            # Calculate prices
            open_price = price
            close_price = price * (1 + daily_return)
            
            # Random volume between 1M and 10M
            volume = np.random.randint(1_000_000, 10_000_000)
            
            records.append({
                'date': date.strftime('%Y-%m-%d'),
                'security_id': security,
                'open_price': round(open_price, 2),
                'close_price': round(close_price, 2),
                'adjustment_factor': 1.0,
                'volume': volume
            })
            
            # Update price for next day
            price = close_price
    
    print(f"  [OK] Generated {len(records)} records")
    print(f"       Expected: {len(dates) * len(securities)} records")
    print(f"       Match: {len(records) == len(dates) * len(securities)}")
    
    # Step 4: Create DataFrame
    print("\n[Step 4] Creating DataFrame...")
    df = pd.DataFrame(records)
    
    print(f"  [OK] DataFrame created")
    print(f"       Shape: {df.shape}")
    print(f"       Columns: {list(df.columns)}")
    print(f"       Dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"         {col:25s} -> {dtype}")
    
    # Step 5: Display sample data
    print("\n[Step 5] Sample data (first 10 rows):")
    print(df.head(10).to_string(index=False))
    
    print("\n  Sample data (last 10 rows):")
    print(df.tail(10).to_string(index=False))
    
    # Step 6: Validate data quality
    print("\n[Step 6] Validating data quality...")
    
    # Check for missing values
    missing = df.isnull().sum()
    print(f"  Missing values:")
    for col, count in missing.items():
        status = "[OK]" if count == 0 else "[WARN]"
        print(f"    {status} {col}: {count}")
    
    # Check price ranges
    print(f"\n  Price statistics:")
    print(f"    Open price:  min={df['open_price'].min():.2f}, max={df['open_price'].max():.2f}")
    print(f"    Close price: min={df['close_price'].min():.2f}, max={df['close_price'].max():.2f}")
    print(f"    Volume:      min={df['volume'].min():,}, max={df['volume'].max():,}")
    
    # Check adjustment factor
    unique_adj = df['adjustment_factor'].unique()
    print(f"\n  Adjustment factors: {unique_adj}")
    if len(unique_adj) == 1 and unique_adj[0] == 1.0:
        print("    [OK] All adjustment factors are 1.0")
    
    # Step 7: Create data directory and save Parquet
    print("\n[Step 7] Saving to Parquet...")
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    output_path = data_dir / 'pricevolume.parquet'
    df.to_parquet(output_path, index=False)
    
    print(f"  [OK] Parquet file saved: {output_path}")
    
    # Step 8: Verify file
    print("\n[Step 8] Verifying saved file...")
    file_size = output_path.stat().st_size
    print(f"  File size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
    
    # Read back and verify
    df_verify = pd.read_parquet(output_path)
    print(f"  Rows after read: {len(df_verify)}")
    print(f"  Columns after read: {len(df_verify.columns)}")
    print(f"  Match original: {df.equals(df_verify)}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("[SUCCESS] Mock Parquet data created successfully!")
    print()
    print("File Details:")
    print(f"  Path: {output_path}")
    print(f"  Rows: {len(df_verify)} (15 days × 6 securities)")
    print(f"  Columns: {', '.join(df_verify.columns)}")
    print(f"  Date range: {df_verify['date'].min()} to {df_verify['date'].max()}")
    print(f"  Securities: {', '.join(sorted(df_verify['security_id'].unique()))}")
    print()
    print("Ready for DuckDB query experiments!")
    print("=" * 70)


if __name__ == '__main__':
    main()

