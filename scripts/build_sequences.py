import os
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')

os.makedirs(DATA_PROCESSED, exist_ok=True)

# Must match MAINSHOCKS from fetch_aftershocks.py for metadata
MAINSHOCKS = {
    'Ridgecrest_2019': {'time': pd.to_datetime('2019-07-06T03:19:53Z'), 'mc': 2.5},
    'Northridge_1994': {'time': pd.to_datetime('1994-01-17T12:30:55Z'), 'mc': 2.5},
    'Landers_1992': {'time': pd.to_datetime('1992-06-28T11:57:34Z'), 'mc': 2.5},
    'Hector_Mine_1999': {'time': pd.to_datetime('1999-10-16T09:46:44Z'), 'mc': 2.5},
    'Loma_Prieta_1989': {'time': pd.to_datetime('1989-10-18T00:04:15Z'), 'mc': 2.5},
    'Tohoku_2011': {'time': pd.to_datetime('2011-03-11T05:46:24Z'), 'mc': 4.5},
    'Kumamoto_2016': {'time': pd.to_datetime('2016-04-15T16:25:06Z'), 'mc': 4.5},
    'Kobe_1995': {'time': pd.to_datetime('1995-01-16T20:46:52Z'), 'mc': 4.5},
    'Maule_2010': {'time': pd.to_datetime('2010-02-27T06:34:11Z'), 'mc': 4.5}
}

def build_sequences():
    summary_data = []
    
    for name, info in MAINSHOCKS.items():
        csv_path = os.path.join(DATA_RAW, f"{name}_aftershocks.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping {name}: Raw data not found.")
            continue
            
        print(f"Processing {name}...")
        df = pd.read_csv(csv_path)
        
        # Ensure time is datetime with timezone
        df['time'] = pd.to_datetime(df['time'], utc=True)
        
        # Filter by Mc
        df = df[df['mag'] >= info['mc']]
        
        # Filter strictly AFTER the mainshock
        df = df[df['time'] > info['time']]
        
        # Calculate time since mainshock in days and hours
        df['days_since_mainshock'] = (df['time'] - info['time']).dt.total_seconds() / (24 * 3600)
        df['hours_since_mainshock'] = (df['time'] - info['time']).dt.total_seconds() / 3600
        
        # Daily Counts (bins of 1 day up to 90 days)
        bins_daily = np.arange(0, 91, 1)
        daily_counts, _ = np.histogram(df['days_since_mainshock'], bins=bins_daily)
        cumulative_counts = np.cumsum(daily_counts)
        
        # Hourly Counts (bins of 1 hour up to 168 hours = 7 days)
        bins_hourly = np.arange(0, 169, 1)
        hourly_counts, _ = np.histogram(df['hours_since_mainshock'], bins=bins_hourly)
        
        # We save the main daily/cumulative sequence to parquet
        seq_df = pd.DataFrame({
            'days_since_mainshock': np.arange(1, 91),
            'daily_count': daily_counts,
            'cumulative_count': cumulative_counts
        })
        
        # Optional: Save hourly separately or just save both, we will save to same parquet padded with NaNs
        hourly_df = pd.DataFrame({
            'hours_since_mainshock': np.arange(1, 169),
            'hourly_count': hourly_counts
        })
        
        # Let's save them as separate parquet tables to avoid messy padding
        seq_df.to_parquet(os.path.join(DATA_PROCESSED, f"{name}_sequences_daily.parquet"))
        hourly_df.to_parquet(os.path.join(DATA_PROCESSED, f"{name}_sequences_hourly.parquet"))
        
        summary_data.append({
            'Sequence': name,
            'Duration (days)': 90,
            'Mc': info['mc'],
            'Total Aftershocks': cumulative_counts[-1],
            'Max Daily Rate': np.max(daily_counts)
        })
        
    # Generate Summary Table
    summary_df = pd.DataFrame(summary_data)
    print("\n--- Sequence Summary ---")
    print(summary_df.to_markdown(index=False))
    summary_df.to_csv(os.path.join(DATA_PROCESSED, 'sequence_summary.csv'), index=False)

if __name__ == "__main__":
    build_sequences()
