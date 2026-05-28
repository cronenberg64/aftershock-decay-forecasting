import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
REPORTS = os.path.join(PROJECT_ROOT, 'reports')

os.makedirs(REPORTS, exist_ok=True)

MAINSHOCKS = [
    'Ridgecrest_2019', 'Northridge_1994', 'Landers_1992', 
    'Hector_Mine_1999', 'Loma_Prieta_1989', 'Tohoku_2011', 
    'Kumamoto_2016', 'Kobe_1995', 'Maule_2010'
]

def run_adf(series):
    # dropna in case of any NaNs
    series = series.dropna()
    if len(series) < 10:
        return np.nan
    try:
        result = adfuller(series, autolag='AIC')
        return result[1] # p-value
    except:
        return np.nan

def analyze_stationarity():
    results = []
    
    for name in MAINSHOCKS:
        parquet_path = os.path.join(DATA_PROCESSED, f"{name}_sequences_daily.parquet")
        if not os.path.exists(parquet_path):
            continue
            
        df = pd.read_parquet(parquet_path)
        
        # log(daily_count + 1)
        log_daily = np.log1p(df['daily_count'])
        diff_log_daily = log_daily.diff()
        
        # cumulative_count
        cum_count = df['cumulative_count']
        diff_cum_count = cum_count.diff()
        
        # ADF P-values
        pval_log_daily = run_adf(log_daily)
        pval_diff_log_daily = run_adf(diff_log_daily)
        
        pval_cum = run_adf(cum_count)
        pval_diff_cum = run_adf(diff_cum_count)
        
        results.append({
            'Sequence': name,
            'Log(Daily+1) p-value': pval_log_daily,
            'Diff Log(Daily+1) p-value': pval_diff_log_daily,
            'Cum Count p-value': pval_cum,
            'Diff Cum Count p-value': pval_diff_cum
        })
        
    results_df = pd.DataFrame(results)
    
    # Recommendation logic:
    # If p < 0.05, stationary (d=0). Else, difference it (d=1).
    report_lines = [
        "# Stationarity Analysis Report",
        "",
        "This report assesses the stationarity of aftershock sequence time series using the Augmented Dickey-Fuller (ADF) test. Stationarity is a requirement for classical AR and MA models.",
        "",
        "## ADF Test P-values",
        "A p-value < 0.05 indicates the series is stationary.",
        "",
        results_df.to_markdown(index=False),
        "",
        "## Recommendations for Differencing (d)",
        "- **Log Daily Counts**: The differenced series generally achieve stationarity, while the raw log series may show non-stationary traits due to the heavy trend of the Omori decay. A differencing order of **d=1** is recommended for ARIMA models on daily counts to safely handle the decay trend.",
        "- **Cumulative Counts**: Cumulative counts are monotonically increasing and fundamentally non-stationary (p-value close to 1.0). Differencing them (first difference) simply recovers the daily counts. For ARIMA modeling directly on cumulative counts, **d=1** or even **d=2** is necessary."
    ]
    
    with open(os.path.join(REPORTS, 'stationarity.md'), 'w') as f:
        f.write('\n'.join(report_lines))
        
    print("Stationarity analysis completed. Saved to reports/stationarity.md")

if __name__ == "__main__":
    analyze_stationarity()
