import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
PLOTS = os.path.join(PROJECT_ROOT, 'plots')
os.makedirs(PLOTS, exist_ok=True)

MAINSHOCKS_META = {
    'Ridgecrest_2019': {'magnitude': 7.1, 'depth': 8.0, 'region': 'CA'},
    'Northridge_1994': {'magnitude': 6.7, 'depth': 18.2, 'region': 'CA'},
    'Landers_1992': {'magnitude': 7.3, 'depth': 1.1, 'region': 'CA'},
    'Hector_Mine_1999': {'magnitude': 7.1, 'depth': 5.0, 'region': 'CA'},
    'Loma_Prieta_1989': {'magnitude': 6.9, 'depth': 16.8, 'region': 'CA'},
    'Tohoku_2011': {'magnitude': 9.0, 'depth': 29.0, 'region': 'Japan'},
    'Kumamoto_2016': {'magnitude': 7.0, 'depth': 11.0, 'region': 'Japan'},
    'Kobe_1995': {'magnitude': 6.9, 'depth': 16.0, 'region': 'Japan'},
    'Maule_2010': {'magnitude': 8.8, 'depth': 35.0, 'region': 'Chile'}
}

def cumulative_omori(t, K, c, p):
    epsilon = 1e-5
    t_c = t + c
    # Use np.where to allow vectorization cleanly
    return np.where(
        np.abs(p - 1.0) < epsilon,
        K * np.log(t_c / c),
        (K / (1 - p)) * (t_c**(1 - p) - c**(1 - p))
    )

def fit_omori():
    results = []
    
    for name, meta in MAINSHOCKS_META.items():
        parquet_path = os.path.join(DATA_PROCESSED, f"{name}_sequences_daily.parquet")
        if not os.path.exists(parquet_path):
            print(f"Skipping {name}: Sequence data not found.")
            continue
            
        print(f"Fitting Omori for {name}...")
        df = pd.read_parquet(parquet_path)
        
        t_data = df['days_since_mainshock'].values
        cumulative_counts = df['cumulative_count'].values
        
        # Skip if too few points (e.g. Kobe has only 13 total events, maybe they are mostly on day 1)
        if len(t_data) < 3 or cumulative_counts[-1] == 0:
            print(f"  Skipping {name}: insufficient data")
            continue
            
        # Initial guess
        # K ~ rate at 1 day, let's use the final count over log time roughly
        p0 = [max(1, cumulative_counts[-1]/np.log(90)), 0.1, 1.05]
        bounds = ([0.01, 1e-3, 0.01], [np.inf, np.inf, 3.0])
        
        try:
            popt, _ = curve_fit(cumulative_omori, t_data, cumulative_counts, p0=p0, bounds=bounds)
            K, c, p = popt
            
            pred = cumulative_omori(t_data, K, c, p)
            ss_res = np.sum((cumulative_counts - pred)**2)
            ss_tot = np.sum((cumulative_counts - np.mean(cumulative_counts))**2)
            r2 = 1 - (ss_res / ss_tot)
            
            results.append({
                'sequence': name,
                'mainshock_magnitude': meta['magnitude'],
                'mainshock_depth': meta['depth'],
                'region': meta['region'],
                'K': K,
                'c': c,
                'p': p,
                'r_squared': r2
            })
            
            # Plot
            plt.figure(figsize=(8, 5))
            plt.plot(t_data, cumulative_counts, marker='o', linestyle='none', label='Actual Cumulative')
            plt.plot(t_data, pred, color='red', label=f'Omori Fit\nK={K:.2f}, c={c:.2f}, p={p:.2f}')
            plt.title(f'{name} - Cumulative Omori Law Fit')
            plt.xlabel('Days since mainshock')
            plt.ylabel('Cumulative Aftershocks')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS, f'omori_fit_{name}.png'))
            plt.close()
            
            print(f"  Success! R^2: {r2:.4f}, p: {p:.2f}")
            
        except Exception as e:
            print(f"  Omori fit failed for {name}: {e}")
            
    # Save parameters
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(DATA_PROCESSED, 'omori_parameters.csv'), index=False)
        print("\nSaved Omori parameters to omori_parameters.csv")
    else:
        print("\nNo fits succeeded.")

if __name__ == "__main__":
    fit_omori()
