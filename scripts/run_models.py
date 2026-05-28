import os
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

MAINSHOCKS = [
    'Ridgecrest_2019', 'Northridge_1994', 'Landers_1992', 
    'Hector_Mine_1999', 'Loma_Prieta_1989', 'Tohoku_2011', 
    'Kumamoto_2016', 'Kobe_1995', 'Maule_2010'
]

def cumulative_omori(t, K, c, p):
    epsilon = 1e-5
    t_c = t + c
    return np.where(
        np.abs(p - 1.0) < epsilon,
        K * np.log(t_c / c),
        (K / (1 - p)) * (t_c**(1 - p) - c**(1 - p))
    )

def run_forecasts():
    origins = [20, 30, 40]
    horizon = 20
    results = []

    for name in MAINSHOCKS:
        parquet_path = os.path.join(DATA_PROCESSED, f"{name}_sequences_daily.parquet")
        if not os.path.exists(parquet_path):
            continue
            
        print(f"Running models for {name}...")
        df = pd.read_parquet(parquet_path)
        
        # Ensure we have enough data
        if len(df) < 60:
            print(f"  Skipping {name}: sequence too short.")
            continue
            
        daily = df['daily_count'].values
        cumulative = df['cumulative_count'].values
        
        for origin in origins:
            # Daily setup
            d_train = daily[:origin]
            d_test = daily[origin:origin+horizon]
            d_train_log = np.log1p(d_train)
            
            # Cumulative setup
            c_train = cumulative[:origin]
            c_test = cumulative[origin:origin+horizon]
            
            # ---------------------------
            # Daily Models
            # ---------------------------
            d_preds = {}
            
            # Naive
            d_preds['Naive'] = np.full(horizon, d_train[-1])
            
            # Mean
            d_preds['Mean'] = np.full(horizon, np.mean(d_train))
            
            # AR(1)
            try:
                ar_model = AutoReg(d_train_log, lags=1).fit()
                pred_log = ar_model.predict(start=len(d_train_log), end=len(d_train_log)+horizon-1)
                d_preds['AR'] = np.expm1(pred_log)
            except:
                d_preds['AR'] = np.full(horizon, np.nan)
                
            # MA(1) - ARIMA(0,0,1)
            try:
                ma_model = pm.ARIMA(order=(0,0,1), suppress_warnings=True).fit(d_train_log)
                pred_log = ma_model.predict(n_periods=horizon)
                d_preds['MA'] = np.expm1(pred_log)
            except:
                d_preds['MA'] = np.full(horizon, np.nan)
                
            # ARIMA(p,d,q)
            try:
                arima_model = pm.auto_arima(d_train_log, max_p=3, max_q=3, max_d=1, 
                                            approximation=True, stepwise=True, 
                                            seasonal=False, suppress_warnings=True)
                pred_log = arima_model.predict(n_periods=horizon)
                d_preds['ARIMA'] = np.expm1(pred_log)
            except:
                d_preds['ARIMA'] = np.full(horizon, np.nan)
                
            # ETS
            try:
                ets_model = ExponentialSmoothing(d_train_log, trend='add', damped_trend=True, initialization_method="estimated").fit()
                pred_log = ets_model.forecast(horizon)
                d_preds['ETS'] = np.expm1(pred_log)
            except:
                d_preds['ETS'] = np.full(horizon, np.nan)

            # ---------------------------
            # Cumulative Models
            # ---------------------------
            c_preds = {}
            
            # Naive (flat line from last cumulative)
            c_preds['Naive'] = np.full(horizon, c_train[-1])
            
            # Mean (linear trend using mean daily rate)
            mean_rate = np.mean(d_train)
            c_preds['Mean'] = c_train[-1] + np.cumsum(np.full(horizon, mean_rate))
            
            # AR(1) directly on cumulative
            try:
                ar_c = AutoReg(c_train, lags=1).fit()
                c_preds['AR'] = ar_c.predict(start=len(c_train), end=len(c_train)+horizon-1)
            except:
                c_preds['AR'] = np.full(horizon, np.nan)
                
            # MA(1) directly on cumulative
            try:
                ma_c = pm.ARIMA(order=(0,0,1), suppress_warnings=True).fit(c_train)
                c_preds['MA'] = ma_c.predict(n_periods=horizon)
            except:
                c_preds['MA'] = np.full(horizon, np.nan)
                
            # ARIMA
            try:
                arima_c = pm.auto_arima(c_train, max_p=3, max_q=3, max_d=1, 
                                        approximation=True, stepwise=True, 
                                        seasonal=False, suppress_warnings=True)
                c_preds['ARIMA'] = arima_c.predict(n_periods=horizon)
            except:
                c_preds['ARIMA'] = np.full(horizon, np.nan)
                
            # ETS
            try:
                ets_c = ExponentialSmoothing(c_train, trend='add', damped_trend=True, initialization_method="estimated").fit()
                c_preds['ETS'] = ets_c.forecast(horizon)
            except:
                c_preds['ETS'] = np.full(horizon, np.nan)
                
            # Omori (fit on cumulative)
            try:
                t_train = np.arange(1, len(c_train) + 1)
                t_test = np.arange(len(c_train) + 1, len(c_train) + 1 + horizon)
                p0 = [max(1, c_train[-1]/np.log(len(c_train)+1)), 0.1, 1.05]
                bounds = ([0.01, 1e-3, 0.01], [np.inf, np.inf, 3.0])
                popt, _ = curve_fit(cumulative_omori, t_train, c_train, p0=p0, bounds=bounds)
                c_preds['Omori'] = cumulative_omori(t_test, *popt)
                
                # Derive daily from Omori cumulative (difference)
                # omori daily rate = N(t) - N(t-1). We calculate N(t_test) and prepend N(last_train)
                full_omori_c = cumulative_omori(np.concatenate(([t_train[-1]], t_test)), *popt)
                d_preds['Omori'] = np.diff(full_omori_c)
            except:
                c_preds['Omori'] = np.full(horizon, np.nan)
                d_preds['Omori'] = np.full(horizon, np.nan)
                
            # Calculate metrics
            for model in ['Naive', 'Mean', 'AR', 'MA', 'ARIMA', 'ETS', 'Omori']:
                # Daily metrics
                if not np.any(np.isnan(d_preds[model])):
                    d_mae = mean_absolute_error(d_test, d_preds[model])
                    d_rmse = root_mean_squared_error(d_test, d_preds[model])
                else:
                    d_mae, d_rmse = np.nan, np.nan
                    
                # Cumulative metrics
                if not np.any(np.isnan(c_preds[model])):
                    c_mae = mean_absolute_error(c_test, c_preds[model])
                    c_rmse = root_mean_squared_error(c_test, c_preds[model])
                else:
                    c_mae, c_rmse = np.nan, np.nan
                    
                # Store horizon degradation (MAE per day)
                for h_idx in range(horizon):
                    results.append({
                        'Sequence': name,
                        'Origin': origin,
                        'Horizon': h_idx + 1,
                        'Model': model,
                        'Target': 'Daily',
                        'Actual': d_test[h_idx],
                        'Predicted': d_preds[model][h_idx] if not np.any(np.isnan(d_preds[model])) else np.nan
                    })
                    results.append({
                        'Sequence': name,
                        'Origin': origin,
                        'Horizon': h_idx + 1,
                        'Model': model,
                        'Target': 'Cumulative',
                        'Actual': c_test[h_idx],
                        'Predicted': c_preds[model][h_idx] if not np.any(np.isnan(c_preds[model])) else np.nan
                    })
                    
                # Store aggregate window errors for evaluation
                results.append({
                    'Sequence': name,
                    'Origin': origin,
                    'Model': model,
                    'Target': 'Daily_Aggregate',
                    'MAE': d_mae,
                    'RMSE': d_rmse
                })
                results.append({
                    'Sequence': name,
                    'Origin': origin,
                    'Model': model,
                    'Target': 'Cumulative_Aggregate',
                    'MAE': c_mae,
                    'RMSE': c_rmse
                })

    # Save to parquet
    # First, separate the item-level predictions from aggregate errors
    df_all = pd.DataFrame(results)
    
    df_agg = df_all.dropna(subset=['MAE']).drop(columns=['Horizon', 'Actual', 'Predicted'])
    df_pred = df_all.dropna(subset=['Horizon']).drop(columns=['MAE', 'RMSE'])
    
    df_agg.to_parquet(os.path.join(RESULTS_DIR, 'forecast_metrics.parquet'))
    df_pred.to_parquet(os.path.join(RESULTS_DIR, 'forecast_predictions.parquet'))
    
    print("Forecasting completed. Results saved to data/results/")

if __name__ == "__main__":
    run_forecasts()
