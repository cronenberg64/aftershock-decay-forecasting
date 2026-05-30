import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'results')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

def evaluate_models():
    metrics_path = os.path.join(RESULTS_DIR, 'forecast_metrics.parquet')
    preds_path = os.path.join(RESULTS_DIR, 'forecast_predictions.parquet')
    
    if not os.path.exists(metrics_path):
        print("Error: Metrics not found. Run run_models.py first.")
        return
        
    df_agg = pd.read_parquet(metrics_path)
    df_pred = pd.read_parquet(preds_path)
    
    # Exclude sparse sequences
    sparse_seqs = ['Kumamoto_2016', 'Kobe_1995']
    df_agg = df_agg[~df_agg['Sequence'].isin(sparse_seqs)]
    df_pred = df_pred[~df_pred['Sequence'].isin(sparse_seqs)]
    
    # We focus on Cumulative for the main narrative ("beat naive on cumulative")
    # but we can do tests on both. Let's do tests on Cumulative_Aggregate
    df_cum = df_agg[df_agg['Target'] == 'Cumulative_Aggregate'].copy()
    
    # Pivot to have models as columns for easier paired testing
    pivot_mae = df_cum.pivot_table(index=['Sequence', 'Origin'], columns='Model', values='MAE').reset_index()
    pivot_mae = pivot_mae.dropna() # Drop any origin where a model failed
    
    print("\n--- Wilcoxon Signed-Rank Test (Cumulative MAE) ---")
    
    tests = [
        ('Omori', 'ETS'),
        ('ETS', 'Naive'),
        ('ARIMA', 'Naive')
    ]
    
    for m1, m2 in tests:
        if m1 in pivot_mae.columns and m2 in pivot_mae.columns:
            stat, pval = wilcoxon(pivot_mae[m1], pivot_mae[m2])
            mean1 = pivot_mae[m1].mean()
            mean2 = pivot_mae[m2].mean()
            better = m1 if mean1 < mean2 else m2
            print(f"{m1} vs {m2}: p-value = {pval:.4f}")
            print(f"  Mean MAE: {m1} ({mean1:.1f}), {m2} ({mean2:.1f}) -> {better} is better on average.")
            
    # Boxplot of MAE
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_cum, x='Model', y='MAE')
    plt.title('Distribution of Cumulative Forecast MAE by Model')
    plt.yscale('log') # Log scale helps see spread
    plt.ylabel('Mean Absolute Error (log scale)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_mae_boxplot.png'))
    plt.close()
    
    # Per-Horizon Degradation
    df_cum_pred = df_pred[df_pred['Target'] == 'Cumulative'].copy()
    df_cum_pred['AbsError'] = np.abs(df_cum_pred['Predicted'] - df_cum_pred['Actual'])
    
    horizon_agg = df_cum_pred.groupby(['Model', 'Horizon'])['AbsError'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=horizon_agg, x='Horizon', y='AbsError', hue='Model', marker='o')
    plt.title('Forecast Error Degradation over Horizon (Cumulative Count)')
    plt.xlabel('Horizon (Days)')
    plt.ylabel('Mean Absolute Error')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'horizon_degradation.png'))
    plt.close()
    
    print(f"\nPlots saved to {PLOTS_DIR}")
    
    # Save a summarized table for the report
    mean_metrics = df_cum.groupby('Model')[['MAE', 'RMSE']].mean().reset_index()
    print("\n--- Mean Cumulative Metrics ---")
    print(mean_metrics.to_markdown(index=False))

if __name__ == "__main__":
    evaluate_models()
