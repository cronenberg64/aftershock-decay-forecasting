import nbformat as nbf
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(PROJECT_ROOT, 'notebooks')
os.makedirs(NB_DIR, exist_ok=True)

def build_notebook():
    nb = nbf.v4.new_notebook()

    cells = []

    # Helper function to add markdown and code cells
    def add_md(text):
        cells.append(nbf.v4.new_markdown_cell(text))
        
    def add_code(text):
        cells.append(nbf.v4.new_code_cell(text))

    add_md("""# 1. Project Title and Objective
**Title**: Forecasting Earthquake Aftershock Decay: A Comparison of Classical Time-Series Methods and the Omori Law

**Objective**: To forecast the cumulative aftershock count following major mainshocks using classical time-series models, and to compare generic statistical methods against the physics-based Omori law.""")

    add_md("""# 2. Project Questions
- Can classical time-series models (Autoregressive (AR), Moving Average (MA), ARIMA, and Exponential Smoothing) forecast aftershock decay?
- Does the physics-based Omori law outperform generic statistical trend models?
- How does forecast accuracy change with training window length and forecast horizon?""")

    add_md("""# 3. Select and Download Data
Data was sourced from the United States Geological Survey (USGS) Advanced National Seismic System (ANSS) Comprehensive Catalog (ComCat). We queried for 9 major earthquakes (mainshocks) and their subsequent events within a 90-day window, applying a regional completeness magnitude filter.

The following code loads the data manifest to show the parameters used for each download:""")

    add_code("""import pandas as pd
import numpy as np
import os
from IPython.display import display, Markdown

PROJECT_ROOT = os.path.dirname(os.getcwd())
MANIFEST_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'MANIFEST.md')

if os.path.exists(MANIFEST_PATH):
    with open(MANIFEST_PATH, 'r') as f:
        manifest_text = f.read()
    display(Markdown(manifest_text))
else:
    print("Manifest not found.")
""")

    add_md("""# 4. Explain Your Data
An **aftershock sequence** is a series of smaller earthquakes that follow a larger earthquake (the **mainshock**) in the same general area. They occur as the crust adjusts to the mainshock's displacement. The frequency of aftershocks typically decays over time.

Our dataset consists of 9 distinct aftershock sequences. For each sequence, we have constructed time-series data of daily aftershock counts and the cumulative sum of these counts over 90 days. We applied a completeness magnitude ($M_c$) threshold, which ensures we only count earthquakes large enough to be reliably detected by the regional seismic network.""")

    add_md("""# 5. Descriptive Statistics
Let's look at the summary statistics for each sequence, including the duration, completeness magnitude ($M_c$), total aftershocks recorded, and the maximum daily rate.""")

    add_code("""SUMMARY_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'sequence_summary.csv')
if os.path.exists(SUMMARY_PATH):
    summary_df = pd.read_csv(SUMMARY_PATH)
    display(summary_df)
else:
    print("Summary data not found.")
""")

    add_md("""Below, we visualize the cumulative aftershock counts over time for one of our sequences, overlaid with the fitted Omori law (the empirical law describing aftershock decay).""")

    add_code("""from IPython.display import Image
Image(filename=os.path.join(PROJECT_ROOT, 'plots', 'omori_fit_Ridgecrest_2019.png'))
""")

    add_md("""# 6. Data Transformations
To prepare the data for modeling, several transformations were applied:
1. **Completeness Filtering**: Only events with magnitude $\geq M_c$ were kept.
2. **Aggregation**: Events were binned into daily counts and cumulative sums.
3. **Log Transformation**: For models predicting daily counts, we used $\log(\text{count} + 1)$ to stabilize variance and prevent divergence.
4. **Differencing**: Stationarity tests (Augmented Dickey-Fuller) showed that cumulative counts are highly non-stationary. A differencing order of $d=1$ was applied to ensure stationary residuals when required.""")

    add_code("""STAT_REPORT = os.path.join(PROJECT_ROOT, 'reports', 'stationarity.md')
if os.path.exists(STAT_REPORT):
    with open(STAT_REPORT, 'r') as f:
        stat_text = f.read()
    display(Markdown(stat_text))
""")

    add_md("""# 7. Develop Time Series Models
We applied the following suite of classical time-series models and a domain-specific baseline using rolling-origin cross-validation (training on the first 20, 30, and 40 days; predicting the next 20 days):

**Classical Models:**
- **AR(p)**: Pure Autoregressive model.
- **MA(q)**: Pure Moving Average model.
- **ARIMA(p,d,q)**: Auto-Regressive Integrated Moving Average (parameters auto-selected).
- **ETS**: Exponential Smoothing (Holt-Winters) with a damped trend.

**Baselines:**
- **Naive**: Predicts the last observed value indefinitely.
- **Mean**: Predicts the mean daily rate indefinitely.
- **Omori Law**: A physics-based extrapolation using the cumulative Omori integral $N(t) = \frac{K}{1-p} \left[ (t+c)^{1-p} - c^{1-p} \right]$ fitted via nonlinear least squares.""")

    add_md("""# 8. Execute Your Project
We evaluated the models by calculating the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on the forecasted cumulative counts.""")

    add_code("""RESULTS_PATH = os.path.join(PROJECT_ROOT, 'data', 'results', 'forecast_metrics.parquet')
if os.path.exists(RESULTS_PATH):
    results_df = pd.read_parquet(RESULTS_PATH)
    cum_results = results_df[results_df['Target'] == 'Cumulative_Aggregate']
    mean_metrics = cum_results.groupby('Model')[['MAE', 'RMSE']].mean().reset_index()
    display(mean_metrics.sort_values('MAE'))
""")

    add_md("""The table above shows the pooled average errors across all origins and sequences. The physics-based **Omori model significantly outperforms generic statistical trend models** like ETS and ARIMA.

Below is the distribution of the forecast errors for each model:""")

    add_code("""Image(filename=os.path.join(PROJECT_ROOT, 'plots', 'model_mae_boxplot.png'))""")

    add_md("""And here is how the forecast accuracy degrades as we look further into the future (up to 20 days ahead):""")

    add_code("""Image(filename=os.path.join(PROJECT_ROOT, 'plots', 'horizon_degradation.png'))""")

    add_md("""# 9. Discuss Limitations and Future Directions
**Limitations**:
- We only evaluated 9 sequences.
- Completeness magnitude ($M_c$) varies by region, which changes the total absolute counts and makes sequences difficult to compare directly on scale.
- Early-time incompleteness: The first hours after a major earthquake often miss smaller aftershocks due to overlapping seismic waves.
- Beating a Naive baseline on a cumulative sequence is partly a structural advantage (since cumulative counts always trend up, flat Naive forecasts perform poorly over long horizons).

**Future Directions**:
- **Covariate-Aware Omori Law**: Modeling the Omori decay parameters ($K$, $c$, $p$) as functions of mainshock properties like magnitude, depth, and fault type.
- Investigating the Epidemic-Type Aftershock Sequence (ETAS) model to account for secondary aftershocks.
- Deploying real-time operational forecasting updates.""")

    add_md("""# 10. Prepare Report
This notebook serves as the final executed report covering the entire end-to-end workflow, as required by the capstone rubric.""")

    nb['cells'] = cells

    with open(os.path.join(NB_DIR, 'capstone_main.ipynb'), 'w') as f:
        nbf.write(nb, f)

    print("Notebook generated successfully at notebooks/capstone_main.ipynb")

if __name__ == "__main__":
    build_notebook()
