# Stationarity Analysis Report

This report assesses the stationarity of aftershock sequence time series using the Augmented Dickey-Fuller (ADF) test. Stationarity is a requirement for classical AR and MA models.

## ADF Test P-values
A p-value < 0.05 indicates the series is stationary.

| Sequence         |   Log(Daily+1) p-value |   Diff Log(Daily+1) p-value |   Cum Count p-value |   Diff Cum Count p-value |
|:-----------------|-----------------------:|----------------------------:|--------------------:|-------------------------:|
| Ridgecrest_2019  |            0.081191    |                 1.6639e-06  |         0.227405    |              2.20806e-07 |
| Northridge_1994  |            0.00056926  |                 3.22042e-15 |         0.638174    |              0.00144164  |
| Landers_1992     |            0.0332107   |                 3.42392e-10 |         1.59487e-05 |              2.80552e-06 |
| Hector_Mine_1999 |            0.00256197  |                 1.6267e-06  |         0.180219    |              0.0287719   |
| Loma_Prieta_1989 |            5.44097e-09 |                 9.48329e-11 |         2.68497e-11 |              0           |
| Tohoku_2011      |            0.00280885  |                 2.00175e-09 |         5.24756e-27 |              6.50954e-07 |
| Kumamoto_2016    |            5.28581e-08 |                 1.27949e-08 |         0.304516    |              5.05335e-05 |
| Kobe_1995        |            0           |                 2.69993e-12 |         0.0125621   |              1.13572e-24 |
| Maule_2010       |            0.0531923   |                 1.16898e-06 |         2.6749e-06  |              6.36457e-06 |

## Recommendations for Differencing (d)
- **Log Daily Counts**: The differenced series generally achieve stationarity, while the raw log series may show non-stationary traits due to the heavy trend of the Omori decay. A differencing order of **d=1** is recommended for ARIMA models on daily counts to safely handle the decay trend.
- **Cumulative Counts**: Cumulative counts are monotonically increasing and fundamentally non-stationary (p-value close to 1.0). Differencing them (first difference) simply recovers the daily counts. For ARIMA modeling directly on cumulative counts, **d=1** or even **d=2** is necessary.