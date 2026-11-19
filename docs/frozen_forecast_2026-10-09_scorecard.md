# Scorecard of the 2026-10-09 frozen forecast (origin Dec-2025, targets Jan-Jun 2026)

This forecast was produced by the earlier pipeline on data to Dec-2025 and its file is not in this repository; the numbers below were recorded when it was scored against the 2026 actuals. It used an earlier covariate treatment (drivers lagged to the origin, charging and GSDP back-filled), so it is not comparable with E1-E3. MAE in percentage points of EV share, 24 state-months per policy.

| Model | capped | capped, excl. West Bengal |
|---|---|---|
| Ridge | 2.19 | 1.92 |
| XGBoost | 2.19 | 2.07 |
| Ensemble | 2.23 | 2.10 |
| LightGBM | 2.37 | 2.25 |
| ETS | 2.45 | 2.33 |
| SARIMA | 2.60 | 2.67 |
| Moving average (3m) | 2.79 | 2.80 |
| Seasonal naive | 3.48 | 2.53 |
| Naive (last value) | 4.79 | 2.43 |

Ensemble bias by state (forecast minus actual, pp): Delhi +1.91, Maharashtra -2.25, Tamil Nadu -1.05, West Bengal -2.49.
The 80% interval covered 71% of actuals.  Maharashtra and Tamil Nadu jumped to 12-14% from March 2026 (a level shift no
history-only model anticipates); West Bengal's Nov-Feb drive was unforecastable.

Note: this forecast used covariates that were not release-aligned (drivers lagged to the origin, charging and GSDP back-filled), so it is not comparable with E1-E3. Later frozen files (`data/frozen_forecasts_made_*.csv`) use release-aligned covariates and are scored by stage 06.
