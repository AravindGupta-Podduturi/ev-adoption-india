# Explainable and Forecasting-Based Analysis of Regional EV Adoption in India

AMLBT course project (Goa Institute of Management). Team: Podduturi Aravind Gupta, Gujjari Ruthvik, Ritika Kumar Tulsyan.

Registrations run through **Sep-2026** and covariates through Aug-2026, aligned to their release dates (no look-ahead). The project has a retrospective explanation layer, a rolling-origin forecasting benchmark, a fixed-origin backtest (train to Mar-2026, forecast Apr-Sep 2026) and a final refit with frozen forecasts. Scorecard against every objective: `docs/OBJECTIVES_AND_BACKTEST_SCORECARD.md`. Every reported number: `docs/MASTER_RESULTS_TABLE.md`.

**Question.** Which economic, environmental and infrastructure indicators are associated with regional EV adoption (EV share of new
registrations) in Delhi, Maharashtra, Tamil Nadu and West Bengal - and can we forecast the next six months?

## Two layers, never mixed
| Layer | Stages | Information used | Claim it supports |
|---|---|---|---|
| **Retrospective explanation** | 01-03 | same-month values of all drivers | associations only (fixed-effects regression is primary, SHAP descriptive) |
| **Prospective forecasting** | 00, 04-06 | only what had been *published* at each forecast origin (`data/ev_panel_vintage.csv`) | out-of-sample accuracy and planning ranges |

## Headline results (every number comes from `docs/MASTER_RESULTS_TABLE.md`)

**Model-selection history (not rewritten).** The **primary, pre-declared model** is the *top-3 F0 ensemble* (SARIMA + ETS + XGBoost-F0), chosen on the 2024 validation window before any test result was read. The *equal-weight pool of 7 F0 models* and the *SARIMA specification average (avg4)* were added **after** test results were seen; they are **exploratory** and are labelled so in every table. Nothing was retuned after seeing test results.

**Different evaluations use different samples and are never put in one comparison.** Each has an ID, with exact windows, n of forecasts, horizons, series level, WB treatment, feature set and model status in the master table.

| ID | Test | Targets | Forecasts per model | Level | WB flagged months |
|---|---|---|---|---|---|
| E1 | rolling origin (stage 04) | Jan-2025..Sep-2026 (21 months) | 480 | state-level, h = 1..6 | excluded |
| E2 | fixed-origin backtest, origin Mar-2026 (stage 10) | Apr..Sep-2026 (6 months) | 24 | state-level, h = 1..6 | none in window |
| E3 | pooled four-state series, rolling origin (stage 09) | Jan-2025..Sep-2026 (17 months) | 102 | pooled series, **not India** | excluded |
| E1c | as E1 but WB flagged months included (sensitivity) | Jan-2025..Sep-2026 | 504 | state-level | included |

**E1 (MAE, percentage points of EV share):** naive 2.34; primary top-3 **1.94** (17% better than naive, 90% bootstrap CI excludes zero); SARIMA 1.93; exploratory pool 1.90; exploratory SARIMA-avg4 1.84. Differences among the top models are not statistically distinguishable. 2025 targets (E1a) 1.53, Jan-Sep 2026 targets (E1b) 2.5-2.6 for the top models: 2026 is harder.

**E2 (single origin):** naive **1.70**; SARIMA 1.96; primary top-3 2.41; exploratory pool 2.58. *The pre-declared model did not beat naive here.* Delhi, Maharashtra and Tamil Nadu rose 2-5 pp above seasonal forecasts (a 2026 level shift). The same Apr-Sep 2026 targets scored over all rolling origins give top-3 2.42 vs naive 2.85 (`10_backtest_context_vs_rolling.csv`): one origin is a harsh, noisy test. 80% intervals covered 22 of 24 actuals (92%), because they are wide.

**Why README previously quoted 1.90 and an older comparison 1.94:** 1.90 is the exploratory pool, 1.94 the primary top-3, both on E1. The older comparison file also used a window ending Aug-2026. Both are now in one table with the same window.

* **Covariates do not help once release dates are respected.** F1/F2 *raise* error by 0.04-0.15 pp (E1, `04_covariate_value_F0_F1_F2.csv`).
* **No driver is robust** in the retrospective layer (SHAP + both fixed-effects specs must agree). Charging (4 snapshots) and GSDP (annual) track calendar time; no elasticity or what-if claims.
* **West Bengal Nov-2025..Feb-2026:** evidence consistent with an e-rickshaw registration backlog (excess EVs / excess three-wheelers = 1.01, correlation 0.98; stage 07). **This is not officially verified.** The backlog-removed series is an *assumption-based sensitivity treatment*, not corrected ground truth. In the E2 backtest it made West Bengal *worse* (3.84 vs 0.81 pp), so WB forecasts are shown as a range across as-reported / capped / backlog-removed (`05_forecast_next6m.csv`).
* **80% intervals** cover 75% overall in E1 (88% at h=1, 60% at h=6): long-horizon intervals are too narrow.
* **R-squared is moderate and fades:** E1 pooled R2 0.47 for the primary model; within-state R2 is below 0 (`08_additional_metrics.csv`).
* **Frozen forecasts** (`data/frozen_forecasts_*.csv`, never overwritten, scored by stage 06 when actuals arrive): origin Aug-2026 scored on Sep-2026 (4 forecasts only, anecdotal); later files await actuals.

## Aggregate (pooled) analysis - stage 09
The four states combined into one series (EV registrations / all registrations; Maharashtra dominates; **not India**).
* **Forecast** (as reported, flagged months excluded, test Jan-2025..Sep-2026): spec-averaged SARIMA MAE 0.99 pp, R2 0.53, 41% better than naive; the 3-model ensemble 1.16 pp. R2 falls from 0.65 at 1 month to 0.44 at 6 months. Covariates again hurt (Ridge F1/F2 +0.12/+0.15 pp vs F0, significant).
* **Top-down vs bottom-up:** forecasting the pooled series and summing volume-weighted state forecasts give the same accuracy (MAE 0.99 vs 0.99 pp), so the choice does not matter for accuracy. Errors are smaller in pp than at state level (0.99 vs 1.87) because state-specific noise averages out, but R2 is similar (0.53 vs 0.46): the pooled series is not easier to explain, only smaller in scale.
* **Drivers (same-month, n = 56):** no robust driver. Regression R2 of 0.82 is mostly time: charging stations and NSDP correlate 0.91 / 0.94 with calendar time, a trend-only model beats the driver model on the hold-out (MAE 1.17 vs 2.81 pp), and NSDP's significance disappears once a trend is added.
* Sensitivity: with the backlog-removed series (all months scored) accuracy is lower (best MAE 1.36 pp, R2 0.24) because the adjustment is an estimate.

## Repository layout
```
config/pipeline_config.json   cut-offs, seeds, release-lag rules (change a lag here and re-run)
data/raw/                     original extract before the reference tables were applied
data/external/                reference tables: GSDP per capita by FY, charging snapshots (+definitions), all-India context, CORRECTIONS_LOG.md
data/ev_adoption_ml_dataset.csv   analysis dataset (built by 00a; 232 rows; Sep-Oct 2026 covariates NaN)
data/ev_panel_vintage.csv, release_calendar.csv   covariates as known at each origin (stage 00)
data/frozen_forecasts_made_*.csv  forecasts frozen on the dates shown (scored by stage 06)
00a_apply_reference_data.py   applies data/external tables to data/raw -> analysis dataset
00_build_vintage_panel.py     release-calendar alignment
01_data_audit_eda.py          audit, anomaly screen, EDA figures
02_statistical_diagnostics.py leakage audit, between/within variance, VIF, stationarity
03_explanatory_model_shap.py  SHAP + fixed-effects triangulation (retrospective layer)
04_forecasting_benchmark.py   E1 rolling-origin benchmark (F0/F1/F2), tuning, bootstrap, intervals
07_west_bengal_evidence.py    in-data evidence audit of the WB episode
10_fixed_origin_backtest.py   E2 fixed-origin backtest, origin Mar-2026
05_final_forecast_and_scenarios.py   next-6-month forecast under 3 WB policies + what-if
06_frozen_forecast_scorecard.py      scores frozen files against later actuals
08_additional_metrics.py      R2, percentage errors, skill vs naive
09_aggregate_analysis.py      E3 pooled series: forecast, top-down vs bottom-up, drivers
11_master_results_table.py    docs/MASTER_RESULTS_TABLE.md (single source of every reported number)
12_report_figures.py          report/figs/*.pdf for the paper
forecast_engine.py / common.py / forecasting_utils.py / vintage.py   shared code
run_all.py                    runs all stages in order (about 3.5 min on 2 CPU cores, measured 2026-11-18)
EV_Adoption_End_to_End.ipynb  executed notebook (one cell per stage)
outputs/figures, outputs/tables   all results
report/                       Overleaf (IEEEtran) sources of the paper: main.tex, abstract.tex, body.tex, refs.tex, figs/ (see report/OVERLEAF_GUIDE.md)
docs/                         GITHUB_GUIDE, MASTER_RESULTS_TABLE, OBJECTIVES_AND_BACKTEST_SCORECARD, DATA_SOURCES, frozen-forecast scorecard
scripts/                      data-extraction script for the 2026 NASA POWER weather months
```

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11+
pip install -r requirements.txt
python run_all.py                                       # regenerates every table and figure (about 3.5 min on 2 cores)
python 00a_apply_reference_data.py                      # (part of run_all) rebuilds the dataset from data/raw + data/external
```
Seed 42 throughout; hyper-parameters are tuned on 2022-24 only and then frozen; no model is selected on the test window.

## Method summary
* **Target:** EV share of new registrations per state-month. Direct-count and fuel/category columns are excluded as leaky (stage 02).
* **Vintage rules** (`config/pipeline_config.json`): monthly covariates usable one month later; each charging snapshot has an explicit assumed release month (label+1 for ministry/parliament replies, label+1 for the PIB/ministry snapshots; `data/external/charging_snapshots.csv`), never back-filled; financial-year GSDP is usable 24 months after the FY starts (FY2022-23 from Apr-2024; the projected FY2026-27 value never enters a forecast). These are *assumptions* to be confirmed (see `docs/DATA_SOURCES.md`).
* **Forecast models:** naive family, ETS, per-state SARIMA, SARIMA-average, PSD (pooled seasonal + damped trend), and direct multi-horizon Ridge / LightGBM / XGBoost on feature sets F0 (history only), F1, F2; primary (pre-declared) = validation-selected top-3 F0 ensemble; equal-weight pool and SARIMA-avg4 are exploratory.
* **Inference:** paired block bootstrap by target month; expanding-window empirical log-error quantiles for 80% intervals.

## Known limitations (state these in the paper)
1. Only four states; one metro hub per state proxies air quality and weather.
2. Release dates for charging and NSDP are assumed, and the charging-station definition (operational vs installed-in-last-five-years) is not verified (`docs/DATA_SOURCES.md`).
3. Registrations are used through Sep-2026 (Oct-2026 is a partial month and is dropped; confirm Sep is complete). Sep/Oct covariates were placeholders and are NaN; same-month explanatory stages stop at Aug-2026 (`COVARIATE_CUTOFF`). Charging snapshots use different station definitions (operational vs installed in the last five years) and the Delhi and West Bengal Aug-2026 values are approximate and flagged (`data/external/CORRECTIONS_LOG.md`, `docs/DATA_SOURCES.md`).
4. GSDP for 2023-24 onward is revised/advance/projected, not final; under the 24-month rule only FY2022-23 to FY2024-25 values reach a forecast. The E2 backtest is a single origin (24 forecasts per model).
5. LSTM / Prophet were not run (not installed in the build environment); the exogenous-free linear and tree benchmarks plus SARIMA/ETS cover the comparison.
6. The top models are within sampling noise. The primary rule (top 3 on the 2024 validation window) was fixed before test results were read; the pool and SARIMA-avg4 were added after, so their ranks are exploratory.

## AI-use declaration
Code scaffolding and documentation were produced with AI assistance (Claude). AI-generated content in the report must stay within the 20% course limit; the team verifies results, cites sources and writes the report in its own words.
