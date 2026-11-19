# Do we answer every objective? (scorecard)

Data: registrations Jan-2022..Sep-2026; covariates Jan-2022..Aug-2026; per-capita GSDP by financial year and charging snapshots from `data/external/`.
All numbers below are produced by `python run_all.py` (table names in brackets, all under `outputs/tables/`).

## 1. The forecasting tests (exact samples in `docs/MASTER_RESULTS_TABLE.md`; E1, E2 and E3 are never pooled)

Primary model = pre-declared top-3 F0 ensemble (SARIMA + ETS + XGBoost-F0, chosen on 2024 validation). Pool and SARIMA-avg4 were added after test results were seen and are exploratory.

### E2: fixed-origin backtest, origin Mar-2026, targets Apr-Sep 2026, 24 state-month forecasts (stage 10)
Hyper-parameters, SARIMA specification, seasonal factors and interval calibration use data up to Mar-2026 only; covariates are release-aligned.

| Model | Status | MAE (pp) | pooled R2 | skill vs naive |
|---|---|---|---|---|
| Naive (last value) | benchmark | **1.70** | 0.14 | - |
| SARIMA | comparison | 1.96 | -0.05 | -16% |
| SARIMA-avg4 | exploratory | 2.18 | -0.22 | -28% |
| **Ensemble-F0 top-3** | **primary** | 2.41 | -0.34 | -42% |
| Ensemble-F0 pool | exploratory | 2.58 | -0.59 | -52% |

**Honest reading: the pre-declared model did not beat naive on this single origin.** Delhi, Maharashtra and Tamil Nadu shares rose above seasonal forecasts (`10_backtest_bias_by_state.csv`). Same target months over all rolling origins: top-3 2.42 vs naive 2.85 (`10_backtest_context_vs_rolling.csv`), so the result is origin-specific. 80% intervals covered 22 of 24 actuals (wide, not sharp). Nothing was retuned afterwards. West Bengal: the backlog-removed treatment worsened the WB backtest (3.84 vs 0.81 pp), so it is a sensitivity case, not a correction.

### E1: rolling origin, Jan-2025..Sep-2026, 480 forecasts per model, h = 1..6, flagged WB months excluded (stage 04)
| Model | Status | MAE (pp) |
|---|---|---|
| SARIMA-avg4 | exploratory | 1.84 |
| Ensemble-F0 pool | exploratory | 1.90 |
| SARIMA | comparison | 1.93 |
| **Ensemble-F0 top-3** | **primary** | **1.94** (R2 0.47; 17% better than naive; 90% bootstrap CI excludes zero) |
| Naive | benchmark | 2.34 |

2025 targets 1.53, Jan-Sep 2026 targets harder (E1a/E1b in the master table). 80% intervals covered 75% overall (88% at h=1, 60% at h=6). Top models are not statistically distinguishable.
**1.90 vs 1.94:** exploratory pool vs primary top-3, same E1 sample. An older comparison file used a window ending Aug-2026.

### C. Prospective check (stage 06)
Frozen files in `data/` are scored when actuals arrive. The origin-Aug file scored Sep-2026 only (4 forecasts: anecdotal). Later files await Oct-2026+ actuals.

### D. Final model: all data to Sep-2026, forecast Oct-2026..Mar-2027 (stage 05)
Primary model, EV share %, West Bengal shown with backlog removed (assumption-based; see the range below):

| Month | Delhi | Maharashtra | Tamil Nadu | West Bengal (backlog removed) | WB as reported |
|---|---|---|---|---|---|
| Oct-26 | 14.2 | 10.5 | 9.8 | 8.9 | 9.1 |
| Nov-26 | 13.7 | 8.2 | 10.1 | 8.0 | 14.2 |
| Dec-26 | 19.5 | 10.3 | 12.6 | 11.0 | 26.1 |
| Jan-27 | 14.6 | 10.0 | 9.4 | 10.1 | 20.4 |
| Feb-27 | 15.4 | 10.7 | 10.6 | 10.6 | 18.4 |
| Mar-27 | 19.3 | 16.3 | 16.2 | 17.9 | 18.8 |

Intervals and the capped policy are in `05_forecast_next6m.csv`. **Caution:** the E2 backtest shows these models under-forecast level shifts and repeat the March fiscal-year-end spike. Read ranges, not points. Frozen in `data/frozen_forecasts_made_2026-11-18.csv`.

## 2. Objectives and research questions

| # | Objective / RQ | Status | Evidence | Caveat |
|---|---|---|---|---|
| (i) | Relevance of regional disparities | Met | Stage 01: EV share of registrations Apr-2025..Sep-2026 is 14.5% in Delhi vs 9.7-10.3% in Tamil Nadu, Maharashtra and West Bengal (WB includes the backlog months); public chargers in Aug-2026 range from ~1,200 (WB) to 6,985 (Maharashtra) | - |
| (ii) | Originality: explanation coupled with out-of-sample forecasting | Met | Two layers (stages 01-03 vs 04-10), 3 feature sets, fixed-origin backtest + rolling test + frozen check | The coupling's main finding is negative: drivers do not improve forecasts |
| (iii) | Robust, honestly validated results | Met, with negative findings reported | Release-aligned covariates; backtest loses to naive (reported); intervals too narrow at long horizons; top models statistically tied | Primary model is the pre-declared top-3; the pool and SARIMA-avg4 are exploratory (added after seeing results) |
| (iv) | Reproducibility | Met | `run_all.py`, config, reference tables in `data/external/`, raw extract kept in `data/raw/`, executed notebook, seed 42 | Source register in `docs/DATA_SOURCES.md`; five items need team attestation (section 6) |
| RQ1 | How has adoption evolved across states? | Answered | Stage 01 figs 01-06; WB episode, evidence consistent with an e-rickshaw backlog (stage 07, not officially verified) | Four states only |
| RQ2 | Which economic/environmental/infrastructure variables are most associated? | Answered: none robustly | Stage 03: SHAP ranks charging stations and GSDP first, but all 8 drivers fail the SHAP + fixed-effects robustness test; both are strongly correlated with calendar time | Charging has only 4 snapshots and GSDP changes once a year: association, not causation |
| RQ3 | Do models with exogenous drivers forecast better than classical baselines? | Answered: no | Rolling: adding drivers raises MAE by 0.04-0.15 pp (significant in 4 of 6 comparisons, `04_covariate_value_F0_F1_F2.csv`). Backtest: mixed, n = 24, no consistent benefit (`10_backtest_covariate_value.csv`) | - |
| RQ4 | Can explainable ML give region-specific insights? | Partly | SHAP state contributions and waterfalls (figs 14-15) describe each state's profile | Hold-out R2 of the explanatory models is low (0.06-0.21); present as descriptive |
