"""
forecast_engine.py -- one self-contained "stand at origin o, use ONLY what is known at o, forecast 6 months" routine
===================================================================================================================
Used by stage 10 (fixed-origin backtest, origin Mar-2026 -> Apr..Sep-2026 compared with actuals) and by stage 05
(final refit on ALL data, origin Sep-2026 -> Oct-2026..Mar-2027).  Everything that is estimated or selected is estimated
on observations with t <= origin only:
  * ML hyper-parameters   : RandomizedSearchCV (25 draws) on two expanding folds that cover the last 12 months before the origin
  * SARIMA specification  : chosen per state on pre-origin validation origins
  * seasonal factors, ETS, SARIMA, PSD fits : history up to the origin
  * covariates            : the vintage panel (release-aligned, no back-filling)
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import loguniform, randint, uniform
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import json

from common import ANOMALY_START, ANOMALY_STATE, SEED, STATES, TARGET, TAB_DIR
from forecasting_utils import (SARIMA_SPECS, base_level, fit_forecast_ets, fit_forecast_psd, fit_forecast_sarima, make_rows,
                               ml_feature_names, pooled_seasonal)

HORIZONS = [1, 2, 3, 4, 5, 6]
SPACES = {
    "Ridge": (make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge()),
              {"ridge__alpha": loguniform(1e-1, 1e3)}, 25),
    "LightGBM": (lgb.LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1),
                 {"n_estimators": randint(60, 300), "num_leaves": randint(3, 12), "learning_rate": loguniform(0.02, 0.15),
                  "min_child_samples": randint(5, 20), "subsample": uniform(0.6, 0.4), "subsample_freq": [1],
                  "colsample_bytree": uniform(0.5, 0.5), "reg_lambda": loguniform(0.5, 30)}, 25),
    "XGBoost": (xgb.XGBRegressor(random_state=SEED, n_jobs=1),
                {"n_estimators": randint(60, 300), "max_depth": randint(2, 4), "learning_rate": loguniform(0.02, 0.15),
                 "min_child_weight": randint(2, 10), "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.5, 0.5),
                 "reg_lambda": loguniform(0.5, 30)}, 25),
}
CLASSIC = ["SARIMA", "ETS", "PSD (pooled seasonal + damped trend)"]
F0_POOL = CLASSIC + ["SARIMA-avg4 (all specs averaged)", "Ridge-F0", "LightGBM-F0", "XGBoost-F0"]
PRIMARY = "Ensemble-F0 top-3 (validation-selected)"   # PRE-DECLARED rule: 3 best exogenous-free models on the 2024 validation window (confirmatory headline)
POOL = "Ensemble-F0 pool (equal weights)"             # EXPLORATORY: equal-weight pool of 7 exogenous-free models (added after seeing test results)
AVG4 = "SARIMA-avg4 (all specs averaged)"             # EXPLORATORY: equal log-average of 4 SARIMA specifications (added after a first run)
HEADLINE = PRIMARY
CLASSIC_ENS = "Ensemble-Classical (SARIMA+ETS+PSD)"


def apply_policy(df: pd.DataFrame, Y_raw: dict, policy: str, adj: dict) -> dict:
    """West Bengal data policy for the Nov-2025..Feb-2026 registration-drive months (see stage 07)."""
    Y = {s: v.copy() for s, v in Y_raw.items()}
    wb = Y[ANOMALY_STATE]
    if policy == "capped":
        for idx in df[(df.state == ANOMALY_STATE) & (df.anomaly_flag == 1)].sort_values("t").t:
            wb[int(idx)] = min(wb[int(idx)], wb[int(idx) - 12:int(idx)].max())
    elif policy == "backlog_removed":
        for t_, r_ in adj.items():
            wb[t_] = r_
    return Y


def run_origin(o: int, Y: dict, EX: dict, fsets: dict, dates, n_jobs: int = -1, verbose: bool = True) -> pd.DataFrame:
    """Forecast months o+1..o+6 for every state using information up to origin index o only.

    Y     : {state: array of EV share, at least o+1 long}   (values after o are never read)
    EX    : {state: DataFrame of vintage covariates indexed by t}   (row o = what is known at the end of month o)
    fsets : {"F0": [], "F1": [...], "F2": [...]}  feature sets to fit for the ML models
    Returns long frame: model, state, h, t, date, y_pred.
    """
    Yo = {s: np.asarray(Y[s][:o + 1], dtype=float) for s in STATES}
    out = []

    def add(model, s, h, p):
        out.append({"model": model, "state": s, "h": h, "t": o + h, "date": dates[o + h], "y_pred": float(p)})

    # ---- 1. ML: tune on pre-origin data only, then predict origin o
    for fs, exog in fsets.items():
        feats = ml_feature_names(exog)
        for h in HORIZONS:
            rows = make_rows(Yo, EX, exog, h, range(14, o + 1))
            tr, te = rows[rows.t <= o].reset_index(drop=True), rows[rows.o == o]
            folds = [(tr.index[tr.t < a].to_numpy(), tr.index[(tr.t >= a) & (tr.t <= b)].to_numpy()) for a, b in [(o - 11, o - 6), (o - 5, o)]]
            for name, (est, space, n_iter) in SPACES.items():
                rs = RandomizedSearchCV(est, space, n_iter=n_iter, cv=folds, scoring="neg_mean_absolute_error",
                                        random_state=SEED, n_jobs=n_jobs, error_score=np.nan).fit(tr[feats], tr["target"])
                pred = te["base"].to_numpy() * np.exp(rs.best_estimator_.predict(te[feats]))
                for s, p in zip(te.state, pred):
                    add(f"{name}-{fs}", s, h, p)
        if verbose:
            print(f"    origin t={o}: ML tuned/fitted for {fs}", flush=True)

    # ---- 2. SARIMA specification per state, chosen on pre-origin validation origins
    choice = {}
    for s in STATES:
        sc = {}
        for spec in SARIMA_SPECS:
            errs = []
            for vo in range(o - 12, o - 5):
                f = fit_forecast_sarima(Yo[s][:vo + 1], 6, spec)
                tgt = np.arange(vo + 1, vo + 7); ok = tgt <= o
                errs += list(np.abs(f[ok] - Yo[s][tgt[ok]]))
            sc[spec] = np.mean(errs)
        choice[s] = min(sc, key=sc.get)

    # ---- 3. statistical models and benchmarks
    sf = pooled_seasonal(Yo, o + 1)
    for s in STATES:
        y = Yo[s]
        ets, sar, psd = fit_forecast_ets(y, 6), fit_forecast_sarima(y, 6, choice[s]), fit_forecast_psd(y, 6, sf)
        avg = np.exp(np.mean([np.log(fit_forecast_sarima(y, 6, sp)) for sp in SARIMA_SPECS], axis=0))
        b, b_ly = base_level(y, o), np.mean(y[o - 14:o - 11])
        for h in HORIZONS:
            t = o + h
            add("Naive (last value)", s, h, y[o]); add("Moving average (3m)", s, h, b)
            add("Seasonal naive", s, h, y[t - 12]); add("Seasonal naive + YoY growth", s, h, y[t - 12] * b / b_ly)
            add("ETS", s, h, ets[h - 1]); add("SARIMA", s, h, sar[h - 1]); add("PSD (pooled seasonal + damped trend)", s, h, psd[h - 1])
            add("SARIMA-avg4 (all specs averaged)", s, h, avg[h - 1])
    f = pd.DataFrame(out)
    top3 = json.loads((TAB_DIR / "04_ensemble_membership.json").read_text())["ensemble_members"][PRIMARY]   # fixed on the 2024 validation window
    if all(m in set(f.model) for m in F0_POOL):                       # ensembles
        for name, members in [(PRIMARY, top3), (POOL, F0_POOL), (CLASSIC_ENS, CLASSIC)]:
            e = f[f.model.isin(members)].groupby(["state", "h", "t", "date"], as_index=False)["y_pred"].mean()
            f = pd.concat([f, e.assign(model=name)], ignore_index=True)
    f.attrs["sarima_choice"] = choice
    return f


def interval_quantiles(pred_path, o: int, model: str = HEADLINE) -> pd.DataFrame:
    """10%/90% quantiles of log(actual/forecast) by horizon, using ONLY errors realised by origin o (target t <= o), WB drive months excluded."""
    ev = pd.read_csv(pred_path)
    e = ev[(ev.model == model) & (ev.flag == 0) & (ev.t <= o)].copy()
    e["log_err"] = np.log(e.y_true / e.y_pred)
    q = e.groupby("h").log_err.quantile([0.10, 0.90]).unstack(); q.columns = ["q10", "q90"]
    q["n_errors"] = e.groupby("h").size()
    return q
