"""
forecasting_utils.py -- reusable forecasting building blocks (used by stages 04 and 05)
=====================================================================================
Feature construction for the direct multi-horizon ML forecasters and wrappers for ETS / SARIMA.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

STATES = ["Delhi", "Maharashtra", "Tamil Nadu", "West Bengal"]
LOG_EXOG = ("charging_stations", "nsdp_per_capita")


def base_level(y, o):
    """Mean of the last three observed months at origin o (the level the ML model corrects)."""
    return np.mean(y[o - 2:o + 1])


def make_rows(y_by_state: dict, ex: dict, exog: list, h: int, origins, states=None) -> pd.DataFrame:
    """One row per (state, origin o): features known at o for target month t = o + h.

    Target = log(y[t] / base).  y_true / target are NaN when t lies in the future.
    """
    rows = []
    sf_cache = {}
    for s in (states or STATES):
        y = y_by_state[s]
        for o in origins:
            if o < 14:
                continue
            t = o + h
            if o not in sf_cache:                                   # pooled seasonal factors estimated from data <= o
                sf_cache[o] = pooled_seasonal(y_by_state, o + 1)
            sf = sf_cache[o]
            b = base_level(y, o)
            b_ly = np.mean(y[o - 14:o - 11])                      # same 3-month window one year earlier
            r = {"state": s, "o": o, "t": t, "h": h, "base": b, "lvl": np.log(b),
                 "mom0": np.log(y[o] / b), "mom1": np.log(y[o - 1] / b),
                 "yoy_level": np.log(b / b_ly),
                 "seas_move": np.log(y[t - 12] / b_ly),             # last year's move between the same two points
                 "pool_seas": sf[t % 12] - np.mean(sf[[(o - 2) % 12, (o - 1) % 12, o % 12]]),   # pooled seasonal move base->target
                 "m_sin": np.sin(2 * np.pi * ((t % 12) + 1) / 12), "m_cos": np.cos(2 * np.pi * ((t % 12) + 1) / 12),
                 "t_is_march": int((t % 12) == 2),
                 "y_true": y[t] if t < len(y) else np.nan}
            r["target"] = np.log(r["y_true"] / b) if t < len(y) else np.nan
            for c in exog:
                v = ex[s].loc[o, c]
                r[f"x_{c}"] = np.log(v) if c in LOG_EXOG else v
            rows.append(r)
    return pd.DataFrame(rows)


BASE_FEATURES = ["lvl", "mom0", "mom1", "yoy_level", "seas_move", "pool_seas", "m_sin", "m_cos", "t_is_march"]


def ml_feature_names(exog: list) -> list:
    return BASE_FEATURES + [f"x_{c}" for c in exog]


def pooled_seasonal(y_by_state: dict, n: int) -> np.ndarray:
    """Seasonal factors (log scale, sum to zero) pooled over all states using observations 0..n-1 only.

    Each month is compared with the trailing 12-month mean of log-share; the trailing window lags the trend by the
    same amount in every calendar month, so after centring the bias cancels.  Pooling 4 states x 3-4 years makes the
    factors far less noisy than any single state's own seasonal pattern.
    """
    devs = {m: [] for m in range(12)}
    for y in y_by_state.values():
        ly = np.log(np.asarray(y[:n], dtype=float))
        for t in range(11, n):
            devs[t % 12].append(ly[t] - ly[t - 11:t + 1].mean())
    sf = np.array([np.mean(devs[m]) if devs[m] else 0.0 for m in range(12)])
    return sf - sf.mean()


def fit_forecast_psd(y, steps, sf):
    """Pooled-seasonal + damped-trend model: deseasonalise log y with pooled factors, damped Holt on the rest."""
    n = len(y)
    z = np.log(y) - sf[np.arange(n) % 12]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ExponentialSmoothing(z, trend="add", damped_trend=True, initialization_method="estimated").fit()
        zf = m.forecast(steps)
    except Exception:
        zf = np.repeat(z[-1], steps)
    return np.exp(zf + sf[(n + np.arange(steps)) % 12])


def fit_forecast_ets(y, steps):
    """Damped-trend additive-seasonal exponential smoothing on log(y)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ExponentialSmoothing(np.log(y), trend="add", damped_trend=True, seasonal="add", seasonal_periods=12,
                                     initialization_method="estimated").fit()
        return np.exp(m.forecast(steps))
    except Exception:
        return np.repeat(y[-1], steps)


SARIMA_SPECS = {
    "(1,1,1)(0,1,0,12)": ((1, 1, 1), (0, 1, 0, 12)),
    "(0,1,1)(0,1,0,12)": ((0, 1, 1), (0, 1, 0, 12)),
    "(1,1,0)(1,0,0,12)": ((1, 1, 0), (1, 0, 0, 12)),
    "(0,1,1)(1,0,0,12)": ((0, 1, 1), (1, 0, 0, 12)),
}


def fit_forecast_sarima(y, steps, spec):
    order, sorder = SARIMA_SPECS[spec]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SARIMAX(np.log(y), order=order, seasonal_order=sorder, enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False, maxiter=200)
        return np.exp(m.forecast(steps))
    except Exception:
        return np.repeat(y[-1], steps)
