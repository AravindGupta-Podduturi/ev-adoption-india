"""
vintage.py -- release-calendar ("vintage") alignment of covariates and West Bengal backlog adjustment
====================================================================================================
Future-information back-filling must be avoided: a covariate value that was only
published months after the forecast origin must not be visible at that origin.  This module implements the rules
stored in config/pipeline_config.json:

* monthly covariates  : usable one month after the month they describe (lag 1)
* charging stations   : each snapshot has an EXPLICIT assumed release month (data/external/charging_snapshots.csv); never back-filled
* NSDP per capita     : financial-year value is usable 24 months after the financial year starts (FY2022-23 -> Apr-2024)

Everything returned here is "as known at the END of month m" so a forecast origin at month m may use it.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import BASE_DIR, STATES, ANOMALY_STATE, ANOMALY_START, ANOMALY_END, TARGET

CONFIG = json.loads((BASE_DIR / "config" / "pipeline_config.json").read_text())
RULES = CONFIG["vintage_rules"]
MONTHLY_COLS = RULES["monthly_covariates"]["columns"]
MONTHLY_LAG = RULES["monthly_covariates"]["lag_months"]
CHARGING_LAG = RULES["charging_stations"]["release_lag_months"]
NSDP_LAG_MONTHS = RULES["nsdp_per_capita"]["release_lag_months"]


EXT = BASE_DIR / "data" / "external"


def charging_snapshots(raw=None) -> pd.DataFrame:
    """Charging-station snapshots from data/external/charging_snapshots.csv with their EXPLICIT assumed release month."""
    t = pd.read_csv(EXT / "charging_snapshots.csv", parse_dates=["label_month", "assumed_release_month"])
    return t[["state", "label_month", "value", "approximate", "source", "assumed_release_month"]]


def nsdp_values(raw=None) -> pd.DataFrame:
    """Financial-year per-capita GSDP from data/external/gsdp_per_capita_fy.csv; usable NSDP_LAG_MONTHS after the FY starts."""
    t = pd.read_csv(EXT / "gsdp_per_capita_fy.csv", parse_dates=["fy_start"])
    t["assumed_release_month"] = t.fy_start + pd.DateOffset(months=NSDP_LAG_MONTHS)
    return t.rename(columns={"value_inr": "value"})[["state", "fy", "fy_start", "value", "status", "assumed_release_month"]]


def build_vintage_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Return the panel with *_asof columns = what a forecaster standing at the end of month `date` could see."""
    raw = raw.sort_values(["state", "date"]).reset_index(drop=True)
    out = raw[["state", "date", "year", "month", TARGET]].copy()
    snaps, nsdp = charging_snapshots(raw), nsdp_values(raw)
    for c in MONTHLY_COLS:
        out[c] = raw.groupby("state")[c].shift(MONTHLY_LAG)
    ch, ns, ch_label, ns_label, ns_status, ch_apx = [], [], [], [], [], []
    for s, d in zip(out.state, out.date):
        sn = snaps[(snaps.state == s) & (snaps.assumed_release_month <= d)]
        if len(sn):
            r = sn.sort_values("label_month").iloc[-1]; ch.append(r.value); ch_label.append(r.label_month); ch_apx.append(int(r.approximate))
        else:
            ch.append(np.nan); ch_label.append(pd.NaT); ch_apx.append(np.nan)
        nn = nsdp[(nsdp.state == s) & (nsdp.assumed_release_month <= d)]
        if len(nn):
            r = nn.sort_values("fy_start").iloc[-1]; ns.append(r.value); ns_label.append(r.fy); ns_status.append(r.status)
        else:
            ns.append(np.nan); ns_label.append(np.nan); ns_status.append(np.nan)
    out["charging_stations"], out["charging_snapshot_label"], out["charging_approx"] = ch, ch_label, ch_apx
    out["nsdp_per_capita"], out["nsdp_fy_label"], out["nsdp_status_asof"] = ns, ns_label, ns_status
    return out


def release_calendar(raw=None) -> pd.DataFrame:
    snaps, nsdp = charging_snapshots(), nsdp_values()
    rows = []
    for (lab, src), g in snaps.groupby(["label_month", "source"]):
        rows.append({"variable": "charging_stations", "label": f"{lab:%Y-%m}", "assumed_release": f"{g.assumed_release_month.iloc[0]:%Y-%m}",
                     "definition": src + ("; some state values approximate" if g.approximate.any() else ""),
                     "status": "ASSUMED release month (not the date printed on the source) - confirm; definitions differ across replies (operational vs installed)"})
    for fy, g in nsdp.groupby("fy"):
        rows.append({"variable": "nsdp_per_capita", "label": f"FY{fy} ({g.status.iloc[0]})", "assumed_release": f"{g.assumed_release_month.iloc[0]:%Y-%m}",
                     "definition": "per-capita GSDP at current prices, financial year (Apr-Mar)",
                     "status": "official/state ESTIMATE" if g.status.iloc[0] != "Actual" else "published actual (24-month lag rule, conservative)"})
    for c in MONTHLY_COLS:
        rows.append({"variable": c, "label": "each month", "assumed_release": f"month + {MONTHLY_LAG}", "definition": "monthly aggregate",
                     "status": "rule from config"})
    return pd.DataFrame(rows)


def backlog_adjusted_rate(df: pd.DataFrame) -> dict:
    """West Bengal backlog adjustment (data-driven; see stage 07). ASSUMPTION-BASED sensitivity treatment, NOT a corrected ground-truth series:
    the evidence is consistent with an e-rickshaw registration backlog but is not officially verified.

    baseline = median three-wheeler registrations in the 12 months before the episode;
    excess  = max(0, three_wheeler - baseline) in the flagged months;
    adjusted rate = (EV - excess) / (total - excess).  Returns {t_index: adjusted_rate}.
    """
    w = df[df.state == ANOMALY_STATE].sort_values("date").reset_index(drop=True)
    pre = w[(w.date < ANOMALY_START) & (w.date >= ANOMALY_START - pd.DateOffset(months=12))]
    base = float(pre.three_wheeler.median())
    out = {}
    for i, r in w[w.anomaly_flag == 1].iterrows():
        excess = max(0.0, r.three_wheeler - base)
        ev_adj = max(r.ev_count - excess, 0.0)
        out[int(r.t)] = ev_adj / max(r.vehicle_total - excess, 1.0)
    return out, base
