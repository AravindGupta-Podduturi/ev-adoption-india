"""
00a_apply_reference_data.py -- rebuild the model dataset from the ORIGINAL extract plus explicit reference tables
==================================================================================================================
Replaces the two infrastructure/income columns with documented reference tables (documented in data/external/CORRECTIONS_LOG.md):
  data/external/gsdp_per_capita_fy.csv        per-capita GSDP, current prices, by FINANCIAL YEAR (Actual/Revised/Advance/Projected)
  data/external/charging_snapshots.csv        public-charging-station snapshots (label month, value, approximate flag, assumed release)
  data/external/charging_all_india_context.csv  all-India totals (context only, not a model input)
Month mapping
  * NSDP: financial year FY(Y-(Y+1)) applies to Apr-Y .. Mar-(Y+1).  Jan-Mar 2022 belong to FY2021-22, which was not supplied, so they
    take the FY2022-23 value and are flagged nsdp_backfilled=1 (they are excluded from every vintage-aligned forecast input anyway).
  * Charging: step function - a month takes the latest snapshot whose label month <= that month.  Jan-Nov 2022 precede the first snapshot,
    take its value and are flagged charging_backfilled=1.
Cut-offs (common.py): registrations are used through Sep-2026; monthly covariates (air quality, weather, fuel prices) only through Aug-2026 -
Sep/Oct-2026 values in the original file were identical across states (placeholders) and are set to NaN here.
Input : data/raw/ev_adoption_ml_dataset_original.csv   Output: data/ev_adoption_ml_dataset.csv
Run from the project root:  python 00a_apply_reference_data.py
"""
from common import BASE_DIR, COVARIATE_CUTOFF
import numpy as np
import pandas as pd

D = BASE_DIR / "data"
raw = pd.read_csv(D / "raw" / "ev_adoption_ml_dataset_original.csv", parse_dates=["date"])
gs = pd.read_csv(D / "external" / "gsdp_per_capita_fy.csv", parse_dates=["fy_start"])
ch = pd.read_csv(D / "external" / "charging_snapshots.csv", parse_dates=["label_month", "assumed_release_month"])

out = raw.copy()
# ---- NSDP by financial year
fy_start = out.date.where(out.date.dt.month >= 4, out.date - pd.DateOffset(years=1)).dt.year  # FY start calendar year
fy_start = np.where(out.date.dt.month >= 4, out.date.dt.year, out.date.dt.year - 1)
out["nsdp_fy_start"] = fy_start
first_fy = int(gs.fy_start.dt.year.min())
backfilled = out.nsdp_fy_start < first_fy
out.loc[backfilled, "nsdp_fy_start"] = first_fy
gs["nsdp_fy_start"] = gs.fy_start.dt.year
out = out.merge(gs.rename(columns={"fy": "nsdp_fy", "value_inr": "nsdp_per_capita", "status": "nsdp_status"})[
    ["state", "nsdp_fy_start", "nsdp_fy", "nsdp_per_capita", "nsdp_status"]], on=["state", "nsdp_fy_start"], how="left", suffixes=("_old", ""))
out["nsdp_backfilled"] = backfilled.astype(int).to_numpy()
out = out.drop(columns=["nsdp_per_capita_old", "nsdp_fy_start"])

# ---- charging snapshots (step function)
out = out.sort_values(["state", "date"]).reset_index(drop=True)
vals, labs, apx, bf = [], [], [], []
for s, d in zip(out.state, out.date):
    sn = ch[(ch.state == s) & (ch.label_month <= d)]
    if len(sn):
        r = sn.sort_values("label_month").iloc[-1]; bf.append(0)
    else:
        r = ch[ch.state == s].sort_values("label_month").iloc[0]; bf.append(1)
    vals.append(float(r.value)); labs.append(r.label_month.strftime("%Y-%m")); apx.append(int(r.approximate))
out["charging_stations"], out["charging_snapshot_label"], out["charging_approx"], out["charging_backfilled"] = vals, labs, apx, bf

# ---- mask monthly covariates after the covariate cut-off (placeholders in the original extract)
mask = out.date > COVARIATE_CUTOFF
for c in ["aqi_mean", "pm25_mean", "pm10_mean", "no2_mean", "so2_mean", "aqi_max", "mean_temp_c", "precipitation_mm", "petrol_price", "diesel_price"]:
    out.loc[mask, c] = np.nan
out.loc[mask, "dominant_pollutant"] = np.nan
out.to_csv(D / "ev_adoption_ml_dataset.csv", index=False)
print(f"wrote data/ev_adoption_ml_dataset.csv  {out.shape}; monthly covariates set to NaN for {int(mask.sum())} rows after {COVARIATE_CUTOFF:%b-%Y}")
print(out.groupby("state").apply(lambda g: g.drop_duplicates("nsdp_fy")[["nsdp_fy", "nsdp_status", "nsdp_per_capita"]].assign(first=g.drop_duplicates("nsdp_fy").date.dt.strftime("%Y-%m"))).to_string())
print(out[out.date.isin(pd.to_datetime(["2022-06-01", "2024-02-01", "2026-08-01", "2026-09-01"]))][["state", "date", "charging_stations", "charging_snapshot_label", "charging_backfilled", "nsdp_fy", "nsdp_per_capita"]].to_string())
