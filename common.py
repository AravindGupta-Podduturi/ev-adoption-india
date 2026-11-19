"""
common.py -- shared configuration, data loading, metrics and plotting helpers
==============================================================================
Project : Explainable ML for Regional EV Adoption in India:
          Identifying Key Drivers and Forecasting Future Demand
Course  : Advanced Machine Learning for Business Transformation (AMLBT)

Every stage script (01_ ... 05_) imports from this module so that paths,
random seeds, feature groups and evaluation metrics are defined exactly once.
"""

import json
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths (works both as a script and inside a notebook)
# --------------------------------------------------------------------------- #
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:  # notebook
    BASE_DIR = Path.cwd()

DATA_PATH = BASE_DIR / "data" / "ev_adoption_ml_dataset.csv"
FIG_DIR = BASE_DIR / "outputs" / "figures"
TAB_DIR = BASE_DIR / "outputs" / "tables"
for _d in (FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
rng_global = np.random.default_rng(SEED)

# --------------------------------------------------------------------------- #
# Problem definition
# --------------------------------------------------------------------------- #
TARGET = "ev_adoption_rate"  # EV share of new registrations in a state-month
# Two cut-offs.  The supplied file holds Sep-Oct 2026, but
#   * Oct-2026 is a partial month (~10 days of registrations) -> dropped, and
#   * Sep/Oct-2026 air-quality / weather / fuel columns were identical in all four states (placeholders) -> set to NaN (stage 00a).
# Registrations are therefore used through Sep-2026 (DATA_CUTOFF; forecasting stages), whereas the same-month
# explanatory stages (01-03, 09 part C) need complete covariates and stop at COVARIATE_CUTOFF (Aug-2026).
DATA_CUTOFF = pd.Timestamp("2026-09-01")          # last month with verified REGISTRATIONS (Oct-2026 is a partial month) - TO CONFIRM with team
COVARIATE_CUTOFF = pd.Timestamp("2026-08-01")      # last month with verified air-quality / weather / fuel-price values (Sep/Oct-2026 were placeholders -> NaN)
TRAIN_END = pd.Timestamp("2024-12-01")  # hyper-parameters are tuned on 2022-24 only
TEST_START = pd.Timestamp("2025-01-01")  # out-of-sample evaluation window = Jan-2025 .. DATA_CUTOFF
TEST_END = DATA_CUTOFF

# West Bengal "registration-drive episode" (Nov-2025 .. Feb-2026).
# EV share: 3.8% (Oct) -> 10.4% -> 23.6% (Dec) -> 17.5% -> 13.4% (Feb) -> ~10% (Mar).
# In-data evidence (stage 07): West Bengal's THREE-WHEELER registrations jump from ~4,000 to 13,000 / 29,000 / 19,000 /
# 13,000 in exactly these months and the excess EVs almost equal the excess three-wheelers - i.e. e-rickshaws.
# Trade-press reports (eMobilityPlus, Feb-2026) link this to the state's mandatory e-rickshaw registration drive
# (announced Oct-2025, deadline 31-Dec-2025 extended to 31-Jan-2026), i.e. a BACKLOG of previously unregistered
# vehicles, not new sales.  The official notification should still be cited in the paper (see docs/DATA_SOURCES.md).
# Flagged months are excluded from the "clean" metrics, and stage 05 forecasts under three data policies.
ANOMALY_STATE = "West Bengal"
ANOMALY_START = pd.Timestamp("2025-11-01")
ANOMALY_END = pd.Timestamp("2026-02-01")
# Columns that must NOT be used as predictors (leakage audit, stage 02)
LEAKY_COLS = {
    "ev_count": "numerator of the target",
    "electric": "identical to ev_count (fuel-type split of the same registrations)",
    "vehicle_total": "denominator of the target; same-month value is unknown at forecast time",
    "two_wheeler": "same-month registrations that contain the EVs",
    "three_wheeler": "same-month registrations that contain the EVs",
    "four_wheeler": "same-month registrations that contain the EVs",
    "light_vehicle": "same-month registrations that contain the EVs",
    "medium_vehicle": "same-month registrations that contain the EVs",
    "heavy_vehicle": "same-month registrations that contain the EVs",
    "other": "same-month registrations that contain the EVs",
    "petrol": "fuel split of same-month registrations (shares sum to the target's denominator)",
    "diesel": "fuel split of same-month registrations",
    "gas": "fuel split of same-month registrations",
    "hybrid": "fuel split of same-month registrations",
    "lpg": "fuel split of same-month registrations",
    "alternative": "fuel split of same-month registrations",
    "not_applicable": "fuel split of same-month registrations",
}

# Candidate drivers, grouped for business interpretation
DRIVER_GROUPS = {
    "Air quality": ["aqi_mean", "pm25_mean", "pm10_mean", "no2_mean", "so2_mean", "aqi_max"],
    "Fuel prices": ["petrol_price", "diesel_price"],
    "Charging infrastructure": ["charging_stations"],
    "Income": ["nsdp_per_capita"],
    "Weather": ["mean_temp_c", "precipitation_mm"],
}
CANDIDATE_DRIVERS = [c for cols in DRIVER_GROUPS.values() for c in cols]
COL_TO_GROUP = {c: g for g, cols in DRIVER_GROUPS.items() for c in cols}
COL_TO_GROUP.update({"month_sin": "Seasonality", "month_cos": "Seasonality", "fy_end": "Seasonality"})

# --------------------------------------------------------------------------- #
# Plot style  (validated categorical palette: blue, orange, aqua, yellow)
# --------------------------------------------------------------------------- #
STATES = ["Delhi", "Maharashtra", "Tamil Nadu", "West Bengal"]
STATE_COLORS = dict(zip(STATES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"


def set_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5, "legend.frameon": False, "lines.linewidth": 1.8,
        "figure.dpi": 110, "savefig.dpi": 200, "font.family": "DejaVu Sans",
    })


set_style()


def save_fig(fig, name: str) -> Path:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    try:  # show the figure inline when running inside a Jupyter notebook
        from IPython import get_ipython
        if get_ipython() is not None and "IPKernelApp" in get_ipython().config:
            from IPython.display import Image, display
            display(Image(filename=str(path), width=900))
    except Exception:
        pass
    print(f"  [figure] {path.relative_to(BASE_DIR)}")
    return path


def save_table(df: pd.DataFrame, name: str, index: bool = True) -> Path:
    path = TAB_DIR / f"{name}.csv"
    df.to_csv(path, index=index)
    print(f"  [table]  {path.relative_to(BASE_DIR)}")
    return path


def save_json(obj, name: str) -> Path:
    path = TAB_DIR / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=str))
    print(f"  [json]   {path.relative_to(BASE_DIR)}")
    return path


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_data(apply_cutoff: bool = True, complete: bool = True) -> pd.DataFrame:
    """Load the state-month panel, sort it and add calendar / flag columns.

    complete=True  -> rows through COVARIATE_CUTOFF (every covariate observed): explanatory / retrospective stages.
    complete=False -> rows through DATA_CUTOFF (registrations only; covariates of the last month are NaN): forecasting stages.
    apply_cutoff=False returns the whole file (used only by the stage-01 audit of the unusable trailing months).
    """
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    if apply_cutoff:
        df = df[df["date"] <= (COVARIATE_CUTOFF if complete else DATA_CUTOFF)]
    df = df.sort_values(["state", "date"]).reset_index(drop=True)
    df["t"] = df.groupby("state").cumcount()  # 0 .. n_months-1
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["fy_end"] = (df["month"] == 3).astype(int)  # Indian fiscal year-end (March)
    df["anomaly_flag"] = ((df["state"] == ANOMALY_STATE) & df["date"].between(ANOMALY_START, ANOMALY_END)).astype(int)
    return df


def wide(df: pd.DataFrame, col: str = TARGET) -> pd.DataFrame:
    """Date x state matrix of one column."""
    return df.pivot(index="date", columns="state", values=col)[STATES]


# --------------------------------------------------------------------------- #
# Metrics  (all on the adoption-rate scale; MAE/RMSE reported in percentage points)
# --------------------------------------------------------------------------- #
def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def smape(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p))))


def wape(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y)))


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def score_table(df: pd.DataFrame, y="y_true", p="y_pred") -> dict:
    return {"MAE_pp": 100 * mae(df[y], df[p]), "RMSE_pp": 100 * rmse(df[y], df[p]),
            "sMAPE_%": 100 * smape(df[y], df[p]), "WAPE_%": 100 * wape(df[y], df[p]), "n": len(df)}
