"""
06_frozen_forecast_scorecard.py -- score a FROZEN forecast file against actuals that arrived afterwards
=======================================================================================================
Stage 05 freezes its forecast once (data/frozen_forecasts_made_<date>.csv, never overwritten).  Re-run this script
whenever new months are added to the raw file and DATA_CUTOFF in common.py is moved forward: it scores every frozen
month that now has an actual.  The earlier 2026-10-09 freeze (origin Dec-2025, targets Jan-Jun 2026, produced by the earlier pipeline) is recorded in
docs/frozen_forecast_2026-10-09_scorecard.md.
Run from the project root:  python 06_frozen_forecast_scorecard.py [frozen_file.csv]
"""
import sys
from common import *

files = sorted((BASE_DIR / "data").glob("frozen_forecasts_made_*.csv"))
if len(sys.argv) > 1 and sys.argv[1].endswith(".csv"):
    files = [BASE_DIR / sys.argv[1]]
raw = load_data(complete=False)
actual = raw[["state", "date", TARGET]].assign(actual=lambda d: 100 * d[TARGET]).drop(columns=TARGET)
summary = {}
for path in files:
    frozen = pd.read_csv(path, parse_dates=["date"])
    f = frozen.merge(actual, on=["state", "date"], how="inner")
    print(f"\nFrozen file: {path.name} | forecast months {frozen.date.min():%b-%Y}..{frozen.date.max():%b-%Y} | verified actuals available through {DATA_CUTOFF:%b-%Y}")
    if f.empty:
        print("  no frozen month has a verified actual yet"); summary[path.name] = {"scored_months": 0}; continue
    cols = [c for c in ["Headline", "Pool (equal weights)", "SARIMA", "ETS", "XGBoost-F0"] if c in f.columns]
    rows = {}
    for pol, g in f.groupby("policy"):
        rows[pol] = {c: (g[c] - g.actual).abs().mean() for c in cols}
        rows[pol]["80% interval coverage %"] = 100 * ((g.actual >= g.lo80) & (g.actual <= g.hi80)).mean()
        rows[pol]["n forecasts"] = len(g)
    score = pd.DataFrame(rows); print(score.round(2).to_string())
    save_table(score.round(3), f"06_frozen_forecast_mae_{path.stem[-10:]}"); summary[path.name] = {"scored_months": int(f.date.nunique())}
save_json(summary, "06_summary")
