"""
11_master_results_table.py -- ONE master table of every reported accuracy number, with its exact sample
=========================================================================================================
Rule: results from different samples must never be mixed into one headline comparison.  Every row below states
the evaluation window (first/last TARGET month), the number of target months and state-month forecasts, the horizons,
whether the flagged West Bengal months are excluded, the series level (state-level vs pooled four-state series), the model
and its status (primary pre-declared / exploratory / benchmark), and the feature set.
Run from the project root after stages 04, 09 and 10:  python 11_master_results_table.py
Outputs: outputs/tables/11_master_results.csv, docs/MASTER_RESULTS_TABLE.md
"""
from common import *

NAIVE = "Naive (last value)"
PRIMARY = "Ensemble-F0 top-3 (validation-selected)"
EXPLORATORY = ["Ensemble-F0 pool (equal weights)", "SARIMA-avg4 (all specs averaged)"]


def status(m):
    if m == PRIMARY: return "PRIMARY (pre-declared)"
    if m in EXPLORATORY: return "exploratory (added after seeing test results)"
    if m.startswith(("Naive", "Moving", "Seasonal")): return "benchmark"
    return "comparison model"


def fset(m):
    for fs in ("F0", "F1", "F2"):
        if m.endswith(f"-{fs}"): return fs
    return "F0 (history only)" if not m.startswith("Ensemble-All") else "F0"


def score(g, naive_mae):
    y, p = g.y_true.to_numpy(), g.y_pred.to_numpy()
    return {"MAE_pp": 100 * np.mean(np.abs(y - p)), "RMSE_pp": 100 * np.sqrt(np.mean((y - p) ** 2)), "sMAPE_%": 100 * smape(y, p),
            "R2_pooled": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2), "skill_vs_naive_%": 100 * (1 - np.mean(np.abs(y - p)) / naive_mae)}


rows = []


def add_eval(eid, label, g, level, wb_excl, horizons, source, model_filter=None):
    g = g.copy()
    nm = g[g.model == NAIVE].eval("abs_err").mean() if (g.model == NAIVE).any() else np.nan
    n_t = g.date.nunique()
    for m, gm in g.groupby("model"):
        if model_filter is not None and m not in model_filter: continue
        rows.append({"eval_id": eid, "evaluation": label, "first_target_month": f"{gm.date.min():%Y-%m}", "last_target_month": f"{gm.date.max():%Y-%m}",
                     "n_target_months": n_t, "n_forecasts_per_model": len(gm), "horizons": horizons, "series_level": level,
                     "WB_flagged_months_excluded": wb_excl, "model": m, "model_status": status(m), "feature_set": fset(m), **score(gm, nm), "source_table": source})


ev = pd.read_csv(TAB_DIR / "04_rolling_origin_predictions.csv", parse_dates=["date"])
ev = ev[~ev.model.str.contains(r"\[backlog")]
test = ev[(ev.period == "test")]
KEEP = [PRIMARY] + EXPLORATORY + ["SARIMA", "ETS", "PSD (pooled seasonal + damped trend)", "Ridge-F0", "LightGBM-F0", "XGBoost-F0", "Ridge-F2", "LightGBM-F2", "XGBoost-F2",
                                  "Moving average (3m)", "Seasonal naive + YoY growth", NAIVE]
clean = test[test.flag == 0]
add_eval("E1", "State-level rolling origin, all test targets", clean, "state-level (4 states)", "yes", "1-6", "04_rolling_origin_predictions.csv", KEEP)
add_eval("E1a", "State-level rolling origin, 2025 targets only", clean[clean.date < "2026-01-01"], "state-level (4 states)", "yes", "1-6", "04_rolling_origin_predictions.csv", KEEP)
add_eval("E1b", "State-level rolling origin, 2026 targets only", clean[clean.date >= "2026-01-01"], "state-level (4 states)", "yes", "1-6", "04_rolling_origin_predictions.csv", KEEP)
add_eval("E1c", "State-level rolling origin, flagged WB months INCLUDED (sensitivity)", test, "state-level (4 states)", "no", "1-6", "04_rolling_origin_predictions.csv", KEEP)

bt = pd.read_csv(TAB_DIR / "10_backtest_forecasts.csv", parse_dates=["date"])
bt = bt[bt.policy == "as_reported"]
add_eval("E2", "Fixed-origin backtest: train to Mar-2026, forecast Apr-Sep 2026 (single origin)", bt, "state-level (4 states)", "yes (none in window)", "1-6", "10_backtest_forecasts.csv",
         [PRIMARY] + EXPLORATORY + ["SARIMA", "ETS", "Ridge-F0", "LightGBM-F0", "XGBoost-F0", "Ridge-F2", "LightGBM-F2", "XGBoost-F2", NAIVE])

ag = pd.read_csv(TAB_DIR / "09_aggregate_metrics_as_reported.csv")
agp = pd.read_csv(TAB_DIR / "09_aggregate_rolling_predictions.csv", parse_dates=["date"]) if (TAB_DIR / "09_aggregate_rolling_predictions.csv").exists() else None
for _, r in ag.iterrows():
    rows.append({"eval_id": "E3", "evaluation": "Pooled four-state series, rolling origin (stage 09)", "first_target_month": "2025-01", "last_target_month": "2026-09",
                 "n_target_months": int(r.n / 6), "n_forecasts_per_model": int(r.n), "horizons": "1-6", "series_level": "pooled four-state series (NOT India)",
                 "WB_flagged_months_excluded": "yes", "model": r.model, "model_status": "exploratory (added after seeing test results)" if "avg4" in r.model or r.model.startswith("Ensemble") else status(r.model),
                 "feature_set": fset(r.model), "MAE_pp": r.MAE_pp, "RMSE_pp": r.RMSE_pp, "sMAPE_%": r["sMAPE_%"], "R2_pooled": r.R2, "skill_vs_naive_%": r["skill_vs_naive_%"], "source_table": "09_aggregate_metrics_as_reported.csv"})

M = pd.DataFrame(rows)
M = M.sort_values(["eval_id", "MAE_pp"]).reset_index(drop=True)
save_table(M.round(4), "11_master_results", index=False)

# markdown version for the docs
md = ["# Master results table (generated by `11_master_results_table.py`)", "",
      "Every number reported in the README, the scorecard and the paper comes from this table. **Rows from different evaluation IDs use different samples and must not be compared with each other.**", "",
      "| Eval | Targets (first..last) | Months | Forecasts / model | Horizons | Level | WB flagged excl. | n windows |", "|---|---|---|---|---|---|---|---|"]
hdr = M.drop_duplicates("eval_id")
for _, r in hdr.iterrows():
    md.append(f"| {r.eval_id}: {r.evaluation} | {r.first_target_month}..{r.last_target_month} | {r.n_target_months} | {r.n_forecasts_per_model} | {r.horizons} | {r.series_level} | {r.WB_flagged_months_excluded} | |".replace(" |  |", " |"))
for eid, g in M.groupby("eval_id", sort=True):
    md += ["", f"## {eid}: {g.evaluation.iloc[0]}", "", "| Model | Status | Feature set | MAE (pp) | RMSE (pp) | sMAPE (%) | R2 (pooled) | Skill vs naive (%) |", "|---|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        f = lambda v, d=2: "" if pd.isna(v) else f"{v:.{d}f}"
        md.append(f"| {r.model} | {r.model_status} | {r.feature_set} | {f(r.MAE_pp)} | {f(r.RMSE_pp)} | {f(r['sMAPE_%'], 1)} | {f(r.R2_pooled)} | {f(r['skill_vs_naive_%'], 1)} |")
(BASE_DIR / "docs" / "MASTER_RESULTS_TABLE.md").write_text("\n".join(md) + "\n")
print(M[M.model.isin([PRIMARY, EXPLORATORY[0], NAIVE, "SARIMA"])][["eval_id", "model", "n_target_months", "n_forecasts_per_model", "MAE_pp", "R2_pooled", "skill_vs_naive_%"]].round(2).to_string(index=False))
print("wrote docs/MASTER_RESULTS_TABLE.md")
