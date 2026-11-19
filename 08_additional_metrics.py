"""
08_additional_metrics.py -- R2, percentage errors and skill scores for the rolling-origin forecasts
==================================================================================================
Reader-friendly views of the SAME forecasts as stage 04 (test window, flagged West Bengal months excluded).
  * pooled R2 mixes between-state differences (easy) with within-state movement (hard) -> also report within-state R2
    (each state's own mean removed) and R2 of CHANGES (forecast minus last observed value vs actual change).
  * skill vs naive = 1 - MAE / MAE(naive).   Percentages are of the actual EV share.
Run from the project root after stage 04:  python 08_additional_metrics.py
"""
from common import *

ev = pd.read_csv(TAB_DIR / "04_rolling_origin_predictions.csv", parse_dates=["date"])
ev = ev[(ev.period == "test") & (ev.flag == 0) & ~ev.model.str.contains(r"\[backlog")]
naive = ev[ev.model == "Naive (last value)"].set_index(["state", "h", "t"]).y_pred.rename("last_obs")
rows = []
mae_naive = ev[ev.model == "Naive (last value)"].eval("abs(y_true-y_pred)").mean()
for m, g in ev.groupby("model"):
    g = g.set_index(["state", "h", "t"]).join(naive)
    y, p = g.y_true.to_numpy(), g.y_pred.to_numpy()
    within_y = y - g.groupby("state").y_true.transform("mean").to_numpy()
    r2_pooled = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    r2_within = 1 - np.sum((y - p) ** 2) / np.sum(within_y ** 2)
    dy, dp = y - g.last_obs.to_numpy(), p - g.last_obs.to_numpy()
    r2_change = 1 - np.sum((dy - dp) ** 2) / np.sum((dy - dy.mean()) ** 2)
    rows.append({"model": m, "R2_pooled": r2_pooled, "R2_within_state": r2_within, "R2_of_changes": r2_change,
                 "MAE_pp": 100 * np.mean(np.abs(y - p)), "MAPE_%": 100 * np.mean(np.abs(y - p) / y),
                 "median_abs_%_error": 100 * np.median(np.abs(y - p) / y), "within_1pp_%": 100 * np.mean(np.abs(y - p) <= 0.01),
                 "within_2pp_%": 100 * np.mean(np.abs(y - p) <= 0.02), "skill_vs_naive_%": 100 * (1 - np.mean(np.abs(y - p)) / mae_naive)})
out = pd.DataFrame(rows).set_index("model").sort_values("MAE_pp").round(3)
print(out.to_string())
save_table(out, "08_additional_metrics")
h = ev[ev.model == "SARIMA-avg4 (all specs averaged)"]
byh = h.groupby("h").apply(lambda d: pd.Series({"MAE_pp": 100 * d.abs_err.mean(), "MAPE_%": 100 * (d.abs_err / d.y_true).mean()})).round(2)
print("\nBest single model by horizon:\n", byh.to_string()); save_table(byh, "08_best_model_by_horizon")
