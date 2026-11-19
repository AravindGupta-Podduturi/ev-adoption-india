"""
10_fixed_origin_backtest.py -- "train to March 2026, forecast April-September 2026, compare with what actually happened"
=======================================================================================================================
A single, easy-to-explain pseudo-future test that complements the rolling-origin benchmark of stage 04:
  * origin = Mar-2026.  Every estimate (hyper-parameters, SARIMA specification, seasonal factors, model fits, interval
    calibration) uses ONLY data with t <= Mar-2026; covariates come from the release-aligned vintage panel.
  * targets = Apr..Sep-2026 (horizons 1..6) - six months that the models never saw.
  * the same engine (forecast_engine.py) is then run at origin Sep-2026 on ALL data for the real forecast (stage 05).
Headline = the PRE-DECLARED top-3 ensemble (SARIMA, ETS, XGBoost-F0: best exogenous-free models on the 2024 validation window).  The equal-weight
pool and SARIMA-avg4 are EXPLORATORY comparisons that were added after test results were seen.
Policies for the West Bengal drive months (Nov-25..Feb-26 sit INSIDE the training history): as_reported (all models) and
backlog_removed (history repaired, exogenous-free models only).  No flagged month is a target here.
Caveat to state in the report: 24 forecasts per model (4 states x 6 months), ONE origin - conclusions are indicative; the
statistical weight comes from the 456-forecast rolling test in stage 04.
Run from the project root after stages 00a-04:  python 10_fixed_origin_backtest.py
"""
from common import *
from forecast_engine import HEADLINE, POOL, AVG4, CLASSIC_ENS, run_origin, interval_quantiles, apply_policy
from vintage import backlog_adjusted_rate

df = load_data(complete=False)
vp = pd.read_csv(BASE_DIR / "data" / "ev_panel_vintage.csv", parse_dates=["date"]).sort_values(["state", "date"])
protocol = json.loads((TAB_DIR / "02_selected_features_and_protocol.json").read_text())
SEL = protocol["selected_drivers"]
EXOG_F1 = [c for c in SEL if c not in ("charging_stations", "nsdp_per_capita")]
EXOG_F2 = EXOG_F1 + [c for c in ("charging_stations", "nsdp_per_capita") if c in SEL]
N_T = int(df.t.max() + 1)
ORIGIN = pd.Timestamp("2026-03-01")
O = int(df.loc[df.date == ORIGIN, "t"].iloc[0])
DATES = df[df.state == STATES[0]].sort_values("t").date.to_numpy()
Y_RAW = {s: df[df.state == s].sort_values("t")[TARGET].to_numpy() for s in STATES}
EX = {s: vp[vp.state == s].reset_index(drop=True)[EXOG_F2] for s in STATES}
adj, base3w = backlog_adjusted_rate(df)
print(f"origin {ORIGIN:%b-%Y} (t={O}); targets {pd.Timestamp(DATES[O + 1]):%b-%Y}..{pd.Timestamp(DATES[O + 6]):%b-%Y}; data end {pd.Timestamp(DATES[-1]):%b-%Y} (t={N_T - 1})")
assert O + 6 == N_T - 1, "six backtest months must exactly fill the data after the origin"

print("\nCovariates visible at the origin (vintage panel):")
print(vp[(vp.date == ORIGIN)][["state", "aqi_mean", "petrol_price", "charging_stations", "charging_snapshot_label", "nsdp_per_capita", "nsdp_fy_label", "nsdp_status_asof"]].to_string(index=False))

# ------------------------------------------------------------------ run the engine (as_reported: F0/F1/F2; backlog_removed: F0)
runs = []
f = run_origin(O, apply_policy(df, Y_RAW, "as_reported", adj), EX, {"F0": [], "F1": EXOG_F1, "F2": EXOG_F2}, DATES)
runs.append(f.assign(policy="as_reported")); print("SARIMA spec chosen:", f.attrs["sarima_choice"])
f2 = run_origin(O, apply_policy(df, Y_RAW, "backlog_removed", adj), EX, {"F0": []}, DATES)
runs.append(f2.assign(policy="backlog_removed"))
fc = pd.concat(runs, ignore_index=True)
truth = pd.DataFrame([{"state": s, "t": t, "y_true": Y_RAW[s][t]} for s in STATES for t in range(O + 1, N_T)])
fc = fc.merge(truth, on=["state", "t"])
fc["flag"] = fc.apply(lambda r: int(r.state == ANOMALY_STATE and ANOMALY_START <= r.date <= ANOMALY_END), axis=1)
assert fc.flag.sum() == 0
fc["abs_err"] = (fc.y_true - fc.y_pred).abs()

# ------------------------------------------------------------------ 80% intervals for the headline (errors realised by the origin only)
q = interval_quantiles(TAB_DIR / "04_rolling_origin_predictions.csv", O)
print("\nInterval calibration errors available at the origin (n per horizon):", q.n_errors.to_dict())
fc = fc.merge(q[["q10", "q90"]], left_on="h", right_index=True, how="left")
fc["lo80"], fc["hi80"] = fc.y_pred * np.exp(fc.q10), fc.y_pred * np.exp(fc.q90)
save_table(fc.drop(columns=["q10", "q90"]).round(5), "10_backtest_forecasts", index=False)

# ------------------------------------------------------------------ metrics
mase_scale = {s: np.mean(np.abs(np.diff(Y_RAW[s][:O + 1]))) for s in STATES}
last = {s: Y_RAW[s][O] for s in STATES}


def metrics(g: pd.DataFrame, naive_mae: float) -> dict:
    y, p = g.y_true.to_numpy(), g.y_pred.to_numpy()
    within = y - g.groupby("state").y_true.transform("mean").to_numpy()
    lo = g.state.map(last).to_numpy()
    return {"MAE_pp": 100 * np.mean(np.abs(y - p)), "RMSE_pp": 100 * np.sqrt(np.mean((y - p) ** 2)),
            "MAPE_%": 100 * np.mean(np.abs(y - p) / y), "sMAPE_%": 100 * smape(y, p),
            "MASE": float(np.mean(np.abs(y - p) / g.state.map(mase_scale).to_numpy())),
            "R2_pooled": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2),
            "R2_within_state": 1 - np.sum((y - p) ** 2) / np.sum(within ** 2),
            "direction_correct_%": 100 * np.mean(np.sign(p - lo) == np.sign(y - lo)),
            "skill_vs_naive_%": 100 * (1 - np.mean(np.abs(y - p)) / naive_mae), "n": len(g)}


main = fc[fc.policy == "as_reported"]
naive_mae = main[main.model == "Naive (last value)"].abs_err.mean()
tab = pd.DataFrame({m: metrics(g, naive_mae) for m, g in main.groupby("model")}).T.sort_values("MAE_pp")
print("\nBACKTEST (origin Mar-2026, targets Apr-Sep 2026, 24 forecasts per model, as_reported history):\n", tab.round(3).to_string())
save_table(tab.round(4), "10_backtest_metrics")

# West Bengal policy comparison (exogenous-free models)
pol_rows = []
for pol, g in fc.groupby("policy"):
    for m in [HEADLINE, POOL, CLASSIC_ENS, "SARIMA", "ETS", "PSD (pooled seasonal + damped trend)", "SARIMA-avg4 (all specs averaged)", "Ridge-F0", "LightGBM-F0", "XGBoost-F0"]:
        d = g[g.model == m]
        pol_rows.append({"policy": pol, "model": m, "MAE_pp_all_states": 100 * d.abs_err.mean(), "MAE_pp_West_Bengal": 100 * d[d.state == ANOMALY_STATE].abs_err.mean(),
                         "MAE_pp_other_3_states": 100 * d[d.state != ANOMALY_STATE].abs_err.mean()})
pol = pd.DataFrame(pol_rows).pivot(index="model", columns="policy")
print("\nWest Bengal data policy (history fed to the models):\n", pol.round(2).to_string())
save_table(pol.round(4), "10_backtest_wb_policy")

# by state / by month / headline vs actual
H = main[main.model == HEADLINE].copy()
H["inside_80"] = (H.y_true >= H.lo80) & (H.y_true <= H.hi80)
by_state = pd.DataFrame({m: 100 * main[main.model == m].groupby("state").abs_err.mean() for m in [HEADLINE, POOL, "SARIMA", "ETS", "Naive (last value)", "Seasonal naive + YoY growth"]})[[HEADLINE, POOL, "SARIMA", "ETS", "Naive (last value)", "Seasonal naive + YoY growth"]].loc[STATES]
by_h = pd.DataFrame({m: 100 * main[main.model == m].groupby("h").abs_err.mean() for m in [HEADLINE, "SARIMA", "ETS", "Naive (last value)"]})
print("\nMAE (pp) by state:\n", by_state.round(2).to_string()); print("\nMAE (pp) by horizon (h = months after Mar-2026):\n", by_h.round(2).to_string())
save_table(by_state.round(3), "10_backtest_mae_by_state"); save_table(by_h.round(3), "10_backtest_mae_by_horizon")
show = H.assign(actual_pct=100 * H.y_true, forecast_pct=100 * H.y_pred, lo80_pct=100 * H.lo80, hi80_pct=100 * H.hi80, error_pp=100 * (H.y_pred - H.y_true))
show["month"] = show.date.dt.strftime("%b-%Y")
wide_tab = show[["state", "month", "actual_pct", "forecast_pct", "lo80_pct", "hi80_pct", "error_pp", "inside_80"]].round(2)
print(f"\nHeadline ({HEADLINE}) vs actual (EV share, %):\n", wide_tab.to_string(index=False))
save_table(wide_tab, "10_backtest_headline_vs_actual", index=False)
cov = float(H.inside_80.mean()); print(f"\n80% interval coverage on the 24 backtest forecasts: {100 * cov:.0f}% (nominal 80%); mean width {100 * (H.hi80 - H.lo80).mean():.2f} pp")

# do covariates help?  (same model, F0 vs F1 vs F2)
cv_rows = [{"model": n, **{fs: 100 * main[main.model == f"{n}-{fs}"].abs_err.mean() for fs in ("F0", "F1", "F2")}} for n in ("Ridge", "LightGBM", "XGBoost")]
cvt = pd.DataFrame(cv_rows).set_index("model"); cvt["F1 - F0"] = cvt.F1 - cvt.F0; cvt["F2 - F0"] = cvt.F2 - cvt.F0
print("\nBacktest MAE (pp) with / without origin-available covariates:\n", cvt.round(3).to_string())
save_table(cvt.round(3), "10_backtest_covariate_value")

# wins against the naive benchmark by state-month cell
piv = main.pivot_table(index=["state", "t"], columns="model", values="abs_err")
wins = {m: float((piv[m] < piv["Naive (last value)"]).mean() * 100) for m in piv.columns if m != "Naive (last value)"}
wins = pd.Series(wins).sort_values(ascending=False).round(1).to_frame("% of 24 cells beating naive")
save_table(wins, "10_backtest_win_rate_vs_naive"); print("\nShare of the 24 state-month cells where the model beat the naive forecast:\n", wins.head(8).to_string())

# ------------------------------------------------------------------ figures
set_style()
fig, axes = plt.subplots(1, 4, figsize=(15, 3.7), sharex=True)
for ax, s in zip(axes, STATES):
    g = df[(df.state == s) & (df.t >= 36)]
    ax.plot(g.date, 100 * g[TARGET], color=INK, lw=1.6, label="actual")
    h = H[H.state == s].sort_values("h")
    ax.fill_between(h.date, 100 * h.lo80, 100 * h.hi80, color="#2a78d6", alpha=.18, label="80% interval")
    ax.plot(h.date, 100 * h.y_pred, color="#2a78d6", lw=2, marker="o", ms=3.5, label="pre-declared top-3 forecast (origin Mar-26)")
    n_ = main[(main.model == "Naive (last value)") & (main.state == s)].sort_values("h")
    ax.plot(n_.date, 100 * n_.y_pred, color="#898781", ls=":", lw=1.6, label="naive")
    ax.axvline(ORIGIN, color=INK2, lw=0.8, ls="--")
    ax.axvspan(df.date[(df.state == s) & (df.anomaly_flag == 1)].min(), df.date[(df.state == s) & (df.anomaly_flag == 1)].max(), color="#eda100", alpha=.15) if s == ANOMALY_STATE else None
    ax.set_title(s, fontsize=10); ax.tick_params(axis="x", labelrotation=45, labelsize=7)
axes[0].set_ylabel("EV share of new registrations (%)")
h_, l_ = axes[0].get_legend_handles_labels(); fig.legend(h_, l_, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.07), fontsize=8)
fig.suptitle("Fixed-origin backtest: trained to Mar-2026 (dashed line), forecast Apr-Sep 2026 vs what actually happened", x=0.01, y=1.04, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout(); save_fig(fig, "fig30_backtest_actual_vs_forecast")

fig, ax = plt.subplots(figsize=(8.5, 5.2))
order = tab.index[::-1]
cols = ["#0b0b0b" if m == HEADLINE else ("#c3c2b7" if m.startswith(("Naive", "Moving", "Seasonal")) else "#2a78d6") for m in order]
ax.barh([m.split(" (")[0] for m in order], tab.loc[order, "MAE_pp"], color=cols)
for i, m in enumerate(order): ax.text(tab.loc[m, "MAE_pp"] + 0.03, i, f"{tab.loc[m, 'MAE_pp']:.2f}", va="center", fontsize=7.5)
ax.set_xlabel("MAE on Apr-Sep 2026 (pp of EV share)"); ax.tick_params(axis="y", labelsize=7.5)
ax.set_title("Backtest accuracy by model (black = pre-declared headline, grey = naive-type benchmarks)", fontsize=10)
fig.tight_layout(); save_fig(fig, "fig31_backtest_mae_by_model")

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
piv_e = show.pivot(index="state", columns="month", values="error_pp").loc[STATES][[d.strftime("%b-%Y") for d in sorted(show.date.unique())]]
im = axes[0].imshow(piv_e.values, cmap="RdBu_r", vmin=-6, vmax=6, aspect="auto"); axes[0].grid(False)
axes[0].set_xticks(range(6), piv_e.columns, fontsize=8); axes[0].set_yticks(range(4), STATES, fontsize=8)
for i in range(4):
    for j in range(6): axes[0].text(j, i, f"{piv_e.values[i, j]:+.1f}", ha="center", va="center", fontsize=8)
axes[0].set_title("Headline forecast error (forecast - actual, pp)", fontsize=10)
mods = ["Ridge", "LightGBM", "XGBoost"]
for i, fs in enumerate(["F0", "F1", "F2"]):
    axes[1].bar(np.arange(3) + (i - 1) * 0.26, cvt.loc[mods, fs], 0.25, color=["#2a78d6", "#eb6834", "#1baf7a"][i], label={"F0": "F0 history only", "F1": "F1 + monthly drivers", "F2": "F2 + charging & GSDP"}[fs])
axes[1].set_xticks(range(3), mods); axes[1].set_ylabel("MAE (pp)"); axes[1].legend(fontsize=7.5)
axes[1].set_title("Do origin-available covariates help in the backtest?", fontsize=10)
fig.tight_layout(); save_fig(fig, "fig32_backtest_errors_and_covariates")

best = tab.index[0]
save_json({"origin": f"{ORIGIN:%Y-%m}", "targets": f"{pd.Timestamp(DATES[O + 1]):%Y-%m}..{pd.Timestamp(DATES[O + 6]):%Y-%m}", "n_forecasts_per_model": 24,
           "headline": HEADLINE, "headline_MAE_pp": float(tab.loc[HEADLINE, "MAE_pp"]), "headline_R2_pooled": float(tab.loc[HEADLINE, "R2_pooled"]),
           "headline_R2_within_state": float(tab.loc[HEADLINE, "R2_within_state"]), "headline_skill_vs_naive_pct": float(tab.loc[HEADLINE, "skill_vs_naive_%"]),
           "naive_MAE_pp": float(tab.loc["Naive (last value)", "MAE_pp"]), "best_model_in_backtest": best, "best_MAE_pp": float(tab.loc[best, "MAE_pp"]),
           "headline_rank": int(list(tab.index).index(HEADLINE) + 1), "interval_coverage_80": cov, "sarima_choice": f.attrs["sarima_choice"]}, "10_summary")

# ------------------------------------------------------------------ diagnostics: bias, and the same six target months under ALL origins (stage 04)
bias = (100 * (main.assign(e=main.y_pred - main.y_true).pivot_table(index="state", columns="model", values="e"))[[HEADLINE, POOL, "SARIMA", "Naive (last value)"]]).loc[STATES]
print("\nMean signed error (forecast - actual, pp) in the backtest - negative = under-forecast:\n", bias.round(2).to_string())
save_table(bias.round(3), "10_backtest_bias_by_state")
ev04 = pd.read_csv(TAB_DIR / "04_rolling_origin_predictions.csv", parse_dates=["date"])
same = ev04[(ev04.date >= "2026-04-01") & (ev04.date <= "2026-09-01") & (ev04.flag == 0) & ~ev04.model.str.contains(r"\[backlog")]
ctx = pd.DataFrame({"rolling_MAE_same_target_months_pp": 100 * same.groupby("model").abs_err.mean(),
                    "rolling_all_test_MAE_pp": 100 * ev04[(ev04.period == "test") & (ev04.flag == 0) & ~ev04.model.str.contains(r"\[backlog")].groupby("model").abs_err.mean()}).loc[[HEADLINE, POOL, "SARIMA", AVG4, "Naive (last value)"]]
print("\nContext - rolling-origin MAE (all origins, horizons 1-6) for the SAME Apr-Sep 2026 targets vs the whole test window:\n", ctx.round(2).to_string())
save_table(ctx.round(3), "10_backtest_context_vs_rolling")
