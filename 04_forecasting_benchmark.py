"""
04_forecasting_benchmark.py -- PROSPECTIVE layer: origin-honest forecasting benchmark with F0 / F1 / F2 feature sets
=====================================================================================================================
Covariates are taken from the VINTAGE panel (data/ev_panel_vintage.csv),
i.e. only what had been published at each forecast origin, with no back-filling, and every comparison is rerun.

Feature sets (direct multi-horizon ML, one global model per horizon, target = log(y_t / base)):
  F0  history only (levels, momentum, own + pooled seasonality, calendar)          -> no exogenous driver at all
  F1  F0 + 1-month-lagged AQI / NO2 / SO2 / petrol / weather
  F2  F1 + release-aligned charging stations and NSDP per capita (NaN until published)
Classical models: naive family, ETS, per-state SARIMA, and NEW  PSD = pooled-seasonal + damped trend.
The pre-declared headline is the top-3 ensemble selected on the 2022-24 validation window ONLY; the equal-weight pool and SARIMA-avg4 are exploratory comparisons.

Protocol: tune on 2022-24 -> freeze -> rolling-origin evaluation, targets Jan-2025 .. data cut-off (test) and
Jan-2024 .. Dec-2024 (validation, used only for model/ensemble selection and interval calibration).
Run from the project root after stages 00 and 02:  python 04_forecasting_benchmark.py
"""
from common import *
from forecasting_utils import *
from vintage import backlog_adjusted_rate

import lightgbm as lgb
import xgboost as xgb
import joblib
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = load_data(complete=False)
vp = pd.read_csv(BASE_DIR / "data" / "ev_panel_vintage.csv", parse_dates=["date"])
vp = vp[vp.date <= DATA_CUTOFF].sort_values(["state", "date"]).reset_index(drop=True)
protocol = json.loads((TAB_DIR / "02_selected_features_and_protocol.json").read_text())
SEL = protocol["selected_drivers"]
EXOG_F1 = [c for c in SEL if c not in ("charging_stations", "nsdp_per_capita")]
EXOG_F2 = EXOG_F1 + [c for c in ("charging_stations", "nsdp_per_capita") if c in SEL]
FSETS = {"F0": [], "F1": EXOG_F1, "F2": EXOG_F2}
print("F1 drivers:", EXOG_F1, "| F2 adds:", [c for c in EXOG_F2 if c not in EXOG_F1])

HORIZONS = [1, 2, 3, 4, 5, 6]
N_T = int(df.t.max() + 1)
T_TRAIN_END = int(df.loc[df.date == TRAIN_END, "t"].iloc[0])           # 35 = Dec-2024
EVAL_T = list(range(24, N_T))                                           # validation (24..35) + test (36..)
Y = {s: df[df.state == s].sort_values("t")[TARGET].to_numpy() for s in STATES}
DATES = df[df.state == STATES[0]].sort_values("t").date.to_numpy()
EX = {s: vp[vp.state == s].reset_index(drop=True)[EXOG_F2] for s in STATES}           # index = t (origin month)
FLAG = {s: df[df.state == s].sort_values("t").anomaly_flag.to_numpy() for s in STATES}
assert all(len(EX[s]) == N_T for s in STATES)

# backlog-adjusted West Bengal history (data-driven, stage 07) for the sensitivity run
adj, wb_base3w = backlog_adjusted_rate(df)
Y_ADJ = {s: v.copy() for s, v in Y.items()}
for t_, r_ in adj.items():
    Y_ADJ[ANOMALY_STATE][t_] = r_
print(f"WB baseline three-wheelers {wb_base3w:.0f}/month; adjusted rates:", {k: round(v, 3) for k, v in adj.items()})

# --------------------------------------------------------------------------------------------- rows + tuning
ROWS = {fs: {h: make_rows(Y, EX, ex, h, range(14, N_T)) for h in HORIZONS} for fs, ex in FSETS.items()}
FEATS = {fs: ml_feature_names(ex) for fs, ex in FSETS.items()}
print("features:", {k: len(v) for k, v in FEATS.items()})

from forecast_engine import SPACES      # same search spaces as the fixed-origin backtest / final refit
BEST, tune_log = {}, []
for fs in FSETS:
    for h in HORIZONS:
        tr = ROWS[fs][h].query("t <= @T_TRAIN_END").reset_index(drop=True)
        folds = [(tr.index[tr.t < a].to_numpy(), tr.index[(tr.t >= a) & (tr.t <= b)].to_numpy()) for a, b in [(24, 29), (30, 35)]]
        for name, (est, space, n_iter) in SPACES.items():
            rs = RandomizedSearchCV(est, space, n_iter=n_iter, cv=folds, scoring="neg_mean_absolute_error",
                                    random_state=SEED, n_jobs=-1, error_score=np.nan).fit(tr[FEATS[fs]], tr["target"])
            BEST[(name, fs, h)] = rs.best_estimator_
            tune_log.append({"model": name, "feature_set": fs, "h": h, "cv_MAE_log_ratio": -rs.best_score_,
                             "params": {k.split("__")[-1]: (round(float(v), 4) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (int, np.integer)) else v)) for k, v in rs.best_params_.items()}})
    print(f"  tuned {fs}", flush=True)
tune_tab = pd.DataFrame(tune_log)
save_table(tune_tab.assign(params=tune_tab.params.astype(str)), "04_ml_tuning_log", index=False)

# --------------------------------------------------------------------------------------------- SARIMA spec (train window only)
val_origins = range(23, 30)
SARIMA_CHOICE, sel_rows = {}, []
for s in STATES:
    scores = {}
    for spec in SARIMA_SPECS:
        errs = []
        for o in val_origins:
            f = fit_forecast_sarima(Y[s][:o + 1], 6, spec)
            tgt = np.arange(o + 1, o + 7); ok = tgt <= T_TRAIN_END
            errs += list(np.abs(f[ok] - Y[s][tgt[ok]]))
        scores[spec] = np.mean(errs) * 100
        sel_rows.append({"state": s, "spec": spec, "validation_MAE_pp": np.mean(errs) * 100})
    SARIMA_CHOICE[s] = min(scores, key=scores.get)
print("SARIMA spec per state:", SARIMA_CHOICE)
save_table(pd.DataFrame(sel_rows), "04_sarima_spec_selection", index=False)

# --------------------------------------------------------------------------------------------- rolling-origin evaluation
records = []


def add(model, s, o, h, pred):
    t = o + h
    records.append({"model": model, "state": s, "o": o, "t": t, "h": h, "date": DATES[t], "y_true": Y[s][t],
                    "y_pred": float(pred), "flag": int(FLAG[s][t])})


origins_by_h = {h: [t - h for t in EVAL_T] for h in HORIZONS}
all_origins = sorted({o for v in origins_by_h.values() for o in v})
for o in all_origins:
    sf = pooled_seasonal(Y, o + 1)
    sf_adj = pooled_seasonal(Y_ADJ, o + 1)
    for s in STATES:
        y = Y[s]; hist = y[:o + 1]
        ets, sar = fit_forecast_ets(hist, 6), fit_forecast_sarima(hist, 6, SARIMA_CHOICE[s])
        psd = fit_forecast_psd(hist, 6, sf)
        sar_avg = np.exp(np.mean([np.log(fit_forecast_sarima(hist, 6, sp)) for sp in SARIMA_SPECS], axis=0))   # spec-free average
        b = base_level(y, o); b_ly = np.mean(y[o - 14:o - 11])
        if s == ANOMALY_STATE:                     # sensitivity: same models fed the backlog-adjusted history
            ha = Y_ADJ[s][:o + 1]
            ets_a, sar_a, psd_a = fit_forecast_ets(ha, 6), fit_forecast_sarima(ha, 6, SARIMA_CHOICE[s]), fit_forecast_psd(ha, 6, sf_adj)
        for h in HORIZONS:
            t = o + h
            if t not in EVAL_T:
                continue
            add("Naive (last value)", s, o, h, y[o]); add("Moving average (3m)", s, o, h, b)
            add("Seasonal naive", s, o, h, y[t - 12]); add("Seasonal naive + YoY growth", s, o, h, y[t - 12] * b / b_ly)
            add("ETS", s, o, h, ets[h - 1]); add("SARIMA", s, o, h, sar[h - 1]); add("SARIMA-avg4 (all specs averaged)", s, o, h, sar_avg[h - 1]); add("PSD (pooled seasonal + damped trend)", s, o, h, psd[h - 1])
            if s == ANOMALY_STATE:
                add("ETS [backlog-adjusted history]", s, o, h, ets_a[h - 1]); add("SARIMA [backlog-adjusted history]", s, o, h, sar_a[h - 1])
                add("PSD [backlog-adjusted history]", s, o, h, psd_a[h - 1])
    if o % 10 == 0:
        print("  classical origin", o, flush=True)

for fs in FSETS:
    for h in HORIZONS:
        rows_h = ROWS[fs][h]
        for o in origins_by_h[h]:
            tr, te = rows_h[rows_h.t <= o], rows_h[rows_h.o == o]
            if len(tr) < 24:                      # too little realised history for a direct ML refit at this origin
                continue
            for name in SPACES:
                m = clone(BEST[(name, fs, h)]).fit(tr[FEATS[fs]], tr["target"])
                pred = te["base"].to_numpy() * np.exp(m.predict(te[FEATS[fs]]))
                for s, p in zip(te.state, pred):
                    add(f"{name}-{fs}", s, o, h, p)
    print(f"  ML rolling refits done for {fs}", flush=True)
ev = pd.DataFrame(records)
ev["period"] = np.where(ev.t <= T_TRAIN_END, "validation", "test")

# ---------------- ensembles chosen from VALIDATION-window accuracy only
val = ev[(ev.period == "validation") & (ev.flag == 0)].assign(ae=lambda d: (d.y_true - d.y_pred).abs())
_piv = val[~val.model.str.contains(r"\[backlog")].pivot_table(index=["state", "h", "t"], columns="model", values="ae").dropna()   # common support
print(f"validation comparison uses {len(_piv)} (state, horizon, month) cells where every model has a forecast")
val_mae = (100 * _piv.mean()).sort_values()
print("\nValidation-window MAE (pp), 2024 targets:\n", val_mae.round(3).to_string())
save_table(val_mae.round(4).to_frame("validation_MAE_pp"), "04_validation_mae")
CLASSIC = ["SARIMA", "ETS", "PSD (pooled seasonal + damped trend)"]
F0_POOL = CLASSIC + ["SARIMA-avg4 (all specs averaged)", "Ridge-F0", "LightGBM-F0", "XGBoost-F0"]
ALL_POOL = CLASSIC + ["SARIMA-avg4 (all specs averaged)"] + [f"{n}-{fs}" for n in SPACES for fs in FSETS]
ENS_DEF = {"Ensemble-Classical (SARIMA+ETS+PSD)": CLASSIC,
           "Ensemble-F0 top-3 (validation-selected)": list(val_mae[val_mae.index.isin(F0_POOL)].index[:3]),
           "Ensemble-All top-3 (validation-selected)": list(val_mae[val_mae.index.isin(ALL_POOL)].index[:3]),
           "Ensemble-F0 pool (equal weights)": F0_POOL}
print("\nEnsemble membership:"); [print(" ", k, "->", v) for k, v in ENS_DEF.items()]
parts = []
for name, members in ENS_DEF.items():
    e_ = ev[ev.model.isin(members)].groupby(["state", "o", "t", "h", "date", "y_true", "flag", "period"], as_index=False)["y_pred"].mean()
    e_["model"] = name; parts.append(e_)
ev = pd.concat([ev] + parts, ignore_index=True)
ev["abs_err"] = (ev.y_true - ev.y_pred).abs()
save_table(ev, "04_rolling_origin_predictions", index=False)
save_json({"ensemble_members": ENS_DEF}, "04_ensemble_membership")

# --------------------------------------------------------------------------------------------- results (TEST window)
test = ev[ev.period == "test"].copy()
mase_scale = {s: np.mean(np.abs(np.diff(Y[s][:T_TRAIN_END + 1]))) for s in STATES}
test["mase_term"] = test.abs_err / test.state.map(mase_scale)
test["subperiod"] = np.where(test.date.dt.year == 2025, "2025", f"{DATA_CUTOFF.year} YTD")


def summarise(frame, by):
    g = frame.groupby(by)
    return pd.DataFrame({"MAE_pp": 100 * g.abs_err.mean(),
                         "RMSE_pp": 100 * np.sqrt(g.apply(lambda d: np.mean((d.y_true - d.y_pred) ** 2))),
                         "sMAPE_%": 100 * g.apply(lambda d: smape(d.y_true, d.y_pred)),
                         "MASE": g.mase_term.mean(), "n": g.size()})


clean = test[test.flag == 0]
main_models = [m for m in clean.model.unique() if "[backlog" not in m]
by_model_clean = summarise(clean[clean.model.isin(main_models)], "model").sort_values("MAE_pp")
by_model_all = summarise(test[test.model.isin(main_models)], "model").sort_values("MAE_pp")
print(f"\nTEST window, targets {TEST_START:%b-%Y}..{DATA_CUTOFF:%b-%Y}, horizons 1-6, flagged WB months excluded:")
print(by_model_clean.round(3).to_string())
save_table(by_model_clean.round(4), "04_metrics_overall_clean"); save_table(by_model_all.round(4), "04_metrics_overall_all")
by_h = summarise(clean[clean.model.isin(main_models)], ["model", "h"])["MAE_pp"].unstack("h").loc[by_model_clean.index]
by_state = summarise(clean[clean.model.isin(main_models)], ["model", "state"])["MAE_pp"].unstack("state").loc[by_model_clean.index][STATES]
by_period = summarise(clean[clean.model.isin(main_models)], ["model", "subperiod"])["MAE_pp"].unstack("subperiod").loc[by_model_clean.index]
print("\nMAE by horizon:\n", by_h.round(2).to_string()); print("\nMAE by sub-period:\n", by_period.round(2).to_string())
save_table(by_h.round(4), "04_mae_by_horizon_clean"); save_table(by_state.round(4), "04_mae_by_state_clean"); save_table(by_period.round(4), "04_mae_by_period_clean")

# sensitivity: backlog-adjusted history (West Bengal only, flagged months excluded)
sens = []
for m in CLASSIC:
    a = clean[(clean.model == m) & (clean.state == ANOMALY_STATE)].abs_err.mean() * 100
    b = clean[(clean.model == m + " [backlog-adjusted history]") & (clean.state == ANOMALY_STATE)].abs_err.mean() * 100 if m != "PSD (pooled seasonal + damped trend)" else \
        clean[(clean.model == "PSD [backlog-adjusted history]") & (clean.state == ANOMALY_STATE)].abs_err.mean() * 100
    sens.append({"model": m, "WB_MAE_as_reported_history_pp": a, "WB_MAE_backlog_adjusted_history_pp": b})
sens = pd.DataFrame(sens).round(3); print("\nWest Bengal sensitivity to the registration-drive episode:\n", sens.to_string(index=False))
save_table(sens, "04_wb_history_sensitivity", index=False)


def boot_diff(model_a, model_b, frame, n=4000, seed=SEED):
    """Paired block bootstrap (blocks = target months) of mean |error| difference A - B, pp."""
    a = frame[frame.model == model_a].set_index(["state", "h", "t"]).abs_err
    b = frame[frame.model == model_b].set_index(["state", "h", "t"]).abs_err
    d = (a - b).dropna().rename("d").reset_index()
    sums = d.groupby("t").d.sum().to_numpy(); counts = d.groupby("t").d.size().to_numpy()
    k = len(sums); rng = np.random.default_rng(seed); picks = rng.integers(0, k, size=(n, k))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    lo, hi = np.percentile(means, [5, 95])
    return 100 * d.d.mean(), 100 * lo, 100 * hi


bench = "Naive (last value)"
rows = []
for m in by_model_clean.index:
    if m == bench: continue
    dm, lo, hi = boot_diff(m, bench, clean)
    rows.append({"model": m, "MAE diff vs naive (pp)": dm, "90% CI low": lo, "90% CI high": hi, "significantly_better": hi < 0})
sig = pd.DataFrame(rows).set_index("model").round(3); print("\nVs naive (negative = better):\n", sig.to_string())
save_table(sig, "04_bootstrap_vs_benchmark")

# THE KEY TABLE: do covariates help once timing is honest?  (same model, F1 or F2 minus F0)
rows = []
for n in SPACES:
    for fs in ("F1", "F2"):
        dm, lo, hi = boot_diff(f"{n}-{fs}", f"{n}-F0", clean)
        rows.append({"model": n, "comparison": f"{fs} minus F0", "MAE diff (pp)": dm, "90% CI low": lo, "90% CI high": hi,
                     "covariates_help": hi < 0, "covariates_hurt": lo > 0})
for fs in ("F1", "F2"):
    pass
cov = pd.DataFrame(rows).round(3)
print("\nDo origin-available covariates improve the forecast? (negative = yes)\n", cov.to_string(index=False))
save_table(cov, "04_covariate_value_F0_F1_F2", index=False)

head_to_head = []
champ = by_model_clean.index[0]    # best model on the test window - descriptive only, NOT used for any choice
for a_, b_ in [(champ, "SARIMA"), ("Ensemble-F0 top-3 (validation-selected)", "SARIMA"), ("Ensemble-F0 pool (equal weights)", "SARIMA"), ("SARIMA-avg4 (all specs averaged)", "SARIMA"), ("PSD (pooled seasonal + damped trend)", "SARIMA"),
               ("LightGBM-F0", "SARIMA"), ("XGBoost-F0", "SARIMA"), ("Ridge-F0", "SARIMA"),
               ("Ensemble-All top-3 (validation-selected)", "Ensemble-F0 top-3 (validation-selected)")]:
    if a_ == b_: continue
    dm, lo, hi = boot_diff(a_, b_, clean)
    head_to_head.append({"A": a_, "B": b_, "MAE(A)-MAE(B) pp": dm, "90% CI low": lo, "90% CI high": hi})
h2h = pd.DataFrame(head_to_head).round(3); print("\nHead-to-head:\n", h2h.to_string(index=False))
save_table(h2h, "04_bootstrap_head_to_head", index=False)

# --------------------------------------------------------------------------------------------- headline model and intervals
HEADLINE = "Ensemble-F0 top-3 (validation-selected)"   # PRE-DECLARED confirmatory headline: the 3 best exogenous-free models on the 2024 validation window. The equal-weight pool and SARIMA-avg4 are EXPLORATORY (added after test results were seen)
print("\nPre-declared headline (validation-selected top-3):", HEADLINE, "->", ENS_DEF[HEADLINE])
e = ev[(ev.model == HEADLINE) & (ev.flag == 0)].copy(); e["log_err"] = np.log(e.y_true / e.y_pred)
cov_rows = []
for _, r in e[e.period == "test"].iterrows():             # expanding-window intervals: only errors realised by the origin
    past = e[(e.h == r.h) & (e.t <= r.o)]
    lo_q, hi_q = past.log_err.quantile([0.10, 0.90])
    cov_rows.append({"covered": r.y_pred * np.exp(lo_q) <= r.y_true <= r.y_pred * np.exp(hi_q), "width_pp": 100 * r.y_pred * (np.exp(hi_q) - np.exp(lo_q)), "h": r.h})
cov_df = pd.DataFrame(cov_rows)
print(f"Out-of-sample coverage of the 80% intervals (expanding-window calibration): {100 * cov_df.covered.mean():.1f}% | mean width {cov_df.width_pp.mean():.2f} pp")
save_table(cov_df.groupby("h").agg(coverage=("covered", "mean"), mean_width_pp=("width_pp", "mean")).round(3), "04_interval_coverage_by_h")
q = e.groupby("h").log_err.quantile([0.10, 0.90]).unstack(); q.columns = ["q10", "q90"]
save_table(q.round(4), "05_interval_quantiles")

# --------------------------------------------------------------------------------------------- figures
set_style()
STY = {"Naive (last value)": ("#c3c2b7", ":"), "Seasonal naive + YoY growth": ("#898781", "--"), "SARIMA": ("#eb6834", "-"), "SARIMA-avg4 (all specs averaged)": ("#a0471e", "-"), "ETS": ("#eda100", "-"),
       "PSD (pooled seasonal + damped trend)": ("#1baf7a", "-"), "Ridge-F0": ("#4a3aa7", "-"), "LightGBM-F0": ("#2a78d6", "-"),
       HEADLINE: ("#0b0b0b", "-")}
fig, ax = plt.subplots(figsize=(8.5, 4.4))
for m, (c, ls) in STY.items():
    ax.plot(by_h.columns, by_h.loc[m], color=c, ls=ls, marker="o", ms=4, lw=2.4 if m == HEADLINE else 1.5, label=m.split(" (")[0] if m != HEADLINE else "Headline (pre-declared top-3)")
ax.set_xlabel("forecast horizon (months ahead)"); ax.set_ylabel("MAE (pp of EV share)")
ax.set_title("Out-of-sample error by horizon (test targets, origin-available information only)", fontsize=10); ax.legend(fontsize=7.5, ncol=2)
fig.tight_layout(); save_fig(fig, "fig16_mae_by_horizon")

fig, ax = plt.subplots(figsize=(8.5, 4))
mods = ["Ridge", "LightGBM", "XGBoost"]
for i, fs in enumerate(["F0", "F1", "F2"]):
    vals = [by_model_clean.loc[f"{n}-{fs}", "MAE_pp"] for n in mods]
    ax.bar(np.arange(3) + (i - 1) * 0.26, vals, 0.25, color=["#2a78d6", "#eb6834", "#1baf7a"][i], label={"F0": "F0 history only", "F1": "F1 + lagged monthly drivers", "F2": "F2 + release-aligned charging & income"}[fs])
    for x, v in zip(np.arange(3) + (i - 1) * 0.26, vals): ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=7.5)
ax.axhline(by_model_clean.loc["SARIMA", "MAE_pp"], color=INK2, ls="--", lw=1); ax.text(2.45, by_model_clean.loc["SARIMA", "MAE_pp"] + 0.03, "SARIMA", fontsize=7.5, ha="right")
ax.set_xticks(range(3), mods); ax.set_ylabel("MAE (pp)"); ax.set_ylim(0, 2.9); ax.legend(fontsize=8, loc="upper center", ncol=3)
ax.set_title("Do covariates help once their release dates are respected?", fontsize=10)
fig.tight_layout(); save_fig(fig, "fig19_covariate_value_F0_F1_F2")

fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
for ax, s in zip(axes, STATES):
    full = df[df.state == s]
    ax.plot(full.date[full.t >= 24], 100 * full[TARGET][full.t >= 24], color=INK, lw=1.5, label="actual")
    for m in [HEADLINE, "Naive (last value)", "SARIMA"]:
        d = ev[(ev.model == m) & (ev.state == s) & (ev.h == 3)].sort_values("t")
        c, ls = STY[m]; ax.plot(d.date, 100 * d.y_pred, color=c, ls=ls, lw=1.8, marker="o", ms=3, label=m.split(" (")[0] if m != HEADLINE else "headline ensemble")
    ax.axvspan(full.date[full.anomaly_flag == 1].min(), full.date[full.anomaly_flag == 1].max(), color="#eda100", alpha=.15) if s == ANOMALY_STATE else None
    ax.set_title(s, fontsize=10); ax.tick_params(axis="x", labelrotation=45, labelsize=7)
axes[0].set_ylabel("EV share (%)")
h_, l_ = axes[0].get_legend_handles_labels(); fig.legend(h_, l_, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.08), fontsize=8)
fig.suptitle("3-month-ahead forecasts vs actuals (validation 2024 + test 2025-26); shaded = West Bengal registration drive", x=0.01, y=1.04, ha="left", fontweight="bold", fontsize=10.5)
fig.tight_layout(); save_fig(fig, "fig17_forecast_vs_actual_h3")

fig, ax = plt.subplots(figsize=(7.5, 5.2)); sub = by_state
im = ax.imshow(sub.values, cmap="Blues", aspect="auto")
ax.set_xticks(range(4), STATES); ax.set_yticks(range(len(sub)), sub.index, fontsize=7.5); ax.grid(False)
for i in range(sub.shape[0]):
    for j in range(4):
        v = sub.iloc[i, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if v > sub.values.max() * 0.6 else INK)
ax.set_title("MAE (pp) by model and state, test window")
fig.tight_layout(); save_fig(fig, "fig18_mae_model_state_heatmap")

(BASE_DIR / "outputs" / "models").mkdir(exist_ok=True, parents=True)
joblib.dump({"ml_templates": BEST, "sarima_choice": SARIMA_CHOICE, "feats": FEATS, "fsets": FSETS, "headline": HEADLINE,
             "ens_def": ENS_DEF}, BASE_DIR / "outputs" / "models" / "forecasting_config.joblib")
save_json({"headline": HEADLINE, "headline_members": ENS_DEF[HEADLINE], "best_test_model": champ, "benchmark": bench, "sarima_choice": SARIMA_CHOICE,
           "interval_coverage_80_test": float(cov_df.covered.mean()), "wb_baseline_three_wheelers": wb_base3w, "feature_sets": FSETS}, "04_summary")
print("Best on test:", champ)
