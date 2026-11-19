"""
09_aggregate_analysis.py -- ONE combined series (4 states pooled): forecast it, then look for the drivers
===========================================================================================================
Aggregate EV share = sum(EV registrations) / sum(all registrations) over Delhi, Maharashtra, Tamil Nadu and West Bengal.
(This is a four-state aggregate, NOT India; Maharashtra's volume dominates it.)

Part A  FORECAST  : same rolling-origin protocol as stage 04 (test targets Jan-2025..cut-off, horizons 1-6, validation 2024),
                    release-aligned covariates (vintage panel), models run on a single series.
Part B  COMPARE   : top-down (this aggregate model) vs bottom-up (state SARIMA forecasts weighted by trailing volumes).
Part C  EXPLAIN   : same-month associations - ML + SHAP and HAC regressions on the aggregate (n ~ 56 months: treat as indicative).
Two versions of the series are used:  as_reported (PRINCIPAL; the 4 flagged West Bengal months are excluded from the scores,
exactly as in stage 04) and backlog_removed (sensitivity: excess e-rickshaw registrations taken out of numerator and
denominator, stage 07, all months scored; its R2 is lower because the removal is an estimate and the series is noisier).
Run from the project root after stages 00-04:  python 09_aggregate_analysis.py
"""
from common import *
from forecasting_utils import *
from vintage import backlog_adjusted_rate

import shap
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = load_data(complete=False)        # registrations through Sep-2026 (forecast part); Part C trims to complete-covariate months
vp = pd.read_csv(BASE_DIR / "data" / "ev_panel_vintage.csv", parse_dates=["date"])
vp = vp[vp.date <= DATA_CUTOFF]
protocol = json.loads((TAB_DIR / "02_selected_features_and_protocol.json").read_text())
SEL = protocol["selected_drivers"]
EXOG_F1 = [c for c in SEL if c not in ("charging_stations", "nsdp_per_capita")]
EXOG_F2 = EXOG_F1 + [c for c in ("charging_stations", "nsdp_per_capita") if c in SEL]
FSETS = {"F0": [], "F1": EXOG_F1, "F2": EXOG_F2}
HORIZONS = [1, 2, 3, 4, 5, 6]
dates = sorted(df.date.unique()); N_T = len(dates)
T_TRAIN_END = int(df.loc[df.date == TRAIN_END, "t"].iloc[0]); EVAL_T = list(range(24, N_T))

# ----------------------------------------------------------------------------- build the aggregate series
adj, base3w = backlog_adjusted_rate(df)
g = df.groupby("date")[["ev_count", "vehicle_total"]].sum()
y_rep = (g.ev_count / g.vehicle_total).to_numpy()
wb = df[df.state == ANOMALY_STATE].sort_values("date").set_index("date")
excess = ((wb.three_wheeler - base3w).clip(lower=0) * wb.anomaly_flag)
ev_adj = (g.ev_count - np.minimum(excess, wb.ev_count)).clip(lower=1); tot_adj = (g.vehicle_total - excess).clip(lower=1)
y_adj = (ev_adj / tot_adj).to_numpy()
FLAG = (df[df.state == ANOMALY_STATE].sort_values("date").anomaly_flag.to_numpy())
SERIES = {"as_reported": y_rep, "backlog_removed": y_adj}
PRIN = "as_reported"
print(f"Aggregate series: {N_T} months | flagged months {int(FLAG.sum())} | WB baseline three-wheelers {base3w:.0f}")
print("Aggregate EV share % (as reported vs backlog removed), flagged window:")
print(pd.DataFrame({"as_reported_%": 100 * y_rep, "backlog_removed_%": 100 * y_adj}, index=pd.to_datetime(dates))[FLAG == 1].round(2).to_string())


def agg_exog(frame: pd.DataFrame, cols: list) -> pd.DataFrame:
    gg = frame.sort_values(["date", "state"]).groupby("date")
    out = pd.DataFrame({c: (gg[c].sum(min_count=4) if c == "charging_stations" else gg[c].mean()) for c in cols})
    return out.reset_index(drop=True)


EX_V = agg_exog(vp, EXOG_F2)                                     # release-aligned (as known at the origin)
EX_S = agg_exog(df, EXOG_F2)                                     # same-month (retrospective layer)
FEATS = {fs: ml_feature_names(ex) for fs, ex in FSETS.items()}

# ----------------------------------------------------------------------------- Part A: forecasting
def run_series(name: str, y: np.ndarray):
    Yd = {"ALL": y}
    rows = {fs: {h: make_rows(Yd, {"ALL": EX_V}, ex, h, range(14, N_T), states=["ALL"]) for h in HORIZONS} for fs, ex in FSETS.items()}
    best = {}
    for fs in FSETS:                                             # tiny data (<40 rows): Ridge alpha by expanding-window validation
        for h in HORIZONS:
            tr = rows[fs][h].query("t <= @T_TRAIN_END").reset_index(drop=True)
            folds = [(tr.index[tr.t < a].to_numpy(), tr.index[(tr.t >= a) & (tr.t <= b)].to_numpy()) for a, b in [(24, 29), (30, 35)]]
            folds = [f for f in folds if len(f[0]) >= 5 and len(f[1]) >= 1]
            est = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge())
            gs = GridSearchCV(est, {"ridge__alpha": [1, 3, 10, 30, 100, 300]}, cv=folds, scoring="neg_mean_absolute_error", error_score=np.nan).fit(tr[FEATS[fs]], tr["target"])
            best[(fs, h)] = gs.best_estimator_
    # SARIMA spec on validation origins
    spec_scores = {}
    for spec in SARIMA_SPECS:
        errs = []
        for o in range(23, 30):
            f = fit_forecast_sarima(y[:o + 1], 6, spec); tgt = np.arange(o + 1, o + 7); ok = tgt <= T_TRAIN_END
            errs += list(np.abs(f[ok] - y[tgt[ok]]))
        spec_scores[spec] = np.mean(errs)
    spec = min(spec_scores, key=spec_scores.get)
    recs = []

    def add(m, o, h, p):
        t = o + h; recs.append({"series": name, "model": m, "o": o, "t": t, "h": h, "date": pd.Timestamp(dates[t]), "y_true": y[t], "y_pred": float(p), "flag": int(FLAG[t])})

    origins = sorted({t - h for t in EVAL_T for h in HORIZONS if t - h >= 17})
    for o in origins:
        hist = y[:o + 1]; ets = fit_forecast_ets(hist, 6); sar = fit_forecast_sarima(hist, 6, spec)
        avg = np.exp(np.mean([np.log(fit_forecast_sarima(hist, 6, sp)) for sp in SARIMA_SPECS], axis=0))
        b = base_level(y, o); b_ly = np.mean(y[o - 14:o - 11])
        for h in HORIZONS:
            t = o + h
            if t not in EVAL_T: continue
            add("Naive (last value)", o, h, y[o]); add("Moving average (3m)", o, h, b); add("Seasonal naive", o, h, y[t - 12])
            add("Seasonal naive + YoY growth", o, h, y[t - 12] * b / b_ly); add("ETS", o, h, ets[h - 1]); add("SARIMA", o, h, sar[h - 1])
            add("SARIMA-avg4 (all specs averaged)", o, h, avg[h - 1])
    for fs in FSETS:
        for h in HORIZONS:
            rh = rows[fs][h]
            for o in [t - h for t in EVAL_T]:
                tr, te = rh[rh.t <= o], rh[rh.o == o]
                if len(tr) < 8 or te.empty: continue
                m = clone(best[(fs, h)]).fit(tr[FEATS[fs]], tr["target"])
                add(f"Ridge-{fs}", o, h, te["base"].iloc[0] * np.exp(m.predict(te[FEATS[fs]])[0]))
    ev = pd.DataFrame(recs)
    ens = ev[ev.model.isin(["SARIMA-avg4 (all specs averaged)", "ETS", "Ridge-F0"])].groupby(["series", "o", "t", "h", "date", "y_true", "flag"], as_index=False).y_pred.mean()
    ens["model"] = "Ensemble (SARIMA-avg4 + ETS + Ridge-F0)"
    ev = pd.concat([ev, ens], ignore_index=True); ev["abs_err"] = (ev.y_true - ev.y_pred).abs()
    ev["period"] = np.where(ev.t <= T_TRAIN_END, "validation", "test")
    return ev, spec


all_ev, specs = [], {}
for name, y in SERIES.items():
    e_, sp_ = run_series(name, y); all_ev.append(e_); specs[name] = sp_
    print(f"series {name}: SARIMA spec chosen on validation = {sp_}", flush=True)
EV = pd.concat(all_ev, ignore_index=True)
save_table(EV, "09_aggregate_rolling_predictions", index=False)

naive_mae = {}
tabs = {}
for name in SERIES:
    t_ = EV[(EV.series == name) & (EV.period == "test")]
    t_ = t_[t_.flag == 0] if name == "as_reported" else t_             # backlog_removed: every test month is scored
    mase_scale = np.mean(np.abs(np.diff(SERIES[name][:T_TRAIN_END + 1])))
    rows = []
    for m, d in t_.groupby("model"):
        y, p = d.y_true.to_numpy(), d.y_pred.to_numpy()
        rows.append({"model": m, "MAE_pp": 100 * np.mean(np.abs(y - p)), "RMSE_pp": 100 * np.sqrt(np.mean((y - p) ** 2)), "sMAPE_%": 100 * smape(y, p),
                     "MAPE_%": 100 * np.mean(np.abs(y - p) / y), "MASE": np.mean(np.abs(y - p)) / mase_scale, "R2": r2(y, p), "n": len(d)})
    tab = pd.DataFrame(rows).set_index("model").sort_values("MAE_pp")
    tab["skill_vs_naive_%"] = 100 * (1 - tab.MAE_pp / tab.loc["Naive (last value)", "MAE_pp"])
    tabs[name] = tab.round(3)
    print(f"\n=== TEST window, aggregate series = {name}  (targets {TEST_START:%b-%Y}..{DATA_CUTOFF:%b-%Y}, horizons 1-6) ===")
    print(tabs[name].to_string())
    save_table(tabs[name], f"09_aggregate_metrics_{name}")
    r2h = t_[t_.model == tab.index[0]].groupby("h").apply(lambda d: r2(d.y_true, d.y_pred)).round(3)
    print(f"R2 by horizon, best model ({tab.index[0]}):", r2h.to_dict())
    save_table(t_.groupby(["model", "h"]).abs_err.mean().mul(100).unstack("h").loc[tab.index].round(3), f"09_aggregate_mae_by_horizon_{name}")


def boot_diff(a, b, frame, n=4000, seed=SEED):
    A = frame[frame.model == a].set_index(["h", "t"]).abs_err; B = frame[frame.model == b].set_index(["h", "t"]).abs_err
    d = (A - B).dropna().rename("d").reset_index(); sums = d.groupby("t").d.sum().to_numpy(); cnt = d.groupby("t").d.size().to_numpy()
    rng = np.random.default_rng(seed); k = len(sums); picks = rng.integers(0, k, size=(n, k)); means = sums[picks].sum(1) / cnt[picks].sum(1)
    lo, hi = np.percentile(means, [5, 95]); return 100 * d.d.mean(), 100 * lo, 100 * hi


main = EV[(EV.series == PRIN) & (EV.period == "test") & (EV.flag == 0)]
rows = []
for m in tabs[PRIN].index:
    if m == "Naive (last value)": continue
    dm, lo, hi = boot_diff(m, "Naive (last value)", main)
    rows.append({"model": m, "MAE diff vs naive (pp)": dm, "90% CI low": lo, "90% CI high": hi, "significantly_better": hi < 0})
sig = pd.DataFrame(rows).set_index("model").round(3); print("\nVs naive, aggregate (as_reported, flagged months excluded):\n", sig.to_string()); save_table(sig, "09_aggregate_bootstrap_vs_naive")
cov = []
for fs in ("F1", "F2"):
    dm, lo, hi = boot_diff(f"Ridge-{fs}", "Ridge-F0", main); cov.append({"comparison": f"Ridge-{fs} minus Ridge-F0", "MAE diff (pp)": dm, "90% CI low": lo, "90% CI high": hi})
cov = pd.DataFrame(cov).round(3); print("\nDo release-aligned covariates help the aggregate forecast? (negative = yes)\n", cov.to_string(index=False)); save_table(cov, "09_aggregate_covariate_value", index=False)

# ----------------------------------------------------------------------------- Part B: top-down vs bottom-up
st = pd.read_csv(TAB_DIR / "04_rolling_origin_predictions.csv", parse_dates=["date"])
st = st[(st.model == "SARIMA-avg4 (all specs averaged)") & (st.period == "test")]
vol = df.pivot(index="t", columns="state", values="vehicle_total")[STATES]
rows = []
for (o, t, h), d in st.groupby(["o", "t", "h"]):
    w = vol.loc[max(0, o - 11):o].sum(); w = w / w.sum()
    rows.append({"o": o, "t": t, "h": h, "bottom_up": float((d.set_index("state").y_pred.reindex(STATES) * w).sum())})
bu = pd.DataFrame(rows)
td = EV[(EV.series == "as_reported") & (EV.model == "SARIMA-avg4 (all specs averaged)") & (EV.period == "test")][["o", "t", "h", "y_true", "y_pred", "flag"]].rename(columns={"y_pred": "top_down"})
cmp_ = td.merge(bu, on=["o", "t", "h"]); cmp_ = cmp_[cmp_.flag == 0]
bu_tab = pd.DataFrame({"top-down (aggregate SARIMA-avg4)": [100 * (cmp_.y_true - cmp_.top_down).abs().mean(), r2(cmp_.y_true, cmp_.top_down)],
                       "bottom-up (state SARIMA-avg4, volume-weighted)": [100 * (cmp_.y_true - cmp_.bottom_up).abs().mean(), r2(cmp_.y_true, cmp_.bottom_up)]}, index=["MAE_pp", "R2"]).round(3)
print("\nTop-down vs bottom-up (as-reported aggregate, flagged months excluded):\n", bu_tab.to_string()); save_table(bu_tab, "09_top_down_vs_bottom_up")

# ----------------------------------------------------------------------------- figures (forecast part)
set_style()
dts = pd.to_datetime(dates)
fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
ax.plot(dts, 100 * y_rep, color=INK, lw=1.8, label="as reported (principal)"); ax.plot(dts, 100 * y_adj, color="#c3c2b7", lw=1.4, label="backlog removed")
ax.axvspan(dts[FLAG == 1].min(), dts[FLAG == 1].max(), color="#eda100", alpha=.15)
for m, c in [("SARIMA-avg4 (all specs averaged)", "#eb6834"), ("Ensemble (SARIMA-avg4 + ETS + Ridge-F0)", "#2a78d6"), ("Naive (last value)", "#898781")]:
    d = EV[(EV.series == PRIN) & (EV.model == m) & (EV.h == 3)].sort_values("t")
    ax.plot(d.date, 100 * d.y_pred, color=c, ls=":" if m.startswith("Naive") else "-", marker="o", ms=2.5, lw=1.4, label=f"{m.split(' (')[0]} (3 months ahead)")
ax.set_ylabel("EV share of registrations (%)"); ax.set_title("Four-state aggregate EV share and 3-month-ahead forecasts", fontsize=10); ax.legend(fontsize=7, ncol=2)
ax = axes[1]; bt = tabs[PRIN]; show = ["Naive (last value)", "Seasonal naive", "ETS", "SARIMA", "SARIMA-avg4 (all specs averaged)", "Ridge-F0", "Ridge-F1", "Ridge-F2", "Ensemble (SARIMA-avg4 + ETS + Ridge-F0)"]
vals = bt.loc[show, "MAE_pp"]; ax.barh(range(len(show)), vals, color=["#c3c2b7" if "Naive" in s or "Seasonal" in s else ("#2a78d6" if "Ridge" in s else "#eb6834") for s in show])
ax.set_yticks(range(len(show)), [s.split(" (")[0] for s in show], fontsize=8); ax.invert_yaxis()
for i, v in enumerate(vals): ax.text(v + .01, i, f"{v:.2f}", va="center", fontsize=8)
ax.set_xlabel("MAE (pp), test window"); ax.set_title("Aggregate accuracy", fontsize=10)
fig.tight_layout(); save_fig(fig, "fig22_aggregate_forecast")

# ----------------------------------------------------------------------------- Part C: explanation on the aggregate
N_C = int((pd.to_datetime(dates) <= COVARIATE_CUTOFF).sum())      # months with every covariate observed (through Aug-2026)
print("\n" + "=" * 90 + "\nPART C - aggregate drivers (same-month associations, n = %d months)\n" % N_C + "=" * 90)
D = EX_S.iloc[:N_C][SEL].copy(); D["month"] = pd.to_datetime(dates[:N_C]).month; D["t"] = np.arange(N_C)
D["month_sin"] = np.sin(2 * np.pi * D.month / 12); D["month_cos"] = np.cos(2 * np.pi * D.month / 12); D["fy_end"] = (D.month == 3).astype(int)
D["y"] = y_adj[:N_C]; D["log_y"] = np.log(y_adj[:N_C])
CAL = ["month_sin", "month_cos", "fy_end"]

from statsmodels.stats.outliers_influence import variance_inflation_factor as _vif
X0 = (D[SEL] - D[SEL].mean()) / D[SEL].std()
vif = pd.Series([_vif(sm.add_constant(X0).values, i + 1) for i in range(len(SEL))], index=SEL).round(1)
print("VIF of aggregate drivers (standardised):", vif.to_dict()); save_table(vif.to_frame("VIF"), "09_aggregate_vif")
print("Correlation of each driver with calendar time (aggregate):", D[SEL + ["t"]].corr()["t"].drop("t").round(2).to_dict())

# --- ML explanatory models with expanding-window validation, hold-out = test window
tr_idx, te_idx = D.index[D.t <= T_TRAIN_END], D.index[D.t > T_TRAIN_END]
te_ok = [i for i in te_idx if FLAG[i] == 0]
folds = [(D.index[D.t < a].to_numpy(), D.index[(D.t >= a) & (D.t <= b)].to_numpy()) for a, b in [(18, 23), (24, 29), (30, 35)]]
folds = [(a, b) for a, b in folds if len(a) >= 8]
FSETS_X = {"calendar only": CAL, "trend + calendar": CAL + ["t"], "drivers + calendar": SEL + CAL}
mods = {"Ridge": (make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [0.1, 1, 3, 10, 30, 100]}),
        "RandomForest": (RandomForestRegressor(random_state=SEED, n_jobs=1), {"n_estimators": [150], "max_depth": [2, 3, 4], "min_samples_leaf": [2, 4, 6], "max_features": [0.5, 1.0]})}
res, fitted = [], {}
for fsn, cols in FSETS_X.items():
    for mn, (est, grid) in mods.items():
        if fsn == "trend + calendar" and mn == "RandomForest": continue          # trees cannot extrapolate a trend
        gs = GridSearchCV(est, grid, cv=folds, scoring="neg_mean_absolute_error").fit(D.loc[tr_idx, cols], D.loc[tr_idx, "log_y"])
        pred = np.exp(gs.predict(D.loc[te_ok, cols])); y_te = D.loc[te_ok, "y"].to_numpy()
        res.append({"feature set": fsn, "model": mn, "cv_MAE (log)": -gs.best_score_, "holdout_MAE_pp": 100 * np.mean(np.abs(y_te - pred)), "holdout_R2": r2(y_te, pred)})
        fitted[(fsn, mn)] = gs.best_estimator_
ref = np.full(len(te_ok), D.loc[tr_idx, "y"].iloc[-12:].mean())
res.append({"feature set": "reference: mean of 2024", "model": "constant", "cv_MAE (log)": np.nan, "holdout_MAE_pp": 100 * np.mean(np.abs(D.loc[te_ok, "y"].to_numpy() - ref)), "holdout_R2": r2(D.loc[te_ok, "y"].to_numpy(), ref)})
res = pd.DataFrame(res).round(3).sort_values("holdout_MAE_pp"); print("\nSame-month explanatory models on the aggregate (hold-out = test window, flagged months excluded):\n", res.to_string(index=False))
save_table(res, "09_aggregate_explanatory_models", index=False)

# --- SHAP on the best driver model fitted to ALL months (descriptive)
best_drv = res[res["feature set"] == "drivers + calendar"].sort_values("holdout_MAE_pp").iloc[0]
rf = clone(fitted[("drivers + calendar", "RandomForest")]).fit(D[SEL + CAL], D["log_y"])
sv = shap.TreeExplainer(rf).shap_values(D[SEL + CAL])
shap_imp = pd.Series(np.abs(sv).mean(0), index=SEL + CAL).sort_values(ascending=False)
shap_dir = pd.Series({c: np.sign(np.corrcoef(D[c], sv[:, (SEL + CAL).index(c)])[0, 1]) if np.std(sv[:, (SEL + CAL).index(c)]) > 0 else 0 for c in SEL})
print("\nSHAP mean |value| (random forest, all months):\n", shap_imp.round(4).to_string())

# --- HAC regressions (spec A: month dummies; spec B: + linear trend; spec C: first differences)
Z = D.copy(); Z[SEL] = (Z[SEL] - Z[SEL].mean()) / Z[SEL].std()
fa = "log_y ~ " + " + ".join(SEL) + " + C(month)"; fb = fa + " + t"
ma = smf.ols(fa, Z).fit(cov_type="HAC", cov_kwds={"maxlags": 3}); mb = smf.ols(fb, Z).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
dZ = Z[["log_y"] + SEL].diff().dropna(); dZ["month"] = Z.month.iloc[1:].values
mc = smf.ols("log_y ~ " + " + ".join(SEL), dZ).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
tri = pd.DataFrame({"coef_A (month FE)": ma.params[SEL], "p_A": ma.pvalues[SEL], "coef_B (+trend)": mb.params[SEL], "p_B": mb.pvalues[SEL],
                    "coef_C (differences)": mc.params[SEL], "p_C": mc.pvalues[SEL], "SHAP importance": shap_imp[SEL], "SHAP direction": shap_dir})
tri["robust"] = ((np.sign(tri["coef_A (month FE)"]) == np.sign(tri["coef_B (+trend)"])) & (np.sign(tri["coef_A (month FE)"]) == tri["SHAP direction"]) & (tri.p_A < .10) & (tri.p_B < .10))
print("\nDriver triangulation, aggregate (coefficients = effect on log share of +1 SD):\n", tri.round(3).to_string())
print(f"\nRobust drivers (same sign in SHAP, A and B; p<0.10 in A and B): {list(tri.index[tri.robust]) or 'none'}")
print(f"Spec A R2 = {ma.rsquared:.2f}, spec B R2 = {mb.rsquared:.2f}, first-difference R2 = {mc.rsquared:.2f}")
save_table(tri.round(4), "09_aggregate_driver_triangulation")
save_json({"robust_drivers": list(tri.index[tri.robust]), "n_months": int(N_C), "spec_A_R2": float(ma.rsquared), "spec_B_R2": float(mb.rsquared), "diff_R2": float(mc.rsquared),
           "best_forecast_model_principal_series": tabs[PRIN].index[0], "best_forecast_MAE_pp": float(tabs[PRIN].MAE_pp.iloc[0]), "best_forecast_R2": float(tabs[PRIN].R2.iloc[0]),
           "sarima_spec_chosen": specs}, "09_summary")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
ax = axes[0]; imp = shap_imp.sort_values(); ax.barh(range(len(imp)), imp.values, color=["#eb6834" if i in SEL else "#c3c2b7" for i in imp.index])
ax.set_yticks(range(len(imp)), imp.index, fontsize=8); ax.set_title("SHAP importance (random forest, aggregate)", fontsize=10); ax.set_xlabel("mean |SHAP| on log share")
ax = axes[1]
for k, (mdl, col, off) in enumerate([(ma, "#2a78d6", -.15), (mb, "#eb6834", .15)]):
    ci = mdl.conf_int().loc[SEL]; ax.errorbar(mdl.params[SEL], np.arange(len(SEL)) + off, xerr=[mdl.params[SEL] - ci[0], ci[1] - mdl.params[SEL]], fmt="o", color=col, ms=4, lw=1.2, label=["A: month effects", "B: + trend"][k])
ax.axvline(0, color=INK2, lw=1); ax.set_yticks(range(len(SEL)), SEL, fontsize=8); ax.invert_yaxis(); ax.legend(fontsize=8, loc="lower right")
ax.set_title("HAC regression effect of +1 SD (95% CI)", fontsize=10); ax.set_xlabel("effect on log EV share")
fig.tight_layout(); save_fig(fig, "fig23_aggregate_drivers")
