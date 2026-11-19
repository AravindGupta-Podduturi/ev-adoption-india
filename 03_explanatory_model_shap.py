"""
03_explanatory_model_shap.py -- Stage 03: explanatory models (SHAP) and fixed-effects triangulation - RETROSPECTIVE layer
Run from the project root:  python 03_explanatory_model_shap.py
"""
from common import *

import shap
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import spearmanr, randint, uniform, loguniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf

df = load_data()
protocol = json.loads((TAB_DIR / "02_selected_features_and_protocol.json").read_text())
DRIVERS = protocol["selected_drivers"]
CAL = protocol["calendar_features"]
FEATURES = DRIVERS + CAL
print("Features:", FEATURES)

train = df[df.date <= TRAIN_END].reset_index(drop=True)
test = df[(df.date >= TEST_START) & (df.date <= COVARIATE_CUTOFF)].reset_index(drop=True)
print(f"train rows={len(train)}  test rows={len(test)}")

# ## 1. Time-aware hyper-parameter search (training period only)
# Three expanding-window folds: validate on 2023-H2, 2024-H1, 2024-H2, always training on the past.

def time_folds(frame: pd.DataFrame):
    cuts = [("2023-07-01", "2023-12-01"), ("2024-01-01", "2024-06-01"), ("2024-07-01", "2024-12-01")]
    folds = []
    for a, b in cuts:
        tr = frame.index[frame.date < a].to_numpy()
        va = frame.index[(frame.date >= a) & (frame.date <= b)].to_numpy()
        folds.append((tr, va))
    return folds


folds = time_folds(train)
Xtr, ytr = train[FEATURES], train[TARGET]

search_space = {
    "Ridge": (make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": loguniform(1e-2, 1e2)}, 25),
    "RandomForest": (RandomForestRegressor(random_state=SEED, n_jobs=-1),
                     {"n_estimators": randint(200, 600), "max_depth": randint(2, 8),
                      "min_samples_leaf": randint(2, 12), "max_features": uniform(0.4, 0.6)}, 30),
    "XGBoost": (xgb.XGBRegressor(random_state=SEED, n_jobs=1, objective="reg:squarederror"),
                {"n_estimators": randint(100, 500), "max_depth": randint(2, 5), "learning_rate": loguniform(0.02, 0.2),
                 "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.5, 0.5),
                 "min_child_weight": randint(1, 8), "reg_lambda": loguniform(0.1, 20)}, 40),
    "LightGBM": (lgb.LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1),
                 {"n_estimators": randint(100, 500), "num_leaves": randint(4, 20), "learning_rate": loguniform(0.02, 0.2),
                  "subsample": uniform(0.6, 0.4), "subsample_freq": [1], "colsample_bytree": uniform(0.5, 0.5),
                  "min_child_samples": randint(4, 20), "reg_lambda": loguniform(0.1, 20)}, 40),
}
tuned, cv_rows = {}, []
for name, (est, space, n_iter) in search_space.items():
    rs = RandomizedSearchCV(est, space, n_iter=n_iter, cv=folds, scoring="neg_mean_absolute_error",
                            random_state=SEED, n_jobs=-1, refit=True).fit(Xtr, ytr)
    tuned[name] = rs.best_estimator_
    cv_rows.append({"model": name, "cv_MAE_pp": -100 * rs.best_score_,
                    "best_params": {k.split("__")[-1]: (round(float(v), 4) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (int, np.integer)) else v))
                                    for k, v in rs.best_params_.items()}})
cv_tab = pd.DataFrame(cv_rows).set_index("model")
print(cv_tab.to_string())
save_table(cv_tab.assign(best_params=cv_tab.best_params.astype(str)), "03_tuning_results")

# ## 2. Validation: chronological hold-out and leave-one-state-out

from sklearn.base import clone

res = []
preds_test = {}
for name, est in tuned.items():
    p = est.predict(test[FEATURES]); preds_test[name] = p
    d = pd.DataFrame({"y_true": test[TARGET], "y_pred": p, "flag": test.anomaly_flag})
    clean = d[d.flag == 0]
    # leave-one-state-out (train on 3 states over the training period, predict the 4th state's 2022-25)
    lo = []
    for s in STATES:
        m = clone(est).fit(df.loc[(df.state != s) & (df.date <= TRAIN_END), FEATURES],
                           df.loc[(df.state != s) & (df.date <= TRAIN_END), TARGET])
        sub = df[(df.state == s) & (df.anomaly_flag == 0)]
        lo.append(pd.DataFrame({"y_true": sub[TARGET], "y_pred": m.predict(sub[FEATURES])}))
    lo = pd.concat(lo)
    res.append({"model": name,
                "holdout_MAE_pp (all)": 100 * mae(d.y_true, d.y_pred),
                "holdout_MAE_pp (excl. flagged)": 100 * mae(clean.y_true, clean.y_pred),
                "holdout_R2 (excl. flagged)": r2(clean.y_true, clean.y_pred),
                "holdout_sMAPE_% (excl. flagged)": 100 * smape(clean.y_true, clean.y_pred),
                "LOSO_MAE_pp": 100 * mae(lo.y_true, lo.y_pred), "LOSO_R2": r2(lo.y_true, lo.y_pred)})
# reference: predict every out-of-sample month with the state's 2024 mean (what a trivial model would do)
ref = test.merge(train[train.year == 2024].groupby("state")[TARGET].mean().rename("ref"), on="state")
ref = ref[ref.anomaly_flag == 0]
res.append({"model": "Reference: state's 2024 mean", "holdout_MAE_pp (all)": np.nan,
            "holdout_MAE_pp (excl. flagged)": 100 * mae(ref[TARGET], ref.ref),
            "holdout_R2 (excl. flagged)": r2(ref[TARGET], ref.ref)})
val_tab = pd.DataFrame(res).set_index("model").round(3)
print(val_tab.to_string())
save_table(val_tab, "03_explanatory_validation")

# Model choice uses ONLY the training-period CV score (never the hold-out), so the hold-out stays an honest test.
best_name = cv_tab["cv_MAE_pp"].idxmin()
print(f"\nSelected explanatory model (lowest time-series-CV MAE on 2022-24): {best_name}")
print(f"  its hold-out MAE (excl. flagged) = {val_tab.loc[best_name, 'holdout_MAE_pp (excl. flagged)']:.2f} pp  vs  "
      f"reference {val_tab.loc[chr(82)+'eference: state' + chr(39) + 's 2024 mean', 'holdout_MAE_pp (excl. flagged)']:.2f} pp")

# ### Hold-out fit plot

set_style()
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=False)
for ax, s in zip(axes, STATES):
    g = df[df.state == s]
    ax.plot(g.date, 100 * g[TARGET], color=INK2, lw=1.2, label="actual")
    mt = test.state == s
    ax.plot(test.date[mt], 100 * preds_test[best_name][mt.values], color=STATE_COLORS[s], lw=2, label=f"{best_name} (out-of-sample)")
    ax.axvline(TEST_START, color=MUTED, ls=":")
    ax.set_title(s, fontsize=10); ax.tick_params(axis="x", labelrotation=45, labelsize=7)
axes[0].set_ylabel("EV share (%)"); axes[0].legend(fontsize=7.5)
fig.tight_layout()
save_fig(fig, "fig10_explanatory_holdout_fit")

# ## 3. SHAP explanations
# The model explains observations inside the range of history; it is refit on **all 192 state-months** for
# interpretation (standard practice for explanation; validation above is what justifies trusting it).

final_model = clone(tuned[best_name]).fit(df[FEATURES], df[TARGET])
X_all = df[FEATURES]
explainer = shap.TreeExplainer(final_model)
sv = explainer(X_all)                       # shap.Explanation (values in EV-share units)
shap_df = pd.DataFrame(sv.values, columns=FEATURES, index=df.index)
base = float(np.ravel(sv.base_values)[0])
print(f"Base value (average prediction) = {100 * base:.2f} %;  additivity check max err = "
      f"{np.abs(shap_df.sum(axis=1) + base - final_model.predict(X_all)).max():.2e}")

glob = shap_df.abs().mean().mul(100).sort_values(ascending=False).rename("mean_|SHAP|_pp")
grp = glob.groupby(lambda c: COL_TO_GROUP[c]).sum().sort_values(ascending=False).rename("group_mean_|SHAP|_pp")
print(glob.round(3).to_string()); print(); print(grp.round(3).to_string())
save_table(glob.round(4).to_frame(), "03_shap_global_importance")
save_table(grp.round(4).to_frame(), "03_shap_group_importance")

# Is a feature's SHAP signal a *state identity* effect or a *within-state over-time* effect?
def between_share(series):
    return float(series.groupby(df.state).transform("mean").var(ddof=0) / series.var(ddof=0))


shap_bw = pd.DataFrame({"mean_|SHAP|_pp": glob,
                        "between_state_share_of_SHAP_variance_%": [100 * between_share(shap_df[c]) for c in glob.index]})
print("\nShare of each feature's SHAP variance that is explained by state identity alone:")
print(shap_bw.round(1).to_string())
save_table(shap_bw.round(3), "03_shap_between_vs_within")

# Fig 11: global importance (by feature, coloured by business group) + group totals
GROUP_COL = {"Charging infrastructure": "#2a78d6", "Income": "#eb6834", "Fuel prices": "#1baf7a",
             "Air quality": "#eda100", "Weather": "#898781", "Seasonality": "#c3c2b7"}
fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.5, 1]})
g_sorted = glob.sort_values()
axes[0].barh(g_sorted.index, g_sorted.values, color=[GROUP_COL[COL_TO_GROUP[c]] for c in g_sorted.index])
axes[0].set_title("Mean |SHAP| by feature (pp of EV share)"); axes[0].set_xlabel("percentage points")
g2 = grp.sort_values()
axes[1].barh(g2.index, g2.values, color=[GROUP_COL[g] for g in g2.index])
axes[1].set_title("By business driver group"); axes[1].set_xlabel("percentage points")
fig.tight_layout()
save_fig(fig, "fig11_shap_global_importance")

# Fig 12: beeswarm
plt.figure(figsize=(8.5, 5))
shap.summary_plot(sv, X_all, show=False, max_display=len(FEATURES), plot_size=(8.5, 5))
plt.title("SHAP summary: direction and size of each feature's effect", loc="left", fontsize=11, fontweight="bold")
save_fig(plt.gcf(), "fig12_shap_beeswarm")

# Fig 13: dependence plots for the 4 most important drivers (colour = state)
top4 = [c for c in glob.index if c in DRIVERS][:4]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for ax, c in zip(axes, top4):
    for s in STATES:
        m = df.state == s
        ax.scatter(df.loc[m, c], 100 * shap_df.loc[m, c], s=14, color=STATE_COLORS[s], alpha=0.85,
                   edgecolor=SURFACE, linewidth=0.4, label=s)
    ax.axhline(0, color=MUTED, lw=0.8); ax.set_xlabel(c); ax.set_title(f"SHAP vs {c}", fontsize=9)
axes[0].set_ylabel("SHAP value (pp)"); axes[0].legend(fontsize=7)
fig.tight_layout()
save_fig(fig, "fig13_shap_dependence")

# Fig 14: why does each state differ?  mean SHAP per state, by group (deviation from the 4-state baseline)
state_grp = shap_df.groupby(df.state).mean().T.groupby(lambda c: COL_TO_GROUP[c]).sum().mul(100)[STATES]
state_grp["_abs"] = state_grp.abs().sum(axis=1); state_grp = state_grp.sort_values("_abs", ascending=False).drop(columns="_abs")
fig, ax = plt.subplots(figsize=(8, 3.6))
im = ax.imshow(state_grp.values, cmap="RdBu_r", vmin=-8, vmax=8, aspect="auto")
ax.set_xticks(range(4), STATES); ax.set_yticks(range(len(state_grp)), state_grp.index); ax.grid(False)
for i in range(state_grp.shape[0]):
    for j in range(4):
        ax.text(j, i, f"{state_grp.iloc[i, j]:+.1f}", ha="center", va="center", fontsize=9,
                color="white" if abs(state_grp.iloc[i, j]) > 4.5 else INK)
ax.set_title(f"Average contribution to EV share vs baseline ({100 * base:.1f}%), pp")
fig.colorbar(im, ax=ax, shrink=0.8)
save_fig(fig, "fig14_state_contribution_heatmap")
save_table(state_grp.round(3), "03_state_group_contributions")
print(state_grp.round(2).to_string())

# Fig 15: local explanations (waterfall) - the latest month of the two states at both ends of the ranking
def waterfall(i, title, name):
    plt.figure(figsize=(7.5, 4.2))
    shap.plots.waterfall(sv[i], max_display=8, show=False)
    plt.title(title, loc="left", fontsize=10, fontweight="bold")
    save_fig(plt.gcf(), name)


LATEST = df.date.max()
i_del = df.index[(df.state == "Delhi") & (df.date == LATEST)][0]
i_wb = df.index[(df.state == "West Bengal") & (df.date == LATEST)][0]
waterfall(i_del, f"Delhi, {LATEST:%b %Y}: drivers of the predicted EV share (0.14 = 14%)", "fig15a_waterfall_delhi")
waterfall(i_wb, f"West Bengal, {LATEST:%b %Y}: drivers of the predicted EV share (fractions)", "fig15b_waterfall_wb")

# ## 4. Robustness of the explanation
# (a) Do the SHAP rankings change when the model is trained on the training period only?
# (b) ... or when a state is left out?  (c) Does a fixed-effects regression agree on the signs?

def rank_vec(model, rows):
    s = pd.DataFrame(shap.TreeExplainer(model)(rows[FEATURES]).values, columns=FEATURES).abs().mean()
    return s.reindex(FEATURES)


ref_rank = glob.reindex(FEATURES)
stab = {"train-period model (2022-24)": rank_vec(clone(tuned[best_name]).fit(train[FEATURES], train[TARGET]), df)}
for s in STATES:
    sub = df[df.state != s]
    stab[f"without {s}"] = rank_vec(clone(tuned[best_name]).fit(sub[FEATURES], sub[TARGET]), df)
stab_tab = pd.DataFrame({k: {"spearman_vs_full_fit": spearmanr(v, ref_rank)[0],
                             "top3": ", ".join(v.sort_values(ascending=False).index[:3])} for k, v in stab.items()}).T
print(stab_tab.to_string())
save_table(stab_tab, "03_shap_stability")

# Fixed-effects regression: log(EV share) on standardised drivers, state + calendar-month fixed effects
fe_df = df.copy()
for c in DRIVERS:
    fe_df[c + "_z"] = (fe_df[c] - fe_df[c].mean()) / fe_df[c].std()
fe_df["log_rate"] = np.log(fe_df[TARGET]); fe_df = fe_df[fe_df.anomaly_flag == 0]
rhs = " + ".join(c + "_z" for c in DRIVERS)
rows = []
for label, extra in [("Spec A: drivers + state FE + month FE", ""), ("Spec B: A + linear time trend", " + t")]:
    m = smf.ols(f"log_rate ~ {rhs} + C(state) + C(month){extra}", fe_df).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    for c in DRIVERS:
        b, se, pv = m.params[c + "_z"], m.bse[c + "_z"], m.pvalues[c + "_z"]
        rows.append({"spec": label, "driver": c, "pct_change_per_+1SD": 100 * (np.exp(b) - 1), "p_value": pv})
    print(f"{label}:  R2 = {m.rsquared:.3f}, n = {int(m.nobs)}")
fe_tab = pd.DataFrame(rows).pivot(index="driver", columns="spec", values=["pct_change_per_+1SD", "p_value"]).round(3)
print(fe_tab.to_string())
save_table(fe_tab, "03_fixed_effects_regression")

# A driver is called "robust" when (i) SHAP direction, (ii) FE spec A and (iii) FE spec B all agree in sign
# and both FE specs are significant at 10%.  Everything else is reported as fragile.
shap_sign = {c: np.sign(spearmanr(df[c], shap_df[c])[0]) for c in DRIVERS}
A, B = "Spec A: drivers + state FE + month FE", "Spec B: A + linear time trend"
verdict = {}
for c in DRIVERS:
    ba, bb = fe_tab.loc[c, ("pct_change_per_+1SD", A)], fe_tab.loc[c, ("pct_change_per_+1SD", B)]
    pa, pb = fe_tab.loc[c, ("p_value", A)], fe_tab.loc[c, ("p_value", B)]
    same = np.sign(ba) == np.sign(bb) == shap_sign[c]
    verdict[c] = "ROBUST" if (same and pa < 0.1 and pb < 0.1) else "fragile"
verdict_tab = pd.DataFrame({"SHAP_direction": {c: "+" if v > 0 else "-" for c, v in shap_sign.items()},
                            "FE_A_effect_%": fe_tab[("pct_change_per_+1SD", A)], "FE_B_effect_%": fe_tab[("pct_change_per_+1SD", B)],
                            "verdict": pd.Series(verdict)})
print("\nTriangulation of driver evidence:"); print(verdict_tab.to_string())
save_table(verdict_tab, "03_driver_evidence_triangulation")
print("\nNOTE: 'ROBUST' = the association is stable across SHAP and both fixed-effects specifications. It is NOT proof of causality.")
n_robust = sum(v == "ROBUST" for v in verdict.values())
print(f"  * Drivers passing the robustness test: {n_robust} of {len(verdict)}"
      + (" -> NO driver has a stable, significant within-state association once state and calendar effects are controlled." if n_robust == 0 else "."))
pe = verdict_tab.loc["petrol_price"]
pet_signs = {int(np.sign(pe["FE_A_effect_%"])), int(np.sign(pe["FE_B_effect_%"])), 1 if pe.SHAP_direction == "+" else -1}
if len(pet_signs) > 1:
    print("  * petrol_price: SHAP and the fixed-effects regressions DISAGREE on the sign -> no reliable fuel-price effect can be claimed."
          "\n    (Most within-state price variation is one fuel-price step in May-Jun 2026 that coincides with a general upswing.)")
elif pet_signs == {-1}:
    print("  * petrol_price enters NEGATIVELY in all methods (dearer petrol <-> lower EV share), the opposite of the economic prior - most likely"
          "\n    a common-trend / state-composition artefact.  Report it as such.")
else:
    print("  * petrol_price enters POSITIVELY in all methods, in line with the economic prior - but identification rests on a single fuel-price step"
          "\n    in May-Jun 2026, so call it suggestive, not an elasticity.")
if verdict.get("so2_mean") == "ROBUST":
    print("  * so2_mean is probably a proxy for unmodelled seasonality / local activity rather than a driver of adoption.")
import joblib
(BASE_DIR / "outputs" / "models").mkdir(exist_ok=True)
joblib.dump(tuned, BASE_DIR / "outputs" / "models" / "explanatory_models_tuned.joblib")
save_json({"best_explanatory_model": best_name, "baseline_pct": 100 * base, "verdict": verdict}, "03_summary")

# ## 5. Limits of the explanation (read before quoting any SHAP number)
# * **4 states only** - state-level variables (income, charging stock) are effectively 4 distinct points, so their
#   SHAP importance largely reflects *which state it is*.
# * **Charging stations and NSDP are step functions** (3-4 values per state; charging has a level break in Feb-2024 and
#   again in Dec-2025, probably a source / definition change) - they cannot explain month-to-month movements.
#   NSDP for 2025-26 are state-government *estimates* (2026 simply repeats 2025); Tamil Nadu's 2025 estimate is below its 2024 value.
# * **Reverse causality**: more EVs also attract more chargers.  **Common trends**: petrol price, charging and
#   adoption all drift over time, so correlation can arise without a causal link.
# * **Air quality** shows a positive pooled association (Delhi) but a *negative* within-state one (winter smog vs
#   festive/fiscal-year registration peaks) - the sign is not a causal statement about pollution.
# print("Limits of explanation printed - see markdown cell above.")

