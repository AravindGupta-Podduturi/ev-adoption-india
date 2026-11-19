"""
02_statistical_diagnostics.py -- Stage 02: statistical diagnostics, leakage audit and feature selection
Run from the project root:  python 02_statistical_diagnostics.py
"""
from common import *

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm

df = load_data()

# ## 1. Leakage audit

roles = []
for c in df.columns:
    if c == TARGET:
        role, why = "TARGET", "EV share of new registrations in the state-month"
    elif c in LEAKY_COLS:
        role, why = "EXCLUDED (leakage)", LEAKY_COLS[c]
    elif c in CANDIDATE_DRIVERS:
        role, why = "candidate driver", COL_TO_GROUP[c]
    elif c in ("state", "date", "year", "month", "month_name", "quarter", "season", "month_index", "t",
               "month_sin", "month_cos", "fy_end", "anomaly_flag"):
        role, why = "id / calendar", "known in advance"
    elif c == "dominant_pollutant":
        role, why = "EXCLUDED", "categorical label derived from the pollutants already used"
    else:
        role, why = "unclassified", ""
    roles.append({"column": c, "role": role, "note": why})
roles = pd.DataFrame(roles)
print(roles.role.value_counts().to_string())
save_table(roles, "02_leakage_and_role_audit", index=False)

# ## 2. Between- vs within-state variance of every driver
# A driver that is (almost) constant *within* each state can only be learned from the 4 state-level
# differences - i.e. effectively 4 data points. This sets the limit of what "driver importance" can mean.

def variance_split(col):
    tot = df[col].var(ddof=0)
    between = df.groupby("state")[col].transform("mean").var(ddof=0)
    return between / tot

vs = pd.DataFrame({c: {"between_state_share_%": 100 * variance_split(c)} for c in [TARGET] + CANDIDATE_DRIVERS}).T
vs["within_state_share_%"] = 100 - vs["between_state_share_%"]
vs["distinct_values_per_state"] = [df.groupby("state")[c].nunique().median() for c in vs.index]
vs = vs.sort_values("between_state_share_%", ascending=False)
print(vs.round(1).to_string())
save_table(vs.round(2), "02_between_within_variance")

# ## 3. Correlation structure and multicollinearity

set_style()
corr_cols = [TARGET] + CANDIDATE_DRIVERS
pooled = df[corr_cols].corr(method="spearman")[TARGET].drop(TARGET)
demeaned = df[corr_cols] - df.groupby("state")[corr_cols].transform("mean")   # remove state effects
within = demeaned.corr(method="spearman")[TARGET].drop(TARGET)
cmp = pd.DataFrame({"pooled_spearman": pooled, "within_state_spearman": within}).loc[CANDIDATE_DRIVERS]
print(cmp.round(2).to_string())
save_table(cmp.round(3), "02_pooled_vs_within_correlation")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.05, 1]})
cm = df[corr_cols].corr(method="spearman")
im = axes[0].imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
axes[0].set_xticks(range(len(cm)), cm.columns, rotation=90); axes[0].set_yticks(range(len(cm)), cm.columns)
for i in range(len(cm)):
    for j in range(len(cm)):
        axes[0].text(j, i, f"{cm.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6.3,
                     color="white" if abs(cm.iloc[i, j]) > 0.6 else INK)
axes[0].grid(False); axes[0].set_title("Spearman correlation matrix (pooled)")
fig.colorbar(im, ax=axes[0], shrink=0.7)
yy = np.arange(len(cmp))
axes[1].barh(yy + 0.2, cmp.pooled_spearman, 0.38, color="#2a78d6", label="pooled (mostly between-state)")
axes[1].barh(yy - 0.2, cmp.within_state_spearman, 0.38, color="#eb6834", label="within-state (state means removed)")
axes[1].set_yticks(yy, cmp.index); axes[1].invert_yaxis(); axes[1].axvline(0, color=MUTED, lw=0.8)
axes[1].set_title("Correlation with EV share: pooled vs within-state"); axes[1].legend(loc="lower right", fontsize=7.5)
fig.tight_layout()
save_fig(fig, "fig07_correlation_pooled_vs_within")

# ### Variance Inflation Factor (VIF) and iterative pruning
# AQI, PM2.5, PM10 and AQI-max are near-duplicates; petrol/diesel move together. Features are removed one at a
# time (highest VIF first) until all VIF < 10, **but** a priority order protects the business-critical drivers.

PROTECT = ["charging_stations", "petrol_price", "nsdp_per_capita", "aqi_mean"]   # keep these if at all possible


def vif_table(X: pd.DataFrame) -> pd.Series:
    Xc = sm.add_constant((X - X.mean()) / X.std())
    return pd.Series([variance_inflation_factor(Xc.values, i + 1) for i in range(X.shape[1])], index=X.columns)


MANUAL_DROP = {"aqi_max": "within-month extreme of the same AQI index; adds noise, not information"}
kept = [c for c in CANDIDATE_DRIVERS if c not in MANUAL_DROP]
history = []
while True:
    v = vif_table(df[kept])
    if v.max() < 10:
        break
    cand = v.drop(PROTECT, errors="ignore").sort_values(ascending=False)
    drop = cand.index[0]
    history.append({"dropped": drop, "its_VIF": round(v[drop], 1), "max_VIF_before": round(v.max(), 1)})
    kept.remove(drop)
final_vif = vif_table(df[kept]).round(2)
print("Manually dropped:", MANUAL_DROP)
print("Pruning history:"); print(pd.DataFrame(history).to_string(index=False))
print("\nFinal driver set and VIF:"); print(final_vif.to_string())
save_table(pd.DataFrame(history), "02_vif_pruning_history", index=False)
save_table(final_vif.to_frame("VIF"), "02_vif_final")

# ## 4. Time-series diagnostics of the target

rows = []
for s, g in df.groupby("state"):
    y = g.set_index("date")[TARGET]
    ly = np.log(y)
    adf_p = adfuller(ly, regression="ct", autolag="AIC")[1]
    adf_d = adfuller(ly.diff().dropna(), regression="c", autolag="AIC")[1]
    kpss_p = kpss(ly, regression="ct", nlags="auto")[1]
    stl = STL(ly, period=12, robust=True).fit()
    Fs = max(0, 1 - stl.resid.var() / (stl.resid + stl.seasonal).var())        # Hyndman seasonal strength
    Ft = max(0, 1 - stl.resid.var() / (stl.resid + stl.trend).var())          # trend strength
    rows.append({"state": s, "ADF_p (log rate, trend)": adf_p, "KPSS_p (log rate, trend)": kpss_p,
                 "ADF_p (diff log rate)": adf_d, "seasonal_strength": Fs, "trend_strength": Ft})
ts_diag = pd.DataFrame(rows).set_index("state").round(3)
print(ts_diag.to_string())
print("\nReading: ADF rejects a unit root once a deterministic trend is allowed (p<0.05) and KPSS does not reject trend-stationarity"
      "\nfor most states -> each series is a steadily rising trend plus seasonality. With only ~56 points per state both tests have low"
      "\npower, so the differencing order of SARIMA is chosen by AIC in stage 04 rather than fixed here.")
save_table(ts_diag, "02_timeseries_diagnostics")

fig, axes = plt.subplots(2, 4, figsize=(13, 5.2), sharey="row")
for j, s in enumerate(STATES):
    y = np.log(df[df.state == s].set_index("date")[TARGET]).diff().dropna()      # growth-rate series
    a = acf(y, nlags=18); p = pacf(y, nlags=18)
    conf = 1.96 / np.sqrt(len(y))
    for ax, vals, name in ((axes[0, j], a, "ACF"), (axes[1, j], p, "PACF")):
        ax.bar(range(len(vals)), vals, color=STATE_COLORS[s], width=0.6)
        ax.axhline(conf, color=MUTED, ls="--", lw=0.8); ax.axhline(-conf, color=MUTED, ls="--", lw=0.8)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(f"{s} - {name} of monthly log-growth" if name == "ACF" else f"{name}", fontsize=9)
    axes[1, j].set_xlabel("lag (months)")
fig.tight_layout()
save_fig(fig, "fig08_acf_pacf")

# STL decomposition of the 4-state pooled EV share
pooled_rate = df.groupby("date").apply(lambda g: g.ev_count.sum() / g.vehicle_total.sum())
stl = STL(np.log(pooled_rate), period=12, robust=True).fit()
fig, axes = plt.subplots(4, 1, figsize=(9, 6.5), sharex=True)
for ax, (nm, ser) in zip(axes, [("log EV share (pooled 4 states)", np.log(pooled_rate)), ("trend", stl.trend),
                                 ("seasonal", stl.seasonal), ("remainder", stl.resid)]):
    ax.plot(ser.index, ser.values, color="#2a78d6" if nm != "remainder" else "#eb6834", lw=1.5)
    ax.set_title(nm, fontsize=9)
fig.tight_layout()
save_fig(fig, "fig09_stl_decomposition")

seas_idx = np.exp(stl.seasonal.groupby(stl.seasonal.index.month).mean())
print("Multiplicative seasonal index by calendar month (1 = average):")
print(seas_idx.round(2).to_string())
save_table(seas_idx.rename("seasonal_index").to_frame(), "02_pooled_seasonal_index")

# ## 5. Final modelling set-up (documented decisions)

SELECTED_DRIVERS = kept
protocol = {
    "target": TARGET,
    "selected_drivers": SELECTED_DRIVERS,
    "calendar_features": ["month_sin", "month_cos", "fy_end"],
    "excluded_for_leakage": sorted(LEAKY_COLS),
    "data_cutoff": f"{COVARIATE_CUTOFF:%Y-%m}",
    "tuning_period": f"{df.date.min():%Y-%m} .. {TRAIN_END:%Y-%m} (hyper-parameters are tuned here and then frozen)",
    "out_of_sample_period": f"{TEST_START:%Y-%m} .. {COVARIATE_CUTOFF:%Y-%m} ({int(df.date.between(TEST_START, COVARIATE_CUTOFF).sum() / 4)} months x 4 states)",
    "flagged_observations": f"{ANOMALY_STATE} {ANOMALY_START:%Y-%m} .. {ANOMALY_END:%Y-%m} surge episode (metrics reported with and without)",
    "cv_scheme": "expanding-window time-series CV inside the training period only (no shuffling)",
    "random_seed": SEED,
}
save_json(protocol, "02_selected_features_and_protocol")
print(json.dumps(protocol, indent=2))

