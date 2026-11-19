"""
05_final_forecast_and_scenarios.py -- RETRAIN ON ALL DATA and forecast the next six months (frozen on creation); what-if scenarios
================================================================================================================================
Origin = Sep-2026 (last month with verified registrations).  Same engine as the backtest (forecast_engine.py): hyper-parameters, SARIMA
specifications and all model fits are re-estimated on every observation through Sep-2026; covariates come from the release-aligned vintage
panel.  HEADLINE = the pre-declared top-3 ensemble (3 best exogenous-free models on the 2024 validation window: SARIMA, ETS, XGBoost-F0).  The equal-weight
pool of 7 exogenous-free models is reported alongside as an EXPLORATORY comparison.  Intervals: 10/90%
quantiles of log forecast errors already realised by the origin (stage 04 rolling-origin errors).
Three West Bengal data policies for the Nov-25..Feb-26 registration-drive months:
  as_reported      - data as supplied (a seasonal model replays the winter spike next winter: implausible)
  capped           - flagged months limited to the pre-episode 12-month maximum
  backlog_removed  - excess three-wheelers removed (stage 07): the PRINCIPLED default
Scenario layer (retrospective explanatory models): associations only, see the cautions printed at the end.
The forecast file is frozen ONCE per run date (data/frozen_forecasts_made_<date>.csv, never overwritten); earlier frozen files are kept.
Run from the project root after stages 00a-04:  python 05_final_forecast_and_scenarios.py
"""
from common import *
from forecast_engine import HEADLINE, POOL, run_origin, interval_quantiles, apply_policy
from vintage import backlog_adjusted_rate
import joblib
from sklearn.base import clone

dfa = load_data(complete=False)                                    # registrations through Sep-2026
vp = pd.read_csv(BASE_DIR / "data" / "ev_panel_vintage.csv", parse_dates=["date"]).sort_values(["state", "date"])
protocol = json.loads((TAB_DIR / "02_selected_features_and_protocol.json").read_text())
EXOG = protocol["selected_drivers"]
EXOG_F2 = EXOG
N_A = int(dfa.t.max() + 1); O = N_A - 1
Y_RAW = {s: dfa[dfa.state == s].sort_values("t")[TARGET].to_numpy() for s in STATES}
EX = {s: vp[vp.state == s].reset_index(drop=True)[EXOG_F2] for s in STATES}
future_dates = pd.date_range(dfa.date.max() + pd.offsets.MonthBegin(1), periods=6, freq="MS")
DATES = list(dfa[dfa.state == STATES[0]].sort_values("t").date) + list(future_dates)
adj, base3w = backlog_adjusted_rate(dfa)
print(f"Retrain on ALL data through {dfa.date.max():%b-%Y} (t={O}); forecast {future_dates[0]:%b-%Y}..{future_dates[-1]:%b-%Y} | headline: {HEADLINE}")
q = interval_quantiles(TAB_DIR / "04_rolling_origin_predictions.csv", O)
print("interval calibration errors per horizon:", q.n_errors.to_dict())

frames = []
for policy in ["as_reported", "capped", "backlog_removed"]:
    print(f"\n--- policy {policy}", flush=True)
    f = run_origin(O, apply_policy(dfa, Y_RAW, policy, adj), EX, {"F0": []}, DATES)
    wide = f.pivot_table(index=["state", "h", "date"], columns="model", values="y_pred").reset_index()
    wide["Headline"] = wide[HEADLINE]; wide["Pool (equal weights)"] = wide[POOL]
    wide = wide.merge(q[["q10", "q90"]], left_on="h", right_index=True)
    wide["lo80"], wide["hi80"] = wide.Headline * np.exp(wide.q10), wide.Headline * np.exp(wide.q90)
    wide["policy"] = policy; frames.append(wide)
fc = pd.concat(frames, ignore_index=True)
pct = [c for c in fc.columns if c not in ("state", "h", "date", "q10", "q90", "policy")]
fc_out = fc.copy(); fc_out[pct] = (100 * fc_out[pct]).round(2); fc_out = fc_out.drop(columns=["q10", "q90"])
save_table(fc_out, "05_forecast_next6m", index=False)
frozen_name = f"frozen_forecasts_made_{pd.Timestamp.today():%Y-%m-%d}.csv"
_already = [p_ for p_ in (BASE_DIR / "data").glob("frozen_forecasts_made_*.csv")
            if {"data_cutoff", "headline_model"} <= set(pd.read_csv(p_, nrows=1).columns)
            and pd.read_csv(p_, nrows=1).data_cutoff.iloc[0] == f"{DATA_CUTOFF:%Y-%m}" and pd.read_csv(p_, nrows=1).headline_model.iloc[0] == HEADLINE]
if not _already:                                                      # freeze ONCE per (data cut-off, headline definition); never overwrite
    fc_out[["state", "h", "date", "policy", "Headline", "Pool (equal weights)", "lo80", "hi80", "SARIMA", "ETS", "XGBoost-F0"]].assign(
        data_cutoff=f"{DATA_CUTOFF:%Y-%m}", headline_model=HEADLINE).to_csv(BASE_DIR / "data" / frozen_name, index=False)
    print("  [frozen]", frozen_name)
else:
    print("  [frozen file for this data cut-off and headline already exists - not overwritten]:", _already[0].name)

main = fc_out[fc_out.policy == "backlog_removed"]
print("\nHeadline (pre-declared top-3) forecast, EV share % (policy = backlog_removed):")
print(main.pivot(index="date", columns="state", values="Headline")[STATES].to_string())
print("\n80% intervals:\n", main[["state", "date", "lo80", "Headline", "hi80"]].to_string(index=False))
print("\nWest Bengal headline forecast by policy (EV share %):")
print(pd.DataFrame({p: fc_out[(fc_out.policy == p) & (fc_out.state == ANOMALY_STATE)].set_index("date").Headline for p in ["as_reported", "capped", "backlog_removed"]}).to_string())
prev = dfa[dfa.date.isin(future_dates - pd.DateOffset(years=1))].groupby("state")[TARGET].mean() * 100
last6 = dfa[dfa.date > dfa.date.max() - pd.DateOffset(months=6)].groupby("state")[TARGET].mean() * 100
summ = main.groupby("state").Headline.mean().to_frame("forecast_next6m_mean_%").join(prev.rename("same_months_last_year_%")).join(last6.rename("last6m_actual_mean_%")).loc[STATES]
summ["change_vs_last_year_pp"] = summ.iloc[:, 0] - summ.iloc[:, 1]; summ["change_vs_last6m_pp"] = summ.iloc[:, 0] - summ.iloc[:, 2]
print("\nNext-6-month average vs the same months a year earlier and vs the last 6 observed months (WB year-earlier base contains the drive):"); print(summ.round(2).to_string())
save_table(summ.round(3), "05_next6m_vs_year_earlier")

set_style()
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True)
for ax, s in zip(axes, STATES):
    g = dfa[(dfa.state == s) & (dfa.t >= 36)]
    ax.plot(g.date, 100 * g[TARGET], color=INK, lw=1.5, label="actual")
    for pol, col, ls in [("backlog_removed", "#1baf7a", "-"), ("as_reported", "#eb6834", "--")]:
        d = fc_out[(fc_out.policy == pol) & (fc_out.state == s)].sort_values("h")
        if pol == "backlog_removed":
            ax.fill_between(d.date, d.lo80, d.hi80, color=col, alpha=.2, label="80% interval")
        ax.plot(d.date, d.Headline, color=col, ls=ls, lw=2, marker="o", ms=3, label=f"forecast ({pol})" if s == ANOMALY_STATE else None)
    ax.set_title(s, fontsize=10); ax.tick_params(axis="x", labelrotation=45, labelsize=7)
axes[0].set_ylabel("EV share (%)"); axes[-1].legend(fontsize=7)
fig.suptitle(f"Retrained on all data: headline forecast {future_dates[0]:%b-%Y} .. {future_dates[-1]:%b-%Y} (pre-declared top-3 ensemble)", x=0.01, y=1.04, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout(); save_fig(fig, "fig19b_forecast_next6m")

# scenarios use the complete-covariate panel (through Aug-2026)
df = load_data(); N_T = int(df.t.max() + 1)
# ----------------------------------------------------------------------------------------------- scenarios (retrospective)
print("\n" + "=" * 100 + "\nWHAT-IF SCENARIOS - retrospective explanatory models, associations only\n" + "=" * 100)
expl = joblib.load(BASE_DIR / "outputs" / "models" / "explanatory_models_tuned.joblib")
FEATURES = EXOG + protocol["calendar_features"]
fit_all = {n: clone(m).fit(df[FEATURES], df[TARGET]) for n, m in expl.items()}
base_rows = df[(df.t > N_T - 13) & (df.anomaly_flag == 0)].copy()

SCENARIOS = {"Charging stations +50%": {"charging_stations": 1.5},
             "Petrol price +10%": {"petrol_price": 1.10},
             "Both": {"charging_stations": 1.5, "petrol_price": 1.10}}
out = []
for sc, mult in SCENARIOS.items():
    shifted = base_rows.copy()
    for c, f in mult.items():
        shifted[c] = shifted[c] * f
    for n, m in fit_all.items():
        delta = 100 * (m.predict(shifted[FEATURES]) - m.predict(base_rows[FEATURES]))
        by = pd.Series(delta, index=base_rows.index).groupby(base_rows.state).mean()
        for s in STATES:
            out.append({"scenario": sc, "model": n, "state": s, "delta_pp": by[s]})
sc_tab = pd.DataFrame(out)
sc_piv = sc_tab.pivot_table(index=["scenario", "state"], columns="model", values="delta_pp")
sc_piv["min"] = sc_piv.min(axis=1); sc_piv["max"] = sc_piv[list(expl)].max(axis=1)
sc_piv["models_agree_on_sign"] = np.sign(sc_piv[list(expl)]).nunique(axis=1) == 1
print(sc_piv.round(2).to_string())
save_table(sc_piv.round(3), "05_scenario_analysis")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
mcol = {"Ridge": "#1baf7a", "RandomForest": "#2a78d6", "XGBoost": "#4a3aa7", "LightGBM": "#eb6834"}
for ax, sc in zip(axes, SCENARIOS):
    d = sc_tab[sc_tab.scenario == sc]
    for i, n in enumerate(expl):
        v = d[d.model == n].set_index("state").loc[STATES, "delta_pp"]
        ax.bar(np.arange(4) + (i - 1.5) * 0.2, v, 0.18, color=mcol[n], label=n)
    ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(range(4), [s.replace(" ", "\n") for s in STATES], fontsize=8)
    ax.set_title(sc, fontsize=10)
axes[0].set_ylabel("change in EV share (pp)"); axes[0].legend(fontsize=7.5, ncol=2)
fig.suptitle("What-if sensitivity - model disagreement is the message", x=0.01, y=1.04, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout(); save_fig(fig, "fig20_scenarios")

agree_share = float(sc_piv.models_agree_on_sign.mean())
print(f"\nShare of scenario cells where all four models agree on the sign: {100 * agree_share:.0f}%")
print("Treat any scenario number as a model-implied association, not as the causal effect of a policy.")
petrol_delta = sc_tab[(sc_tab.scenario == "Petrol price +10%")].delta_pp
if (petrol_delta < 0).all():
    print("CAUTION 1: a +10% petrol price LOWERS predicted EV share in every state-model cell - the OPPOSITE of the economic prior"
          "\n  (dearer petrol should favour EVs). The model has learned a trend/composition effect, not a price elasticity.")
elif (petrol_delta > 0).mean() > 0.75:
    print("CAUTION 1: a +10% petrol price RAISES predicted EV share in most cells, as theory would suggest - but the identifying variation"
          "\n  is mostly ONE fuel-price step in May-Jun 2026 that coincides with a general upswing, so treat the size as indicative only.")
else:
    print("CAUTION 1: the models disagree on the sign of a petrol-price change - the data do not identify a fuel-price effect.")
print("CAUTION 2: charging +50% has near-zero effect in trees (flat between observed step values) and a different sign in some models"
      "\n  - the data contain too few distinct charging values (3-4 per state) to support a what-if on infrastructure.")
