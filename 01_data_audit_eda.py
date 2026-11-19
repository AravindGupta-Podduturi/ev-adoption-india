"""
01_data_audit_eda.py -- Stage 01: data audit and exploratory data analysis (retrospective, same-month data)
Run from the project root:  python 01_data_audit_eda.py
"""
from common import *

import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL

df = load_data()
print(f"Panel used: {df.shape[0]} rows x {df.shape[1]} cols | states={df.state.nunique()} | "
      f"{df.date.min():%b %Y} -> {df.date.max():%b %Y}  (COVARIATE_CUTOFF = {COVARIATE_CUTOFF:%b %Y}; registrations run to {DATA_CUTOFF:%b %Y})")

# ## 0. Screening of the raw file - which trailing months can be trusted?
# A month is excluded when (a) its total registrations are < 40% of the trailing 6-month median
# (partial month) or (b) the environmental columns are *identical in every state* (placeholder fill).

raw = load_data(apply_cutoff=False)
tot = raw.groupby("date").vehicle_total.sum()
ratio = (tot / tot.shift(1).rolling(6).median()).rename("registrations_vs_trailing_median")
ENV_COLS = ["aqi_mean", "pm25_mean", "pm10_mean", "no2_mean", "so2_mean", "aqi_max", "mean_temp_c", "precipitation_mm"]
screen = pd.DataFrame({"registrations_vs_trailing_median": ratio,
                       "env_columns_identical_across_states": [sum(g[c].nunique() == 1 for c in ENV_COLS)
                                                               for _, g in raw.groupby("date")]}, index=tot.index)
screen["dominant_pollutant_unknown"] = raw.groupby("date").dominant_pollutant.apply(lambda x: (x == "Unknown").all()).values
screen["verdict"] = np.where((screen.registrations_vs_trailing_median < 0.4) | (screen.env_columns_identical_across_states >= 4)
                             | screen.dominant_pollutant_unknown, "EXCLUDE", "ok")
print(screen.tail(6).round(2).to_string())
bad = screen[screen.verdict == "EXCLUDE"].index
print(f"\nExcluded months: {[d.strftime('%b-%Y') for d in bad]}  -> analysis cut-off = {DATA_CUTOFF:%b %Y}")
assert all(d > DATA_CUTOFF for d in bad), "an excluded month lies inside the analysis window - revisit DATA_CUTOFF"
print(f"Registrations used through {DATA_CUTOFF:%b-%Y}; same-month explanatory stages use complete covariates only, through {COVARIATE_CUTOFF:%b-%Y} (later covariates are NaN).")
save_table(screen.round(3), "01_trailing_month_screen")

# ## 1. Structural audit - completeness, balance, internal consistency

FUEL = ["petrol", "diesel", "gas", "electric", "hybrid", "lpg", "alternative", "not_applicable"]
CATS = ["two_wheeler", "three_wheeler", "four_wheeler", "light_vehicle", "medium_vehicle", "heavy_vehicle", "other"]

expected_months = pd.date_range(df.date.min(), df.date.max(), freq="MS")
checks = {
    "Missing values (total)": int(df.isna().sum().sum()),
    "Duplicate state-month rows": int(df.duplicated(["state", "date"]).sum()),
    f"Panel balanced ({len(expected_months)} months x 4 states)": bool(
        (df.groupby("state").date.nunique() == len(expected_months)).all()),
    "No gaps in monthly calendar": bool(
        all((g.date.values == expected_months.values).all() for _, g in df.groupby("state"))),
    "ev_count == electric (all rows)": bool((df.ev_count == df.electric).all()),
    "ev_adoption_rate == ev_count / vehicle_total (max abs err)": float(
        (df.ev_count / df.vehicle_total - df[TARGET]).abs().max()),
    "Fuel-type columns sum to vehicle_total": bool((df[FUEL].sum(axis=1) == df.vehicle_total).all()),
    "Vehicle-category columns sum to vehicle_total": bool((df[CATS].sum(axis=1) == df.vehicle_total).all()),
    "Target within (0,1)": bool(df[TARGET].between(0, 1, inclusive="neither").all()),
    "Rows with dominant_pollutant == 'Insufficient data'": int((df.dominant_pollutant == "Insufficient data").sum()),
    "'four_wheeler' max monthly count (looks like a mapping artefact)": int(df.four_wheeler.max()),
}
audit = pd.Series(checks, name="result").to_frame()
display(audit) if "display" in globals() else print(audit.to_string())
save_table(audit, "01_integrity_checks")

# ## 2. Granularity audit - how often does each driver actually change?
# Charging stations and per-capita income are *low-frequency snapshots* repeated across months.
# This limits what any model can learn about them (see stage 03 caveats).

rows = []
for col in ["petrol_price", "diesel_price", "charging_stations", "nsdp_per_capita", "aqi_mean", "mean_temp_c"]:
    for s, g in df.groupby("state"):
        ch = g.loc[g[col].diff().fillna(0) != 0, "date"]
        rows.append({"variable": col, "state": s, "distinct_values": g[col].nunique(),
                     "n_changes": int(len(ch)),
                     "change_dates": ", ".join(d.strftime("%b-%y") for d in ch[:6])
                     if col in ("charging_stations", "nsdp_per_capita") else ""})
gran = pd.DataFrame(rows)
piv = gran.pivot(index="variable", columns="state", values="distinct_values")[STATES]
print(f"Distinct values per state ({df.t.max() + 1} months):")
print(piv.to_string())
print("\nCharging-station / income change dates:")
print(gran[gran.variable.isin(["charging_stations", "nsdp_per_capita"])]
      [["variable", "state", "change_dates"]].to_string(index=False))
save_table(gran, "01_granularity_audit", index=False)

# Per-capita income: annual change by state.  2026 values are state-government ESTIMATES (actuals not yet released).
nsdp = df.groupby(["state", "year"]).nsdp_per_capita.first().unstack("state")[STATES]
nsdp_chg = (nsdp.pct_change() * 100).round(1)
print("\nNSDP per capita, annual % change (latest years are state-government estimates):")
print(nsdp_chg.to_string())
odd = [(s_, int(y_)) for y_ in nsdp_chg.index for s_ in STATES if pd.notna(nsdp_chg.loc[y_, s_]) and (nsdp_chg.loc[y_, s_] < 0 or nsdp_chg.loc[y_, s_] > 15)]
print("Check basis / source (decline or >15% jump):", odd)
carried = [int(y_) for y_ in nsdp_chg.index if (nsdp_chg.loc[y_].fillna(1) == 0).all()]
print("Years whose income value is simply carried forward from the previous year (0% change in all states):", carried)
save_table(nsdp_chg, "01_nsdp_annual_change")

# ## 3. Outlier / anomaly screen
# Seasonal spikes (March, festive season) are real and recurring, so a plain z-score would flag them.
# Instead each month is compared with (a) the highest value of the *previous 12 months* and
# (b) the mean of the previous 3 months.  A month is "suspect" when it exceeds BOTH by a wide margin.

rows = []
for s, g in df.groupby("state"):
    g = g.set_index("date")[TARGET]
    prior_max12 = g.shift(1).rolling(12).max()
    prior_mean3 = g.shift(1).rolling(3).mean()
    r = pd.DataFrame({"state": s, "rate_%": 100 * g, "x_prior12_max": g / prior_max12, "x_prior3_mean": g / prior_mean3}).dropna()
    rows.append(r.reset_index())
scr = pd.concat(rows)
scr["suspect"] = (scr.x_prior12_max > 1.8) & (scr.x_prior3_mean > 2.0)
print("Top-8 months by exceedance of the previous-12-month maximum:")
print(scr.sort_values("x_prior12_max", ascending=False).head(8).round(2).to_string(index=False))
print("\nMonths flagged suspect:", scr.loc[scr.suspect, ["state", "date"]].astype(str).values.tolist())
flag_df = scr.sort_values("x_prior12_max", ascending=False).head(15)
save_table(flag_df, "01_anomaly_screen", index=False)

wb = df[(df.state == ANOMALY_STATE) & (df.date >= "2025-08-01")][["date", "vehicle_total", "ev_count", TARGET, "charging_stations"]]
print("\nWest Bengal, last months:\n", wb.round(4).to_string(index=False))
print(f"\n>> West Bengal {ANOMALY_START:%b-%Y}..{ANOMALY_END:%b-%Y} is flagged (anomaly_flag=1): a surge that decays over ~3 months, so\n   probably a real event rather than a typo - but verify against the raw Vahan extract.")

# ## 4. Descriptive statistics

desc_cols = [TARGET, "ev_count", "vehicle_total"] + CANDIDATE_DRIVERS
desc = df.groupby("state")[desc_cols].agg(["mean", "std", "min", "max"]).T
desc.index.names = ["variable", "stat"]
save_table(desc.round(3), "01_descriptives_by_state")

yearly = (df.groupby(["state", "year"]).agg(ev_count=("ev_count", "sum"), total=("vehicle_total", "sum"), months=("month", "nunique"))
          .assign(rate_pct=lambda x: 100 * x.ev_count / x.total).round(2))
yearly["ev_yoy_growth_%"] = (yearly.groupby("state").ev_count.pct_change() * 100).round(1)
yearly.loc[yearly.months < 12, "ev_yoy_growth_%"] = np.nan          # partial year: not comparable with a full year
print(yearly.to_string())
save_table(yearly, "01_yearly_adoption_summary")

# like-for-like growth of the latest (partial) year: same months of the previous year
LAST_Y = int(df.year.max()); LAST_M = int(df[df.year == LAST_Y].month.max())
ytd = (df[df.month <= LAST_M].groupby(["state", "year"]).agg(ev=("ev_count", "sum"), tot=("vehicle_total", "sum")).reset_index())
ytd["rate_%"] = 100 * ytd.ev / ytd.tot
ytd_cmp = ytd[ytd.year.isin([LAST_Y - 1, LAST_Y])].pivot(index="state", columns="year", values=["ev", "rate_%"]).loc[STATES]
ytd_cmp[("ev_growth_%", LAST_Y)] = 100 * (ytd_cmp[("ev", LAST_Y)] / ytd_cmp[("ev", LAST_Y - 1)] - 1)
print(f"\nJan-{pd.Timestamp(2000, LAST_M, 1):%b} {LAST_Y} vs same months of {LAST_Y - 1}:")
print(ytd_cmp.round(1).to_string())
save_table(ytd_cmp.round(2), "01_ytd_comparison")

# ## 5. Figures
# ### Fig 1 - Adoption trajectories (RQ1)

set_style()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
W = wide(df)
for s in STATES:
    ax.plot(W.index, 100 * W[s], color=STATE_COLORS[s], label=s)
# end-of-line labels, nudged apart so that close values do not overlap
ends = sorted(((100 * W[s].iloc[-1], s) for s in STATES))
pos, last = {}, -99
for v, s in ends:
    pos[s] = max(v, last + 1.1); last = pos[s]
for s in STATES:
    ax.text(W.index[-1] + pd.Timedelta(days=12), pos[s], s, color=INK2, va="center", fontsize=8.5)
ax.axvspan(TEST_START, COVARIATE_CUTOFF + pd.offsets.MonthEnd(0), color="#898781", alpha=0.10, lw=0)
ax.text(TEST_START + pd.Timedelta(days=8), 0.4, f"out-of-sample window ({TEST_START:%b-%y} to {COVARIATE_CUTOFF:%b-%y})", color=INK2, fontsize=8, va="bottom")
ax.annotate("West Bengal drive\nNov-25 to Feb-26", xy=(pd.Timestamp("2025-12-01"), 23.6),
            xytext=(pd.Timestamp("2024-07-01"), 21.5), fontsize=8, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED))
ax.set_title("EV share of new registrations, by state")
ax.set_ylabel("EV adoption rate (%)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
ax.set_xlim(W.index[0], W.index[-1] + pd.Timedelta(days=150))
ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.0, 0.86))

ax = axes[1]
y = yearly.reset_index().pivot(index="year", columns="state", values="rate_pct")[STATES]
x = np.arange(len(y.index)); w = 0.2
for i, s in enumerate(STATES):
    ax.bar(x + (i - 1.5) * w, y[s], w * 0.88, color=STATE_COLORS[s], label=s)
ax.set_xticks(x, [f"{i}{' YTD' if yearly.loc[(STATES[0], i), 'months'] < 12 else ''}" for i in y.index])
ax.set_title("Annual EV share (volume-weighted)"); ax.set_ylabel("%"); ax.tick_params(axis="x", labelsize=7.5)
ax.legend(ncol=2, loc="upper left")
fig.tight_layout()
save_fig(fig, "fig01_adoption_trajectories")

# ### Fig 2 - Seasonality: the registration calendar matters (festive season, fiscal year-end)

fig, axes = plt.subplots(1, 4, figsize=(13, 3.0), sharey=True)
for ax, s in zip(axes, STATES):
    g = df[(df.state == s) & (df.year < LAST_Y)]          # complete calendar years only (a partial year would bias the index)
    # detrend by dividing each month by its calendar-year mean -> seasonal index
    idx = g[TARGET] / g.groupby("year")[TARGET].transform("mean")
    h = pd.DataFrame({"year": g.year, "month": g.month, "idx": idx}).pivot(index="year", columns="month", values="idx")
    im = ax.imshow(h.values, cmap="Blues", vmin=0.4, vmax=2.2, aspect="auto")
    ax.set_xticks(range(12), list("JFMAMJJASOND")); ax.set_yticks(range(len(h.index)), h.index)
    ax.set_title(s, fontsize=10); ax.grid(False)
cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01); cbar.set_label("EV share / that year's mean")
fig.suptitle("Seasonal index of EV share (each month relative to its own year)", x=0.01, y=1.06, ha="left", fontweight="bold", fontsize=11)
save_fig(fig, "fig02_seasonality_heatmap")

# ### Fig 3 - Fuel-type mix and vehicle-category mix (what is being electrified?)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fuel_share = df.groupby("year")[FUEL].sum()
fuel_share = fuel_share.div(fuel_share.sum(axis=1), axis=0) * 100
YEAR_LABEL = {y_: f"{y_} YTD" if y_ == LAST_Y and LAST_M < 12 else str(y_) for y_ in fuel_share.index}
order = ["petrol", "diesel", "gas", "electric", "hybrid"]
cols = ["#898781", "#52514e", "#eda100", "#2a78d6", "#1baf7a"]
fuel_share = fuel_share.assign(other_fuels=100 - fuel_share[order].sum(axis=1))
bottom = np.zeros(len(fuel_share))
for c, col in zip(order + ["other_fuels"], cols + ["#c3c2b7"]):
    axes[0].bar([YEAR_LABEL[i] for i in fuel_share.index], fuel_share[c], bottom=bottom, color=col, label=c, edgecolor=SURFACE, linewidth=1.5)
    bottom += fuel_share[c].values
axes[0].set_title("Fuel mix of new registrations (4-state total)"); axes[0].set_ylabel("%")
axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
cat_share = df.groupby("year")[CATS].sum(); cat_share = cat_share.div(cat_share.sum(axis=1), axis=0) * 100
cat_share[["two_wheeler", "light_vehicle", "three_wheeler", "heavy_vehicle"]].plot(
    ax=axes[1], color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], marker="o")
axes[1].set_title("Vehicle-category mix of new registrations"); axes[1].set_ylabel("%"); axes[1].set_xlabel("")
axes[1].set_xticks(cat_share.index, [YEAR_LABEL[i] for i in cat_share.index], fontsize=8)
fig.tight_layout()
save_fig(fig, "fig03_fuel_and_category_mix")

# ### Fig 4 - Candidate drivers over time (are they aligned in state x month?)

panel = [("aqi_mean", "AQI (monthly mean)"), ("petrol_price", "Petrol price (INR/l)"),
         ("charging_stations", "Charging stations (count)"), ("mean_temp_c", "Mean temperature (C)"),
         ("nsdp_per_capita", "NSDP per capita (INR)"), ("precipitation_mm", "Precipitation (mm)")]
fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), sharex=True)
for ax, (c, t) in zip(axes.ravel(), panel):
    Wc = wide(df, c)
    for s in STATES:
        ax.plot(Wc.index, Wc[s], color=STATE_COLORS[s], lw=1.5, label=s,
                drawstyle="steps-post" if c in ("charging_stations", "nsdp_per_capita") else "default")
    ax.set_title(t, fontsize=10); ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[0, 0].legend(ncol=2, loc="upper left", fontsize=7.5)
fig.tight_layout()
save_fig(fig, "fig04_driver_timeseries")

# ### Fig 5 - Distribution of the target and raw bivariate relationships

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
bp = axes[0].boxplot([100 * df.loc[df.state == s, TARGET] for s in STATES], tick_labels=STATES, patch_artist=True,
                     medianprops=dict(color=INK), flierprops=dict(markersize=3, markeredgecolor=MUTED))
for patch, s in zip(bp["boxes"], STATES):
    patch.set_facecolor(STATE_COLORS[s]); patch.set_alpha(0.55)
axes[0].set_title("EV share by state"); axes[0].set_ylabel("%"); axes[0].tick_params(axis="x", labelsize=7.5)
for ax, (c, t) in zip(axes[1:], [("petrol_price", "Petrol price (INR/l)"), ("aqi_mean", "AQI (monthly mean)")]):
    for s in STATES:
        g = df[df.state == s]
        ax.scatter(g[c], 100 * g[TARGET], s=14, color=STATE_COLORS[s], alpha=0.8, edgecolor=SURFACE, linewidth=0.5, label=s)
    ax.set_xlabel(t); ax.set_ylabel("EV share (%)"); ax.set_title(f"EV share vs {t.split(' (')[0].lower()}")
axes[1].legend(fontsize=7.5, ncol=2)
fig.tight_layout()
save_fig(fig, "fig05_target_distribution_and_scatter")

# ### Fig 6 - EV volumes vs total registrations (denominator effects)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for s in STATES:
    g = df[df.state == s]
    axes[0].plot(g.date, g.ev_count / 1000, color=STATE_COLORS[s], label=s)
    axes[1].plot(g.date, g.vehicle_total / 1000, color=STATE_COLORS[s], label=s)
axes[0].set_title("EV registrations per month ('000)"); axes[1].set_title("All new registrations per month ('000)")
axes[0].legend(ncol=2)
for ax in axes: ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
fig.tight_layout()
save_fig(fig, "fig06_volumes")

# ## 6. Key EDA take-aways (printed so they can be pasted into the report)

def share(frame):
    return frame.groupby("state").apply(lambda g: 100 * g.ev_count.sum() / g.vehicle_total.sum())


s22, s25 = share(df[df.year == 2022]), share(df[df.year == 2025])
s26 = share(df[df.year == LAST_Y])
print("EV share (volume-weighted, %):")
print(pd.DataFrame({"2022": s22, "2025": s25, f"{LAST_Y} YTD": s26, "x 2022->2025": s25 / s22}).round(2).loc[STATES].to_string())
print("\nRanking by 2022 share:", ", ".join(s22.sort_values(ascending=False).index),
      "| by 2025:", ", ".join(s25.sort_values(ascending=False).index), f"| by {LAST_Y} YTD:", ", ".join(s26.sort_values(ascending=False).index))
print(f"Note: West Bengal's 2025 and {LAST_Y} figures include the Nov-25..Feb-26 registration-drive episode.")
print("Seasonal peaks recur around March (fiscal year-end) and Oct-Dec (festive season) - modelled explicitly in later stages.")

