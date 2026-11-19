"""
12_report_figures.py -- print-quality PDF figures for the IEEE/Overleaf report (writes report/figs/*.pdf)
Run from the project root after stages 00a-11:  python 12_report_figures.py
"""
from common import *
import matplotlib as mpl

OUT = BASE_DIR / "report" / "figs"; OUT.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "font.size": 7.5, "axes.titlesize": 8.5,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
                     "legend.fontsize": 6.8, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 1.3, "font.family": "DejaVu Sans"})
COL = {"Delhi": "#1f4e9c", "Maharashtra": "#d6641a", "Tamil Nadu": "#2e8b57", "West Bengal": "#8b3a8b"}
PRIMARY = "Ensemble-F0 top-3 (validation-selected)"
df = load_data(complete=False)

# ---- Fig 1: trajectories and seasonality
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.05), gridspec_kw={"width_ratios": [1.75, 1]})
for s in STATES:
    g = df[df.state == s]; ax[0].plot(g.date, 100 * g[TARGET], color=COL[s], label=s)
ax[0].axvspan(ANOMALY_START, ANOMALY_END + pd.offsets.MonthEnd(0), color="#f2c14e", alpha=.35, lw=0)
ax[0].axvspan(TEST_START, DATA_CUTOFF + pd.offsets.MonthEnd(0), color="#888888", alpha=.10, lw=0)
ax[0].text(TEST_START, 0.8, "test window", fontsize=6.2)
ax[0].set_ylabel("EV share of new registrations (%)"); ax[0].set_title("(a) Monthly EV adoption rate"); ax[0].legend(ncol=4, loc="upper left", frameon=False, fontsize=6.3)
ax[0].set_ylim(0, 30)
si = pd.read_csv(TAB_DIR / "02_pooled_seasonal_index.csv"); si.columns = ["month", "idx"]
ax[1].bar(si.month, si.idx, color=["#1f4e9c" if v > 1.15 else "#9fb4d4" for v in si.idx]); ax[1].axhline(1, color="k", lw=.6)
ax[1].set_xticks(range(1, 13), list("JFMAMJJASOND")); ax[1].set_ylim(0.6, 1.35); ax[1].set_title("(b) Pooled seasonal index (STL)")
fig.tight_layout(pad=0.4); fig.savefig(OUT / "fig1_eda.pdf"); plt.close(fig)

# ---- Fig 2: SHAP
gi = pd.read_csv(TAB_DIR / "03_shap_global_importance.csv", index_col=0).iloc[:, 0].sort_values()
gc = pd.read_csv(TAB_DIR / "03_state_group_contributions.csv", index_col=0)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.0), gridspec_kw={"width_ratios": [1, 1.15]})
lab = {"charging_stations": "Charging stations", "nsdp_per_capita": "GSDP per capita", "petrol_price": "Petrol price", "so2_mean": "SO2", "no2_mean": "NO2", "aqi_mean": "AQI",
       "mean_temp_c": "Temperature", "precipitation_mm": "Rainfall", "month_cos": "Month cos", "month_sin": "Month sin", "fy_end": "FY-end flag"}
ax[0].barh([lab[i] for i in gi.index], gi.values, color="#1f4e9c"); ax[0].set_xlabel("mean |SHAP| (pp of EV share)"); ax[0].set_title("(a) Global SHAP importance")
im = ax[1].imshow(gc.values, cmap="RdBu_r", vmin=-2.6, vmax=2.6, aspect="auto"); ax[1].grid(False)
ax[1].set_xticks(range(4), gc.columns, fontsize=6.5); ax[1].set_yticks(range(len(gc)), gc.index, fontsize=6.5)
for i in range(gc.shape[0]):
    for j in range(gc.shape[1]): ax[1].text(j, i, f"{gc.values[i, j]:+.1f}", ha="center", va="center", fontsize=6.3)
ax[1].set_title("(b) Contribution vs baseline (pp)")
fig.tight_layout(pad=0.4); fig.savefig(OUT / "fig2_shap.pdf"); plt.close(fig)

# ---- Fig 3: fixed-origin backtest
bt = pd.read_csv(TAB_DIR / "10_backtest_forecasts.csv", parse_dates=["date"]); bt = bt[bt.policy == "as_reported"]
fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.95), sharex=True)
for a, s in zip(ax, STATES):
    g = df[(df.state == s) & (df.date >= "2025-07-01")]; a.plot(g.date, 100 * g[TARGET], color="k", lw=1.2, label="actual")
    h = bt[(bt.model == PRIMARY) & (bt.state == s)].sort_values("h"); a.fill_between(h.date, 100 * h.lo80, 100 * h.hi80, color="#1f4e9c", alpha=.15, lw=0, label="80% interval")
    a.plot(h.date, 100 * h.y_pred, color="#1f4e9c", marker="o", ms=2.2, label="top-3 ensemble")
    n = bt[(bt.model == "Naive (last value)") & (bt.state == s)].sort_values("h"); a.plot(n.date, 100 * n.y_pred, color="#777", ls=":", label="naive")
    a.axvline(pd.Timestamp("2026-03-01"), color="#444", lw=.6, ls="--"); a.set_title(s, fontsize=7.5)
    a.tick_params(axis="x", labelrotation=60, labelsize=5.8)
    a.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonth=[7, 1, 4, 9])); a.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b-%y"))
h_, l_ = ax[0].get_legend_handles_labels(); fig.legend(h_, l_, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=6.5)
fig.tight_layout(pad=0.4, rect=(0, 0.07, 1, 1)); fig.savefig(OUT / "fig3_backtest.pdf"); plt.close(fig)

# ---- Fig 4: final forecast
fc = pd.read_csv(TAB_DIR / "05_forecast_next6m.csv", parse_dates=["date"])
fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.95), sharex=True)
for a, s in zip(ax, STATES):
    g = df[(df.state == s) & (df.date >= "2025-10-01")]; a.plot(g.date, 100 * g[TARGET], color="k", lw=1.2, label="actual")
    p = fc[(fc.state == s) & (fc.policy == "as_reported")].sort_values("h")
    a.fill_between(p.date, p.lo80, p.hi80, color="#1f4e9c", alpha=.15, lw=0, label="80% interval (as reported)")
    a.plot(p.date, p.Headline, color="#1f4e9c", marker="o", ms=2.2, label="forecast (as reported)")
    if s == ANOMALY_STATE:
        b = fc[(fc.state == s) & (fc.policy == "backlog_removed")].sort_values("h"); a.plot(b.date, b.Headline, color="#d6641a", marker="s", ms=2.2, ls="--", label="forecast (backlog removed)")
    a.set_title(s, fontsize=7.5); a.tick_params(axis="x", labelrotation=60, labelsize=5.8)
    a.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonth=[10, 1, 3, 6])); a.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b-%y"))
h_, l_ = ax[-1].get_legend_handles_labels(); h0, l0 = ax[0].get_legend_handles_labels()
fig.legend(h0 + h_[-1:], l0 + l_[-1:], ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=6.3)
fig.tight_layout(pad=0.4, rect=(0, 0.07, 1, 1)); fig.savefig(OUT / "fig4_forecast.pdf"); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
