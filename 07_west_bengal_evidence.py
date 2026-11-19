"""
07_west_bengal_evidence.py -- what do the data say about the Nov-2025..Feb-2026 West Bengal surge?
=========================================================================================================
Hypothesis (trade press, to be cited with the official notification): the state made e-rickshaw ("Toto") registration
mandatory in Oct-2025 (deadline 31-Dec-2025, extended to 31-Jan-2026), so the surge could be a BACKLOG of existing vehicles (a hypothesis: the evidence below is consistent with it but does not officially verify it).
Test inside the panel: if true, (a) West Bengal's three-wheeler registrations jump in exactly these months, (b) the
excess EVs almost equal the excess three-wheelers, (c) no other state shows it, (d) removing the excess restores a
normal EV share.  Run from the project root:  python 07_west_bengal_evidence.py
"""
from common import *
from vintage import backlog_adjusted_rate

df = load_data()
adj, base3w = backlog_adjusted_rate(df)
w = df[df.state == ANOMALY_STATE].sort_values("date").copy()
pre = w[(w.date < ANOMALY_START) & (w.date >= ANOMALY_START - pd.DateOffset(months=12))]
base_ev = float(pre.ev_count.median())
w["3W_baseline"] = base3w; w["excess_3W"] = (w.three_wheeler - base3w).clip(lower=0)
w["excess_EV"] = (w.ev_count - base_ev).clip(lower=0)
w["adj_rate_%"] = [100 * adj[int(t)] if int(t) in adj else 100 * r for t, r in zip(w.t, w[TARGET])]
w["rate_%"] = 100 * w[TARGET]; w["3W_share_%"] = 100 * w.three_wheeler / w.vehicle_total
view = w[w.date >= "2025-08-01"][["date", "vehicle_total", "ev_count", "rate_%", "three_wheeler", "3W_share_%", "excess_EV", "excess_3W", "adj_rate_%"]]
print(f"Pre-episode medians (12 months before {ANOMALY_START:%b-%Y}): three-wheelers {base3w:.0f}/month, EVs {base_ev:.0f}/month\n")
print(view.round(1).to_string(index=False))
fl = w[w.anomaly_flag == 1]
ratio = fl.excess_EV.sum() / fl.excess_3W.sum()
corr = np.corrcoef(w[w.date >= "2025-01-01"].ev_count, w[w.date >= "2025-01-01"].three_wheeler)[0, 1]
print(f"\nIn the {len(fl)} flagged months: excess EV registrations / excess three-wheeler registrations = {ratio:.2f} (1.0 = every extra EV is a three-wheeler)")
print(f"Correlation of monthly EV and three-wheeler counts, WB 2025-26: {corr:.2f}")
others = df[(df.state != ANOMALY_STATE) & df.date.between(ANOMALY_START, ANOMALY_END)].groupby("state").three_wheeler.max() / \
         df[(df.state != ANOMALY_STATE) & (df.date < ANOMALY_START) & (df.date >= ANOMALY_START - pd.DateOffset(months=12))].groupby("state").three_wheeler.median()
print("Other states: peak three-wheelers during the window / their pre-episode median:", others.round(2).to_dict())
print(f"West Bengal: peak three-wheelers / pre-episode median = {fl.three_wheeler.max() / base3w:.1f}x")
summ = {"baseline_three_wheelers_per_month": base3w, "baseline_EV_per_month": base_ev, "excess_EV_over_excess_3W_in_flagged_months": float(ratio),
        "corr_EV_vs_3W_2025_26": float(corr), "wb_peak_3W_multiple": float(fl.three_wheeler.max() / base3w),
        "other_states_peak_3W_multiple": {k: float(v) for k, v in others.items()},
        "adjusted_rates_flagged_months_%": {str(w.loc[w.t == k, 'date'].iloc[0].date()): round(100 * v, 2) for k, v in adj.items()},
        "evidence_status": "evidence CONSISTENT WITH an e-rickshaw registration backlog (numerical correspondence only; NOT officially verified); official notification and Vahan class split still to be cited"}
save_json(summ, "07_wb_evidence"); save_table(view.round(3), "07_wb_episode_table", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
v = w[w.date >= "2025-01-01"]
axes[0].bar(v.date, v.three_wheeler, 20, color="#eda100", label="three-wheeler registrations"); axes[0].plot(v.date, v.ev_count, color="#0b0b0b", lw=2, marker="o", ms=3, label="EV registrations")
axes[0].axhline(base3w, color=INK2, ls="--", lw=1); axes[0].set_title("West Bengal: EVs move with three-wheelers", fontsize=10); axes[0].legend(fontsize=8)
axes[1].plot(v.date, v["rate_%"], color="#eb6834", lw=2, label="as reported"); axes[1].plot(v.date, v["adj_rate_%"], color="#1baf7a", lw=2, label="backlog removed")
axes[1].set_title("EV share of registrations (%)", fontsize=10); axes[1].legend(fontsize=8)
for a in axes: a.tick_params(axis="x", labelrotation=45, labelsize=7.5)
fig.tight_layout(); save_fig(fig, "fig21_west_bengal_evidence")
