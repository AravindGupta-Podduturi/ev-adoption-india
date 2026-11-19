"""
00_build_vintage_panel.py -- release-calendar alignment of covariates
Inputs : data/ev_adoption_ml_dataset.csv (raw extract), config/pipeline_config.json
Outputs: data/ev_panel_vintage.csv   (covariates as KNOWN at each forecast origin, no back-fill)
         data/release_calendar.csv   (what was assumed released when)
         outputs/tables/00_vintage_audit.csv, outputs/figures/fig00_vintage_vs_backfilled.png
Run from the project root:  python 00_build_vintage_panel.py
"""
from common import *
from vintage import build_vintage_panel, release_calendar, charging_snapshots

raw = load_data(complete=False)
panel = build_vintage_panel(raw)
panel.to_csv(BASE_DIR / "data" / "ev_panel_vintage.csv", index=False)
cal = release_calendar()
cal.to_csv(BASE_DIR / "data" / "release_calendar.csv", index=False)
print(f"vintage panel: {panel.shape}  ->  data/ev_panel_vintage.csv\n")
print(cal.to_string(index=False))

# Audit: how much of the original (back-filled) covariate information was NOT yet available at each origin?
chk = raw[["state", "date", "charging_stations", "nsdp_per_capita"]].merge(
    panel[["state", "date", "charging_stations", "nsdp_per_capita"]], on=["state", "date"], suffixes=("_backfilled", "_asof"))
rows = []
for c in ["charging_stations", "nsdp_per_capita"]:
    b, a = chk[f"{c}_backfilled"], chk[f"{c}_asof"]
    rows.append({"variable": c, "months_missing_at_origin_%": 100 * a.isna().mean(),
                 "months_where_value_differs_from_backfilled_%": 100 * ((a != b) & a.notna()).mean(),
                 "mean_abs_gap_when_differs_%": 100 * ((a - b).abs() / b)[(a != b) & a.notna()].mean()})
aud = pd.DataFrame(rows).round(2)
print("\nHow different is the vintage-aligned view from the original back-filled column?\n", aud.to_string(index=False))
save_table(aud, "00_vintage_audit", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, c, ttl in zip(axes, ["charging_stations", "nsdp_per_capita"], ["Charging stations", "NSDP per capita"]):
    for s in STATES:
        g = chk[chk.state == s]
        ax.plot(g.date, g[f"{c}_backfilled"], color=STATE_COLORS[s], lw=1, alpha=.45)
        ax.plot(g.date, g[f"{c}_asof"], color=STATE_COLORS[s], lw=2, label=s)
    ax.set_title(f"{ttl}: as known at origin (solid) vs original back-filled column (faint)", fontsize=9.5)
axes[0].legend(fontsize=7.5, ncol=2)
fig.tight_layout(); save_fig(fig, "fig00_vintage_vs_backfilled")
