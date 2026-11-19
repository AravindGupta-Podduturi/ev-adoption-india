"""Run the whole pipeline in order.  Usage:  python run_all.py   (about 3.5 minutes on 2 CPU cores, measured 2026-11-18)."""
import subprocess, sys, time
STAGES = ["00a_apply_reference_data.py", "00_build_vintage_panel.py", "01_data_audit_eda.py", "02_statistical_diagnostics.py", "03_explanatory_model_shap.py",
          "04_forecasting_benchmark.py", "07_west_bengal_evidence.py", "10_fixed_origin_backtest.py", "05_final_forecast_and_scenarios.py", "06_frozen_forecast_scorecard.py", "08_additional_metrics.py", "09_aggregate_analysis.py", "11_master_results_table.py", "12_report_figures.py"]
for s in STAGES:
    t0 = time.time(); print(f"\n{'=' * 90}\n>>> {s}\n{'=' * 90}", flush=True)
    r = subprocess.run([sys.executable, s])
    if r.returncode:
        sys.exit(f"stage {s} failed")
    print(f"<<< {s} done in {time.time() - t0:.0f}s", flush=True)
