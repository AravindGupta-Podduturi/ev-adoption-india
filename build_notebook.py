"""Assemble EV_Adoption_End_to_End.ipynb from the stage scripts (thin wrapper: each stage runs as its own cell).
Execute with:  jupyter nbconvert --to notebook --execute --inplace EV_Adoption_End_to_End.ipynb"""
import nbformat as nbf, re
from pathlib import Path
nb = nbf.v4.new_notebook()
nb.cells.append(nbf.v4.new_markdown_cell(
    "# Explainable and Forecasting-Based Analysis of Regional EV Adoption in India\n"
    "End-to-end notebook.  Every stage is a script in this folder; each cell below runs one stage.  "
    "**Retrospective layer** = stages 01-03 (same-month associations).  **Prospective layer** = stages 04-06 (only "
    "origin-available information, release-aligned covariates).  Stage 00 builds the vintage panel; stage 07 examines the West Bengal episode (evidence only, not officially verified)."))
order = ["00a_apply_reference_data.py", "00_build_vintage_panel.py", "01_data_audit_eda.py", "02_statistical_diagnostics.py", "03_explanatory_model_shap.py",
         "04_forecasting_benchmark.py", "07_west_bengal_evidence.py", "10_fixed_origin_backtest.py", "05_final_forecast_and_scenarios.py", "06_frozen_forecast_scorecard.py", "08_additional_metrics.py", "09_aggregate_analysis.py", "11_master_results_table.py", "12_report_figures.py"]
nb.cells.append(nbf.v4.new_code_cell("import sys, os\nsys.path.insert(0, os.getcwd())"))
for f in order:
    src = Path(f).read_text()
    doc = re.match(r'"""(.*?)"""', src, re.S)
    nb.cells.append(nbf.v4.new_markdown_cell(f"## `{f}`\n" + (doc.group(1).strip() if doc else "")))
    body = src[doc.end():].lstrip("\n") if doc else src
    nb.cells.append(nbf.v4.new_code_cell(body))
nbf.write(nb, "EV_Adoption_End_to_End.ipynb")
print("notebook written:", len(nb.cells), "cells")
