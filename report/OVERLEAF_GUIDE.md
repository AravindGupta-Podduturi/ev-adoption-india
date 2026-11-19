# Generating the report on Overleaf

## Steps
1. Go to overleaf.com, sign in, and choose **New Project > Blank Project** (name it e.g. "EV Adoption Report"). Overleaf already includes the IEEEtran class, so nothing needs installing.
2. Delete the default `main.tex`. Click **Upload** and add these four files from this folder: `main.tex`, `abstract.tex`, `body.tex`, `refs.tex`.
3. Create a folder named `figs` (New Folder) and upload the four PDFs inside it: `fig1_eda.pdf`, `fig2_shap.pdf`, `fig3_backtest.pdf`, `fig4_forecast.pdf`. The folder name must be exactly `figs`.
4. Open the menu (top left) and check: **Compiler = pdfLaTeX**, **Main document = main.tex**.
5. Click **Recompile**. Compile twice if citation numbers show as [?].
6. Check the page count: the PDF must be at most 6 pages **including references**. A test build here gave exactly 6, with page 6 almost full. If it spills onto page 7, shorten a sentence in the Discussion (`body.tex`) or cut the least useful reference.
7. Download the PDF with the **Download PDF** button (top of the preview) and re-read it once against `docs/MASTER_RESULTS_TABLE.md`.

## What you must change by hand before submission (search for `MANUAL` in the files)
| File | What to change |
|---|---|
| `body.tex`, section "Reproducibility" (search `GitHub repository link`) | Replace `[GitHub repository link]` with your repository URL, then delete the trailing `% MANUAL` comment |
| `body.tex`, "AI Use Declaration" (search `team to confirm`) | State the real AI-generated share (must be at most 20%), delete the bracket text and the `% MANUAL` comment. Rewrite as much prose as you can in your own words, since the draft text was AI-assisted |
| `refs.tex`, `\bibitem{nasa}` | Add the date you ran the NASA POWER script (printed in the output file's Metadata sheet) |
| `refs.tex`, `\bibitem{sansad}` and `\bibitem{mospi}` | Open each URL once and check it loads and matches the citation (the Rajya Sabha annexure also gives exact Delhi and West Bengal Aug-2026 charging values; see `docs/DATA_SOURCES.md`) |
| `main.tex` | Check author names, student IDs and email addresses |
| `body.tex`, Acknowledgment | Confirm the wording |

Numbers in the text, tables and figures come from `docs/MASTER_RESULTS_TABLE.md` and `outputs/`. If you re-run the pipeline and a number changes, update the sentence and re-upload `figs/*.pdf` (created by `python 12_report_figures.py`).
