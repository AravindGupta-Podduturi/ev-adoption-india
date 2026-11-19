# Putting this project on GitHub

The folder is ready to upload as it is (about 20 MB, no file near GitHub's 100 MB limit; `.gitignore`, `LICENSE`, `requirements.txt` and `README.md` are included).

## Option A: command line
1. Create the repository: on github.com click **New repository**. Name it (e.g. `ev-adoption-india`), choose Public or Private, and leave **Add a README / .gitignore / licence all unchecked** (the folder already has them). Click **Create repository** and copy the URL it shows.
2. Install Git (git-scm.com) and set your identity once:
   `git config --global user.name "Your Name"` and `git config --global user.email "you@example.com"`
3. Open a terminal inside the project folder (the one containing `README.md`) and run:
```bash
git init
git add .
git commit -m "EV adoption analysis: data, pipeline, results, report sources"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
4. GitHub asks you to sign in. Use the browser prompt, or a Personal Access Token as the password (GitHub > Settings > Developer settings > Personal access tokens, with "repo" permission).
5. Refresh the repository page: you should see `README.md` rendered, the numbered stage scripts, `data/`, `docs/`, `outputs/`, `report/`.

## Option B: no command line
On the new empty repository page click **uploading an existing file**, drag the *contents* of the project folder in (GitHub accepts folders by drag-and-drop, up to 100 files per upload, so you may need several batches), and click **Commit changes**. Option A is simpler for this many files.

## After pushing
* Paste the repository URL into `report/body.tex` (Reproducibility section) and recompile on Overleaf.
* Add a short repository description and topics (e.g. `electric-vehicles`, `forecasting`, `shap`).
* Test it once from a clean copy: `git clone <url>`, `pip install -r requirements.txt`, `python run_all.py` (about 3.5 minutes on 2 cores) should regenerate every table and figure.
* Do not upload raw personal data or credentials. Nothing like that is in this folder. The two team-tracking files (earlier-version comparison and review responses) are deliberately not included.
* Later changes: `git add .`, `git commit -m "message"`, `git push`.
