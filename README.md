# AppScout: is your Play Store app idea viable?

Data Science Ecosystem (23CS2024) project. Type an app idea in one sentence; AppScout
places it in its market, finds its closest competitors (2018 data and live from the Play
Store), estimates its chance of reaching a million installs, explains that estimate input
by input, draws the market around it, and suggests better-aligned niches and the
complaints worth solving.

## Contents

| Path | What |
|---|---|
| `app/app.py` | The AppScout web app (Streamlit) |
| `app/views.py` | What is shown for an idea, shared by the web app and the Colab notebook |
| `start.bat` | One-click start on Windows |
| `AppScout_Colab.ipynb` | The same idea check in Google Colab |
| `notebooks/engine.py` | The viability engine, shared by the app and the notebooks |
| `notebooks/common.py` | Paths, cleaning, the shared split, `record()` into results.json |
| `notebooks/src/*.py` | The analysis notebooks as plain Python (what gets edited) |
| `notebooks/*.ipynb` | The same notebooks, executed, with outputs |
| `results/results.json` | Every number the project reports, written by code |
| `figures/` | Every figure, written by the notebooks |
| `data/` | 2018 Kaggle data (CC BY 3.0, see `license.txt`) and the 2026 scrape |
| `export/scrape_2026.py` | How the 2026 data was scraped |
| `report/` | Report builder and diagrams; the report itself is written to `private/` |

## The chapters

1. **See**: problems in the raw data, cleaning, missing values, outliers, the 2026 scrape.
2. **Explore**: descriptive statistics, categories, correlation, free vs paid, sentiment, 2018 vs 2026.
3. **Predict**: regression ladder for installs; leaked (reviews) against honest (pre-launch) models.
4. **Classify**: idea → category; will it reach 1M installs? (logistic, KNN, tree).
5. **Discover**: k-means and agglomerative clustering of genres into market segments.
6. **Prove**: seven Holm-corrected hypothesis tests, bootstrap, score validation (incl. 2026 apps launched after the data), fairness.
7. **Recommend**: content-based similar apps and hybrid niche suggestions, evaluated; complaint topics; case studies.

## Run it

**Windows, easiest:** double-click `start.bat`. The first run sets up a private Python environment (a few minutes,
needs internet and Python 3.10+), then AppScout opens at http://localhost:8501 (or the next free port).

**Google Colab:** open `AppScout_Colab.ipynb` in Colab, choose Runtime → Run all, and upload `AppScout.zip` when asked.
The "Check an idea" cell has a form with the same inputs as the website.

**By hand:**

```
pip install -r requirements.txt
python notebooks/build.py          # executes every notebook, rewrites results.json and figures
streamlit run app/app.py           # the web app
python report/diagrams.py          # report diagrams
python report/build_report.py      # report -> private/AppScout_DSE_Report.docx (needs the template in private/)
```

Self-checks: `python notebooks/common.py` and `python notebooks/engine.py`.

## Data

* Gupta, L. (2018). *Google Play Store Apps*. Kaggle. CC BY 3.0. `data/googleplaystore*.csv`
* 2026 scrape: `python export/scrape_2026.py` (google-play-scraper), `data/playstore_2026.csv`, log in `data/scrape_log.json`.
