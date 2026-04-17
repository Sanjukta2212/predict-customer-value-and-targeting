# Churn-driven CLV and acquisition targeting

This project demonstrates a **production-shaped** Python workflow that:

1. Trains a **churn classifier** with preprocessing, probability calibration, and standard ranking metrics (ROC-AUC, PR-AUC).
2. Estimates **Customer Lifetime Value (CLV)** by combining historical contribution with a **discounted finite-horizon** extension driven by predicted churn probability.
3. Simulates **acquisition targeting** by ranking prospects on predicted CLV, selecting the top fraction, and comparing outcomes to **random** selection of the same audience size.
4. Reports **incremental lift** (absolute and percent) versus random targeting.

The workflow lives in **Jupyter notebooks** under `notebooks/`.

## Data (where it comes from)

**Training file for churn/CLV** is set in `config.yaml` as `data.customer_training_csv`. The default is **`data/customers_augmented_public_health.csv`**: the original synthetic customers plus one column, **`public_uninsured_rate_proxy`**, built from **scraped** Wikipedia state uninsured tables (see `docs/SCRAPING.md` and `notebooks/03_scraped_data_retrain_and_metrics.ipynb`). The base synthetic table remains available as `data/customers.csv`.

- The core customer rows are **synthetic** (generated for teaching), **not** copied from a real insurer’s warehouse.
- Columns whose names start with **`_`** (`_acquisition_prob`, `_monthly_spend_intensity`) exist **only** for the acquisition simulation. **Do not** use them as normal model inputs in real life—they are teaching aids.

A longer plain-English explanation is at the top of `notebooks/01_preprocessing.ipynb`.

## Quick start

```bash
cd clv_acquisition_targeting
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Open Jupyter from the project root (so paths to `data/` and `config.yaml` resolve):

```bash
jupyter lab
```

Run the notebooks **in order** (refresh scraped files + retrain metrics anytime with step 0):

0. `notebooks/03_scraped_data_retrain_and_metrics.ipynb` *(optional but recommended when changing scrape logic)* — scrape public pages, rebuild the augmented CSV, exploratory plots, retrain the churn classifier, print **train/val/test row counts** and **test accuracy / ROC-AUC / PR-AUC**, and save `outputs/scraped_augment_*.png` plus `outputs/scraped_retrain_metrics.json`.
1. `notebooks/01_preprocessing.ipynb` — load the training CSV from config, exploratory diagrams, preprocessing recipe, train/validation/test split.
2. `notebooks/02_churn_clv_and_targeting.ipynb` — churn model, CLV, targeting simulation, figures, and `metrics.json`.

## CSR web app (single-customer CLV)

A small **FastAPI** site lets a CSR enter account fields, runs the churn + CLV stack on the server, and **streams a calculation trace** (NDJSON) so the browser can show each step while processing.

```bash
cd clv_acquisition_targeting
source .venv/bin/activate
uvicorn webapp.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` in a browser. API: `POST /api/predict` (JSON in one response) or `POST /api/predict/stream` (newline-delimited JSON events: `step` then `result`).

To run headless (CI or terminal):

```bash
jupyter nbconvert --execute --to notebook notebooks/01_preprocessing.ipynb
jupyter nbconvert --execute --to notebook notebooks/02_churn_clv_and_targeting.ipynb
```

Artifacts are written to `outputs/`:

- `churn_model_performance.png` — ROC and precision-recall curves (test).
- `clv_distribution.png` — distribution of predicted CLV (test pool).
- `lift_chart.png` — cumulative lift vs targeting depth (single stochastic draw for visualization).
- `metrics.json` — machine-readable metrics for CI or dashboards.

## Configuration and reproducibility

- Edit `config.yaml` for sample sizes, model choice, CLV economics, and targeting budget.
- The pipeline fixes **NumPy / sklearn random seeds** via `project.random_seed` for repeatable splits and simulations.

## Modeling assumptions (important for stakeholders)

### Churn label

Synthetic churn is generated from a latent “customer health” score influenced by RFM and engagement features. In production, align the label definition with your **observation window** and **business definition** of churn (contract cancel, 90-day inactivity, etc.).

### CLV formula (implemented)

Let `p` be the predicted probability of churn in the **next decision month** (consistent with how the classifier is trained). Let `m` be an estimated **monthly gross margin** from historical spend and tenure:

\[
m \approx \frac{\text{margin\_rate} \times \text{total\_spend}}{\max(\text{tenure\_months}, 1)}
\]

Historical realized contribution is approximated as `m × tenure_months`. Future value is a **capped geometric survival** sum:

\[
\text{future} = \sum_{t=1}^{H} (1-p)^{t-1} \cdot \frac{m}{(1+d)^t}
\]

where `H` is `max_horizon_months`, and `d` is `discount_monthly`. This is a pragmatic ranking model—not a full hierarchical Bayes BTYD implementation.

### Acquisition lift simulation (no leakage)

The CSV includes a **hidden** `_acquisition_prob` column used only for simulation. It is **not** passed as a model feature, so lift reflects whether observable retention economics proxy **unobserved** acquisition quality in this synthetic world. In real campaigns, replace latent probabilities with measured conversion from experiments (holdouts, geo tests, PSA).

## Notebook map

| Notebook | Role |
|----------|------|
| `notebooks/01_preprocessing.ipynb` | Data provenance, EDA, preprocessing, splits |
| `notebooks/02_churn_clv_and_targeting.ipynb` | Churn model, CLV, plots, targeting simulation, `metrics.json` |
| `config.yaml` | Parameters shared by both notebooks |

## Optional extensions

- Swap the random split in the notebooks for a **time-based** split when timestamps exist.
- Install XGBoost (`pip install xgboost`) and set `churn_model.model_type: xgboost` in `config.yaml`.
- Replace the Bernoulli acquisition draw with **uplift models** when randomized treatment logs are available.

## Troubleshooting

If `import sklearn` crashes immediately with a segmentation fault on your machine, it is usually a **binary wheel / OpenMP** mismatch. Try a fresh virtual environment, upgrade `pip`, and reinstall `numpy` and `scikit-learn`, or use the official Python build from python.org / conda-forge.
