# Web scraping log and methodology

This document lists **which public websites** this project scraped to pull reference tables into `data/scraped/`, and the **steps** we follow end to end.

> **Important:** These sources are **public pages** (U.S. government open data and Wikipedia). Scraping is done **lightly** (single GET per site, polite `User-Agent`, pause between hosts). Do **not** point this pattern at paywalled sites, competitor portals, or pages that forbid automated access in their terms of use. For production insurance workflows, use **licensed data feeds** or **official APIs** instead of HTML scraping.

## Websites scraped (as of this project)

| # | Website | Page URL | What we took | Output file(s) |
|---|---------|----------|----------------|----------------|
| 1 | **Federal Deposit Insurance Corporation (FDIC)** | https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/ | The main HTML **table** of failed banks (name, city, state, cert, acquirer, closing date, fund). | `data/scraped/fdic_failed_banks_list.csv` |
| 2 | **Wikipedia** | https://en.wikipedia.org/wiki/Health_insurance_coverage_in_the_United_States | (a) Small **timeline** table: year vs number uninsured / percent uninsured. (b) Larger **census division** table of insurance rates by year (wide format). | `data/scraped/wikipedia_us_uninsured_by_year.csv`, `data/scraped/wikipedia_us_insurance_rates_by_census_division_wide.csv` |

A machine-readable list of URLs and output paths is also written when you run the scraper: `data/scraped/scrape_run_meta.json`.

## Steps we use to scrape (reproducible checklist)

1. **Choose sources**  
   Prefer **government** pages and **Wikipedia** articles that publish facts in HTML tables, not private customer systems.

2. **Read the site’s rules**  
   Check `robots.txt` and the site’s terms. Wikipedia expects a **descriptive `User-Agent`** (see [Wikimedia Bot policy](https://meta.wikimedia.org/wiki/Bot_policy)). Federal sites generally allow reasonable access to public listings.

3. **Set request headers**  
   Use a clear `User-Agent` string that identifies the project and stack (see `scraping/scrape_public_sources.py`).

4. **Fetch HTML over HTTPS**  
   One GET per source, with a timeout (we use ~45 seconds).

5. **Parse HTML tables**  
   Use `pandas.read_html` (with `lxml`) to extract `<table>` elements into DataFrames. For FDIC, we drop stray rows that look like UI text (e.g. “Sort ascending”).

6. **Normalize and save**  
   Strip column names, write UTF-8 CSV under `data/scraped/`, and record URLs + paths in `scrape_run_meta.json`.

7. **Rate limiting**  
   Sleep ~1.5 seconds between **different hosts** (FDIC → Wikipedia) to avoid hammering servers.

8. **Verify outputs**  
   Open the CSVs locally or in a notebook; row counts and headers should match expectations.

## How to run the scraper yourself

From the **project root** (same folder as `config.yaml`):

```bash
source .venv/bin/activate   # if you use a venv
pip install -r requirements.txt
python scraping/scrape_public_sources.py
```

Dependencies used: `requests`, `pandas`, `lxml`.

## Augmented training CSV (merged onto customers)

After scraping, run:

```bash
python scraping/build_augmented_customers_from_scraped.py
```

That writes **`data/customers_augmented_public_health.csv`**: every row from `data/customers.csv` plus **`public_uninsured_rate_proxy`**, computed by averaging uninsured percentages (across years in the scraped Wikipedia state table) within U.S. states, then averaging again within each macro region code (`NE`, `SE`, `MW`, `W`, `S`) used in the synthetic data.

`config.yaml` points `data.customer_training_csv` at this file so notebooks **01**/**02** and the CSR **webapp** retrain on the augmented feature set.

## How this relates to the CLV notebooks

The **CLV / churn teaching pipeline** trains on the CSV named in **`data.customer_training_csv`** (default: augmented file above). Raw scraped tables remain under **`data/scraped/`** for transparency and for the plots in **`notebooks/03_scraped_data_retrain_and_metrics.ipynb`**.
