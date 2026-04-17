#!/usr/bin/env python3
"""
Fetch small public tables from government and Wikipedia pages.

Run from project root:
    python scraping/scrape_public_sources.py
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "scraped"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "clv-acquisition-targeting/1.0 (educational research; "
        "+https://github.com) python-requests"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_html(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=45)
    r.raise_for_status()
    return r.text


def clean_fdic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Drop pagination / header junk rows
    mask = df.iloc[:, 0].astype(str).str.contains("Sort ascending", case=False, na=False)
    df = df.loc[~mask]
    mask2 = df.iloc[:, 0].astype(str).str.match(r"^\s*$", na=False)
    df = df.loc[~mask2]
    return df.reset_index(drop=True)


def main() -> None:
    # --- Source 1: FDIC (U.S. federal government, public financial data) ---
    fdic_url = "https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/"
    html = fetch_html(fdic_url)
    fdic_tables = pd.read_html(io.StringIO(html))
    fdic = clean_fdic(fdic_tables[0])
    fdic_path = OUT_DIR / "fdic_failed_banks_list.csv"
    fdic.to_csv(fdic_path, index=False)
    print("Wrote", fdic_path, "rows", len(fdic))

    time.sleep(1.5)

    # --- Source 2: Wikipedia — health insurance coverage in the U.S. ---
    wiki_url = "https://en.wikipedia.org/wiki/Health_insurance_coverage_in_the_United_States"
    html2 = fetch_html(wiki_url)
    wiki_tables = pd.read_html(io.StringIO(html2))

    uninsured = wiki_tables[1]
    uninsured_path = OUT_DIR / "wikipedia_us_uninsured_by_year.csv"
    uninsured.to_csv(uninsured_path, index=False)
    print("Wrote", uninsured_path, "rows", len(uninsured))

    by_div = wiki_tables[4]
    by_div_path = OUT_DIR / "wikipedia_us_insurance_rates_by_census_division_wide.csv"
    by_div.to_csv(by_div_path, index=False)
    print("Wrote", by_div_path, "rows", len(by_div))

    meta = {
        "sources": [
            {"name": "FDIC failed bank list", "url": fdic_url, "output": str(fdic_path.relative_to(ROOT))},
            {"name": "Wikipedia — health insurance coverage in the United States", "url": wiki_url, "outputs": [
                str(uninsured_path.relative_to(ROOT)),
                str(by_div_path.relative_to(ROOT)),
            ]},
        ]
    }
    import json

    (OUT_DIR / "scrape_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
