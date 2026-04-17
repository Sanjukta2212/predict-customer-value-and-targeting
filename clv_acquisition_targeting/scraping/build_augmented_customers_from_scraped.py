#!/usr/bin/env python3
"""
Merge public uninsured-rate signals (from scraped Wikipedia state table) onto
synthetic customers by macro-region (NE/SE/MW/W/S), then write a training CSV.

Requires:
  - data/customers.csv
  - data/scraped/wikipedia_us_insurance_rates_by_census_division_wide.csv
    (run scraping/scrape_public_sources.py first)

Usage (from project root):
    python scraping/build_augmented_customers_from_scraped.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Macro region codes used in synthetic customers.csv — map states → one of these.
STATE_TO_REGION: dict[str, str] = {
    **{s: "NE" for s in ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont", "New York", "New Jersey", "Pennsylvania"]},
    **{s: "SE" for s in ["Delaware", "Maryland", "District of Columbia", "Virginia", "West Virginia", "Kentucky", "Tennessee", "North Carolina", "South Carolina", "Georgia", "Florida"]},
    **{s: "MW" for s in ["Ohio", "Indiana", "Illinois", "Michigan", "Wisconsin", "Minnesota", "Iowa", "Missouri", "North Dakota", "South Dakota", "Nebraska", "Kansas"]},
    **{s: "W" for s in ["Montana", "Idaho", "Wyoming", "Colorado", "New Mexico", "Arizona", "Utah", "Nevada", "Washington", "Oregon", "California", "Alaska", "Hawaii"]},
    **{s: "S" for s in ["Alabama", "Mississippi", "Arkansas", "Louisiana", "Oklahoma", "Texas"]},
}


def region_public_uninsured_proxy(wiki_wide: pd.DataFrame) -> pd.Series:
    """Mean uninsured % (across years in table) averaged across states in each macro region."""
    df = wiki_wide.copy()
    df = df.loc[df["Division"].astype(str) != "United States"].copy()
    year_cols = [c for c in df.columns if c != "Division" and str(c).isdigit()]
    for c in year_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["_state_mean_uninsured_pct"] = df[year_cols].mean(axis=1)
    df["_macro"] = df["Division"].map(STATE_TO_REGION)
    df = df.dropna(subset=["_macro"])
    return df.groupby("_macro", as_index=True)["_state_mean_uninsured_pct"].mean()


def main() -> None:
    customers = pd.read_csv(ROOT / "data" / "customers.csv")
    wiki_path = ROOT / "data" / "scraped" / "wikipedia_us_insurance_rates_by_census_division_wide.csv"
    if not wiki_path.exists():
        raise FileNotFoundError(f"Missing {wiki_path}; run: python scraping/scrape_public_sources.py")

    wiki = pd.read_csv(wiki_path)
    proxy_by_region = region_public_uninsured_proxy(wiki)
    customers = customers.copy()
    customers["public_uninsured_rate_proxy"] = customers["region"].map(proxy_by_region).astype(float)
    if customers["public_uninsured_rate_proxy"].isna().any():
        missing = customers.loc[customers["public_uninsured_rate_proxy"].isna(), "region"].unique()
        raise ValueError(f"Unmapped regions after merge: {missing}")

    out = ROOT / "data" / "customers_augmented_public_health.csv"
    customers.to_csv(out, index=False)
    print("Wrote", out, "shape", customers.shape)
    print("public_uninsured_rate_proxy by region:\n", proxy_by_region)


if __name__ == "__main__":
    main()
