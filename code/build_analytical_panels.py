"""
Builds the country x year outcome panels (devrel_country_year.csv,
emergency_country_year.csv) from the cleaned project-level ODA data and
checks the pre-built inputs in data/analytical/ (see docs/DATA_CONSTRUCTION.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from _common import DATA_ANALYTICAL, get_logger, require_file

log = get_logger("build_analytical_panels")

ODA_CLEAN = DATA_ANALYTICAL / "oda_clean.csv"
DEVREL_OUT = DATA_ANALYTICAL / "devrel_country_year.csv"
EMERGENCY_OUT = DATA_ANALYTICAL / "emergency_country_year.csv"

MULTILATERAL_LABEL = "다자간"
# OECD CRS emergency/humanitarian-response purpose codes (same codes
# preprocess_oda.py maps toward SDG1/SDG13 in the original crosswalk;
# reproduced here as the standalone aid-modality definition this study's
# emergency-ODA analyses actually use).
HUMANITARIAN_EXACT = {"72010", "72040", "72050", "73010", "74010"}
HUMANITARIAN_PREFIX = {"720", "730", "740"}


def _full_country_year_grid(df: pd.DataFrame) -> pd.DataFrame:
    countries = sorted(df.loc[df["country_iso3"].notna() & (df["country_iso3"] != "---"),
                               "country_iso3"].unique())
    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    return pd.MultiIndex.from_product(
        [countries, range(year_min, year_max + 1)], names=["country_iso3", "year"]
    ).to_frame(index=False)


def build_devrel_panel(df: pd.DataFrame) -> pd.DataFrame:
    """devrel = all ODA commitments EXCLUDING International Organization /
    Multilateral Development Bank (IO/MDB) core-capital subscriptions.
    devrel_flag = NOT (bilateral_multilateral == multilateral AND sdg is
    null) -- a multilateral row that DOES carry an SDG tag (real sector
    attribution, e.g. Korea-directed multilateral trust-fund spending) is
    correctly retained in devrel. This is the outcome side of every
    development-ODA regression in the study; it deliberately differs from
    the simpler "multilateral alone" filter used for Table 2's descriptive
    sample-construction counts (see generate_tables.py) -- the two serve
    different purposes and are not meant to coincide."""
    is_multilateral = df["bilateral_multilateral"] == MULTILATERAL_LABEL
    has_sdg = df["sdg"].notna()
    io_mdb = is_multilateral & ~has_sdg
    devrel = ~io_mdb

    total_musd = df["commitment_musd"].sum()
    log.info(f"Total ODA: n={len(df):,} commitment=${total_musd:,.2f}M")
    log.info(f"IO/MDB (excluded): n={io_mdb.sum():,} (${df.loc[io_mdb, 'commitment_musd'].sum():,.2f}M)")
    log.info(f"Devrel (= total - IO/MDB): n={devrel.sum():,} "
             f"(${df.loc[devrel, 'commitment_musd'].sum():,.2f}M)")

    df_c = df[df["country_iso3"].notna() & (df["country_iso3"] != "---")].copy()
    devrel_by_cy = (
        df_c[devrel.loc[df_c.index]]
        .groupby(["country_iso3", "year"])
        .agg(devrel_commitment_musd=("commitment_musd", "sum"),
             devrel_n_projects=("project_id", "count"))
        .reset_index()
    )
    total_by_cy = (
        df_c.groupby(["country_iso3", "year"])
        .agg(total_commitment_musd=("commitment_musd", "sum"),
             total_n_projects=("project_id", "count"))
        .reset_index()
    )

    # Zero-fill: a (country, year) with no recorded project at all is a
    # true zero, not a missing row -- built from the explicit full grid so
    # it is present with commitment=0 rather than silently absent (a
    # missing row would implicitly condition every downstream regression
    # on "received some ODA that year," which is not this study's design).
    grid = _full_country_year_grid(df_c)
    out = grid.merge(total_by_cy, on=["country_iso3", "year"], how="left")
    out = out.merge(devrel_by_cy, on=["country_iso3", "year"], how="left")
    fill_cols = ["total_commitment_musd", "total_n_projects", "devrel_commitment_musd", "devrel_n_projects"]
    out[fill_cols] = out[fill_cols].fillna(0.0)
    out["devrel_share"] = out["devrel_commitment_musd"] / out["total_commitment_musd"].replace(0, pd.NA)
    return out


def build_emergency_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Emergency/humanitarian ODA identified directly from the OECD CRS
    sector_code (the aid-modality category as it exists in the raw data),
    not from a downstream SDG label -- a disaster-response project is
    emergency-modality spending regardless of which SDG Korea's own
    crosswalk happened to also tag it with."""
    df = df.copy()
    df["sector_code_str"] = df["sector_code"].astype(str).str.replace(r"\.0$", "", regex=True)
    is_emergency = (df["sector_code_str"].isin(HUMANITARIAN_EXACT)
                     | df["sector_code_str"].str[:3].isin(HUMANITARIAN_PREFIX))
    log.info(f"Emergency/humanitarian rows: {is_emergency.sum():,} of {len(df):,} "
             f"({is_emergency.sum() / len(df) * 100:.2f}%)")

    df_c = df[df["country_iso3"].notna() & (df["country_iso3"] != "---")].copy()
    is_emergency_c = is_emergency.loc[df_c.index]

    emergency_by_cy = (
        df_c[is_emergency_c]
        .groupby(["country_iso3", "year"])
        .agg(emergency_commitment_musd=("commitment_musd", "sum"),
             emergency_n_projects=("project_id", "count"))
        .reset_index()
    )
    total_by_cy = (
        df_c.groupby(["country_iso3", "year"])
        .agg(total_commitment_musd=("commitment_musd", "sum"))
        .reset_index()
    )

    grid = _full_country_year_grid(df)  # full universe, not just SDG-tagged rows -- "true zero" is judged against all ODA
    out = grid.merge(total_by_cy, on=["country_iso3", "year"], how="left")
    out = out.merge(emergency_by_cy, on=["country_iso3", "year"], how="left")
    fill_cols = ["total_commitment_musd", "emergency_commitment_musd", "emergency_n_projects"]
    out[fill_cols] = out[fill_cols].fillna(0.0)
    out["emergency_share"] = out["emergency_commitment_musd"] / out["total_commitment_musd"].replace(0, pd.NA)
    out["emergency_flag"] = (out["emergency_commitment_musd"] > 0).astype(int)
    return out


def main() -> None:
    for f in (ODA_CLEAN, DATA_ANALYTICAL / "panel_country_sdg_month.csv",
              DATA_ANALYTICAL / "country_year_controls.csv",
              DATA_ANALYTICAL / "crisis_news_country_month.csv"):
        require_file(f, "See DATA_AVAILABILITY.md for how each included analytical input is built.")

    df = pd.read_csv(ODA_CLEAN, encoding="utf-8-sig", low_memory=False)
    df["sdg"] = pd.to_numeric(df["sdg"], errors="coerce")

    devrel = build_devrel_panel(df)
    devrel.to_csv(DEVREL_OUT, index=False, encoding="utf-8-sig")
    log.info(f"Wrote {DEVREL_OUT} ({len(devrel):,} rows, {devrel['country_iso3'].nunique():,} countries)")

    emergency = build_emergency_panel(df)
    emergency.to_csv(EMERGENCY_OUT, index=False, encoding="utf-8-sig")
    log.info(f"Wrote {EMERGENCY_OUT} ({len(emergency):,} rows, {emergency['country_iso3'].nunique():,} countries)")


if __name__ == "__main__":
    main()
