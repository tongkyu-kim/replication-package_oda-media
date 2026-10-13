"""
Stacked panel for H4: each country-year contributes an emergency-ODA and a
non-emergency-ODA observation (which sum to the development-ODA outcome),
both matched to the same development-relevant media measures, with
country x aid-type and year x aid-type fixed-effect identifiers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import DATA_ANALYTICAL, require_file
from lib import devrel_panel as dp
from lib import panel_windows as pw

EMERGENCY_PATH = DATA_ANALYTICAL / "emergency_country_year.csv"
CONTROL_COLS = dp.CONTROL_COLS

MEDIA_COLS = ["devrel_media_contemp", "devrel_media_m1_13", "devrel_media_m14_25", "devrel_media_m26_37",
              "devrel_media_roll24", "devrel_media_roll36", "devrel_media_roll60",
              "streak_ge24", "streak_ge36", "streak_ge60"]


def build_outcomes() -> pd.DataFrame:
    """Country-year pooled, emergency, and non-emergency commitments ($M, unlogged)."""
    require_file(dp.DEVREL_PATH, "Run build_analytical_panels.py first.")
    require_file(EMERGENCY_PATH, "Run build_analytical_panels.py first, or see DATA_AVAILABILITY.md.")
    pooled = pd.read_csv(dp.DEVREL_PATH, encoding="utf-8-sig")[["country_iso3", "year", "devrel_commitment_musd"]]
    emg = pd.read_csv(EMERGENCY_PATH, encoding="utf-8-sig")[["country_iso3", "year", "emergency_commitment_musd"]]
    out = pooled.merge(emg, on=["country_iso3", "year"], how="left")
    out["emergency_commitment_musd"] = out["emergency_commitment_musd"].fillna(0.0)
    out = out.rename(columns={"country_iso3": "entity", "devrel_commitment_musd": "y_pooled",
                              "emergency_commitment_musd": "y_emergency"})
    out["y_nonemergency"] = out["y_pooled"] - out["y_emergency"]
    return out


def build_stacked_panel(country_month_long: pd.DataFrame | None = None,
                         sentiment_month: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long panel: one emergency and one non-emergency row per country-year.
    Outcome `y` is the unlogged commitment level, clipped at 0 for PPML."""
    if country_month_long is None:
        country_month_long = pw.build_country_month_long()
    if sentiment_month is None:
        sentiment_month = dp.build_country_sentiment_month()
    annual = dp.build_country_annual(True, country_month_long, sentiment_month)
    min_ym, max_ym = pw.corpus_month_bounds(country_month_long)
    annual = pw.attach_window(annual, country_month_long, True, start_k=-11, end_k=0,
                               colname="devrel_media_contemp", agg="sum", min_ym=min_ym, max_ym=max_ym)
    for L in (24, 36, 60):
        annual[f"streak_ge{L}"] = (annual["consecutive_periods_covered"] >= L).astype(float)

    base = annual[["entity", "year"] + MEDIA_COLS + CONTROL_COLS].merge(
        build_outcomes()[["entity", "year", "y_nonemergency", "y_emergency"]], on=["entity", "year"], how="left")

    rows = []
    for aid_type, col in (("nonemergency", "y_nonemergency"), ("emergency", "y_emergency")):
        part = base.drop(columns=["y_nonemergency", "y_emergency"]).copy()
        part["y"] = base[col].clip(lower=0)
        part["aid_type"] = aid_type
        rows.append(part)
    stacked = pd.concat(rows, ignore_index=True)
    stacked["is_emergency"] = (stacked["aid_type"] == "emergency").astype(float)
    stacked["entity_x"] = stacked["entity"] + "::" + stacked["aid_type"]
    stacked["time_x"] = stacked["year"].astype(str) + "::" + stacked["aid_type"]
    return stacked


def verify_panel(stacked: pd.DataFrame) -> dict:
    """Common-exposure and partition checks, computed (not assumed)."""
    a = stacked[stacked["aid_type"] == "nonemergency"].set_index(["entity", "year"])
    b = stacked[stacked["aid_type"] == "emergency"].set_index(["entity", "year"])
    a, b = a.align(b, join="inner", axis=0)
    mismatches = 0
    for c in MEDIA_COLS:
        both_nan = a[c].isna() & b[c].isna()
        mismatches += int((((a[c] - b[c]).abs() > 1e-12) | (a[c].isna() != b[c].isna()))[~both_nan].sum())
    out = build_outcomes()
    recon = float((out["y_nonemergency"] + out["y_emergency"] - out["y_pooled"]).abs().max())
    return {
        "country_years": int(len(a)),
        "stacked_rows": int(len(stacked)),
        "media_cells_checked": int(len(a) * len(MEDIA_COLS)),
        "media_exposure_mismatches": mismatches,
        "max_abs_partition_error_musd": recon,
        "negative_nonemergency_rows_clipped": int((out["y_nonemergency"] < 0).sum()),
        "emergency_zero_share": float((out["y_emergency"] <= 0).mean()),
        "nonemergency_zero_share": float((out["y_nonemergency"] <= 0).mean()),
    }
