"""
Outlet ideology classification and its aggregation to a country x year panel
of conservative-bloc, progressive-bloc and cross-partisan development-relevant
coverage over the 2-13-month window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import DATA_ANALYTICAL, require_file
from lib import panel_windows as pw

OUTLET_LONG_PATH = DATA_ANALYTICAL / "outlet_article_counts_long.csv"

CONSERVATIVE_OUTLETS: list[str] = ["조선일보", "중앙일보", "동아일보"]  # Chosun / JoongAng / Dong-A Ilbo
PROGRESSIVE_OUTLETS: list[str] = ["한겨레", "경향신문"]  # Hankyoreh / Kyunghyang Shinmun
CLASSIFIED_OUTLETS: list[str] = CONSERVATIVE_OUTLETS + PROGRESSIVE_OUTLETS

START_K, END_K = 2, 13  # CORRECTED 2026-09-20 -- see WINDOW NOTE above


def load_outlet_long() -> pd.DataFrame:
    require_file(OUTLET_LONG_PATH, "See DATA_AVAILABILITY.md / code/lib/ideology_panel.py for what this "
                                    "file is and where it comes from.")
    return pd.read_csv(OUTLET_LONG_PATH, encoding="utf-8-sig")


def pool_outlet_window(target: pd.DataFrame, long_df: pd.DataFrame,
                        start_k: int = START_K, end_k: int = END_K) -> pd.DataFrame:
    """One row per (target country, target year, source year, month, outlet,
    n_articles): the outlet-level long file pooled over the k=start_k..end_k
    pre-allocation window via the same calendar-lookup logic as the rest of
    this package (panel_windows.calendar_month_and_offset), inner-joined so
    a (country, source year, month) with no rows in long_df (i.e. zero
    articles at every classified/unclassified outlet that month) simply
    contributes nothing to the pooled sum -- the original script's own
    join semantics, not the NaN-vs-zero corpus-boundary convention used by
    panel_windows.attach_window elsewhere in this package."""
    membership_rows = []
    for k in range(start_k, end_k + 1):
        cal_month, n_years_back = pw.calendar_month_and_offset(k)
        sub = target[["country_iso3", "year"]].drop_duplicates().copy()
        sub["month"] = cal_month
        sub["src_year"] = sub["year"] - n_years_back
        membership_rows.append(sub[["country_iso3", "year", "src_year", "month"]])
    membership = pd.concat(membership_rows, ignore_index=True)
    pooled = membership.merge(
        long_df, left_on=["country_iso3", "src_year", "month"],
        right_on=["country_iso3", "year", "month"], how="inner", suffixes=("", "_src"))
    return pooled.rename(columns={"year": "target_year"})


def build_ideology_measures(target: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    """target: any frame with (country_iso3, year) columns (duplicates ok).
    Returns one row per distinct (country_iso3, year) with:
      total_window_articles_all12 -- all 12 outlets (classified + not), k=1-13 pooled
      conservative_media / progressive_media -- raw article counts, classified blocs only
      ln_conservative_media / ln_progressive_media -- ln(1+x)
      cross_partisan -- 1 if BOTH blocs have >=1 article in the window, else 0
      conservative_share -- conservative_media / (conservative_media + progressive_media),
                             NaN where the classified total is 0
      ideological_balance -- 1 - |2*conservative_share - 1|, so 1 = perfectly
                              balanced, 0 = entirely one bloc; NaN where conservative_share is NaN
    """
    pooled = pool_outlet_window(target, long_df)
    total_by_cy = pooled.groupby(["country_iso3", "target_year"])["n_articles"].sum().rename(
        "total_window_articles_all12")

    cons = pooled[pooled["outlet"].isin(CONSERVATIVE_OUTLETS)]
    prog = pooled[pooled["outlet"].isin(PROGRESSIVE_OUTLETS)]
    cons_by_cy = cons.groupby(["country_iso3", "target_year"])["n_articles"].sum().rename("conservative_media")
    prog_by_cy = prog.groupby(["country_iso3", "target_year"])["n_articles"].sum().rename("progressive_media")

    out = target[["country_iso3", "year"]].drop_duplicates().merge(
        total_by_cy.reset_index().rename(columns={"target_year": "year"}), on=["country_iso3", "year"], how="left")
    out = out.merge(cons_by_cy.reset_index().rename(columns={"target_year": "year"}), on=["country_iso3", "year"], how="left")
    out = out.merge(prog_by_cy.reset_index().rename(columns={"target_year": "year"}), on=["country_iso3", "year"], how="left")
    for c in ["total_window_articles_all12", "conservative_media", "progressive_media"]:
        out[c] = out[c].fillna(0.0)

    out["ln_conservative_media"] = np.log1p(out["conservative_media"])
    out["ln_progressive_media"] = np.log1p(out["progressive_media"])
    out["cross_partisan"] = ((out["conservative_media"] > 0) & (out["progressive_media"] > 0)).astype(float)
    classified_total = out["conservative_media"] + out["progressive_media"]
    out["conservative_share"] = np.where(classified_total > 0, out["conservative_media"] / classified_total, np.nan)
    out["ideological_balance"] = 1 - (2 * out["conservative_share"] - 1).abs()
    return out
