"""
True rolling-month lag/window construction over the country x SDG x month
media panel.

Ported unchanged (statistical logic identical) from the original research
repository's pipeline/monthly_lags.py. Core primitive: month_lag_k = k
months before January 1 of year t, located by explicit
(calendar_month, year_offset) lookup rather than by relabeling a whole
prior-year block -- this generalizes cleanly to any k.

Zero-fill convention: a source (entity, year, month) with no rows at all is
true zero coverage IF that month falls inside the corpus's active range
[min_ym, max_ym]; a request reaching before the corpus began returns NaN
(coverage there is unknown, not zero).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _common import DATA_ANALYTICAL, require_file

PANEL_PATH = DATA_ANALYTICAL / "panel_country_sdg_month.csv"


def build_month_long() -> pd.DataFrame:
    """(entity, year, month, freq_articles) collapsed to entity = country_iso3 + '_' + sdg."""
    require_file(PANEL_PATH, "Run build_analytical_panels.py first, or see DATA_AVAILABILITY.md.")
    df = pd.read_csv(PANEL_PATH, usecols=["year", "month", "sdg", "country_iso3", "freq_articles"],
                      low_memory=False)
    df = df[df["country_iso3"].notna() & (df["country_iso3"] != "---")]
    df["sdg"] = pd.to_numeric(df["sdg"], errors="coerce")
    df = df[df["sdg"].notna()]
    df["sdg"] = df["sdg"].astype(int)
    df["freq_articles"] = df["freq_articles"].fillna(0.0)
    df["entity"] = df["country_iso3"] + "_" + df["sdg"].astype(str)
    monthly = df.groupby(["entity", "year", "month"])["freq_articles"].sum().reset_index()
    return monthly[["entity", "year", "month", "freq_articles"]]


def build_country_month_long() -> pd.DataFrame:
    """Same shape, but entity = country_iso3 alone (summed across SDG) --
    used by every country-level model (all of Models/H1-H4 and the
    secondary analyses; the SDG-level entity above is not used by anything
    reported in the final manuscript)."""
    require_file(PANEL_PATH, "Run build_analytical_panels.py first, or see DATA_AVAILABILITY.md.")
    df = pd.read_csv(PANEL_PATH, usecols=["year", "month", "sdg", "country_iso3", "freq_articles"],
                      low_memory=False)
    df = df[df["country_iso3"].notna() & (df["country_iso3"] != "---")]
    df["freq_articles"] = df["freq_articles"].fillna(0.0)
    monthly = df.groupby(["country_iso3", "year", "month"])["freq_articles"].sum().reset_index()
    monthly = monthly.rename(columns={"country_iso3": "entity"})
    return monthly[["entity", "year", "month", "freq_articles"]]


def corpus_month_bounds(month_long: pd.DataFrame) -> tuple[int, int]:
    ym = month_long["year"] * 12 + (month_long["month"] - 1)
    return int(ym.min()), int(ym.max())


def calendar_month_and_offset(k: int) -> tuple[int, int]:
    calendar_month = ((12 - k) % 12) + 1
    n_years_back = (k + 11) // 12  # ceil(k / 12)
    return calendar_month, n_years_back


def attach_month_lag(annual: pd.DataFrame, month_long: pd.DataFrame, k: int,
                      min_ym: int, max_ym: int) -> pd.Series:
    """Single month_lag_k column: k months before Jan 1 of each row's year.
    NaN where that source month falls outside the corpus's active range;
    0.0 where it's inside the range but genuinely uncovered."""
    calendar_month, n_years_back = calendar_month_and_offset(k)
    src = month_long[month_long["month"] == calendar_month][["entity", "year", "freq_articles"]].copy()
    src["year"] = src["year"] + n_years_back
    src = src.rename(columns={"freq_articles": "_val"})
    merged = annual[["entity", "year"]].merge(src, on=["entity", "year"], how="left")

    target_ym = annual["year"].to_numpy() * 12 - k
    in_range = (target_ym >= min_ym) & (target_ym <= max_ym)
    out = merged["_val"].to_numpy()
    out = np.where(in_range, np.nan_to_num(out, nan=0.0), np.nan)
    return pd.Series(out, index=annual.index)


def attach_window(annual: pd.DataFrame, month_long: pd.DataFrame, use_log: bool,
                   start_k: int, end_k: int, colname: str, agg: str = "sum",
                   min_ym: int | None = None, max_ym: int | None = None) -> pd.DataFrame:
    """Adds `colname` = sum or mean of month_lag_start_k .. month_lag_end_k
    (inclusive). agg='sum' for accumulation windows (recent/lagged-cycle
    analyses); agg='mean' for rolling-average windows (persistent-attention
    analysis). NaN if ANY constituent month falls outside the corpus's
    active range."""
    if min_ym is None or max_ym is None:
        min_ym, max_ym = corpus_month_bounds(month_long)

    out = annual.copy()
    parts = []
    for k in range(start_k, end_k + 1):
        col = f"_wk_{k}"
        out[col] = attach_month_lag(out, month_long, k, min_ym, max_ym)
        parts.append(col)

    complete = out[parts].notna().all(axis=1)
    raw_sum = out[parts].sum(axis=1, min_count=1)
    if agg == "sum":
        value = raw_sum
    elif agg == "mean":
        value = raw_sum / len(parts)
    else:
        raise ValueError(f"unknown agg {agg!r}")
    value = value.where(complete)

    if use_log:
        out[colname] = np.log1p(value)
    else:
        out[colname] = value

    out = out.drop(columns=parts)
    return out
