"""
Builds the country x year analytical panel used by every development-ODA
regression in the study (recent/lagged-cycle, persistent-attention,
sustained-attention/streak, and evaluative-tone analyses).

Ported unchanged (statistical logic identical) from the original research
repository's pipeline/devrel_baseline/estimate_devrel_models.py
(build_country_annual, build_country_sentiment_month,
attach_sentiment_annual). This is shared plumbing, not a model in its own
right -- see main_baseline.py, lag_windows.py, rolling_persistence.py,
sustained_coverage.py, and evaluative_tone.py for the actual specifications
that consume this panel.

Timing convention: the 2-13 month pre-allocation window (not 1-13) excludes
December of year Y-1, which can contain post-decision media given Korea's
constitutional budget deadline (within 30 days before the start of the fiscal
year, i.e., early December). Every window in this module shares that
same near edge (k=2) for the same reason.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _common import DATA_ANALYTICAL, require_file
from lib import panel_windows as pw

DEVREL_PATH = DATA_ANALYTICAL / "devrel_country_year.csv"
CONTROLS_PATH = DATA_ANALYTICAL / "country_year_controls.csv"

CONTROL_COLS = ["ln_gdppc", "ln_population", "log_bilateral_trade",
                "ln_conflict_deaths", "ln_disaster_affected"]

SENTIMENT_COLS = ["avg_sentiment", "share_positive", "share_negative"]

COVERED_ARTICLE_BAR = 3  # a month counts toward a coverage streak if it has >=3 devrel articles


def build_country_sentiment_month() -> pd.DataFrame:
    """Article-count-weighted mean of avg_sentiment / share_positive /
    share_negative across all SDG rows per country-month."""
    df = pd.read_csv(pw.PANEL_PATH, usecols=["year", "month", "sdg", "country_iso3", "freq_articles"]
                      + SENTIMENT_COLS, low_memory=False)
    df = df[df["country_iso3"].notna() & (df["country_iso3"] != "---")]
    df["freq_articles"] = df["freq_articles"].fillna(0.0)
    df = df.dropna(subset=SENTIMENT_COLS)
    for c in SENTIMENT_COLS:
        df[f"_w_{c}"] = df[c] * df["freq_articles"]
    agg = {f"_wsum_{c}": (f"_w_{c}", "sum") for c in SENTIMENT_COLS}
    agg["_n"] = ("freq_articles", "sum")
    g = df.groupby(["country_iso3", "year", "month"]).agg(**agg).reset_index()
    for c in SENTIMENT_COLS:
        g[c] = g[f"_wsum_{c}"] / g["_n"].replace(0, np.nan)
    g = g.rename(columns={"country_iso3": "entity"})
    return g[["entity", "year", "month"] + SENTIMENT_COLS]


def attach_sentiment_annual(annual: pd.DataFrame, sentiment_month: pd.DataFrame) -> pd.DataFrame:
    """Annual sentiment = mean of the 12 monthly values over the same
    k=2-13 window as the recent-attention measure."""
    out = annual.copy()
    frames = []
    for k in range(2, 14):
        cal_month, n_years_back = pw.calendar_month_and_offset(k)
        src = sentiment_month[sentiment_month["month"] == cal_month][["entity", "year"] + SENTIMENT_COLS].copy()
        src["year"] = src["year"] + n_years_back
        src = src.rename(columns={c: f"_{c}_k{k}" for c in SENTIMENT_COLS})
        frames.append(src)
    merged = out[["entity", "year"]]
    for f in frames:
        merged = merged.merge(f, on=["entity", "year"], how="left")
    for c in SENTIMENT_COLS:
        c_cols = [f"_{c}_k{k}" for k in range(2, 14)]
        out[c] = merged[c_cols].mean(axis=1, skipna=True)
    return out


def attach_streak(annual: pd.DataFrame, country_month_long: pd.DataFrame, min_ym: int, max_ym: int,
                   covered_article_bar: float | None = None) -> pd.DataFrame:
    """Consecutive-coverage streak, true monthly granularity: the number of
    uninterrupted prior months (as of November of year Y-1, the near edge
    of the 2-13 month window) in which the recipient received at least
    covered_article_bar devrel-relevant articles (default: the module-level
    COVERED_ARTICLE_BAR, i.e. every specification reported in the
    manuscript), resetting to zero on any month below that bar.
    covered_article_bar is exposed as a parameter only so
    h3_bar_sensitivity.py can rebuild this measure at alternative monthly
    thresholds for a robustness check (Appendix Table A6); every other
    caller in this package relies on the default and is unaffected."""
    if covered_article_bar is None:
        covered_article_bar = COVERED_ARTICLE_BAR
    out = annual.copy()
    month_grid = pd.MultiIndex.from_product(
        [out["entity"].unique(), range(min_ym, max_ym + 1)], names=["entity", "_ym"]
    ).to_frame(index=False)
    month_grid["year"] = month_grid["_ym"] // 12
    month_grid["month"] = month_grid["_ym"] % 12 + 1
    month_grid = month_grid.merge(
        country_month_long[["entity", "year", "month", "freq_articles"]],
        on=["entity", "year", "month"], how="left")
    month_grid["_covered"] = month_grid["freq_articles"].fillna(0.0) >= covered_article_bar
    month_grid = month_grid.sort_values(["entity", "_ym"])

    def _running_streak(s: pd.Series) -> pd.Series:
        counter, vals = 0, []
        for covered in s:
            counter = counter + 1 if covered else 0
            vals.append(counter)
        return pd.Series(vals, index=s.index)

    month_grid["_streak_months"] = (
        month_grid.groupby("entity")["_covered"].apply(_running_streak).reset_index(level=0, drop=True)
    )
    streak_lookup = month_grid[["entity", "_ym", "_streak_months"]].rename(
        columns={"_streak_months": "consecutive_periods_covered"})
    out["_target_ym"] = (out["year"] - 1) * 12 + 10  # November of year Y-1 (k=2)
    out = out.merge(streak_lookup.rename(columns={"_ym": "_target_ym"}), on=["entity", "_target_ym"], how="left")
    out = out.drop(columns=["_target_ym"])
    out["consecutive_periods_covered"] = out["consecutive_periods_covered"].fillna(0.0)
    return out


def build_country_annual(use_log: bool, country_month_long: pd.DataFrame,
                          sentiment_month: pd.DataFrame,
                          covered_article_bar: float | None = None) -> pd.DataFrame:
    """The single shared country x year panel: recent (2-13mo), two further
    lag cycles (14-25, 26-37mo, for the temporal-decomposition diagnostic),
    persistent-attention rolling averages (roll24/36/60), the sustained-
    coverage streak, annual sentiment shares, and the five controls --
    merged onto the devrel-ODA base panel (`devrel_country_year.csv`).
    covered_article_bar passes through to attach_streak() (see its
    docstring); leave it at the default for every use except the H3
    coverage-threshold robustness check."""
    require_file(DEVREL_PATH, "Run build_analytical_panels.py first.")
    devrel = pd.read_csv(DEVREL_PATH, encoding="utf-8-sig")
    annual = devrel.copy()
    annual["entity"] = annual["country_iso3"]

    min_ym, max_ym = pw.corpus_month_bounds(country_month_long)

    annual = pw.attach_window(annual, country_month_long, use_log, start_k=2, end_k=13,
                               colname="devrel_media_m1_13", agg="sum", min_ym=min_ym, max_ym=max_ym)
    annual = pw.attach_window(annual, country_month_long, use_log, start_k=14, end_k=25,
                               colname="devrel_media_m14_25", agg="sum", min_ym=min_ym, max_ym=max_ym)
    annual = pw.attach_window(annual, country_month_long, use_log, start_k=26, end_k=37,
                               colname="devrel_media_m26_37", agg="sum", min_ym=min_ym, max_ym=max_ym)
    annual = pw.attach_window(annual, country_month_long, use_log, start_k=2, end_k=25,
                               colname="devrel_media_roll24", agg="mean", min_ym=min_ym, max_ym=max_ym)
    annual = pw.attach_window(annual, country_month_long, use_log, start_k=2, end_k=37,
                               colname="devrel_media_roll36", agg="mean", min_ym=min_ym, max_ym=max_ym)
    annual = pw.attach_window(annual, country_month_long, use_log, start_k=2, end_k=61,
                               colname="devrel_media_roll60", agg="mean", min_ym=min_ym, max_ym=max_ym)

    annual = attach_sentiment_annual(annual, sentiment_month)
    annual = attach_streak(annual, country_month_long, min_ym, max_ym, covered_article_bar=covered_article_bar)

    if use_log:
        annual["devrel_commitment_musd"] = np.log1p(annual["devrel_commitment_musd"].clip(lower=0))

    require_file(CONTROLS_PATH, "Run build_analytical_panels.py first.")
    controls = pd.read_csv(CONTROLS_PATH, encoding="utf-8-sig")
    annual = annual.merge(controls[["country_iso3", "year"] + CONTROL_COLS], on=["country_iso3", "year"], how="left")
    return annual
