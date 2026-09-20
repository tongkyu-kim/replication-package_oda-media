"""
Builds the country x year (crisis-media -> emergency-ODA) panel and the
stacked development/emergency PPML panel used by code/emergency_temporal.py
(H4 -- development ODA vs. emergency ODA timing) and code/leave_one_out.py's
crisis-timing LOO section.

Ported unchanged (statistical logic identical) from the original research
repository's pipeline/crisis_news/estimate_crisis_model.py
(build_crisis_month_long, build_country_annual -- the within-emergency,
country x year specifications) and
pipeline/aid_type_temporal/ppml_diagnostic/sbx_ppml_windows.py +
sbx_emg_roll.py (build_dev_panel / build_emg_panel / build_stacked -- the
stacked PPML panel). This is shared plumbing, not a model in its own
right -- see emergency_temporal.py for the actual specifications that
consume it.

Two distinct outcome/predictor universes share this module because the
original repository's own scripts share them: the within-emergency
country x year panel (emergency_country_year.csv + crisis_news_country_
month.csv) feeds the log1p(y)-OLS association tests (item 1 of the H4
family: pooled contemporaneous/forward windows, per-crisis-type
heterogeneity); a STACKED development+emergency panel
(build_stacked_ppml_panel) feeds the formal PPML temporal-contrast test
(item 3 / Figure 5). The two are related but not interchangeable: the
within-emergency panel starts from emergency_country_year.csv alone; the
PPML panel also needs the devrel-ODA panel (via lib.devrel_panel) stacked
alongside it.

CONTROL_COLS/CONTROLS_PATH are re-exported from lib.devrel_panel rather
than redefined here -- the original repository's own estimate_crisis_
model.py imports the identical list from its own shared
estimate_hypotheses module (eh.CONTROL_COLS) rather than maintaining a
second copy, and this package preserves that "one control list, shared by
every model" convention.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import DATA_ANALYTICAL, require_file
from lib import devrel_panel as dp
from lib import panel_windows as pw

CRISIS_PANEL_PATH = DATA_ANALYTICAL / "crisis_news_country_month.csv"
EMERGENCY_PATH = DATA_ANALYTICAL / "emergency_country_year.csv"

CONTROL_COLS = dp.CONTROL_COLS
CONTROLS_PATH = dp.CONTROLS_PATH

CRISIS_TYPES = ["disaster", "climate", "conflict", "health", "food", "refugee"]

# Within-emergency association windows (item 1a/1b): "lag0" is the pooled
# CONTEMPORANEOUS window (k=-11..0, i.e. months 0-11 of the commitment
# year) -- the featured/primary spec for this analysis, since crisis/
# emergency response is theoretically a fast, event-triggered channel;
# "m1_13" is the forward-looking, pre-allocation budget-cycle window
# (k=2..13) reported for comparison (Appendix Table A4). Name kept as
# "m1_13" for continuity with the original script even though the actual
# bound is k=2, not k=1 (see lib/devrel_panel.py's own docstring on the
# same point for the devrel side).
WINDOWS = {
    "lag0": (-11, 0),
    "m1_13": (2, 13),
}

# The four temporal windows used by the stacked PPML contrast test (item 3
# / Figure 5): contemporaneous plus the same three non-overlapping
# pre-allocation cycles used throughout the devrel-baseline analyses.
PPML_WINDOWS = {
    "contemp": (-11, 0),
    "m1_13": (2, 13),
    "m14_25": (14, 25),
    "m26_37": (26, 37),
}


def build_crisis_month_long() -> pd.DataFrame:
    """Country x year x month crisis-news article counts, pooled
    (crisis_articles) and by crisis type (disaster/climate/conflict/
    health/food/refugee). Not "long" in the melted sense -- one row per
    country-month, one column per crisis type -- named for consistency with
    the original script's own build_crisis_month_long()."""
    require_file(CRISIS_PANEL_PATH, "Run build_analytical_panels.py first, or see DATA_AVAILABILITY.md.")
    df = pd.read_csv(CRISIS_PANEL_PATH, encoding="utf-8-sig")
    return df.rename(columns={"country_iso3": "entity"})


def build_country_annual(use_log: bool, crisis_month_long: pd.DataFrame, media_col: str,
                          window: str) -> tuple[pd.DataFrame, str]:
    """Country x year panel: emergency ODA commitment (base grid) plus one
    crisis-media window column (pooled or a single crisis type, selected via
    `media_col`), plus the 5 shared controls. Mirrors the original
    estimate_crisis_model.py's build_country_annual() exactly, including its
    column-naming convention (`crisis_media_{window}`, generic regardless of
    which crisis type was requested -- callers merge each type's column
    under its own name when several are needed at once, see
    emergency_temporal.py's run_crisis_type_heterogeneity)."""
    require_file(EMERGENCY_PATH, "Run build_analytical_panels.py first, or see DATA_AVAILABILITY.md.")
    emergency = pd.read_csv(EMERGENCY_PATH, encoding="utf-8-sig")
    annual = emergency.copy()
    annual["entity"] = annual["country_iso3"]

    month_long_single = crisis_month_long[["entity", "year", "month", media_col]].rename(
        columns={media_col: "freq_articles"})
    min_ym, max_ym = pw.corpus_month_bounds(month_long_single)
    start_k, end_k = WINDOWS[window]
    xcol = f"crisis_media_{window}"
    annual = pw.attach_window(annual, month_long_single, use_log, start_k=start_k, end_k=end_k,
                               colname=xcol, agg="sum", min_ym=min_ym, max_ym=max_ym)

    if use_log:
        annual["emergency_commitment_musd"] = np.log1p(annual["emergency_commitment_musd"].clip(lower=0))

    require_file(CONTROLS_PATH, "Run build_analytical_panels.py first.")
    controls = pd.read_csv(CONTROLS_PATH, encoding="utf-8-sig")
    annual = annual.merge(controls[["country_iso3", "year"] + CONTROL_COLS], on=["country_iso3", "year"], how="left")
    return annual, xcol


def build_dev_ppml_panel() -> pd.DataFrame:
    """Development-ODA side of the stacked PPML panel (item 3): the shared
    devrel country x year panel, plus a new contemporaneous window (devrel
    media has no pre-existing contemporaneous measure -- Models H1-H3 are
    pre-allocation only), with the RAW (unlogged) commitment level restored
    as the outcome -- PPML needs levels, not log1p(y). Mirrors
    sbx_ppml_windows.py's build_dev_panel()."""
    country_month_long = pw.build_country_month_long()
    sentiment_month = dp.build_country_sentiment_month()
    annual = dp.build_country_annual(True, country_month_long, sentiment_month)  # logged media + logged y
    min_ym, max_ym = pw.corpus_month_bounds(country_month_long)

    annual = pw.attach_window(annual, country_month_long, True, start_k=-11, end_k=0,
                               colname="devrel_media_contemp", agg="sum", min_ym=min_ym, max_ym=max_ym)

    require_file(dp.DEVREL_PATH, "Run build_analytical_panels.py first.")
    raw = pd.read_csv(dp.DEVREL_PATH, encoding="utf-8-sig")
    raw = raw.rename(columns={"country_iso3": "entity"})[["entity", "year", "devrel_commitment_musd"]]
    annual = annual.drop(columns=["devrel_commitment_musd"]).merge(raw, on=["entity", "year"], how="left")

    out = annual.rename(columns={
        "devrel_media_contemp": "contemp", "devrel_media_m1_13": "m1_13",
        "devrel_media_m14_25": "m14_25", "devrel_media_m26_37": "m26_37",
        "devrel_commitment_musd": "y",
    })
    return out[["entity", "year", "y", "contemp", "m1_13", "m14_25", "m26_37"] + CONTROL_COLS]


def build_emg_ppml_panel(crisis_month_long: pd.DataFrame) -> pd.DataFrame:
    """Emergency-ODA side of the stacked PPML panel (item 3): all four
    temporal windows built on POOLED crisis media (crisis_articles), RAW
    (unlogged) emergency commitment level as the outcome. Mirrors
    sbx_ppml_windows.py's build_emg_panel()."""
    require_file(EMERGENCY_PATH, "Run build_analytical_panels.py first, or see DATA_AVAILABILITY.md.")
    emg = pd.read_csv(EMERGENCY_PATH, encoding="utf-8-sig")
    annual = emg.copy()
    annual["entity"] = annual["country_iso3"]
    min_ym, max_ym = pw.corpus_month_bounds(crisis_month_long)

    month_long_pooled = crisis_month_long[["entity", "year", "month", "crisis_articles"]].rename(
        columns={"crisis_articles": "freq_articles"})
    for name, (sk, ek) in PPML_WINDOWS.items():
        annual = pw.attach_window(annual, month_long_pooled, True, start_k=sk, end_k=ek,
                                   colname=name, agg="sum", min_ym=min_ym, max_ym=max_ym)

    require_file(CONTROLS_PATH, "Run build_analytical_panels.py first.")
    controls = pd.read_csv(CONTROLS_PATH, encoding="utf-8-sig")
    annual = annual.merge(controls[["country_iso3", "year"] + CONTROL_COLS], on=["country_iso3", "year"], how="left")
    annual = annual.rename(columns={"emergency_commitment_musd": "y"})
    return annual[["entity", "year", "y", "contemp", "m1_13", "m14_25", "m26_37"] + CONTROL_COLS]


def build_stacked_ppml_panel() -> pd.DataFrame:
    """Development + emergency rows stacked into one long panel for the
    formal PPML temporal-contrast test (item 3 / Figure 5): entity_x =
    country::aid_type and time_x = factorize(year::aid_type) give each aid
    type its own country and year fixed effects; `entity` (the ORIGINAL
    country) is kept so SE can be clustered on it -- a country's
    development-side and emergency-side residuals are allowed to correlate,
    which is what makes the joint model's covariance matrix valid for a
    formal cross-process difference test. Mirrors sbx_ppml_windows.py's
    build_stacked()."""
    dev = build_dev_ppml_panel()
    dev["aid_type"] = "dev"
    crisis_month_long = build_crisis_month_long()
    emg = build_emg_ppml_panel(crisis_month_long)
    emg["aid_type"] = "emg"
    stacked = pd.concat([dev, emg], ignore_index=True)
    stacked = stacked.dropna(subset=["y", "contemp", "m1_13", "m14_25", "m26_37"] + CONTROL_COLS).copy()

    # PPML requires a weakly nonnegative outcome. The original panel has one
    # row (ECU, 2025, dev) with a small negative net commitment (-0.61 musd,
    # presumably a cancellation/correction exceeding new commitments that
    # year) -- clipped to 0 here, matching the original script exactly (not
    # a silent drop; n_neg is 0 or 1 in practice).
    stacked["y"] = stacked["y"].clip(lower=0)
    stacked["is_emg"] = (stacked["aid_type"] == "emg").astype(float)
    stacked["entity_x"] = stacked["entity"] + "::" + stacked["aid_type"]
    stacked["time_x"] = pd.factorize(stacked["year"].astype(str) + "::" + stacked["aid_type"])[0]
    return stacked
