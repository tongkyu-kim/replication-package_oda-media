"""
Country x year panel for the exploratory crisis-media diagnostics: emergency
ODA, crisis-related media coverage (all types and by type) and the shared
controls.
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
