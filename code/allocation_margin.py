"""
Secondary analysis -- extensive vs. intensive margin. Does media attention
mainly determine WHETHER a country receives any development ODA at all
(extensive margin), or HOW MUCH it receives conditional on already being a
recipient (intensive margin)? Reported as Table 3 column [6].

Ported unchanged (statistical logic identical) from the original research
repository's pipeline/mechanisms/estimate_extensive_intensive.py ("Model D"
/ "Model 6" -- both names refer to the same script across the project's
renumbering history).

ZERO-FILL PANEL: unlike every other analysis in this package, this one
needs true zero-ODA country-years, which the finalized devrel-ODA baseline
sample deliberately excludes (devrel_country_year.csv only has a row for a
country-year with at least one realized devrel ODA project). This script
therefore builds its own zero-filled country x year grid -- every country
in devrel_country_year.csv's own country set, crossed with every year
2010-2025 -- and fills missing commitments with 0. This is a genuine zero
commitment, not a data-collection gap: the underlying project-level ODA
source has complete, uninterrupted year coverage for the full recipient
universe (see the original script's docstring). Media construction
(2-13 month pre-allocation window, log1p), FE structure, and clustering
are otherwise the finalized baseline's, unchanged.

Two specifications:
  Extensive margin (D1) -- LPM: P(received any devrel ODA) ~ media +
    controls, country + year FE, country-clustered SE, fit on the FULL
    zero-filled grid. Outcome is a binary "received any ODA" indicator,
    NOT log-transformed.
  Intensive margin (D2) -- ln(ODA) ~ media + controls, country + year FE,
    country-clustered SE, fit on POSITIVE-COMMITMENT country-years only.
    Outcome is ln(devrel_commitment_musd) with NO "+1" -- unlike every
    other outcome in this package, this sample is strictly positive by
    construction (it excludes the zero rows precisely because they would
    make the log outcome ill-defined), so the original script logs the
    raw commitment directly rather than using the usual ln(x+1) convention.

Writes: results/models/allocation_margin_results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from _common import RESULTS_MODELS, get_logger, require_file, write_model_results
from lib import devrel_panel as dp
from lib import panel_windows as pw
from lib.regressions import fit_fe_ols, result_rows_from_fit

log = get_logger("allocation_margin")

OUT_PATH = RESULTS_MODELS / "allocation_margin_results.csv"

YEAR_MIN, YEAR_MAX = 2010, 2025  # matches the finalized baseline's own panel span

MEDIA_COL = "devrel_media_m1_13"  # ln(1 + article count), 2-13 months pre-allocation window


def build_zero_filled_panel(country_month_long: pd.DataFrame) -> pd.DataFrame:
    """Every (country, year) combination for 2010-2025 across the
    devrel-ODA panel's own country set, with missing devrel_commitment_musd
    filled to 0.0 -- the base panel for both margins below."""
    require_file(dp.DEVREL_PATH, "Run build_analytical_panels.py first, or see DATA_AVAILABILITY.md.")
    devrel = pd.read_csv(dp.DEVREL_PATH, encoding="utf-8-sig")
    countries = sorted(devrel["country_iso3"].unique())
    grid = pd.MultiIndex.from_product(
        [countries, range(YEAR_MIN, YEAR_MAX + 1)], names=["country_iso3", "year"]
    ).to_frame(index=False)
    grid = grid.merge(devrel[["country_iso3", "year", "devrel_commitment_musd"]],
                       on=["country_iso3", "year"], how="left")
    grid["devrel_commitment_musd"] = grid["devrel_commitment_musd"].fillna(0.0)
    grid["oda_positive"] = (grid["devrel_commitment_musd"] > 0).astype(float)
    grid["entity"] = grid["country_iso3"]

    min_ym, max_ym = pw.corpus_month_bounds(country_month_long)
    grid = pw.attach_window(grid, country_month_long, use_log=True, start_k=2, end_k=13,
                             colname=MEDIA_COL, agg="sum", min_ym=min_ym, max_ym=max_ym)

    require_file(dp.CONTROLS_PATH, "Run build_analytical_panels.py first.")
    controls = pd.read_csv(dp.CONTROLS_PATH, encoding="utf-8-sig")
    grid = grid.merge(controls[["country_iso3", "year"] + dp.CONTROL_COLS], on=["country_iso3", "year"], how="left")
    return grid


def run_extensive_margin(d0: pd.DataFrame) -> list:
    """D1: LPM on the full zero-filled grid. Outcome = 1{received any devrel
    ODA that year}, untransformed."""
    res = fit_fe_ols(d0, "oda_positive", [MEDIA_COL] + dp.CONTROL_COLS, verbose_label="extensive margin (LPM)")
    rows = result_rows_from_fit(
        res, [MEDIA_COL], specification_prefix="allocation_margin_extensive",
        outcome="oda_positive (binary: 1 if devrel ODA received, else 0)",
        controls=dp.CONTROL_COLS,
        sample_definition="Zero-filled country x year grid (every devrel-panel country crossed with "
                           f"{YEAR_MIN}-{YEAR_MAX}), including true zero-ODA rows",
        timing_window="2-13 months pre-allocation",
        transformation="outcome: binary indicator, untransformed; predictor: ln(x+1)",
    )
    if res is not None:
        log.info(f"Extensive margin (D1): coef={res.params[MEDIA_COL]:.4f} se={res.std_errors[MEDIA_COL]:.4f} "
                  f"p={res.pvalues[MEDIA_COL]:.4f} n={int(res.nobs)}")
    return rows


def run_intensive_margin(d0: pd.DataFrame) -> list:
    """D2: fit on positive-commitment country-years only. Outcome =
    ln(devrel_commitment_musd), NO "+1" -- strictly positive by
    construction on this restricted sample."""
    d_pos = d0[d0["oda_positive"] == 1].copy()
    d_pos["log_oda"] = np.log(d_pos["devrel_commitment_musd"])
    res = fit_fe_ols(d_pos, "log_oda", [MEDIA_COL] + dp.CONTROL_COLS, verbose_label="intensive margin")
    rows = result_rows_from_fit(
        res, [MEDIA_COL], specification_prefix="allocation_margin_intensive",
        outcome="ln(devrel_commitment_musd)",
        controls=dp.CONTROL_COLS,
        sample_definition="Zero-filled country x year grid, restricted to positive-commitment "
                           "country-years only",
        timing_window="2-13 months pre-allocation",
        transformation="outcome: ln(ODA), NOT ln(ODA+1) -- positive-only sample; predictor: ln(x+1)",
    )
    if res is not None:
        log.info(f"Intensive margin (D2): coef={res.params[MEDIA_COL]:.4f} se={res.std_errors[MEDIA_COL]:.4f} "
                  f"p={res.pvalues[MEDIA_COL]:.4f} n={int(res.nobs)}")
    return rows


def main() -> None:
    country_month_long = pw.build_country_month_long()
    d0 = build_zero_filled_panel(country_month_long)
    log.info(f"Zero-filled panel: {len(d0):,} rows ({d0['country_iso3'].nunique()} countries x "
              f"{YEAR_MAX - YEAR_MIN + 1} years), {d0['oda_positive'].sum():,.0f} positive-ODA rows "
              f"({d0['oda_positive'].mean():.1%})")

    rows = run_extensive_margin(d0) + run_intensive_margin(d0)
    write_model_results(OUT_PATH, rows)
    log.info(f"Wrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
