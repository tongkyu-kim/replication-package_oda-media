"""
Crisis-related media and emergency ODA -- EXPLORATORY diagnostics from the
former H4 specification (manuscript Section 4.7, Appendix Table A4 Panel B,
Appendix Table A7, Table 4's exploratory crisis-type row).

UPDATED 2026-10-12 (revised H4): H4 is now tested in h4_common_exposure.py,
which compares emergency and non-emergency ODA against the SAME
development-relevant media exposure. The former formal H4 test -- a stacked
PPML contrast of crisis-related media -> emergency ODA against DevRel
media -> development ODA, previously item (2) of this script and the
source of the old Figure 5 and results/models/emergency_ppml_contrast_
results.csv -- has been removed from the package, because it changed the
media measure and the outcome at the same time. The analyses below use the
crisis-related media measure, not the H4 exposure, and are reported only as
exploratory diagnostics.

Within-emergency association: pooled crisis media -> emergency ODA,
country + year FE, country-clustered SE, 5 controls, log1p(y)-OLS:
  (a) contemporaneous window (k=-11..0, months 0-11 of the commitment
      year) -- Appendix Table A4 Panel B.
  (b) forward/pre-allocation window (k=2..13) -- Appendix Table A4 Panel B.
  (c) per-crisis-type heterogeneity (disaster/climate/conflict/health/
      food/refugee), contemporaneous window, each its own standalone
      regression -- Appendix Table A7 -- plus a joint Wald test (all 6
      jointly zero, and all 6 jointly equal) from ONE simultaneous
      regression with all 6 crisis-type media measures entered together.

Ported unchanged (statistical logic identical) from the original research
repository's pipeline/crisis_news/estimate_crisis_model.py.

Writes:
  results/models/emergency_temporal_results.csv       (items a/b/c coefficients)
  results/models/emergency_temporal_wald_results.csv   (item c's joint crisis-type Wald tests)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from _common import (RESULTS_MODELS, ModelResultRow, HYPOTHESIS_TEST_FIELDS, get_logger,
                      write_model_results, write_rows)
from lib import emergency_panel as ep
from lib.regressions import fit_fe_ols, result_rows_from_fit, wald_zero, hypothesis_test_row

log = get_logger("emergency_temporal")

OUT_RESULTS = RESULTS_MODELS / "emergency_temporal_results.csv"
OUT_WALD = RESULTS_MODELS / "emergency_temporal_wald_results.csv"

WINDOW_LABELS = {
    "lag0": "contemporaneous (months 0-11 of the commitment year, k=-11..0)",
    "m1_13": "2-13 months pre-allocation (forward/pre-allocation-style window)",
}

# ---------------------------------------------------------------------------
# (1a)/(1b) pooled within-emergency association
# ---------------------------------------------------------------------------

SPEC_IDS = {"lag0": "H4_within_emergency_contemporaneous", "m1_13": "H4_within_emergency_forward"}


def run_within_emergency(crisis_month_long: pd.DataFrame) -> list[ModelResultRow]:
    """Item (1a)/(1b): pooled crisis media -> emergency ODA, contemporaneous
    and forward windows."""
    rows: list[ModelResultRow] = []
    for window in ["lag0", "m1_13"]:
        annual, xcol = ep.build_country_annual(True, crisis_month_long, "crisis_articles", window)
        res = fit_fe_ols(annual, "emergency_commitment_musd", [xcol] + ep.CONTROL_COLS,
                          verbose_label=f"within-emergency, {window}")
        rows += result_rows_from_fit(
            res, [xcol], specification_prefix=SPEC_IDS[window],
            outcome="ln_emergency_commitment_musd", controls=ep.CONTROL_COLS,
            sample_definition="Emergency-ODA country-year panel (pooled crisis media -> emergency-reclassified "
                               "ODA), complete cases on media/outcome/controls",
            timing_window=WINDOW_LABELS[window], transformation="ln(x+1)")
        if res is not None:
            log.info(f"Within-emergency [{window}]: coef={res.params[xcol]:.4f} se={res.std_errors[xcol]:.4f} "
                      f"p={res.pvalues[xcol]:.4f} n={int(res.nobs)}")
    return rows


# ---------------------------------------------------------------------------
# (1c) per-crisis-type heterogeneity + joint Wald tests
# ---------------------------------------------------------------------------

def _wald_all_equal(res, coef_names: list[str]) -> tuple[float, int, float]:
    """H0: all named coefficients are pairwise equal, as a SINGLE joint
    restriction (beta_2=beta_1, beta_3=beta_1, ...), not
    len(coef_names)-choose-2 separate pairwise tests. Ported from
    estimate_crisis_model.py's _wald_all_equal -- kept local to this script
    rather than added to the shared lib.regressions, which only offers a
    single-pair wald_equal."""
    params = list(res.params.index)
    R = np.zeros((len(coef_names) - 1, len(params)))
    for i, name in enumerate(coef_names[1:]):
        R[i, params.index(coef_names[0])] = 1.0
        R[i, params.index(name)] = -1.0
    wt = res.wald_test(restriction=R, value=np.zeros(len(coef_names) - 1))
    return float(wt.stat), int(wt.df), float(wt.pval)


def run_crisis_type_heterogeneity(crisis_month_long: pd.DataFrame) -> tuple[list[ModelResultRow], list[dict]]:
    """Item (1c): six standalone per-crisis-type regressions (contemporaneous
    window, log spec) plus a joint Wald test across all 6 types from ONE
    simultaneous regression."""
    rows: list[ModelResultRow] = []
    wald_rows: list[dict] = []

    for ct in ep.CRISIS_TYPES:
        annual, xcol = ep.build_country_annual(True, crisis_month_long, ct, "lag0")
        res = fit_fe_ols(annual, "emergency_commitment_musd", [xcol] + ep.CONTROL_COLS,
                          verbose_label=f"crisis type: {ct}")
        if res is None:
            log.warning(f"Crisis type {ct}: insufficient data, skipping")
            continue
        # xcol is generically named "crisis_media_lag0" regardless of crisis
        # type (see build_country_annual's docstring); relabeled per-type
        # here purely for the OUTPUT row's key_predictor, via a rename on a
        # throwaway fit object is unnecessary -- result_rows_from_fit reads
        # coefficients off `res` by the xs names passed to fit_fe_ols, so we
        # pass xcol itself and then relabel only the emitted row below.
        type_rows = result_rows_from_fit(
            res, [xcol], specification_prefix=f"H4_crisis_type_{ct}",
            outcome="ln_emergency_commitment_musd", controls=ep.CONTROL_COLS,
            sample_definition=f"Emergency-ODA country-year panel, {ct}-type crisis media only, complete cases",
            timing_window=WINDOW_LABELS["lag0"], transformation="ln(x+1)")
        for r in type_rows:
            r.key_predictor = f"crisis_{ct}_lag0"
            r.specification_id = f"H4_crisis_type_{ct}__crisis_{ct}_lag0"
        rows += type_rows
        log.info(f"Crisis type [{ct}]: coef={res.params[xcol]:.4f} se={res.std_errors[xcol]:.4f} "
                  f"p={res.pvalues[xcol]:.4f} n={int(res.nobs)}")

    # Joint Wald tests: ONE simultaneous regression, all 6 crisis-type media
    # measures entered together (contemporaneous window, log spec).
    annual, _ = ep.build_country_annual(True, crisis_month_long, "crisis_articles", "lag0")  # base grid only
    xcols = []
    for ct in ep.CRISIS_TYPES:
        a, generic_xcol = ep.build_country_annual(True, crisis_month_long, ct, "lag0")
        xcol = f"crisis_{ct}_lag0"
        a = a.rename(columns={generic_xcol: xcol})
        annual = annual.merge(a[["entity", "year", xcol]], on=["entity", "year"], how="left")
        xcols.append(xcol)

    res_joint = fit_fe_ols(annual, "emergency_commitment_musd", xcols + ep.CONTROL_COLS,
                            verbose_label="joint crisis-type regression (simultaneous)")
    if res_joint is not None:
        stat, dfree, pval = wald_zero(res_joint, xcols)
        wald_rows.append(hypothesis_test_row(
            "H4_crisis_type_joint", "Wald: all 6 crisis-type coefficients = 0 (joint, simultaneous regression)",
            stat, dfree, pval, "b_disaster=b_climate=b_conflict=b_health=b_food=b_refugee=0", int(res_joint.nobs)))
        log.info(f"Joint crisis-type Wald (all=0): chi2({dfree})={stat:.3f} p={pval:.5f} n={int(res_joint.nobs)}")

        stat2, dfree2, pval2 = _wald_all_equal(res_joint, xcols)
        wald_rows.append(hypothesis_test_row(
            "H4_crisis_type_joint", "Wald: all 6 crisis-type coefficients are equal (single joint restriction, "
                                     "not pairwise)",
            stat2, dfree2, pval2, "b_disaster=b_climate=b_conflict=b_health=b_food=b_refugee", int(res_joint.nobs)))
        log.info(f"Joint crisis-type Wald (all equal): chi2({dfree2})={stat2:.3f} p={pval2:.5f}")

    return rows, wald_rows


def main() -> None:
    crisis_month_long = ep.build_crisis_month_long()

    results_rows = run_within_emergency(crisis_month_long)
    type_rows, type_wald_rows = run_crisis_type_heterogeneity(crisis_month_long)
    results_rows += type_rows
    write_model_results(OUT_RESULTS, results_rows)
    log.info(f"Wrote {len(results_rows)} rows -> {OUT_RESULTS}")

    wald_rows = type_wald_rows
    write_rows(OUT_WALD, HYPOTHESIS_TEST_FIELDS, wald_rows)
    log.info(f"Wrote {len(wald_rows)} rows -> {OUT_WALD}")


if __name__ == "__main__":
    main()
