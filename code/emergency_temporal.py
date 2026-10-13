"""
Crisis-related media and emergency ODA: exploratory diagnostics (Section 4.7;
Table 4; Appendix Table A7).

Emergency ODA (log) on crisis-related media coverage, country and year FE,
five controls, country-clustered standard errors:
  (a) all crisis types, contemporaneous window (months 0-11 of the commitment
      year) and 2-13-month pre-allocation window;
  (b) each crisis type separately (disaster, climate, conflict, health,
      food, refugee), contemporaneous window;
  (c) joint Wald test that all six crisis-type coefficients are zero, from
      one regression with all six measures.

Output: results/models/emergency_temporal_results.csv,
        results/models/emergency_temporal_wald_results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from _common import (RESULTS_MODELS, ModelResultRow, HYPOTHESIS_TEST_FIELDS, get_logger,
                     write_model_results, write_rows)
from lib import emergency_panel as ep
from lib.regressions import fit_fe_ols, result_rows_from_fit, wald_zero, hypothesis_test_row

log = get_logger("emergency_temporal")

OUT_RESULTS = RESULTS_MODELS / "emergency_temporal_results.csv"
OUT_WALD = RESULTS_MODELS / "emergency_temporal_wald_results.csv"
OUTCOME = "ln_emergency_commitment_musd"
WINDOW_LABELS = {"lag0": "contemporaneous (months 0-11 of the commitment year)",
                 "m1_13": "2-13 months pre-allocation"}
SPEC_IDS = {"lag0": "crisis_media_contemporaneous", "m1_13": "crisis_media_m1_13"}


def run_all_crisis_types(crisis_month_long: pd.DataFrame) -> list[ModelResultRow]:
    rows: list[ModelResultRow] = []
    for window in ["lag0", "m1_13"]:
        annual, xcol = ep.build_country_annual(True, crisis_month_long, "crisis_articles", window)
        res = fit_fe_ols(annual, "emergency_commitment_musd", [xcol] + ep.CONTROL_COLS)
        rows += result_rows_from_fit(
            res, [xcol], specification_prefix=SPEC_IDS[window], outcome=OUTCOME, controls=ep.CONTROL_COLS,
            sample_definition="Emergency-ODA country-year panel, complete cases",
            timing_window=WINDOW_LABELS[window], transformation="ln(x+1)")
    return rows


def run_crisis_types(crisis_month_long: pd.DataFrame) -> tuple[list[ModelResultRow], list[dict]]:
    rows: list[ModelResultRow] = []
    for ct in ep.CRISIS_TYPES:
        annual, xcol = ep.build_country_annual(True, crisis_month_long, ct, "lag0")
        res = fit_fe_ols(annual, "emergency_commitment_musd", [xcol] + ep.CONTROL_COLS)
        type_rows = result_rows_from_fit(
            res, [xcol], specification_prefix=f"crisis_type_{ct}", outcome=OUTCOME, controls=ep.CONTROL_COLS,
            sample_definition=f"Emergency-ODA country-year panel, {ct} crisis media, complete cases",
            timing_window=WINDOW_LABELS["lag0"], transformation="ln(x+1)")
        for r in type_rows:
            r.key_predictor = f"crisis_{ct}_lag0"
            r.specification_id = f"crisis_type_{ct}__crisis_{ct}_lag0"
        rows += type_rows

    annual, _ = ep.build_country_annual(True, crisis_month_long, "crisis_articles", "lag0")
    xcols = []
    for ct in ep.CRISIS_TYPES:
        a, generic_xcol = ep.build_country_annual(True, crisis_month_long, ct, "lag0")
        xcol = f"crisis_{ct}_lag0"
        annual = annual.merge(a.rename(columns={generic_xcol: xcol})[["entity", "year", xcol]],
                              on=["entity", "year"], how="left")
        xcols.append(xcol)
    res_joint = fit_fe_ols(annual, "emergency_commitment_musd", xcols + ep.CONTROL_COLS)
    stat, dfree, pval = wald_zero(res_joint, xcols)
    wald = [hypothesis_test_row("crisis_type_joint", "Wald: all 6 crisis-type coefficients = 0",
                                stat, dfree, pval, "b_disaster=b_climate=b_conflict=b_health=b_food=b_refugee=0",
                                int(res_joint.nobs))]
    return rows, wald


def main() -> None:
    crisis_month_long = ep.build_crisis_month_long()
    rows = run_all_crisis_types(crisis_month_long)
    type_rows, wald = run_crisis_types(crisis_month_long)
    write_model_results(OUT_RESULTS, rows + type_rows)
    write_rows(OUT_WALD, HYPOTHESIS_TEST_FIELDS, wald)
    for r in rows + type_rows:
        log.info(f"{r.specification_id}: coef={r.coefficient:.4f} se={r.se:.4f} p={r.p_value:.4f} n={r.n}")
    log.info(f"{wald[0]['test_name']}: chi2({wald[0]['df']})={wald[0]['test_statistic']:.3f}")


if __name__ == "__main__":
    main()
