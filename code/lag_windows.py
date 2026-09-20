"""
Temporal decomposition -- a diagnostic, not a separate hypothesis test (see
manuscript Section 2.2 / 4.2). Development-related media attention is
decomposed into three non-overlapping windows (2-13, 14-25, 26-37 months
pre-allocation) entered JOINTLY, so each coefficient captures the
association within that window conditional on attention in the other two.
Reported as Table 3 column [2].

Writes: results/models/lag_windows_results.csv (coefficients) and
results/models/lag_windows_wald_results.csv (joint/equality tests).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import RESULTS_MODELS, get_logger, write_model_results, write_rows, HYPOTHESIS_TEST_FIELDS
from lib import devrel_panel as dp
from lib import panel_windows as pw
from lib.regressions import fit_fe_ols, result_rows_from_fit, wald_zero, wald_equal, hypothesis_test_row

log = get_logger("lag_windows")

OUT_RESULTS = RESULTS_MODELS / "lag_windows_results.csv"
OUT_WALD = RESULTS_MODELS / "lag_windows_wald_results.csv"

XS = ["devrel_media_m1_13", "devrel_media_m14_25", "devrel_media_m26_37"]


def main() -> None:
    country_month_long = pw.build_country_month_long()
    sentiment_month = dp.build_country_sentiment_month()
    annual_log = dp.build_country_annual(use_log=True, country_month_long=country_month_long,
                                          sentiment_month=sentiment_month)

    res = fit_fe_ols(annual_log, "devrel_commitment_musd", XS + dp.CONTROL_COLS,
                      verbose_label="temporal decomposition (3-cycle joint)")
    rows = result_rows_from_fit(res, XS, specification_prefix="lag_windows_joint",
                                 outcome="ln_devrel_commitment_musd", controls=dp.CONTROL_COLS,
                                 sample_definition="Devrel-ODA panel, complete cases",
                                 timing_window="2-13 / 14-25 / 26-37 months pre-allocation (joint)",
                                 transformation="ln(x+1)")
    write_model_results(OUT_RESULTS, rows)
    for r in rows:
        log.info(f"{r.key_predictor}: coef={r.coefficient:.4f} se={r.se:.4f} p={r.p_value:.4f}")

    wald_rows = []
    if res is not None:
        stat, dfree, pval = wald_zero(res, XS)
        wald_rows.append(hypothesis_test_row("lag_windows_joint", "Wald: all three windows = 0 (joint)",
                                              stat, dfree, pval, "b_m1_13=b_m14_25=b_m26_37=0", int(res.nobs)))
        stat, dfree, pval = wald_equal(res, "devrel_media_m1_13", "devrel_media_m26_37")
        wald_rows.append(hypothesis_test_row("lag_windows_joint", "Wald: recent window = most-distant window",
                                              stat, dfree, pval, "b_m1_13=b_m26_37", int(res.nobs)))
    write_rows(OUT_WALD, HYPOTHESIS_TEST_FIELDS, wald_rows)
    log.info(f"Wrote {len(rows)} coefficient rows -> {OUT_RESULTS}; {len(wald_rows)} Wald rows -> {OUT_WALD}")


if __name__ == "__main__":
    main()
