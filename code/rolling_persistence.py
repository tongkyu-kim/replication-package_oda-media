"""
H2 persistent attention (Section 4.3; Table 3 column [3]): ln(ODA) on the
log of the mean monthly development-relevant article count over 24-, 36- and
60-month windows (months 2-25, 2-37, 2-61), one model per window, with
country and year FE, five controls and country-clustered standard errors.

Output: results/models/rolling_persistence_results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import RESULTS_MODELS, get_logger, write_model_results
from lib import devrel_panel as dp
from lib import panel_windows as pw
from lib.regressions import fit_fe_ols, result_rows_from_fit

log = get_logger("rolling_persistence")

OUT_PATH = RESULTS_MODELS / "rolling_persistence_results.csv"

WINDOWS = [
    ("devrel_media_roll24", "24-month rolling average", "2-25 months pre-allocation (24mo window)"),
    ("devrel_media_roll36", "36-month rolling average", "2-37 months pre-allocation (36mo window)"),
    ("devrel_media_roll60", "60-month rolling average", "2-61 months pre-allocation (60mo window)"),
]


def main() -> None:
    country_month_long = pw.build_country_month_long()
    sentiment_month = dp.build_country_sentiment_month()
    annual_log = dp.build_country_annual(use_log=True, country_month_long=country_month_long,
                                          sentiment_month=sentiment_month)

    rows = []
    for predictor, label, window in WINDOWS:
        res = fit_fe_ols(annual_log, "devrel_commitment_musd", [predictor] + dp.CONTROL_COLS,
                          verbose_label=label)
        rows += result_rows_from_fit(res, [predictor], specification_prefix=f"H2_{predictor}",
                                      outcome="ln_devrel_commitment_musd", controls=dp.CONTROL_COLS,
                                      sample_definition="Devrel-ODA panel, complete cases (all constituent "
                                                         "months must fall inside the corpus's active range)",
                                      timing_window=window, transformation="ln(mean(x)+1), logged after averaging")
        if res is not None:
            log.info(f"{label}: coef={res.params[predictor]:.4f} se={res.std_errors[predictor]:.4f} "
                     f"p={res.pvalues[predictor]:.4f} n={int(res.nobs)}")

    write_model_results(OUT_PATH, rows)
    log.info(f"Wrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
