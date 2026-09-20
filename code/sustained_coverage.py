"""
H3 -- Sustained media attention: development-related coverage sustained
WITHOUT INTERRUPTION for longer consecutive periods is positively
associated with subsequent development ODA.

Three binary threshold indicators (>=24, >=36, >=60 consecutive covered
months -- a month is "covered" if it has at least 3 devrel-relevant
articles; the running count resets to zero on any month below that bar).
Each threshold enters its own specification with a single binary
regressor, mirroring H2's three-separate-rows presentation over the same
three horizons -- H2 asks whether the AVERAGE level of attention over a
window matters; H3 asks whether UNBROKEN duration matters. Reported as
Table 3 column [4].

Writes: results/models/sustained_coverage_results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import RESULTS_MODELS, get_logger, write_model_results
from lib import devrel_panel as dp
from lib import panel_windows as pw
from lib.regressions import fit_fe_ols, result_rows_from_fit

log = get_logger("sustained_coverage")

OUT_PATH = RESULTS_MODELS / "sustained_coverage_results.csv"

THRESHOLDS = [24, 36, 60]


def main() -> None:
    country_month_long = pw.build_country_month_long()
    sentiment_month = dp.build_country_sentiment_month()
    annual_log = dp.build_country_annual(use_log=True, country_month_long=country_month_long,
                                          sentiment_month=sentiment_month)

    rows = []
    for thresh in THRESHOLDS:
        col = f"streak_ge{thresh}"
        d = annual_log.copy()
        d[col] = (d["consecutive_periods_covered"] >= thresh).astype(float)
        res = fit_fe_ols(d, "devrel_commitment_musd", [col] + dp.CONTROL_COLS,
                          verbose_label=f"Streak >= {thresh} months")
        rows += result_rows_from_fit(res, [col], specification_prefix=f"H3_streak_ge{thresh}",
                                      outcome="ln_devrel_commitment_musd", controls=dp.CONTROL_COLS,
                                      sample_definition="Devrel-ODA panel, complete cases",
                                      timing_window=f"consecutive coverage streak >= {thresh} months, "
                                                     "evaluated as of November of year Y-1",
                                      transformation="binary indicator")
        if res is not None:
            log.info(f"streak>={thresh}: coef={res.params[col]:.4f} se={res.std_errors[col]:.4f} "
                     f"p={res.pvalues[col]:.4f} n={int(res.nobs)}")

    write_model_results(OUT_PATH, rows)
    log.info(f"Wrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
