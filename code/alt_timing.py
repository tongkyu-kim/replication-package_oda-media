"""
Alternative timing of the baseline association (Appendix Table A3). Five
single-window regressions with country and year FE and country-clustered
standard errors (no controls): the contemporaneous year (lag0), the 2-13-month
window (lag1), and the three preceding twelve-month blocks (lag2-lag4).

Output: results/models/alt_timing_baseline_results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import RESULTS_MODELS, get_logger, write_model_results
from lib import devrel_panel as dp
from lib import panel_windows as pw
from lib.regressions import fit_fe_ols, result_rows_from_fit

log = get_logger("alt_timing")

OUT_PATH = RESULTS_MODELS / "alt_timing_baseline_results.csv"

# (specification suffix, column name, window bounds or None if already
# present in the shared devrel panel, human-readable timing window)
LAGS = [
    ("lag0", "devrel_media_lag0", (-11, 0), "Jan-Dec of the allocation year itself (contemporaneous, 12mo)"),
    ("lag1", "devrel_media_m1_13", None, "2-13 months pre-allocation"),
    ("lag2", "devrel_media_m14_25", None, "14-25 months pre-allocation"),
    ("lag3", "devrel_media_m26_37", None, "26-37 months pre-allocation"),
    ("lag4", "devrel_media_lag4", (38, 49), "38-49 months pre-allocation"),
]


def build_lag_variants(annual_log, country_month_long) -> "pd.DataFrame":
    min_ym, max_ym = pw.corpus_month_bounds(country_month_long)
    d = annual_log
    for _, col, bounds, _ in LAGS:
        if bounds is None:
            continue  # devrel_media_m1_13 / m14_25 / m26_37 already built by dp.build_country_annual
        start_k, end_k = bounds
        d = pw.attach_window(d, country_month_long, True, start_k=start_k, end_k=end_k,
                              colname=col, agg="sum", min_ym=min_ym, max_ym=max_ym)
    return d


def main() -> None:
    country_month_long = pw.build_country_month_long()
    sentiment_month = dp.build_country_sentiment_month()
    annual_log = dp.build_country_annual(use_log=True, country_month_long=country_month_long,
                                          sentiment_month=sentiment_month)
    annual_log = build_lag_variants(annual_log, country_month_long)

    rows = []
    for suffix, col, _, window in LAGS:
        res = fit_fe_ols(annual_log, "devrel_commitment_musd", [col], verbose_label=f"alt-timing {suffix}")
        rows += result_rows_from_fit(res, [col], specification_prefix=f"alt_timing_{suffix}",
                                      outcome="ln_devrel_commitment_musd", controls=[],
                                      sample_definition="Devrel-ODA panel (country FE + year FE, no controls, "
                                                         "bivariate single-regressor)",
                                      timing_window=window, transformation="ln(x+1)")
        if res is not None:
            log.info(f"{suffix} ({col}): coef={res.params[col]:.4f} se={res.std_errors[col]:.4f} "
                     f"p={res.pvalues[col]:.4f} n={int(res.nobs)}")

    write_model_results(OUT_PATH, rows)
    log.info(f"Wrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
