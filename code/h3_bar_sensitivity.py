"""
Robustness check: does H3 (sustained/uninterrupted media attention, Section
4.4) depend on the monthly coverage bar used to define a "covered" month?

The primary specification (sustained_coverage.py) counts a month as covered
if it has >= COVERED_ARTICLE_BAR (3) devrel-relevant articles. This script
rebuilds the identical H3 specification -- same sample, controls, two-way
FE, country-clustered SE, outcome -- at five alternative monthly bars
(>=1, >=2, >=4, >=5, >=10 articles; >=3 is the already-reported baseline)
crossed with the three streak lengths (>=24, >=36, >=60 consecutive
months), using lib/devrel_panel.py's attach_streak()/build_country_annual()
with their covered_article_bar parameter -- no duplicated logic, no
shortcuts. Added 2026-09-20 in response to a reviewer/editor-facing
threshold-sensitivity request; reported as Appendix Table A7.

Writes:
  results/models/h3_bar_sensitivity_results.csv        (standardized schema)
  results/diagnostics/h3_bar_sensitivity_coverage.csv   (share/count meeting
                                                          each streak definition)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import RESULTS_DIAGNOSTICS, RESULTS_MODELS, get_logger, write_model_results, write_rows
from lib import devrel_panel as dp
from lib import panel_windows as pw
from lib.regressions import fit_fe_ols, result_rows_from_fit

log = get_logger("h3_bar_sensitivity")

RESULTS_PATH = RESULTS_MODELS / "h3_bar_sensitivity_results.csv"
COVERAGE_PATH = RESULTS_DIAGNOSTICS / "h3_bar_sensitivity_coverage.csv"
COVERAGE_FIELDS = ["covered_article_bar", "streak_threshold_months", "n_sample", "n_meeting", "share_meeting"]

BARS = [1, 2, 3, 4, 5, 10]  # 3 is dp.COVERED_ARTICLE_BAR, the manuscript's baseline
THRESHOLDS = [24, 36, 60]


def main() -> None:
    country_month_long = pw.build_country_month_long()
    sentiment_month = dp.build_country_sentiment_month()

    result_rows = []
    coverage_rows = []

    for bar in BARS:
        annual_log = dp.build_country_annual(use_log=True, country_month_long=country_month_long,
                                              sentiment_month=sentiment_month, covered_article_bar=bar)

        for thresh in THRESHOLDS:
            col = f"streak_ge{thresh}"
            d = annual_log.copy()
            d[col] = (d["consecutive_periods_covered"] >= thresh).astype(float)

            dd = d.dropna(subset=["devrel_commitment_musd", col] + dp.CONTROL_COLS)
            n_sample = int(len(dd))
            n_meeting = int(dd[col].sum())
            coverage_rows.append({
                "covered_article_bar": bar, "streak_threshold_months": thresh,
                "n_sample": n_sample, "n_meeting": n_meeting,
                "share_meeting": (n_meeting / n_sample) if n_sample else "",
            })

            res = fit_fe_ols(d, "devrel_commitment_musd", [col] + dp.CONTROL_COLS,
                              verbose_label=f"bar>={bar} streak>={thresh}")
            result_rows += result_rows_from_fit(
                res, [col], specification_prefix=f"H3_bar_sensitivity_bar{bar}_ge{thresh}",
                outcome="ln_devrel_commitment_musd", controls=dp.CONTROL_COLS,
                sample_definition="Devrel-ODA panel, complete cases",
                timing_window=f"consecutive coverage streak >= {thresh} months at a >= {bar} "
                               "devrel-article/month coverage bar, evaluated as of November of year Y-1",
                transformation="binary indicator")
            if res is not None:
                log.info(f"bar>={bar} streak>={thresh}: coef={res.params[col]:.4f} "
                         f"se={res.std_errors[col]:.4f} p={res.pvalues[col]:.4f} n={int(res.nobs)} "
                         f"({n_meeting}/{n_sample} = {n_meeting / n_sample:.1%} meet the definition)")

    write_model_results(RESULTS_PATH, result_rows)
    log.info(f"Wrote {len(result_rows)} rows -> {RESULTS_PATH}")

    write_rows(COVERAGE_PATH, COVERAGE_FIELDS, coverage_rows)
    log.info(f"Wrote {len(coverage_rows)} rows -> {COVERAGE_PATH}")


if __name__ == "__main__":
    main()
