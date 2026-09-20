"""
Secondary analysis -- evaluative tone. Does the tone of development-related
coverage (positive vs. negative share) carry information beyond its volume?

Positive and negative sentiment shares are entered SEPARATELY (not as a
single signed average), since a country-year that is 40% positive / 10%
negative and one that is 25% / 25% can share the same average sentiment
while describing very different coverage. Reported as Table 3 column [8].

Writes: results/models/evaluative_tone_results.csv and
results/models/evaluative_tone_wald_results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import RESULTS_MODELS, get_logger, write_model_results, write_rows, HYPOTHESIS_TEST_FIELDS
from lib import devrel_panel as dp
from lib import panel_windows as pw
from lib.regressions import fit_fe_ols, result_rows_from_fit, wald_zero, hypothesis_test_row

log = get_logger("evaluative_tone")

OUT_RESULTS = RESULTS_MODELS / "evaluative_tone_results.csv"
OUT_WALD = RESULTS_MODELS / "evaluative_tone_wald_results.csv"


def main() -> None:
    country_month_long = pw.build_country_month_long()
    sentiment_month = dp.build_country_sentiment_month()
    annual_log = dp.build_country_annual(use_log=True, country_month_long=country_month_long,
                                          sentiment_month=sentiment_month)

    d = annual_log.dropna(subset=["devrel_commitment_musd", "devrel_media_m1_13", "share_positive",
                                   "share_negative"] + dp.CONTROL_COLS).copy()

    rows, wald_rows = [], []

    res_main = fit_fe_ols(d, "devrel_commitment_musd", ["share_positive", "share_negative"] + dp.CONTROL_COLS,
                           verbose_label="sentiment main effect only (no media)")
    if res_main is not None:
        stat, dfree, pval = wald_zero(res_main, ["share_positive", "share_negative"])
        wald_rows.append(hypothesis_test_row("evaluative_tone_additive_no_media",
                                              "Wald: positive share = negative share = 0",
                                              stat, dfree, pval, "b_positive=b_negative=0", int(res_main.nobs)))

    res_additive = fit_fe_ols(d, "devrel_commitment_musd",
                               ["devrel_media_m1_13", "share_positive", "share_negative"] + dp.CONTROL_COLS,
                               verbose_label="media + sentiment (additive)")
    rows += result_rows_from_fit(res_additive, ["devrel_media_m1_13", "share_positive", "share_negative"],
                                  specification_prefix="evaluative_tone_additive",
                                  outcome="ln_devrel_commitment_musd",
                                  controls=dp.CONTROL_COLS,
                                  sample_definition="Devrel-ODA panel, complete cases on media/sentiment/controls",
                                  timing_window="2-13 months pre-allocation (media); annual mean (sentiment)",
                                  transformation="ln(x+1) media; raw share [0,1] sentiment")

    if res_additive is not None:
        c = res_additive.params["devrel_media_m1_13"]
        p = res_additive.pvalues["devrel_media_m1_13"]
        log.info(f"Additive spec, media coef={c:.4f} p={p:.4f} n={int(res_additive.nobs)}")

    write_model_results(OUT_RESULTS, rows)
    write_rows(OUT_WALD, HYPOTHESIS_TEST_FIELDS, wald_rows)
    log.info(f"Wrote {len(rows)} rows -> {OUT_RESULTS}; {len(wald_rows)} Wald rows -> {OUT_WALD}")


if __name__ == "__main__":
    main()
