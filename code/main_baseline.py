"""
H1 baseline (Section 4.1; Table 3 columns [0]-[1]): an unadjusted pooled
benchmark and the preferred specification with country and year FE, five
controls and country-clustered standard errors, using media attention over
the 2-13 months preceding the allocation year.

Output: results/models/main_baseline_results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import statsmodels.formula.api as smf

from _common import (DATA_ANALYTICAL, RESULTS_MODELS, ModelResultRow, get_logger,
                      write_model_results)
from lib import devrel_panel as dp
from lib import panel_windows as pw
from lib.regressions import fit_fe_ols, result_rows_from_fit

log = get_logger("main_baseline")

OUT_PATH = RESULTS_MODELS / "main_baseline_results.csv"


def run_pooled_benchmark(annual_log) -> list[ModelResultRow]:
    """Table 3 column [0]: plain OLS, no FE, no controls, classical SE --
    deliberately naive, a reference point for the raw bivariate relationship."""
    d = annual_log.dropna(subset=["devrel_commitment_musd", "devrel_media_m1_13"])
    res = smf.ols("devrel_commitment_musd ~ devrel_media_m1_13", data=d).fit()
    coef, se, pval, n = (res.params["devrel_media_m1_13"], res.bse["devrel_media_m1_13"],
                          res.pvalues["devrel_media_m1_13"], int(res.nobs))
    log.info(f"Unadjusted pooled benchmark: coef={coef:.4f} se={se:.4f} p={pval:.4f} n={n}")
    return [ModelResultRow(
        specification_id="H1_unadjusted_pooled",
        outcome="ln_devrel_commitment_musd",
        key_predictor="devrel_media_m1_13",
        coefficient=coef, se=se, p_value=pval, n=n, n_entities=None,
        fe_structure="none", cluster_level="none (classical SE)", controls="none",
        sample_definition="Full devrel-ODA panel, complete cases on media/outcome only",
        timing_window="2-13 months pre-allocation",
        transformation="ln(x+1)",
    )]


def run_preferred(annual_log) -> list[ModelResultRow]:
    """Table 3 column [1]: the preferred, fully-adjusted H1 specification."""
    res_nc = fit_fe_ols(annual_log, "devrel_commitment_musd", ["devrel_media_m1_13"],
                         verbose_label="H1 no-controls")
    res_full = fit_fe_ols(annual_log, "devrel_commitment_musd",
                           ["devrel_media_m1_13"] + dp.CONTROL_COLS,
                           verbose_label="H1 preferred")
    rows = []
    rows += result_rows_from_fit(res_nc, ["devrel_media_m1_13"], specification_prefix="H1_no_controls",
                                  outcome="ln_devrel_commitment_musd", controls=[],
                                  sample_definition="Devrel-ODA panel (country FE + year FE, no controls)",
                                  timing_window="2-13 months pre-allocation", transformation="ln(x+1)")
    rows += result_rows_from_fit(res_full, ["devrel_media_m1_13"] + dp.CONTROL_COLS,
                                  specification_prefix="H1_preferred",
                                  outcome="ln_devrel_commitment_musd", controls=dp.CONTROL_COLS,
                                  sample_definition="Devrel-ODA panel, complete cases on media/outcome/controls",
                                  timing_window="2-13 months pre-allocation", transformation="ln(x+1)")
    if res_full is not None:
        c, s, p = (res_full.params["devrel_media_m1_13"], res_full.std_errors["devrel_media_m1_13"],
                   res_full.pvalues["devrel_media_m1_13"])
        log.info(f"H1 preferred: coef={c:.4f} se={s:.4f} p={p:.4f} n={int(res_full.nobs)}")
    return rows


def main() -> None:
    log.info("Building analytical panel (log spec)...")
    country_month_long = pw.build_country_month_long()
    sentiment_month = dp.build_country_sentiment_month()
    annual_log = dp.build_country_annual(use_log=True, country_month_long=country_month_long,
                                          sentiment_month=sentiment_month)

    rows = run_pooled_benchmark(annual_log) + run_preferred(annual_log)
    write_model_results(OUT_PATH, rows)
    log.info(f"Wrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
