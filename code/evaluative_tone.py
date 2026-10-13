"""
Evaluative tone (Section 4.6; Table 3 column [8]). Positive and negative
sentiment shares of development-relevant coverage are entered together with
the 2-13-month media volume.

Output: results/models/evaluative_tone_results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import RESULTS_MODELS, get_logger, write_model_results
from lib import devrel_panel as dp
from lib import panel_windows as pw
from lib.regressions import fit_fe_ols, result_rows_from_fit

log = get_logger("evaluative_tone")

OUT_RESULTS = RESULTS_MODELS / "evaluative_tone_results.csv"
XS = ["devrel_media_m1_13", "share_positive", "share_negative"]


def main() -> None:
    country_month_long = pw.build_country_month_long()
    annual_log = dp.build_country_annual(use_log=True, country_month_long=country_month_long,
                                         sentiment_month=dp.build_country_sentiment_month())
    d = annual_log.dropna(subset=["devrel_commitment_musd"] + XS + dp.CONTROL_COLS).copy()
    res = fit_fe_ols(d, "devrel_commitment_musd", XS + dp.CONTROL_COLS)
    rows = result_rows_from_fit(res, XS, specification_prefix="evaluative_tone_additive",
                                outcome="ln_devrel_commitment_musd", controls=dp.CONTROL_COLS,
                                sample_definition="Development-ODA panel, complete cases on media, sentiment and controls",
                                timing_window="2-13 months pre-allocation (media); annual mean (sentiment)",
                                transformation="ln(x+1) media; raw share [0,1] sentiment")
    write_model_results(OUT_RESULTS, rows)
    for r in rows:
        log.info(f"{r.key_predictor}: coef={r.coefficient:.4f} se={r.se:.4f} p={r.p_value:.4f}")


if __name__ == "__main__":
    main()
