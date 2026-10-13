"""
Temporal decomposition (Section 4.2; Table 3 column [2]). Development-relevant
attention in the 2-13-, 14-25- and 26-37-month windows is entered jointly.

Output: results/models/lag_windows_results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import RESULTS_MODELS, get_logger, write_model_results
from lib import devrel_panel as dp
from lib import panel_windows as pw
from lib.regressions import fit_fe_ols, result_rows_from_fit

log = get_logger("lag_windows")

OUT_RESULTS = RESULTS_MODELS / "lag_windows_results.csv"
XS = ["devrel_media_m1_13", "devrel_media_m14_25", "devrel_media_m26_37"]


def main() -> None:
    country_month_long = pw.build_country_month_long()
    annual_log = dp.build_country_annual(use_log=True, country_month_long=country_month_long,
                                         sentiment_month=dp.build_country_sentiment_month())
    res = fit_fe_ols(annual_log, "devrel_commitment_musd", XS + dp.CONTROL_COLS)
    rows = result_rows_from_fit(res, XS, specification_prefix="lag_windows_joint",
                                outcome="ln_devrel_commitment_musd", controls=dp.CONTROL_COLS,
                                sample_definition="Development-ODA panel, complete cases",
                                timing_window="2-13 / 14-25 / 26-37 months pre-allocation (joint)",
                                transformation="ln(x+1)")
    write_model_results(OUT_RESULTS, rows)
    for r in rows:
        log.info(f"{r.key_predictor}: coef={r.coefficient:.4f} se={r.se:.4f} p={r.p_value:.4f}")


if __name__ == "__main__":
    main()
