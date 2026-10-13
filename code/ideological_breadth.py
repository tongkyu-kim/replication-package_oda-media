"""
Ideological breadth of coverage (Section 4.6; Table 3 column [7]; Figure 7).

CrossPartisan = 1 when development-relevant coverage of the recipient appears
in both the conservative bloc (Chosun Ilbo, JoongAng Ilbo, Dong-A Ilbo) and the
progressive bloc (Kyunghyang Shinmun, Hankyoreh) within the 2-13-month window.
The model interacts CrossPartisan with 2-13-month media volume (country and
year FE, five controls, country-clustered standard errors). Figure 7 data:
predicted ln(ODA) over the 5th-95th percentile of media attention, controls
at their means, with delta-method 90% confidence intervals.

Output: results/models/ideological_breadth_results.csv,
        results/figure_data/figure7_ideological_breadth_data.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from _common import (RESULTS_MODELS, RESULTS_FIGURE_DATA, SIGNIFICANCE_CI_LEVEL, Z_90,
                      get_logger, write_model_results, write_rows)
from lib import devrel_panel as dp
from lib import ideology_panel as ip
from lib import panel_windows as pw
from lib.regressions import fit_fe_ols, result_rows_from_fit

log = get_logger("ideological_breadth")

OUT_RESULTS = RESULTS_MODELS / "ideological_breadth_results.csv"
OUT_FIGDATA = RESULTS_FIGURE_DATA / "figure7_ideological_breadth_data.csv"

MEDIA_COL = "devrel_media_m1_13"  # ln(1 + article count), 2-13 months pre-allocation window
FIGDATA_FIELDS = ["media_attention_level", "predicted_log_oda", "ci_lower", "ci_upper", "ci_level", "coverage_type"]
N_GRID_POINTS = 100


def build_panel() -> pd.DataFrame:
    country_month_long = pw.build_country_month_long()
    sentiment_month = dp.build_country_sentiment_month()
    annual_log = dp.build_country_annual(use_log=True, country_month_long=country_month_long,
                                          sentiment_month=sentiment_month)

    long_df = ip.load_outlet_long()
    ideo = ip.build_ideology_measures(annual_log[["country_iso3", "year"]], long_df)
    d = annual_log.merge(ideo, on=["country_iso3", "year"], how="left")
    d["media_x_crosspartisan"] = d[MEDIA_COL] * d["cross_partisan"]
    return d


def run_regression(d: pd.DataFrame):
    xs = [MEDIA_COL, "cross_partisan", "media_x_crosspartisan"] + dp.CONTROL_COLS
    res = fit_fe_ols(d, "devrel_commitment_musd", xs, verbose_label="ideological breadth (cross-partisan salience)")
    rows = result_rows_from_fit(
        res, [MEDIA_COL, "cross_partisan", "media_x_crosspartisan"],
        specification_prefix="ideological_breadth_crosspartisan",
        outcome="ln_devrel_commitment_musd",
        controls=dp.CONTROL_COLS,
        sample_definition="Devrel-ODA panel, complete cases on media/cross-partisan measures/controls",
        timing_window="2-13 months pre-allocation (media, controls, and conservative/progressive "
                       "bloc pooling -- see code/lib/ideology_panel.py)",
        transformation="ln(x+1) media; binary cross-partisan indicator; product term",
    )
    if res is not None:
        c = res.params["media_x_crosspartisan"]
        p = res.pvalues["media_x_crosspartisan"]
        log.info(f"media_x_crosspartisan: coef={c:.4f} p={p:.4f} n={int(res.nobs)}")
    return res, rows


def build_figure_data(d: pd.DataFrame, res) -> list[dict]:
    """Predicted ln(ODA) for cross-partisan (coverage_type='cross_partisan')
    vs. single-bloc (coverage_type='single_bloc') coverage across the
    observed 5th-95th percentile range of media attention, with
    delta-method 90% CI ribbons -- same computation as the original
    repository's make_figure_model6_crosspartisan_prediction_ci.py."""
    if res is None:
        return []
    xs = [MEDIA_COL, "cross_partisan", "media_x_crosspartisan"] + dp.CONTROL_COLS
    dd = d.dropna(subset=["devrel_commitment_musd"] + xs).copy()

    b_media = res.params[MEDIA_COL]
    b_cp = res.params["cross_partisan"]
    b_int = res.params["media_x_crosspartisan"]
    cov3 = res.cov.loc[[MEDIA_COL, "cross_partisan", "media_x_crosspartisan"],
                        [MEDIA_COL, "cross_partisan", "media_x_crosspartisan"]].to_numpy()

    y_bar = dd["devrel_commitment_musd"].mean()
    x_bar = dd[MEDIA_COL].mean()
    cp_bar = dd["cross_partisan"].mean()

    x_lo, x_hi = dd[MEDIA_COL].quantile([0.05, 0.95])
    grid = np.linspace(x_lo, x_hi, N_GRID_POINTS)

    def predict_and_ci(x: np.ndarray, cp: float):
        pred = y_bar + b_media * (x - x_bar) + b_cp * (cp - cp_bar) + b_int * (x * cp - x_bar * cp_bar)
        a = x - x_bar
        b = np.full_like(x, cp - cp_bar)
        c = x * cp - x_bar * cp_bar
        grad = np.vstack([a, b, c])
        var = np.einsum("in,ij,jn->n", grad, cov3, grad)
        se = np.sqrt(var)
        return pred, pred - Z_90 * se, pred + Z_90 * se

    rows = []
    for cp_val, label in [(1, "cross_partisan"), (0, "single_bloc")]:
        pred, lo, hi = predict_and_ci(grid, cp_val)
        for xv, pv, lv, hv in zip(grid, pred, lo, hi):
            rows.append({
                "media_attention_level": float(xv),
                "predicted_log_oda": float(pv),
                "ci_lower": float(lv),
                "ci_upper": float(hv),
                "ci_level": SIGNIFICANCE_CI_LEVEL,
                "coverage_type": label,
            })
        log.info(f"Figure 7 data ({label}): predicted range [{pred[0]:.3f}, {pred[-1]:.3f}] "
                  f"across media attention [{grid[0]:.3f}, {grid[-1]:.3f}]")
    return rows


def main() -> None:
    d = build_panel()
    res, rows = run_regression(d)
    write_model_results(OUT_RESULTS, rows)
    log.info(f"Wrote {len(rows)} rows -> {OUT_RESULTS}")

    fig_rows = build_figure_data(d, res)
    write_rows(OUT_FIGDATA, FIGDATA_FIELDS, fig_rows)
    log.info(f"Wrote {len(fig_rows)} rows -> {OUT_FIGDATA}")

    # RESOLVED 2026-09-20 (final reconciliation pass): the k=1-13 vs. k=2-13
    # window mismatch this script used to log as an open discrepancy was
    # corrected in code/lib/ideology_panel.py (START_K now 2, was 1). No
    # discrepancy remains to log -- see that module's WINDOW NOTE.


if __name__ == "__main__":
    main()
