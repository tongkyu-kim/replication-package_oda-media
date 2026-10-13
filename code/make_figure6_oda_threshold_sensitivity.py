"""
Figure 6: media coefficient across minimum ODA thresholds. For each threshold
(percentiles 1-99 of positive annual country-level ODA), a linear probability
model for ODA at or above the threshold is fitted on the zero-filled
country x year panel of allocation_margin.py (2-13-month media, country and
year FE, five controls, country-clustered standard errors; 90% confidence
intervals).

Output: results/figures/figure6_oda_threshold_sensitivity.png,
        results/figure_data/figure6_oda_threshold_data.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from _common import (RESULTS_FIGURES, RESULTS_FIGURE_DATA, SIGNIFICANCE_CI_LEVEL, Z_90,  # noqa: E402
                      get_logger, write_rows)
from lib import panel_windows as pw  # noqa: E402
from lib import devrel_panel as dp  # noqa: E402
from lib.regressions import fit_fe_ols  # noqa: E402
from allocation_margin import build_zero_filled_panel, MEDIA_COL  # noqa: E402

log = get_logger("make_figure6_oda_threshold_sensitivity")

OUT_FIGDATA = RESULTS_FIGURE_DATA / "figure6_oda_threshold_data.csv"
OUT_PNG = RESULTS_FIGURES / "figure6_oda_threshold_sensitivity.png"

FIGDATA_FIELDS = ["percentile", "threshold_usd", "coefficient", "se", "ci_lower", "ci_upper",
                   "ci_level", "p_value", "n"]

PERCENTILES = list(range(1, 100))  # p1..p99 of the empirical positive-ODA distribution


def run_one(d0: pd.DataFrame, thr_musd: float) -> dict | None:
    d = d0.copy()
    d["oda_above_thr"] = (d["devrel_commitment_musd"] >= thr_musd).astype(float)
    res = fit_fe_ols(d, "oda_above_thr", [MEDIA_COL] + dp.CONTROL_COLS,
                      verbose_label=f"threshold sensitivity (thr={thr_musd:.5f} MUSD)")
    if res is None:
        return None
    coef = float(res.params[MEDIA_COL])
    se = float(res.std_errors[MEDIA_COL])
    p = float(res.pvalues[MEDIA_COL])
    return {"coefficient": coef, "se": se, "p_value": p, "n": int(res.nobs),
            "ci_lower": coef - Z_90 * se, "ci_upper": coef + Z_90 * se}


def build_figure_data(d0: pd.DataFrame) -> list[dict]:
    positive = d0.loc[d0["devrel_commitment_musd"] > 0, "devrel_commitment_musd"]
    log.info(f"n_positive={len(positive)} country-years for the empirical threshold grid")
    rows = []
    for pct in PERCENTILES:
        thr_musd = float(positive.quantile(pct / 100))
        r = run_one(d0, thr_musd)
        if r is None:
            log.warning(f"p{pct}: insufficient data, skipping")
            continue
        rows.append({"percentile": pct, "threshold_usd": thr_musd * 1_000_000,
                      "ci_level": SIGNIFICANCE_CI_LEVEL, **r})
        log.info(f"[p{pct:>2d}, thr=${thr_musd * 1e6:,.0f}] coef={r['coefficient']:.5f} se={r['se']:.5f} "
                  f"p={r['p_value']:.4f} n={r['n']}")
    return rows


def make_plot(fig_df: pd.DataFrame) -> None:
    fig_df = fig_df.sort_values("threshold_usd")
    x = fig_df["threshold_usd"].to_numpy()
    coef = fig_df["coefficient"].to_numpy()
    lo = fig_df["ci_lower"].to_numpy()
    hi = fig_df["ci_upper"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.fill_between(x, lo, hi, color="gray", alpha=0.25, linewidth=0, label="90% CI", zorder=1)
    ax.plot(x, coef, color="black", lw=1.4, label="Media coefficient", zorder=3)
    ax.axhline(0, color="black", lw=0.8, ls="--", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("Minimum annual ODA threshold, $ (log scale)")
    ax.set_ylabel("Coefficient on development-related media\nattention (extensive margin, LPM)")
    ax.set_title("Fig. 6: Sensitivity of the media coefficient across minimum ODA thresholds")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    country_month_long = pw.build_country_month_long()
    d0 = build_zero_filled_panel(country_month_long)
    log.info(f"Zero-filled panel: {len(d0):,} rows ({d0['country_iso3'].nunique()} countries), "
              f"{d0['oda_positive'].sum():,.0f} positive-ODA rows ({d0['oda_positive'].mean():.1%})")

    rows = build_figure_data(d0)
    write_rows(OUT_FIGDATA, FIGDATA_FIELDS, rows)
    log.info(f"Wrote {len(rows)} rows -> {OUT_FIGDATA}")

    fig_df = pd.read_csv(OUT_FIGDATA, encoding="utf-8-sig")
    make_plot(fig_df)
    log.info(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
