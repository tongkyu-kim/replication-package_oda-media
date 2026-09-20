"""
Figure 2 -- Monthly temporal profile of development-related media attention.

50 separate single-month regressions, t=-37 (37 months before the
allocation year) through t=+12 (12 months after), each a bivariate
single-regressor specification: devrel_commitment_musd ~ media_month +
controls, entity+year FE, country-clustered SE. NOT an event study, NOT a
joint distributed lag -- each point is estimated independently, exactly as
the original repository's pipeline/devrel_baseline/
estimate_monthly_temporal_profile.py (Panel A) and plotted by
pipeline/manuscript_tables/make_figure3_monthly_temporal_profile.py (old
filename -- "figure3" predates a manuscript renumbering; this is now
Figure 2). t=-37..-2 spans exactly the pre-allocation range covered by
this study's three lag-window specifications (k=2-13/14-25/26-37, see
code/lag_windows.py); t=-1..+12 is a post-allocation placebo/falsification
extension -- does media attention AFTER the allocation decision also
"predict" it, which would undercut the pre-allocation timing story.

Sign convention (matches the original exactly): codebase_k = -relative_month,
i.e. relative_month is standard event-time (positive = after t0) while k is
"months before Jan 1 of year t" as used throughout lib/panel_windows.py.

Manuscript text: "the left panel shows month-specific associations are
generally positive across a broad pre-allocation period."

CI convention: this package's figures use a 90% CI throughout (Z_90 =
1.6448536269514722, see code/_common.py), recomputed here from each
month's own coefficient/SE -- the same convention every other figure in
this package uses, whatever CI multiplier a given script in the original
research repository happened to use for its own intermediate CSV.

Writes: results/figure_data/figure2_monthly_profile_data.csv (one row per
relative month) and results/figures/figure2_monthly_temporal_profile.png(.pdf).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import (RESULTS_FIGURES, RESULTS_FIGURE_DATA, SIGNIFICANCE_CI_LEVEL, Z_90,
                      get_logger, write_rows)
from lib import devrel_panel as dp
from lib import panel_windows as pw
from lib.regressions import fit_fe_ols

log = get_logger("make_figure2_monthly_temporal_profile")

OUT_FIGDATA = RESULTS_FIGURE_DATA / "figure2_monthly_profile_data.csv"
OUT_PNG = RESULTS_FIGURES / "figure2_monthly_temporal_profile.png"
OUT_PDF = RESULTS_FIGURES / "figure2_monthly_temporal_profile.pdf"

FIGDATA_FIELDS = ["relative_month", "codebase_k", "coefficient", "se", "p_value",
                   "ci_lower", "ci_upper", "ci_level", "n", "n_entities"]

REL_MONTH_RANGE = range(-37, 13)  # t=-37 .. t=+12 inclusive, matches the original


def build_figure_data() -> list[dict]:
    country_month_long = pw.build_country_month_long()
    sentiment_month = dp.build_country_sentiment_month()
    annual = dp.build_country_annual(use_log=True, country_month_long=country_month_long,
                                      sentiment_month=sentiment_month)
    min_ym, max_ym = pw.corpus_month_bounds(country_month_long)

    base_cols = ["entity", "year", "devrel_commitment_musd"] + dp.CONTROL_COLS
    base = annual[base_cols].copy()

    rows = []
    for rel_month in REL_MONTH_RANGE:
        k = -rel_month
        raw = pw.attach_month_lag(annual, country_month_long, k, min_ym, max_ym)
        d = base.copy()
        d["media_month"] = np.log1p(raw.clip(lower=0))
        xs = ["media_month"] + dp.CONTROL_COLS
        res = fit_fe_ols(d, "devrel_commitment_musd", xs, verbose_label=f"t={rel_month:+d}")
        if res is None:
            continue
        c = float(res.params["media_month"])
        se = float(res.std_errors["media_month"])
        p = float(res.pvalues["media_month"])
        dd = d.dropna(subset=["devrel_commitment_musd"] + xs)
        rows.append({
            "relative_month": rel_month,
            "codebase_k": k,
            "coefficient": c,
            "se": se,
            "p_value": p,
            "ci_lower": c - Z_90 * se,
            "ci_upper": c + Z_90 * se,
            "ci_level": SIGNIFICANCE_CI_LEVEL,
            "n": int(res.nobs),
            "n_entities": int(dd["entity"].nunique()),
        })
        log.info(f"[t={rel_month:+3d}] coef={c:.5f} se={se:.5f} p={p:.4f} n={int(res.nobs)}")
    return rows


def plot_figure(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    x = df["relative_month"].to_numpy()
    coef = df["coefficient"].to_numpy()
    ci_lo = df["ci_lower"].to_numpy()
    ci_hi = df["ci_upper"].to_numpy()

    ax.fill_between(x, ci_lo, ci_hi, color="#8C8C8C", alpha=0.2, linewidth=0, zorder=1)
    ax.plot(x, coef, color="black", lw=0.9, zorder=2)
    ax.scatter(x, coef, color="black", marker="o", s=16, zorder=3, edgecolors="none")

    ax.axvline(0, color="#8C1515", lw=1.1, ls=":", zorder=1.5)
    ax.axhline(0, color="black", lw=0.7, ls="-", zorder=1)

    ax.set_xlim(x.min() - 1, x.max() + 1)
    ax.set_xticks(np.arange(-36, 13, 6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)
    ax.set_xlabel("Relative month (t = 0 is the ODA allocation year boundary)")
    ax.set_ylabel("Coefficient on single-month media attention (90% CI)")
    ax.set_title("Monthly temporal profile of development-related media attention", loc="left")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Wrote {OUT_PNG}")
    log.info(f"Wrote {OUT_PDF}")


def main() -> None:
    rows = build_figure_data()
    write_rows(OUT_FIGDATA, FIGDATA_FIELDS, rows)
    log.info(f"Wrote {len(rows)} rows -> {OUT_FIGDATA}")

    df = pd.read_csv(OUT_FIGDATA, encoding="utf-8-sig")
    plot_figure(df)


if __name__ == "__main__":
    main()
