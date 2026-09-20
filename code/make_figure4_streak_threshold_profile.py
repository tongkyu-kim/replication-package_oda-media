"""
Figure 4 -- Streak-length threshold profile of sustained media coverage.

Extends the study's H3 (sustained-coverage) specification --
results/models/sustained_coverage_results.csv, three separate binary-
threshold regressions at >=24/36/60 consecutive covered months -- to a
DENSE grid of every integer minimum streak length from 1 to 60 months, one
separate single-regressor regression per threshold (60 regressions, not
mutually adjusted, matching H3's own three-separate-specifications
design): devrel_commitment_musd ~ 1(consecutive_periods_covered >= L) +
controls.

Ported from the original research repository's
pipeline/devrel_baseline/estimate_streak_threshold_profile.py (the DATA-
computation script; its own streak_threshold_profile_results.csv is a
dense L=1..60 companion to H3's headline streak_ge24/36/60 specs) and
pipeline/manuscript_tables/make_figure5_streak_threshold_profile.py (old
filename -- "figure5" predates a manuscript renumbering; this is now
Figure 4) for the plotting logic. consecutive_periods_covered itself
(lib/devrel_panel.py's attach_streak(), a month covers if it has >=3
devrel-relevant articles, streak resets on any month below that bar,
evaluated as of November of year Y-1) is exactly the same column H3's
three headline specifications already use -- this script only re-
thresholds it at every integer L instead of only 24/36/60, so there is no
new panel construction here, only a denser sweep over the existing
measure.

Cross-check against results/models/sustained_coverage_results.csv (already
in this package): at L=24 this script reproduces coef=0.21547085839148536,
se=0.08894733233353389 exactly; L=36 coef=0.22850879161311385; L=60
coef=0.20190540590658285 -- all three match the original repository's own
dense-grid CSV (streak_threshold_profile_results.csv) to full float
precision, confirming the port. n is constant at every threshold (the
streak column is never missing, so which countries/years enter the
complete-case sample never changes with L) -- also verified against the
original's own dense-grid CSV, where n=2214 at every one of the 60
thresholds.

Manuscript text: "Figure 4 shows this duration profile across the full
range of minimum consecutive-coverage lengths (1 to 60 months)." Unlike
the original journal-quality figure (which drops L=1-5 because a streak
that short is satisfied by nearly every country-year and is not
comparable to the headline thresholds), this script keeps the FULL 1-60
range in both the figure-data CSV and the plot, per that manuscript
sentence.

CI convention: 90% (Z_90, this package's standard, see code/_common.py).

Writes: results/figure_data/figure4_streak_threshold_data.csv (one row per
threshold, 1..60) and results/figures/figure4_streak_threshold_profile.png(.pdf).
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

log = get_logger("make_figure4_streak_threshold_profile")

OUT_FIGDATA = RESULTS_FIGURE_DATA / "figure4_streak_threshold_data.csv"
OUT_PNG = RESULTS_FIGURES / "figure4_streak_threshold_profile.png"
OUT_PDF = RESULTS_FIGURES / "figure4_streak_threshold_profile.pdf"

FIGDATA_FIELDS = ["threshold", "coefficient", "se", "ci_lower", "ci_upper", "ci_level", "p_value", "n", "n_entities"]

THRESHOLDS = range(1, 61)  # minimum consecutive covered months, 1..60 inclusive


def build_figure_data() -> list[dict]:
    country_month_long = pw.build_country_month_long()
    sentiment_month = dp.build_country_sentiment_month()
    annual = dp.build_country_annual(use_log=True, country_month_long=country_month_long,
                                      sentiment_month=sentiment_month)

    rows = []
    for L in THRESHOLDS:
        col = f"streak_ge{L}"
        d = annual.copy()
        d[col] = (d["consecutive_periods_covered"] >= L).astype(float)
        xs = [col] + dp.CONTROL_COLS
        res = fit_fe_ols(d, "devrel_commitment_musd", xs, verbose_label=f"streak>={L}")
        if res is None:
            continue
        c = float(res.params[col])
        se = float(res.std_errors[col])
        p = float(res.pvalues[col])
        dd = d.dropna(subset=["devrel_commitment_musd"] + xs)
        rows.append({
            "threshold": L,
            "coefficient": c,
            "se": se,
            "ci_lower": c - Z_90 * se,
            "ci_upper": c + Z_90 * se,
            "ci_level": SIGNIFICANCE_CI_LEVEL,
            "p_value": p,
            "n": int(res.nobs),
            "n_entities": int(dd["entity"].nunique()),
        })
        log.info(f"[streak>={L:3d}mo] coef={c:.5f} se={se:.5f} p={p:.4f} n={int(res.nobs)}")
    return rows


def plot_figure(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    x = df["threshold"].to_numpy()
    coef = df["coefficient"].to_numpy()
    ci_lo = df["ci_lower"].to_numpy()
    ci_hi = df["ci_upper"].to_numpy()

    ax.fill_between(x, ci_lo, ci_hi, color="#8C8C8C", alpha=0.2, linewidth=0, zorder=1)
    ax.plot(x, coef, color="black", lw=0.9, zorder=2)

    mark = (x % 6 == 0)
    ax.scatter(x[mark], coef[mark], color="black", marker="o", s=18, zorder=3, edgecolors="none")

    for L in (24, 36, 60):
        if L in x:
            i = int(np.where(x == L)[0][0])
            ax.scatter([x[i]], [coef[i]], color="#8C1515", marker="D", s=34, zorder=4,
                       edgecolors="black", linewidths=0.5)

    ax.axhline(0, color="black", lw=0.7, ls="-", zorder=1)
    ax.set_xlim(0, 61)
    ax.set_xticks(np.arange(0, 61, 6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)
    ax.set_xlabel("Minimum consecutive-coverage length (months)")
    ax.set_ylabel("Coefficient on sustained media coverage (90% CI)")
    ax.set_title("Streak-length threshold profile of sustained media coverage", loc="left")

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

    for L in (24, 36, 60):
        r = next((r for r in rows if r["threshold"] == L), None)
        if r:
            log.info(f"Cross-check L={L}: coef={r['coefficient']:.4f} "
                      f"(sustained_coverage_results.csv should match)")

    df = pd.read_csv(OUT_FIGDATA, encoding="utf-8-sig")
    plot_figure(df)


if __name__ == "__main__":
    main()
