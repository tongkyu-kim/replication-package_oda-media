"""
Figure 4: sustained attention across minimum streak lengths of 1-60 months,
one regression per threshold, as in sustained_coverage.py (90% confidence
intervals).

Output: results/figures/figure4_streak_threshold_profile.png,
        results/figure_data/figure4_streak_threshold_data.csv
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
    plt.close(fig)
    log.info(f"Wrote {OUT_PNG}")


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
