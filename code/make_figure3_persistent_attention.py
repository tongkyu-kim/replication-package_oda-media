"""
Figure 3 -- Persistent media attention across rolling exposure horizons.

Extends the study's H2 (persistent-attention) specification --
results/models/rolling_persistence_results.csv, three separate
regressions at rolling-average window widths of 24/36/60 months -- to a
DENSE grid of every window width from 1 to 60 months, one separate
single-regressor regression per width (60 regressions, not mutually
adjusted, matching H2's own three-separate-specifications design).

Ported from the original research repository's
pipeline/devrel_baseline/estimate_monthly_temporal_profile.py, Panel B
(the dense L=1..60 companion to the headline roll24/roll36/roll60
specifications, saved there as cumulative_window_profile_results.csv) --
same construction as H2's own devrel_media_roll24/36/60 measures (rolling
mean of the k=2..(L+1) window, log1p'd after averaging), just swept across
every width rather than only the three headline ones. This package
reproduces that dense sweep directly with lib/panel_windows.py's
attach_window() (the exact function -- "ported unchanged" -- the original
Panel B calls), rather than reading a cached CSV, so it needs no input
beyond this package's own analytical data.

Cross-check: at L=24/36/60 this reproduces H2's own coefficients exactly
(rolling_persistence_results.csv: roll24=0.2011, roll36=0.2031,
roll60=0.2072) -- verified against the original repository's own dense-grid
CSV (streak/window profile) before writing this script; not re-verified
programmatically here since results/models/rolling_persistence_results.csv
already carries the L=24/36/60 rows from the identical construction.

Manuscript text: "the right panel shows that the rolling-average estimates
remain positive as the averaging window expands." Full dense grid (no
windows dropped) is written to the figure-data CSV and plotted; markers
are drawn only at 6-month multiples to keep the line legible, matching the
original figure's convention, but the underlying line/CI band is drawn
from the complete L=1..60 series.

CI convention: 90% (Z_90, this package's standard, see code/_common.py).

Writes: results/figure_data/figure3_persistent_attention_data.csv (one row
per window width, L=1..60) and
results/figures/figure3_persistent_attention.png.
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

log = get_logger("make_figure3_persistent_attention")

OUT_FIGDATA = RESULTS_FIGURE_DATA / "figure3_persistent_attention_data.csv"
OUT_PNG = RESULTS_FIGURES / "figure3_persistent_attention.png"

FIGDATA_FIELDS = ["window_length_months", "coefficient", "se", "p_value",
                   "ci_lower", "ci_upper", "ci_level", "n", "n_entities"]

WINDOW_WIDTHS = range(1, 61)  # months, 1..60 inclusive


def build_figure_data() -> list[dict]:
    country_month_long = pw.build_country_month_long()
    sentiment_month = dp.build_country_sentiment_month()
    annual = dp.build_country_annual(use_log=True, country_month_long=country_month_long,
                                      sentiment_month=sentiment_month)
    min_ym, max_ym = pw.corpus_month_bounds(country_month_long)

    base_cols = ["entity", "year", "devrel_commitment_musd"] + dp.CONTROL_COLS
    base = annual[base_cols].copy()

    rows = []
    for L in WINDOW_WIDTHS:
        colname = f"devrel_media_roll{L}"
        # Same construction as devrel_panel.py's roll24/36/60 (k=2..(L+1),
        # width preserved at L months) -- attach_window is the exact
        # ported-unchanged helper every other script in this package uses.
        d = pw.attach_window(base.copy(), country_month_long, True, start_k=2, end_k=L + 1,
                              colname=colname, agg="mean", min_ym=min_ym, max_ym=max_ym)
        xs = [colname] + dp.CONTROL_COLS
        res = fit_fe_ols(d, "devrel_commitment_musd", xs, verbose_label=f"L={L}mo")
        if res is None:
            continue
        c = float(res.params[colname])
        se = float(res.std_errors[colname])
        p = float(res.pvalues[colname])
        dd = d.dropna(subset=["devrel_commitment_musd"] + xs)
        rows.append({
            "window_length_months": L,
            "coefficient": c,
            "se": se,
            "p_value": p,
            "ci_lower": c - Z_90 * se,
            "ci_upper": c + Z_90 * se,
            "ci_level": SIGNIFICANCE_CI_LEVEL,
            "n": int(res.nobs),
            "n_entities": int(dd["entity"].nunique()),
        })
        log.info(f"[L={L:3d}mo] coef={c:.5f} se={se:.5f} p={p:.4f} n={int(res.nobs)}")
    return rows


def plot_figure(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    x = df["window_length_months"].to_numpy()
    coef = df["coefficient"].to_numpy()
    ci_lo = df["ci_lower"].to_numpy()
    ci_hi = df["ci_upper"].to_numpy()

    ax.fill_between(x, ci_lo, ci_hi, color="#8C8C8C", alpha=0.2, linewidth=0, zorder=1)
    ax.plot(x, coef, color="black", lw=0.9, zorder=2)

    mark = (x % 6 == 0)
    ax.scatter(x[mark], coef[mark], color="black", marker="o", s=18, zorder=3, edgecolors="none")

    for L, marker_x in [(24, 24), (36, 36), (60, 60)]:
        if marker_x in x:
            i = int(np.where(x == marker_x)[0][0])
            ax.scatter([x[i]], [coef[i]], color="#8C1515", marker="D", s=34, zorder=4,
                       edgecolors="black", linewidths=0.5)

    ax.axhline(0, color="black", lw=0.7, ls="-", zorder=1)
    ax.set_xlim(0, 61)
    ax.set_xticks(np.arange(0, 61, 6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)
    ax.set_xlabel("Rolling-average window width (months)")
    ax.set_ylabel("Coefficient on rolling-average media attention (90% CI)")
    ax.set_title("Persistent media attention across rolling exposure horizons", loc="left")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Wrote {OUT_PNG}")


def main() -> None:
    rows = build_figure_data()
    write_rows(OUT_FIGDATA, FIGDATA_FIELDS, rows)
    log.info(f"Wrote {len(rows)} rows -> {OUT_FIGDATA}")

    for L in (24, 36, 60):
        r = next((r for r in rows if r["window_length_months"] == L), None)
        if r:
            log.info(f"Cross-check L={L}: coef={r['coefficient']:.4f} "
                      f"(rolling_persistence_results.csv should match)")

    df = pd.read_csv(OUT_FIGDATA, encoding="utf-8-sig")
    plot_figure(df)


if __name__ == "__main__":
    main()
