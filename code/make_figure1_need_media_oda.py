"""
Figure 1 -- Media attention, development need, and Korean ODA by recipient.

Purely DESCRIPTIVE (no regression, no significance test, no causal
language) -- one bubble per recipient country in media-attention x
development-need space, sized by average annual Korean ODA. Ported from
the original research repository's pipeline/make_figure_oda_attention_need.py
(data-computation logic only; that script's exact bubble-size legend inset
and Korean-name label lookup are not reproduced here -- see SIMPLIFICATIONS
below).

DEVELOPMENT-NEED COMPOSITE (country-year, then averaged to country level):
  development_need_it = mean( z(-ln_gdppc_it), z(ln_conflict_deaths_it),
                               z(ln_disaster_affected_it) ), skipna
  z(.) = standardized ACROSS THE FULL country-year analytical panel (every
  recipient, every year) using that variable's panel-wide mean/SD.
  ln_gdppc is negated before standardizing so higher score = lower income =
  greater need; the other two components are already "higher = more need"
  (conflict deaths / disaster-affected population). Equal weights (mean of
  3 z-scores) -- matches the manuscript's own description: "Development
  need is measured descriptively as an equally weighted composite of
  income, conflict, and disaster indicators." Country-level
  development_need = mean of the annual score across the years that
  country appears in the panel.

MEDIA ATTENTION: mean of the RAW (not logged) k=2-13 pre-allocation-window
article count (devrel_media_m1_13 computed with use_log=False, which is
exactly the original script's "current_media_raw" -- see
build_mechanism_panel.py in the original repo: both call
monthly_lags.attach_window(..., start_k=2, end_k=13, agg="sum",
use_log=False)) across a country's panel years, THEN log1p-transformed for
the x-axis (average-then-transform, not transform-then-average, matching
the original).

ODA: mean of devrel_commitment_musd (raw, million USD) across a country's
panel years.

TOP_N / ODA_COLOR_THRESHOLD: identical constants to the original (top 50
by average annual ODA highlighted/sized; countries above $10M average
annual ODA get the colored/sized bubble treatment, everyone else is a
plain small gray background dot).

SIMPLIFICATIONS relative to the original journal-quality script (this
package's figures are meant to be correct and legible, not
publication-polished -- see task brief):
  - Recipients are labeled by ISO3 code, not full English country name --
    the original's Korean/English name lookup (reference/countries_ko.py)
    is a project-internal reference file not part of this public package's
    data directory.
  - Quadrant reference lines: the original HARDCODES x_ref=4.0 (its own
    docstring explains this is a "clean round number" chosen because it
    was indistinguishable from that run's actual country-level median of
    4.038 -- not a principled zero). Rather than carry over a number
    tuned to a data snapshot this package does not necessarily reproduce
    exactly, this script computes x_ref as the ACTUAL country-level
    median of avg_annual_media_log in this package's own panel every time
    it runs. y_ref=0.0 is preserved unchanged -- that one IS principled:
    each need component is z-scored across the full country-year panel,
    so 0 is that panel's true average need by construction, not a tuned
    constant.
  - No bubble-size legend inset / no adjustText label-collision avoidance
    -- plain scatter with labels on the largest recipients only.

Writes: results/figure_data/figure1_need_media_oda_data.csv (one row per
country) and results/figures/figure1_need_media_oda.png(.pdf).
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

from _common import RESULTS_FIGURES, RESULTS_FIGURE_DATA, get_logger, write_rows
from lib import devrel_panel as dp
from lib import panel_windows as pw

log = get_logger("make_figure1_need_media_oda")

OUT_FIGDATA = RESULTS_FIGURE_DATA / "figure1_need_media_oda_data.csv"
OUT_PNG = RESULTS_FIGURES / "figure1_need_media_oda.png"
OUT_PDF = RESULTS_FIGURES / "figure1_need_media_oda.pdf"

FIGDATA_FIELDS = ["country_iso3", "oda_rank", "top50", "quadrant", "avg_annual_oda",
                   "avg_annual_media_raw", "avg_annual_media_log", "development_need",
                   "gdppc_component", "conflict_component", "disaster_component", "n_years"]

TOP_N = 50
ODA_COLOR_THRESHOLD = 10.0  # $M -- only recipients above this are colored/sized on the plot
LABEL_ODA_THRESHOLD = 50.0  # $M -- only recipients above this get a text label


def build_country_year() -> pd.DataFrame:
    """Country-year panel with raw ODA, raw k=2-13 media attention, and the
    3 need components -- built from this package's own analytical data via
    the same shared lib/devrel_panel.py + lib/panel_windows.py helpers
    every other script in this package uses (use_log=False reproduces the
    original's "current_media_raw" / raw devrel_commitment_musd exactly:
    both are the k=2-13 sum / unlogged commitment before any log1p)."""
    country_month_long = pw.build_country_month_long()
    sentiment_month = dp.build_country_sentiment_month()
    d = dp.build_country_annual(use_log=False, country_month_long=country_month_long,
                                 sentiment_month=sentiment_month)

    for col in ["ln_gdppc", "ln_conflict_deaths", "ln_disaster_affected"]:
        mu, sd = d[col].mean(), d[col].std()
        d[f"z_{col}"] = (d[col] - mu) / sd
    d["z_need_gdppc"] = -d["z_ln_gdppc"]
    d["z_need_conflict"] = d["z_ln_conflict_deaths"]
    d["z_need_disaster"] = d["z_ln_disaster_affected"]
    d["development_need"] = d[["z_need_gdppc", "z_need_conflict", "z_need_disaster"]].mean(axis=1, skipna=True)
    return d


def build_country_level(d: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    g = d.groupby("country_iso3").agg(
        avg_annual_oda=("devrel_commitment_musd", "mean"),
        avg_annual_media_raw=("devrel_media_m1_13", "mean"),
        development_need=("development_need", "mean"),
        gdppc_component=("z_need_gdppc", "mean"),
        conflict_component=("z_need_conflict", "mean"),
        disaster_component=("z_need_disaster", "mean"),
        n_years=("year", "count"),
    ).reset_index()
    g["avg_annual_media_log"] = np.log1p(g["avg_annual_media_raw"])

    g = g.sort_values("avg_annual_oda", ascending=False).reset_index(drop=True)
    g["oda_rank"] = np.arange(1, len(g) + 1)
    g["top50"] = g["oda_rank"] <= TOP_N

    y_ref = 0.0  # principled: 0 is the panel-wide standardized mean by construction
    x_ref = float(g["avg_annual_media_log"].median())  # see SIMPLIFICATIONS in the module docstring
    log.info(f"Reference lines: x={x_ref:.3f} (country-level median), y={y_ref} (panel-wide standardized mean)")

    g["quadrant"] = np.select(
        [
            (g["avg_annual_media_log"] >= x_ref) & (g["development_need"] >= y_ref),
            (g["avg_annual_media_log"] >= x_ref) & (g["development_need"] < y_ref),
            (g["avg_annual_media_log"] < x_ref) & (g["development_need"] >= y_ref),
        ],
        ["High attention / High need", "High attention / Low need", "Low attention / High need"],
        default="Low attention / Low need",
    )
    return g, x_ref, y_ref


def plot_figure(g: pd.DataFrame, x_ref: float, y_ref: float) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.6))

    bg = g[g["avg_annual_oda"] <= ODA_COLOR_THRESHOLD]
    top = g[g["avg_annual_oda"] > ODA_COLOR_THRESHOLD].copy()

    ax.scatter(bg["avg_annual_media_log"], bg["development_need"], s=12, color="#BFBFBF",
               edgecolors="none", alpha=0.85, zorder=2, label=f"<= ${ODA_COLOR_THRESHOLD:.0f}M avg. annual ODA")

    if not top.empty:
        oda_min, oda_max = top["avg_annual_oda"].min(), top["avg_annual_oda"].max()
        span = max(oda_max - oda_min, 1e-9)
        sizes = 20.0 + (top["avg_annual_oda"] - oda_min) / span * 700.0
        ax.scatter(top["avg_annual_media_log"], top["development_need"], s=sizes, color="#8C1515",
                   edgecolors="black", linewidths=0.6, alpha=0.65, zorder=3,
                   label=f"> ${ODA_COLOR_THRESHOLD:.0f}M avg. annual ODA (bubble area ~ ODA)")

        labeled = top[top["avg_annual_oda"] > LABEL_ODA_THRESHOLD]
        for _, r in labeled.iterrows():
            ax.annotate(r["country_iso3"], (r["avg_annual_media_log"], r["development_need"]),
                        fontsize=7.4, color="black", xytext=(3, 3), textcoords="offset points", zorder=5)

    ax.axvline(x_ref, color="#808080", lw=0.9, ls="--", zorder=1)
    ax.axhline(y_ref, color="#808080", lw=0.9, ls="--", zorder=1)

    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    pad_x, pad_y = (xhi - xlo) * 0.015, (yhi - ylo) * 0.025
    quad_kw = dict(fontsize=7.3, color="#808080", style="italic")
    ax.text(xhi - pad_x, yhi - pad_y, "High attention / High need", ha="right", va="top", **quad_kw)
    ax.text(xhi - pad_x, ylo + pad_y, "High attention / Low need", ha="right", va="bottom", **quad_kw)
    ax.text(xlo + pad_x, yhi - pad_y, "Low attention / High need", ha="left", va="top", **quad_kw)
    ax.text(xlo + pad_x, ylo + pad_y, "Low attention / Low need", ha="left", va="bottom", **quad_kw)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Media attention (log avg. annual development-related articles)")
    ax.set_ylabel("Development need (composite, standardized)")
    ax.set_title("Media attention, development need, and Korean ODA by recipient", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=7.8, frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Wrote {OUT_PNG}")
    log.info(f"Wrote {OUT_PDF}")


def main() -> None:
    cy = build_country_year()
    g, x_ref, y_ref = build_country_level(cy)
    log.info(f"Country-level sample: {len(g)} recipients, {int(g['top50'].sum())} in top {TOP_N} by avg annual ODA")

    write_rows(OUT_FIGDATA, FIGDATA_FIELDS, g[FIGDATA_FIELDS].to_dict("records"))
    log.info(f"Wrote {len(g)} rows -> {OUT_FIGDATA}")

    # Read back from disk before plotting, per this package's convention that
    # every figure is drawn from its own already-written figure-data file.
    g_plot = pd.read_csv(OUT_FIGDATA, encoding="utf-8-sig")
    plot_figure(g_plot, x_ref, y_ref)


if __name__ == "__main__":
    main()
