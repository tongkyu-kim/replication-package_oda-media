"""
Figure 8 -- Temporal and substantive placebo tests of the media-ODA
association.

Data: results/models/placebo_nondevrel_results.csv (14 rows: 7 devrel/
sports pairs -- the SUBSTANTIVE placebo, does non-devrel sports Korean
media coverage "predict" devrel ODA the same way devrel-specific coverage
does, on the identical matched sample for each of the 7 headline attention
measures) and results/models/placebo_future_news_results.csv (2 rows --
the TEMPORAL placebo, does unexpected FUTURE media coverage "predict"
this year's devrel ODA, raw and AR(2)-innovation variants). Both files are
already-complete output of code/placebo_tests.py; no new estimation
happens here.

Layout ported (visual design only) from the original research repository's
pipeline/manuscript_tables/make_figure_robustness_placebos.py: point + 90%
CI whiskers, no connecting lines. Substantive/real = solid whisker + filled
circle; placebo = dashed whisker + hollow square. No color coding --
real-vs-placebo is carried by line style and marker shape alone.

SCOPE NOTE (this package differs from the original repository's figure):
the original figure's temporal panel compared THREE pre-allocation windows
(2-13/14-25/26-37mo) each against its own future-media placebo variant.
This package's placebo_tests.py only ports the 2-13mo window's future-news
placebo (raw future_news_t1 and its AR(2) innovation) -- the 14-25mo/
26-37mo mirror-window placebos were not ported (not present in
placebo_future_news_results.csv). This figure's temporal panel is
therefore SCOPE-LIMITED to the 2-13mo window only, shown against both of
its future-news placebo variants (raw and AR(2) innovation); the "real"
2-13mo devrel estimate is the SAME estimate in both pairs (it is also the
first entry of the substantive-placebo panel,
placebo_nondevrel_m1_13_devrel, which matches code/main_baseline.py's
H1_preferred__devrel_media_m1_13 to full precision since both fit the
identical spec on the identical sample).

Writes: results/figure_data/figure8_placebo_battery_data.csv and
results/figures/figure8_placebo_battery.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from _common import RESULTS_MODELS, RESULTS_FIGURES, RESULTS_FIGURE_DATA, get_logger, write_rows  # noqa: E402

log = get_logger("make_figure8_placebo_battery")

IN_NONDEVREL = RESULTS_MODELS / "placebo_nondevrel_results.csv"
IN_FUTURE_NEWS = RESULTS_MODELS / "placebo_future_news_results.csv"
OUT_FIGDATA = RESULTS_FIGURE_DATA / "figure8_placebo_battery_data.csv"
OUT_PNG = RESULTS_FIGURES / "figure8_placebo_battery.png"

FIGDATA_FIELDS = ["panel", "measure", "measure_order", "measure_label", "series", "placebo_type",
                   "coefficient", "se", "ci_lower", "ci_upper", "ci_level", "p_value", "n"]

# ---- Panel A: substantive (devrel vs. non-devrel sports media) placebo ----
NONDEVREL_MEASURES = [
    ("m1_13", "2-13 mo\nattention"), ("roll24", "Rolling\n24-mo"), ("roll36", "Rolling\n36-mo"),
    ("roll60", "Rolling\n60-mo"), ("streak24", "Streak\n>=24mo"), ("streak36", "Streak\n>=36mo"),
    ("streak60", "Streak\n>=60mo"),
]

# ---- Panel B: temporal (future-news) placebo, 2-13mo window only (see SCOPE NOTE) ----
FUTURE_NEWS_MEASURES = [
    ("future_news_raw", "Future news\n(raw, t+1)"),
    ("future_news_ar2_innovation", "Future news\n(AR(2) innovation)"),
]


def _row_to_dict(panel: str, measure: str, order: int, label: str, series: str, placebo_type: str,
                  r: pd.Series) -> dict:
    return {
        "panel": panel, "measure": measure, "measure_order": order, "measure_label": label, "series": series,
        "placebo_type": placebo_type, "coefficient": float(r["coefficient"]), "se": float(r["se"]),
        "ci_lower": float(r["ci_lower"]), "ci_upper": float(r["ci_upper"]), "ci_level": float(r["ci_level"]),
        "p_value": float(r["p_value"]), "n": int(r["n"]),
    }


def build_figure_data(df_nondevrel: pd.DataFrame, df_future: pd.DataFrame) -> list[dict]:
    rows = []
    for i, (measure, label) in enumerate(NONDEVREL_MEASURES):
        real = df_nondevrel[df_nondevrel["specification_id"] == f"placebo_nondevrel_{measure}_devrel"].iloc[0]
        alln = df_nondevrel[df_nondevrel["specification_id"] == f"placebo_nondevrel_{measure}_allnews"].iloc[0]
        pla = df_nondevrel[df_nondevrel["specification_id"] == f"placebo_nondevrel_{measure}_generic"].iloc[0]
        rows.append(_row_to_dict("nondevrel", measure, i, label, "real", "devrel media (headline measure)", real))
        rows.append(_row_to_dict("nondevrel", measure, i, label, "allnews",
                                  "all news mentioning the country (total visibility)", alln))
        rows.append(_row_to_dict("nondevrel", measure, i, label, "placebo",
                                  "sports media (non-development placebo)", pla))

    real_m1_13 = df_nondevrel[df_nondevrel["specification_id"] == "placebo_nondevrel_m1_13_devrel"].iloc[0]
    future_specs = {
        "future_news_raw": "placebo_future_news_raw__future_news_t1",
        "future_news_ar2_innovation": "placebo_future_news_ar2_innovation__innovation_ar2",
    }
    placebo_labels = {
        "future_news_raw": "future-news placebo (raw, calendar year t+1)",
        "future_news_ar2_innovation": "future-news placebo (AR(2) innovation)",
    }
    for i, (measure, label) in enumerate(FUTURE_NEWS_MEASURES):
        pla = df_future[df_future["specification_id"] == future_specs[measure]].iloc[0]
        rows.append(_row_to_dict("temporal", measure, i, label, "real",
                                  "devrel media, 2-13 months pre-allocation (real, H1 headline window)",
                                  real_m1_13))
        rows.append(_row_to_dict("temporal", measure, i, label, "placebo", placebo_labels[measure], pla))
    return rows


def draw_pair(ax, x: float, real_val, real_lo, real_hi, pla_val, pla_lo, pla_hi, offset=0.14, cap=0.05):
    xr, xp = x - offset, x + offset
    ax.vlines(xr, real_lo, real_hi, color="black", lw=1.2, linestyles="solid", zorder=3)
    ax.hlines([real_lo, real_hi], xr - cap, xr + cap, color="black", lw=1.2, zorder=3)
    ax.vlines(xp, pla_lo, pla_hi, color="black", lw=1.2, linestyles=(0, (1.5, 1)), zorder=3)
    ax.hlines([pla_lo, pla_hi], xp - cap, xp + cap, color="black", lw=1.2, zorder=3)
    ax.scatter(xr, real_val, facecolors="black", edgecolors="black", marker="o", s=34, zorder=4)
    ax.scatter(xp, pla_val, facecolors="white", edgecolors="black", marker="s", s=34, linewidths=1.1, zorder=4)


def draw_triple(ax, x: float, real, alln, pla, offset=0.22, cap=0.045):
    """devrel (solid, filled circle) | all news (dash-dot, open triangle) | sports (dotted, open square)."""
    for xi, r, ls, marker, face in [(x - offset, real, "solid", "o", "black"),
                                    (x, alln, (0, (4, 1.5, 1, 1.5)), "^", "white"),
                                    (x + offset, pla, (0, (1.5, 1)), "s", "white")]:
        ax.vlines(xi, r["ci_lower"], r["ci_upper"], color="black", lw=1.2, linestyles=ls, zorder=3)
        ax.hlines([r["ci_lower"], r["ci_upper"]], xi - cap, xi + cap, color="black", lw=1.2, zorder=3)
        ax.scatter(xi, r["coefficient"], facecolors=face, edgecolors="black", marker=marker, s=34,
                   linewidths=1.1, zorder=4)


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def make_plot(fig_df: pd.DataFrame) -> None:
    from matplotlib.lines import Line2D

    fig, (axB, axA) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [2, 4]})

    # Panel A -- substantive (devrel vs. nondevrel) placebo
    for i, (measure, label) in enumerate(NONDEVREL_MEASURES):
        sel = fig_df[(fig_df["panel"] == "nondevrel") & (fig_df["measure"] == measure)].set_index("series")
        draw_triple(axA, i, sel.loc["real"], sel.loc["allnews"], sel.loc["placebo"])
    axA.axhline(0, color="black", lw=0.7, ls=(0, (1.5, 1)), zorder=2)
    axA.set_xticks(range(len(NONDEVREL_MEASURES)))
    axA.set_xticklabels([lbl for _, lbl in NONDEVREL_MEASURES], fontsize=8.5)
    axA.set_xlim(-0.5, len(NONDEVREL_MEASURES) - 0.5)
    style_axes(axA)
    axA.set_ylabel("Coefficient on media attention")
    axA.set_title("News-substance placebo: development-relevant vs.\nall country news vs. sports news")
    handles_a = [
        Line2D([0], [0], color="none", marker="o", markerfacecolor="black", markeredgecolor="black",
               markersize=6, label="DevRel media"),
        Line2D([0], [0], color="none", marker="^", markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=1.1, markersize=6, label="All country news"),
        Line2D([0], [0], color="none", marker="s", markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=1.1, markersize=6, label="Sports news"),
    ]
    axA.legend(handles=handles_a, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=8)

    # Panel B -- temporal (future-news) placebo
    for i, (measure, label) in enumerate(FUTURE_NEWS_MEASURES):
        real = fig_df[(fig_df["panel"] == "temporal") & (fig_df["measure"] == measure) & (fig_df["series"] == "real")].iloc[0]
        pla = fig_df[(fig_df["panel"] == "temporal") & (fig_df["measure"] == measure) & (fig_df["series"] == "placebo")].iloc[0]
        draw_pair(axB, i, real["coefficient"], real["ci_lower"], real["ci_upper"],
                  pla["coefficient"], pla["ci_lower"], pla["ci_upper"])
    axB.axhline(0, color="black", lw=0.7, ls=(0, (1.5, 1)), zorder=2)
    axB.set_xticks(range(len(FUTURE_NEWS_MEASURES)))
    axB.set_xticklabels([lbl for _, lbl in FUTURE_NEWS_MEASURES], fontsize=8.5)
    axB.set_xlim(-0.5, len(FUTURE_NEWS_MEASURES) - 0.5)
    style_axes(axB)
    axB.set_ylabel("Coefficient on media attention")
    axB.set_title("Temporal placebo: 2-13mo pre-allocation\nmedia vs. future-news media")
    handles_b = [
        Line2D([0], [0], color="none", marker="o", markerfacecolor="black", markeredgecolor="black",
               markersize=6, label="Pre-allocation media, 2-13mo (real)"),
        Line2D([0], [0], color="none", marker="s", markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=1.1, markersize=6, label="Future-news media (placebo)"),
    ]
    axB.legend(handles=handles_b, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False, fontsize=8)

    fig.suptitle("Fig. 8: Temporal and substantive placebo tests of the media-ODA association", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df_nondevrel = pd.read_csv(IN_NONDEVREL, encoding="utf-8-sig")
    df_future = pd.read_csv(IN_FUTURE_NEWS, encoding="utf-8-sig")

    rows = build_figure_data(df_nondevrel, df_future)
    write_rows(OUT_FIGDATA, FIGDATA_FIELDS, rows)
    log.info(f"Wrote {len(rows)} rows -> {OUT_FIGDATA}")

    fig_df = pd.read_csv(OUT_FIGDATA, encoding="utf-8-sig")
    make_plot(fig_df)
    log.info(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
