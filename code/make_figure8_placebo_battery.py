"""
Figure 8: temporal and substantive placebo tests.

Left panel: 2-13-, 14-25- and 26-37-month pre-allocation coefficients, each
with its future-media placebo (AR(2) innovation for 2-13 months; mirror-image
future windows for the other two). Right panel: news-substance placebo for
the seven headline measures (development-relevant, all country, and sports
news). Point estimates with 90% confidence intervals.

Input:  results/models/main_baseline_results.csv,
        results/models/placebo_future_news_results.csv,
        results/models/placebo_news_substance_results.csv
Output: results/figures/figure8_placebo_battery.png,
        results/figure_data/figure8_placebo_battery_data.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from _common import RESULTS_FIGURE_DATA, RESULTS_FIGURES, RESULTS_MODELS, get_logger  # noqa: E402

log = get_logger("make_figure8_placebo_battery")

INK, GRID, WHITE = "#000000", "#E5E5E5", "#FFFFFF"
plt.rcParams.update({"figure.facecolor": WHITE, "axes.facecolor": WHITE, "savefig.facecolor": WHITE,
                     "font.size": 9, "font.family": ["Cambria Math", "DejaVu Serif"], "text.color": INK,
                     "pdf.fonttype": 42})

TEMPORAL = [("2-13 months", ("main", "H1_preferred__devrel_media_m1_13"),
             ("future", "placebo_future_news_ar2_innovation__innovation_ar2")),
            ("14-25 months", ("future", "placebo_future_news_m14_25_pre__devrel_media_m14_25"),
             ("future", "placebo_future_news_m14_25_future__future_m14_25")),
            ("26-37 months", ("future", "placebo_future_news_m26_37_pre__devrel_media_m26_37"),
             ("future", "placebo_future_news_m26_37_future__future_m26_37"))]
SUBSTANCE = [("m1_13", "2-13 mo\nattention"), ("roll24", "Rolling\n24-mo"), ("roll36", "Rolling\n36-mo"),
             ("roll60", "Rolling\n60-mo"), ("streak24", "Streak\n>=24mo"), ("streak36", "Streak\n>=36mo"),
             ("streak60", "Streak\n>=60mo")]
SERIES_STYLE = {  # whisker line style, marker, marker face
    "real": ("solid", "o", INK), "allnews": ((0, (4, 1.5, 1, 1.5)), "^", WHITE),
    "placebo": ((0, (1.5, 1)), "s", WHITE), "sports": ((0, (1.5, 1)), "s", WHITE)}


def figure_data() -> pd.DataFrame:
    src = {"main": pd.read_csv(RESULTS_MODELS / "main_baseline_results.csv").set_index("specification_id"),
           "future": pd.read_csv(RESULTS_MODELS / "placebo_future_news_results.csv").set_index("specification_id"),
           "substance": pd.read_csv(RESULTS_MODELS / "placebo_news_substance_results.csv")
           .set_index("specification_id")}
    rows = []

    def add(panel, order, label, series, r):
        rows.append({"panel": panel, "measure_order": order, "measure_label": label, "series": series,
                     "coefficient": r["coefficient"], "se": r["se"], "ci_lower": r["ci_lower"],
                     "ci_upper": r["ci_upper"], "ci_level": r["ci_level"], "p_value": r["p_value"], "n": r["n"]})

    for i, (label, (s1, real_id), (s2, pla_id)) in enumerate(TEMPORAL):
        add("temporal", i, label, "real", src[s1].loc[real_id])
        add("temporal", i, label, "placebo", src[s2].loc[pla_id])
    for i, (m, label) in enumerate(SUBSTANCE):
        for series in ["devrel", "allnews", "sports"]:
            add("substance", i, label.replace("\n", " "), "real" if series == "devrel" else series,
                src["substance"].loc[f"placebo_substance_{m}_{series}"])
    return pd.DataFrame(rows)


def style_axes(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.7, length=3, labelsize=7.6)
    ax.axhline(0, color=INK, lw=0.7, ls=(0, (1.5, 1)), zorder=2)


def draw_group(ax, x: float, rows: pd.DataFrame, offset: float) -> None:
    n = len(rows)
    for j, (_, r) in enumerate(rows.iterrows()):
        xi = x + (j - (n - 1) / 2) * offset
        ls, marker, face = SERIES_STYLE[r["series"]]
        ax.vlines(xi, r["ci_lower"], r["ci_upper"], color=INK, lw=1.1, linestyles=ls, zorder=3)
        ax.hlines([r["ci_lower"], r["ci_upper"]], xi - 0.045, xi + 0.045, color=INK, lw=1.1, zorder=3)
        ax.scatter(xi, r["coefficient"], facecolors=face, edgecolors=INK, marker=marker, s=30, linewidths=1.0,
                   zorder=4)


def legend(ax, items: list[tuple[str, str, str]], ncol: int) -> None:
    handles = [Line2D([0], [0], color="none", marker=m, markerfacecolor=f, markeredgecolor=INK,
                      markeredgewidth=1.1, markersize=5.2, label=lab) for lab, m, f in items]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=ncol, frameon=False,
              fontsize=6.4, handletextpad=0.5, columnspacing=1.4)


def main() -> None:
    fd = figure_data()
    fd.to_csv(RESULTS_FIGURE_DATA / "figure8_placebo_battery_data.csv", index=False, encoding="utf-8-sig")

    fig, (axB, axA) = plt.subplots(1, 2, figsize=(13.0, 4.4), gridspec_kw={"width_ratios": [3, 7]})
    tmp = fd[fd["panel"] == "temporal"]
    for i, (label, _, _) in enumerate(TEMPORAL):
        draw_group(axB, i, tmp[tmp["measure_order"] == i], 0.28)
    axB.set_xticks(range(len(TEMPORAL)))
    axB.set_xticklabels([t[0] for t in TEMPORAL])
    axB.set_xlim(-0.5, len(TEMPORAL) - 0.5)
    style_axes(axB)
    axB.set_ylabel("Coefficient on media attention", fontsize=8.7)
    axB.set_title("Pre-allocation attention versus\nfuture-media placebo", fontsize=8.7, pad=10)
    legend(axB, [(r"Pre-allocation media ($T_{t-m}$)", "o", INK), ("Future-media placebo", "s", WHITE)], 2)

    sub = fd[fd["panel"] == "substance"]
    for i, _ in enumerate(SUBSTANCE):
        draw_group(axA, i, sub[sub["measure_order"] == i], 0.22)
    axA.set_xticks(range(len(SUBSTANCE)))
    axA.set_xticklabels([s[1] for s in SUBSTANCE])
    axA.set_xlim(-0.5, len(SUBSTANCE) - 0.5)
    style_axes(axA)
    axA.set_ylabel("Coefficient on media attention", fontsize=8.7)
    axA.set_title("News-substance placebo\n(development-relevant vs. all country vs. sports news)",
                  fontsize=8.3, pad=10)
    legend(axA, [("Development-relevant news", "o", INK), ("All country news", "^", WHITE),
                 ("Sports news", "s", WHITE)], 3)

    fig.tight_layout()
    out = RESULTS_FIGURES / "figure8_placebo_battery.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
