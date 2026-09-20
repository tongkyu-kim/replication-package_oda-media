"""
Figure 7 -- Predicted development ODA by ideological breadth of media
coverage (cross-partisan vs. single-bloc salience).

Data already complete: results/figure_data/figure7_ideological_breadth_data.csv,
written by code/ideological_breadth.py's build_figure_data() (predicted
ln(ODA) for cross-partisan vs. single-bloc coverage across the observed
5th-95th percentile range of media attention, with delta-method 90% CI
ribbons, predicted at the estimation sample's means). This script only
plots that already-written data -- no new computation happens here.

Writes: results/figures/figure7_ideological_breadth.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from _common import RESULTS_FIGURES, RESULTS_FIGURE_DATA, get_logger  # noqa: E402

log = get_logger("make_figure7_ideological_breadth")

IN_PATH = RESULTS_FIGURE_DATA / "figure7_ideological_breadth_data.csv"
OUT_PNG = RESULTS_FIGURES / "figure7_ideological_breadth.png"

SERIES = [
    ("cross_partisan", "Cross-partisan coverage (both blocs)", "black", "-"),
    ("single_bloc", "Single-bloc coverage", "#555555", "--"),
]


def make_plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for coverage_type, label, color, ls in SERIES:
        sub = df[df["coverage_type"] == coverage_type].sort_values("media_attention_level")
        ax.plot(sub["media_attention_level"], sub["predicted_log_oda"], color=color, ls=ls, lw=1.6, label=label,
                zorder=3)
        ax.fill_between(sub["media_attention_level"], sub["ci_lower"], sub["ci_upper"], color=color, alpha=0.18,
                         linewidth=0, zorder=1)

    ci_level = df["ci_level"].iloc[0]
    ax.set_xlabel("Development-related media attention (2-13 months pre-allocation, ln(1+articles))")
    ax.set_ylabel("Predicted ln(devrel ODA commitment)")
    ax.set_title("Fig. 7: Predicted ODA by ideological breadth of media coverage\n"
                  f"(shaded bands = {ci_level:.0%} CI)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(IN_PATH, encoding="utf-8-sig")
    make_plot(df)
    log.info(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
