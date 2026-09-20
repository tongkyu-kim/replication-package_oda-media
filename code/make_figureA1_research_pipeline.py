"""
Figure A1 -- Research pipeline (Appendix). A pure diagram, no data or
regression -- box/arrow flowchart of this REPLICATION PACKAGE's own
analytical pipeline, box-drawn with matplotlib patches/annotations.

This is a fresh diagram of THIS package's stages, not a port of the
original research repository's pipeline/manuscript_tables/
make_figure1_research_flow.py -- that figure diagrams the original,
non-public repository's OWN pipeline (raw BigKinds/ODAKorea collection,
classifier training, the original script-renumbering history), most of
which this package deliberately does not reproduce (see
DATA_AVAILABILITY.md: this package starts from already-built analytical
CSVs, and classifier_validation.py only re-runs inference against an
already-fine-tuned checkpoint, never training). The stages below instead
match this package's own code/00_run_all.py execution order exactly:
data construction -> classifier validation & reliability -> main/temporal
analyses (H1-H4) -> secondary analyses -> robustness & diagnostics ->
tables & figures.

Writes: results/figures/figureA1_research_pipeline.png
No figure-data CSV -- this is a diagram, not a data plot.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from _common import RESULTS_FIGURES, get_logger  # noqa: E402

log = get_logger("make_figureA1_research_pipeline")

OUT_PNG = RESULTS_FIGURES / "figureA1_research_pipeline.png"

INK = "#000000"
WHITE = "#FFFFFF"

FIG_W, FIG_H = 11.0, 13.5

# (title, subtitle scripts, detail lines)
STAGES = [
    ("1. Data construction",
     "code/build_analytical_panels.py",
     ["Assembles country x year/month analytical panels from the included",
      "data/analytical/*.csv (ODA, media counts, controls -- see",
      "DATA_AVAILABILITY.md for what each input is and why it ships pre-built)"]),
    ("2. Classifier validation & reliability",
     "code/classifier_validation.py  +  code/reliability_alpha.py",
     ["Held-out inference of the fine-tuned KLUE-RoBERTa devrel classifier",
      "(N=1,954 gold-labeled articles; Table 1, Fig. A2) and Krippendorff's",
      "alpha for the three hand-coded variables (devrel/sentiment/crisis)"]),
    ("3. Main and temporal analyses (H1-H4)",
     "main_baseline  |  lag_windows  |  rolling_persistence  |  sustained_coverage  |  emergency_temporal",
     ["H1 recent attention -> H2 persistent attention -> H3 sustained attention,",
      "plus the temporal-decomposition diagnostic and the H4 development-vs-",
      "emergency ODA timing contrast (stacked PPML, Fig. 5)"]),
    ("4. Secondary analyses",
     "allocation_margin  |  ideological_breadth  |  evaluative_tone",
     ["Extensive vs. intensive margin (Fig. 6); cross-partisan salience of",
      "coverage (Fig. 7); evaluative tone (positive/negative share)"]),
    ("5. Robustness and diagnostics",
     "placebo_tests  |  alt_timing  |  leave_one_out",
     ["Future-news and NonDevRel substantive placebos (Fig. 8); alternative",
      "timing windows; leave-one-country-out sensitivity"]),
    ("6. Tables and figures",
     "generate_tables.py  +  make_figureN*.py  +  build_reported_text_results.py",
     ["Every manuscript table (1-4, A1-A6) and figure (1-8, A1-A2) built",
      "directly from results/models/ and results/diagnostics/ -- no manually",
      "typed statistical value anywhere downstream of this stage"]),
]


def box(ax, x, y, w, h, title, subtitle, details, fontsize_title=11.5):
    r = Rectangle((x, y), w, h, facecolor=WHITE, edgecolor=INK, linewidth=1.3, zorder=2)
    ax.add_patch(r)
    ax.text(x + w / 2, y + h - 0.34, title, ha="center", va="top", fontsize=fontsize_title,
            fontweight="bold", zorder=3)
    ax.text(x + w / 2, y + h - 0.72, subtitle, ha="center", va="top", fontsize=8.6,
            fontstyle="italic", zorder=3, family="monospace")
    detail_text = "\n".join(details)
    ax.text(x + w / 2, y + h - 1.10, detail_text, ha="center", va="top", fontsize=8.6,
            zorder=3, linespacing=1.5)


def arrow(ax, x, y0, y1):
    a = FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=16, color=INK, linewidth=1.3, zorder=1)
    ax.add_patch(a)


def make_figure() -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    n = len(STAGES)
    box_w = 9.4
    box_h = 1.85
    gap = 0.55
    x0 = (FIG_W - box_w) / 2
    top_y = FIG_H - 0.5

    ys = [top_y - box_h - i * (box_h + gap) for i in range(n)]

    for (title, subtitle, details), y in zip(STAGES, ys):
        box(ax, x0, y, box_w, box_h, title, subtitle, details)

    for i in range(n - 1):
        y_top_box_bottom = ys[i]
        y_next_box_top = ys[i + 1] + box_h
        arrow(ax, FIG_W / 2, y_top_box_bottom, y_next_box_top)

    ax.text(FIG_W / 2, FIG_H - 0.15, "Fig. A1: Research pipeline (replication package execution order)",
            ha="center", va="top", fontsize=13, fontweight="bold")

    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    make_figure()
    log.info(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
