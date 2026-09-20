"""
Figure 5 -- Contemporaneous versus lagged media responsiveness, development
ODA versus emergency ODA (stacked PPML). The manuscript's key H4 figure.

Data: results/models/emergency_ppml_contrast_results.csv (already complete
-- written by code/emergency_temporal.py's run_ppml_contrast(): 12 rows,
development / emergency / emergency-minus-development-difference
coefficients x 4 temporal windows (contemporaneous, 2-13, 14-25, 26-37
months pre-allocation)). No new estimation happens here -- this script only
reshapes that file for plotting and renders the figure.

Layout ported (visual design only, no statistics) from the original
research repository's
pipeline/manuscript_tables/make_figure8_ppml_temporal_contrast.py (old
filename "figure8" -- promoted to the permanent pipeline 2026-09-08,
renumbered here as manuscript Figure 5): two panels, windows ordered
most-distant to nearest-in-time (26-37mo, 14-25mo, 2-13mo, contemporaneous
-- a chronological left-to-right read). Panel A: raw development (filled
circle, solid whisker) vs. emergency (hollow square, dashed whisker)
coefficients, 90% CI. Panel B: emergency-minus-development difference
(filled square, dashed whisker), 90% CI, zero line. Per the source data,
the contemporaneous window is the only one where the two aid processes are
statistically distinguishable (development~0.096, emergency~0.929,
diff p<0.001); the three lagged windows are not (diff p>0.20 in each).

Writes: results/figure_data/figure5_ppml_contrast_data.csv and
results/figures/figure5_ppml_contrast.png (+ .pdf)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import pandas as pd  # noqa: E402

from _common import RESULTS_MODELS, RESULTS_FIGURES, RESULTS_FIGURE_DATA, get_logger, write_rows  # noqa: E402

log = get_logger("make_figure5_ppml_contrast")

IN_PATH = RESULTS_MODELS / "emergency_ppml_contrast_results.csv"
OUT_FIGDATA = RESULTS_FIGURE_DATA / "figure5_ppml_contrast_data.csv"
OUT_PNG = RESULTS_FIGURES / "figure5_ppml_contrast.png"
OUT_PDF = RESULTS_FIGURES / "figure5_ppml_contrast.pdf"

FIGDATA_FIELDS = ["window", "window_order", "window_label", "aid_type", "coefficient", "se",
                   "ci_lower", "ci_upper", "ci_level", "p_value", "n"]

# Explicit plotting order: most-distant to nearest-in-time (chronological
# left-to-right read), matching the source figure's own convention.
WINDOW_ORDER = {"m26_37": 0, "m14_25": 1, "m1_13": 2, "contemp": 3}
WINDOW_LABELS = {"m26_37": "26-37 mo", "m14_25": "14-25 mo", "m1_13": "2-13 mo", "contemp": "Contemporaneous"}
AID_TYPE_ORDER = {"development": 0, "emergency": 1, "diff": 2}


def _parse_row(key_predictor: str, timing_window: str) -> tuple[str, str]:
    """(aid_type, window) parsed from the source CSV's own key_predictor /
    timing_window text -- no new classification logic, just reading back
    the labels emergency_temporal.py already wrote."""
    if timing_window.startswith("contemporaneous"):
        window = "contemp"
    elif timing_window.startswith("2-13"):
        window = "m1_13"
    elif timing_window.startswith("14-25"):
        window = "m14_25"
    elif timing_window.startswith("26-37"):
        window = "m26_37"
    else:
        raise ValueError(f"Unrecognized timing_window: {timing_window!r}")

    if key_predictor.startswith("diff_emergency_minus_development_"):
        aid_type = "diff"
    elif key_predictor.startswith("development_"):
        aid_type = "development"
    elif key_predictor.startswith("emergency_"):
        aid_type = "emergency"
    else:
        raise ValueError(f"Unrecognized key_predictor: {key_predictor!r}")
    return aid_type, window


def build_figure_data(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in df.iterrows():
        aid_type, window = _parse_row(r["key_predictor"], r["timing_window"])
        rows.append({
            "window": window,
            "window_order": WINDOW_ORDER[window],
            "window_label": WINDOW_LABELS[window],
            "aid_type": aid_type,
            "coefficient": float(r["coefficient"]),
            "se": float(r["se"]),
            "ci_lower": float(r["ci_lower"]),
            "ci_upper": float(r["ci_upper"]),
            "ci_level": float(r["ci_level"]),
            "p_value": float(r["p_value"]),
            "n": int(r["n"]),
        })
    rows.sort(key=lambda r: (r["window_order"], AID_TYPE_ORDER[r["aid_type"]]))
    return rows


def make_plot(fig_df: pd.DataFrame) -> None:
    windows = ["m26_37", "m14_25", "m1_13", "contemp"]
    labels = [WINDOW_LABELS[w] for w in windows]
    x = list(range(len(windows)))
    offset = 0.13

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # ---- Panel A: raw dev vs. emergency coefficients, side by side ----
    for i, w in enumerate(windows):
        dev = fig_df[(fig_df["window"] == w) & (fig_df["aid_type"] == "development")].iloc[0]
        emg = fig_df[(fig_df["window"] == w) & (fig_df["aid_type"] == "emergency")].iloc[0]
        axA.vlines(i - offset, dev["ci_lower"], dev["ci_upper"], color="black", lw=1.3, zorder=3)
        axA.plot(i - offset, dev["coefficient"], marker="o", color="black", ms=7, zorder=4)
        axA.vlines(i + offset, emg["ci_lower"], emg["ci_upper"], color="black", lw=1.3,
                   linestyles=(0, (1.5, 1)), zorder=3)
        axA.plot(i + offset, emg["coefficient"], marker="s", mfc="white", mec="black", ms=7, zorder=4)
    axA.axhline(0, color="black", lw=0.8, ls=(0, (1.5, 1)), zorder=2)
    axA.axvspan(windows.index("contemp") - 0.5, windows.index("contemp") + 0.5, color="#8C1515", alpha=0.06, zorder=0)
    axA.set_xticks(x)
    axA.set_xticklabels(labels)
    axA.set_xlim(-0.55, len(windows) - 0.45)
    axA.set_ylabel("PPML coefficient")
    axA.set_title("Raw temporal-window coefficients")
    axA.spines[["top", "right"]].set_visible(False)
    axA.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)
    axA.set_axisbelow(True)
    handles_a = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="black", markeredgecolor="black",
               markersize=7, label="Development ODA"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor="black",
               markersize=7, label="Emergency ODA"),
    ]
    axA.legend(handles=handles_a, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)

    # ---- Panel B: emergency - development difference ----
    for i, w in enumerate(windows):
        diff = fig_df[(fig_df["window"] == w) & (fig_df["aid_type"] == "diff")].iloc[0]
        axB.vlines(i, diff["ci_lower"], diff["ci_upper"], color="black", lw=1.3, linestyles=(0, (1.5, 1)), zorder=3)
        axB.plot(i, diff["coefficient"], marker="s", color="black", ms=7, zorder=4)
    axB.axhline(0, color="black", lw=0.9, zorder=2)
    axB.axvspan(windows.index("contemp") - 0.5, windows.index("contemp") + 0.5, color="#8C1515", alpha=0.06, zorder=0)
    axB.set_xticks(x)
    axB.set_xticklabels(labels)
    axB.set_xlim(-0.55, len(windows) - 0.45)
    axB.set_ylabel("PPML coefficient")
    axB.set_title("Emergency minus development difference")
    axB.spines[["top", "right"]].set_visible(False)
    axB.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)
    axB.set_axisbelow(True)
    handles_b = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="black", markeredgecolor="black",
               markersize=7, label="Emergency - development (interaction)"),
    ]
    axB.legend(handles=handles_b, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False)

    fig.suptitle("Fig. 5: Contemporaneous vs. lagged media responsiveness -- development vs. emergency ODA "
                  "(stacked PPML, country x aid-type + year x aid-type FE, clustered SE)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(IN_PATH, encoding="utf-8-sig")
    rows = build_figure_data(df)
    write_rows(OUT_FIGDATA, FIGDATA_FIELDS, rows)
    log.info(f"Wrote {len(rows)} rows -> {OUT_FIGDATA}")

    fig_df = pd.read_csv(OUT_FIGDATA, encoding="utf-8-sig")
    make_plot(fig_df)
    log.info(f"Wrote {OUT_PNG} and {OUT_PDF}")


if __name__ == "__main__":
    main()
