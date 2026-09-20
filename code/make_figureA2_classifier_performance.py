"""
Figure A2 -- Classification performance of the development-relevance
("devrel") model (Appendix).

Data: results/diagnostics/classifier_predictions.csv (row_id, article_id,
y_true, y_pred_proba, y_pred_label -- N=1,954 held-out gold-labeled
articles) and results/diagnostics/classifier_performance.csv (summary
metrics), both already-complete output of code/classifier_validation.py.
No new inference happens here -- this script only derives plotting curves
(confusion matrix counts, ROC, precision-recall) from those predictions
via sklearn.metrics (a pinned dependency) and plots them.

Three panels in one figure: (1) confusion matrix (y_true x y_pred_label),
(2) ROC curve (sklearn.metrics.roc_curve / roc_auc_score), (3)
precision-recall curve (sklearn.metrics.precision_recall_curve).

Writes:
  results/figure_data/figureA2_classifier_performance_data.csv (long
    format, "curve" column distinguishes "roc" and "pr" point series)
  results/figure_data/figureA2_confusion_matrix_data.csv (2x2 counts --
    not a "curve", kept as its own small file rather than forced into the
    curve schema above)
  results/figures/figureA2_classifier_performance.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (confusion_matrix, precision_recall_curve, roc_auc_score,  # noqa: E402
                              roc_curve)

from _common import RESULTS_DIAGNOSTICS, RESULTS_FIGURES, RESULTS_FIGURE_DATA, get_logger, write_rows  # noqa: E402

log = get_logger("make_figureA2_classifier_performance")

IN_PREDICTIONS = RESULTS_DIAGNOSTICS / "classifier_predictions.csv"
IN_PERFORMANCE = RESULTS_DIAGNOSTICS / "classifier_performance.csv"
OUT_CURVE_DATA = RESULTS_FIGURE_DATA / "figureA2_classifier_performance_data.csv"
OUT_CM_DATA = RESULTS_FIGURE_DATA / "figureA2_confusion_matrix_data.csv"
OUT_PNG = RESULTS_FIGURES / "figureA2_classifier_performance.png"

CURVE_FIELDS = ["curve", "x", "y", "threshold"]
CM_FIELDS = ["true_label", "pred_label", "count"]


def build_curve_data(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[list[dict], float, float]:
    fpr, tpr, roc_thr = roc_curve(y_true, y_proba)
    auc = float(roc_auc_score(y_true, y_proba))
    precision, recall, pr_thr = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns one more point than thresholds (the
    # final point is recall=0, precision=1 with no corresponding
    # threshold) -- pad with NaN so every point has a row.
    pr_thr_padded = np.append(pr_thr, np.nan)

    rows: list[dict] = []
    for x, y, t in zip(fpr, tpr, roc_thr):
        rows.append({"curve": "roc", "x": float(x), "y": float(y), "threshold": float(t)})
    for x, y, t in zip(recall, precision, pr_thr_padded):
        rows.append({"curve": "pr", "x": float(x), "y": float(y),
                      "threshold": float(t) if not np.isnan(t) else ""})
    return rows, auc, float(np.trapezoid(precision[::-1], recall[::-1]))


def build_confusion_matrix_data(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    rows = []
    for i, true_label in enumerate([0, 1]):
        for j, pred_label in enumerate([0, 1]):
            rows.append({"true_label": true_label, "pred_label": pred_label, "count": int(cm[i, j])})
    return rows


def make_plot(curve_df: pd.DataFrame, cm_df: pd.DataFrame, auc: float, ap: float) -> None:
    fig, (ax_cm, ax_roc, ax_pr) = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # ---- Panel 1: confusion matrix ----
    cm = np.zeros((2, 2), dtype=int)
    for _, r in cm_df.iterrows():
        cm[int(r["true_label"]), int(r["pred_label"])] = int(r["count"])
    im = ax_cm.imshow(cm, cmap="Greys", vmin=0)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax_cm.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13, color=color)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Not devrel (0)", "Devrel (1)"])
    ax_cm.set_yticklabels(["Not devrel (0)", "Devrel (1)"])
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("True label")
    ax_cm.set_title("Confusion matrix")

    # ---- Panel 2: ROC curve ----
    roc = curve_df[curve_df["curve"] == "roc"].sort_values("x")
    ax_roc.plot(roc["x"], roc["y"], color="black", lw=1.6, label=f"ROC (AUC = {auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], color="gray", lw=1.0, ls="--", label="Chance")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curve")
    ax_roc.legend(loc="lower right", frameon=False, fontsize=9)
    ax_roc.spines[["top", "right"]].set_visible(False)

    # ---- Panel 3: precision-recall curve ----
    pr = curve_df[curve_df["curve"] == "pr"].sort_values("x")
    ax_pr.plot(pr["x"], pr["y"], color="black", lw=1.6, label=f"PR (AP = {ap:.3f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-recall curve")
    ax_pr.legend(loc="lower left", frameon=False, fontsize=9)
    ax_pr.spines[["top", "right"]].set_visible(False)

    for ax in (ax_roc, ax_pr):
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(color="#E5E5E5", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle("Fig. A2: Classification performance of the development-relevance model "
                  "(held-out gold-labeled test set, N=1,954)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    preds = pd.read_csv(IN_PREDICTIONS, encoding="utf-8-sig")
    y_true = preds["y_true"].to_numpy()
    y_proba = preds["y_pred_proba"].to_numpy()
    y_pred = preds["y_pred_label"].to_numpy()

    curve_rows, auc, ap = build_curve_data(y_true, y_proba)
    write_rows(OUT_CURVE_DATA, CURVE_FIELDS, curve_rows)
    log.info(f"Wrote {len(curve_rows)} rows -> {OUT_CURVE_DATA} (AUC-ROC={auc:.4f}, AP={ap:.4f})")

    cm_rows = build_confusion_matrix_data(y_true, y_pred)
    write_rows(OUT_CM_DATA, CM_FIELDS, cm_rows)
    log.info(f"Wrote {len(cm_rows)} rows -> {OUT_CM_DATA}")

    if IN_PERFORMANCE.exists():
        perf = pd.read_csv(IN_PERFORMANCE, encoding="utf-8-sig").set_index("metric")["value"]
        reported_auc = float(perf.get("auc_roc", float("nan")))
        log.info(f"classifier_performance.csv reports auc_roc={reported_auc:.4f} "
                  f"(recomputed here from the same predictions: {auc:.4f})")

    curve_df = pd.read_csv(OUT_CURVE_DATA, encoding="utf-8-sig")
    cm_df = pd.read_csv(OUT_CM_DATA, encoding="utf-8-sig")
    make_plot(curve_df, cm_df, auc, ap)
    log.info(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
