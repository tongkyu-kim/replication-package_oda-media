"""
Held-out performance of the development-relevance ("devrel") classifier on
the gold-labeled validation file (N=1,954) -- Manuscript Table 1 and Figure
A2 (confusion matrix / ROC / PR curve).

Ported from the original repository's pipeline/evaluate_devrel_classifier_gold.py,
which fine-tunes and evaluates a KLUE-RoBERTa-base binary classifier
(1 = article discusses a development-related situation in a developing
country; 0 = otherwise). This script only re-runs INFERENCE with the
already-fine-tuned checkpoint; no training happens here.

Text construction (build_text) and the inference Dataset class are ported
verbatim from the original pipeline/train_devrel_classifier.py: for each
row, the first two non-empty fields among
["title", "keywords", "top_keywords", "제목", "키워드"] are concatenated
with a space and truncated to 512 characters. This must match exactly what
the checkpoint was fine-tuned on.

Two known bugs in the original script were fixed during porting (confirmed
by an earlier audit pass, not reproduced here):

1. Path resolution. The original script computed
   ORIG_MODEL_DIR = PROJECT_ROOT / "models" / "devrel_classifier", which
   does not exist -- the checkpoint actually lives at
   PROJECT_ROOT / "pipeline" / "models" / "devrel_classifier". In this
   package the checkpoint is copied to DATA_VALIDATION/classifier_checkpoint/
   (resolved via _common.py, no hardcoded absolute path).

2. Hardcoded scratch path. The original script unconditionally copied the
   ~440MB checkpoint to a hardcoded C:\\tmp_devrel_model before loading, to
   work around a claimed transformers from_pretrained() local-path
   detection failure attributed to the special characters ("'", "[...]")
   in this project's directory names. That workaround was tested directly
   against this package's own (similarly special-character-laden) path
   during porting and was found to be UNNECESSARY: transformers loads the
   checkpoint directly from DATA_VALIDATION/classifier_checkpoint/ without
   any copying, even though that path also contains an apostrophe and
   brackets. So this script loads the checkpoint in place. As a purely
   defensive fallback (in case some future environment really does hit a
   local-path detection issue), if direct loading fails, it falls back to
   copying the checkpoint into a Python tempfile.TemporaryDirectory (auto-
   cleaned on exit) rather than any hardcoded location -- nothing is ever
   written outside this package or the OS temp directory.

Not ported: the original script's train/gold title-overlap leakage check.
That check reads src/labels/sample_labeled.csv (the raw training-label
file), which is not part of this public package (see DATA_AVAILABILITY.md)
and is not needed to compute the performance metrics themselves. The
original repo's own docstring already records that check's result (zero
title overlap, N=1,954 gold vs. training file) -- it is a one-time
provenance check on the gold set's construction, not a re-derivable
research finding, so it is not reproduced here.

Two run modes (pick automatically, or force with --mode):
  - "infer":     load the checkpoint, run inference over the gold set on
                 CPU (matches the original script exactly -- deterministic,
                 fixed batch order, no train/val split), write
                 classifier_predictions.csv, then compute metrics from it.
  - "recompute": if classifier_predictions.csv already exists, just
                 recompute classifier_performance.csv from the saved
                 predictions -- no torch/transformers import needed, runs
                 in well under a second.
  "auto" (the default) does "recompute" if predictions already exist on
  disk, else "infer". This script was tested end-to-end in "infer" mode
  (full CPU inference over N=1,954) and reproduced the manuscript's stated
  Table 1 values exactly (Accuracy=0.9637, Precision=0.8656, Recall=0.9388,
  F1=0.9007, AUC-ROC=0.9896); "recompute" mode was then verified against
  the resulting classifier_predictions.csv.

Writes:
  results/diagnostics/classifier_performance.csv  (metric, value)
  results/diagnostics/classifier_predictions.csv  (row_id, article_id,
                                                     y_true, y_pred_proba,
                                                     y_pred_label)

Usage:
  python classifier_validation.py               # auto mode
  python classifier_validation.py --mode infer
  python classifier_validation.py --mode recompute
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

from _common import DATA_VALIDATION, RESULTS_DIAGNOSTICS, RANDOM_SEED, get_logger, write_rows

log = get_logger("classifier_validation")

MODEL_DIR = DATA_VALIDATION / "classifier_checkpoint"
GOLD_PATH = DATA_VALIDATION / "precision_gold_devrel.csv"
PERF_OUT = RESULTS_DIAGNOSTICS / "classifier_performance.csv"
PRED_OUT = RESULTS_DIAGNOSTICS / "classifier_predictions.csv"

MAX_LEN = 128          # matches train_devrel_classifier.py: title + keywords fits in 128 tokens
BATCH_INFER = 64        # matches train_devrel_classifier.py
THRESHOLD = 0.5
LABEL_COL = "label_development_relevant"


# ---------------------------------------------------------------------------
# Ported verbatim from the original pipeline/train_devrel_classifier.py so
# that inference reproduces exactly the input representation the checkpoint
# was fine-tuned on.
# ---------------------------------------------------------------------------

def build_text(row: "pd.Series") -> str:
    """Concatenate title + keywords for classification input."""
    parts = []
    for col in ["title", "keywords", "top_keywords", "제목", "키워드"]:
        v = str(row.get(col, "") or "").strip()
        if v:
            parts.append(v)
        if len(parts) == 2:
            break
    return " ".join(parts)[:512]


class ArticleDataset:
    """Minimal torch Dataset, ported from train_devrel_classifier.py."""

    def __init__(self, encodings: dict, labels: "list[int] | None" = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        import torch
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_gold() -> pd.DataFrame:
    gold = pd.read_csv(GOLD_PATH, dtype=str, encoding="utf-8-sig")
    gold = gold[gold[LABEL_COL].isin(["0", "1"])].copy()
    gold = gold.reset_index(drop=True)
    gold["row_id"] = gold.index
    return gold


def _load_model_and_tokenizer(model_dir: Path):
    """Load the checkpoint from model_dir. Tries the path directly first
    (this is expected to always work -- see module docstring, bug 2); falls
    back to a self-cleaning tempfile copy only if that somehow fails."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
        return tokenizer, model
    except Exception as exc:  # pragma: no cover -- defensive fallback, not expected to trigger
        log.warning(f"Direct load from {model_dir} failed ({exc}); "
                    f"falling back to a temporary local copy.")
        required = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
        with tempfile.TemporaryDirectory(prefix="devrel_classifier_") as tmp:
            tmp_path = Path(tmp)
            import shutil
            for fname in required:
                shutil.copy2(model_dir / fname, tmp_path / fname)
            tokenizer = AutoTokenizer.from_pretrained(str(tmp_path), local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(str(tmp_path), local_files_only=True)
            return tokenizer, model


def run_inference() -> pd.DataFrame:
    """Load the checkpoint and score the full gold set. Returns a dataframe
    with row_id, article_id, y_true, y_pred_proba."""
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cpu")  # matches the original script -- deterministic, no GPU dependency

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found at {MODEL_DIR}. "
            f"See DATA_AVAILABILITY.md -- it must be copied there from the original "
            f"repository's pipeline/models/devrel_classifier/ before running this script."
        )

    gold = load_gold()
    texts = [build_text(row) for _, row in gold.iterrows()]
    labels = gold[LABEL_COL].astype(int).tolist()
    log.info(f"Loaded gold set: N={len(labels)}  positives={sum(labels)}  "
             f"negatives={len(labels) - sum(labels)}")

    log.info(f"Loading checkpoint from {MODEL_DIR} ...")
    tokenizer, model = _load_model_and_tokenizer(MODEL_DIR)
    model = model.to(device)
    model.eval()

    enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    ds = ArticleDataset({k: v.numpy() for k, v in enc.items()})
    loader = DataLoader(ds, batch_size=BATCH_INFER, shuffle=False)  # fixed order -> reproducible

    probs_all: list[float] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
            probs_all.extend(probs.tolist())
            log.info(f"  batch {i + 1}/{len(loader)} done")

    return pd.DataFrame({
        "row_id": gold["row_id"],
        "article_id": gold["article_id"],
        "y_true": labels,
        "y_pred_proba": probs_all,
    })


def compute_metrics(pred_df: pd.DataFrame) -> list[dict]:
    y_true = pred_df["y_true"].astype(int).to_numpy()
    y_proba = pred_df["y_pred_proba"].astype(float).to_numpy()
    y_pred = (y_proba >= THRESHOLD).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_proba),
        "test_n": len(y_true),
        "positive_n": int(y_true.sum()),
        "negative_n": int(len(y_true) - y_true.sum()),
        "positive_share": float(y_true.mean()),
    }
    log.info("Classifier performance on gold set (N={test_n}): "
             "Accuracy={accuracy:.4f}  Precision={precision:.4f}  Recall={recall:.4f}  "
             "F1={f1:.4f}  AUC-ROC={auc_roc:.4f}".format(**metrics))
    return [{"metric": k, "value": v} for k, v in metrics.items()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "infer", "recompute"], default="auto",
                         help="'infer' always re-runs the checkpoint; 'recompute' always reuses "
                              "an existing classifier_predictions.csv; 'auto' (default) recomputes "
                              "if that file already exists, else infers.")
    args = parser.parse_args()

    mode = args.mode
    if mode == "auto":
        mode = "recompute" if PRED_OUT.exists() else "infer"
        log.info(f"Mode 'auto' resolved to '{mode}' "
                 f"({'found' if mode == 'recompute' else 'did not find'} {PRED_OUT}).")

    if mode == "infer":
        pred_df = run_inference()
        write_rows(PRED_OUT, ["row_id", "article_id", "y_true", "y_pred_proba"],
                   pred_df.to_dict("records"))
        log.info(f"Wrote {len(pred_df)} rows -> {PRED_OUT}")
    else:
        if not PRED_OUT.exists():
            raise FileNotFoundError(
                f"--mode recompute requires an existing {PRED_OUT}; run with --mode infer first."
            )
        pred_df = pd.read_csv(PRED_OUT)
        log.info(f"Recomputing metrics from saved predictions -> {PRED_OUT} (N={len(pred_df)})")

    pred_df["y_pred_label"] = (pred_df["y_pred_proba"].astype(float) >= THRESHOLD).astype(int)
    write_rows(PRED_OUT, ["row_id", "article_id", "y_true", "y_pred_proba", "y_pred_label"],
               pred_df.to_dict("records"))

    metric_rows = compute_metrics(pred_df)
    write_rows(PERF_OUT, ["metric", "value"], metric_rows)
    log.info(f"Wrote {len(metric_rows)} rows -> {PERF_OUT}")


if __name__ == "__main__":
    main()
