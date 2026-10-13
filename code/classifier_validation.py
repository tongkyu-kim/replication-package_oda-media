"""
Held-out performance of the fine-tuned KLUE-RoBERTa development-relevance
classifier on the gold-labeled test set (N = 1,954): Table 1 and Figure A2.

Inference only: the fine-tuned checkpoint is applied to the gold set on CPU
(fixed batch order). The classifier input is the first two non-empty fields
among title, keywords and top keywords, truncated to 512 characters.

Modes: "infer" runs the checkpoint and writes the predictions; "recompute"
recomputes the metrics from saved predictions; "auto" (default) recomputes
when predictions exist and otherwise infers.

Input:  data/validation/classifier_checkpoint/, data/validation/precision_gold_devrel.csv
Output: results/diagnostics/classifier_performance.csv,
        results/diagnostics/classifier_predictions.csv
"""
from __future__ import annotations

import argparse
import sys
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

MAX_LEN = 128
BATCH_INFER = 64
THRESHOLD = 0.5
LABEL_COL = "label_development_relevant"


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
    """Minimal torch Dataset for tokenized inputs."""

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
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    return tokenizer, model


def run_inference() -> pd.DataFrame:
    """Load the checkpoint and score the full gold set. Returns a dataframe
    with row_id, article_id, y_true, y_pred_proba."""
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cpu")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found at {MODEL_DIR}; see DATA_AVAILABILITY.md.")

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
    loader = DataLoader(ds, batch_size=BATCH_INFER, shuffle=False)

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
