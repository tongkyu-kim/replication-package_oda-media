"""
Shared configuration and result-writing utilities for the replication package.

Every analysis script in this package imports from here rather than
hardcoding paths or inventing its own output schema. All paths are relative
to the package root (the parent of this file's parent directory), resolved
via PACKAGE_ROOT below -- nothing in this package refers to any location
outside itself except the explicitly-documented external sources listed in
DATA_AVAILABILITY.md.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_ROOT / "data"
DATA_ANALYTICAL = DATA_DIR / "analytical"
DATA_VALIDATION = DATA_DIR / "validation"
DATA_EXTERNAL = DATA_DIR / "external"
RESULTS_DIR = PACKAGE_ROOT / "results"
RESULTS_MODELS = RESULTS_DIR / "models"
RESULTS_TABLES = RESULTS_DIR / "tables"
RESULTS_FIGURES = RESULTS_DIR / "figures"
RESULTS_FIGURE_DATA = RESULTS_DIR / "figure_data"
RESULTS_DIAGNOSTICS = RESULTS_DIR / "diagnostics"
LOGS_DIR = PACKAGE_ROOT / "logs"

# Study-wide constants. Fixed once here so every script agrees with every
# other script; nothing below is a "new" analytical choice -- these mirror
# what the canonical scripts in the original research repository already do.
SIGNIFICANCE_CI_LEVEL = 0.90          # this study's figures/CI convention throughout
Z_90 = 1.6448536269514722             # two-sided 90% normal critical value
CLUSTER_LEVEL = "country"
FE_STRUCTURE_BASELINE = "country + year"
RANDOM_SEED = 2025                    # matches the original classifier train/test split seed


def ensure_dirs() -> None:
    for d in (DATA_ANALYTICAL, DATA_VALIDATION, DATA_EXTERNAL, RESULTS_MODELS,
              RESULTS_TABLES, RESULTS_FIGURES, RESULTS_FIGURE_DATA,
              RESULTS_DIAGNOSTICS, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    ensure_dirs()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)-28s  %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(LOGS_DIR / "run_all.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def require_file(path: Path, hint: str = "") -> Path:
    """Fail loudly and early rather than letting a script silently run on stale/missing input."""
    if not path.exists():
        msg = f"Required input not found: {path}"
        if hint:
            msg += f"\n  -> {hint}"
        raise FileNotFoundError(msg)
    return path


# ---------------------------------------------------------------------------
# Standardized result-file schema (see README.md "Outputs" / step 10 of the
# release spec). Every regression-style specification exported by this
# package is one row in this schema. Hypothesis tests and bootstrap results
# use the two schemas further below. No script should write a model-result
# CSV with any other column layout.
# ---------------------------------------------------------------------------

MODEL_RESULT_FIELDS = [
    "specification_id",   # stable, human-readable id, e.g. "H1_preferred"
    "outcome",             # e.g. "ln_devrel_commitment"
    "key_predictor",       # e.g. "devrel_media_2_13"
    "coefficient",
    "se",
    "p_value",
    "sig",                 # "", "+", "*", "**", "***" at .10/.05/.01/.001
    "ci_lower",
    "ci_upper",
    "ci_level",            # e.g. 0.90
    "n",
    "n_entities",           # number of countries / entities
    "year_range",          # e.g. "2010-2025"
    "fe_structure",
    "cluster_level",
    "controls",             # semicolon-separated control variable list, or "none"
    "sample_definition",
    "timing_window",
    "transformation",      # e.g. "ln(x+1)"
]

HYPOTHESIS_TEST_FIELDS = [
    "specification_id",
    "test_name",
    "test_statistic",
    "df",
    "p_value",
    "restriction_tested",
    "n",
]

BOOTSTRAP_FIELDS = [
    "specification_id",
    "seed",
    "n_reps_requested",
    "n_reps_success",
    "n_reps_fail",
    "point_estimate",
    "ci_lower",
    "ci_upper",
    "ci_level",
    "p_value",
]


def stars(p: Optional[float]) -> str:
    if p is None:
        return ""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "+"
    return ""


@dataclass
class ModelResultRow:
    specification_id: str
    outcome: str
    key_predictor: str
    coefficient: float
    se: float
    p_value: float
    n: int
    n_entities: int = None
    year_range: str = ""
    fe_structure: str = FE_STRUCTURE_BASELINE
    cluster_level: str = CLUSTER_LEVEL
    controls: str = "none"
    sample_definition: str = ""
    timing_window: str = ""
    transformation: str = "ln(x+1)"
    ci_level: float = SIGNIFICANCE_CI_LEVEL
    sig: str = field(default="")
    ci_lower: float = None
    ci_upper: float = None

    def __post_init__(self):
        if not self.sig:
            self.sig = stars(self.p_value)
        if self.ci_lower is None and self.coefficient is not None and self.se is not None:
            self.ci_lower = self.coefficient - Z_90 * self.se
            self.ci_upper = self.coefficient + Z_90 * self.se

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in MODEL_RESULT_FIELDS}


def write_rows(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_model_results(path: Path, rows: Iterable[ModelResultRow]) -> None:
    write_rows(path, MODEL_RESULT_FIELDS, [r.as_dict() for r in rows])


def append_reported_text_result(rows_accum: list, **kwargs) -> None:
    """Helper used by build_reported_text_results.py callers; kept here so
    every script that contributes a prose-only number uses the same field set."""
    rows_accum.append(kwargs)


KNOWN_DISCREPANCIES_PATH = RESULTS_DIAGNOSTICS / "known_discrepancies.csv"
KNOWN_DISCREPANCY_FIELDS = ["id", "table_or_figure", "manuscript_value", "package_value",
                            "likely_cause", "recommended_action"]


def ensure_known_discrepancies_file() -> None:
    """Guarantees results/diagnostics/known_discrepancies.csv exists (header
    only if nothing has ever logged a row this run) -- called at the end of
    the pipeline so a completely fresh run always produces this file, even
    though no discrepancy currently logs anything to it (all four found
    during this package's construction were resolved 2026-09-20; see
    README.md's "Status: frozen" section). Never overwrites existing rows."""
    if not KNOWN_DISCREPANCIES_PATH.exists():
        write_rows(KNOWN_DISCREPANCIES_PATH, KNOWN_DISCREPANCY_FIELDS, [])


def log_known_discrepancy(**kwargs) -> None:
    """Multiple independent scripts each discover and log their own
    manuscript-vs-package discrepancies into the same shared file. This
    merges by 'id' (read-modify-write) rather than overwriting, so
    whichever script runs last never erases another script's row, and
    rerunning any one script is idempotent (it just replaces its own id)."""
    import csv as _csv
    existing: dict[str, dict] = {}
    if KNOWN_DISCREPANCIES_PATH.exists():
        with open(KNOWN_DISCREPANCIES_PATH, newline="", encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                existing[r["id"]] = r
    existing[kwargs["id"]] = kwargs
    write_rows(KNOWN_DISCREPANCIES_PATH, KNOWN_DISCREPANCY_FIELDS, list(existing.values()))
