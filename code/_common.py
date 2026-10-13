"""
Shared paths, constants and the standardized result-file schema. All paths are
relative to the package root.
"""
from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_ROOT / "data"
DATA_ANALYTICAL = DATA_DIR / "analytical"
DATA_VALIDATION = DATA_DIR / "validation"
RESULTS_DIR = PACKAGE_ROOT / "results"
RESULTS_MODELS = RESULTS_DIR / "models"
RESULTS_TABLES = RESULTS_DIR / "tables"
RESULTS_FIGURES = RESULTS_DIR / "figures"
RESULTS_FIGURE_DATA = RESULTS_DIR / "figure_data"
RESULTS_DIAGNOSTICS = RESULTS_DIR / "diagnostics"
LOGS_DIR = PACKAGE_ROOT / "logs"

SIGNIFICANCE_CI_LEVEL = 0.90          # confidence level used in all figures and tables
Z_90 = 1.6448536269514722             # two-sided 90% normal critical value
CLUSTER_LEVEL = "country"
FE_STRUCTURE_BASELINE = "country + year"
RANDOM_SEED = 2025                    # classifier train/test split seed


def ensure_dirs() -> None:
    for d in (DATA_ANALYTICAL, DATA_VALIDATION, RESULTS_MODELS, RESULTS_TABLES, RESULTS_FIGURES,
              RESULTS_FIGURE_DATA, RESULTS_DIAGNOSTICS, LOGS_DIR):
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
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}" + (f"\n  -> {hint}" if hint else ""))
    return path


# ---------------------------------------------------------------------------
# Result-file schemas
# ---------------------------------------------------------------------------

MODEL_RESULT_FIELDS = [
    "specification_id", "outcome", "key_predictor", "coefficient", "se", "p_value",
    "sig",                 # "+", "*", "**", "***" at .10/.05/.01/.001
    "ci_lower", "ci_upper", "ci_level", "n",
    "n_entities",          # number of countries
    "year_range", "fe_structure", "cluster_level",
    "controls",            # semicolon-separated control variables, or "none"
    "sample_definition", "timing_window", "transformation",
]

HYPOTHESIS_TEST_FIELDS = ["specification_id", "test_name", "test_statistic", "df", "p_value",
                          "restriction_tested", "n"]


def stars(p: Optional[float]) -> str:
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
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_model_results(path: Path, rows: Iterable[ModelResultRow]) -> None:
    write_rows(path, MODEL_RESULT_FIELDS, [r.as_dict() for r in rows])
