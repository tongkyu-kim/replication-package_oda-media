"""
Inter-coder reliability (Section 3.1): Krippendorff's alpha for development
relevance (nominal), sentiment (ordinal) and the crisis flag (nominal).

The two coder files are aligned on article_id. Codes outside each variable's
domain are treated as missing. Following the codebook, a coder's crisis flag
is used only where that coder marked the article development-relevant.

Input:  data/validation/precision_final_coderA.csv, precision_final_coderB.csv
Output: results/diagnostics/reliability_alpha.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import krippendorff
import numpy as np
import pandas as pd

from _common import DATA_VALIDATION, RESULTS_DIAGNOSTICS, get_logger, write_rows

log = get_logger("reliability_alpha")

CODER_A_PATH = DATA_VALIDATION / "precision_final_coderA.csv"
CODER_B_PATH = DATA_VALIDATION / "precision_final_coderB.csv"
OUT_PATH = RESULTS_DIAGNOSTICS / "reliability_alpha.csv"

VARIABLES = {  # variable -> (column, valid codes, level of measurement)
    "development_relevance": ("label_development_relevant", [0, 1], "nominal"),
    "sentiment": ("label_sentiment_country", [-2, -1, 0, 1, 2], "ordinal"),
    "crisis_flag": ("label_crisis_flag", [0, 1], "nominal"),
}


def _coerce(series: pd.Series, domain: list) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return x.where(x.isin(domain), np.nan)


def main() -> None:
    a = pd.read_csv(CODER_A_PATH, dtype=str, encoding="utf-8-sig")
    b = pd.read_csv(CODER_B_PATH, dtype=str, encoding="utf-8-sig")
    if len(a) != len(b) or not (a["article_id"] == b["article_id"]).all():
        raise ValueError("Coder files are not aligned on article_id")

    devrel_a = _coerce(a["label_development_relevant"], [0, 1])
    devrel_b = _coerce(b["label_development_relevant"], [0, 1])
    rows = []
    for var, (col, domain, level) in VARIABLES.items():
        va, vb = _coerce(a[col], domain), _coerce(b[col], domain)
        if var == "crisis_flag":
            va, vb = va.where(devrel_a == 1), vb.where(devrel_b == 1)
        data = np.array([va.to_numpy(dtype=float), vb.to_numpy(dtype=float)])
        alpha = krippendorff.alpha(reliability_data=data, level_of_measurement=level)
        n_units = int(np.sum(~np.isnan(data).any(axis=0)))
        rows.append({"variable": var, "level_of_measurement": level, "n_units": n_units, "n_coders": 2,
                     "alpha": alpha})
        log.info(f"{var} ({level}): alpha={alpha:.4f}, n={n_units}")
    write_rows(OUT_PATH, ["variable", "level_of_measurement", "n_units", "n_coders", "alpha"], rows)


if __name__ == "__main__":
    main()
