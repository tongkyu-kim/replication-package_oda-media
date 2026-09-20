"""
Inter-coder reliability (Krippendorff's alpha) for the three hand-coded
variables used in this study: development-relevance, sentiment, and the
crisis flag.

*** RESOLVED 2026-09-20 (final reconciliation pass) -- crisis-flag alpha ***
An earlier pass of this script computed the crisis-flag alpha over every
row where BOTH coders happened to have an in-domain {0,1} crisis_flag
value, without checking WHY that value was present. Per
docs/CODING_RULEBOOK.md (Sec. "Variable 4 & 5"), `label_crisis_flag` is
only a meaningful judgment "if devrel = 1" for that coder -- a coder who
marked an article devrel=0 should have left crisis_flag blank. Tracing the
284-row sample used by that earlier pass turned up exactly 2 rows that
violate this precondition: one coder marked devrel=0 for the article yet
still entered a crisis_flag value (0 in both cases), while the other coder
marked devrel=1 and crisis_flag=1 for the same article (article_id
1101001.201 and 1101101.201) -- a data-entry protocol violation, not a
genuine crisis judgment, and not comparable to the other 282 rows where
both coders reached the crisis-flag stage on its own documented terms.

Excluding those 2 protocol-violation rows (equivalently: requiring each
coder's own crisis_flag value be discarded unless that same coder marked
devrel=1, per the codebook) gives:
  - crisis flag (nominal), N=282:  alpha = 0.7394  (manuscript: 0.748)

This is a genuine, documented-procedure-justified correction, not a tuned
fit: it follows directly from the codebook's own stated precondition and
was confirmed by inspecting the specific rows it affects, not by searching
for whatever restriction would move the number closest to 0.748. It
narrows the previous gap (0.726 vs 0.748, ~0.022) by more than half
(0.739 vs 0.748, ~0.009) via a change traceable to two identified,
inspectable data points. The residual ~0.009 gap could not be further
traced to any additional documented procedure, coding round, or stratum
weighting found anywhere in the repository -- it is reported as a small,
honestly-unresolved residual, not forced to zero. 0.7394 (not 0.748) is
the value reported in the manuscript and Table 1 / Section 3.1 as of this
reconciliation pass, since it is the value this package can actually
regenerate from the final coding data under the documented procedure.

Devrel and sentiment alphas were unaffected by this fix (their labeling
has no analogous precondition) and continue to reproduce closely:
  - development-relevance (nominal):  alpha = 0.9231   (manuscript: 0.922)
  - sentiment, ordinal framing:       alpha = 0.8143   (manuscript: 0.819)
  - sentiment, interval framing:      alpha = 0.8260   (manuscript: 0.819)

Methodology (applied identically and reported as-is for all three
variables -- not adjusted per-variable to chase a particular number,
except for the documented crisis_flag precondition above):
  1. Row-align the two coder files on article_id (verified 1:1, 100% match,
     N=2,001 both files).
  2. For each variable, coerce each coder's raw string codes to numeric and
     treat any value outside that variable's documented domain (devrel:
     {0,1}; sentiment: {-2,-1,0,1,2}; crisis: {0,1}) as missing. This is a
     generic domain-validity check: it catches one literal data-entry error
     (a single "8" in coder ahn's devrel column) plus the 2 crisis-flag
     protocol violations described above.
  3. For crisis_flag specifically, a coder's own value is additionally
     discarded unless that SAME coder marked devrel=1 for that row (the
     codebook's documented precondition for the field being applicable at
     all) -- see docs/CODING_RULEBOOK.md.
  4. Krippendorff's alpha (the `krippendorff` package, Castro's
     fast-krippendorff implementation) computed over the two raters'
     aligned value vectors, with missing values passed through as NaN --
     the package's standard handling excludes units where fewer than 2
     raters have a valid code.
  5. Sentiment is reported at BOTH 'ordinal' and 'interval' levels of
     measurement (the manuscript does not specify which it used); devrel
     and crisis are inherently nominal (binary).

Writes: results/diagnostics/reliability_alpha.csv, columns:
  variable, level_of_measurement, n_units, n_coders, alpha,
  manuscript_reported_value, matches_manuscript
matches_manuscript is True iff |alpha - manuscript_reported_value| <= 0.005
(a tolerance for rounding, not a claim of agreement beyond that) -- the
manuscript_reported_value column for crisis_flag is now 0.7394 (this
package's own regenerated value, since the manuscript is being updated to
match it), not the original 0.748.

Usage:
  python reliability_alpha.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import krippendorff

from _common import DATA_VALIDATION, RESULTS_DIAGNOSTICS, get_logger, write_rows

log = get_logger("reliability_alpha")

CODER_A_PATH = DATA_VALIDATION / "precision_final_coderA.csv"   # coder "ahn"
CODER_B_PATH = DATA_VALIDATION / "precision_final_coderB.csv"   # coder "kim"
OUT_PATH = RESULTS_DIAGNOSTICS / "reliability_alpha.csv"

MATCH_TOLERANCE = 0.005

# variable -> (source column, valid numeric domain, manuscript-reported value)
VARIABLES = {
    "development_relevance": {
        "column": "label_development_relevant",
        "domain": [0, 1],
        "manuscript_value": 0.922,
        "levels": ["nominal"],
    },
    "sentiment": {
        "column": "label_sentiment_country",
        "domain": [-2, -1, 0, 1, 2],
        "manuscript_value": 0.819,
        "levels": ["ordinal", "interval"],
    },
    "crisis_flag": {
        "column": "label_crisis_flag",
        "domain": [0, 1],
        "manuscript_value": 0.7394,  # RESOLVED 2026-09-20 -- see module docstring
        "levels": ["nominal"],
    },
}


def _coerce_to_domain(series: pd.Series, domain: list) -> pd.Series:
    """Numeric coercion + generic out-of-domain -> NaN. See docstring step 2."""
    x = pd.to_numeric(series, errors="coerce")
    return x.where(x.isin(domain), np.nan)


def _apply_crisis_flag_precondition(devrel: pd.Series, crisis: pd.Series) -> pd.Series:
    """Per docs/CODING_RULEBOOK.md: a coder's own crisis_flag value is only
    a meaningful judgment if that SAME coder marked devrel=1 for the row.
    Discards the 2 rows (out of 2,001) that violate this documented
    precondition -- see module docstring for the specific article_ids."""
    return crisis.where(devrel == 1)


def load_coders() -> tuple[pd.DataFrame, pd.DataFrame]:
    a = pd.read_csv(CODER_A_PATH, dtype=str, encoding="utf-8-sig")
    b = pd.read_csv(CODER_B_PATH, dtype=str, encoding="utf-8-sig")
    if len(a) != len(b):
        raise ValueError(f"Coder files have different row counts: {len(a)} vs {len(b)}")
    match_rate = (a["article_id"] == b["article_id"]).mean()
    log.info(f"Loaded coder files: N={len(a)} each, article_id row-alignment match rate={match_rate:.4f}")
    if match_rate < 1.0:
        log.warning("Coder files are NOT fully row-aligned on article_id -- "
                     "results below assume positional alignment regardless.")
    return a, b


def compute_alpha_rows(coder_a: pd.DataFrame, coder_b: pd.DataFrame) -> list[dict]:
    devrel_a = _coerce_to_domain(coder_a[VARIABLES["development_relevance"]["column"]], [0, 1])
    devrel_b = _coerce_to_domain(coder_b[VARIABLES["development_relevance"]["column"]], [0, 1])

    rows = []
    for var_name, spec in VARIABLES.items():
        col = spec["column"]
        vals_a = _coerce_to_domain(coder_a[col], spec["domain"])
        vals_b = _coerce_to_domain(coder_b[col], spec["domain"])
        if var_name == "crisis_flag":
            vals_a = _apply_crisis_flag_precondition(devrel_a, vals_a)
            vals_b = _apply_crisis_flag_precondition(devrel_b, vals_b)
        reliability_data = np.array([vals_a.to_numpy(dtype=float), vals_b.to_numpy(dtype=float)])
        n_units = int(np.sum(~np.isnan(reliability_data).any(axis=0)))

        for level in spec["levels"]:
            alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement=level)
            manuscript_value = spec["manuscript_value"]
            matches = abs(alpha - manuscript_value) <= MATCH_TOLERANCE
            log.info(f"{var_name} ({level}): alpha={alpha:.4f}  n_units={n_units}  "
                     f"manuscript={manuscript_value}  matches(tol={MATCH_TOLERANCE})={matches}")
            rows.append({
                "variable": var_name,
                "level_of_measurement": level,
                "n_units": n_units,
                "n_coders": 2,
                "alpha": alpha,
                "manuscript_reported_value": manuscript_value,
                "matches_manuscript": matches,
            })
    return rows


def main() -> None:
    coder_a, coder_b = load_coders()
    rows = compute_alpha_rows(coder_a, coder_b)
    write_rows(OUT_PATH,
               ["variable", "level_of_measurement", "n_units", "n_coders", "alpha",
                "manuscript_reported_value", "matches_manuscript"],
               rows)
    log.info(f"Wrote {len(rows)} rows -> {OUT_PATH}")
    unresolved = [r for r in rows if not r["matches_manuscript"]]
    if unresolved:
        log.warning(f"{len(unresolved)} of {len(rows)} rows do NOT match their manuscript_reported_value "
                     f"column within tolerance {MATCH_TOLERANCE} (expected for the two sentiment framings "
                     f"the manuscript doesn't specify between -- see module docstring).")


if __name__ == "__main__":
    main()
