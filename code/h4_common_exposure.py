"""
H4 -- emergency vs. non-emergency ODA under a COMMON development-relevant
media exposure (manuscript Section 4.5, Table 3 column [5], Figure 5,
Appendix Table A4 Panel A, Table 4).

H4: contemporaneous development-relevant media attention exhibits a
stronger positive association with emergency ODA commitments than with
non-emergency ODA commitments.

Design (lib/h4_panel.py): each country-year contributes an emergency and a
non-emergency observation -- mutually exclusive components that sum exactly
to the pooled outcome used for H1-H3 -- both matched to the IDENTICAL
DevRel media measures. Stacked PPML (pyfixest.fepois):

  E[y_ca] = exp(a_ca + d_ta + sum_w [b_w DevRel_w + th_w DevRel_w x Emerg_a]
                + sum_c [g_c X_c + h_c X_c x Emerg_a])

with country x aid-type and year x aid-type fixed effects (entity_x,
time_x), aid-type-specific control slopes, and SEs clustered on the
ORIGINAL country (so a recipient's emergency and non-emergency residuals
may correlate). For each window: non-emergency coefficient = b_w;
emergency-minus-non-emergency difference = th_w (the interaction, with its
own clustered SE); emergency coefficient = b_w + th_w (delta method, z test).

Specifications (all written to h4_common_exposure_results.csv):
  H4_primary     contemporaneous (k=-11..0) + 2-13 months, jointly -- the
                 headline H4 test (manuscript: 0.712 / 0.038 / diff 0.674)
  H4_extended    + 14-25 and 26-37 months, jointly (Figure 5)
  H4_roll24/36/60, H4_streak24/36/60   one measure per model (temporal
                 specificity of the contrast)
  H4_trim_top1pct  primary model without the top 1% of emergency
                 commitments (threshold from the complete-case primary
                 sample's emergency rows) -- sensitivity only
  H4_loglinear   primary model with ln(1+y) OLS and the same FE structure
                 (pyfixest.feols) -- estimator sensitivity only

The former H4 specification (crisis-related media -> emergency ODA vs.
DevRel media -> development ODA) is no longer estimated here; its
within-emergency crisis-media analyses remain in emergency_temporal.py as
exploratory diagnostics.

Ported (statistical logic identical) from the original research
repository's pipeline/devrel_baseline/h4r_primary_ppml.py,
h4r_rolling_streak.py, h4r_diagnostics.py and h4r_ci90.py.

Writes:
  results/models/h4_common_exposure_results.csv
  results/models/h4_common_exposure_wald_results.csv
  results/diagnostics/h4_common_exposure_data_checks.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats as sstats

from _common import (RESULTS_MODELS, RESULTS_DIAGNOSTICS, ModelResultRow, HYPOTHESIS_TEST_FIELDS,
                      get_logger, write_model_results, write_rows)
from lib import h4_panel as hp
from lib.regressions import hypothesis_test_row

log = get_logger("h4_common_exposure")

OUT_RESULTS = RESULTS_MODELS / "h4_common_exposure_results.csv"
OUT_WALD = RESULTS_MODELS / "h4_common_exposure_wald_results.csv"
OUT_CHECKS = RESULTS_DIAGNOSTICS / "h4_common_exposure_data_checks.csv"

CTRL = hp.CONTROL_COLS
LABELS = {"devrel_media_contemp": "contemporaneous", "devrel_media_m1_13": "m1_13",
          "devrel_media_m14_25": "m14_25", "devrel_media_m26_37": "m26_37",
          "devrel_media_roll24": "roll24", "devrel_media_roll36": "roll36", "devrel_media_roll60": "roll60",
          "streak_ge24": "streak_ge24", "streak_ge36": "streak_ge36", "streak_ge60": "streak_ge60"}
WINDOW_DESC = {"devrel_media_contemp": "contemporaneous (months 0-11 of the commitment year, k=-11..0)",
               "devrel_media_m1_13": "2-13 months pre-allocation", "devrel_media_m14_25": "14-25 months pre-allocation",
               "devrel_media_m26_37": "26-37 months pre-allocation",
               "devrel_media_roll24": "24-month rolling average (months 2-25)",
               "devrel_media_roll36": "36-month rolling average (months 2-37)",
               "devrel_media_roll60": "60-month rolling average (months 2-61)",
               "streak_ge24": "uninterrupted coverage >= 24 months", "streak_ge36": "uninterrupted coverage >= 36 months",
               "streak_ge60": "uninterrupted coverage >= 60 months"}
PRIMARY = ["devrel_media_contemp", "devrel_media_m1_13"]
EXTENDED = PRIMARY + ["devrel_media_m14_25", "devrel_media_m26_37"]
SINGLES = [("H4_roll24", "devrel_media_roll24"), ("H4_roll36", "devrel_media_roll36"),
           ("H4_roll60", "devrel_media_roll60"), ("H4_streak24", "streak_ge24"),
           ("H4_streak36", "streak_ge36"), ("H4_streak60", "streak_ge60")]
FE = "country x aid_type + year x aid_type"
CLUSTER = "country (original, pre-stack)"
CONTROLS_TXT = ";".join(CTRL) + " (each also interacted with the emergency indicator)"
SAMPLE = ("Stacked emergency + non-emergency ODA country-years (common DevRel media exposure), "
          "complete cases on media/outcome/controls; PPML drops observations separated by the FE")


def formula(windows: list[str], y: str = "y") -> str:
    terms = []
    for w in windows:
        terms += [w, f"{w}:is_emergency"]
    for c in CTRL:
        terms += [c, f"{c}:is_emergency"]
    return f"{y} ~ {' + '.join(terms)} | entity_x + time_x"


def complete_cases(d: pd.DataFrame, windows: list[str]) -> pd.DataFrame:
    return d.dropna(subset=windows + ["y"] + CTRL).copy()


def fit_ppml(d: pd.DataFrame, windows: list[str]):
    return pf.fepois(formula(windows), data=complete_cases(d, windows), vcov={"CRV1": "entity"},
                     separation_check=["fe"])


def lincomb(m, names: list[str], weights: list[float]) -> tuple[float, float]:
    idx = list(m.coef().index)
    pos = [idx.index(n) for n in names]
    b = m.coef().values[pos]
    V = m._vcov[np.ix_(pos, pos)]
    w = np.array(weights)
    return float(w @ b), float(np.sqrt(w @ V @ w))


def n_clusters(m) -> int:
    g = getattr(m, "_G", None)
    return int(g[0] if isinstance(g, (list, tuple, np.ndarray)) else g)


def rows_for(m, windows: list[str], prefix: str, outcome: str, transformation: str,
             quantities=("nonemergency", "emergency", "difference")) -> list[ModelResultRow]:
    tidy = m.tidy()
    p_col = "Pr(>|t|)"
    out = []
    for w in windows:
        lab = LABELS[w]
        vals = {
            "nonemergency": (float(tidy.loc[w, "Estimate"]), float(tidy.loc[w, "Std. Error"]),
                             float(tidy.loc[w, p_col])),
            "difference": (float(tidy.loc[f"{w}:is_emergency", "Estimate"]),
                           float(tidy.loc[f"{w}:is_emergency", "Std. Error"]),
                           float(tidy.loc[f"{w}:is_emergency", p_col])),
        }
        e, se = lincomb(m, [w, f"{w}:is_emergency"], [1.0, 1.0])
        vals["emergency"] = (e, se, float(2 * sstats.norm.sf(abs(e / se))))
        for q in quantities:
            c, s, p = vals[q]
            key = {"nonemergency": f"nonemergency_{lab}", "emergency": f"emergency_{lab}",
                   "difference": f"diff_emergency_minus_nonemergency_{lab}"}[q]
            out.append(ModelResultRow(
                specification_id=f"{prefix}__{key}", outcome=outcome, key_predictor=key,
                coefficient=c, se=s, p_value=p, n=int(m._N), n_entities=n_clusters(m),
                fe_structure=FE, cluster_level=CLUSTER, controls=CONTROLS_TXT, sample_definition=SAMPLE,
                timing_window=WINDOW_DESC[w], transformation=transformation))
    return out


def joint_wald(m, windows: list[str], spec_id: str) -> dict:
    idx = list(m.coef().index)
    names = [f"{w}:is_emergency" for w in windows]
    R = np.zeros((len(names), len(idx)))
    for i, n in enumerate(names):
        R[i, idx.index(n)] = 1.0
    wt = m.wald_test(R=R, q=np.zeros(len(names)), distribution="chi2")
    return hypothesis_test_row(spec_id, f"Wald: all {len(names)} emergency-minus-non-emergency interactions = 0",
                               float(wt["statistic"]), len(names), float(wt["pvalue"]),
                               " = ".join(names) + " = 0", int(m._N))


def main() -> None:
    stacked = hp.build_stacked_panel()
    checks = hp.verify_panel(stacked)
    write_rows(OUT_CHECKS, ["metric", "value"], [{"metric": k, "value": v} for k, v in checks.items()])
    log.info(f"Data checks: {checks}")
    if checks["media_exposure_mismatches"] != 0 or checks["max_abs_partition_error_musd"] > 1e-6:
        raise RuntimeError(f"H4 common-exposure / partition check failed: {checks}")

    ppml_tf = "none (PPML log link on level commitments, $M); ln(x+1) media predictors"
    rows: list[ModelResultRow] = []
    wald: list[dict] = []

    m = fit_ppml(stacked, PRIMARY)
    rows += rows_for(m, PRIMARY, "H4_primary", "commitment_musd (PPML level)", ppml_tf)
    wald.append(joint_wald(m, PRIMARY, "H4_primary"))
    m = fit_ppml(stacked, EXTENDED)
    rows += rows_for(m, EXTENDED, "H4_extended", "commitment_musd (PPML level)", ppml_tf)
    wald.append(joint_wald(m, EXTENDED, "H4_extended"))
    for prefix, w in SINGLES:
        m = fit_ppml(stacked, [w])
        rows += rows_for(m, [w], prefix, "commitment_musd (PPML level)", ppml_tf)

    # sensitivity: top-1% of emergency commitments removed
    cc = complete_cases(stacked, PRIMARY)
    thr = float(cc.loc[cc["aid_type"] == "emergency", "y"].quantile(0.99))
    trimmed = cc[~((cc["aid_type"] == "emergency") & (cc["y"] > thr))]
    log.info(f"Top-1% trim: threshold ${thr:.2f}M, {len(cc) - len(trimmed)} emergency rows removed")
    m = fit_ppml(trimmed, PRIMARY)
    rows += rows_for(m, ["devrel_media_contemp"], "H4_trim_top1pct", "commitment_musd (PPML level)",
                     ppml_tf + f"; emergency rows above ${thr:.2f}M removed")

    # sensitivity: log-linear counterpart, same FE structure
    cc["log_y"] = np.log1p(cc["y"])
    m = pf.feols(formula(PRIMARY, "log_y"), data=cc, vcov={"CRV1": "entity"})
    rows += rows_for(m, PRIMARY, "H4_loglinear", "ln(1 + commitment_musd) (OLS)", "ln(x+1) outcome and media")

    write_model_results(OUT_RESULTS, rows)
    write_rows(OUT_WALD, HYPOTHESIS_TEST_FIELDS, wald)
    for r in rows:
        if r.specification_id.startswith("H4_primary") and "contemporaneous" in r.key_predictor:
            log.info(f"{r.key_predictor}: {r.coefficient:.4f} (se {r.se:.4f}) p={r.p_value:.2e} n={r.n}")
    log.info(f"Wrote {len(rows)} rows -> {OUT_RESULTS}; {len(wald)} Wald rows -> {OUT_WALD}")


if __name__ == "__main__":
    main()
