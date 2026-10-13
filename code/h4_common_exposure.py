"""
H4: emergency vs. non-emergency ODA under common development-relevant media
exposure (Section 4.5; Table 3 column [5]; Table 4; Appendix Table A4;
Figure 5).

Stacked PPML (lib/h4_panel.py): each country-year contributes an emergency
and a non-emergency observation, both matched to the same development-
relevant media measures,

  E[y_ca] = exp(a_ca + d_ta + sum_w [b_w M_w + th_w M_w x Emerg_a]
                + sum_c [g_c X_c + h_c X_c x Emerg_a]),

with country x aid-type and year x aid-type fixed effects, aid-type-specific
control slopes, and standard errors clustered by country. For each window:
non-emergency coefficient = b_w; emergency-minus-non-emergency difference =
th_w; emergency coefficient = b_w + th_w (delta method).

Specifications:
  H4_primary           contemporaneous and 2-13-month windows, jointly
  H4_extended          adds the 14-25- and 26-37-month windows
  H4_roll*/H4_streak*  one persistent or sustained measure per model
  H4_loglinear         primary model estimated by OLS on ln(1 + y)

Output: results/models/h4_common_exposure_results.csv,
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

from _common import RESULTS_MODELS, RESULTS_DIAGNOSTICS, ModelResultRow, get_logger, write_model_results, write_rows
from lib import h4_panel as hp

log = get_logger("h4_common_exposure")

OUT_RESULTS = RESULTS_MODELS / "h4_common_exposure_results.csv"
OUT_CHECKS = RESULTS_DIAGNOSTICS / "h4_common_exposure_data_checks.csv"

CTRL = hp.CONTROL_COLS
LABELS = {"devrel_media_contemp": "contemporaneous", "devrel_media_m1_13": "m1_13",
          "devrel_media_m14_25": "m14_25", "devrel_media_m26_37": "m26_37",
          "devrel_media_roll24": "roll24", "devrel_media_roll36": "roll36", "devrel_media_roll60": "roll60",
          "streak_ge24": "streak_ge24", "streak_ge36": "streak_ge36", "streak_ge60": "streak_ge60"}
WINDOW_DESC = {"devrel_media_contemp": "contemporaneous (months 0-11 of the commitment year)",
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
CLUSTER = "country"
CONTROLS_TXT = ";".join(CTRL) + " (each also interacted with the emergency indicator)"
SAMPLE = ("Stacked emergency and non-emergency ODA country-years, complete cases; "
          "PPML drops observations separated by the fixed effects")


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


def rows_for(m, windows: list[str], prefix: str, outcome: str, transformation: str) -> list[ModelResultRow]:
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
        for q in ("nonemergency", "emergency", "difference"):
            c, s, p = vals[q]
            key = {"nonemergency": f"nonemergency_{lab}", "emergency": f"emergency_{lab}",
                   "difference": f"diff_emergency_minus_nonemergency_{lab}"}[q]
            out.append(ModelResultRow(
                specification_id=f"{prefix}__{key}", outcome=outcome, key_predictor=key,
                coefficient=c, se=s, p_value=p, n=int(m._N), n_entities=n_clusters(m),
                fe_structure=FE, cluster_level=CLUSTER, controls=CONTROLS_TXT, sample_definition=SAMPLE,
                timing_window=WINDOW_DESC[w], transformation=transformation))
    return out


def main() -> None:
    stacked = hp.build_stacked_panel()
    checks = hp.verify_panel(stacked)
    write_rows(OUT_CHECKS, ["metric", "value"], [{"metric": k, "value": v} for k, v in checks.items()])
    if checks["media_exposure_mismatches"] != 0 or checks["max_abs_partition_error_musd"] > 1e-6:
        raise RuntimeError(f"H4 panel check failed: {checks}")

    ppml_tf = "PPML (log link, commitments in $M); ln(x+1) media"
    outcome = "commitment_musd (PPML level)"
    rows: list[ModelResultRow] = []
    rows += rows_for(fit_ppml(stacked, PRIMARY), PRIMARY, "H4_primary", outcome, ppml_tf)
    rows += rows_for(fit_ppml(stacked, EXTENDED), EXTENDED, "H4_extended", outcome, ppml_tf)
    for prefix, w in SINGLES:
        rows += rows_for(fit_ppml(stacked, [w]), [w], prefix, outcome, ppml_tf)

    cc = complete_cases(stacked, PRIMARY)
    cc["log_y"] = np.log1p(cc["y"])
    m = pf.feols(formula(PRIMARY, "log_y"), data=cc, vcov={"CRV1": "entity"})
    rows += rows_for(m, PRIMARY, "H4_loglinear", "ln(1 + commitment_musd) (OLS)", "ln(x+1) outcome and media")

    write_model_results(OUT_RESULTS, rows)
    log.info(f"Wrote {len(rows)} rows -> {OUT_RESULTS}")


if __name__ == "__main__":
    main()
