"""
Leave-one-country-out robustness for the persistent-attention (H2:
roll24/36/60) and sustained-coverage (H3: streak >=24/36/60mo)
specifications: refit each preferred spec once per country, excluding
that country, and summarize how much the focal coefficient moves.

This does not fit the standard ModelResultRow schema (one row = one
estimate); it reports a DISTRIBUTION of refit estimates per analysis, so
it uses its own simple schema instead (see LOO_FIELDS below).

Ported unchanged (statistical logic identical) from the original research
repository's pipeline/devrel_baseline/robustness_models_2to5.py
(leave_one_out(), Model3 battery -- roll24/roll36/roll60) and
pipeline/devrel_baseline/robustness_model4_streak_loo.py (streak
>=24/36/60mo). Both source scripts report the coefficient range and a
significance-flip count (at p<0.10); this port additionally derives a
sign-reversal count directly from the same per-country refit coefficients
each source script already collects (not a new statistic -- both source
scripts' own printed diagnostics describe "N sign-flips" alongside the
N sig-flips they write to CSV; this just also persists it as its own
column here).

Also includes H4 (revised 2026-10-12): leave-one-country-out for the
contemporaneous emergency-minus-non-emergency difference in the stacked
PPML model with common development-relevant media exposure
(h4_common_exposure.py's primary specification; manuscript Appendix Table
A5's H4 row). Ported from the original research repository's
pipeline/devrel_baseline/h4r_ci90.py. This replaces the former
crisis-timing row (crisis media -> emergency ODA), which belonged to the
former H4 specification. See run_h4_loo().

Writes: results/models/leave_one_out_results.csv (overwritten on every run;
see main()).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from _common import RESULTS_MODELS, get_logger, write_rows
from lib import devrel_panel as dp
from lib import h4_panel as hp
from lib import panel_windows as pw
from lib.regressions import fit_fe_ols

log = get_logger("leave_one_out")

OUT_PATH = RESULTS_MODELS / "leave_one_out_results.csv"

LOO_FIELDS = [
    "analysis",
    "focal_term",
    "full_sample_coef",
    "full_sample_p",
    "n_exclusions",
    "min_coef",
    "max_coef",
    "n_sign_reversals",
    "n_sig_flips_at_p10",
]

# (analysis label, focal predictor column)
PERSISTENT_ANALYSES = [
    ("persistent_roll24", "devrel_media_roll24"),
    ("persistent_roll36", "devrel_media_roll36"),
    ("persistent_roll60", "devrel_media_roll60"),
]
SUSTAINED_THRESHOLDS = [24, 36, 60]


def leave_one_out(d: pd.DataFrame, y: str, focal: str, other_xs: list[str]) -> dict | None:
    """Refit `y ~ focal + other_xs` (entity+year FE, country-clustered SE)
    once on the full sample, then once per excluded country. Returns the
    full-sample estimate plus the distribution of refit coefficients."""
    xs = [focal] + other_xs
    dd = d.dropna(subset=[y] + xs).copy()
    full = fit_fe_ols(dd, y, xs, verbose_label=f"LOO full-sample ({focal})")
    if full is None:
        return None
    full_coef, full_p = float(full.params[focal]), float(full.pvalues[focal])
    full_sig = full_p < 0.10

    coefs, sig_flips = [], 0
    for c in sorted(dd["entity"].unique()):
        sub = dd[dd["entity"] != c]
        res = fit_fe_ols(sub, y, xs)
        if res is None:
            continue
        coefs.append(float(res.params[focal]))
        if (float(res.pvalues[focal]) < 0.10) != full_sig:
            sig_flips += 1
    coefs = np.array(coefs)
    n_sign_rev = int(np.sum(np.sign(coefs) != np.sign(full_coef)))

    return {
        "focal_term": focal,
        "full_sample_coef": full_coef,
        "full_sample_p": full_p,
        "n_exclusions": len(coefs),
        "min_coef": float(coefs.min()),
        "max_coef": float(coefs.max()),
        "n_sign_reversals": n_sign_rev,
        "n_sig_flips_at_p10": sig_flips,
    }


def run_persistent_loo(annual_log: pd.DataFrame) -> list[dict]:
    rows = []
    for analysis, focal in PERSISTENT_ANALYSES:
        res = leave_one_out(annual_log, "devrel_commitment_musd", focal, dp.CONTROL_COLS)
        if res is None:
            log.info(f"{analysis}: not estimable, skipped")
            continue
        res["analysis"] = analysis
        rows.append(res)
        log.info(f"{analysis}: full_coef={res['full_sample_coef']:.4f} p={res['full_sample_p']:.4f} "
                  f"n_excl={res['n_exclusions']} range=[{res['min_coef']:.4f},{res['max_coef']:.4f}] "
                  f"sign_rev={res['n_sign_reversals']} sig_flips={res['n_sig_flips_at_p10']}")
    return rows


def run_sustained_loo(annual_log: pd.DataFrame) -> list[dict]:
    rows = []
    for thresh in SUSTAINED_THRESHOLDS:
        analysis = f"sustained_streak{thresh}"
        col = f"streak_ge{thresh}"
        d = annual_log.copy()
        d[col] = (d["consecutive_periods_covered"] >= thresh).astype(float)
        res = leave_one_out(d, "devrel_commitment_musd", col, dp.CONTROL_COLS)
        if res is None:
            log.info(f"{analysis}: not estimable, skipped")
            continue
        res["analysis"] = analysis
        rows.append(res)
        log.info(f"{analysis}: full_coef={res['full_sample_coef']:.4f} p={res['full_sample_p']:.4f} "
                  f"n_excl={res['n_exclusions']} range=[{res['min_coef']:.4f},{res['max_coef']:.4f}] "
                  f"sign_rev={res['n_sign_reversals']} sig_flips={res['n_sig_flips_at_p10']}")
    return rows


def run_h4_loo(country_month_long: pd.DataFrame, sentiment_month: pd.DataFrame) -> list[dict]:
    """H4 -- contemporaneous emergency-minus-non-emergency difference
    (interaction term) in h4_common_exposure.py's primary stacked PPML
    model, refit once per excluded country on the complete-case primary
    sample. Significance flips use the interaction's own p-value at 0.10."""
    import h4_common_exposure as h4
    focal = "devrel_media_contemp:is_emergency"
    cc = h4.complete_cases(hp.build_stacked_panel(country_month_long, sentiment_month), h4.PRIMARY)

    def fit(d):
        tidy = h4.fit_ppml(d, h4.PRIMARY).tidy()
        return float(tidy.loc[focal, "Estimate"]), float(tidy.loc[focal, "Pr(>|t|)"])

    full_coef, full_p = fit(cc)
    full_sig = full_p < 0.10
    coefs, sig_flips = [], 0
    for c in sorted(cc["entity"].unique()):
        coef, p = fit(cc[cc["entity"] != c])
        coefs.append(coef)
        sig_flips += int((p < 0.10) != full_sig)
    coefs = np.array(coefs)
    res = {"analysis": "h4_common_exposure_contemp_diff", "focal_term": focal,
           "full_sample_coef": full_coef, "full_sample_p": full_p, "n_exclusions": len(coefs),
           "min_coef": float(coefs.min()), "max_coef": float(coefs.max()),
           "n_sign_reversals": int(np.sum(np.sign(coefs) != np.sign(full_coef))),
           "n_sig_flips_at_p10": sig_flips}
    log.info(f"{res['analysis']}: full_coef={full_coef:.4f} p={full_p:.2e} n_excl={len(coefs)} "
             f"range=[{res['min_coef']:.4f},{res['max_coef']:.4f}] sign_rev={res['n_sign_reversals']} "
             f"sig_flips={sig_flips}")
    return [res]


def main() -> None:
    """Computes all 7 LOO analyses (persistent x3, sustained x3, H4 x1) in
    a single run and OVERWRITES results/models/leave_one_out_results.csv --
    this script is self-sufficient (it builds the H4 stacked panel itself
    via lib/h4_panel.py), so reruns are idempotent by construction."""
    country_month_long = pw.build_country_month_long()
    sentiment_month = dp.build_country_sentiment_month()
    annual_log = dp.build_country_annual(use_log=True, country_month_long=country_month_long,
                                          sentiment_month=sentiment_month)

    rows = run_persistent_loo(annual_log) + run_sustained_loo(annual_log)

    rows += run_h4_loo(country_month_long, sentiment_month)

    write_rows(OUT_PATH, LOO_FIELDS, rows)
    log.info(f"Wrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
