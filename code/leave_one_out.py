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

Also includes (H4 family): crisis-timing LOO -- the pooled within-emergency
contemporaneous association (crisis media -> emergency ODA), ported from
pipeline/crisis_news/robustness_model6.py's leave_one_out check (manuscript
Appendix Table A5's crisis-timing row). Reuses the exact same leave_one_out()
helper below, applied to lib.emergency_panel's within-emergency panel
instead of the devrel one -- see run_crisis_timing_loo().

Writes: results/models/leave_one_out_results.csv -- if that file already
exists (e.g. a separate workstream's crisis-timing LOO row got there
first), this script's rows are APPENDED to whatever is already there,
reusing that file's own column schema, rather than overwriting it.
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
from lib import emergency_panel as ep
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


def run_crisis_timing_loo(crisis_annual: pd.DataFrame, focal: str) -> list[dict]:
    """H4 -- pooled within-emergency contemporaneous association (crisis
    media, months 0-11 of the commitment year -> emergency ODA),
    leave-one-country-out. Ported from pipeline/crisis_news/
    robustness_model6.py's leave_one_out check (manuscript Appendix Table
    A5's crisis-timing row) -- same focal spec as
    emergency_temporal.py's run_within_emergency "lag0" window: pooled
    crisis_media_lag0, log spec, country+year FE, 5 controls, clustered SE.
    Reuses the leave_one_out() helper above, applied to
    lib.emergency_panel's within-emergency panel instead of the devrel
    one."""
    analysis = "crisis_timing_lag0"
    res = leave_one_out(crisis_annual, "emergency_commitment_musd", focal, ep.CONTROL_COLS)
    if res is None:
        log.info(f"{analysis}: not estimable, skipped")
        return []
    res["analysis"] = analysis
    log.info(f"{analysis}: full_coef={res['full_sample_coef']:.4f} p={res['full_sample_p']:.4f} "
              f"n_excl={res['n_exclusions']} range=[{res['min_coef']:.4f},{res['max_coef']:.4f}] "
              f"sign_rev={res['n_sign_reversals']} sig_flips={res['n_sig_flips_at_p10']}")
    return [res]


def main() -> None:
    """Computes all 7 LOO analyses (persistent x3, sustained x3, crisis-
    timing x1) in a single run and OVERWRITES results/models/leave_one_out_
    results.csv -- this script is self-sufficient (it builds the crisis
    panel itself via lib/emergency_panel.py, not dependent on any other
    script having run first), so reruns are idempotent by construction:
    no append/merge step, no risk of duplicate rows across repeated runs."""
    country_month_long = pw.build_country_month_long()
    sentiment_month = dp.build_country_sentiment_month()
    annual_log = dp.build_country_annual(use_log=True, country_month_long=country_month_long,
                                          sentiment_month=sentiment_month)

    rows = run_persistent_loo(annual_log) + run_sustained_loo(annual_log)

    crisis_month_long = ep.build_crisis_month_long()
    crisis_annual, crisis_xcol = ep.build_country_annual(True, crisis_month_long, "crisis_articles", "lag0")
    rows += run_crisis_timing_loo(crisis_annual, crisis_xcol)

    write_rows(OUT_PATH, LOO_FIELDS, rows)
    log.info(f"Wrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
