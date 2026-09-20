"""
H4 -- development ODA vs. emergency ODA timing.

Two families of analysis, both using crisis-related news coverage as the
media measure and emergency/humanitarian-reclassified ODA
(emergency_country_year.csv) as the outcome on the within-emergency side,
plus development ODA (via lib.devrel_panel) as the comparison process on
the PPML side.

(1) WITHIN-EMERGENCY association -- secondary/supplementary analysis, NOT
    the formal H4 test. Pooled crisis media -> emergency ODA, country +
    year FE, country-clustered SE, 5 controls, log1p(y)-OLS (the LOG spec
    only -- the original script also computes a levels/no-transformation
    spec purely as an internal diagnostic never reported in the
    manuscript, and that variant is not ported here):
      (a) contemporaneous window (k=-11..0, i.e. months 0-11 of the
          commitment year) -- the featured/primary spec for this
          analysis (Table 3 column [5]).
      (b) forward/pre-allocation window (k=2..13) -- reported for
          comparison (Appendix Table A4).
      (c) per-crisis-type heterogeneity (disaster/climate/conflict/
          health/food/refugee), contemporaneous window, each its own
          standalone regression -- Appendix Table A6 -- plus a joint Wald
          test (all 6 jointly zero, and all 6 jointly equal) computed from
          ONE simultaneous regression with all 6 crisis-type media
          measures entered together (the only way to support a valid
          joint test; the 6 standalone per-type regressions cannot, since
          each is its own separate model).

(2) The formal H4 test (manuscript Figure 5): development and emergency
    ODA observations STACKED into one panel with an aid-type indicator,
    four temporal windows (contemporaneous, 2-13, 14-25, 26-37 months)
    entered as main effects AND as interactions with the aid-type
    indicator, estimated by Poisson pseudo-maximum likelihood (PPML, via
    `pyfixest.fepois`) with aid-type-specific fixed effects (country x
    aid-type, year x aid-type) and SE clustered on the ORIGINAL country.
    For each window: development coefficient = the uninteracted term;
    emergency-minus-development difference = the interaction term (its own
    clustered SE, taken directly from the fit); emergency coefficient =
    development + interaction, via the delta method. The original script
    does not report a p-value for the two combined (delta-method) dev/emg
    estimates, only for the difference -- this port fills that field in
    via a two-sided normal (z) test on the same delta-method SE, the same
    z-test approach the original script itself uses for its own timing-
    contrast check (see the joint Wald tests below).

key_predictor naming for the PPML section
(emergency_ppml_contrast_results.csv): "development_{window}",
"emergency_{window}", "diff_emergency_minus_development_{window}" for
window in {contemporaneous, m1_13, m14_25, m26_37} ("contemporaneous"
spelled out for the k=-11..0 window; the other three keep this study's
established "m{start}_{end}" window labels). specification_id is
"H4_ppml_contrast__{key_predictor}".

Leave-one-country-out robustness for (1a) lives in code/leave_one_out.py,
not here (shared LOO file across the whole package).

Ported unchanged (statistical logic identical) from the original research
repository's:
  - pipeline/crisis_news/estimate_crisis_model.py (within-emergency
    pooled/forward/per-type/joint-Wald sections)
  - pipeline/aid_type_temporal/ppml_diagnostic/sbx_ppml_windows.py +
    sbx_ppml_fit.py (stacked PPML temporal-contrast test)

Writes:
  results/models/emergency_temporal_results.csv       (items 1a/1b/1c coefficients)
  results/models/emergency_temporal_wald_results.csv   (item 1c's joint crisis-type
                                                         Wald tests, and item 2's own
                                                         joint/timing-contrast Wald tests)
  results/models/emergency_ppml_contrast_results.csv   (item 2: dev/emg/diff, all 4 windows)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats as sstats

from _common import (RESULTS_MODELS, ModelResultRow, HYPOTHESIS_TEST_FIELDS, get_logger,
                      write_model_results, write_rows)
from lib import emergency_panel as ep
from lib.regressions import fit_fe_ols, result_rows_from_fit, wald_zero, hypothesis_test_row

log = get_logger("emergency_temporal")

OUT_RESULTS = RESULTS_MODELS / "emergency_temporal_results.csv"
OUT_WALD = RESULTS_MODELS / "emergency_temporal_wald_results.csv"
OUT_PPML = RESULTS_MODELS / "emergency_ppml_contrast_results.csv"

WINDOW_LABELS = {
    "lag0": "contemporaneous (months 0-11 of the commitment year, k=-11..0)",
    "m1_13": "2-13 months pre-allocation (forward/pre-allocation-style window)",
}

# ---------------------------------------------------------------------------
# (1a)/(1b) pooled within-emergency association
# ---------------------------------------------------------------------------

SPEC_IDS = {"lag0": "H4_within_emergency_contemporaneous", "m1_13": "H4_within_emergency_forward"}


def run_within_emergency(crisis_month_long: pd.DataFrame) -> list[ModelResultRow]:
    """Item (1a)/(1b): pooled crisis media -> emergency ODA, contemporaneous
    and forward windows."""
    rows: list[ModelResultRow] = []
    for window in ["lag0", "m1_13"]:
        annual, xcol = ep.build_country_annual(True, crisis_month_long, "crisis_articles", window)
        res = fit_fe_ols(annual, "emergency_commitment_musd", [xcol] + ep.CONTROL_COLS,
                          verbose_label=f"within-emergency, {window}")
        rows += result_rows_from_fit(
            res, [xcol], specification_prefix=SPEC_IDS[window],
            outcome="ln_emergency_commitment_musd", controls=ep.CONTROL_COLS,
            sample_definition="Emergency-ODA country-year panel (pooled crisis media -> emergency-reclassified "
                               "ODA), complete cases on media/outcome/controls",
            timing_window=WINDOW_LABELS[window], transformation="ln(x+1)")
        if res is not None:
            log.info(f"Within-emergency [{window}]: coef={res.params[xcol]:.4f} se={res.std_errors[xcol]:.4f} "
                      f"p={res.pvalues[xcol]:.4f} n={int(res.nobs)}")
    return rows


# ---------------------------------------------------------------------------
# (1c) per-crisis-type heterogeneity + joint Wald tests
# ---------------------------------------------------------------------------

def _wald_all_equal(res, coef_names: list[str]) -> tuple[float, int, float]:
    """H0: all named coefficients are pairwise equal, as a SINGLE joint
    restriction (beta_2=beta_1, beta_3=beta_1, ...), not
    len(coef_names)-choose-2 separate pairwise tests. Ported from
    estimate_crisis_model.py's _wald_all_equal -- kept local to this script
    rather than added to the shared lib.regressions, which only offers a
    single-pair wald_equal."""
    params = list(res.params.index)
    R = np.zeros((len(coef_names) - 1, len(params)))
    for i, name in enumerate(coef_names[1:]):
        R[i, params.index(coef_names[0])] = 1.0
        R[i, params.index(name)] = -1.0
    wt = res.wald_test(restriction=R, value=np.zeros(len(coef_names) - 1))
    return float(wt.stat), int(wt.df), float(wt.pval)


def run_crisis_type_heterogeneity(crisis_month_long: pd.DataFrame) -> tuple[list[ModelResultRow], list[dict]]:
    """Item (1c): six standalone per-crisis-type regressions (contemporaneous
    window, log spec) plus a joint Wald test across all 6 types from ONE
    simultaneous regression."""
    rows: list[ModelResultRow] = []
    wald_rows: list[dict] = []

    for ct in ep.CRISIS_TYPES:
        annual, xcol = ep.build_country_annual(True, crisis_month_long, ct, "lag0")
        res = fit_fe_ols(annual, "emergency_commitment_musd", [xcol] + ep.CONTROL_COLS,
                          verbose_label=f"crisis type: {ct}")
        if res is None:
            log.warning(f"Crisis type {ct}: insufficient data, skipping")
            continue
        # xcol is generically named "crisis_media_lag0" regardless of crisis
        # type (see build_country_annual's docstring); relabeled per-type
        # here purely for the OUTPUT row's key_predictor, via a rename on a
        # throwaway fit object is unnecessary -- result_rows_from_fit reads
        # coefficients off `res` by the xs names passed to fit_fe_ols, so we
        # pass xcol itself and then relabel only the emitted row below.
        type_rows = result_rows_from_fit(
            res, [xcol], specification_prefix=f"H4_crisis_type_{ct}",
            outcome="ln_emergency_commitment_musd", controls=ep.CONTROL_COLS,
            sample_definition=f"Emergency-ODA country-year panel, {ct}-type crisis media only, complete cases",
            timing_window=WINDOW_LABELS["lag0"], transformation="ln(x+1)")
        for r in type_rows:
            r.key_predictor = f"crisis_{ct}_lag0"
            r.specification_id = f"H4_crisis_type_{ct}__crisis_{ct}_lag0"
        rows += type_rows
        log.info(f"Crisis type [{ct}]: coef={res.params[xcol]:.4f} se={res.std_errors[xcol]:.4f} "
                  f"p={res.pvalues[xcol]:.4f} n={int(res.nobs)}")

    # Joint Wald tests: ONE simultaneous regression, all 6 crisis-type media
    # measures entered together (contemporaneous window, log spec).
    annual, _ = ep.build_country_annual(True, crisis_month_long, "crisis_articles", "lag0")  # base grid only
    xcols = []
    for ct in ep.CRISIS_TYPES:
        a, generic_xcol = ep.build_country_annual(True, crisis_month_long, ct, "lag0")
        xcol = f"crisis_{ct}_lag0"
        a = a.rename(columns={generic_xcol: xcol})
        annual = annual.merge(a[["entity", "year", xcol]], on=["entity", "year"], how="left")
        xcols.append(xcol)

    res_joint = fit_fe_ols(annual, "emergency_commitment_musd", xcols + ep.CONTROL_COLS,
                            verbose_label="joint crisis-type regression (simultaneous)")
    if res_joint is not None:
        stat, dfree, pval = wald_zero(res_joint, xcols)
        wald_rows.append(hypothesis_test_row(
            "H4_crisis_type_joint", "Wald: all 6 crisis-type coefficients = 0 (joint, simultaneous regression)",
            stat, dfree, pval, "b_disaster=b_climate=b_conflict=b_health=b_food=b_refugee=0", int(res_joint.nobs)))
        log.info(f"Joint crisis-type Wald (all=0): chi2({dfree})={stat:.3f} p={pval:.5f} n={int(res_joint.nobs)}")

        stat2, dfree2, pval2 = _wald_all_equal(res_joint, xcols)
        wald_rows.append(hypothesis_test_row(
            "H4_crisis_type_joint", "Wald: all 6 crisis-type coefficients are equal (single joint restriction, "
                                     "not pairwise)",
            stat2, dfree2, pval2, "b_disaster=b_climate=b_conflict=b_health=b_food=b_refugee", int(res_joint.nobs)))
        log.info(f"Joint crisis-type Wald (all equal): chi2({dfree2})={stat2:.3f} p={pval2:.5f}")

    return rows, wald_rows


# ---------------------------------------------------------------------------
# (2) formal H4 test -- stacked PPML temporal contrast (Figure 5)
# ---------------------------------------------------------------------------

PPML_WINDOW_NAMES = ["contemp", "m1_13", "m14_25", "m26_37"]
PPML_KEY_LABELS = {"contemp": "contemporaneous", "m1_13": "m1_13", "m14_25": "m14_25", "m26_37": "m26_37"}
PPML_WINDOW_DESC = {
    "contemp": "contemporaneous (k=-11..0)",
    "m1_13": "2-13 months pre-allocation (window k=2-13)",
    "m14_25": "14-25 months pre-allocation (window k=14-25)",
    "m26_37": "26-37 months pre-allocation (window k=26-37)",
}


def _ppml_formula(ctrl: list[str]) -> str:
    terms = []
    for w in PPML_WINDOW_NAMES:
        terms.append(w)
        terms.append(f"{w}:is_emg")
    for c in ctrl:
        terms.append(c)
        terms.append(f"{c}:is_emg")
    rhs = " + ".join(terms)
    return f"y ~ {rhs} | entity_x + time_x"


def _wald_lincomb(m, coef_names: list[str], weights: list[float]) -> tuple[float, float]:
    """Point estimate + SE for a linear combination of named coefficients,
    via the delta method using the model's own (clustered) vcov. Ported
    unchanged from sbx_ppml_fit.py's wald_lincomb."""
    idx = list(m.coef().index)
    pos = [idx.index(c) for c in coef_names]
    b = m.coef().values[pos]
    V = m._vcov[np.ix_(pos, pos)]
    w = np.array(weights)
    est = float(w @ b)
    var = float(w @ V @ w)
    se = float(np.sqrt(var))
    return est, se


def run_ppml_contrast() -> tuple[list[ModelResultRow], list[dict]]:
    """Item (2) -- the formal H4 test / Figure 5."""
    stacked = ep.build_stacked_ppml_panel()
    fml = _ppml_formula(ep.CONTROL_COLS)
    log.info(f"PPML formula: {fml}")
    m = pf.fepois(fml, data=stacked, vcov={"CRV1": "entity"}, separation_check=["fe"])
    n_total = int(m._N)
    n_entities = int(stacked["entity"].nunique())
    tidy = m.tidy()
    log.info(f"PPML fit: N={n_total} (dev={int((stacked['aid_type'] == 'dev').sum())}, "
              f"emg={int((stacked['aid_type'] == 'emg').sum())}), countries={n_entities}")

    sample_def = ("Development + emergency ODA country-year panels stacked; aid-type-specific "
                   "country x aid-type and year x aid-type fixed effects (entity_x, time_x); "
                   "SE clustered on the ORIGINAL country (not entity_x)")
    transformation_dev_emg = "none (PPML log link on level y; ln(x+1) media predictors)"
    transformation_diff = transformation_dev_emg + "; interaction term = direct emergency-minus-development difference"

    rows: list[ModelResultRow] = []
    for w in PPML_WINDOW_NAMES:
        label = PPML_KEY_LABELS[w]
        dev_est, dev_se = _wald_lincomb(m, [w], [1.0])
        emg_est, emg_se = _wald_lincomb(m, [w, f"{w}:is_emg"], [1.0, 1.0])
        diff_est = float(tidy.loc[f"{w}:is_emg", "Estimate"])
        diff_se = float(tidy.loc[f"{w}:is_emg", "Std. Error"])
        diff_p = float(tidy.loc[f"{w}:is_emg", "Pr(>|t|)"])

        # The original script does not report a p-value for the combined
        # dev/emg (delta-method) estimates, only for the difference (taken
        # directly from the PPML fit's own t-test). Filled in here via a
        # two-sided normal (z) test on the same delta-method SE -- the same
        # z-test approach the original script itself uses for its own
        # timing-contrast check below.
        dev_p = float(2 * (1 - sstats.norm.cdf(abs(dev_est / dev_se)))) if dev_se else float("nan")
        emg_p = float(2 * (1 - sstats.norm.cdf(abs(emg_est / emg_se)))) if emg_se else float("nan")

        common_kwargs = dict(
            outcome="oda_commitment_musd (PPML level)", n=n_total, n_entities=n_entities,
            fe_structure="country x aid_type + year x aid_type",
            cluster_level="country (original, pre-stack)", controls=";".join(ep.CONTROL_COLS),
            sample_definition=sample_def, timing_window=PPML_WINDOW_DESC[w],
        )
        rows.append(ModelResultRow(
            specification_id=f"H4_ppml_contrast__development_{label}",
            key_predictor=f"development_{label}", coefficient=dev_est, se=dev_se, p_value=dev_p,
            transformation=transformation_dev_emg, **common_kwargs))
        rows.append(ModelResultRow(
            specification_id=f"H4_ppml_contrast__emergency_{label}",
            key_predictor=f"emergency_{label}", coefficient=emg_est, se=emg_se, p_value=emg_p,
            transformation=transformation_dev_emg, **common_kwargs))
        rows.append(ModelResultRow(
            specification_id=f"H4_ppml_contrast__diff_emergency_minus_development_{label}",
            key_predictor=f"diff_emergency_minus_development_{label}", coefficient=diff_est, se=diff_se,
            p_value=diff_p, transformation=transformation_diff, **common_kwargs))
        log.info(f"PPML [{w}]: dev={dev_est:.4f}(se={dev_se:.4f}) emg={emg_est:.4f}(se={emg_se:.4f}) "
                  f"diff={diff_est:.4f}(se={diff_se:.4f}) p={diff_p:.2e}")

    # Joint hypothesis tests the original script reports alongside the
    # window-specific effects table (printed there, not written to a CSV in
    # the original -- captured here as HYPOTHESIS_TEST_FIELDS rows in this
    # script's own wald-results file, for completeness of the port).
    wald_rows: list[dict] = []
    idx = list(m.coef().index)
    int_names = [f"{w}:is_emg" for w in PPML_WINDOW_NAMES]
    R = np.zeros((len(int_names), len(idx)))
    for i, name in enumerate(int_names):
        R[i, idx.index(name)] = 1.0
    wt = m.wald_test(R=R, q=np.zeros(len(int_names)), distribution="chi2")
    wald_rows.append(hypothesis_test_row(
        "H4_ppml_contrast_joint",
        "Wald: temporal response profiles are equal (all 4 emergency-minus-development interactions = 0)",
        float(wt["statistic"]), len(int_names), float(wt["pvalue"]),
        "int_contemp=int_m1_13=int_m14_25=int_m26_37=0", n_total))

    R2 = np.zeros((1, len(idx)))
    R2[0, idx.index("contemp:is_emg")] = 1.0
    R2[0, idx.index("m14_25:is_emg")] = -1.0
    wt2 = m.wald_test(R=R2, q=np.zeros(1), distribution="chi2")
    wald_rows.append(hypothesis_test_row(
        "H4_ppml_contrast_joint",
        "Wald: timing contrast, (emg_contemp - emg_m14_25) - (dev_contemp - dev_m14_25) = 0 "
        "[algebraically = int_contemp - int_m14_25]",
        float(wt2["statistic"]), 1, float(wt2["pvalue"]), "int_contemp - int_m14_25 = 0", n_total))
    log.info(f"PPML joint Wald (4 interactions=0): chi2({len(int_names)})={float(wt['statistic']):.3f} "
              f"p={float(wt['pvalue']):.4e}")
    log.info(f"PPML timing-contrast Wald: chi2(1)={float(wt2['statistic']):.3f} p={float(wt2['pvalue']):.4f}")

    return rows, wald_rows


def main() -> None:
    crisis_month_long = ep.build_crisis_month_long()

    results_rows = run_within_emergency(crisis_month_long)
    type_rows, type_wald_rows = run_crisis_type_heterogeneity(crisis_month_long)
    results_rows += type_rows
    write_model_results(OUT_RESULTS, results_rows)
    log.info(f"Wrote {len(results_rows)} rows -> {OUT_RESULTS}")

    ppml_rows, ppml_wald_rows = run_ppml_contrast()
    write_model_results(OUT_PPML, ppml_rows)
    log.info(f"Wrote {len(ppml_rows)} rows -> {OUT_PPML}")

    wald_rows = type_wald_rows + ppml_wald_rows
    write_rows(OUT_WALD, HYPOTHESIS_TEST_FIELDS, wald_rows)
    log.info(f"Wrote {len(wald_rows)} rows -> {OUT_WALD}")


if __name__ == "__main__":
    main()
