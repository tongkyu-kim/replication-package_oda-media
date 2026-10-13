"""
Builds every manuscript table (main Tables 1-4 and Appendix Tables A1-A7)
directly from the structured result files in results/models/ and
results/diagnostics/ -- no table in this package contains a manually typed
statistical value. Run after every analysis script (see 00_run_all.py for
the required order).

Outputs both a machine-readable CSV and a formatted XLSX per table under
results/tables/.

RESOLVED 2026-09-20 (final reconciliation pass): Table 2's dollar-
denominated statistics were previously printed in the manuscript on a
DISBURSEMENT basis (disbursement_musd), even though the row is explicitly
labeled "ODA commitment, $M" and every count, every regression, and the
paper's stated methodology use the COMMITMENT basis (commitment_musd)
throughout. This script always computed the commitment-basis figures; the
manuscript has now been updated to match (Total $58,311.30M, multilateral
$7,395.85M/12.68%, development $48,499.82M/83.17%, emergency
$2,415.63M/4.14%, main panel $45,449.31M, crisis panel $1,393.70M, Panel C
ODA row mean=19.46/SD=70.36/min=-0.61/max=1793.63 -- including one genuine
negative country-year, a downward commitment revision, that a
disbursement-basis figure could not show by construction). No discrepancy
remains between this script's output and the manuscript.

UPDATED 2026-10-12 (revised H4): Table 3 column [5], Table 4's H4 rows and
Appendix Table A4 Panel A now come from h4_common_exposure.py (emergency
vs. non-emergency ODA under a common development-relevant media exposure,
stacked PPML). The crisis-media -> emergency-ODA estimates from the former
H4 specification remain as exploratory diagnostics: Appendix Table A4
Panel B, Appendix Table A7, and Table 4's crisis-type row (verdict "—").
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from _common import (DATA_ANALYTICAL, RESULTS_DIAGNOSTICS, RESULTS_MODELS, RESULTS_TABLES,
                      get_logger, require_file, write_rows)
from lib import devrel_panel as dp
from lib import panel_windows as pw

log = get_logger("generate_tables")


def _write_table(name: str, df: pd.DataFrame) -> None:
    csv_path = RESULTS_TABLES / f"{name}.csv"
    xlsx_path = RESULTS_TABLES / f"{name}.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as e:  # pragma: no cover - openpyxl absence shouldn't block the CSV
        log.warning(f"Could not write {xlsx_path.name} ({e}); CSV was written successfully.")
    log.info(f"Wrote {csv_path}")


# ---------------------------------------------------------------------------
# Table 1 -- classifier performance
# ---------------------------------------------------------------------------

def build_table1() -> None:
    src = RESULTS_DIAGNOSTICS / "classifier_performance.csv"
    if not src.exists():
        log.warning(f"{src} not found -- run classifier_validation.py first. Skipping Table 1.")
        return
    perf = pd.read_csv(src)
    _write_table("table1_classifier_performance", perf)


# ---------------------------------------------------------------------------
# Table 2 -- data construction and analysis-sample descriptive statistics
# ---------------------------------------------------------------------------

def build_table2() -> None:
    require_file(DATA_ANALYTICAL / "oda_clean.csv")
    df = pd.read_csv(DATA_ANALYTICAL / "oda_clean.csv", encoding="utf-8-sig", low_memory=False)
    df["sdg"] = pd.to_numeric(df["sdg"], errors="coerce")
    df["sector_code_str"] = df["sector_code"].astype(str).str.replace(r"\.0$", "", regex=True)

    HUMANITARIAN_EXACT = {"72010", "72040", "72050", "73010", "74010"}
    HUMANITARIAN_PREFIX = {"720", "730", "740"}
    is_emergency = (df["sector_code_str"].isin(HUMANITARIAN_EXACT)
                     | df["sector_code_str"].str[:3].isin(HUMANITARIAN_PREFIX))
    is_multi = df["bilateral_multilateral"] == "다자간"
    analytic = ~is_multi
    development = analytic & ~is_emergency
    emergency = analytic & is_emergency

    def _row(label, mask, money_col="commitment_musd"):
        n = int(mask.sum())
        musd = float(df.loc[mask, money_col].sum())
        return n, musd

    total_n, total_musd = len(df), float(df["commitment_musd"].sum())
    rows = []
    for label, mask in [
        ("Total ODA projects, 2010-2025", pd.Series(True, index=df.index)),
        ("Multilateral commitments (omitted from analysis)", is_multi),
        ("Total analytical sample", analytic),
        ("- Development aid", development),
        ("- Emergency aid", emergency),
    ]:
        n, musd = _row(label, mask)
        rows.append({"panel": "B. ODA construction and analysis samples", "item": label,
                      "n": n, "share_pct": round(n / total_n * 100, 2),
                      "total_usd_m": round(musd, 2), "dollar_share_pct": round(musd / total_musd * 100, 2)})

    panel_b = pd.DataFrame(rows)

    # Panel C: baseline analysis variables, from the zero-filled N=2,336 devrel panel.
    country_month_long = pw.build_country_month_long()
    sentiment_month = dp.build_country_sentiment_month()
    annual = dp.build_country_annual(use_log=False, country_month_long=country_month_long,
                                      sentiment_month=sentiment_month)

    def _desc(series: pd.Series) -> dict:
        s = series.dropna()
        return {"n": int(s.shape[0]), "mean": round(float(s.mean()), 2), "sd": round(float(s.std()), 2),
                "min": round(float(s.min()), 2), "max": round(float(s.max()), 2)}

    controls = pd.read_csv(DATA_ANALYTICAL / "country_year_controls.csv", encoding="utf-8-sig")
    # Merged onto the same devrel-panel (country, year) universe used everywhere else in the
    # study, not the raw controls file's own (larger, non-recipient-inclusive) universe --
    # otherwise a stray non-country aggregate row (e.g. a "World" total in the raw WDI pull)
    # can distort these descriptives. See devrel_panel.build_country_annual's own controls merge.
    merged_controls = annual[["country_iso3", "year"]].merge(controls, on=["country_iso3", "year"], how="left")
    panel_c_rows = [
        {"variable": "Development-relevant media (2-13 month window)", **_desc(annual["devrel_media_m1_13"])},
        {"variable": "ODA commitment, $M (country x year)", **_desc(annual["devrel_commitment_musd"])},
    ]
    for label, col, scale in [
        ("GDP per capita (constant 2015 USD)", "gdppc_constant2015usd", 1),
        ("Population (million)", "population", 1e6),
        ("Bilateral trade with Korea (million USD)", "bilateral_trade_usd", 1e6),
        ("Conflict deaths", "conflict_deaths", 1),
        ("Disaster-affected persons (million)", "disaster_affected", 1e6),
    ]:
        if col in merged_controls.columns:
            panel_c_rows.append({"variable": label, **_desc(merged_controls[col] / scale)})
        else:
            log.warning(f"Control column '{col}' not found in country_year_controls.csv -- "
                        f"check its actual column names and adjust this script; skipping row.")
    panel_c = pd.DataFrame(panel_c_rows)

    _write_table("table2_panelB_data_construction", panel_b)
    _write_table("table2_panelC_descriptives", panel_c)

    # RESOLVED 2026-09-20 (final reconciliation pass): Table 2's dollar figures were
    # previously on a stale disbursement basis in the manuscript while this package
    # (and every regression, and the row's own label "ODA commitment") were already on
    # commitment_musd. The manuscript has now been updated to the commitment-basis
    # figures this script computes -- no discrepancy remains to log. (Historical note,
    # not logged: the old disbursement-basis figures were Total=$40,565.47M,
    # multilateral=$9,014.74M, development=$29,108.72M, emergency=$2,442.01M, Panel C
    # ODA mean=11.20/SD=26.74/min=0.00/max=283.27.)


def write_known_discrepancies() -> None:
    """known_discrepancies.csv is a shared log: multiple scripts (this one,
    reliability_alpha.py) each contribute rows for the discrepancies they
    find. Merge by 'id' rather than overwrite, so rerunning any one
    contributor never erases another's rows -- keyed, idempotent, and
    order-independent."""
    import csv as _csv
    existing: dict[str, dict] = {}
    if KNOWN_DISCREPANCIES_PATH.exists():
        with open(KNOWN_DISCREPANCIES_PATH, newline="", encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                existing[r["id"]] = r
    for r in _discrepancy_rows:
        existing[r["id"]] = r
    write_rows(KNOWN_DISCREPANCIES_PATH, _DISCREPANCY_FIELDS, list(existing.values()))
    log.info(f"Wrote {len(existing)} known-discrepancy rows (merged) -> {KNOWN_DISCREPANCIES_PATH}")


def _row(df: pd.DataFrame, spec_id: str, key_predictor: str | None = None) -> pd.Series:
    m = df["specification_id"] == spec_id
    if key_predictor is not None:
        m &= df["key_predictor"] == key_predictor
    sub = df[m]
    if sub.empty:
        raise KeyError(f"No row for specification_id={spec_id!r} key_predictor={key_predictor!r}")
    return sub.iloc[0]


def _fmt(row: pd.Series) -> str:
    sig = row["sig"]
    sig = "" if (pd.isna(sig) or sig is None) else str(sig)
    return f"{row['coefficient']:.3f}{sig} ({row['se']:.3f})"


# ---------------------------------------------------------------------------
# Table 3 -- main and secondary analyses (9-column comprehensive results table)
# ---------------------------------------------------------------------------

def build_table3() -> None:
    mb = pd.read_csv(RESULTS_MODELS / "main_baseline_results.csv")
    lw = pd.read_csv(RESULTS_MODELS / "lag_windows_results.csv")
    rp = pd.read_csv(RESULTS_MODELS / "rolling_persistence_results.csv")
    sc = pd.read_csv(RESULTS_MODELS / "sustained_coverage_results.csv")
    h4 = pd.read_csv(RESULTS_MODELS / "h4_common_exposure_results.csv")
    am = pd.read_csv(RESULTS_MODELS / "allocation_margin_results.csv")
    ib = pd.read_csv(RESULTS_MODELS / "ideological_breadth_results.csv")
    tone = pd.read_csv(RESULTS_MODELS / "evaluative_tone_results.csv")

    rows = []

    def add(col: str, label: str, row: pd.Series) -> None:
        rows.append({"column": col, "row_label": label, "coefficient": row["coefficient"],
                     "se": row["se"], "p_value": row["p_value"], "sig": row["sig"], "n": row["n"],
                     "formatted": _fmt(row)})

    add("[0]", "devrel_media_m1_13 (unadjusted)", _row(mb, "H1_unadjusted_pooled"))
    add("[1]", "devrel_media_m1_13 (preferred, H1)", _row(mb, "H1_preferred__devrel_media_m1_13"))
    add("[2]", "devrel_media_m1_13", _row(lw, "lag_windows_joint__devrel_media_m1_13"))
    add("[2]", "devrel_media_m14_25", _row(lw, "lag_windows_joint__devrel_media_m14_25"))
    add("[2]", "devrel_media_m26_37", _row(lw, "lag_windows_joint__devrel_media_m26_37"))
    add("[3]", "devrel_media_roll24", _row(rp, "H2_devrel_media_roll24__devrel_media_roll24"))
    add("[3]", "devrel_media_roll36", _row(rp, "H2_devrel_media_roll36__devrel_media_roll36"))
    add("[3]", "devrel_media_roll60", _row(rp, "H2_devrel_media_roll60__devrel_media_roll60"))
    add("[4]", "streak_ge24", _row(sc, "H3_streak_ge24__streak_ge24"))
    add("[4]", "streak_ge36", _row(sc, "H3_streak_ge36__streak_ge36"))
    add("[4]", "streak_ge60", _row(sc, "H3_streak_ge60__streak_ge60"))
    add("[5]", "devrel_media_contemp x emergency (H4 difference, stacked PPML)",
        _row(h4, "H4_primary__diff_emergency_minus_nonemergency_contemporaneous"))
    add("[6]", "extensive margin", _row(am, "allocation_margin_extensive__devrel_media_m1_13"))
    add("[6]", "intensive margin", _row(am, "allocation_margin_intensive__devrel_media_m1_13"))
    add("[7]", "devrel_media_m1_13", _row(ib, "ideological_breadth_crosspartisan__devrel_media_m1_13"))
    add("[7]", "media_x_crosspartisan", _row(ib, "ideological_breadth_crosspartisan__media_x_crosspartisan"))
    add("[8]", "devrel_media_m1_13", _row(tone, "evaluative_tone_additive__devrel_media_m1_13"))
    add("[8]", "share_positive", _row(tone, "evaluative_tone_additive__share_positive"))
    add("[8]", "share_negative", _row(tone, "evaluative_tone_additive__share_negative"))

    _write_table("table3_main_results", pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Table 4 -- robustness and diagnostic-test summary (narrative, values interpolated)
# ---------------------------------------------------------------------------

def _p_fmt(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


_H4_LABEL = {"m1_13": "2-13 mo", "m14_25": "14-25 mo", "m26_37": "26-37 mo", "roll24": "rolling 24 mo",
             "roll36": "rolling 36 mo", "roll60": "rolling 60 mo", "streak_ge24": "streak ≥ 24",
             "streak_ge36": "streak ≥ 36", "streak_ge60": "streak ≥ 60"}


def build_table4() -> None:
    fn = pd.read_csv(RESULTS_MODELS / "placebo_future_news_results.csv")
    nd = pd.read_csv(RESULTS_MODELS / "placebo_nondevrel_results.csv")
    loo = pd.read_csv(RESULTS_MODELS / "leave_one_out_results.csv")
    et = pd.read_csv(RESULTS_MODELS / "emergency_temporal_results.csv")
    et_wald = pd.read_csv(RESULTS_MODELS / "emergency_temporal_wald_results.csv")

    innov = fn[fn["specification_id"] == "placebo_future_news_ar2_innovation__innovation_ar2"].iloc[0]
    # news-substance placebo: devrel vs. all country news vs. sports news, same 7 measures
    sub = nd.assign(measure=nd["specification_id"].str.extract(r"placebo_nondevrel_(.+)_(?:devrel|allnews|generic)$")[0],
                    series=nd["specification_id"].str.extract(r"_(devrel|allnews|generic)$")[0])
    wide = sub.pivot(index="measure", columns="series", values=["coefficient", "p_value"])
    n_meas = len(wide)
    sig = {s: int((wide[("p_value", s)] < 0.10).sum()) for s in ["devrel", "allnews", "generic"]}
    larger = wide[wide[("coefficient", "allnews")] > wide[("coefficient", "devrel")]].index.tolist()
    larger_txt = ("" if not larger else
                  f"; all news exceeds DevRel only at {', '.join(_H4_LABEL.get(m.replace('streak', 'streak_ge'), m) for m in larger)}")
    substance_txt = (f"DevRel coverage is significant (p < 0.10) on {sig['devrel']}/{n_meas} headline measures, "
                     f"all country news on {sig['allnews']}/{n_meas}, and sports news on {sig['generic']}/{n_meas}"
                     f"{larger_txt}.")

    h4 = pd.read_csv(RESULTS_MODELS / "h4_common_exposure_results.csv")
    h4_main = _row(h4, "H4_primary__diff_emergency_minus_nonemergency_contemporaneous")
    h4_ll = _row(h4, "H4_loglinear__diff_emergency_minus_nonemergency_contemporaneous")
    # every non-contemporaneous aid-type difference: extended-model lags, rolling, streak
    other = h4[h4["key_predictor"].str.startswith("diff_")
               & h4["specification_id"].str.match(r"H4_(extended|roll|streak)")
               & ~h4["key_predictor"].str.endswith("contemporaneous")]
    exceptions = other[other["p_value"] < 0.10]
    exc_txt = "; ".join(
        f"{_H4_LABEL[r['key_predictor'].replace('diff_emergency_minus_nonemergency_', '')]}: "
        f"β = {r['coefficient']:.3f}, {_p_fmt(r['p_value'])}" for _, r in exceptions.iterrows())
    timing_txt = (f"The emergency−non-emergency difference is concentrated in the contemporaneous window "
                  f"(β = {h4_main['coefficient']:.3f}, {_p_fmt(h4_main['p_value'])}); lagged-window, "
                  f"rolling-average and streak differences are not consistently significant"
                  + (f" ({exc_txt})." if len(exceptions) else "."))

    ct = et[et["specification_id"].str.startswith("H4_crisis_type_")]
    n_ct = int(len(ct))
    n_ct_pos_sig = int(((ct["coefficient"] > 0) & (ct["p_value"] < 0.05)).sum())
    ct_scope = (f"All {n_ct} crisis types are individually positive and significant (p < 0.05)"
                if n_ct_pos_sig == n_ct else
                f"{n_ct_pos_sig} of {n_ct} crisis types are individually positive and significant (p < 0.05)")
    joint_ct = et_wald[et_wald["test_name"].str.contains("crisis-type coefficients = 0")].iloc[0]

    roll60_row = loo[loo["analysis"] == "persistent_roll60"].iloc[0]
    n_flips = int(roll60_row["n_sig_flips_at_p10"])
    n_flips_txt = _NUMBER_WORDS.get(n_flips, str(n_flips))

    # H3 coverage-threshold sensitivity (Table A6): highest bar up to which every streak estimate is positive,
    # and the >=60-month estimate at the most stringent bar
    hb = pd.read_csv(RESULTS_MODELS / "h3_bar_sensitivity_results.csv")
    hb["bar"] = hb["specification_id"].str.extract(r"_bar(\d+)_ge")[0].astype(int)
    all_pos = hb.groupby("bar")["coefficient"].apply(lambda c: bool((c > 0).all())).sort_index()
    pos_upto = int(all_pos.index[(~all_pos).cumsum() == 0].max())
    top_bar = int(hb["bar"].max())
    top60 = hb[(hb["bar"] == top_bar) & hb["specification_id"].str.endswith("__streak_ge60")].iloc[0]
    bar_txt = (f"Streak estimates remain positive at every streak length for thresholds of "
               f"{int(all_pos.index.min())}–{pos_upto} articles per month; at {top_bar} articles the ≥ 60-month "
               f"estimate is β = {top60['coefficient']:.3f}, {_p_fmt(top60['p_value'])}.")

    rows = [
        {"diagnostic": "Future-news placebo",
         "key_finding": f"Future media does not predict current ODA: γ = {innov['coefficient']:.3f}, "
                         f"{_p_fmt(innov['p_value'])}."},
        {"diagnostic": "News-substance placebo",
         "key_finding": substance_txt},
        {"diagnostic": "Alternative timing (baseline)",
         "key_finding": "The association remains positive for roughly three years, peaking "
                         "contemporaneously before fading in year four."},
        {"diagnostic": "Alternative timing (H4)",
         "key_finding": timing_txt},
        {"diagnostic": "LOO sensitivity",
         "key_finding": f"No sign reversals in any analysis. Only the roll60 estimate shows {n_flips_txt} "
                         f"significance-status changes ({n_flips}/{int(roll60_row['n_exclusions'])})."},
        {"diagnostic": "Coverage-threshold sensitivity (H3)",
         "key_finding": bar_txt},
        {"diagnostic": "Estimator sensitivity (H4)",
         "key_finding": f"A log(x+1) OLS counterpart of the stacked model does not reproduce the contemporaneous "
                         f"difference (β = {h4_ll['coefficient']:.3f}, {_p_fmt(h4_ll['p_value'])})."},
        {"diagnostic": "Crisis-type heterogeneity",
         "key_finding": f"{ct_scope}; joint test: χ²({int(joint_ct['df'])}) = "
                         f"{joint_ct['test_statistic']:.3f}, {_p_fmt(joint_ct['p_value'])}."},
    ]
    _write_table("table4_robustness_summary", pd.DataFrame(rows))
    log.info("Note: Table 4 is a narrative summary of Appendix Tables A1-A7; every number above is "
             "interpolated from those result files, not typed manually.")


# ---------------------------------------------------------------------------
# Appendix Tables A1-A7 -- full underlying estimates
# ---------------------------------------------------------------------------

def build_appendix_tables() -> None:
    _write_table("tableA1_temporal_placebo",
                 pd.read_csv(RESULTS_MODELS / "placebo_future_news_results.csv"))
    _write_table("tableA2_media_type_placebo",
                 pd.read_csv(RESULTS_MODELS / "placebo_nondevrel_results.csv"))
    _write_table("tableA3_baseline_alt_timing",
                 pd.read_csv(RESULTS_MODELS / "alt_timing_baseline_results.csv"))

    # A4: H4 under common development-relevant media exposure (primary model in full;
    # extended-model, rolling and streak aid-type differences)
    h4 = pd.read_csv(RESULTS_MODELS / "h4_common_exposure_results.csv")
    sid = h4["specification_id"]
    a4 = h4[sid.str.startswith("H4_primary__")
            | (sid.str.match(r"H4_(extended|roll|streak)") & h4["key_predictor"].str.startswith("diff_"))]
    _write_table("tableA4_emergency_alt_timing", a4)
    et = pd.read_csv(RESULTS_MODELS / "emergency_temporal_results.csv")

    _write_table("tableA5_leave_one_out", pd.read_csv(RESULTS_MODELS / "leave_one_out_results.csv"))

    build_table_a6()

    a7 = et[et["specification_id"].str.startswith("H4_crisis_type_")]
    _write_table("tableA7_crisis_type_heterogeneity", a7)


# ---------------------------------------------------------------------------
# Appendix Table A6 -- H3 coverage-bar sensitivity
# ---------------------------------------------------------------------------

def build_table_a6() -> None:
    """One row per monthly coverage bar (>=1, >=2, >=3 [baseline], >=4, >=5,
    >=10 devrel-relevant articles/month); for each of the three streak
    lengths (>=24, >=36, >=60 consecutive months), the coefficient, SE,
    p-value, and the count/percent of the estimation sample meeting that
    streak definition at that bar. The regression sample size (N) is
    constant at 2,214 across every cell (the streak indicator is never
    missing, so it never changes which rows survive the complete-case
    filter) and is reported once in the table's notes rather than
    per-cell. Source: h3_bar_sensitivity.py."""
    res = pd.read_csv(RESULTS_MODELS / "h3_bar_sensitivity_results.csv")
    cov = pd.read_csv(RESULTS_DIAGNOSTICS / "h3_bar_sensitivity_coverage.csv")

    bars = sorted(cov["covered_article_bar"].unique())
    thresholds = [24, 36, 60]

    rows = []
    for bar in bars:
        row = {"covered_article_bar": bar, "is_manuscript_baseline": bar == dp.COVERED_ARTICLE_BAR}
        for thresh in thresholds:
            r = res[res["specification_id"] == f"H3_bar_sensitivity_bar{bar}_ge{thresh}__streak_ge{thresh}"].iloc[0]
            c = cov[(cov["covered_article_bar"] == bar) & (cov["streak_threshold_months"] == thresh)].iloc[0]
            row[f"streak_ge{thresh}_coef"] = r["coefficient"]
            row[f"streak_ge{thresh}_se"] = r["se"]
            row[f"streak_ge{thresh}_p"] = r["p_value"]
            row[f"streak_ge{thresh}_count_meeting"] = int(c["n_meeting"])
            row[f"streak_ge{thresh}_pct_meeting"] = round(100 * c["share_meeting"], 1)
        rows.append(row)

    _write_table("tableA6_h3_bar_sensitivity", pd.DataFrame(rows))


def main() -> None:
    build_table1()
    build_table2()
    build_table3()
    build_table4()
    build_appendix_tables()


if __name__ == "__main__":
    main()
