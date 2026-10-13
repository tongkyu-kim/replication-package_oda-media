"""
Builds Tables 1-4 and Appendix Tables A1-A7 from the result files in
results/models/ and results/diagnostics/ (CSV and XLSX in results/tables/).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from _common import DATA_ANALYTICAL, RESULTS_DIAGNOSTICS, RESULTS_MODELS, RESULTS_TABLES, get_logger, require_file
from lib import devrel_panel as dp
from lib import panel_windows as pw

log = get_logger("generate_tables")


def _write_table(name: str, df: pd.DataFrame) -> None:
    df.to_csv(RESULTS_TABLES / f"{name}.csv", index=False, encoding="utf-8-sig")
    df.to_excel(RESULTS_TABLES / f"{name}.xlsx", index=False)
    log.info(f"Wrote {name}")


def _row(df: pd.DataFrame, spec_id: str) -> pd.Series:
    sub = df[df["specification_id"] == spec_id]
    if sub.empty:
        raise KeyError(f"No row for specification_id={spec_id!r}")
    return sub.iloc[0]


def _fmt(row: pd.Series) -> str:
    sig = "" if pd.isna(row["sig"]) else str(row["sig"])
    return f"{row['coefficient']:.3f}{sig} ({row['se']:.3f})"


def _p_fmt(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ---------------------------------------------------------------------------
# Table 1: classifier performance
# ---------------------------------------------------------------------------

def build_table1() -> None:
    _write_table("table1_classifier_performance", pd.read_csv(RESULTS_DIAGNOSTICS / "classifier_performance.csv"))


# ---------------------------------------------------------------------------
# Table 2: data construction (Panel B) and descriptive statistics (Panel C)
# ---------------------------------------------------------------------------

def build_table2() -> None:
    require_file(DATA_ANALYTICAL / "oda_clean.csv")
    df = pd.read_csv(DATA_ANALYTICAL / "oda_clean.csv", encoding="utf-8-sig", low_memory=False)
    df["sector_code_str"] = df["sector_code"].astype(str).str.replace(r"\.0$", "", regex=True)

    humanitarian_exact = {"72010", "72040", "72050", "73010", "74010"}
    humanitarian_prefix = {"720", "730", "740"}
    is_emergency = (df["sector_code_str"].isin(humanitarian_exact)
                    | df["sector_code_str"].str[:3].isin(humanitarian_prefix))
    is_multi = df["bilateral_multilateral"] == "다자간"
    analytic = ~is_multi

    total_n, total_musd = len(df), float(df["commitment_musd"].sum())
    rows = []
    for label, mask in [
        ("Total ODA projects, 2010-2025", pd.Series(True, index=df.index)),
        ("Multilateral commitments (omitted from analysis)", is_multi),
        ("Total analytical sample", analytic),
        ("- Development aid", analytic & ~is_emergency),
        ("- Emergency aid", analytic & is_emergency),
    ]:
        n, musd = int(mask.sum()), float(df.loc[mask, "commitment_musd"].sum())
        rows.append({"panel": "B. ODA construction and analysis samples", "item": label,
                     "n": n, "share_pct": round(n / total_n * 100, 2),
                     "total_usd_m": round(musd, 2), "dollar_share_pct": round(musd / total_musd * 100, 2)})
    _write_table("table2_panelB_data_construction", pd.DataFrame(rows))

    annual = dp.build_country_annual(use_log=False, country_month_long=pw.build_country_month_long(),
                                     sentiment_month=dp.build_country_sentiment_month())

    def _desc(series: pd.Series) -> dict:
        s = series.dropna()
        return {"n": int(s.shape[0]), "mean": round(float(s.mean()), 2), "sd": round(float(s.std()), 2),
                "min": round(float(s.min()), 2), "max": round(float(s.max()), 2)}

    controls = pd.read_csv(DATA_ANALYTICAL / "country_year_controls.csv", encoding="utf-8-sig")
    merged = annual[["country_iso3", "year"]].merge(controls, on=["country_iso3", "year"], how="left")
    panel_c = [
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
        panel_c.append({"variable": label, **_desc(merged[col] / scale)})
    _write_table("table2_panelC_descriptives", pd.DataFrame(panel_c))


# ---------------------------------------------------------------------------
# Table 3: main and secondary analyses
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
# Table 4: summary of robustness and diagnostic tests
# ---------------------------------------------------------------------------

_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
_LABEL = {"m1_13": "2-13 mo", "m14_25": "14-25 mo", "m26_37": "26-37 mo", "roll24": "rolling 24 mo",
          "roll36": "rolling 36 mo", "roll60": "rolling 60 mo", "streak_ge24": "streak ≥ 24",
          "streak_ge36": "streak ≥ 36", "streak_ge60": "streak ≥ 60"}


def build_table4() -> None:
    fn = pd.read_csv(RESULTS_MODELS / "placebo_future_news_results.csv")
    ns = pd.read_csv(RESULTS_MODELS / "placebo_news_substance_results.csv")
    loo = pd.read_csv(RESULTS_MODELS / "leave_one_out_results.csv")
    et = pd.read_csv(RESULTS_MODELS / "emergency_temporal_results.csv")
    et_wald = pd.read_csv(RESULTS_MODELS / "emergency_temporal_wald_results.csv")
    h4 = pd.read_csv(RESULTS_MODELS / "h4_common_exposure_results.csv")
    hb = pd.read_csv(RESULTS_MODELS / "h3_bar_sensitivity_results.csv")

    innov = _row(fn, "placebo_future_news_ar2_innovation__innovation_ar2")

    parts = ns["specification_id"].str.extract(r"placebo_substance_(.+)_(devrel|allnews|sports)$")
    wide = ns.assign(measure=parts[0], series=parts[1]).pivot(index="measure", columns="series",
                                                              values=["coefficient", "p_value"])
    sig = {s: int((wide[("p_value", s)] < 0.10).sum()) for s in ["devrel", "allnews", "sports"]}
    larger = wide[wide[("coefficient", "allnews")] > wide[("coefficient", "devrel")]].index.tolist()
    larger_txt = "" if not larger else (
        "; all news exceeds DevRel only at "
        + ", ".join(_LABEL.get(m.replace("streak", "streak_ge"), m) for m in larger))
    substance_txt = (f"DevRel coverage is significant (p < 0.10) on {sig['devrel']}/{len(wide)} headline measures, "
                     f"all country news on {sig['allnews']}/{len(wide)}, and sports news on "
                     f"{sig['sports']}/{len(wide)}{larger_txt}.")

    h4_main = _row(h4, "H4_primary__diff_emergency_minus_nonemergency_contemporaneous")
    h4_ll = _row(h4, "H4_loglinear__diff_emergency_minus_nonemergency_contemporaneous")
    other = h4[h4["key_predictor"].str.startswith("diff_")
               & h4["specification_id"].str.match(r"H4_(extended|roll|streak)")
               & ~h4["key_predictor"].str.endswith("contemporaneous")]
    exceptions = other[other["p_value"] < 0.10]
    exc_txt = "; ".join(
        f"{_LABEL[r['key_predictor'].replace('diff_emergency_minus_nonemergency_', '')]}: "
        f"β = {r['coefficient']:.3f}, {_p_fmt(r['p_value'])}" for _, r in exceptions.iterrows())
    timing_txt = (f"The emergency−non-emergency difference is concentrated in the contemporaneous window "
                  f"(β = {h4_main['coefficient']:.3f}, {_p_fmt(h4_main['p_value'])}); lagged-window, "
                  f"rolling-average and streak differences are not consistently significant"
                  + (f" ({exc_txt})." if len(exceptions) else "."))

    ct = et[et["specification_id"].str.startswith("crisis_type_")]
    n_ct_pos_sig = int(((ct["coefficient"] > 0) & (ct["p_value"] < 0.05)).sum())
    ct_scope = (f"All {len(ct)} crisis types are individually positive and significant (p < 0.05)"
                if n_ct_pos_sig == len(ct) else
                f"{n_ct_pos_sig} of {len(ct)} crisis types are individually positive and significant (p < 0.05)")
    joint_ct = _row(et_wald, "crisis_type_joint")

    roll60 = loo[loo["analysis"] == "persistent_roll60"].iloc[0]
    n_flips = int(roll60["n_sig_flips_at_p10"])

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
        {"diagnostic": "News-substance placebo", "key_finding": substance_txt},
        {"diagnostic": "Alternative timing (baseline)",
         "key_finding": "The association remains positive for roughly three years, peaking "
                        "contemporaneously before fading in year four."},
        {"diagnostic": "Alternative timing (emergency vs. non-emergency)", "key_finding": timing_txt},
        {"diagnostic": "LOO sensitivity",
         "key_finding": f"No sign reversals in any analysis. Only the roll60 estimate shows "
                        f"{_NUMBER_WORDS.get(n_flips, str(n_flips))} significance-status changes "
                        f"({n_flips}/{int(roll60['n_exclusions'])})."},
        {"diagnostic": "Coverage-threshold sensitivity (sustained attention)", "key_finding": bar_txt},
        {"diagnostic": "Estimator sensitivity (emergency vs. non-emergency)",
         "key_finding": f"A log(x+1) OLS counterpart of the stacked model does not reproduce the contemporaneous "
                        f"difference (β = {h4_ll['coefficient']:.3f}, {_p_fmt(h4_ll['p_value'])})."},
        {"diagnostic": "Crisis-type heterogeneity",
         "key_finding": f"{ct_scope}; joint test: χ²({int(joint_ct['df'])}) = "
                        f"{joint_ct['test_statistic']:.3f}, {_p_fmt(joint_ct['p_value'])}."},
    ]
    _write_table("table4_robustness_summary", pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Appendix Tables A1-A7
# ---------------------------------------------------------------------------

def build_table_a1() -> None:
    """Reference 2-13-month estimate, future-news placebos, and the 14-25/26-37-month
    windows with their mirror-image future windows (manuscript row order)."""
    fn = pd.read_csv(RESULTS_MODELS / "placebo_future_news_results.csv")
    mb = pd.read_csv(RESULTS_MODELS / "main_baseline_results.csv")
    order = [("2-13 mo pre-allocation (reference)", mb, "H1_preferred__devrel_media_m1_13"),
             ("Future-news raw placebo (t+1)", fn, "placebo_future_news_raw__future_news_t1"),
             ("Future-news innovation placebo (AR(2) residual of t+1)", fn,
              "placebo_future_news_ar2_innovation__innovation_ar2"),
             ("14-25 mo pre-allocation window", fn, "placebo_future_news_m14_25_pre__devrel_media_m14_25"),
             ("14-25 mo future-media placebo (t+13..t+24)", fn, "placebo_future_news_m14_25_future__future_m14_25"),
             ("26-37 mo pre-allocation window", fn, "placebo_future_news_m26_37_pre__devrel_media_m26_37"),
             ("26-37 mo future-media placebo (t+25..t+36)", fn, "placebo_future_news_m26_37_future__future_m26_37")]
    rows = []
    for label, src, sid in order:
        r = _row(src, sid)
        rows.append({"specification": label, "specification_id": sid, "coefficient": r["coefficient"],
                     "se": r["se"], "p_value": r["p_value"], "n": r["n"], "n_entities": r["n_entities"]})
    _write_table("tableA1_temporal_placebo", pd.DataFrame(rows))


def build_table_a6() -> None:
    """One row per monthly coverage threshold; coefficient, SE, p and the number (share) of
    country-years meeting each streak length."""
    res = pd.read_csv(RESULTS_MODELS / "h3_bar_sensitivity_results.csv")
    cov = pd.read_csv(RESULTS_DIAGNOSTICS / "h3_bar_sensitivity_coverage.csv")
    rows = []
    for bar in sorted(cov["covered_article_bar"].unique()):
        row = {"covered_article_bar": bar, "is_manuscript_baseline": bar == dp.COVERED_ARTICLE_BAR}
        for thresh in [24, 36, 60]:
            r = _row(res, f"H3_bar_sensitivity_bar{bar}_ge{thresh}__streak_ge{thresh}")
            c = cov[(cov["covered_article_bar"] == bar) & (cov["streak_threshold_months"] == thresh)].iloc[0]
            row[f"streak_ge{thresh}_coef"] = r["coefficient"]
            row[f"streak_ge{thresh}_se"] = r["se"]
            row[f"streak_ge{thresh}_p"] = r["p_value"]
            row[f"streak_ge{thresh}_count_meeting"] = int(c["n_meeting"])
            row[f"streak_ge{thresh}_pct_meeting"] = round(100 * c["share_meeting"], 1)
        rows.append(row)
    _write_table("tableA6_h3_bar_sensitivity", pd.DataFrame(rows))


def build_appendix_tables() -> None:
    build_table_a1()
    _write_table("tableA2_news_substance_placebo", pd.read_csv(RESULTS_MODELS / "placebo_news_substance_results.csv"))
    _write_table("tableA3_baseline_alt_timing", pd.read_csv(RESULTS_MODELS / "alt_timing_baseline_results.csv"))
    h4 = pd.read_csv(RESULTS_MODELS / "h4_common_exposure_results.csv")
    sid = h4["specification_id"]
    a4 = h4[sid.str.startswith("H4_primary__")
            | (sid.str.match(r"H4_(extended|roll|streak)") & h4["key_predictor"].str.startswith("diff_"))]
    _write_table("tableA4_emergency_alt_timing", a4)
    _write_table("tableA5_leave_one_out", pd.read_csv(RESULTS_MODELS / "leave_one_out_results.csv"))
    build_table_a6()
    et = pd.read_csv(RESULTS_MODELS / "emergency_temporal_results.csv")
    _write_table("tableA7_crisis_type_heterogeneity", et[et["specification_id"].str.startswith("crisis_type_")])


def main() -> None:
    build_table1()
    build_table2()
    build_table3()
    build_table4()
    build_appendix_tables()


if __name__ == "__main__":
    main()
