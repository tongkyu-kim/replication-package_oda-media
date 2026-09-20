"""
Consolidates every manuscript number that appears only in prose (not
already visible in Table 1-4 or Appendix Tables A1-A7) into one
machine-readable file, each row traced to its source result file. Run
after every analysis script and generate_tables.py.

Writes: results/models/reported_text_results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from _common import (DATA_ANALYTICAL, RESULTS_DIAGNOSTICS, RESULTS_MODELS, get_logger, write_rows,
                      ensure_known_discrepancies_file)

log = get_logger("build_reported_text_results")

OUT_PATH = RESULTS_MODELS / "reported_text_results.csv"
FIELDS = ["item", "manuscript_section", "value", "source_file", "source_row_or_field"]

rows: list[dict] = []


def add(item: str, section: str, value, source_file: str, source_ref: str) -> None:
    rows.append({"item": item, "manuscript_section": section, "value": value,
                 "source_file": source_file, "source_row_or_field": source_ref})


def main() -> None:
    # --- Sample construction counts (Section 3.1-3.2) ---
    oda = pd.read_csv(DATA_ANALYTICAL / "oda_clean.csv", encoding="utf-8-sig", low_memory=False, usecols=["project_id"])
    add("Total ODA projects, 2010-2025", "3.1", len(oda), "data/analytical/oda_clean.csv", "row count")

    devrel = pd.read_csv(DATA_ANALYTICAL / "devrel_country_year.csv")
    add("Main development-ODA panel N", "3.2", len(devrel), "data/analytical/devrel_country_year.csv", "row count")
    add("Main development-ODA panel countries", "3.2", devrel["country_iso3"].nunique(),
        "data/analytical/devrel_country_year.csv", "nunique(country_iso3)")

    emergency = pd.read_csv(DATA_ANALYTICAL / "emergency_country_year.csv")
    add("Full emergency-ODA panel N (pre crisis-media merge)", "3.2", len(emergency),
        "data/analytical/emergency_country_year.csv", "row count")

    # --- Classifier performance (Section 3.1, Table 1) ---
    if (RESULTS_DIAGNOSTICS / "classifier_performance.csv").exists():
        perf = pd.read_csv(RESULTS_DIAGNOSTICS / "classifier_performance.csv").set_index("metric")["value"]
        for metric, section_label in [("f1", "F1 score"), ("precision", "Precision"), ("recall", "Recall"),
                                       ("auc_roc", "AUC-ROC"), ("test_n", "Held-out test N")]:
            add(f"Classifier {section_label}", "3.1", perf[metric], "results/diagnostics/classifier_performance.csv",
                f"metric=={metric}")

    # --- Reliability alphas (Section 3.1) ---
    if (RESULTS_DIAGNOSTICS / "reliability_alpha.csv").exists():
        alpha = pd.read_csv(RESULTS_DIAGNOSTICS / "reliability_alpha.csv")
        for _, r in alpha.iterrows():
            add(f"Krippendorff's alpha, {r['variable']} ({r['level_of_measurement']})", "3.1", r["alpha"],
                "results/diagnostics/reliability_alpha.csv", f"variable=={r['variable']}")

    # --- H1 baseline (Section 4.1) ---
    mb = pd.read_csv(RESULTS_MODELS / "main_baseline_results.csv")
    pref = mb[mb["specification_id"] == "H1_preferred__devrel_media_m1_13"].iloc[0]
    add("H1 preferred coefficient", "4.1", pref["coefficient"], "results/models/main_baseline_results.csv",
        "specification_id==H1_preferred__devrel_media_m1_13")
    add("H1 preferred p-value", "4.1", pref["p_value"], "results/models/main_baseline_results.csv",
        "specification_id==H1_preferred__devrel_media_m1_13")

    # --- Temporal decomposition (Section 4.2) ---
    lw = pd.read_csv(RESULTS_MODELS / "lag_windows_results.csv")
    for pred, label in [("devrel_media_m1_13", "2-13mo"), ("devrel_media_m14_25", "14-25mo"),
                         ("devrel_media_m26_37", "26-37mo")]:
        r = lw[lw["key_predictor"] == pred].iloc[0]
        add(f"Temporal decomposition, {label} coefficient", "4.2", r["coefficient"],
            "results/models/lag_windows_results.csv", f"key_predictor=={pred}")

    # --- H2 persistent attention (Section 4.3) ---
    rp = pd.read_csv(RESULTS_MODELS / "rolling_persistence_results.csv")
    for pred in ["devrel_media_roll24", "devrel_media_roll36", "devrel_media_roll60"]:
        r = rp[rp["key_predictor"] == pred].iloc[0]
        add(f"H2 {pred} coefficient", "4.3", r["coefficient"], "results/models/rolling_persistence_results.csv",
            f"key_predictor=={pred}")

    # --- H3 sustained attention (Section 4.4) ---
    sc = pd.read_csv(RESULTS_MODELS / "sustained_coverage_results.csv")
    for pred in ["streak_ge24", "streak_ge36", "streak_ge60"]:
        r = sc[sc["key_predictor"] == pred].iloc[0]
        add(f"H3 {pred} coefficient", "4.4", r["coefficient"], "results/models/sustained_coverage_results.csv",
            f"key_predictor=={pred}")
        add(f"H3 {pred} p-value", "4.4", r["p_value"], "results/models/sustained_coverage_results.csv",
            f"key_predictor=={pred}")

    # --- H4 development vs. emergency (Section 4.5) ---
    et = pd.read_csv(RESULTS_MODELS / "emergency_temporal_results.csv")
    pooled = et[et["specification_id"] == "H4_within_emergency_contemporaneous__crisis_media_lag0"].iloc[0]
    add("H4 within-emergency pooled contemporaneous coefficient", "4.5", pooled["coefficient"],
        "results/models/emergency_temporal_results.csv",
        "specification_id==H4_within_emergency_contemporaneous__crisis_media_lag0")

    ppml = pd.read_csv(RESULTS_MODELS / "emergency_ppml_contrast_results.csv")
    for pred, label in [("development_contemporaneous", "development"), ("emergency_contemporaneous", "emergency"),
                         ("diff_emergency_minus_development_contemporaneous", "emergency-minus-development difference")]:
        r = ppml[ppml["key_predictor"] == pred].iloc[0]
        add(f"H4 stacked PPML, contemporaneous {label} coefficient", "4.5", r["coefficient"],
            "results/models/emergency_ppml_contrast_results.csv", f"key_predictor=={pred}")
        add(f"H4 stacked PPML, contemporaneous {label} p-value", "4.5", r["p_value"],
            "results/models/emergency_ppml_contrast_results.csv", f"key_predictor=={pred}")

    # --- Secondary analyses (Section 4.6) ---
    am = pd.read_csv(RESULTS_MODELS / "allocation_margin_results.csv")
    for spec, label in [("allocation_margin_extensive__devrel_media_m1_13", "extensive margin"),
                         ("allocation_margin_intensive__devrel_media_m1_13", "intensive margin")]:
        r = am[am["specification_id"] == spec].iloc[0]
        add(f"Allocation-margin {label} coefficient", "4.6", r["coefficient"],
            "results/models/allocation_margin_results.csv", f"specification_id=={spec}")

    ib = pd.read_csv(RESULTS_MODELS / "ideological_breadth_results.csv")
    r = ib[ib["specification_id"] == "ideological_breadth_crosspartisan__media_x_crosspartisan"].iloc[0]
    add("Ideological-breadth interaction coefficient", "4.6", r["coefficient"],
        "results/models/ideological_breadth_results.csv",
        "specification_id==ideological_breadth_crosspartisan__media_x_crosspartisan")

    tone = pd.read_csv(RESULTS_MODELS / "evaluative_tone_results.csv")
    r = tone[tone["specification_id"] == "evaluative_tone_additive__devrel_media_m1_13"].iloc[0]
    add("Evaluative-tone spec, media coefficient", "4.6", r["coefficient"],
        "results/models/evaluative_tone_results.csv",
        "specification_id==evaluative_tone_additive__devrel_media_m1_13")

    # --- Robustness (Section 4.7) ---
    fn = pd.read_csv(RESULTS_MODELS / "placebo_future_news_results.csv")
    innov = fn[fn["specification_id"] == "placebo_future_news_ar2_innovation__innovation_ar2"].iloc[0]
    add("Future-news innovation placebo coefficient", "4.7", innov["coefficient"],
        "results/models/placebo_future_news_results.csv",
        "specification_id==placebo_future_news_ar2_innovation__innovation_ar2")
    add("Future-news innovation placebo p-value", "4.7", innov["p_value"],
        "results/models/placebo_future_news_results.csv",
        "specification_id==placebo_future_news_ar2_innovation__innovation_ar2")

    et_wald = pd.read_csv(RESULTS_MODELS / "emergency_temporal_wald_results.csv")
    joint = et_wald[et_wald["test_name"].str.contains("crisis-type coefficients = 0")].iloc[0]
    add("Crisis-type joint Wald chi-square", "4.7", joint["test_statistic"],
        "results/models/emergency_temporal_wald_results.csv", "test_name contains 'crisis-type coefficients = 0'")
    add("Crisis-type joint Wald p-value", "4.7", joint["p_value"],
        "results/models/emergency_temporal_wald_results.csv", "test_name contains 'crisis-type coefficients = 0'")

    loo = pd.read_csv(RESULTS_MODELS / "leave_one_out_results.csv")
    for _, r in loo.iterrows():
        add(f"LOO {r['analysis']} sig-flip count", "4.7", r["n_sig_flips_at_p10"],
            "results/models/leave_one_out_results.csv", f"analysis=={r['analysis']}")

    # --- Supplementary IV diagnostics (Section 4.8) ---
    # RESOLVED 2026-09-20 (final reconciliation pass): a second exhaustive search
    # (numeric scan across pipeline/crowdout_iv/'s 8 active scripts and its archived
    # counterparts, tolerance +/-0.015) still found no script producing the previously
    # -reported F=5.86 (sports) / F=1.50 (entertainment) pair. Per instruction, these
    # two specific numbers were REMOVED from the manuscript; Section 4.8 now retains
    # only the qualitative statement that both instruments had partial first-stage
    # F-statistics "well below the conventional relevance threshold," which every
    # located IV variant in pipeline/crowdout_iv/ does in fact support (the many
    # sports/entertainment first-stage F's actually present there range from ~0.03 to
    # ~22, but the sports/entertainment PAIR specifically named in the manuscript could
    # not be isolated). No further numeric claim is made in this package for this item.
    add("Sports-based IV partial first-stage F", "4.8",
        "Not reported (manuscript revised 2026-09-20 to a qualitative-only statement; "
        "specific F-statistic could not be traced to any script in the source repository)",
        "N/A", "N/A")
    add("Entertainment-based IV partial first-stage F", "4.8",
        "Not reported (manuscript revised 2026-09-20 to a qualitative-only statement; "
        "specific F-statistic could not be traced to any script in the source repository)",
        "N/A", "N/A")

    write_rows(OUT_PATH, FIELDS, rows)
    log.info(f"Wrote {len(rows)} rows -> {OUT_PATH}")

    ensure_known_discrepancies_file()


if __name__ == "__main__":
    main()
