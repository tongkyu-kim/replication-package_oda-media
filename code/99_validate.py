"""
Validates the regenerated release outputs. Two kinds of checks:

1. INTERNAL CONSISTENCY (the primary purpose): required inputs exist,
   panel dimensions and key counts match what the pipeline itself should
   produce, no duplicate keys, no missing required output file, figure
   data matches the model-result file it was built from. These checks do
   NOT depend on manuscript values -- they would catch a broken pipeline
   even if the manuscript were never written.

2. VERIFICATION SNAPSHOT comparison against expected_final_results.csv --
   a small, hand-curated set of values confirmed during this package's
   construction to match the manuscript (with small tolerances for
   legitimate floating-point/rerun drift). This is a SANITY CHECK, not a
   source of truth: expected_final_results.csv was generated ONCE from a
   verified reconciliation pass and is never used to compute or adjust any
   analytical result -- see README.md's "Known discrepancies" section for
   the (small number of) values that are EXPECTED to differ from the
   manuscript's currently-printed text, and are therefore deliberately
   excluded from this snapshot.

Exit code 0 = all checks passed. Exit code 1 = one or more checks failed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from _common import (DATA_ANALYTICAL, DATA_VALIDATION, RESULTS_DIAGNOSTICS, RESULTS_FIGURES,
                      RESULTS_FIGURE_DATA, RESULTS_MODELS, RESULTS_TABLES, get_logger)

log = get_logger("validate")
CODE_DIR = Path(__file__).parent

failures: list[str] = []
passes: int = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global passes
    if condition:
        passes += 1
        log.info(f"PASS  {label}")
    else:
        failures.append(label)
        log.error(f"FAIL  {label}" + (f" -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 1. Required inputs present
# ---------------------------------------------------------------------------

def check_required_inputs() -> None:
    for f in ["oda_clean.csv", "panel_country_sdg_month.csv", "country_year_controls.csv",
              "crisis_news_country_month.csv", "devrel_country_year.csv", "emergency_country_year.csv",
              "outlet_article_counts_long.csv", "stage0_noncrisis_country_month.csv"]:
        check(f"data/analytical/{f} exists", (DATA_ANALYTICAL / f).exists())
    check("classifier checkpoint present", (DATA_VALIDATION / "classifier_checkpoint" / "model.safetensors").exists())
    check("classifier gold set present", (DATA_VALIDATION / "precision_gold_devrel.csv").exists())


# ---------------------------------------------------------------------------
# 2. Panel dimensions and key counts
# ---------------------------------------------------------------------------

def check_panel_dimensions() -> None:
    oda = pd.read_csv(DATA_ANALYTICAL / "oda_clean.csv", encoding="utf-8-sig", low_memory=False,
                       usecols=["project_id"])
    check("ODA project count == 94,412", len(oda) == 94412, f"got {len(oda)}")

    devrel = pd.read_csv(DATA_ANALYTICAL / "devrel_country_year.csv")
    check("devrel panel N == 2,336", len(devrel) == 2336, f"got {len(devrel)}")
    check("devrel panel countries == 146", devrel["country_iso3"].nunique() == 146,
          f"got {devrel['country_iso3'].nunique()}")
    check("devrel panel has no duplicate (country, year) keys",
          not devrel.duplicated(subset=["country_iso3", "year"]).any())

    emergency = pd.read_csv(DATA_ANALYTICAL / "emergency_country_year.csv")
    check("emergency panel N == 2,336 (full universe, pre crisis-media merge)", len(emergency) == 2336,
          f"got {len(emergency)}")
    check("emergency panel has no duplicate (country, year) keys",
          not emergency.duplicated(subset=["country_iso3", "year"]).any())


# ---------------------------------------------------------------------------
# 3. Every required model-result file exists with required columns
# ---------------------------------------------------------------------------

REQUIRED_MODEL_RESULT_FILES = [
    "main_baseline_results.csv", "lag_windows_results.csv", "lag_windows_wald_results.csv",
    "rolling_persistence_results.csv", "sustained_coverage_results.csv",
    "emergency_temporal_results.csv", "emergency_temporal_wald_results.csv",
    "emergency_ppml_contrast_results.csv", "allocation_margin_results.csv",
    "ideological_breadth_results.csv", "evaluative_tone_results.csv", "evaluative_tone_wald_results.csv",
    "placebo_future_news_results.csv", "placebo_nondevrel_results.csv",
    "alt_timing_baseline_results.csv", "leave_one_out_results.csv", "h3_bar_sensitivity_results.csv",
    "reported_text_results.csv",
]

REQUIRED_TABLES = [
    "table1_classifier_performance", "table2_panelB_data_construction", "table2_panelC_descriptives",
    "table3_main_results", "table4_robustness_summary",
    "tableA1_temporal_placebo", "tableA2_media_type_placebo", "tableA3_baseline_alt_timing",
    "tableA4_emergency_alt_timing", "tableA5_leave_one_out", "tableA6_crisis_type_heterogeneity",
    "tableA7_h3_bar_sensitivity",
]

REQUIRED_FIGURES = [
    "figure1_need_media_oda", "figure2_monthly_temporal_profile", "figure3_persistent_attention",
    "figure4_streak_threshold_profile", "figure5_ppml_contrast", "figure6_oda_threshold_sensitivity",
    "figure7_ideological_breadth", "figure8_placebo_battery",
    "figureA1_research_pipeline", "figureA2_classifier_performance",
]


def check_required_outputs() -> None:
    for f in REQUIRED_MODEL_RESULT_FILES:
        p = RESULTS_MODELS / f
        check(f"results/models/{f} exists", p.exists())
        if p.exists():
            df = pd.read_csv(p)
            check(f"results/models/{f} is non-empty", len(df) > 0, f"0 rows")

    for t in REQUIRED_TABLES:
        check(f"results/tables/{t}.csv exists", (RESULTS_TABLES / f"{t}.csv").exists())

    for fig in REQUIRED_FIGURES:
        png_exists = (RESULTS_FIGURES / f"{fig}.png").exists()
        check(f"results/figures/{fig}.png exists", png_exists)

    for diag in ["classifier_performance.csv", "classifier_predictions.csv", "reliability_alpha.csv",
                 "known_discrepancies.csv", "h3_bar_sensitivity_coverage.csv"]:
        check(f"results/diagnostics/{diag} exists", (RESULTS_DIAGNOSTICS / diag).exists())


# ---------------------------------------------------------------------------
# 4. Standardized schema compliance (every model-result file has the
#    required columns from _common.MODEL_RESULT_FIELDS, where applicable)
# ---------------------------------------------------------------------------

_MAIN_RESULT_FILES = [
    "main_baseline_results.csv", "lag_windows_results.csv", "rolling_persistence_results.csv",
    "sustained_coverage_results.csv", "emergency_temporal_results.csv", "emergency_ppml_contrast_results.csv",
    "allocation_margin_results.csv", "ideological_breadth_results.csv", "evaluative_tone_results.csv",
    "placebo_future_news_results.csv", "placebo_nondevrel_results.csv", "alt_timing_baseline_results.csv",
    "h3_bar_sensitivity_results.csv",
]


def check_schema_compliance() -> None:
    from _common import MODEL_RESULT_FIELDS
    for f in _MAIN_RESULT_FILES:
        p = RESULTS_MODELS / f
        if not p.exists():
            continue
        df = pd.read_csv(p)
        missing = [c for c in MODEL_RESULT_FIELDS if c not in df.columns]
        check(f"{f} has all standardized schema columns", not missing, f"missing: {missing}")
        check(f"{f} specification_id values are unique",
              not df["specification_id"].duplicated().any() if "specification_id" in df.columns else False)


# ---------------------------------------------------------------------------
# 5. No stale/orphaned output in results/ (files not written by any known
#    script or referenced by REQUIRED_* lists above)
# ---------------------------------------------------------------------------

def check_no_stale_outputs() -> None:
    known_model_files = set(REQUIRED_MODEL_RESULT_FILES) | {"evaluative_tone_wald_results.csv",
                                                              "lag_windows_wald_results.csv"}
    on_disk = {p.name for p in RESULTS_MODELS.glob("*.csv")}
    unexpected = on_disk - known_model_files
    check("no unexpected files in results/models/", not unexpected, f"unexpected: {sorted(unexpected)}")

    known_table_files = {f"{t}.csv" for t in REQUIRED_TABLES} | {f"{t}.xlsx" for t in REQUIRED_TABLES}
    on_disk_t = {p.name for p in RESULTS_TABLES.glob("*")}
    unexpected_t = on_disk_t - known_table_files
    check("no unexpected files in results/tables/", not unexpected_t, f"unexpected: {sorted(unexpected_t)}")


# ---------------------------------------------------------------------------
# 6. Figure data matches the model-result file it was built from (spot check)
# ---------------------------------------------------------------------------

def check_figure_data_consistency() -> None:
    fig7 = RESULTS_FIGURE_DATA / "figure7_ideological_breadth_data.csv"
    ib = RESULTS_MODELS / "ideological_breadth_results.csv"
    if fig7.exists() and ib.exists():
        fd = pd.read_csv(fig7)
        mr = pd.read_csv(ib)
        interaction = mr[mr["specification_id"] == "ideological_breadth_crosspartisan__media_x_crosspartisan"]
        check("figure7 data file is non-empty and interaction coefficient exists in source model results",
              len(fd) > 0 and len(interaction) == 1)


# ---------------------------------------------------------------------------
# 7. Verification snapshot (expected_final_results.csv) -- sanity check only
# ---------------------------------------------------------------------------

def check_expected_snapshot() -> None:
    snap_path = CODE_DIR / "expected_final_results.csv"
    if not snap_path.exists():
        log.warning("expected_final_results.csv not found -- skipping snapshot comparison")
        return
    snap = pd.read_csv(snap_path)
    for _, row in snap.iterrows():
        p = RESULTS_MODELS / row["source_file"] if (RESULTS_MODELS / row["source_file"]).exists() else \
            RESULTS_DIAGNOSTICS / row["source_file"]
        if not p.exists():
            check(f"snapshot: {row['source_file']} :: {row['specification_id_or_key']}", False, "source file missing")
            continue
        df = pd.read_csv(p)
        key_col = "specification_id" if "specification_id" in df.columns else \
                  ("metric" if "metric" in df.columns else "variable")
        sub = df[df[key_col] == row["specification_id_or_key"]]
        if sub.empty:
            check(f"snapshot: {row['source_file']} :: {row['specification_id_or_key']}", False, "row not found")
            continue
        actual = float(sub.iloc[0][row["field"]])
        expected = float(row["expected_value"])
        tol = float(row["tolerance"])
        ok = abs(actual - expected) <= tol
        check(f"snapshot: {row['source_file']} :: {row['specification_id_or_key']} :: {row['field']} "
              f"(expected {expected}, got {actual:.4f}, tol {tol})", ok)


def main() -> None:
    check_required_inputs()
    check_panel_dimensions()
    check_required_outputs()
    check_schema_compliance()
    check_no_stale_outputs()
    check_figure_data_consistency()
    check_expected_snapshot()

    log.info(f"\n{'=' * 70}\n{passes} passed, {len(failures)} failed\n{'=' * 70}")
    if failures:
        log.error("FAILED checks:")
        for f in failures:
            log.error(f"  - {f}")
        sys.exit(1)
    else:
        log.info("All validation checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
