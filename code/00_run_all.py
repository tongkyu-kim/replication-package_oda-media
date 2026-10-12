"""
Single entry point for the full replication pipeline. Runs every stage in
the required order and writes a timestamped execution log to logs/.

Usage:
  python 00_run_all.py             # full pipeline
  python 00_run_all.py --skip-classifier   # skip classifier_validation.py
                                             (the slowest stage; useful for
                                             a quick rerun of everything else)
  python 00_run_all.py --only STAGE1,STAGE2   # run only named stages (see
                                                 STAGES below for names)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import ensure_dirs, get_logger, LOGS_DIR

log = get_logger("run_all")
CODE_DIR = Path(__file__).parent

# (stage name, script filename, human description) -- executed strictly in
# this order. Each script is independently runnable on its own (see each
# script's own module docstring), but later stages assume earlier stages'
# results/models/*.csv and data/analytical/*.csv already exist.
STAGES: list[tuple[str, str, str]] = [
    ("build_data",            "build_analytical_panels.py",     "1. Build analytical datasets"),
    ("classifier",            "classifier_validation.py",       "2. Classifier validation"),
    ("reliability",           "reliability_alpha.py",           "2b. Human-coding reliability (Krippendorff's alpha)"),
    ("main_baseline",         "main_baseline.py",               "3. Main model: H1 recent attention"),
    ("lag_windows",           "lag_windows.py",                 "3b. Temporal decomposition diagnostic"),
    ("rolling_persistence",   "rolling_persistence.py",         "4. Temporal model: H2 persistent attention"),
    ("sustained_coverage",    "sustained_coverage.py",           "4b. Temporal model: H3 sustained attention"),
    ("emergency_temporal",    "emergency_temporal.py",          "4c. Exploratory: crisis media and emergency ODA (former H4 specification)"),
    ("h4_common_exposure",    "h4_common_exposure.py",          "4d. Main model: H4 emergency vs. non-emergency ODA (common DevRel exposure)"),
    ("allocation_margin",     "allocation_margin.py",           "5. Secondary analysis: allocation margin"),
    ("ideological_breadth",   "ideological_breadth.py",         "5b. Secondary analysis: ideological breadth"),
    ("evaluative_tone",       "evaluative_tone.py",             "5c. Secondary analysis: evaluative tone"),
    ("placebo_tests",         "placebo_tests.py",               "6. Robustness: placebo tests"),
    ("alt_timing",            "alt_timing.py",                  "6b. Robustness: alternative timing"),
    ("leave_one_out",         "leave_one_out.py",               "6c. Robustness: leave-one-out sensitivity"),
    ("h3_bar_sensitivity",    "h3_bar_sensitivity.py",          "6d. Robustness: H3 coverage-bar sensitivity (Appendix Table A6)"),
    ("fig1",  "make_figure1_need_media_oda.py",          "7a. Figure 1 data + plot"),
    ("fig2",  "make_figure2_monthly_temporal_profile.py", "7b. Figure 2 data + plot"),
    ("fig3",  "make_figure3_persistent_attention.py",     "7c. Figure 3 data + plot"),
    ("fig4",  "make_figure4_streak_threshold_profile.py", "7d. Figure 4 data + plot"),
    ("fig5",  "make_figure5_ppml_contrast.py",             "7e. Figure 5 data + plot"),
    ("fig6",  "make_figure6_oda_threshold_sensitivity.py", "7f. Figure 6 data + plot"),
    ("fig7",  "make_figure7_ideological_breadth.py",       "7g. Figure 7 plot (data already built by ideological_breadth.py)"),
    ("fig8",  "make_figure8_placebo_battery.py",           "7h. Figure 8 data + plot"),
    ("figA2", "make_figureA2_classifier_performance.py",   "7i. Figure A2 data + plot"),
    ("tables",                "generate_tables.py",             "8. Generate all tables (Table 1-4, A1-A7)"),
    ("reported_text",         "build_reported_text_results.py", "9. Consolidate text-only reported numbers"),
    ("validate",              "99_validate.py",                 "10. Validation checks"),
]


def run_stage(name: str, script: str, desc: str) -> tuple[bool, float]:
    path = CODE_DIR / script
    if not path.exists():
        log.error(f"[{name}] script not found: {path} -- skipping")
        return False, 0.0
    log.info(f"{'=' * 70}\n{desc}\n{'=' * 70}")
    t0 = time.time()
    result = subprocess.run([sys.executable, str(path)], cwd=str(CODE_DIR))
    elapsed = time.time() - t0
    ok = result.returncode == 0
    level = log.info if ok else log.error
    level(f"[{name}] {'OK' if ok else 'FAILED'} ({elapsed:.1f}s)")
    return ok, elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-classifier", action="store_true",
                     help="Skip classifier_validation.py (the slowest stage)")
    ap.add_argument("--only", type=str, default=None,
                     help="Comma-separated stage names to run (see STAGES for names)")
    args = ap.parse_args()

    ensure_dirs()
    stages = STAGES
    if args.only:
        wanted = set(args.only.split(","))
        stages = [s for s in STAGES if s[0] in wanted]
    if args.skip_classifier:
        stages = [s for s in stages if s[0] != "classifier"]

    log.info(f"Running {len(stages)} stage(s)")
    results = []
    t_start = time.time()
    for name, script, desc in stages:
        ok, elapsed = run_stage(name, script, desc)
        results.append((name, ok, elapsed))
        if not ok:
            log.error(f"Stopping: stage '{name}' failed. Fix the error above, then rerun with "
                      f"--only {name},<remaining stages> to continue without repeating earlier stages.")
            break

    total = time.time() - t_start
    log.info(f"\n{'=' * 70}\nSUMMARY ({total:.1f}s total)\n{'=' * 70}")
    for name, ok, elapsed in results:
        log.info(f"  {'OK  ' if ok else 'FAIL'}  {name:<22} {elapsed:6.1f}s")

    if all(ok for _, ok, _ in results) and len(results) == len(stages):
        log.info("\nAll stages completed successfully.")
        sys.exit(0)
    else:
        log.error("\nOne or more stages failed or were not run. See log above and logs/run_all.log.")
        sys.exit(1)


if __name__ == "__main__":
    main()
