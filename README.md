# Replication Package: Media Attention and Foreign Aid Allocation in South Korea

## Overview

This package reproduces the data construction, models, robustness checks,
tables, and figures for "Beyond the CNN Effect: Media Attention and Foreign
Aid Allocation in South Korea." It contains only technical replication
material: code, data, and generated outputs. It does not contain the
manuscript itself, manuscript-building scripts, or interpretive material.

## Requirements

- Python 3.10+ (developed and tested on 3.13.1)
- See `requirements.txt` for pinned package versions; `environment/README.md`
  for setup instructions
- No network access, API keys, or `.env` file required to run the pipeline
  as included
- Optional CUDA GPU speeds up classifier validation; CPU works, just slower

## Data

Every panel needed to run the full pipeline is included under
`data/analytical/` and `data/validation/`. Two upstream sources cannot be
redistributed (licensed BigKinds article text; see `DATA_AVAILABILITY.md`
for exactly what is and is not included, and how to obtain the restricted
source if you want to rebuild the media panel from raw text rather than
from the already-classified, already-aggregated counts included here).

## Directory structure

```
replication_package_public/
├── README.md                  this file
├── DATA_AVAILABILITY.md       what's included / restricted, and how to get the rest
├── LICENSE
├── requirements.txt
├── environment/                setup notes
├── data/
│   ├── analytical/              panels every script reads (included)
│   └── validation/               classifier checkpoint + gold/coder labels (included)
├── code/
│   ├── 00_run_all.py            <- run this
│   ├── 99_validate.py
│   ├── expected_final_results.csv   verification snapshot (see 99_validate.py's docstring)
│   ├── _common.py               shared config, paths, standardized result schema
│   ├── lib/                      shared panel-construction and regression helpers
│   └── *.py                      one file per analysis / table / figure (see OUTPUT_MANIFEST.csv)
├── results/
│   ├── models/                   every specification's structured coefficients/tests
│   ├── tables/                    Table 1-4, Appendix Tables A1-A7 (CSV + XLSX)
│   ├── figures/                   Figures 1-8, A1-A2 (PNG/PDF)
│   ├── figure_data/                underlying plotted values for every figure
│   └── diagnostics/                classifier performance, reliability alpha, known discrepancies
├── logs/                        execution log from the last 00_run_all.py run
├── OUTPUT_MANIFEST.csv          every table/figure/text-result -> generating script
└── FILE_MANIFEST.csv            every included file -> purpose
```

## Replication

```bash
cd code
python 00_run_all.py
```

This builds the analytical panels, runs every model and robustness check,
generates every table and figure, consolidates text-only reported numbers,
and validates the output -- in that order, logging to `logs/run_all.log`.

Every script is also independently runnable (`python main_baseline.py`,
etc.) once `data/analytical/` is populated (run `build_analytical_panels.py`
first if starting from a subset of stages).

## Outputs

- **Tables**: `results/tables/table{1,2,3,4}_*.csv` (+ `.xlsx`) and
  `results/tables/tableA{1-7}_*.csv`
- **Figures**: `results/figures/figure{1-8}_*.png` and `figureA{1,2}_*.png`,
  each with underlying data at `results/figure_data/`
- **Every reported model coefficient, SE, p-value, CI, and test statistic**:
  `results/models/*.csv`, in the standardized schema documented in
  `code/_common.py`
- **Every manuscript number reported only in prose** (not already visible in
  a table): `results/models/reported_text_results.csv`
- **Classifier performance and human-coding reliability**:
  `results/diagnostics/classifier_performance.csv`,
  `results/diagnostics/reliability_alpha.csv`

## Status: frozen, 2026-09-20

This package underwent a final reconciliation pass on 2026-09-20. All four
discrepancies found during construction have been resolved, and the
manuscript (`00. Submission/manuscript.docx`) has been synchronized to
match this package's output exactly -- confirmed by a full
manuscript-to-pipeline consistency check covering every reported
coefficient, SE, p-value, N, country count, total, predicted value,
classifier metric, reliability statistic, and timing window. No open
discrepancies remain; `results/diagnostics/known_discrepancies.csv` is
present but empty (header only) as of this freeze.

**Resolution summary** (see each script's module docstring for full detail
and the specific rows/article_ids each fix traces to):

1. **Table 2's dollar-denominated figures.** Were on a disbursement basis
   in the manuscript while the package was already on the commitment basis
   used by every count, regression, and the row's own label. The
   manuscript's Table 2 (Panel B and Panel C) has been updated to the
   package's commitment-basis figures (`generate_tables.py`).
2. **The ideological-breadth measure's timing window.** Corrected from
   k=1-13 to k=2-13 months, matching every other measure in the study
   (`code/lib/ideology_panel.py`). This changed the interaction
   coefficient (Table 3 col.[7]: 0.079\*→0.084\*) and Figure 7's predicted
   ranges; the manuscript has been updated to the corrected values.
3. **The Section 4.8 IV first-stage F-statistics.** A second exhaustive
   search still could not trace the previously-reported F=5.86 (sports) /
   F=1.50 (entertainment) pair to any script in the source repository. Per
   instruction, these two numbers were removed from the manuscript, which
   now retains only the qualitative statement that both instruments had
   first-stage F-statistics well below the conventional relevance
   threshold (`build_reported_text_results.py`).
4. **The crisis-flag reliability alpha.** Traced to a documented
   precondition in `docs/CODING_RULEBOOK.md` ("`label_crisis_flag` only if
   devrel=1") that the alpha computation had not been enforcing --  2 of
   2,001 coded rows violated it. Applying the documented precondition
   (`code/reliability_alpha.py`) moves the regenerated alpha from 0.726 to
   0.739; the manuscript has been updated from 0.748 to 0.739. The
   residual ~0.009 gap from 0.748 could not be further traced and is
   reported, not forced to zero.

**Addendum, 2026-09-20 (same day, post-freeze).** A reviewer/editor-facing
threshold-sensitivity check was added for H3 (sustained attention):
`code/h3_bar_sensitivity.py` re-estimates the >=24/36/60-month streak
specifications at six monthly coverage bars (>=1, >=2, >=3 [manuscript
baseline], >=4, >=5, >=10 devrel-relevant articles/month), holding sample,
controls, FE, clustering, and outcome fixed. Reported as **Appendix Table
A7**; the manuscript's Section 4.7 (Robustness) now cites it. This is an
added robustness check, not a correction -- it does not change any
existing table, figure, or reported coefficient. See
`results/tables/tableA7_h3_bar_sensitivity.csv` and the script's own
docstring for detail.

## Runtime

Approximately 10-15 minutes end to end on a standard laptop CPU (most of
that is classifier validation on CPU; a few seconds on GPU). Use
`--skip-classifier` to skip that stage on a rerun.

## Validation

```bash
cd code
python 99_validate.py
```

Checks required-input presence, panel dimensions and key uniqueness,
existence and non-emptiness of every required table/figure/result file,
standardized-schema compliance, absence of stale/unexpected output files,
and a small verification-snapshot comparison against values confirmed
during this package's construction (`expected_final_results.csv` -- a
sanity check, not a source of truth; see that file and `99_validate.py`'s
docstring for how it's used).
