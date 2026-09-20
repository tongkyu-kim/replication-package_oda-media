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
│   ├── figures/                   Figures 1-8, A2 (PNG)
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
- **Figures**: `results/figures/figure{1-8}_*.png` and `figureA2_*.png`,
  each with underlying data at `results/figure_data/`. Figure A1 (the
  research-pipeline diagram) is produced manually and is not part of this
  package.
- **Every reported model coefficient, SE, p-value, CI, and test statistic**:
  `results/models/*.csv`, in the standardized schema documented in
  `code/_common.py`
- **Every manuscript number reported only in prose** (not already visible in
  a table): `results/models/reported_text_results.csv`
- **Classifier performance and human-coding reliability**:
  `results/diagnostics/classifier_performance.csv`,
  `results/diagnostics/reliability_alpha.csv`

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
