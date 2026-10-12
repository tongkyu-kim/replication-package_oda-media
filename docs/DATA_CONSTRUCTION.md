# Data Construction

Technical reference for every variable and panel this package uses. No
interpretation or rationale for design choices is included here -- only
what each variable is and how it is built. See `DATA_AVAILABILITY.md` for
what is and is not included/redistributable.

## Raw and pre-built inputs (`data/analytical/`)

| File | Rows | Built by | Notes |
|---|---|---|---|
| `oda_clean.csv` | 94,412 | Cleaned, project-level ODA records, 2010-2025 | One row per ODA project. Columns include `country_iso3`, `year`, `commitment_musd`, `disbursement_musd`, `net_disbursement_musd`, `bilateral_multilateral` ("다자간" = multilateral, else bilateral), `sector_code` (OECD CRS purpose code), `sdg` (SDG tag where assigned by the source crosswalk), `project_id` |
| `panel_country_sdg_month.csv` | ~large | Classified media corpus, aggregated to country x SDG x month article counts | Included pre-built (article COUNTS only, not text) -- see `DATA_AVAILABILITY.md` item 1 for why the raw corpus is not included/reconstructed here |
| `country_year_controls.csv` | 5,386 | Country x year controls, already merged from cached external-API pulls | Columns: `gdppc_constant2015usd`, `ln_gdppc`, `population`, `ln_population`, `bilateral_trade_usd`, `log_bilateral_trade`, `conflict_deaths`, `ln_conflict_deaths`, `disaster_affected`, `ln_disaster_affected` |
| `crisis_news_country_month.csv` | -- | Crisis-type media counts by country-month | Six crisis types: disaster, climate, conflict, health, food, refugee |

## Derived panels (built by `build_analytical_panels.py`)

### `devrel_country_year.csv` -- development-ODA outcome panel

- **Grid**: every (country, year) combination for 2010-2025 across all 146 countries with any recorded ODA project -- zero-filled, not presence-only. A country-year with no ODA project of any kind is an explicit `0.0`, not a missing row.
- **devrel_flag**: `NOT (bilateral_multilateral == "다자간" AND sdg is null)`. Excludes only International Organization / Multilateral Development Bank (IO/MDB) core-capital subscriptions (1,127 projects, $5,452.79M as of the current `oda_clean.csv`). A multilateral-flagged row that DOES carry an SDG tag (a real sector attribution, e.g. Korea-directed multilateral trust-fund spending) is retained in devrel.
- **Output columns**: `total_commitment_musd`, `total_n_projects`, `devrel_commitment_musd`, `devrel_n_projects`, `devrel_share`.
- Confirmed: 2,336 rows, 146 countries.

### `emergency_country_year.csv` -- emergency-ODA outcome panel

- Same zero-filled (country, year) grid, built from the full `oda_clean.csv` universe (not just SDG-tagged rows).
- **Emergency definition**: OECD CRS sector_code in `{72010, 72040, 72050, 73010, 74010}` or with prefix in `{720, 730, 740}` -- the humanitarian-response purpose-code family, read directly from the raw sector code (not from any downstream SDG label).
- Confirmed: 2,167 emergency-flagged projects (2.30% of all ODA projects), matching Table 2.

### H4 stacked panel (`lib/h4_panel.py`, built in memory by `h4_common_exposure.py` and `leave_one_out.py`)

- **Outcomes**: the pooled development-relevant outcome used for H1-H3 (`devrel_commitment_musd`) is partitioned into two mutually exclusive components per country-year: **emergency ODA** (`emergency_commitment_musd` from `emergency_country_year.csv`) and **non-emergency ODA** (pooled minus emergency). Every emergency-flagged project already satisfies `devrel_flag`, so the partition is exact: emergency + non-emergency = pooled in all 2,336 country-years (max absolute error 3e-14 $M). One non-emergency country-year is negative (a downward commitment revision) and is clipped to 0 for PPML, as everywhere else.
- **Stacking**: each country-year contributes one emergency and one non-emergency row (4,672 rows), with `is_emergency`, `entity_x` = country x aid type, and `time_x` = year x aid type for the aid-type-specific fixed effects. Standard errors are clustered on the original country.
- **Common exposure**: both rows carry IDENTICAL development-relevant media measures -- the production DevRel measure, which intentionally includes crisis-related coverage. Windows: contemporaneous (`devrel_media_contemp`, months 0-11 of the commitment year, k=-11..0, ln(1+sum)), plus the H1-H3 measures (`devrel_media_m1_13`, `_m14_25`, `_m26_37`, `_roll24/36/60`, `streak_ge24/36/60`).
- **Checks** (`results/diagnostics/h4_common_exposure_data_checks.csv`, enforced by `h4_common_exposure.py` and `99_validate.py`): 0 media-exposure mismatches across 23,360 country-year x measure cells; partition reconciles to the pooled outcome; emergency ODA is zero in 65.3% of country-years, non-emergency in 11.9%.

## Country x year analytical panel (`lib/devrel_panel.py`, used by every regression script)

Built fresh by each analysis script from the panels above (not itself cached to disk, since it's inexpensive to rebuild and every script needs a slightly different subset of its columns):

- **Recent attention** (`devrel_media_m1_13`): sum of monthly development-relevant article counts over months 2-13 before the allocation year (calendar month arithmetic, not a relabeled prior-year block -- generalizes cleanly to any window width). December of year Y-1 (k=1) is excluded because Korea's constitutional budget deadline falls within 30 days before the start of the fiscal year (early December), so that month can contain post-decision media.
- **Lagged cycles** (`devrel_media_m14_25`, `devrel_media_m26_37`): same construction, non-overlapping 12-month blocks further back.
- **Persistent attention** (`devrel_media_roll24/36/60`): the MEAN monthly article count over a 24/36/60-month window (months 2-25, 2-37, 2-61 respectively), log-transformed after averaging, not before.
- **Sustained attention / streak** (`consecutive_periods_covered`, then `streak_ge24/36/60`): built at true monthly granularity on a dense (entity, month) grid. A month is "covered" if it has >=3 development-relevant articles. The running count resets to zero on any month below that bar. Evaluated as of November of year Y-1 (the near edge of the 2-13 month window) for each target year.
- **Zero-fill vs. missing**: a source (entity, year, month) with no rows at all is true zero coverage IF that month falls inside the news corpus's active date range; a request reaching before the corpus began returns a genuine missing value (coverage before the corpus started is unknown, not zero).
- **Sentiment** (`avg_sentiment`, `share_positive`, `share_negative`): article-count-weighted mean across SDG rows per country-month, then averaged over the same 2-13 month window as recent attention.
- **Outcome transformation**: `ln(devrel_commitment_musd + 1)` in the log specification (the primary specification throughout); untransformed in the levels specification.
- **Controls**: `ln_gdppc`, `ln_population`, `log_bilateral_trade`, `ln_conflict_deaths`, `ln_disaster_affected` -- merged from `country_year_controls.csv` on (country_iso3, year).

## Missingness

A regression's estimation sample is always the complete-case intersection of its outcome, key predictor(s), and (where included) the five controls -- rows with any missing value among those are dropped for that specification only, not dropped from the underlying panel. This is why different specifications report different N even though they draw on the same underlying panel (e.g. N=2,336 with no controls vs. N=2,214 once the five controls are required).
