# Data Availability

This package cannot redistribute two of its inputs because they are licensed
from third-party providers. This document specifies exactly what is missing,
how to obtain it, and what the package can still reproduce without it.

## Summary

| Source | Included in package? | Reason |
|---|---|---|
| BigKinds news corpus (article-level, raw text) | No | Licensed data; redistribution of full article text is prohibited by the provider and by this project's own license terms |
| Korea ODA Statistics (project-level) | Yes, as cleaned CSV | Public government statistics |
| World Bank WDI (GDP per capita, population) | Yes, as cached extract | Public API; small, redistributable derived extract |
| UN Comtrade (bilateral trade) | Yes, as cached extract | Public API; small, redistributable derived extract |
| UCDP GED (conflict deaths) | Yes, as cached extract | Open research license, redistributable with citation |
| EM-DAT / HDX (disaster-affected population) | Yes, as cached extract | Redistributable aggregate proxy (see note below) |
| Fine-tuned classifier checkpoint | Yes | Derived from an open-license base model (`klue/roberta-base`); contains no licensed article text |
| Human-coded reliability/gold labels | Yes | Contains coder-assigned labels and short (~200-character) title/snippet fields only, not licensed full article text |

## 1. BigKinds news corpus — NOT included, must be independently obtained

- **Provider:** Korea Press Foundation (한국언론진흥재단)
- **Dataset:** BigKinds (빅카인즈), https://www.bigkinds.or.kr
- **Scope used in this study:** national daily newspapers only (전국일간지), 2007–2025
- **Access procedure:** BigKinds requires an account (free registration for individuals; institutional/bulk access requires separate arrangement with the Korea Press Foundation). Articles are queried and exported through BigKinds' own search/export interface; full-text redistribution outside BigKinds is not permitted under its terms of use.
- **What this package expects, once you have obtained the raw export:** monthly CSV files named `news_YYYY_MM.csv` (e.g. `news_2010_03.csv`), one file per calendar month from `2007_01` through `2025_12` (228 files), placed under `data/external/news_raw/`. Required fields per row: article ID, publication date, news outlet name, title, keyword list, and body text (or the truncated preview BigKinds provides, whichever your export includes — this pipeline was built and validated against a ~200-character body preview per article, not full text).
- **What this package includes instead:** the *outputs* of classifying that corpus — article counts, not article text. Specifically, the monthly country×development-relevance×crisis-type aggregates already sit in `data/analytical/`, so every regression, table, and figure in this package reproduces without needing the raw corpus at all. Only the classifier's own accuracy/reliability re-verification (`code/03_classifier_validation.py`) and any full from-scratch rebuild of the media panel require the raw corpus. The same holds for `data/analytical/outlet_article_counts_long.csv` (country×year×month×outlet article counts, used only by `code/ideological_breadth.py` / `code/lib/ideology_panel.py` to build the cross-partisan coverage measure) — it is a count-only aggregate with no article text, derived from the same corpus, and its own from-scratch rebuild (`build_outlet_breadth.py` in the original research repository) requires the raw corpus, but the included aggregate does not.

## 2. Korea ODA Statistics — included

- **Provider:** Korea International Cooperation Agency / Ministry of Foreign Affairs, published via the Korea ODA Statistics portal
- **Dataset:** project-level ODA disbursement records, https://www.odakorea.go.kr
- **Years:** 2010–2025
- **Included as:** `data/analytical/oda_clean.csv` (94,412 projects, cleaned and harmonized). Raw per-year source files (`oda_2010.xlsx` … `oda_2025.xlsx`) are public government statistics and are included under `data/external/oda_raw/` for full traceability of the cleaning step.

## 3. World Bank World Development Indicators — included

- **Provider:** World Bank
- **Series:** `NY.GDP.PCAP.KD` (GDP per capita, constant 2015 US$), `SP.POP.TOTL` (population)
- **Access:** public API, no authentication required (https://api.worldbank.org/v2/)
- **Included as:** cached JSON pulls under `data/external/controls/worldbank_*.json`, plus the merged `data/analytical/country_year_controls.csv`

## 4. UN Comtrade — included

- **Provider:** United Nations
- **Series:** bilateral trade with Korea (reporter code 410), both flow directions
- **Access:** public API (https://comtradeplus.un.org/), rate-limited; an API key is recommended for bulk pulls but not required for the volume used here
- **Included as:** cached JSON pulls under `data/external/controls/comtrade_*.json`

## 5. UCDP Georeferenced Event Dataset — included

- **Provider:** Uppsala Conflict Data Program, Uppsala University
- **Series:** state-based armed conflict events, `type_of_violence==1`, summed `best` battle-death estimate by country-year
- **License:** UCDP data are freely available for research use with citation (Sundberg & Melander, 2013; Pettersson et al.)
- **Included as:** cached extract under `data/external/controls/ucdp_ged.csv`

## 6. EM-DAT / HDX disaster data — included, as an aggregate proxy

- **Provider:** Centre for Research on the Epidemiology of Disasters (CRED); this project uses an HDX (Humanitarian Data Exchange) country-profile aggregate rather than the raw event-level EM-DAT export, because the raw EM-DAT export requires an individual registration that is not transferable
- **Series:** total persons affected, natural-disaster group only (no technological/industrial accidents)
- **Included as:** `data/external/controls/emdat_hdx_aggregate.csv`
- **Known limitation:** this control does not cover technological or industrial disasters, only the "natural disaster" group — documented in `docs/DATA_CONSTRUCTION.md`

## 7. Fine-tuned development-relevance classifier — included

- **Base model:** `klue/roberta-base` (Park et al., 2021), open license, available on HuggingFace
- **Fine-tuning data:** hand-labeled Korean-language article samples (title + keyword list + short body preview only — see item 1)
- **Included as:** `data/validation/classifier_checkpoint/` (config, weights, tokenizer) and `data/validation/precision_gold_devrel.csv` (the held-out test set, N=1,954 — title, keyword, short preview, and label only, no restricted full text)

## What you can reproduce without any external access

Running `code/00_run_all.py` out of the box reproduces every table, figure, and text-reported statistic in the manuscript, because every analytical panel needed for estimation is already included in `data/analytical/`. External data acquisition (§1 above) is only required if you want to rebuild the media panel from the raw news corpus rather than start from the already-classified, already-aggregated panel.
