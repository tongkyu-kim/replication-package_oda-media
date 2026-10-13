"""
Placebo tests (Table 4; Appendix Tables A1-A2; Figure 8).

(A) Temporal placebo. Current-year development ODA is regressed on media
    attention measured after the allocation decision:
      - future_news_t1: development-relevant coverage in calendar year t+1;
      - innovation_ar2: the residual of future_news_t1 from an AR(2) first
        stage (country and year FE) on the 2-13-month window and its lag;
      - the 14-25- and 26-37-month pre-allocation windows, each paired with
        its mirror-image future window (t+13..t+24, t+25..t+36).
    Each regression is estimated standalone on its own complete-case sample.

(B) News-substance placebo. For each of the seven headline attention
    measures (2-13-month window; 24/36/60-month rolling averages; streaks of
    at least 24/36/60 months), the same model is estimated on one matched
    sample with three media series: development-relevant coverage, all news
    mentioning the country, and non-development-relevant sports coverage.

Output: results/models/placebo_future_news_results.csv,
        results/models/placebo_news_substance_results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from _common import DATA_ANALYTICAL, RESULTS_MODELS, ModelResultRow, get_logger, require_file, write_model_results
from lib import devrel_panel as dp
from lib import panel_windows as pw
from lib.regressions import fit_fe_ols, result_rows_from_fit

log = get_logger("placebo_tests")

OUT_FUTURE_NEWS = RESULTS_MODELS / "placebo_future_news_results.csv"
OUT_SUBSTANCE = RESULTS_MODELS / "placebo_news_substance_results.csv"
SPORTS_PATH = DATA_ANALYTICAL / "sports_country_month.csv"
ALLNEWS_PATH = DATA_ANALYTICAL / "allnews_country_month.csv"
OUTCOME = "ln_devrel_commitment_musd"


# ---------------------------------------------------------------------------
# (A) Temporal placebo
# ---------------------------------------------------------------------------

MIRROR_WINDOWS = [  # (label, pre-allocation column, future window start_k, end_k)
    ("m14_25", "devrel_media_m14_25", -24, -13),
    ("m26_37", "devrel_media_m26_37", -36, -25),
]


def build_future_news_panel(annual_log: pd.DataFrame, country_month_long: pd.DataFrame) -> pd.DataFrame:
    min_ym, max_ym = pw.corpus_month_bounds(country_month_long)
    d = annual_log[["entity", "year", "devrel_commitment_musd", "devrel_media_m1_13", "devrel_media_m14_25",
                    "devrel_media_m26_37"] + dp.CONTROL_COLS].copy()
    d = d.rename(columns={"devrel_media_m1_13": "news_t"})
    d = pw.attach_window(d, country_month_long, True, start_k=-23, end_k=-12,
                         colname="future_news_t1", agg="sum", min_ym=min_ym, max_ym=max_ym)
    for label, _, start_k, end_k in MIRROR_WINDOWS:
        d = pw.attach_window(d, country_month_long, True, start_k=start_k, end_k=end_k,
                             colname=f"future_{label}", agg="sum", min_ym=min_ym, max_ym=max_ym)
    d = d.sort_values(["entity", "year"]).reset_index(drop=True)
    d["news_tm1"] = d.groupby("entity")["news_t"].shift(1)
    return d


def _standalone(d: pd.DataFrame, x: str, prefix: str, sample: str, window: str, transformation: str):
    res = fit_fe_ols(d, "devrel_commitment_musd", [x] + dp.CONTROL_COLS, verbose_label=prefix)
    return result_rows_from_fit(res, [x], specification_prefix=prefix, outcome=OUTCOME, controls=dp.CONTROL_COLS,
                                sample_definition=sample, timing_window=window, transformation=transformation)


def run_future_news_placebo(annual_log: pd.DataFrame, country_month_long: pd.DataFrame) -> list[ModelResultRow]:
    d = build_future_news_panel(annual_log, country_month_long)

    ar2 = fit_fe_ols(d, "future_news_t1", ["news_t", "news_tm1"], verbose_label="AR(2) first stage")
    resid = ar2.resids.rename("innovation_ar2").reset_index()
    d = d.merge(resid, on=["entity", "year"], how="left")

    rows = _standalone(d, "future_news_t1", "placebo_future_news_raw",
                       "Development-ODA panel, complete cases", "calendar year t+1", "ln(x+1)")
    rows += _standalone(d, "innovation_ar2", "placebo_future_news_ar2_innovation",
                        "Development-ODA panel, AR(2) first-stage sample",
                        "AR(2) innovation of calendar-year t+1 coverage", "AR(2) residual")
    for label, pre_col, start_k, end_k in MIRROR_WINDOWS:
        rows += _standalone(d, pre_col, f"placebo_future_news_{label}_pre",
                            "Development-ODA panel, complete cases",
                            f"{label.replace('m', '').replace('_', '-')} months pre-allocation", "ln(x+1)")
        rows += _standalone(d, f"future_{label}", f"placebo_future_news_{label}_future",
                            "Development-ODA panel, complete cases", f"t+{-end_k}..t+{-start_k} months", "ln(x+1)")
    for r in rows:
        log.info(f"{r.specification_id}: coef={r.coefficient:.4f} se={r.se:.4f} p={r.p_value:.4f} n={r.n}")
    return rows


# ---------------------------------------------------------------------------
# (B) News-substance placebo
# ---------------------------------------------------------------------------

COVERED_ARTICLE_BAR = 3  # articles per month for a month to count as covered (as in the DevRel streak)
MEASURES = [("m1_13", "devrel_media_m1_13"), ("roll24", "devrel_media_roll24"),
            ("roll36", "devrel_media_roll36"), ("roll60", "devrel_media_roll60")]
STREAK_THRESHOLDS = [24, 36, 60]
TIMING_WINDOWS = {
    "m1_13": "2-13 months pre-allocation",
    "roll24": "24-month rolling average (months 2-25)",
    "roll36": "36-month rolling average (months 2-37)",
    "roll60": "60-month rolling average (months 2-61)",
    "streak24": "consecutive coverage streak >= 24 months as of November of year t-1",
    "streak36": "consecutive coverage streak >= 36 months as of November of year t-1",
    "streak60": "consecutive coverage streak >= 60 months as of November of year t-1",
}
TRANSFORMATIONS = {"m1_13": "ln(x+1)", "roll24": "ln(mean+1)", "roll36": "ln(mean+1)", "roll60": "ln(mean+1)",
                   "streak24": "indicator", "streak36": "indicator", "streak60": "indicator"}


def load_month_series(path: Path, col: str) -> pd.DataFrame:
    """Country x month article counts; a country-month absent from the file is zero."""
    require_file(path, f"country x year x month counts (column {col}); see docs/DATA_CONSTRUCTION.md")
    d = pd.read_csv(path, encoding="utf-8-sig").rename(columns={"country_iso3": "entity", col: "freq_articles"})
    return d[["entity", "year", "month", "freq_articles"]]


def attach_window_safe(annual: pd.DataFrame, month_long: pd.DataFrame, valid_ym: set, use_log: bool,
                       start_k: int, end_k: int, colname: str, agg: str = "sum") -> pd.DataFrame:
    """As lib/panel_windows.attach_window, with coverage decided per calendar month (valid_ym)."""
    out = annual.copy()
    parts = []
    for k in range(start_k, end_k + 1):
        cal_month, n_years_back = pw.calendar_month_and_offset(k)
        src = month_long[month_long["month"] == cal_month][["entity", "year", "freq_articles"]].copy()
        src["year"] = src["year"] + n_years_back
        src = src.rename(columns={"freq_articles": f"_wk_{k}"})
        out = out.merge(src, on=["entity", "year"], how="left")
        target_year = out["year"] - n_years_back
        in_valid_month = [(y, cal_month) in valid_ym for y in target_year]
        out[f"_wk_{k}"] = np.where(in_valid_month, out[f"_wk_{k}"].fillna(0.0), np.nan)
        parts.append(f"_wk_{k}")
    complete = out[parts].notna().all(axis=1)
    raw_sum = out[parts].sum(axis=1, min_count=1)
    value = raw_sum if agg == "sum" else raw_sum / len(parts)
    value = value.where(complete)
    out[colname] = np.log1p(value) if use_log else value
    return out.drop(columns=parts)


def _running_streak(s: pd.Series) -> pd.Series:
    counter, out_vals = 0, []
    for v in s:
        if pd.isna(v):
            out_vals.append(np.nan)
            counter = 0
        else:
            counter = counter + 1 if v > 0 else 0
            out_vals.append(counter)
    return pd.Series(out_vals, index=s.index)


def attach_streak(base: pd.DataFrame, month_long: pd.DataFrame, valid_ym: set, min_ym: int, max_ym: int,
                  colname: str) -> pd.DataFrame:
    """Consecutive covered months (>= COVERED_ARTICLE_BAR articles) as of November of year t-1."""
    grid = pd.MultiIndex.from_product(
        [base["entity"].unique(), range(min_ym, max_ym + 1)], names=["entity", "_ym"]).to_frame(index=False)
    grid["year"] = grid["_ym"] // 12
    grid["month"] = grid["_ym"] % 12 + 1
    grid = grid.merge(month_long, on=["entity", "year", "month"], how="left")
    in_valid_month = [(y, m) in valid_ym for y, m in zip(grid["year"], grid["month"])]
    grid["_covered"] = np.where(in_valid_month,
                                (grid["freq_articles"].fillna(0.0) >= COVERED_ARTICLE_BAR).astype(float), np.nan)
    grid = grid.sort_values(["entity", "_ym"])
    grid["_streak"] = grid.groupby("entity")["_covered"].apply(_running_streak).reset_index(level=0, drop=True)
    lookup = grid[["entity", "_ym", "_streak"]].rename(columns={"_ym": "_target_ym"})
    out = base.copy()
    out["_target_ym"] = (out["year"] - 1) * 12 + 10
    out = out.merge(lookup, on=["entity", "_target_ym"], how="left").drop(columns=["_target_ym"])
    out[colname] = out["_streak"].fillna(0.0)
    return out.drop(columns=["_streak"])


def build_substance_panel(annual_log: pd.DataFrame, country_month_long: pd.DataFrame) -> pd.DataFrame:
    min_ym, max_ym = pw.corpus_month_bounds(country_month_long)
    base = annual_log[["entity", "year", "devrel_commitment_musd", "devrel_media_m1_13", "devrel_media_roll24",
                       "devrel_media_roll36", "devrel_media_roll60", "consecutive_periods_covered"]
                      + dp.CONTROL_COLS].copy()
    for name, path, col in [("allnews", ALLNEWS_PATH, "all_news"), ("sports", SPORTS_PATH, "sports")]:
        month_long = load_month_series(path, col)
        valid_ym = set(zip(month_long["year"], month_long["month"]))
        base = attach_window_safe(base, month_long, valid_ym, True, 2, 13, f"{name}_m1_13", agg="sum")
        base = attach_window_safe(base, month_long, valid_ym, True, 2, 25, f"{name}_roll24", agg="mean")
        base = attach_window_safe(base, month_long, valid_ym, True, 2, 37, f"{name}_roll36", agg="mean")
        base = attach_window_safe(base, month_long, valid_ym, True, 2, 61, f"{name}_roll60", agg="mean")
        base = attach_streak(base, month_long, valid_ym, min_ym, max_ym, colname=f"{name}_streak")
    return base


def run_news_substance_placebo(annual_log: pd.DataFrame, country_month_long: pd.DataFrame) -> list[ModelResultRow]:
    base = build_substance_panel(annual_log, country_month_long)
    names = ["allnews", "sports"]
    specs = [(label, dcol, {n: f"{n}_{label}" for n in names}) for label, dcol in MEASURES]
    for thresh in STREAK_THRESHOLDS:
        label = f"streak{thresh}"
        base[f"_devrel_{label}"] = (base["consecutive_periods_covered"] >= thresh).astype(float)
        for n in names:
            base[f"_{n}_{label}"] = (base[f"{n}_streak"] >= thresh).astype(float)
        specs.append((label, f"_devrel_{label}", {n: f"_{n}_{label}" for n in names}))

    rows: list[ModelResultRow] = []
    for label, dcol, cols in specs:
        sample = base.dropna(subset=["devrel_commitment_musd", dcol] + list(cols.values()) + dp.CONTROL_COLS)
        for kind, col in [("devrel", dcol)] + [(n, cols[n]) for n in names]:
            res = fit_fe_ols(sample, "devrel_commitment_musd", [col] + dp.CONTROL_COLS)
            rows.append(ModelResultRow(
                specification_id=f"placebo_substance_{label}_{kind}",
                outcome=OUTCOME,
                key_predictor=col,
                coefficient=float(res.params[col]),
                se=float(res.std_errors[col]),
                p_value=float(res.pvalues[col]),
                n=int(res.nobs),
                n_entities=int(sample["entity"].nunique()),
                controls=";".join(dp.CONTROL_COLS),
                sample_definition="Development-ODA panel, sample matched across the three media series",
                timing_window=TIMING_WINDOWS[label],
                transformation=TRANSFORMATIONS[label],
            ))
            log.info(f"placebo_substance_{label}_{kind}: coef={res.params[col]:.4f} "
                     f"se={res.std_errors[col]:.4f} p={res.pvalues[col]:.4f} n={int(res.nobs)}")
    return rows


def main() -> None:
    country_month_long = pw.build_country_month_long()
    annual_log = dp.build_country_annual(use_log=True, country_month_long=country_month_long,
                                         sentiment_month=dp.build_country_sentiment_month())
    write_model_results(OUT_FUTURE_NEWS, run_future_news_placebo(annual_log, country_month_long))
    write_model_results(OUT_SUBSTANCE, run_news_substance_placebo(annual_log, country_month_long))


if __name__ == "__main__":
    main()
