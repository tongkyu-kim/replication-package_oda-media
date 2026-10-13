"""
Two placebo checks against the devrel-baseline (H1/H2/H3) media-ODA
association, reported in Table 4 / Appendix Tables A1-A2.

(A) Future-news placebo -- does UNEXPECTED future media coverage (which
cannot causally influence an ODA decision already made) "predict" this
year's devrel ODA the same way past coverage does? Two variants:
  raw            : future_news_t1 = ln(1 + sum of articles, Jan-Dec of
                   calendar year t+1) -- a full future calendar year, not
                   a single volatile month.
  ar2_innovation : the UNEXPECTED component of future_news_t1 -- the
                   residual from an AR(2) first stage (country + year FE)
                   that predicts future_news_t1 from already-observable
                   pre-decision history (news_t = devrel_media_m1_13,
                   news_t-1 = the same measure one year earlier). Tests
                   only the innovation, not the persistence any
                   attention-persistence process would already explain.
  Both placebo regressions are standalone (devrel ODA ~ placebo + controls
  + FE), each on its own maximal complete-case sample -- never estimated
  jointly with the pre-allocation regressor.

(B) Non-development (sports) news placebo (negative control) -- does Korean
sports coverage of a country (BigKinds categories 스포츠*), excluding
devrel-relevant articles, "predict" devrel ODA the same way devrel-specific
coverage does? Sports coverage carries no plausible information about
development needs, so an association would point to generic visibility or
a shared confounder rather than development-relevant information.
Entertainment/celebrity coverage, also counted in the source file, is not
used: a plausible affinity channel (favorable celebrity exposure ->
goodwill toward the country -> aid) makes it an invalid negative control. Compared against DEVREL on the IDENTICAL (matched) sample for
each of the 7 headline attention measures: the 2-13mo window (H1), the
three persistent-attention rolling averages (H2: roll24/36/60), and the
three sustained-coverage streak thresholds (H3: >=24/36/60 consecutive
covered months).

Ported unchanged (statistical logic identical) from the original research
repository's pipeline/devrel_baseline/estimate_future_news_innovation.py,
estimate_panelB_future_news_placebo.py, build_nondevrel_placebo.py, and
estimate_panelA_devrel_vs_nondevrel.py.

Writes: results/models/placebo_future_news_results.csv and
results/models/placebo_nondevrel_results.csv
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
OUT_NONDEVREL = RESULTS_MODELS / "placebo_nondevrel_results.csv"

TOPIC_PLACEBO_PATH = DATA_ANALYTICAL / "topic_placebo_country_month.csv"


# ---------------------------------------------------------------------------
# (A) Future-news placebo
# ---------------------------------------------------------------------------

def build_future_news_panel(annual_log: pd.DataFrame, country_month_long: pd.DataFrame) -> pd.DataFrame:
    """news_t (=devrel_media_m1_13, the SAME pre-decision window H1 uses),
    future_news_t1 (ln(1+sum of articles), Jan-Dec of calendar year t+1 --
    start_k=-23..-12 resolves to that full future calendar year), and
    news_tm1 (news_t shifted one year within each country's own panel)."""
    min_ym, max_ym = pw.corpus_month_bounds(country_month_long)
    d = annual_log[["entity", "year", "devrel_commitment_musd", "devrel_media_m1_13"] + dp.CONTROL_COLS].copy()
    d = d.rename(columns={"devrel_media_m1_13": "news_t"})
    d = pw.attach_window(d, country_month_long, True, start_k=-23, end_k=-12,
                          colname="future_news_t1", agg="sum", min_ym=min_ym, max_ym=max_ym)
    d = d.sort_values(["entity", "year"]).reset_index(drop=True)
    d["news_tm1"] = d.groupby("entity")["news_t"].shift(1)
    return d


def run_future_news_placebo(annual_log: pd.DataFrame, country_month_long: pd.DataFrame) -> list[ModelResultRow]:
    d = build_future_news_panel(annual_log, country_month_long)

    # First stage (AR(2), the primary/production spec): future_news_t1 ~
    # news_t + news_tm1 + country FE + year FE. The residual is the
    # "innovation" -- the unexpected component of future news, net of what
    # already-observable pre-decision history would predict.
    ar2_res = fit_fe_ols(d, "future_news_t1", ["news_t", "news_tm1"], verbose_label="future-news AR(2) first stage")
    if ar2_res is not None:
        resid = ar2_res.resids.rename("innovation_ar2").reset_index()
        d = d.merge(resid, on=["entity", "year"], how="left")
        log.info(f"AR(2) first stage: n={int(ar2_res.nobs)}, "
                 f"rho[news_t]={ar2_res.params['news_t']:.4f}, rho[news_tm1]={ar2_res.params['news_tm1']:.4f}")
    else:
        d["innovation_ar2"] = np.nan

    # Raw (unresidualized) placebo -- its own maximal sample, independent of
    # news_t/news_tm1 availability (only future_news_t1 + controls need be
    # non-missing).
    res_raw = fit_fe_ols(d, "devrel_commitment_musd", ["future_news_t1"] + dp.CONTROL_COLS,
                          verbose_label="future-news placebo (raw)")
    rows = result_rows_from_fit(
        res_raw, ["future_news_t1"], specification_prefix="placebo_future_news_raw",
        outcome="ln_devrel_commitment_musd", controls=dp.CONTROL_COLS,
        sample_definition="Devrel-ODA panel, complete cases (raw future-news placebo, its own maximal sample)",
        timing_window="future news, Jan-Dec of calendar year t+1 (raw, unresidualized)",
        transformation="ln(x+1)")

    # AR(2)-innovation placebo -- restricted to the AR(2) first stage's own
    # sample (innovation_ar2 is only defined there).
    res_ar2 = fit_fe_ols(d, "devrel_commitment_musd", ["innovation_ar2"] + dp.CONTROL_COLS,
                          verbose_label="future-news placebo (AR(2) innovation)")
    rows += result_rows_from_fit(
        res_ar2, ["innovation_ar2"], specification_prefix="placebo_future_news_ar2_innovation",
        outcome="ln_devrel_commitment_musd", controls=dp.CONTROL_COLS,
        sample_definition="Devrel-ODA panel, complete cases (AR(2)-innovation placebo, restricted to the "
                           "AR(2) first-stage sample)",
        timing_window="AR(2) innovation (residual) of Jan-Dec year t+1 future news, net of news_t and news_t-1",
        transformation="AR(2) residual (unitless)")

    for r in rows:
        log.info(f"{r.specification_id}: coef={r.coefficient:.4f} se={r.se:.4f} p={r.p_value:.4f} n={r.n}")
    return rows


# ---------------------------------------------------------------------------
# (B) NonDevRel (generic-coverage) placebo
# ---------------------------------------------------------------------------

COVERED_ARTICLE_BAR = 3  # same "covered month" bar as the devrel-side streak (lib/devrel_panel.py)

MEASURES = [
    ("m1_13", "devrel_media_m1_13", "nondevrel_m1_13"),
    ("roll24", "devrel_media_roll24", "nondevrel_roll24"),
    ("roll36", "devrel_media_roll36", "nondevrel_roll36"),
    ("roll60", "devrel_media_roll60", "nondevrel_roll60"),
]
STREAK_THRESHOLDS = [24, 36, 60]

TIMING_WINDOWS = {
    "m1_13": "2-13 months pre-allocation",
    "roll24": "2-25 months pre-allocation (24-month rolling average)",
    "roll36": "2-37 months pre-allocation (36-month rolling average)",
    "roll60": "2-61 months pre-allocation (60-month rolling average)",
    "streak24": "consecutive coverage streak >= 24 months, evaluated as of November of year Y-1",
    "streak36": "consecutive coverage streak >= 36 months, evaluated as of November of year Y-1",
    "streak60": "consecutive coverage streak >= 60 months, evaluated as of November of year Y-1",
}
TRANSFORMATIONS = {
    "m1_13": "ln(x+1)", "roll24": "ln(mean(x)+1), logged after averaging",
    "roll36": "ln(mean(x)+1), logged after averaging", "roll60": "ln(mean(x)+1), logged after averaging",
    "streak24": "binary indicator", "streak36": "binary indicator", "streak60": "binary indicator",
}


def build_nondevrel_month_long(country_month_long: pd.DataFrame) -> pd.DataFrame:
    """Generic placebo series: monthly count of non-devrel sports articles
    mentioning the country (built from the
    raw corpus by the research pipeline's
    stage0_general_media/build_topic_placebo_panel.py, with the devrel
    pipeline's own country detector). Rows exist only for country-months
    with at least one such article; within the months the file covers, a
    missing row is a true zero."""
    require_file(TOPIC_PLACEBO_PATH,
                 "Country x year x month counts of non-devrel sports / entertainment articles "
                 "(columns sports, entertainment, topic) -- see docs/DATA_CONSTRUCTION.md.")
    d = pd.read_csv(TOPIC_PLACEBO_PATH, encoding="utf-8-sig").rename(columns={"country_iso3": "entity"})
    d = d[d["sports"] > 0]  # row convention: a missing country-month is a true zero
    log.info(f"Sports placebo: {len(d):,} country-months, {int(d['sports'].sum()):,} article-country mentions")
    return d[["entity", "year", "month", "sports"]].rename(columns={"sports": "freq_articles"})


def attach_window_safe(annual: pd.DataFrame, month_long: pd.DataFrame, valid_ym: set, use_log: bool,
                        start_k: int, end_k: int, colname: str, agg: str = "sum") -> pd.DataFrame:
    """Same construction as lib/panel_windows.attach_window, but "in range"
    is decided per-(year,month) against valid_ym (the set of (year,month)
    the generic placebo source panel actually has rows for) rather than
    a single contiguous [min_ym,max_ym] bound -- gap-aware, kept for
    correctness even though the current corpus has no whole-year gaps."""
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


def _running_streak_nan_aware(s: pd.Series) -> pd.Series:
    counter, out_vals = 0, []
    for v in s:
        if pd.isna(v):
            out_vals.append(np.nan)
            counter = 0
        else:
            counter = counter + 1 if v > 0 else 0
            out_vals.append(counter)
    return pd.Series(out_vals, index=s.index)


def attach_nondevrel_streak(base: pd.DataFrame, nondevrel_m: pd.DataFrame, valid_ym: set,
                             min_ym: int, max_ym: int) -> pd.DataFrame:
    """Consecutive-coverage streak on the generic (non-devrel) side, at the
    SAME true monthly granularity and >=3-articles/month bar as the
    devrel-side streak (lib/devrel_panel.attach_streak) -- an
    apples-to-apples placebo comparison, not a raw article-count streak on
    a different scale."""
    grid = pd.MultiIndex.from_product(
        [base["entity"].unique(), range(min_ym, max_ym + 1)], names=["entity", "_ym"]
    ).to_frame(index=False)
    grid["year"] = grid["_ym"] // 12
    grid["month"] = grid["_ym"] % 12 + 1
    grid = grid.merge(nondevrel_m[["entity", "year", "month", "freq_articles"]],
                       on=["entity", "year", "month"], how="left")
    in_valid_month = [(y, m) in valid_ym for y, m in zip(grid["year"], grid["month"])]
    grid["_covered"] = np.where(in_valid_month,
                                 (grid["freq_articles"].fillna(0.0) >= COVERED_ARTICLE_BAR).astype(float), np.nan)
    grid = grid.sort_values(["entity", "_ym"])
    grid["_streak_months"] = (
        grid.groupby("entity")["_covered"].apply(_running_streak_nan_aware).reset_index(level=0, drop=True)
    )
    streak_lookup = grid[["entity", "_ym", "_streak_months"]].rename(columns={"_ym": "_target_ym"})

    out = base.copy()
    out["_target_ym"] = (out["year"] - 1) * 12 + 10  # November of year Y-1 (k=2), matching the devrel side
    out = out.merge(streak_lookup, on=["entity", "_target_ym"], how="left")
    out = out.drop(columns=["_target_ym"])
    # A target month before the corpus began is true zero streak, not
    # missing -- matches the devrel side's own fillna(0) treatment.
    out["nondevrel_streak"] = out["_streak_months"].fillna(0.0)
    return out.drop(columns=["_streak_months"])


def build_nondevrel_panel(annual_log: pd.DataFrame, country_month_long: pd.DataFrame) -> pd.DataFrame:
    nondevrel_m = build_nondevrel_month_long(country_month_long)
    valid_ym = set(zip(nondevrel_m["year"], nondevrel_m["month"]))
    min_ym, max_ym = pw.corpus_month_bounds(country_month_long)

    base = annual_log[["entity", "year", "devrel_commitment_musd", "devrel_media_m1_13", "devrel_media_roll24",
                        "devrel_media_roll36", "devrel_media_roll60", "consecutive_periods_covered"]
                       + dp.CONTROL_COLS].copy()
    base = attach_window_safe(base, nondevrel_m, valid_ym, True, 2, 13, "nondevrel_m1_13", agg="sum")
    base = attach_window_safe(base, nondevrel_m, valid_ym, True, 2, 25, "nondevrel_roll24", agg="mean")
    base = attach_window_safe(base, nondevrel_m, valid_ym, True, 2, 37, "nondevrel_roll36", agg="mean")
    base = attach_window_safe(base, nondevrel_m, valid_ym, True, 2, 61, "nondevrel_roll60", agg="mean")
    base = attach_nondevrel_streak(base, nondevrel_m, valid_ym, min_ym, max_ym)
    return base


def run_nondevrel_placebo(annual_log: pd.DataFrame, country_month_long: pd.DataFrame) -> list[ModelResultRow]:
    base = build_nondevrel_panel(annual_log, country_month_long)

    pairs = list(MEASURES)
    for thresh in STREAK_THRESHOLDS:
        label = f"streak{thresh}"
        dcol, ncol = f"_devrel_{label}", f"_nondevrel_{label}"
        base[dcol] = (base["consecutive_periods_covered"] >= thresh).astype(float)
        base[ncol] = (base["nondevrel_streak"] >= thresh).astype(float)
        pairs.append((label, dcol, ncol))

    rows: list[ModelResultRow] = []
    for label, dcol, ncol in pairs:
        # Matched sample: dropna on BOTH regressors (+ controls), so DEVREL
        # and NonDevRel/generic are each fit on the identical set of rows.
        sample = base.dropna(subset=["devrel_commitment_musd", dcol, ncol] + dp.CONTROL_COLS).copy()
        for kind, col in [("devrel", dcol), ("generic", ncol)]:
            res = fit_fe_ols(sample, "devrel_commitment_musd", [col] + dp.CONTROL_COLS,
                              verbose_label=f"nondevrel placebo {label} ({kind})")
            if res is None:
                continue
            rows.append(ModelResultRow(
                specification_id=f"placebo_nondevrel_{label}_{kind}",
                outcome="ln_devrel_commitment_musd",
                key_predictor=col,
                coefficient=float(res.params[col]),
                se=float(res.std_errors[col]),
                p_value=float(res.pvalues[col]),
                n=int(res.nobs),
                n_entities=int(sample["entity"].nunique()),
                controls=";".join(dp.CONTROL_COLS),
                sample_definition=f"Devrel-ODA panel, matched sample (dropna on both devrel and generic "
                                   f"{label} regressors + controls)",
                timing_window=TIMING_WINDOWS[label],
                transformation=TRANSFORMATIONS[label],
            ))
            log.info(f"placebo_nondevrel_{label}_{kind}: coef={res.params[col]:.4f} "
                     f"se={res.std_errors[col]:.4f} p={res.pvalues[col]:.4f} n={int(res.nobs)}")
    return rows


def main() -> None:
    country_month_long = pw.build_country_month_long()
    sentiment_month = dp.build_country_sentiment_month()
    annual_log = dp.build_country_annual(use_log=True, country_month_long=country_month_long,
                                          sentiment_month=sentiment_month)

    future_news_rows = run_future_news_placebo(annual_log, country_month_long)
    write_model_results(OUT_FUTURE_NEWS, future_news_rows)
    log.info(f"Wrote {len(future_news_rows)} rows -> {OUT_FUTURE_NEWS}")

    nondevrel_rows = run_nondevrel_placebo(annual_log, country_month_long)
    write_model_results(OUT_NONDEVREL, nondevrel_rows)
    log.info(f"Wrote {len(nondevrel_rows)} rows -> {OUT_NONDEVREL}")


if __name__ == "__main__":
    main()
