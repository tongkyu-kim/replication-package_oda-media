"""
Figure 6 -- Sensitivity of the media coefficient across minimum ODA
thresholds.

Refits the SAME extensive-margin specification code/allocation_margin.py
already estimates -- LPM: P(devrel ODA received >= threshold) ~ media +
controls, country + year FE, country-clustered SE, fit on the full
zero-filled country x year grid (allocation_margin.build_zero_filled_panel)
-- at a dense grid of alternative minimum-commitment thresholds, instead of
the single fixed ">$0" cut point allocation_margin.py itself reports as its
D1 "extensive margin" row.

Ported (statistical logic identical) from the original research
repository's pipeline/mechanisms/estimate_model5_dense_threshold_profile.py:
thresholds are the p1, p2, ..., p99 percentiles of the EMPIRICAL
distribution of POSITIVE annual country-level devrel ODA (not arbitrary
round-dollar values, and not selected on significance) -- 99 points. Every
regression uses the full zero-filled panel (same n_countries x n_years);
only the binary threshold cut point moves, so only the share of rows coded
1 shrinks toward the tails. The original script's 7 round-dollar benchmark
thresholds ($0/$100k/$250k/$500k/$1M/$2M/$5M) are reference-only there and
are not part of the plotted dense grid -- not reproduced here either.

Regenerated from this package's OWN current data/analytical/
devrel_country_year.csv via allocation_margin.py's own
build_zero_filled_panel() and MEDIA_COL (rather than copying the original
repository's cached results file), so the figure is guaranteed consistent
with this package's own current data and code -- per the release note that
an earlier audit pass found the original repo's cached dense-grid CSV
current as of Sep 4, but this port re-derives it directly rather than
trusting that snapshot stayed in sync with this package's own build.

Writes: results/figure_data/figure6_oda_threshold_data.csv and
results/figures/figure6_oda_threshold_sensitivity.png (+ .pdf)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from _common import (RESULTS_FIGURES, RESULTS_FIGURE_DATA, SIGNIFICANCE_CI_LEVEL, Z_90,  # noqa: E402
                      get_logger, write_rows)
from lib import panel_windows as pw  # noqa: E402
from lib import devrel_panel as dp  # noqa: E402
from lib.regressions import fit_fe_ols  # noqa: E402
from allocation_margin import build_zero_filled_panel, MEDIA_COL  # noqa: E402

log = get_logger("make_figure6_oda_threshold_sensitivity")

OUT_FIGDATA = RESULTS_FIGURE_DATA / "figure6_oda_threshold_data.csv"
OUT_PNG = RESULTS_FIGURES / "figure6_oda_threshold_sensitivity.png"
OUT_PDF = RESULTS_FIGURES / "figure6_oda_threshold_sensitivity.pdf"

FIGDATA_FIELDS = ["percentile", "threshold_usd", "coefficient", "se", "ci_lower", "ci_upper",
                   "ci_level", "p_value", "n"]

PERCENTILES = list(range(1, 100))  # p1..p99 of the empirical positive-ODA distribution


def run_one(d0: pd.DataFrame, thr_musd: float) -> dict | None:
    d = d0.copy()
    d["oda_above_thr"] = (d["devrel_commitment_musd"] >= thr_musd).astype(float)
    res = fit_fe_ols(d, "oda_above_thr", [MEDIA_COL] + dp.CONTROL_COLS,
                      verbose_label=f"threshold sensitivity (thr={thr_musd:.5f} MUSD)")
    if res is None:
        return None
    coef = float(res.params[MEDIA_COL])
    se = float(res.std_errors[MEDIA_COL])
    p = float(res.pvalues[MEDIA_COL])
    return {"coefficient": coef, "se": se, "p_value": p, "n": int(res.nobs),
            "ci_lower": coef - Z_90 * se, "ci_upper": coef + Z_90 * se}


def build_figure_data(d0: pd.DataFrame) -> list[dict]:
    positive = d0.loc[d0["devrel_commitment_musd"] > 0, "devrel_commitment_musd"]
    log.info(f"n_positive={len(positive)} country-years for the empirical threshold grid")
    rows = []
    for pct in PERCENTILES:
        thr_musd = float(positive.quantile(pct / 100))
        r = run_one(d0, thr_musd)
        if r is None:
            log.warning(f"p{pct}: insufficient data, skipping")
            continue
        rows.append({"percentile": pct, "threshold_usd": thr_musd * 1_000_000,
                      "ci_level": SIGNIFICANCE_CI_LEVEL, **r})
        log.info(f"[p{pct:>2d}, thr=${thr_musd * 1e6:,.0f}] coef={r['coefficient']:.5f} se={r['se']:.5f} "
                  f"p={r['p_value']:.4f} n={r['n']}")
    return rows


def make_plot(fig_df: pd.DataFrame) -> None:
    fig_df = fig_df.sort_values("threshold_usd")
    x = fig_df["threshold_usd"].to_numpy()
    coef = fig_df["coefficient"].to_numpy()
    lo = fig_df["ci_lower"].to_numpy()
    hi = fig_df["ci_upper"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.fill_between(x, lo, hi, color="gray", alpha=0.25, linewidth=0, label="90% CI", zorder=1)
    ax.plot(x, coef, color="black", lw=1.4, label="Media coefficient", zorder=3)
    ax.axhline(0, color="black", lw=0.8, ls="--", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("Minimum annual ODA threshold, $ (log scale)")
    ax.set_ylabel("Coefficient on development-related media\nattention (extensive margin, LPM)")
    ax.set_title("Fig. 6: Sensitivity of the media coefficient across minimum ODA thresholds")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    country_month_long = pw.build_country_month_long()
    d0 = build_zero_filled_panel(country_month_long)
    log.info(f"Zero-filled panel: {len(d0):,} rows ({d0['country_iso3'].nunique()} countries), "
              f"{d0['oda_positive'].sum():,.0f} positive-ODA rows ({d0['oda_positive'].mean():.1%})")

    rows = build_figure_data(d0)
    write_rows(OUT_FIGDATA, FIGDATA_FIELDS, rows)
    log.info(f"Wrote {len(rows)} rows -> {OUT_FIGDATA}")

    fig_df = pd.read_csv(OUT_FIGDATA, encoding="utf-8-sig")
    make_plot(fig_df)
    log.info(f"Wrote {OUT_PNG} and {OUT_PDF}")


if __name__ == "__main__":
    main()
