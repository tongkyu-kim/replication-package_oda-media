"""
Shared two-way fixed-effects estimation and Wald-test helpers.

Ported from the original research repository's shared `_fe_ols` /
`_wald_zero` / `_wald_equal` pattern (used identically across every
regression script in the study). Every specification in this package uses
entity + time fixed effects, country-clustered standard errors -- this
module is the single place that logic lives, so every analysis script
calls it the same way instead of re-implementing panel fitting.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from _common import ModelResultRow, HYPOTHESIS_TEST_FIELDS, CLUSTER_LEVEL, FE_STRUCTURE_BASELINE


def fit_fe_ols(d: pd.DataFrame, y: str, xs: list[str], *, entity_col: str = "entity",
               time_col: str = "year", verbose_label: str = ""):
    """Two-way FE panel OLS, country-clustered SE. Returns the fitted
    linearmodels result (or None if the sample is degenerate), exactly as
    the original pipeline's _fe_ols did."""
    dd = d.dropna(subset=[y] + xs).copy()
    if dd.empty or dd[entity_col].nunique() < 2:
        if verbose_label:
            print(f"  [{verbose_label}] insufficient data, skipping")
        return None
    pdata = dd.set_index([entity_col, time_col])
    exog = pdata[xs].assign(const=1.0)
    mod = PanelOLS(pdata[y], exog, entity_effects=True, time_effects=True)
    return mod.fit(cov_type="clustered", cluster_entity=True)


def result_rows_from_fit(res, xs: list[str], *, specification_prefix: str, outcome: str,
                          controls: list[str], sample_definition: str, timing_window: str,
                          transformation: str, year_range: str = "") -> list[ModelResultRow]:
    """One ModelResultRow per requested coefficient in `xs` (skips control
    coefficients unless explicitly listed in xs)."""
    if res is None:
        return []
    rows = []
    n_entities = int(pd.Series(res.params.index).shape[0]) if False else None
    for x in xs:
        rows.append(ModelResultRow(
            specification_id=f"{specification_prefix}__{x}",
            outcome=outcome,
            key_predictor=x,
            coefficient=float(res.params[x]),
            se=float(res.std_errors[x]),
            p_value=float(res.pvalues[x]),
            n=int(res.nobs),
            n_entities=_n_entities(res),
            year_range=year_range,
            fe_structure=FE_STRUCTURE_BASELINE,
            cluster_level=CLUSTER_LEVEL,
            controls=";".join(controls) if controls else "none",
            sample_definition=sample_definition,
            timing_window=timing_window,
            transformation=transformation,
        ))
    return rows


def _n_entities(res) -> Optional[int]:
    try:
        return int(pd.Index(res.model.dependent.entities).nunique())
    except Exception:
        try:
            return int(len(set(idx[0] for idx in res.resids.index)))
        except Exception:
            return None


def wald_zero(res, coef_names: list[str]) -> tuple[float, int, float]:
    """H0: all named coefficients == 0, jointly."""
    params = list(res.params.index)
    R = np.zeros((len(coef_names), len(params)))
    for i, name in enumerate(coef_names):
        R[i, params.index(name)] = 1.0
    wt = res.wald_test(restriction=R, value=np.zeros(len(coef_names)))
    return float(wt.stat), int(wt.df), float(wt.pval)


def wald_equal(res, coef_a: str, coef_b: str) -> tuple[float, int, float]:
    """H0: coef_a == coef_b."""
    params = list(res.params.index)
    R = np.zeros((1, len(params)))
    R[0, params.index(coef_a)] = 1.0
    R[0, params.index(coef_b)] = -1.0
    wt = res.wald_test(restriction=R, value=np.zeros(1))
    return float(wt.stat), int(wt.df), float(wt.pval)


def hypothesis_test_row(specification_id: str, test_name: str, stat: float, df: int,
                         pval: float, restriction_tested: str, n: int) -> dict:
    return {
        "specification_id": specification_id,
        "test_name": test_name,
        "test_statistic": stat,
        "df": df,
        "p_value": pval,
        "restriction_tested": restriction_tested,
        "n": n,
    }
