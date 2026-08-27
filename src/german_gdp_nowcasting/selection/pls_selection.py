"""PLS + VIP indicator selection.

Variable Importance in Projection (VIP) scores from Partial Least Squares
(Wold, 1966; Mehmood et al., 2012). Components maximise covariance with the
target rather than predictor variance. On this German panel the method is
the extreme hard-data selector (fixed at 30 series). It is not a more
accurate nowcast than the elastic net: Part II does not reject equal DFM
accuracy across input sets.

References
----------
Wold, H. (1966). Estimation of principal components and related models by
    iterative least squares. In P. R. Krishnaiah (Ed.), Multivariate Analysis.
Kelly, B. & Pruitt, S. (2015). The three-pass regression filter: A new approach
    to forecasting using many predictors. Journal of Econometrics, 186(2).
Mehmood, T. et al. (2012). A review of variable selection methods in PLS.
    Chemom. Intell. Lab. Syst., 118, 62-69.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

from .core_utils import (
    FixedKFit,
    align_quarterly_xy,
    build_coverage_mask,
    monthly_to_quarterly,
    training_end_quarter,
)


def _make_imputer(
    imputer_strategy: str,
) -> IterativeImputer | SimpleImputer:
    """Return an sklearn imputer for the given strategy string."""
    if imputer_strategy == "iterative":
        return IterativeImputer(max_iter=10, random_state=42, skip_complete=True)
    return SimpleImputer(strategy="mean")


# ---------------------------------------------------------------------------
# PLS + VIP selection
# ---------------------------------------------------------------------------

def pls_vip_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_components: int = 5,
    top_k: int = 30,
    imputer_strategy: str = "mean",
) -> FixedKFit:
    """Select top_k variables by PLS Variable Importance in Projection (VIP) scores.

    VIP formula (Wold 1994):

        VIP_j = sqrt( p × Σ_h(W*_jh² × SSY_h) / SSY_total )

    where W* is the normalised PLS weight matrix, SSY_h is the sum of squared
    y-variance explained by component h, and p is the number of predictors.

    Parameters
    ----------
    X : pd.DataFrame
        Quarterly predictor matrix (training window).
    y : pd.Series
        Quarterly GDP log-growth target.
    n_components : int
        Number of PLS latent components.  Capped at min(n_obs − 1, n_vars).
    top_k : int
        Number of variables to select (by descending VIP score).
    imputer_strategy : str
        ``"mean"`` (default) or ``"iterative"``.

    Returns
    -------
    FixedKFit with ``selected_variables`` ranked by VIP score (highest first)
    and ``alpha`` set to NaN (not applicable for PLS).
    """
    valid = y.notna()
    X_fit = X.loc[valid].copy()
    y_fit = y.loc[valid].values.reshape(-1, 1)

    scl = StandardScaler()
    X_arr = scl.fit_transform(
        _make_imputer(imputer_strategy).fit_transform(X_fit)
    )

    n_comp = min(n_components, X_arr.shape[0] - 1, X_arr.shape[1])
    pls = PLSRegression(n_components=n_comp, scale=False)
    pls.fit(X_arr, y_fit)

    # W*: (p, n_comp) normalised x-weights; T: (n, n_comp) x-scores
    # Q:  (1, n_comp) y-loadings
    W = pls.x_weights_        # (p, n_comp)
    T = pls.x_scores_         # (n, n_comp)
    Q = pls.y_loadings_       # (1, n_comp)

    p = X_arr.shape[1]
    ssy = np.sum(T ** 2, axis=0) * (Q ** 2).ravel()   # (n_comp,)
    ssy_total = ssy.sum()

    if ssy_total == 0:
        warnings.warn(
            "PLS: total SSY is zero; returning first top_k columns by order.",
            stacklevel=2,
        )
        selected = X.columns[:top_k].tolist()
        return FixedKFit(alpha=np.nan, selected_variables=selected, n_selected=len(selected))

    # Normalise weights column-wise before squaring
    W_norm = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    vip = np.sqrt(p * np.sum((W_norm ** 2) * ssy[None, :], axis=1) / ssy_total)

    vip_series = pd.Series(vip, index=X.columns, name="vip")
    top_vars = vip_series.nlargest(top_k).index.tolist()

    return FixedKFit(alpha=np.nan, selected_variables=top_vars, n_selected=len(top_vars))


# ---------------------------------------------------------------------------
# Expanding-window PLS selection
# ---------------------------------------------------------------------------

def run_expanding_selection_pls(
    X_monthly: pd.DataFrame,
    y_quarterly: pd.Series,
    trafo_map: pd.Series,
    forecast_origins: Iterable[pd.Period | str],
    coverage_mask: pd.DataFrame | None = None,
    train_start_quarter: str = "1991Q1",
    n_components: int = 5,
    top_k: int = 30,
    imputer_strategy: str = "mean",
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Expanding-window PLS + VIP selection for each monthly forecast origin.

    Follows the same expanding-window and within-quarter caching logic as
    ``run_expanding_selection``.  Produces a separate ``selection_matrix_pls``
    suitable for cross-method comparison or as an alternative DFM input.

    Publication-lag constraints (unbalanced panel, ragged edge) are the
    responsibility of the DFM — the PLS operates on the full historical
    training sample (backward-looking predictive relevance).

    Parameters
    ----------
    X_monthly : monthly predictor panel.
    y_quarterly : first-release GDP log-growth.
    trafo_map : pd.Series mapping series id → trafo_applied code.
    forecast_origins : monthly PeriodIndex.
    coverage_mask : optional precomputed mask.
    train_start_quarter : start of expanding window.
    n_components : number of PLS latent components.
    top_k : number of variables to select per origin.
    imputer_strategy : ``"mean"`` (default) or ``"iterative"``.

    Returns
    -------
    selection_matrix_pls : pd.DataFrame
        Binary origins × series matrix.
    results_pls : dict
        Per-origin diagnostics (selected variables, n_components, top_k).
    """
    X_quarterly = monthly_to_quarterly(X_monthly, trafo_map)
    X_quarterly, y_quarterly = align_quarterly_xy(X_quarterly, y_quarterly)
    train_start = pd.Period(train_start_quarter, freq="Q")

    origins = [pd.Period(o) for o in forecast_origins]
    if coverage_mask is None:
        coverage_mask = build_coverage_mask(X_monthly, origins)

    selection_rows: dict[str, pd.Series] = {}
    results: dict[str, dict] = {}
    _cache: dict[str, tuple[pd.Series, dict]] = {}

    for origin in tqdm(origins, desc="Expanding PLS selection", unit="origin"):
        origin_key = str(origin)
        end_q = training_end_quarter(origin)
        cache_key = str(end_q)

        if cache_key in _cache:
            cached_row, cached_template = _cache[cache_key]
            selection_rows[origin_key] = cached_row.copy()
            results[origin_key] = {**cached_template, "origin": origin_key}
            continue

        train_idx = X_quarterly.index[
            (X_quarterly.index >= train_start) & (X_quarterly.index <= end_q)
        ]
        if train_idx.empty:
            raise ValueError(f"No training data for origin {origin_key}.")

        y_train_full = y_quarterly.reindex(train_idx)
        valid_idx = y_train_full.index[y_train_full.notna()]
        if len(valid_idx) < n_components + 2:
            raise ValueError(
                f"Too few target observations for origin {origin_key}: "
                f"{len(valid_idx)} < {n_components + 2}."
            )

        if origin_key not in coverage_mask.index:
            raise ValueError(f"Coverage mask is missing origin {origin_key}.")

        candidate_cols = coverage_mask.columns[coverage_mask.loc[origin_key]].tolist()

        X_train = X_quarterly.loc[valid_idx, candidate_cols]
        y_train = y_quarterly.loc[valid_idx]

        nonempty = X_train.notna().any()
        X_train = X_train.loc[:, nonempty]
        if X_train.empty:
            raise ValueError(
                f"No non-empty candidate predictors for origin {origin_key}."
            )

        fit = pls_vip_selection(
            X_train, y_train,
            n_components=n_components,
            top_k=top_k,
            imputer_strategy=imputer_strategy,
        )

        row = pd.Series(0, index=X_monthly.columns, dtype=int)
        row.loc[[s for s in fit.selected_variables if s in row.index]] = 1

        template = {
            "train_start": str(train_start),
            "train_end": cache_key,
            "n_train_quarters": int(len(valid_idx)),
            "n_candidate_variables": int(X_train.shape[1]),
            "n_selected_variables": fit.n_selected,
            "selected_variables": fit.selected_variables,
            "n_components": n_components,
            "top_k": top_k,
        }
        _cache[cache_key] = (row, template)
        selection_rows[origin_key] = row
        results[origin_key] = {**template, "origin": origin_key}

    selection_matrix_pls = pd.DataFrame.from_dict(
        selection_rows, orient="index"
    ).astype(int)
    selection_matrix_pls.index.name = "forecast_origin"
    return selection_matrix_pls, results
