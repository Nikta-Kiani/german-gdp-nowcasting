"""Utilities for XGBoost-based nowcasting of German GDP.

Provides the analogue of ``nowcast_utils`` in stage 04, but for a tabular
gradient-boosting nowcasting model. The design mirrors the DFM evaluation
pipeline so results are directly comparable:

    - Same evaluation window: 2011Q1 – 2025Q4.
    - Same forecast origin: month 3 of each evaluation quarter.
    - Same expanding training window: 1991Q1 → quarter before the target.
    - Same real-time information set as the DFM: ``apply_pub_lag_mask`` then
      ``fill_ragged_edge_ar`` (Bańbura et al. 2013) before quarterly aggregation.
    - M3-only forecast grid (60 quarters) with DFM-compatible CSV columns.
    - Re-uses the same selection matrices (core / en_only / pls_only)
      plus an additional XGB-Full input set with SHAP-guided pruning inside
      the expanding window.

Key functions
-------------
build_xgb_design_matrix
    Construct a quarterly feature matrix from monthly indicators given a
    monthly forecast origin and a pub-lag map.
tune_xgb
    Time-series-aware ``RandomizedSearchCV`` hyperparameter search.
shap_guided_pruning
    Pre-loop SHAP screening (mean |φ| cumulative-mass + minimum count).
compute_xgb_intervals
    Empirical-error quantile prediction intervals (rolling-history sleeves).
run_xgb_nowcast_loop
    Expanding-window pseudo-real-time evaluation; writes one CSV per spec.

References
----------
Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
    Proc. KDD '16, 785–794.
Lundberg, S. & Lee, S.-I. (2017). A unified approach to interpreting model
    predictions. NeurIPS.
Medeiros, M. C., Vasconcelos, G. F. R., Veiga, Á. & Zilberman, E. (2021).
    Forecasting inflation in a data-rich environment: The benefits of
    machine learning methods. JBES, 39(1), 98–119.
Goulet Coulombe, P., Leroux, M., Stevanovic, D. & Surprenant, S. (2022).
    How is machine learning useful for macroeconomic forecasting? JAE,
    37(5), 920–964.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

try:  # SHAP is optional at import; runtime check is performed where needed.
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover
    shap = None
    SHAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

DEFAULT_LAGS: tuple[int, ...] = (0, 1, 2)
HEADLINE_MIQ: int = 3
RANDOM_STATE: int = 42

DEFAULT_SEARCH_SPACE: dict = {
    "n_estimators":     [200, 300, 400, 500, 600],
    "max_depth":        [2, 3, 4, 5, 6],
    "learning_rate":    [0.01, 0.025, 0.05, 0.075, 0.1],
    "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 2, 4, 6, 10],
    "reg_alpha":        [0.0, 0.001, 0.01, 0.1, 1.0],
    "reg_lambda":       [0.5, 1.0, 2.0, 5.0],
}


@dataclass(frozen=True)
class XGBHyperparams:
    """Container for a fitted XGB hyperparameter set."""

    params: dict
    cv_rmse: float
    n_iter: int


# ---------------------------------------------------------------------------
# Real-time feature construction
# ---------------------------------------------------------------------------

def prepare_monthly_realtime_panel(
    X_monthly: pd.DataFrame,
    origin: pd.Period | str,
    selected_cols: list[str],
    pub_lag_map: pd.Series,
    fill_method: Literal["ar_bic", "none"] = "ar_bic",
    ar_max_p: int = 4,
    ar_min_train: int = 24,
) -> pd.DataFrame:
    """Build the monthly predictor panel at ``origin`` (DFM-aligned).

    Same two-stage pipeline as ``build_dfm_endog`` in stage 04:

    1. ``apply_pub_lag_mask`` — real-time information set.
    2. ``fill_ragged_edge_ar`` — complete current-quarter months (optional).

    The panel spans from the global sample start through the quarter-end
    month of the quarter containing ``origin``.
    """
    from ..dfm.nowcast_utils import get_current_quarter, quarter_end_timestamp
    from ..dfm.ragged_edge import apply_pub_lag_mask, fill_ragged_edge_ar

    origin_p = pd.Period(origin, freq="M")
    current_q = get_current_quarter(origin_p)
    quarter_end_ts = quarter_end_timestamp(current_q)

    monthly_idx = pd.date_range(
        start=X_monthly.index[0],
        end=quarter_end_ts,
        freq="MS",
    )
    X_sub = X_monthly[selected_cols].reindex(monthly_idx)
    lag_sub = pub_lag_map.reindex(selected_cols).fillna(0).astype(int)

    X_masked = apply_pub_lag_mask(X_sub, origin_p, lag_sub)
    if fill_method == "ar_bic":
        X_filled, _ = fill_ragged_edge_ar(
            X_masked, origin_p, lag_sub,
            max_p=ar_max_p, min_train=ar_min_train,
        )
        return X_filled
    return X_masked


def real_time_monthly_mask(
    X_monthly: pd.DataFrame,
    origin: pd.Period,
    pub_lag_map: pd.Series | None,
) -> pd.DataFrame:
    """Legacy mask-only helper. Prefer ``prepare_monthly_realtime_panel``."""
    if pub_lag_map is None:
        X = X_monthly.copy()
        future = X.index > origin.to_timestamp()
        if future.any():
            X.loc[future, :] = np.nan
        return X
    from ..dfm.ragged_edge import apply_pub_lag_mask
    return apply_pub_lag_mask(X_monthly, origin, pub_lag_map)


def aggregate_to_quarterly(
    X_monthly: pd.DataFrame,
    trafo_map: pd.Series | None = None,
) -> pd.DataFrame:
    """Fallback monthly→quarterly aggregation of a *transformed* panel.

    The headline pipeline uses the central raw-level bridge via
    :func:`aggregation.quarterly_block_realtime` in
    :func:`build_xgb_design_matrix`.  This helper applies mean/sum rules
    directly on transformed monthly values when a caller does not use the
    real-time bridge.
    """
    q_idx = X_monthly.index.to_period("Q")

    if trafo_map is None:
        return X_monthly.groupby(q_idx).mean()

    trafo = trafo_map.reindex(X_monthly.columns)
    missing = trafo.index[trafo.isna()].tolist()
    if missing:
        warnings.warn(
            f"trafo_map missing for {len(missing)} columns; defaulting to mean "
            f"aggregation for: {missing[:5]}{'...' if len(missing) > 5 else ''}",
            RuntimeWarning,
            stacklevel=2,
        )
    level_cols = trafo.index[trafo.fillna(0).eq(0)].tolist()
    diff_cols  = trafo.index[~trafo.fillna(0).eq(0)].tolist()

    parts: list[pd.DataFrame] = []
    if level_cols:
        parts.append(X_monthly[level_cols].groupby(q_idx).mean())
    if diff_cols:
        parts.append(X_monthly[diff_cols].groupby(q_idx).sum(min_count=1))

    out = pd.concat(parts, axis=1).reindex(columns=X_monthly.columns)
    out.index.name = "quarter"
    return out


def build_xgb_design_matrix(
    X_monthly: pd.DataFrame,
    y_quarterly: pd.Series,
    origin: pd.Period,
    selected_cols: list[str],
    pub_lag_map: pd.Series | None = None,
    lags: Iterable[int] = DEFAULT_LAGS,
    train_start: str = "1991Q1",
    trafo_map: pd.Series | None = None,
    fill_method: Literal["ar_bic", "none"] = "ar_bic",
    ar_max_p: int = 4,
    ar_min_train: int = 24,
    gdp_lags: tuple[int, ...] = (1, 2),
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build (X_train, y_train, X_pred) for a single forecast origin.

    Procedure
    ---------
    1. Mask with publication lags, then AR(p) BIC fill in Q(t) (``ragged_edge``).
    2. Aggregate to quarterly (mean / sum by ``trafo_map``).
    3. Build a lag panel with lags 0, 1, ..., max(lags). Lag-0 of the current
       (target) quarter is the partial-quarter aggregate available at the
       origin (e.g. mean of January-March readings observed at month 3 given
       pub_lag).
    4. Append GDP autoregressive features (``gdp_lags``, default L1 and L2).
       GDP for q-1 is always released before M3 of quarter q (standard
       nowcasting convention; Bańbura et al. 2013), so lags ≥ 1 are strictly
       real-time safe.  These AR terms close the structural gap with the DFM,
       which embeds past GDP in its Mariano–Murasawa state space.
    5. Align with ``y_quarterly`` over the training window
       ``[train_start, target_quarter - 1]``; rows with all-NaN features are
       dropped, remaining NaNs filled with the column training-window mean
       (XGBoost natively handles NaN but a small fill improves CV stability).
    6. Return the single prediction row indexed by the target quarter.

    Parameters
    ----------
    X_monthly       : full monthly predictor panel.
    y_quarterly     : quarterly GDP first-release series (PeriodIndex).
    origin          : monthly forecast origin (Period).
    selected_cols   : columns of ``X_monthly`` to include as base features.
    pub_lag_map     : optional pd.Series of integer publication lags.
    lags            : iterable of integer quarter lags (0 = current quarter).
    train_start     : training-window start quarter (inclusive).
    trafo_map       : optional pd.Series mapping series id → ``trafo_applied``
        flag (0 = level-stationary, !=0 = log-growth / diff). When provided,
        quarterly aggregation uses the economically motivated rule from
        ``core_utils.monthly_to_quarterly`` (mean for level series, sum for
        log-growth). When None, falls back to mean for all columns.
    gdp_lags        : tuple of positive integer quarter lags for GDP
        autoregressive features (default (1, 2)).  Pass ``()`` to disable.

    Returns
    -------
    X_train : pd.DataFrame, training feature matrix (quarters × features).
    y_train : pd.Series,   training target (quarterly GDP growth).
    X_pred  : pd.DataFrame, single-row feature matrix for the target quarter.
    """
    lags = tuple(int(L) for L in lags)
    if not selected_cols:
        raise ValueError("selected_cols is empty.")
    target_q = origin.asfreq("Q")
    train_start_q = pd.Period(train_start, freq="Q")

    if pub_lag_map is None:
        raise ValueError(
            "pub_lag_map is required for DFM-aligned real-time feature construction."
        )
    X_filled = prepare_monthly_realtime_panel(
        X_monthly=X_monthly,
        origin=origin,
        selected_cols=selected_cols,
        pub_lag_map=pub_lag_map,
        fill_method=fill_method,
        ar_max_p=ar_max_p,
        ar_min_train=ar_min_train,
    )
    # Raw-level bridge: back-transform the masked + AR-filled transformed panel
    # to completed raw levels, aggregate (mean) to quarterly, and re-transform
    # to the stationary quarterly series (central aggregation module).
    from ...selection.aggregation import quarterly_block_realtime

    if trafo_map is None:
        from ...config import paths as _tp
        from ...selection.core_utils import load_trafo_map
        trafo_map = load_trafo_map(_tp.DATA_DICT_ENRICHED_CSV)
    lag_sub = pub_lag_map.reindex(selected_cols).fillna(0).astype(int)
    X_q = quarterly_block_realtime(
        X_filled, origin, lag_sub, trafo_map.reindex(selected_cols),
    )

    parts: dict[str, pd.Series] = {}
    for L in lags:
        shifted = X_q.shift(L)
        for col in shifted.columns:
            parts[f"{col}__L{L}"] = shifted[col]
    feature_panel = pd.DataFrame(parts).sort_index()

    # GDP autoregressive features: shift(k) on a quarterly PeriodIndex means
    # feature_panel["gdp__Lk"][q] == y_quarterly[q - k], which is observed at
    # M3 of q for k >= 1 (no look-ahead bias).
    if gdp_lags:
        y_q_aligned = y_quarterly.reindex(feature_panel.index)
        for k in gdp_lags:
            feature_panel[f"gdp__L{k}"] = y_q_aligned.shift(k)

    train_idx = pd.period_range(train_start_q, target_q - 1, freq="Q")
    X_train_full = feature_panel.reindex(train_idx)
    y_train_full = y_quarterly.reindex(train_idx)

    valid_rows = ~X_train_full.isna().all(axis=1) & y_train_full.notna()
    X_train = X_train_full.loc[valid_rows]
    y_train = y_train_full.loc[valid_rows]

    col_means = X_train.mean(axis=0)
    X_train = X_train.fillna(col_means)

    if target_q not in feature_panel.index:
        raise KeyError(
            f"Target quarter {target_q} not in aggregated feature panel "
            f"(panel range: {feature_panel.index.min()} – "
            f"{feature_panel.index.max()})."
        )
    X_pred = feature_panel.loc[[target_q]].copy()
    X_pred = X_pred.fillna(col_means)

    return X_train, y_train, X_pred


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 40,
    cv_splits: int = 5,
    random_state: int = RANDOM_STATE,
    search_space: dict | None = None,
    verbose: int = 0,
) -> XGBHyperparams:
    """Time-series-aware randomized hyperparameter search for XGBRegressor.

    Uses ``TimeSeriesSplit`` so each validation fold is strictly later than
    its training fold (no leakage). Scoring is negative MSE; the returned
    ``cv_rmse`` is the square root of the best mean MSE. Quick by design
    (n_iter ≈ 40) to keep the §3 tuning step lightweight.

    The found hyperparameters are then **reused for all expanding-window
    origins** in §4 (fixed-design strategy; standard practice in machine-
    learning nowcasting, e.g. Medeiros et al. 2021, Goulet Coulombe et al.
    2022). Optionally a notebook caller may re-invoke this function on a
    different schedule (e.g. every 4 quarters) for robustness checks.
    """
    space = search_space or DEFAULT_SEARCH_SPACE
    base = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    cv = TimeSeriesSplit(n_splits=cv_splits)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=space,
        n_iter=n_iter,
        scoring="neg_mean_squared_error",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=verbose,
        refit=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X_train.values, y_train.values)

    best_mse = -search.best_score_
    return XGBHyperparams(
        params=dict(search.best_params_),
        cv_rmse=float(np.sqrt(max(best_mse, 0.0))),
        n_iter=n_iter,
    )


def make_xgb(params: dict, random_state: int = RANDOM_STATE) -> XGBRegressor:
    """Construct an XGBRegressor with the given tuned params plus fixed
    objective / determinism settings."""
    full = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": 0,
    }
    full.update(params)
    return XGBRegressor(**full)


# ---------------------------------------------------------------------------
# SHAP-guided pruning
# ---------------------------------------------------------------------------

def _base_feature_name(col: str) -> str:
    """Strip the ``__L<n>`` lag suffix to recover the base series id."""
    if "__L" in col:
        return col.rsplit("__L", 1)[0]
    return col


def model_shap_mean_abs(model: XGBRegressor, X: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per column for a fitted XGB model over rows ``X``.

    Uses ``TreeExplainer`` when SHAP is available and falls back to the model's
    gain-based ``feature_importances_`` otherwise. Returned as a Series indexed
    by ``X.columns`` (name ``mean_abs_shap``).
    """
    if SHAP_AVAILABLE:
        try:
            sv = shap.TreeExplainer(model).shap_values(X.values)
            return pd.Series(np.mean(np.abs(sv), axis=0), index=X.columns,
                             name="mean_abs_shap")
        except Exception:
            pass
    return pd.Series(model.feature_importances_, index=X.columns,
                     name="mean_abs_shap")


def shap_guided_pruning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
    keep_frac: float = 0.90,
    min_features: int = 20,
    random_state: int = RANDOM_STATE,
) -> tuple[list[str], pd.Series]:
    """Prune features by SHAP mean-|φ| cumulative mass.

    1. Fit an XGB model with the supplied (tuned) params on (X_train, y_train).
    2. Compute SHAP values on the training rows via ``TreeExplainer``; falls
       back to model ``feature_importances_`` if SHAP is unavailable.
    3. Sum mean |SHAP| over the lag suffixes to a **base-feature** importance
       ranking (avoids picking the same indicator's lag-0 and lag-1 twice).
    4. Keep the top base features whose cumulative mass reaches ``keep_frac``,
       enforcing ``min_features`` lower bound.
    5. Expand the chosen base features back to all their lag columns.

    Returns
    -------
    selected_cols : list[str] of column names from ``X_train`` to keep.
    shap_series   : pd.Series of mean |SHAP| per kept column (for export /
                    visualisation).

    References
    ----------
    Lundberg & Lee (2017); the cumulative-mass screening rule follows the
    common 90% convention in applied SHAP feature-selection (e.g. Buckmann
    et al. 2023, BoE Working Paper No. 1063).
    """
    model = make_xgb(params, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train.values, y_train.values)

    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_train.values)
            per_col = pd.Series(
                np.mean(np.abs(sv), axis=0),
                index=X_train.columns,
                name="mean_abs_shap",
            )
        except Exception:
            per_col = pd.Series(
                model.feature_importances_,
                index=X_train.columns,
                name="mean_abs_shap",
            )
    else:
        per_col = pd.Series(
            model.feature_importances_,
            index=X_train.columns,
            name="mean_abs_shap",
        )

    base = per_col.copy()
    base.index = [_base_feature_name(c) for c in per_col.index]
    base = base.groupby(level=0).sum().sort_values(ascending=False)

    total = float(base.sum())
    if total <= 0:
        return list(X_train.columns), per_col

    cumshare = base.cumsum() / total
    n_keep = int((cumshare < keep_frac).sum() + 1)
    n_keep = max(n_keep, min_features)
    n_keep = min(n_keep, len(base))
    kept_bases = set(base.index[:n_keep])

    selected_cols = [c for c in X_train.columns
                     if _base_feature_name(c) in kept_bases]
    return selected_cols, per_col.loc[selected_cols].sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Empirical-error prediction intervals (rolling sleeves)
# ---------------------------------------------------------------------------

def compute_xgb_intervals(
    nowcast_df: pd.DataFrame,
    alpha: float = 0.10,
    min_history: int = 8,
) -> pd.DataFrame:
    """Add empirical prediction intervals to a nowcast DataFrame.

    For each row (in chronological order) the prediction interval at
    coverage ``1 - alpha`` is

        [nowcast + q_{α/2}(past errors), nowcast + q_{1-α/2}(past errors)]

    where ``past errors`` are the strictly-prior realised errors. Until
    ``min_history`` past errors are available, the interval is NaN. This
    follows the prequential coverage construction in the reference notebook
    (``_preq_cov``) and the classical interval-from-residuals approach
    (Chatfield 1993).

    Modifies ``nowcast_df`` in place by adding ``ci_lower``, ``ci_upper`` and
    returns it.
    """
    cov_label = int(round((1 - alpha) * 100))
    lo_col = f"ci_lower_{cov_label}"
    hi_col = f"ci_upper_{cov_label}"

    df = nowcast_df.copy()
    df[lo_col] = np.nan
    df[hi_col] = np.nan

    errors = df["error"].to_numpy()
    nowcasts = df["nowcast"].to_numpy()

    q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0

    for i in range(len(df)):
        past = errors[:i]
        past = past[~np.isnan(past)]
        if len(past) < min_history or np.isnan(nowcasts[i]):
            continue
        ql = float(np.quantile(past, q_lo))
        qh = float(np.quantile(past, q_hi))
        df.iat[i, df.columns.get_loc(lo_col)] = nowcasts[i] + ql
        df.iat[i, df.columns.get_loc(hi_col)] = nowcasts[i] + qh

    return df


# ---------------------------------------------------------------------------
# Expanding-window XGBoost nowcast loop
# ---------------------------------------------------------------------------

def quarter_to_m3_period(q: pd.Period) -> pd.Period:
    """Return the month-3 monthly Period of a quarterly Period."""
    return pd.Period(q.asfreq("M", how="end"), freq="M")


def run_xgb_nowcast_loop(
    selection_matrix: pd.DataFrame | None,
    X_monthly: pd.DataFrame,
    y_quarterly: pd.Series,
    params: dict,
    quarterly_origins: Iterable[pd.Period | str],
    pub_lag_map: pd.Series | None = None,
    lags: Iterable[int] = DEFAULT_LAGS,
    train_start: str = "1991Q1",
    trafo_map: pd.Series | None = None,
    full_panel_columns: list[str] | None = None,
    coverage_mask: pd.DataFrame | None = None,
    fill_method: Literal["ar_bic", "none"] = "ar_bic",
    ar_max_p: int = 4,
    ar_min_train: int = 24,
    gdp_lags: tuple[int, ...] = (1, 2),
    shap_pruning: bool = False,
    shap_refit_every: int = 4,
    shap_keep_frac: float = 0.90,
    shap_min_features: int = 20,
    shap_log_every: int = 0,
    verbose: bool = True,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window XGBoost nowcast loop with optional SHAP pruning.

    At each quarter ``q`` in ``quarterly_origins``:
        1. Determine the active feature set:
             - From ``selection_matrix`` (if provided) at the month-3 row of
               quarter q (apples-to-apples comparison with DFM specs); OR
             - From ``coverage_mask`` / ``full_panel_columns`` (XGB-Full).
        2. Build the real-time design matrix via ``build_xgb_design_matrix``,
           including GDP autoregressive lags (``gdp_lags``, default L1 and L2).
        3. (Optional) Apply SHAP-guided pruning. To bound cost the pruning
           re-runs every ``shap_refit_every`` quarters; in between, the
           previously kept feature list is reused.  SHAP pruning targets
           indicator columns; GDP lag columns are always retained.
        4. Fit an XGB model with the supplied (tuned) ``params`` on the
           training rows; predict the single target-quarter row.

    Parameters
    ----------
    gdp_lags : tuple of positive integers for GDP AR features (default (1, 2)).
        Passed through to ``build_xgb_design_matrix``.  Pass ``()`` to disable.

    Returns
    -------
    results_df : pd.DataFrame with columns quarter, monthly_origin,
        month_in_quarter (= 3), n_indicators, n_features, nowcast, actual, error.
    shap_log   : pd.DataFrame indexed by (quarter, feature) with the SHAP
        mean |φ| per feature. When ``shap_log_every > 0`` it is recorded for the
        **deployed** model at every ``shap_log_every``-th quarter (1 = quarterly,
        decoupled from the pruning refit so the nowcasts are unaffected);
        otherwise it falls back to the SHAP computed at pruning-refit quarters.
    """
    records: list[dict] = []
    shap_rows: list[dict] = []

    cached_kept: list[str] | None = None
    quarters_since_refit = math.inf
    n_fit = 0

    for q in quarterly_origins:
        q = pd.Period(q, freq="Q")
        m3 = quarter_to_m3_period(q)
        m3_key = str(m3)

        if selection_matrix is not None:
            if m3_key not in selection_matrix.index:
                if verbose:
                    print(f"  {q}: monthly origin {m3_key} not in selection_matrix — skipped.")
                continue
            selected_cols = selection_matrix.columns[
                selection_matrix.loc[m3_key].astype(bool)
            ].tolist()
        else:
            if coverage_mask is not None:
                if m3_key not in coverage_mask.index:
                    if verbose:
                        print(f"  {q}: origin {m3_key} not in coverage_mask — skipped.")
                    continue
                row = coverage_mask.loc[m3_key]
                selected_cols = row.index[row.astype(bool)].tolist()
            elif full_panel_columns is not None:
                selected_cols = list(full_panel_columns)
            else:
                raise ValueError(
                    "Provide selection_matrix, coverage_mask, or full_panel_columns."
                )

        if len(selected_cols) == 0:
            if verbose:
                print(f"  {q}: no indicators selected — skipped.")
            continue

        if verbose:
            tag = "FULL" if selection_matrix is None else "SEL"
            print(f"  {q} ({m3_key}): [{tag}] N={len(selected_cols)} indicators ...",
                  end=" ", flush=True)

        try:
            X_train, y_train, X_pred = build_xgb_design_matrix(
                X_monthly=X_monthly,
                y_quarterly=y_quarterly,
                origin=m3,
                selected_cols=selected_cols,
                pub_lag_map=pub_lag_map,
                lags=lags,
                train_start=train_start,
                trafo_map=trafo_map,
                fill_method=fill_method,
                ar_max_p=ar_max_p,
                ar_min_train=ar_min_train,
                gdp_lags=gdp_lags,
            )

            if len(X_train) < 16:
                if verbose:
                    print(f"too few training rows ({len(X_train)}) — skipped.")
                continue

            n_features_pre = X_train.shape[1]

            if shap_pruning:
                # GDP lag columns are always kept; SHAP pruning targets indicator cols.
                gdp_lag_cols = [c for c in X_train.columns if c.startswith("gdp__L")]
                if (cached_kept is None
                        or quarters_since_refit >= shap_refit_every):
                    kept, shap_vals = shap_guided_pruning(
                        X_train=X_train,
                        y_train=y_train,
                        params=params,
                        keep_frac=shap_keep_frac,
                        min_features=shap_min_features,
                        random_state=random_state,
                    )
                    # Ensure GDP lags are not pruned out
                    kept = list(dict.fromkeys(kept + gdp_lag_cols))
                    cached_kept = kept
                    quarters_since_refit = 0
                    # Only log the pruning-time SHAP when per-quarter logging of
                    # the deployed model is disabled (avoids duplicate quarters).
                    if not shap_log_every:
                        for feat, val in shap_vals.items():
                            shap_rows.append({
                                "quarter": str(q),
                                "feature": feat,
                                "mean_abs_shap": float(val),
                            })
                else:
                    kept = [c for c in cached_kept if c in X_train.columns]
                    # Re-guarantee GDP lags in case they dropped from cached list
                    for col in gdp_lag_cols:
                        if col not in kept:
                            kept.append(col)
                X_train = X_train[kept]
                X_pred = X_pred[kept]
                quarters_since_refit += 1

            model = make_xgb(params, random_state=random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train.values, y_train.values)

            nowcast = float(model.predict(X_pred.values)[0])
            actual = float(y_quarterly.get(q, np.nan))
            error = nowcast - actual if not np.isnan(actual) else np.nan

            # Per-quarter SHAP of the deployed model (decoupled from pruning).
            if shap_log_every and (n_fit % shap_log_every == 0):
                for feat, val in model_shap_mean_abs(model, X_train).items():
                    shap_rows.append({
                        "quarter": str(q),
                        "feature": feat,
                        "mean_abs_shap": float(val),
                    })
            n_fit += 1

            records.append({
                "quarter":          str(q),
                "monthly_origin":   m3_key,
                "month_in_quarter": HEADLINE_MIQ,
                "n_indicators":     len(selected_cols),
                "n_features":       X_train.shape[1],
                "n_features_pre":   n_features_pre,
                "nowcast":          nowcast,
                "actual":           actual,
                "error":            error,
            })
            if verbose:
                print(f"feat={X_train.shape[1]:>4d}  "
                      f"nowcast={nowcast:.3f}  actual={actual:.3f}")
        except Exception as exc:
            warnings.warn(f"{q} ({m3_key}): {exc}", RuntimeWarning, stacklevel=2)
            if verbose:
                print(f"ERROR: {exc}")
            records.append({
                "quarter":          str(q),
                "monthly_origin":   m3_key,
                "month_in_quarter": HEADLINE_MIQ,
                "n_indicators":     len(selected_cols),
                "n_features":       np.nan,
                "n_features_pre":   np.nan,
                "nowcast":          np.nan,
                "actual":           float(y_quarterly.get(q, np.nan)),
                "error":            np.nan,
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.set_index("quarter")
    shap_log = (pd.DataFrame(shap_rows).set_index(["quarter", "feature"])
                if shap_rows else pd.DataFrame())
    return df, shap_log


# ---------------------------------------------------------------------------
# Result loading / caching helpers
# ---------------------------------------------------------------------------

def needs_run(csv_path: str | Path, force: bool = False) -> bool:
    """True if the CSV does not exist or ``force`` is set."""
    return force or not Path(csv_path).exists()


def save_params(path: str | Path, hp: XGBHyperparams) -> None:
    """Serialize tuned XGBoost hyperparameters and their CV score."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump({
            "params":  hp.params,
            "cv_rmse": hp.cv_rmse,
            "n_iter":  hp.n_iter,
        }, fh, indent=2)


def load_params(path: str | Path) -> dict | None:
    """Load serialized XGBoost hyperparameters when available."""
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)
