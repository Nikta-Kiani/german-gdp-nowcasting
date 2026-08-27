"""Utilities for mixed-frequency DFM nowcasting.

Implements the A-CD-TPN DFM pipeline from Franjic & Schweikert (2025):
    A   = Arithmetic aggregation of monthly predictors to quarterly
    CD  = Coordinate Descent (sklearn ElasticNet CV)
    TPN = Targeted Predictor Nowcasting

Also implements evaluation helpers (RMSFE, DM, MCS, NSR) shared with XGB/MLP stages.

References
----------
Franjic, D. & Schweikert, K. (2025). "4DFM ..."
Giannone, D., Reichlin, L. & Small, D. (2008). Nowcasting: The real-time informational
    content of macroeconomic data. Journal of Monetary Economics, 55(4), 665–676.
Doz, C., Giannone, D. & Reichlin, L. (2011). A two-step estimator for large approximate
    dynamic factor models based on Kalman filtering. Journal of Econometrics, 164, 188–205.
Mariano, R. S. & Murasawa, Y. (2003). A new coincident index of business cycles based
    on monthly and quarterly series. Journal of Applied Econometrics, 18(4), 427–443.
Bai, J. & Ng, S. (2002). Determining the number of factors in approximate factor models.
    Econometrica, 70(1), 191–221.
Bai, J. & Ng, S. (2008). Forecasting economic time series using targeted predictors.
    Journal of Econometrics, 146(2), 304–317.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ


# ---------------------------------------------------------------------------
# GDP representation helpers
# ---------------------------------------------------------------------------

def quarterly_to_monthly_rep(
    y_quarterly: pd.Series,
    monthly_index: pd.DatetimeIndex,
) -> pd.Series:
    """Convert quarterly GDP series to monthly representation (Mariano-Murasawa).

    Places the quarterly value at the last month of each quarter (month 3, 6, 9, 12).
    All other months receive NaN.  This is the standard mixed-frequency DFM encoding
    following Mariano & Murasawa (2003).

    Parameters
    ----------
    y_quarterly : pd.Series with PeriodIndex (quarterly frequency).
    monthly_index : pd.DatetimeIndex (monthly frequency, month-start or any day).

    Returns
    -------
    pd.Series with the same monthly_index, NaN except at quarter-end months.
    """
    gdp_m = pd.Series(np.nan, index=monthly_index, name=y_quarterly.name)
    for period, value in y_quarterly.items():
        # Quarter-end month timestamp
        end_month = period.asfreq("M", how="end").to_timestamp()
        if end_month in gdp_m.index:
            gdp_m.at[end_month] = value
    return gdp_m


def get_current_quarter(origin: pd.Period | str) -> pd.Period:
    """Return the quarter being nowcasted at a given monthly origin.

    The current quarter is the quarter that the monthly origin belongs to.
    E.g., 2011-02 → 2011Q1  (month 2 is inside Q1).
    """
    p = pd.Period(origin)
    if p.freqstr.startswith("M"):
        return p.asfreq("Q")
    if p.freqstr.startswith("Q"):
        return p
    raise ValueError(f"Unsupported origin frequency: {p.freqstr!r}")


def quarter_end_timestamp(quarter: pd.Period) -> pd.Timestamp:
    """Return the last-month-start timestamp for a quarter (e.g., 2011Q1 → 2011-03-01)."""
    return quarter.asfreq("M", how="end").to_timestamp()


# ---------------------------------------------------------------------------
# Bai & Ng (2002) information criterion for number of factors
# ---------------------------------------------------------------------------

def bai_ng_ic2(X_scaled: np.ndarray, max_k: int = 8) -> int:
    """Select number of factors by Bai & Ng (2002) IC2 criterion.

    Parameters
    ----------
    X_scaled : (T × N) array, mean-zero, unit-variance standardised.
    max_k    : maximum number of factors to consider.

    Returns
    -------
    Optimal k ≥ 1.
    """
    T, N = X_scaled.shape
    if T == 0 or N == 0:
        return 1

    max_k = min(max_k, T - 1, N)

    # IC2 penalty g(N,T) = (N+T)/(NT) * log(min(N,T))
    # Using Bai-Ng (2002) equation (9)
    g = (N + T) / (N * T) * np.log(min(N, T))

    # SVD for all eigenvalues at once (efficient)
    try:
        U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    except np.linalg.LinAlgError:
        return 1

    ic_values: list[float] = []
    for k in range(1, max_k + 1):
        factors = U[:, :k] * s[:k]           # T × k
        loadings = Vt[:k, :]                  # k × N
        residuals = X_scaled - factors @ loadings
        sigma2_k = float(np.mean(residuals ** 2))
        if sigma2_k <= 0:
            ic_values.append(-np.inf)
        else:
            ic_values.append(np.log(sigma2_k) + k * g)

    return int(np.argmin(ic_values)) + 1  # +1 because range starts at 1


def select_n_factors(
    X_monthly_train: pd.DataFrame,
    n_selected: int,
    max_k: int = 8,
) -> int:
    """Choose number of DFM factors: Bai-Ng IC2 capped at min(N/2, 5).

    Parameters
    ----------
    X_monthly_train : monthly predictor data (training window only; may contain
                      NaN for ragged-start series — handled via mean imputation).
    n_selected      : number of selected indicators (N).
    max_k           : hard upper bound before the cap is applied.

    Returns
    -------
    k ≥ 1, capped at min(floor(N/2), 5).
    """
    cap = min(int(np.floor(n_selected / 2)), 5)
    cap = max(cap, 1)  # at least 1 factor

    # Drop rows that are entirely NaN (no information at all), then apply
    # column-wise mean imputation for the remaining sporadic NaN entries
    # (ragged-start series). Bai-Ng (2002) derived IC2 for a balanced panel;
    # mean imputation is the standard practical approximation for unbalanced
    # macro panels and preserves the covariance structure far better than
    # zero-fill, which biases the SVD after standardisation.
    X_vals_raw = X_monthly_train.dropna(how="all").values
    if X_vals_raw.shape[0] < 10 or X_vals_raw.shape[1] < 2:
        return min(1, cap)

    imputer = SimpleImputer(strategy="mean")
    X_vals = imputer.fit_transform(X_vals_raw)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_vals)

    k_ic = bai_ng_ic2(X_std, max_k=min(max_k, cap))
    return min(k_ic, cap)


# ---------------------------------------------------------------------------
# DFM endog builder
# ---------------------------------------------------------------------------

def build_dfm_endog(
    X_monthly_sel: pd.DataFrame,
    y_quarterly: pd.Series,
    origin: pd.Period | str,
    pub_lag_map: pd.Series | None = None,
    fill_method: Literal["ar_bic", "none"] = "ar_bic",
    ar_max_p: int = 4,
    ar_min_train: int = 24,
) -> tuple[pd.DataFrame, int]:
    """Assemble the mixed-frequency endog DataFrame for DynamicFactorMQ.

    Procedure
    ---------
    1. Monthly index spans 1991-01-01 to the **quarter-end** of the current
       quarter (= last month of the quarter that origin belongs to).
    2. Per-series real-time masking: for each column j, observations after
       ``origin - pub_lag_j`` are set to NaN (ragged-edge, no leakage).
    3. AR(p) BIC fill (when ``fill_method="ar_bic"``): univariate AR fitted
       only on observed history through ``origin - pub_lag_j``, used to
       complete missing months up to the quarter-end of Q(t).  This ensures
       the DFM receives a *complete* monthly indicator panel for the current
       quarter at every origin.
    4. The GDP column is placed at month-3 of each completed quarter through
       the last completed quarter before the origin; GDP for the current
       (target) quarter is NaN.
    5. Monthly variables must come FIRST in the DataFrame (k_endog_M columns),
       followed by the quarterly GDP variable.

    Parameters
    ----------
    X_monthly_sel : monthly predictor panel (all dates × selected columns).
    y_quarterly   : quarterly GDP first-release series (PeriodIndex).
    origin        : monthly forecast origin (pd.Period or str).
    pub_lag_map   : optional pd.Series mapping column id → publication lag
        in months.  When provided, each column is masked at origin - pub_lag
        (real-time unbalanced-panel setup).  When None, all columns are masked
        from origin + 1 onwards (balanced panel, original behaviour).
    fill_method   : 'ar_bic' (default) — fill ragged months in Q(t) using
        univariate AR(p) BIC; 'none' — leave NaN as in the previous pipeline.
    ar_max_p      : maximum AR order for BIC selection (default 4).
    ar_min_train  : minimum non-NaN training observations for AR(1+) (default 24).

    Returns
    -------
    endog     : pd.DataFrame with monthly DatetimeIndex, shape (T_m, N+1).
    k_endog_M : number of monthly variables (= N after degenerate-column drop).
    """
    from .ragged_edge import apply_pub_lag_mask, fill_ragged_edge_ar

    if not isinstance(y_quarterly.index, pd.PeriodIndex):
        y_quarterly = y_quarterly.copy()
        y_quarterly.index = pd.PeriodIndex(y_quarterly.index, freq="Q")

    origin = pd.Period(origin)
    current_q = get_current_quarter(origin)
    quarter_end_ts = quarter_end_timestamp(current_q)

    # Full monthly DatetimeIndex from panel start to quarter-end
    panel_start = X_monthly_sel.index[0]
    monthly_idx = pd.date_range(start=panel_start, end=quarter_end_ts, freq="MS")

    # Monthly indicators: reindex to full monthly grid first
    X_sel_full = X_monthly_sel.reindex(monthly_idx)

    if pub_lag_map is None:
        # Balanced-panel fallback: mask all months after origin
        origin_ts = origin.to_timestamp()
        future_mask = monthly_idx > origin_ts
        X_sel_full.loc[future_mask, :] = np.nan
    else:
        # Stage 1 — per-series real-time masking (ragged edge)
        X_sel_full = apply_pub_lag_mask(X_sel_full, origin, pub_lag_map)

        # Stage 2 — AR(p) BIC fill of current-quarter missing months
        if fill_method == "ar_bic":
            # Restrict pub_lag_map to the columns in the panel
            lag_sub = pub_lag_map.reindex(X_sel_full.columns, fill_value=0)
            X_sel_full, _ = fill_ragged_edge_ar(
                X_sel_full, origin, lag_sub,
                max_p=ar_max_p, min_train=ar_min_train,
            )

    # GDP column: place values at quarter-end months through last completed quarter
    last_completed_q = current_q - 1
    y_obs = y_quarterly.loc[
        y_quarterly.index <= last_completed_q
    ] if not y_quarterly.empty else y_quarterly

    gdp_monthly = quarterly_to_monthly_rep(y_obs, monthly_idx)

    # Assemble: monthly variables first, GDP last
    endog = pd.concat([X_sel_full, gdp_monthly.rename("gdp")], axis=1)

    k_endog_M = X_sel_full.shape[1]
    return endog, k_endog_M


# ---------------------------------------------------------------------------
# DFM fitting and nowcast extraction
# ---------------------------------------------------------------------------

def _drop_degenerate_endog_columns(
    endog: pd.DataFrame,
    k_endog_M: int,
) -> tuple[pd.DataFrame, int]:
    """Remove monthly columns with no usable observations; keep GDP last."""
    if k_endog_M <= 0 or endog.shape[1] <= 1:
        return endog, k_endog_M

    monthly = endog.iloc[:, :k_endog_M]
    gdp_col = endog.columns[-1]
    keep_monthly = monthly.columns[monthly.notna().any()]
    if len(keep_monthly) == k_endog_M:
        return endog, k_endog_M

    endog_clean = pd.concat([monthly[keep_monthly], endog[[gdp_col]]], axis=1)
    return endog_clean, len(keep_monthly)


#: Generous upper bound on |log-likelihood| for a fitted DFM. Valid fits on
#: these mixed-frequency panels have llf of order 1e3-1e4; a degenerate EM
#: solution collapses to ~1e30+. The 1e8 threshold leaves >20 orders of
#: magnitude of margin, so it flags only genuine divergence.
_DFM_LLF_ABSURD_BOUND = 1e8


def _dfm_fit_is_sane(result: Any) -> bool:
    """Return True if a fitted DFM is numerically usable.

    Guards against EM converging to a degenerate solution on thin/ill-
    conditioned real-time panels: such a fit does not raise, but its
    log-likelihood collapses to an absurd value (e.g. -1e32) and the implied
    nowcast explodes (e.g. -1.6e17). Detected here so ``fit_dfm`` can fall
    through to a more robust configuration instead of returning the blow-up.
    """
    llf = getattr(result, "llf", None)
    if llf is None:
        return False
    llf = float(llf)
    if not np.isfinite(llf):
        return False
    if abs(llf) > _DFM_LLF_ABSURD_BOUND:
        return False
    return True


def fit_dfm(
    endog: pd.DataFrame,
    k_endog_M: int,
    k_factors: int,
    factor_order: int = 2,
    idiosyncratic_ar1: bool = True,
    maxiter: int = 200,
) -> object:
    """Fit DynamicFactorMQ via EM algorithm on mixed-frequency data.

    Parameters
    ----------
    endog       : mixed-frequency DataFrame (monthly index, monthly cols first,
                  quarterly GDP last).
    k_endog_M   : number of monthly columns.
    k_factors   : number of global factors (from Bai-Ng IC2).
    factor_order: VAR order for factor dynamics (default 2).
                  Lehmann, Reif & Wollmershäuser (2020) use AR(2) factor
                  dynamics in ifoCAST; AR(2) better captures typical business-
                  cycle persistence (~6–12 months) than AR(1).
    idiosyncratic_ar1 : if True, adds AR(1) to idiosyncratic components
                        (default True; follows Doz et al. 2011 and is closer
                        to ifoCAST's AR(2) idiosyncratic structure).
    maxiter     : maximum EM iterations.

    Returns
    -------
    Fitted DynamicFactorMQResults object.

    Notes
    -----
    EM can fail numerically on ill-conditioned panels (e.g. thin real-time
    panels at the 2020Q1 COVID break). A small fallback ladder is tried before
    raising: ``standardize=False`` and/or ``idiosyncratic_ar1=False``.  Point
    nowcasts from ``predict()`` remain valid under these fallbacks.

    Two distinct failure modes are handled:
      1. The fit raises (``infs or NaNs`` / ``LinAlgError``) — caught and the
         next configuration is tried.
      2. The EM *converges to a degenerate solution* without raising: the
         log-likelihood collapses to an absurd value (e.g. −1e32) and
         ``predict()`` returns astronomically large nowcasts (e.g. 1e17). This
         is detected via ``_dfm_fit_is_sane`` and treated exactly like a failed
         configuration, so the ladder degrades to a robust specification (here
         ``idiosyncratic_ar1=False`` recovers a sane fit). Returning the first
         non-raising fit unconditionally — as before — would silently propagate
         the blow-up into the nowcast (Doz et al. 2011; Bańbura et al. 2013).
    """
    endog, k_endog_M = _drop_degenerate_endog_columns(endog, k_endog_M)
    if k_endog_M == 0:
        raise ValueError("No monthly endogenous variables with observations.")

    fit_configs = [
        {"standardize": True, "idiosyncratic_ar1": idiosyncratic_ar1},
        {"standardize": False, "idiosyncratic_ar1": idiosyncratic_ar1},
        {"standardize": True, "idiosyncratic_ar1": False},
        {"standardize": False, "idiosyncratic_ar1": False},
    ]

    last_exc: Exception | None = None
    diverged_fallback = None  # least-preferred: a diverged fit, used only if
    #                           every configuration diverges (kept so the caller
    #                           can still inspect/skip rather than crash).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cfg in fit_configs:
            try:
                model = DynamicFactorMQ(
                    endog,
                    k_endog_monthly=k_endog_M,
                    factors=k_factors,
                    factor_orders=factor_order,
                    idiosyncratic_ar1=cfg["idiosyncratic_ar1"],
                    standardize=cfg["standardize"],
                )
                res = model.fit(disp=False, maxiter=maxiter)
            except ValueError as exc:
                if "infs or NaNs" not in str(exc):
                    raise
                last_exc = exc
                continue
            except np.linalg.LinAlgError as exc:
                last_exc = exc
                continue

            if _dfm_fit_is_sane(res):
                return res
            if diverged_fallback is None:
                diverged_fallback = res  # remember the first diverged fit

    # Every configuration either raised or diverged. Prefer raising (so the
    # nowcast loop records a clean NaN and excludes the origin) over returning a
    # blown-up fit that would poison RMSFE with a ~1e17 nowcast.
    if last_exc is not None:
        raise last_exc
    if diverged_fallback is not None:
        raise RuntimeError(
            "DFM EM diverged for every fallback configuration "
            "(degenerate log-likelihood); origin skipped."
        )
    raise RuntimeError("DFM fit failed without a specific error.")


def extract_nowcast(
    result: Any,
    origin: pd.Period | str,
) -> float:
    """Extract the nowcast of the current quarter's GDP from a fitted DFM.

    The DFM is estimated through the quarter-end month. The GDP variable has
    NaN at the current quarter's month-3, so the Kalman smoother provides the
    best linear unbiased prediction for that missing observation.

    Parameters
    ----------
    result : fitted DynamicFactorMQResults (from fit_dfm).
    origin : monthly forecast origin.

    Returns
    -------
    Nowcast value in the same units as the GDP target (pp log-growth).
    """
    current_q = get_current_quarter(origin)
    target_month = quarter_end_timestamp(current_q)

    # predict() returns DataFrame aligned to endog index, in original units
    predicted = result.predict()

    # GDP is the last column of the endog
    gdp_col = predicted.columns[-1]

    if target_month not in predicted.index:
        raise KeyError(
            f"Quarter-end month {target_month.date()} not in predicted index "
            f"(index range: {predicted.index[0].date()} – {predicted.index[-1].date()})"
        )
    return float(predicted.loc[target_month, gdp_col])


# ---------------------------------------------------------------------------
# Single-origin nowcast (combines build → fit → extract)
# ---------------------------------------------------------------------------

def nowcast_for_origin(
    X_monthly: pd.DataFrame,
    y_quarterly: pd.Series,
    selected_cols: list[str],
    origin: pd.Period | str,
    k_factors: int = 2,
    factor_order: int = 2,
    idiosyncratic_ar1: bool = True,
    maxiter: int = 200,
    pub_lag_map: pd.Series | None = None,
    fill_method: Literal["ar_bic", "none"] = "ar_bic",
    ar_max_p: int = 4,
    ar_min_train: int = 24,
) -> dict:
    """Produce a DFM nowcast for a single forecast origin.

    Parameters
    ----------
    X_monthly     : full monthly predictor panel.
    y_quarterly   : quarterly GDP first-release series.
    selected_cols : list of selected indicator column names.
    origin        : monthly forecast origin.
    k_factors     : number of global factors (default 2; fixed across origins,
                    in line with ifoCAST / headline thesis specification).
    factor_order  : VAR order p for factor dynamics.
    idiosyncratic_ar1 : see fit_dfm.
    maxiter       : EM iterations.
    pub_lag_map   : optional pd.Series mapping id → publication lag (months).
        When provided, ``build_dfm_endog`` applies per-series ragged-edge
        masking followed by AR(p) BIC fill (controlled by ``fill_method``).
    fill_method   : 'ar_bic' (default) — fill ragged current-quarter months
        with univariate AR(p) BIC before passing to DFM; 'none' — leave NaN.
    ar_max_p      : maximum AR order for BIC selection.
    ar_min_train  : minimum training observations for AR(1+).

    Returns
    -------
    dict with keys: origin, current_quarter, k_factors, factor_order,
        n_indicators, nowcast.
    """
    origin = pd.Period(origin)
    origin_key = str(origin)

    if len(selected_cols) == 0:
        raise ValueError(f"No selected columns for origin {origin_key}.")

    X_sel = X_monthly[selected_cols]

    endog, k_endog_M = build_dfm_endog(
        X_sel, y_quarterly, origin,
        pub_lag_map=pub_lag_map,
        fill_method=fill_method,
        ar_max_p=ar_max_p,
        ar_min_train=ar_min_train,
    )

    result = fit_dfm(
        endog,
        k_endog_M=k_endog_M,
        k_factors=k_factors,
        factor_order=factor_order,
        idiosyncratic_ar1=idiosyncratic_ar1,
        maxiter=maxiter,
    )

    nowcast = extract_nowcast(result, origin)

    return {
        "origin": origin_key,
        "current_quarter": str(get_current_quarter(origin)),
        "k_factors": k_factors,
        "factor_order": factor_order,
        "n_indicators": len(selected_cols),
        "nowcast": nowcast,
    }


# ---------------------------------------------------------------------------
# Expanding nowcast loop — A-CD-TPN
# ---------------------------------------------------------------------------

def run_actpn_nowcast_loop(
    selection_matrix: pd.DataFrame,
    X_monthly: pd.DataFrame,
    y_quarterly: pd.Series,
    quarterly_origins: Iterable[pd.Period | str] | None = None,
    k_factors: int = 2,
    factor_order: int = 2,
    idiosyncratic_ar1: bool = True,
    maxiter: int = 200,
    verbose: bool = True,
    pub_lag_map: pd.Series | None = None,
    fill_method: Literal["ar_bic", "none"] = "ar_bic",
    ar_max_p: int = 4,
    ar_min_train: int = 24,
) -> pd.DataFrame:
    """Expanding A-CD-TPN nowcast loop: three nowcasts per quarter (M1, M2, M3).

    For each quarterly origin q and each month-in-quarter m ∈ {1, 2, 3}:
      1. Compute the monthly origin = first/second/third month of q.
      2. Load selected columns from selection_matrix at that monthly origin.
      3. Build mixed-frequency DFM endog:
           a. Real-time masking via pub_lag_map (ragged edge).
           b. AR(p) BIC fill of missing months in Q(q) (when fill_method='ar_bic').
      4. Fit DynamicFactorMQ with fixed ``k_factors`` (default 2).
      5. Extract nowcast for quarter q.

    M1 (first month of q): mostly survey data available.
    M2 (second month of q): surveys + some early hard data.
    M3 (third month of q): most hard data released.

    Parameters
    ----------
    selection_matrix  : binary DataFrame (monthly origins × series).
    X_monthly         : full monthly predictor panel.
    y_quarterly       : quarterly GDP first-release series.
    quarterly_origins : explicit quarterly PeriodIndex to evaluate. If None,
                        inferred as all quarters spanned by selection_matrix.
    factor_order      : VAR order p (default 2).
    idiosyncratic_ar1 : see fit_dfm (default True).
    k_factors         : number of global factors (default 2).
    maxiter           : EM iterations.
    verbose           : print progress.
    pub_lag_map       : pd.Series mapping id → publication lag (months).
        When provided, real-time masking and AR(p) fill are applied.
    fill_method       : 'ar_bic' (default) — fill ragged months before DFM;
        'none' — leave NaN (old behaviour).
    ar_max_p          : maximum AR order for BIC selection.
    ar_min_train      : minimum training observations for AR(1+).

    Returns
    -------
    pd.DataFrame with a MultiIndex (quarter, month_in_quarter) or flat index,
    with columns: quarter, monthly_origin, month_in_quarter, n_indicators,
    k_factors, nowcast, actual, error.
    """
    if quarterly_origins is None:
        all_m = pd.PeriodIndex(selection_matrix.index, freq="M")
        q_start = all_m[0].asfreq("Q")
        q_end = all_m[-1].asfreq("Q")
        quarterly_origins = pd.period_range(q_start, q_end, freq="Q")

    records = []
    for q in quarterly_origins:
        q = pd.Period(q, freq="Q")
        actual = float(y_quarterly.get(q, np.nan))

        for m_in_q in (1, 2, 3):
            # Monthly origin: m_in_q-th month of the quarter
            q_m1 = q.asfreq("M", how="start")
            origin_p = q_m1 + (m_in_q - 1)
            origin_key = str(origin_p)

            if origin_key not in selection_matrix.index:
                if verbose:
                    print(f"  {q} M{m_in_q} ({origin_key}): not in selection_matrix — skipped.")
                continue

            selected_cols = selection_matrix.columns[
                selection_matrix.loc[origin_key].astype(bool)
            ].tolist()

            if len(selected_cols) == 0:
                if verbose:
                    print(f"  {q} M{m_in_q} ({origin_key}): no indicators selected — skipped.")
                continue

            if verbose:
                print(
                    f"  {q} M{m_in_q} ({origin_key}): N={len(selected_cols)} ...",
                    end=" ", flush=True,
                )

            try:
                res = nowcast_for_origin(
                    X_monthly=X_monthly,
                    y_quarterly=y_quarterly,
                    selected_cols=selected_cols,
                    origin=origin_p,
                    k_factors=k_factors,
                    factor_order=factor_order,
                    idiosyncratic_ar1=idiosyncratic_ar1,
                    maxiter=maxiter,
                    pub_lag_map=pub_lag_map,
                    fill_method=fill_method,
                    ar_max_p=ar_max_p,
                    ar_min_train=ar_min_train,
                )
                error = res["nowcast"] - actual if not np.isnan(actual) else np.nan

                records.append({
                    "quarter":          str(q),
                    "monthly_origin":   origin_key,
                    "month_in_quarter": m_in_q,
                    "n_indicators":     res["n_indicators"],
                    "k_factors":        res["k_factors"],
                    "nowcast":          res["nowcast"],
                    "actual":           actual,
                    "error":            error,
                })
                if verbose:
                    print(
                        f"k={res['k_factors']}  nowcast={res['nowcast']:.3f}"
                        f"  actual={actual:.3f}"
                    )
            except Exception as exc:
                if verbose:
                    print(f"ERROR: {exc}")
                records.append({
                    "quarter":          str(q),
                    "monthly_origin":   origin_key,
                    "month_in_quarter": m_in_q,
                    "n_indicators":     len(selected_cols),
                    "k_factors":        np.nan,
                    "nowcast":          np.nan,
                    "actual":           actual,
                    "error":            np.nan,
                })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.set_index("monthly_origin")
    return df


# ---------------------------------------------------------------------------
# AR(1) baseline on GDP growth
# ---------------------------------------------------------------------------

def run_ar1_baseline(
    y_quarterly: pd.Series,
    quarterly_origins: Iterable[pd.Period | str],
    train_start_quarter: str = "1991Q1",
) -> pd.DataFrame:
    """Direct AR(1) nowcast: y_q = c + phi * y_{q-1} + eps, fit by OLS expanding window.

    Standard naive baseline in the nowcasting literature (Stock & Watson, 2002).
    """
    train_start = pd.Period(train_start_quarter, freq="Q")
    rows: list[dict] = []
    for q in quarterly_origins:
        q = pd.Period(q, freq="Q")
        train_idx = y_quarterly.index[
            (y_quarterly.index >= train_start) & (y_quarterly.index < q)
        ]
        y_train = y_quarterly.reindex(train_idx).dropna()
        if len(y_train) < 8:
            rows.append({"quarter": str(q), "nowcast": np.nan,
                         "actual": float(y_quarterly.get(q, np.nan)), "error": np.nan})
            continue
        y_t, y_lag = y_train.values[1:], y_train.values[:-1]
        x = np.column_stack([np.ones_like(y_lag), y_lag])
        beta, *_ = np.linalg.lstsq(x, y_t, rcond=None)
        last_obs = y_quarterly.loc[y_quarterly.index < q].dropna()
        y_prev = float(last_obs.iloc[-1]) if len(last_obs) else 0.0
        nowcast = float(beta[0] + beta[1] * y_prev)
        actual = float(y_quarterly.get(q, np.nan))
        rows.append({
            "quarter": str(q), "nowcast": nowcast, "actual": actual,
            "error": nowcast - actual if not np.isnan(actual) else np.nan,
        })
    return pd.DataFrame(rows).set_index("quarter")


# ---------------------------------------------------------------------------
# Random walk and AR(p)-BIC baselines
# ---------------------------------------------------------------------------

def run_rw_baseline(
    y_quarterly: pd.Series,
    quarterly_origins: Iterable[pd.Period | str],
) -> pd.DataFrame:
    """Naive random-walk nowcast: y_q^hat = y_{q-1}.

    Standard naive benchmark in the nowcasting literature (Stock & Watson
    2002a/b; Franjic & Schweikert 2025).
    """
    rows: list[dict] = []
    for q in quarterly_origins:
        q = pd.Period(q, freq="Q")
        last_obs = y_quarterly.loc[y_quarterly.index < q].dropna()
        nowcast = float(last_obs.iloc[-1]) if len(last_obs) else np.nan
        actual = float(y_quarterly.get(q, np.nan))
        rows.append({
            "quarter": str(q),
            "nowcast": nowcast,
            "actual": actual,
            "error": (nowcast - actual) if not (np.isnan(nowcast) or np.isnan(actual)) else np.nan,
        })
    return pd.DataFrame(rows).set_index("quarter")


# ---------------------------------------------------------------------------
# Diebold & Mariano (1995) test with Harvey-Leybourne-Newbold (1997) correction
# ---------------------------------------------------------------------------

def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int = 1,
    loss: str = "se",
) -> dict[str, float]:
    """Two-sided test of equal predictive accuracy.

    Tests H0: E[L(e_a) - L(e_b)] = 0. Negative DM statistic means model A is
    more accurate. Uses the Harvey-Leybourne-Newbold (1997) small-sample
    correction. ``loss`` in {'se', 'ae'} for squared / absolute error.

    References
    ----------
    Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy.
        Journal of Business and Economic Statistics, 13(3), 253-263.
    Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of
        prediction mean squared errors. International Journal of Forecasting,
        13(2), 281-291.
    """
    a = np.asarray(errors_a, dtype=float)
    b = np.asarray(errors_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = a.shape[0]
    if n < 8:
        return {"DM": np.nan, "p_value": np.nan, "n": n}

    if loss == "se":
        d = a ** 2 - b ** 2
    elif loss == "ae":
        d = np.abs(a) - np.abs(b)
    else:
        raise ValueError("loss must be 'se' or 'ae'")

    d_mean = float(np.mean(d))
    gamma_0 = float(np.var(d, ddof=0))
    var_d = gamma_0
    for k in range(1, h):
        cov = float(np.mean((d[k:] - d_mean) * (d[:-k] - d_mean)))
        var_d += 2.0 * cov
    if var_d <= 0:
        return {"DM": np.nan, "p_value": np.nan, "n": n}
    dm = d_mean / np.sqrt(var_d / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * hln
    from scipy import stats as _st
    p = 2 * (1 - _st.t.cdf(abs(dm_hln), df=n - 1))
    return {"DM": float(dm_hln), "p_value": float(p), "n": int(n)}


def mincer_zarnowitz_test(
    nowcast_df: pd.DataFrame,
    eval_start: str | None = None,
    eval_end: str | None = None,
    month_in_quarter: int | None = 3,
) -> dict[str, float]:
    """Mincer-Zarnowitz (1969) bias and efficiency test: y = alpha + beta * yhat + eps.

    Returns OLS coefficients, standard errors, and joint Wald p-value for
    H0: alpha = 0 and beta = 1 (forecast rationality under quadratic loss).
    """
    sub = _subset_eval_window(
        nowcast_df, eval_start, eval_end, month_in_quarter=month_in_quarter
    )
    sub = sub.dropna(subset=["nowcast", "actual"])
    if len(sub) < 8:
        return {
            "alpha": np.nan, "beta": np.nan,
            "se_alpha": np.nan, "se_beta": np.nan,
            "p_alpha_zero": np.nan, "p_joint_wald": np.nan, "n": len(sub),
        }

    y = sub["actual"].to_numpy(dtype=float)
    yhat = sub["nowcast"].to_numpy(dtype=float)
    n = len(y)
    x = np.column_stack([np.ones(n), yhat])
    beta_hat, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta_hat
    sigma2 = float(np.sum(resid ** 2) / max(n - 2, 1))
    xtx_inv = np.linalg.inv(x.T @ x)
    se = np.sqrt(np.clip(np.diag(sigma2 * xtx_inv), 0.0, None))

    from scipy import stats as _st
    t_alpha = beta_hat[0] / se[0] if se[0] > 0 else np.nan
    t_beta = beta_hat[1] / se[1] if se[1] > 0 else np.nan
    p_alpha = 2 * (1 - _st.t.cdf(abs(t_alpha), df=n - 2)) if not np.isnan(t_alpha) else np.nan

    # Joint Wald: R @ beta = r  with R = [[1,0],[0,1]], r = [0,1]
    r_vec = np.array([0.0, 1.0])
    diff = beta_hat - r_vec
    wald = float(diff.T @ np.linalg.inv(xtx_inv * sigma2) @ diff)
    p_joint = 1 - _st.f.cdf(wald / 2, dfn=2, dfd=n - 2)

    return {
        "alpha": float(beta_hat[0]),
        "beta": float(beta_hat[1]),
        "se_alpha": float(se[0]),
        "se_beta": float(se[1]),
        "p_alpha_zero": float(p_alpha),
        "p_joint_wald": float(p_joint),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Ahn & Horenstein (2013) eigenvalue ratio for number of factors (diagnostic)
# ---------------------------------------------------------------------------

def eigenvalue_ratio_er(X_scaled: np.ndarray, max_k: int = 8) -> int:
    """Ahn & Horenstein (2013) ER estimator for the number of factors.

    Robust alternative to Bai-Ng IC criteria; tends to pick fewer factors and
    avoids the cap-binding behaviour of IC2 in data-rich settings.

    Reference: Ahn, S. C. & Horenstein, A. R. (2013). Eigenvalue ratio test
    for the number of factors. Econometrica, 81(3), 1203-1227.
    """
    T, N = X_scaled.shape
    if T < 5 or N < 5:
        return 1
    max_k = min(max_k, T - 1, N - 1)
    s = np.linalg.svd(X_scaled, compute_uv=False)
    eig = (s ** 2) / (T * N)
    ratios = eig[:max_k] / np.clip(eig[1: max_k + 1], 1e-12, None)
    return int(np.argmax(ratios)) + 1



# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def _subset_eval_window(
    nowcast_df: pd.DataFrame,
    eval_start: str | None = None,
    eval_end: str | None = None,
    month_in_quarter: int | None = None,
) -> pd.DataFrame:
    """Restrict a nowcast DataFrame to the evaluation window (and optional M1/M2/M3)."""
    sub = nowcast_df.copy()
    qcol = "quarter" if "quarter" in sub.columns else None
    if qcol is not None:
        if eval_start is not None:
            sub = sub.loc[sub[qcol] >= eval_start]
        if eval_end is not None:
            sub = sub.loc[sub[qcol] <= eval_end]
    else:
        if eval_start is not None:
            sub = sub.loc[sub.index >= eval_start]
        if eval_end is not None:
            sub = sub.loc[sub.index <= eval_end]
    if month_in_quarter is not None and "month_in_quarter" in sub.columns:
        sub = sub.loc[sub["month_in_quarter"] == month_in_quarter]
    return sub


def compute_rmsfe(
    nowcast_df: pd.DataFrame,
    eval_start: str | None = None,
    eval_end: str | None = None,
    month_in_quarter: int | None = None,
) -> float:
    """Compute RMSFE over the evaluation window, excluding NaN pairs.

    Parameters
    ----------
    nowcast_df : DataFrame with 'error' column (nowcast - actual).
    eval_start : start quarter string, e.g. '2011Q1'. Uses all if None.
    eval_end   : end quarter string. Uses all if None.
    month_in_quarter : if set (1, 2, or 3), restrict to that monthly origin
        within each target quarter. Use ``3`` for the standard headline nowcast
        (end-of-quarter information set). If ``None``, all monthly origins in
        the DataFrame are pooled (180 errors when M1--M3 are present).
    """
    sub = _subset_eval_window(
        nowcast_df, eval_start, eval_end, month_in_quarter
    )
    errors = sub["error"].dropna()
    if errors.empty:
        return np.nan
    return float(np.sqrt(np.mean(errors ** 2)))


def compute_rmsfe_by_month_in_quarter(
    nowcast_df: pd.DataFrame,
    eval_start: str | None = None,
    eval_end: str | None = None,
) -> pd.Series:
    """RMSFE at M1, M2, and M3 separately (requires ``month_in_quarter`` column)."""
    if "month_in_quarter" not in nowcast_df.columns:
        return pd.Series(dtype=float)
    out = {}
    for m in (1, 2, 3):
        out[m] = compute_rmsfe(
            nowcast_df,
            eval_start=eval_start,
            eval_end=eval_end,
            month_in_quarter=m,
        )
    return pd.Series(out, name="RMSFE")


def expand_quarterly_nowcasts_to_monthly(
    nowcast_df: pd.DataFrame,
) -> pd.DataFrame:
    """Replicate quarterly nowcasts at M1, M2, and M3 for alignment with DFM output.

    Random-walk and AR(1) baselines depend only on completed quarterly GDP; the
    point nowcast for target quarter *q* is the same at each month-in-quarter
    within *q*. Expansion duplicates rows so Diebold--Mariano tests and CSV
    schemas match ``run_actpn_nowcast_loop`` without changing RMSFE at M3.
    """
    if "month_in_quarter" in nowcast_df.columns:
        return nowcast_df.copy()

    records: list[dict] = []
    for idx, row in nowcast_df.iterrows():
        q_str = str(row["quarter"]) if "quarter" in nowcast_df.columns else str(idx)
        q = pd.Period(q_str, freq="Q")
        q_m1 = q.asfreq("M", how="start")
        for m_in_q in (1, 2, 3):
            origin_p = q_m1 + (m_in_q - 1)
            rec = {
                "quarter": str(q),
                "monthly_origin": str(origin_p),
                "month_in_quarter": m_in_q,
                "nowcast": row.get("nowcast", np.nan),
                "actual": row.get("actual", np.nan),
                "error": row.get("error", np.nan),
            }
            for col in ("n_indicators", "k_factors"):
                if col in row.index:
                    rec[col] = row[col]
            records.append(rec)
    out = pd.DataFrame(records)
    if out.empty:
        return out
    return out.set_index("monthly_origin")


def _ensure_quarter_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a string ``quarter`` column (never rely on duplicate index)."""
    out = df.copy()
    if "quarter" not in out.columns:
        out = out.reset_index()
        if "quarter" not in out.columns and out.index.name:
            out = out.rename(columns={out.index.name: "quarter"})
    out["quarter"] = out["quarter"].astype(str)
    return out


def align_forecast_errors(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    month_in_quarter: int | None = 3,
    eval_start: str | None = None,
    eval_end: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired forecast errors on a common quarter grid for DM tests.

    Always merges on the string quarter key (one row per target quarter at the
    requested ``month_in_quarter``). This avoids accidental row-order alignment
    when DFM DataFrames carry duplicate quarter index labels (M1/M2/M3).
    """
    a = _subset_eval_window(
        df_a, eval_start, eval_end, month_in_quarter=month_in_quarter
    )
    b = _subset_eval_window(
        df_b, eval_start, eval_end, month_in_quarter=month_in_quarter
    )
    a = _ensure_quarter_column(a)
    b = _ensure_quarter_column(b)

    merged = a[["quarter", "error"]].merge(
        b[["quarter", "error"]],
        on="quarter",
        suffixes=("_a", "_b"),
        how="inner",
    )
    ea = merged["error_a"].to_numpy(dtype=float)
    eb = merged["error_b"].to_numpy(dtype=float)
    mask = ~(np.isnan(ea) | np.isnan(eb))
    return ea[mask], eb[mask]


def build_forecast_loss_matrix(
    models: dict[str, pd.DataFrame],
    eval_start: str | None = None,
    eval_end: str | None = None,
    month_in_quarter: int | None = 3,
    loss: Literal["se", "ae"] = "se",
) -> pd.DataFrame:
    """Return common-date forecast losses with quarters as rows and models as columns."""
    if loss not in {"se", "ae"}:
        raise ValueError("loss must be 'se' or 'ae'")

    errors: dict[str, pd.Series] = {}
    for name, df in models.items():
        sub = _subset_eval_window(
            df, eval_start, eval_end, month_in_quarter=month_in_quarter
        )
        sub = _ensure_quarter_column(sub)[["quarter", "error"]].dropna()
        if sub["quarter"].duplicated().any():
            raise ValueError(f"{name} has duplicate observations for a quarter")
        errors[name] = sub.set_index("quarter")["error"].astype(float)

    if len(errors) < 2:
        raise ValueError("at least two models are required")

    aligned = pd.concat(errors, axis=1, join="inner").dropna().sort_index()
    if aligned.empty:
        raise ValueError("models have no common forecast-error observations")
    return aligned.pow(2) if loss == "se" else aligned.abs()


def compute_model_confidence_set(
    losses: pd.DataFrame,
    size: float = 0.10,
    reps: int = 10_000,
    block_size: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute the Hansen--Lunde--Nason model confidence set.

    Uses the range statistic and a stationary block bootstrap. Models with an
    elimination p-value greater than ``size`` remain in the confidence set.

    References
    ----------
    Hansen, P. R., Lunde, A. & Nason, J. M. (2011). The Model Confidence Set.
        Econometrica, 79(2), 453--497.
    """
    if losses.shape[0] < 8 or losses.shape[1] < 2:
        raise ValueError("losses must contain at least 8 rows and 2 models")
    if not 0 < size < 1:
        raise ValueError("size must lie strictly between 0 and 1")

    from arch.bootstrap import MCS  # pyright: ignore[reportMissingImports]

    mcs = MCS(
        losses,
        size=size,
        reps=reps,
        block_size=block_size,
        method="R",
        bootstrap="stationary",
        seed=seed,
    )
    mcs.compute()

    pvalues = mcs.pvalues["Pvalue"].reindex(losses.columns)
    result = pd.DataFrame({
        "mean_loss": losses.mean(),
        "MCS_p_value": pvalues,
        "in_MCS": losses.columns.isin(mcs.included),
    })
    result.index.name = "model"
    return result.sort_values("mean_loss")


def compute_nsr(
    nowcast_df: pd.DataFrame,
    y_quarterly: pd.Series,
    eval_start: str | None = None,
    eval_end: str | None = None,
) -> float:
    """Compute the Noise-to-Signal Ratio (NSR) as defined in Lehmann et al. (2020).

    NSR = RMSFE / std(y_GDP)

    A model with NSR < 1 produces forecast errors that are smaller than the
    natural variability of the target series — the standard threshold for
    "practical relevance" in the nowcasting literature (ifoCAST paper, p. 8).
    The standard deviation is computed over the same evaluation window as the
    RMSFE, using the actual GDP values observed in that window.

    Parameters
    ----------
    nowcast_df  : DataFrame with 'error' and 'actual' columns.
    y_quarterly : quarterly GDP series (PeriodIndex). Used to compute the
                  target variability over the evaluation window.
    eval_start  : start quarter (inclusive), e.g. '2011Q1'.
    eval_end    : end quarter (inclusive), e.g. '2025Q4'.

    Returns
    -------
    NSR (float); NaN if RMSFE or target std cannot be computed.

    References
    ----------
    Lehmann, R., Reif, M. & Wollmershäuser, T. (2020). ifoCAST — Das neue
        Kurzfristprognosemodell des ifo Instituts. ifo Schnelldienst, 73(11).
    """
    # Headline NSR uses M3 when monthly origins are present (one actual per quarter).
    miq = 3 if "month_in_quarter" in nowcast_df.columns else None
    rmsfe = compute_rmsfe(
        nowcast_df,
        eval_start=eval_start,
        eval_end=eval_end,
        month_in_quarter=miq,
    )
    if np.isnan(rmsfe):
        return np.nan

    sub = _subset_eval_window(
        nowcast_df, eval_start, eval_end, month_in_quarter=miq
    )
    actuals = sub["actual"].dropna()

    if len(actuals) < 2:
        return np.nan

    target_std = float(actuals.std(ddof=1))
    if target_std <= 0:
        return np.nan

    return rmsfe / target_std


def compute_crps_gaussian(
    nowcast_df: pd.DataFrame,
    eval_start: str | None = None,
    eval_end: str | None = None,
) -> float:
    """Compute mean CRPS for a Gaussian predictive distribution (SV models).

    The Continuous Ranked Probability Score (CRPS) is the proper scoring rule
    for evaluating predictive distributions. For a Gaussian predictive
    N(μ, σ²) it has the closed-form expression of Gneiting and Raftery
    (2007, p. 367):

        CRPS(F, y) = σ · [z (2Φ(z) − 1) + 2φ(z) − 1/√π],
        z = (y − μ) / σ,

    where φ and Φ are the standard normal PDF and CDF. The score is
    nonnegative and lower is better; it reduces to absolute error as
    σ → 0, so the mean CRPS of a degenerate point forecast equals MAE.

    The predictive parameters are reconstructed from the stored columns:
      μ = nowcast (integrated SV can shift the point slightly from EM)
      σ = sigma_em · √(rel_vol)  (SV-scaled Kalman predictive SD)

    Parameters
    ----------
    nowcast_df : DataFrame with columns 'nowcast', 'actual', 'sigma_em',
                 'rel_vol'. Produced by ``run_actpn_nowcast_loop_sv``.
    eval_start : start quarter (inclusive).
    eval_end   : end quarter (inclusive).

    Returns
    -------
    Mean CRPS (float); NaN if required columns are absent.

    References
    ----------
    Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules,
        prediction, and estimation. Journal of the American Statistical
        Association, 102(477), 359–378.
    """
    from scipy import stats as _st

    required = {"nowcast", "actual", "sigma_em", "rel_vol"}
    if not required.issubset(nowcast_df.columns):
        return np.nan

    sub = nowcast_df.copy()
    if eval_start is not None:
        sub = sub.loc[sub.index >= eval_start]
    if eval_end is not None:
        sub = sub.loc[sub.index <= eval_end]
    sub = sub.dropna(subset=list(required))

    if sub.empty:
        return np.nan

    mu    = sub["nowcast"].values
    y     = sub["actual"].values
    sigma = sub["sigma_em"].values * np.sqrt(
        np.clip(sub["rel_vol"].values, 1e-8, None)
    )
    sigma = np.clip(sigma, 1e-10, None)

    z    = (y - mu) / sigma
    crps = sigma * (
        z * (2.0 * _st.norm.cdf(z) - 1.0)
        + 2.0 * _st.norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    return float(np.mean(crps))


def build_rmsfe_table(
    results: dict[str, pd.DataFrame],
    reference_key: str | None = "DFM-EN",
    eval_start: str | None = None,
    eval_end: str | None = None,
    y_quarterly: pd.Series | None = None,
    month_in_quarter: int | None = 3,
) -> pd.DataFrame:
    """Build RMSFE summary table with relative ratios and Noise-to-Signal Ratio.

    Parameters
    ----------
    results       : dict mapping model name → nowcast DataFrame.
    reference_key : key in ``results`` used for RMSFE-relative ratios (optional).
    eval_start  : evaluation window start quarter, e.g. '2011Q1'.
    eval_end    : evaluation window end quarter, e.g. '2025Q4'.
    y_quarterly : quarterly GDP series. When provided, an NSR column is added
                  (NSR = RMSFE / std(GDP actuals)). NSR < 1 indicates practical
                  relevance (Lehmann, Reif & Wollmershäuser 2020, ifoCAST p. 8).
    month_in_quarter : headline RMSFE month (default 3 = M3). Pass ``None`` to
        pool all monthly origins.

    Returns
    -------
    pd.DataFrame with columns: RMSFE, RMSFE_relative [, NSR].
    """
    rows = []
    for name, df in results.items():
        rmsfe = compute_rmsfe(
            df,
            eval_start=eval_start,
            eval_end=eval_end,
            month_in_quarter=month_in_quarter,
        )
        nsr   = (
            compute_nsr(df, y_quarterly, eval_start=eval_start, eval_end=eval_end)
            if y_quarterly is not None
            else np.nan
        )
        rows.append({"model": name, "RMSFE": rmsfe, "NSR": nsr})

    table = pd.DataFrame(rows).set_index("model")

    if (
        reference_key is not None
        and reference_key in table.index
        and not np.isnan(table.at[reference_key, "RMSFE"])
    ):
        ref_rmsfe = table.at[reference_key, "RMSFE"]
        table["RMSFE_relative"] = table["RMSFE"] / ref_rmsfe
    else:
        table["RMSFE_relative"] = np.nan

    # Reorder columns: RMSFE, RMSFE_relative, NSR
    cols = ["RMSFE", "RMSFE_relative"]
    if y_quarterly is not None:
        cols.append("NSR")
    table = table[cols]

    return table.round(4)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def build_interval_calibration_table(
    sv_results: dict[str, pd.DataFrame],
    eval_start: str | None = None,
    eval_end: str | None = None,
    credibility: float = 0.9,
) -> pd.DataFrame:
    """Evaluate empirical coverage of SV prediction intervals.

    For each model that produced CI columns (ci_lower_XX / ci_upper_XX),
    computes:
      - empirical coverage  (fraction of actuals inside the PI)
      - mean interval width
      - RMSFE of the point nowcast

    Full-sample coverage near the nominal level can average severe
    under-coverage in a shock and over-coverage afterwards; report the
    three evaluation windows alongside the pooled figure.

    Parameters
    ----------
    sv_results  : dict mapping model name → DataFrame produced by
                  run_actpn_nowcast_loop_sv (must have ci_lower_XX,
                  ci_upper_XX, actual, and error columns).
    eval_start  : start quarter (inclusive), e.g. '2011Q1'.
    eval_end    : end quarter (inclusive), e.g. '2025Q4'.
    credibility : nominal coverage level (default 0.9).

    Returns
    -------
    pd.DataFrame with columns:
        model, RMSFE, coverage_empirical, coverage_nominal, mean_width.
    """
    cov_label = int(round(credibility * 100))
    lo_col = f"ci_lower_{cov_label}"
    hi_col = f"ci_upper_{cov_label}"

    rows = []
    for name, df in sv_results.items():
        sub = _subset_eval_window(
            df,
            eval_start=eval_start,
            eval_end=eval_end,
            month_in_quarter=3,
        )
        sub = sub.dropna(subset=["actual", "nowcast"])

        rmsfe = compute_rmsfe(
            sub,
            eval_start=None,
            eval_end=None,
            month_in_quarter=None,
        ) if "error" in sub and sub["error"].notna().any() else np.nan

        if lo_col in sub.columns and hi_col in sub.columns:
            valid = sub.dropna(subset=[lo_col, hi_col, "actual"])
            inside = (
                (valid["actual"] >= valid[lo_col])
                & (valid["actual"] <= valid[hi_col])
            )
            coverage  = float(inside.mean()) if len(valid) > 0 else np.nan
            mean_width = float((valid[hi_col] - valid[lo_col]).mean()) \
                if len(valid) > 0 else np.nan
        else:
            coverage   = np.nan
            mean_width = np.nan

        crps = compute_crps_gaussian(
            sub, eval_start=None, eval_end=None,
        )

        rows.append({
            "model":              name,
            "RMSFE":              round(rmsfe, 4),
            "coverage_empirical": round(coverage, 3) if not np.isnan(coverage) else np.nan,
            "coverage_nominal":   credibility,
            "mean_width":         round(mean_width, 4) if not np.isnan(mean_width) else np.nan,
            "CRPS":               round(crps, 4) if not np.isnan(crps) else np.nan,
        })

    return pd.DataFrame(rows).set_index("model")


def save_nowcast_outputs(
    output_dir: str | Path,
    nowcast_results: dict[str, pd.DataFrame],
    rmsfe_table: pd.DataFrame,
) -> None:
    """Persist nowcast results and RMSFE table to CSV files.

    Parameters
    ----------
    output_dir      : directory to write files (created if absent).
    nowcast_results : dict of model-name → DataFrame with columns
                      quarter, n_indicators, k_factors, nowcast, actual, error.
    rmsfe_table     : output of build_rmsfe_table.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, df in nowcast_results.items():
        safe_name = model_name.lower().replace("-", "_").replace(" ", "_")
        df.to_csv(output_dir / f"nowcast_results_{safe_name}.csv")

    rmsfe_table.to_csv(output_dir / "rmsfe_table.csv")
    print(f"Nowcast outputs saved to {output_dir}")
