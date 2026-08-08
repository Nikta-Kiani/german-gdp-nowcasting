"""Ragged-edge panel utilities for real-time DFM nowcasting.

Two-stage pipeline (Part II only):
  Stage 1 — Real-time masking: apply publication lags to set unreleased
            observations to NaN, creating the ragged-edge panel.
  Stage 2 — AR(p) BIC fill: univariate AR(p) with BIC order selection per
            series, estimated only on observed history, to complete missing
            months through the end of the current quarter Q(t).

The DFM then receives a monthly indicator panel with no NaN *within* the
estimation window for each indicator (except the deliberate GDP target NaN).

No future information is used at any stage: the AR is estimated only on
observations through last_observed_month(j, t) = t - pub_lag_j.

References
----------
Bańbura, M., Giannone, D., Modugno, M. & Reichlin, L. (2013). Now-casting
    and the real-time data flow. In G. Elliott & A. Timmermann (Eds.),
    Handbook of Economic Forecasting, vol. 2A, 195–237. Elsevier.
Bańbura, M. & Rünstler, G. (2011). A look into the factor model black box:
    Publication lags and the role of hard and soft data in forecasting GDP.
    International Journal of Forecasting, 27(2), 333–346.
Giannone, D., Reichlin, L. & Small, D. (2008). Nowcasting: The real-time
    informational content of macroeconomic data. Journal of Monetary
    Economics, 55(4), 665–676.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Stage 1 — Real-time publication-lag mask
# ---------------------------------------------------------------------------

def last_observed_month(
    origin: pd.Period,
    pub_lag: int,
) -> pd.Period:
    """Return the most recent reference month that has been released at origin.

    Semantics (from core_utils.load_pub_lag_map docstring):
      - Value for reference month T with pub_lag=P is available at the *start*
        of month T + P + 1.
      - At monthly origin t, series j may use observations through month
        t - pub_lag_j.

    Parameters
    ----------
    origin  : monthly forecast origin (pd.Period, freq='M').
    pub_lag : publication lag in whole months (≥ 0).

    Returns
    -------
    pd.Period (monthly) — most recent reference month available at origin.

    Examples
    --------
    >>> last_observed_month(pd.Period('2011-03', 'M'), 0)
    Period('2011-03', 'M')   # same month released: lag-0 series
    >>> last_observed_month(pd.Period('2011-03', 'M'), 2)
    Period('2011-01', 'M')   # two months behind: hard data
    """
    return origin - pub_lag


def apply_pub_lag_mask(
    X: pd.DataFrame,
    origin: pd.Period | str,
    pub_lag_map: pd.Series,
) -> pd.DataFrame:
    """Apply per-series publication-lag masking to create a real-time panel.

    For series j at monthly origin t, sets all observations strictly after
    last_observed_month(t, lag_j) to NaN.  The resulting panel represents
    exactly the information set available to a forecaster at origin t.

    Parameters
    ----------
    X           : monthly predictor panel (DatetimeIndex × series).
    origin      : monthly forecast origin.
    pub_lag_map : pd.Series mapping series id → publication lag in months.
                  Series not in pub_lag_map are assumed lag=0.

    Returns
    -------
    X_masked : copy of X with future (not-yet-released) cells set to NaN.
    """
    origin_p = pd.Period(origin, freq="M")
    X_masked = X.copy()

    for col in X_masked.columns:
        lag = int(pub_lag_map.get(col, 0))
        last_obs_p = last_observed_month(origin_p, lag)
        last_obs_ts = last_obs_p.to_timestamp()
        future_mask = X_masked.index > last_obs_ts
        if future_mask.any():
            X_masked.loc[future_mask, col] = np.nan

    return X_masked


# ---------------------------------------------------------------------------
# Stage 2 — Univariate AR(p) BIC fill
# ---------------------------------------------------------------------------

def _fit_ar_bic(
    series: np.ndarray,
    max_p: int = 4,
    min_train: int = 24,
) -> tuple[int, np.ndarray]:
    """Fit AR(p) selected by BIC to a 1-D series; return (p_opt, coefficients).

    The constant is always included.  Coefficients are ordered [const, ar_1,
    ..., ar_p].  Falls back to AR(1) when insufficient history; uses AR(0)
    (random walk with drift = sample mean) as the ultimate fallback.

    Parameters
    ----------
    series    : 1-D float array, may contain leading NaN but must end with
                at least min_train non-NaN observations.
    max_p     : maximum AR order to consider (BIC grid: 0 … max_p).
    min_train : minimum number of usable observations required to fit AR(1+).

    Returns
    -------
    p_opt   : selected AR order.
    coefs   : coefficient array [const, ar_1, …, ar_p].
    """
    valid = series[~np.isnan(series)]
    n = len(valid)

    if n < max(min_train, max_p + 2):
        p_cap = max(0, min(max_p, n - 2))
    else:
        p_cap = max_p

    if n < 2:
        # No usable history: return AR(0) with mean=0
        return 0, np.array([0.0])

    best_p = 0
    best_bic = np.inf
    best_coefs = np.array([float(np.mean(valid))])

    for p in range(0, p_cap + 1):
        if n <= p + 1:
            break
        n_eff = n - p
        y = valid[p:]
        if p == 0:
            X_reg = np.ones((n_eff, 1))
        else:
            X_reg = np.column_stack(
                [np.ones(n_eff)] + [valid[p - j - 1: n - j - 1] for j in range(p)]
            )
        try:
            coefs, residuals, _, _ = np.linalg.lstsq(X_reg, y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        if residuals.size == 0:
            fitted = X_reg @ coefs
            ss_res = float(np.sum((y - fitted) ** 2))
        else:
            ss_res = float(residuals[0])

        sigma2 = ss_res / n_eff if n_eff > 0 else np.inf
        if sigma2 <= 0:
            # Perfect fit → accept this order
            best_p, best_bic, best_coefs = p, -np.inf, coefs
            break

        k = p + 1  # number of parameters (including constant)
        bic = n_eff * np.log(sigma2) + k * np.log(n_eff)
        if bic < best_bic:
            best_bic = bic
            best_p = p
            best_coefs = coefs

    return best_p, best_coefs


def _ar_forecast(
    history: np.ndarray,
    steps_ahead: int,
    p: int,
    coefs: np.ndarray,
) -> np.ndarray:
    """Multi-step AR(p) recursive forecast.

    Parameters
    ----------
    history     : observed series (non-NaN), most recent at the end.
    steps_ahead : number of future values to forecast.
    p           : AR order.
    coefs       : [const, ar_1, …, ar_p].

    Returns
    -------
    forecasts : 1-D array of length steps_ahead.
    """
    const = coefs[0]
    ar_coefs = coefs[1:] if p > 0 else np.array([])

    buf = list(history[-max(p, 1):]) if len(history) > 0 else [0.0]
    preds = []
    for _ in range(steps_ahead):
        if p == 0:
            yhat = const
        else:
            lag_vals = np.array(buf[-p:])
            yhat = const + float(ar_coefs @ lag_vals[::-1])
        preds.append(yhat)
        buf.append(yhat)

    return np.array(preds)


def fill_ragged_edge_ar(
    X_masked: pd.DataFrame,
    origin: pd.Period | str,
    pub_lag_map: pd.Series,
    max_p: int = 4,
    min_train: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill missing months in the current quarter using univariate AR(p) BIC.

    For each series j at monthly origin t:
      1. Training sample: all non-NaN months in X_masked up to
         last_observed_month(j, t)  — real-time only, no leakage.
      2. BIC selects order p ∈ {0, …, min(max_p, T_eff−2)}.
      3. Fill: every month s in the DFM window that is NaN AND
         s ≤ quarter-end of Q(t) receives the AR multi-step forecast.
         Filling starts from the first missing month after last release.
      4. No series is filled *beyond* the quarter-end of Q(t).

    Series with too little history are left as NaN (they will be dropped
    by _drop_degenerate_endog_columns in nowcast_utils).

    Parameters
    ----------
    X_masked  : real-time masked panel (output of apply_pub_lag_mask),
                monthly DatetimeIndex × N series.
    origin    : monthly forecast origin.
    pub_lag_map: pd.Series mapping series id → publication lag in months.
    max_p     : maximum AR order for BIC selection (default 4).
    min_train : minimum number of non-NaN observations required before AR
                order ≥ 1 is allowed (default 24 months ≈ 2 years).

    Returns
    -------
    X_filled  : copy of X_masked with NaN cells in Q(t) replaced by AR
                forecasts where feasible.
    fill_flags: pd.DataFrame of same shape, dtype='object', with values
                'observed' | 'filled' | 'missing' for diagnostics.
    """
    origin_p = pd.Period(origin, freq="M")
    current_q = origin_p.asfreq("Q")
    q_end_month_p = current_q.asfreq("M", how="end")
    q_end_ts = q_end_month_p.to_timestamp()

    X_filled = X_masked.copy()
    fill_flags = pd.DataFrame("observed", index=X_masked.index, columns=X_masked.columns)
    fill_flags[X_masked.isna()] = "missing"

    for col in X_filled.columns:
        lag = int(pub_lag_map.get(col, 0))
        last_obs_p = last_observed_month(origin_p, lag)
        last_obs_ts = last_obs_p.to_timestamp()

        # Identify months to fill: NaN, within the current quarter, after last release
        fill_mask = (
            X_filled[col].isna()
            & (X_filled.index > last_obs_ts)
            & (X_filled.index <= q_end_ts)
        )
        if not fill_mask.any():
            continue

        # Training data: all non-NaN values up to last observed month (real-time only)
        train_mask = X_filled.index <= last_obs_ts
        train_vals = X_filled.loc[train_mask, col].dropna().values

        if len(train_vals) < 2:
            # Insufficient history — leave NaN; will be dropped downstream
            continue

        p_opt, coefs = _fit_ar_bic(train_vals, max_p=max_p, min_train=min_train)

        # Recursive forecasts advance one release month at a time through Q(t).
        fill_dates = X_filled.index[fill_mask]

        for i, fill_date in enumerate(fill_dates):
            steps = i + 1  # consecutive months from last release
            preds = _ar_forecast(train_vals, steps_ahead=steps, p=p_opt, coefs=coefs)
            X_filled.at[fill_date, col] = preds[-1]
            fill_flags.at[fill_date, col] = "filled"

    return X_filled, fill_flags


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def get_information_set_summary(
    X_raw: pd.DataFrame,
    origin: pd.Period | str,
    pub_lag_map: pd.Series,
    max_p: int = 4,
    min_train: int = 24,
) -> pd.DataFrame:
    """Summarise the real-time information set for a given forecast origin.

    Produces a per-series DataFrame that shows:
      - publication lag
      - last observed month (real-time)
      - whether the current-quarter months are observed / filled / missing
      - number of months filled by AR in the current quarter

    Useful for validating that M1 has fewer observed series than M3, and that
    filled months > 0 for lagged series.

    Parameters
    ----------
    X_raw       : full monthly predictor panel (no masking applied yet).
    origin      : monthly forecast origin.
    pub_lag_map : pd.Series mapping series id → publication lag.
    max_p       : passed to fill_ragged_edge_ar.
    min_train   : passed to fill_ragged_edge_ar.

    Returns
    -------
    pd.DataFrame indexed by series id with columns:
        pub_lag, last_obs_month, n_obs_q, n_filled_q, n_missing_q, status.
    """
    origin_p = pd.Period(origin, freq="M")
    current_q = origin_p.asfreq("Q")
    q_start_ts = current_q.asfreq("M", how="start").to_timestamp()
    q_end_ts = current_q.asfreq("M", how="end").to_timestamp()

    # Select the subpanel that spans the current quarter
    q_months = X_raw.index[(X_raw.index >= q_start_ts) & (X_raw.index <= q_end_ts)]

    X_masked = apply_pub_lag_mask(X_raw, origin_p, pub_lag_map)
    _, fill_flags = fill_ragged_edge_ar(
        X_masked, origin_p, pub_lag_map, max_p=max_p, min_train=min_train
    )

    rows = []
    for col in X_raw.columns:
        lag = int(pub_lag_map.get(col, 0))
        last_obs_p = last_observed_month(origin_p, lag)

        flags_q = fill_flags.loc[q_months, col] if len(q_months) > 0 else pd.Series(dtype=str)
        n_obs = int((flags_q == "observed").sum())
        n_filled = int((flags_q == "filled").sum())
        n_missing = int((flags_q == "missing").sum())

        if n_missing > 0:
            status = "partially_missing"
        elif n_filled > 0:
            status = "ar_filled"
        else:
            status = "fully_observed"

        rows.append({
            "series": col,
            "pub_lag": lag,
            "last_obs_month": str(last_obs_p),
            "n_obs_q": n_obs,
            "n_filled_q": n_filled,
            "n_missing_q": n_missing,
            "status": status,
        })

    return pd.DataFrame(rows).set_index("series")
