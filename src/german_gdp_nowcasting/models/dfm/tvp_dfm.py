"""Two-step Time-Varying-Parameter DFM (TVP-DFM) for German GDP nowcasting.

Motivation
----------
A fixed-loading DFM estimates the factor → GDP transmission *once* and applies
it forever. After a structural break (e.g. the post-2022 energy-shock /
stagnation regime) the historical ifo → GDP relationship can weaken
permanently, and a fixed-loading model stays anchored to the pre-break
estimate. Del Negro & Otrok (2008) address this by letting the loadings drift
as a random walk, so the mapping adapts to the new regime.

Why a *two-step* TVP-DFM (and not the full Gibbs sampler)
---------------------------------------------------------
The full Del Negro–Otrok model is a Bayesian Gibbs sampler that jointly draws
the latent factors, the time-varying loadings of *every* series, and the
stochastic volatilities. With mixed frequency (Mariano–Murasawa) and a ragged
edge on top, that is heavy and fragile. This module instead keeps the project's
existing **two-step** philosophy (Doz, Giannone & Reichlin 2011), which is
exactly the estimator used everywhere else in the pipeline:

    Stage 1 — identical EM-DFM front-end (``nowcast_utils.fit_dfm``) extracts the
              smoothed monthly factors F_t. Bates, Plagborg-Møller, Stock &
              Watson (2013) show that two-step / PCA factor estimates remain
              *consistent even when the loadings drift*, which is precisely what
              licenses estimating the factors with a fixed-loading step and
              allowing the second stage to be time-varying.

    Stage 2 — a TVP **bridge equation** maps the factor(s) to GDP with
              random-walk loadings, estimated by a Gaussian Kalman
              filter/smoother:

                  y_q = a_q + λ_q' f_q + ε_q,     ε_q ~ N(0, σ²)
                  [a_q, λ_q]' = [a_{q-1}, λ_{q-1}]' + ω_q,  ω_q ~ N(0, σ² Q*)

              where f_q is the current-quarter factor (3-month average of the
              smoothed monthly factor) and Q* = q_ratio · I is a diagonal
              signal-to-noise matrix. q_ratio controls how fast the loadings are
              allowed to drift and is chosen by profile (concentrated) maximum
              likelihood at every origin; q_ratio → 0 recovers the fixed-loading
              OLS bridge. The COVID quarters (2020Q1–2021Q4) receive an inflated
              observation variance (Lenza & Primiceri 2022) so the loadings are
              not dragged by — and do not overshoot after — the pandemic swings;
              this is what unlocks the post-2022 accuracy gain.

This targets exactly the post-2022 concern (a drifting factor → GDP
transmission), runs in milliseconds per origin (no MCMC), and re-uses the
*identical* Stage-1 information set (real-time publication-lag masking + AR(p)
BIC fill via ``build_dfm_endog``) so it is directly comparable to DFM-EN,
DFM-ifoCAST, the SV variants, XGB and MLP.

The single-origin function and the expanding loop mirror
``nowcast_utils.run_actpn_nowcast_loop`` /
``dfm_sv_bayes.run_actpn_nowcast_loop_sv`` so the output CSV is consumable by
the same evaluation helpers (``compute_rmsfe``, ``diebold_mariano_test`` …).

References
----------
Del Negro, M. & Otrok, C. (2008). Dynamic factor models with time-varying
    parameters: Measuring changes in international business cycles. FRBNY Staff
    Report No. 326.
Bates, B. J., Plagborg-Møller, M., Stock, J. H. & Watson, M. W. (2013).
    Consistent factor estimation in dynamic factor models with structural
    instability. Journal of Econometrics, 177(2), 289–304.
Doz, C., Giannone, D. & Reichlin, L. (2011). A two-step estimator for large
    approximate dynamic factor models based on Kalman filtering. Journal of
    Econometrics, 164(1), 188–205.
Lenza, M. & Primiceri, G. E. (2022). How to estimate a VAR after March 2020.
    Journal of Applied Econometrics, 37(4), 688–699.
Stock, J. H. & Watson, M. W. (2002). Macroeconomic forecasting using diffusion
    indexes. Journal of Business & Economic Statistics, 20(2), 147–162.
Primiceri, G. E. (2005). Time varying structural VARs and monetary policy.
    Review of Economic Studies, 72(3), 821–852.
Cogley, T. & Sargent, T. J. (2005). Drifts and volatilities. Review of Economic
    Dynamics, 8(2), 262–302.
Eickmeier, S., Lemke, W. & Marcellino, M. (2015). Classical time-varying
    factor-augmented vector autoregressive models. Journal of Applied
    Econometrics, 30(3), 493–516.
Durbin, J. & Koopman, S. J. (2012). Time Series Analysis by State Space
    Methods, 2nd ed. OUP. (Kalman filter, diffuse init, concentrated likelihood.)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

#: Plausibility bound for a single nowcast (pp QoQ log-growth). Mirrors the guard
#: used in run_blockbalanced_benchmark.py: German quarterly GDP growth never left
#: [-11, +9] even in 2020, so |nowcast| beyond this signals a diverged fit.
NOWCAST_CAP = 20.0

#: Minimum number of observed (factor, GDP) quarterly pairs before the TVP
#: Kalman bridge is estimated; below this we fall back to a static OLS bridge.
MIN_TRAIN_QUARTERS = 16

#: COVID quarters whose observation variance is inflated in the Stage-2 bridge
#: (Lenza & Primiceri 2022). Down-weighting these prevents the random-walk
#: loadings from being dragged by — and overshooting after — the pandemic swings.
COVID_QUARTERS = pd.period_range("2020Q1", "2021Q4", freq="Q")

#: Relative observation-variance multiplier applied to the COVID quarters.
COVID_VAR_SCALE = 100.0


# ---------------------------------------------------------------------------
# 1. Smoothed factor extraction + quarterly aggregation
# ---------------------------------------------------------------------------

def extract_smoothed_factors(result: Any) -> pd.DataFrame:
    """Return the smoothed monthly factors as a DataFrame (monthly index × k).

    Mirrors the extraction in ``dfm_sv_bayes.extract_factor_innovations`` but
    keeps the time index so the factors can be aggregated to quarterly frequency.
    """
    try:
        factors_df = result.factors.smoothed          # DataFrame (T × k)
        if not isinstance(factors_df, pd.DataFrame):
            raise AttributeError
        return factors_df.astype(float)
    except AttributeError:
        k_factors = result.model.k_factors
        F = result.smoothed_state[:k_factors, :].T    # (T, k)
        idx = getattr(result.model, "_index", None)
        if idx is None or len(idx) != F.shape[0]:
            idx = pd.RangeIndex(F.shape[0])
        cols = [f"factor_{j + 1}" for j in range(F.shape[1])]
        return pd.DataFrame(F, index=idx, columns=cols).astype(float)


def aggregate_factor_to_quarterly(factors_m: pd.DataFrame) -> pd.DataFrame:
    """Average the smoothed monthly factors within each calendar quarter.

    The 3-month average is the standard "bridge" aggregation of a monthly
    regressor to quarterly frequency for a flow target such as QoQ GDP growth.
    Quarters with at least one observed month are kept (the current quarter is
    complete because the EM front-end fills the ragged edge by AR(p) BIC through
    the quarter-end month).

    Returns
    -------
    pd.DataFrame indexed by quarterly PeriodIndex, columns = factor columns.
    """
    if not isinstance(factors_m.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise TypeError("factors_m must have a Datetime or Period index.")
    q_idx = pd.PeriodIndex(factors_m.index, freq="Q")
    fq = factors_m.copy()
    fq.index = q_idx
    return fq.groupby(level=0).mean().sort_index()


# ---------------------------------------------------------------------------
# 2. TVP regression via concentrated-likelihood Kalman filter
# ---------------------------------------------------------------------------

def _tvp_kalman_filter(
    Z: np.ndarray,
    y: np.ndarray,
    q_ratio: float,
    kappa: float = 1e6,
    obs_var: np.ndarray | None = None,
) -> dict:
    """Run the (scale-concentrated) Kalman filter for a random-walk-coefficient
    regression ``y_t = Z_t' β_t + ε_t``, ``β_t = β_{t-1} + ω_t``.

    All variances are expressed in units of the observation variance σ²:
    R*_t = obs_var[t] (default 1), Q* = q_ratio · I, and the diffuse prior is
    P_0 = κ · I (κ large). The common scale σ² is concentrated out analytically
    (Durbin & Koopman 2012, §2.10 / §6.2). This makes maximum likelihood a
    stable 1-D search over ``q_ratio``.

    Outlier / COVID robustness
    --------------------------
    ``obs_var`` is a per-observation relative variance multiplier. Inflating it
    for the COVID quarters (2020Q1–2021Q4) implements the Lenza & Primiceri
    (2022) "estimate a VAR after March 2020" device: extreme observations are
    down-weighted so the random-walk loadings are not dragged toward (and do not
    overshoot after) the pandemic swings, while still contributing information.

    Parameters
    ----------
    Z       : (n, d) design matrix (rows include the intercept column).
    y       : (n,) target vector.
    q_ratio : state-innovation-to-observation variance ratio (signal-to-noise).
    kappa   : diffuse initial state variance (in σ² units).
    obs_var : optional (n,) array of relative observation-variance multipliers
              (≥ 1; default all-ones = homoskedastic Gaussian).

    Returns
    -------
    dict with:
        loglik     : concentrated log-likelihood.
        sigma2     : MLE of the observation variance σ².
        a_filt     : (n, d) filtered state means β_{t|t}.
        P_filt     : (n, d, d) filtered state covariances (σ²-scaled units).
        a_last     : (d,) last filtered state mean.
        P_last     : (d, d) last filtered state covariance (σ²-scaled).
    """
    n, d = Z.shape
    a = np.zeros(d)
    P = np.eye(d) * kappa
    Q = np.eye(d) * q_ratio
    if obs_var is None:
        obs_var = np.ones(n)

    a_filt = np.zeros((n, d))
    P_filt = np.zeros((n, d, d))

    sum_v2_f = 0.0
    sum_log_f = 0.0
    n_used = 0

    for t in range(n):
        z_t = Z[t]
        y_t = y[t]

        # Predict (random-walk transition: identity)
        P_pred = P + Q

        if not np.isfinite(y_t):
            # Missing target: propagate the prediction, no update.
            a_filt[t] = a
            P_filt[t] = P_pred
            P = P_pred
            continue

        v = y_t - z_t @ a                       # one-step-ahead error
        r_t = float(obs_var[t]) if np.isfinite(obs_var[t]) else 1.0
        f = z_t @ P_pred @ z_t + r_t            # error variance (σ² units)
        f = max(f, 1e-12)
        K = (P_pred @ z_t) / f                   # Kalman gain

        a = a + K * v
        P = P_pred - np.outer(K, z_t @ P_pred)
        P = 0.5 * (P + P.T)                      # numerical symmetrisation

        a_filt[t] = a
        P_filt[t] = P

        sum_v2_f += v * v / f
        sum_log_f += np.log(f)
        n_used += 1

    if n_used == 0:
        return {
            "loglik": -np.inf, "sigma2": np.nan,
            "a_filt": a_filt, "P_filt": P_filt,
            "a_last": a, "P_last": P,
        }

    sigma2 = sum_v2_f / n_used
    sigma2 = max(sigma2, 1e-12)
    loglik = -0.5 * (
        n_used * np.log(2.0 * np.pi)
        + n_used * np.log(sigma2)
        + sum_log_f
        + n_used
    )
    return {
        "loglik": float(loglik),
        "sigma2": float(sigma2),
        "a_filt": a_filt,
        "P_filt": P_filt,
        "a_last": a,
        "P_last": P,
    }


def fit_tvp_regression(
    Z: np.ndarray,
    y: np.ndarray,
    q_ratio: float | None = None,
    q_ratio_bounds: tuple[float, float] = (1e-6, 1.0),
    kappa: float = 1e6,
    obs_var: np.ndarray | None = None,
) -> dict:
    """Fit the random-walk-coefficient regression, choosing ``q_ratio`` by MLE.

    When ``q_ratio`` is None the signal-to-noise ratio is selected by profile
    (concentrated) maximum likelihood via a bounded 1-D search on log10 scale.
    A larger q_ratio lets the loadings drift faster; q_ratio → 0 is the static
    OLS bridge. The bounded search caps the drift to avoid over-fitting on the
    short post-2011 evaluation sample.

    Returns the filter dict from ``_tvp_kalman_filter`` augmented with the
    selected ``q_ratio``.
    """
    if q_ratio is not None:
        out = _tvp_kalman_filter(Z, y, q_ratio, kappa=kappa, obs_var=obs_var)
        out["q_ratio"] = float(q_ratio)
        return out

    lo, hi = np.log10(q_ratio_bounds[0]), np.log10(q_ratio_bounds[1])

    def _neg_ll(log_q: float) -> float:
        """Return the negative concentrated likelihood at a log drift ratio."""
        res = _tvp_kalman_filter(Z, y, float(10.0 ** log_q), kappa=kappa,
                                 obs_var=obs_var)
        ll = res["loglik"]
        return -ll if np.isfinite(ll) else 1e18

    try:
        opt = minimize_scalar(_neg_ll, bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-3})
        best_log_q = float(opt.x)
    except Exception:
        # Robust grid fallback
        grid = np.linspace(lo, hi, 25)
        best_log_q = float(grid[int(np.argmin([_neg_ll(g) for g in grid]))])

    out = _tvp_kalman_filter(Z, y, float(10.0 ** best_log_q), kappa=kappa,
                             obs_var=obs_var)
    out["q_ratio"] = float(10.0 ** best_log_q)
    return out


# ---------------------------------------------------------------------------
# 3. Single-origin TVP-DFM nowcast
# ---------------------------------------------------------------------------

def nowcast_with_tvp(
    X_monthly: pd.DataFrame,
    y_quarterly: pd.Series,
    selected_cols: list[str],
    origin: pd.Period | str,
    k_factors: int = 2,
    factor_order: int = 2,
    idiosyncratic_ar1: bool = True,
    maxiter: int = 200,
    pub_lag_map: pd.Series | None = None,
    fill_method: str = "ar_bic",
    ar_max_p: int = 4,
    ar_min_train: int = 24,
    q_ratio: float | None = None,
    q_ratio_bounds: tuple[float, float] = (1e-6, 1.0),
    covid_var_scale: float = COVID_VAR_SCALE,
    credibility: float = 0.9,
    return_details: bool = False,
) -> dict:
    """Produce a two-step COVID-robust TVP-DFM nowcast for a single origin.

    Stage 1 — EM-DFM (identical to every other model in the pipeline) using the
    real-time publication-lag-masked, AR(p)-BIC-filled information set; yields
    the smoothed monthly factors.

    Stage 2 — quarterly bridge ``y_q = a_q + λ_q' f_q + ε_q`` with random-walk
    coefficients estimated by the concentrated-likelihood Kalman filter. The
    current-quarter nowcast uses the random-walk forecast of the coefficients
    (= last filtered state, since the transition is the identity) applied to the
    current-quarter factor. The COVID quarters are down-weighted via
    ``covid_var_scale`` (Lenza & Primiceri 2022); the drift speed ``q_ratio`` is
    selected by profile maximum likelihood when None.

    Returns
    -------
    dict with keys: origin, current_quarter, k_factors, n_indicators, nowcast,
        q_ratio, ci_lower, ci_upper, sigma, intercept, loadings (list).
    """
    from .nowcast_utils import (
        build_dfm_endog,
        fit_dfm,
        get_current_quarter,
        quarter_end_timestamp,
    )
    from scipy import stats as _st

    origin = pd.Period(origin)
    if len(selected_cols) == 0:
        raise ValueError(f"No selected columns for origin {origin}.")

    current_q = get_current_quarter(origin)
    X_sel = X_monthly[selected_cols]

    # ── Stage 1: EM-DFM front-end (same information set as DFM-EN) ──────────
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

    # ── Stage 2: build quarterly bridge regressors from smoothed factors ───
    factors_m = extract_smoothed_factors(result)
    factors_q = aggregate_factor_to_quarterly(factors_m)

    # Align GDP (observed only through the last completed quarter) with factors.
    y_q = y_quarterly.copy()
    if not isinstance(y_q.index, pd.PeriodIndex):
        y_q.index = pd.PeriodIndex(y_q.index, freq="Q")
    last_completed_q = current_q - 1

    train_q = factors_q.index[
        (factors_q.index <= last_completed_q) & factors_q.index.isin(y_q.index)
    ]
    Fq_train = factors_q.loc[train_q]
    y_train = y_q.reindex(train_q)
    mask = y_train.notna() & np.isfinite(Fq_train).all(axis=1)
    Fq_train = Fq_train.loc[mask]
    y_train = y_train.loc[mask]

    if current_q not in factors_q.index:
        raise KeyError(f"Current quarter {current_q} missing from factor panel.")
    f_now = factors_q.loc[current_q].to_numpy(dtype=float)

    k = factors_q.shape[1]
    z_now = np.concatenate([[1.0], f_now])              # intercept + factors

    # COVID-robust observation-variance multipliers aligned to the training rows.
    is_covid = np.array([qp in COVID_QUARTERS for qp in y_train.index])
    obs_var = np.where(is_covid, float(covid_var_scale), 1.0)

    if len(y_train) < MIN_TRAIN_QUARTERS:
        # Too short for stable drift estimation → weighted static OLS bridge.
        Z = np.column_stack([np.ones(len(y_train)),
                            Fq_train.to_numpy(dtype=float)])
        yv = y_train.to_numpy(dtype=float)
        w = 1.0 / np.sqrt(obs_var)
        beta, *_ = np.linalg.lstsq(Z * w[:, None], yv * w, rcond=None)
        resid = yv - Z @ beta
        sigma = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else np.nan
        nowcast = float(z_now @ beta)
        q_ratio_used = 0.0
        a_pred, P_pred_scaled = beta, None
    else:
        Z = np.column_stack([np.ones(len(y_train)),
                            Fq_train.to_numpy(dtype=float)])
        yv = y_train.to_numpy(dtype=float)
        fit = fit_tvp_regression(
            Z, yv, q_ratio=q_ratio, q_ratio_bounds=q_ratio_bounds,
            obs_var=obs_var,
        )
        # Random-walk forecast of the coefficients = last filtered state.
        a_pred = fit["a_last"]
        P_pred_scaled = fit["P_last"] + np.eye(k + 1) * fit["q_ratio"]
        nowcast = float(z_now @ a_pred)
        sigma = float(np.sqrt(fit["sigma2"]))
        q_ratio_used = fit["q_ratio"]

    # Predictive interval (Gaussian): var = σ²·(z' P_pred z + 1) when available.
    if P_pred_scaled is not None and np.isfinite(sigma):
        pred_var = (sigma ** 2) * float(z_now @ P_pred_scaled @ z_now + 1.0)
        pred_sd = float(np.sqrt(max(pred_var, 0.0)))
    else:
        pred_sd = sigma if np.isfinite(sigma) else np.nan

    zc = float(_st.norm.ppf((1.0 + credibility) / 2.0))
    if np.isfinite(pred_sd):
        ci_lower, ci_upper = nowcast - zc * pred_sd, nowcast + zc * pred_sd
    else:
        ci_lower = ci_upper = np.nan

    out = {
        "origin": str(origin),
        "current_quarter": str(current_q),
        "k_factors": k_factors,
        "n_indicators": len(selected_cols),
        "nowcast": nowcast,
        "q_ratio": q_ratio_used,
        "sigma": sigma,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "intercept": float(a_pred[0]),
        "loadings": [float(v) for v in np.asarray(a_pred[1:])],
    }
    if return_details:
        # Internals needed for the factor-bridge category decomposition:
        # the fitted Stage-1 DFM (for indicator->factor loadings), the current
        # quarter's factor vector, and the selected column order.
        out["result"] = result
        out["factors_now"] = np.asarray(f_now, dtype=float)
        out["current_quarter_period"] = current_q
        out["selected_cols"] = list(selected_cols)
    return out


# ---------------------------------------------------------------------------
# 4. Expanding TVP-DFM nowcast loop  (mirrors run_actpn_nowcast_loop)
# ---------------------------------------------------------------------------

def run_actpn_nowcast_loop_tvp(
    selection_matrix: pd.DataFrame,
    X_monthly: pd.DataFrame,
    y_quarterly: pd.Series,
    quarterly_origins: Iterable[pd.Period | str] | None = None,
    k_factors: int = 2,
    factor_order: int = 2,
    idiosyncratic_ar1: bool = True,
    maxiter: int = 200,
    pub_lag_map: pd.Series | None = None,
    fill_method: str = "ar_bic",
    ar_max_p: int = 4,
    ar_min_train: int = 24,
    q_ratio: float | None = None,
    q_ratio_bounds: tuple[float, float] = (1e-6, 1.0),
    covid_var_scale: float = COVID_VAR_SCALE,
    credibility: float = 0.9,
    save_path: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Expanding A-CD-TPN nowcast loop with a two-step TVP bridge — M1/M2/M3.

    Mirrors ``nowcast_utils.run_actpn_nowcast_loop`` exactly (same selection
    matrix, real-time masking and AR(p) BIC fill) but replaces the fixed
    factor → GDP mapping with the COVID-robust random-walk-loading Kalman bridge.

    The returned DataFrame is schema-compatible with the evaluation helpers in
    ``nowcast_utils`` (quarter, monthly_origin, month_in_quarter, n_indicators,
    k_factors, nowcast, actual, error) and additionally stores the drift
    diagnostics (q_ratio, tvp_intercept, tvp_loading_1 …) plus a Gaussian
    prediction interval (ci_lower_{cov}, ci_upper_{cov}).

    Returns
    -------
    pd.DataFrame indexed by monthly_origin.
    """
    cov_label = int(round(credibility * 100))

    if quarterly_origins is None:
        all_m = pd.PeriodIndex(selection_matrix.index, freq="M")
        quarterly_origins = pd.period_range(
            all_m[0].asfreq("Q"), all_m[-1].asfreq("Q"), freq="Q"
        )

    save_path = Path(save_path) if save_path is not None else None
    records: list[dict] = []

    for q in quarterly_origins:
        q = pd.Period(q, freq="Q")
        actual = float(y_quarterly.get(q, np.nan))

        for m_in_q in (1, 2, 3):
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
                print(f"  {q} M{m_in_q} ({origin_key}): N={len(selected_cols)} — EM + TVP ...",
                      end=" ", flush=True)

            try:
                res = nowcast_with_tvp(
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
                    q_ratio=q_ratio,
                    q_ratio_bounds=q_ratio_bounds,
                    covid_var_scale=covid_var_scale,
                    credibility=credibility,
                )
                nc = res["nowcast"]
                if not (np.isfinite(nc) and abs(nc) <= NOWCAST_CAP):
                    raise ValueError(f"implausible nowcast {nc}")

                error = nc - actual if not np.isnan(actual) else np.nan
                row = {
                    "quarter":               str(q),
                    "monthly_origin":        origin_key,
                    "month_in_quarter":      m_in_q,
                    "n_indicators":          res["n_indicators"],
                    "k_factors":             res["k_factors"],
                    "nowcast":               nc,
                    "actual":                actual,
                    "error":                 error,
                    "q_ratio":               res["q_ratio"],
                    "tvp_intercept":         res["intercept"],
                    f"ci_lower_{cov_label}": res["ci_lower"],
                    f"ci_upper_{cov_label}": res["ci_upper"],
                }
                for j, lam in enumerate(res["loadings"], start=1):
                    row[f"tvp_loading_{j}"] = lam
                records.append(row)

                if verbose:
                    print(f"q_ratio={res['q_ratio']:.1e}  nowcast={nc:.3f}  actual={actual:.3f}")

                if save_path is not None:
                    pd.DataFrame(records).set_index("monthly_origin").to_csv(save_path)

            except Exception as exc:
                if verbose:
                    print(f"ERROR: {exc}")
                records.append({
                    "quarter":               str(q),
                    "monthly_origin":        origin_key,
                    "month_in_quarter":      m_in_q,
                    "n_indicators":          len(selected_cols),
                    "k_factors":             np.nan,
                    "nowcast":               np.nan,
                    "actual":                actual,
                    "error":                 np.nan,
                    "q_ratio":               np.nan,
                    "tvp_intercept":         np.nan,
                    f"ci_lower_{cov_label}": np.nan,
                    f"ci_upper_{cov_label}": np.nan,
                })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.set_index("monthly_origin")
    return df


# ---------------------------------------------------------------------------
# 5. Factor-bridge category decomposition (mirrors the DFM-EN contribution
#    panel, but attributes the *TVP* nowcast)
# ---------------------------------------------------------------------------

#: Label of the neutral bar carrying the TVP bridge intercept a_q, so that the
#: category bars plus this baseline sum exactly to the TVP nowcast (preserving
#: the "bars sum to the nowcast" identity of the DFM-EN decomposition chart).
BASELINE_LABEL = "Baseline"

#: Cap on the *aggregate* (summed, unsigned) category attribution as a
#: multiple of |nowcast|. Unlike ``nowcast_plots._contrib_frame`` (which can
#: hit a genuine divide-by-near-zero), the TVP bridge's per-factor shares
#: ``|L[j,i]| / sum(|L[j,:]|)`` are always well-conditioned; the outsized
#: bars instead come from the *state equation* itself: ``lambda_j * f_jq``
#: for one factor can be legitimately large while the other factor and the
#: intercept net it down to a small nowcast (a real random-walk-bridge
#: property, not an estimation artefact -- see the TVP model card). Either
#: way, showing +50 pp / -45 pp category bars for a 7 pp nowcast is not
#: economically readable, so we apply the same leverage cap and reconciliation
#: convention as ``_contrib_frame``: no category carries more than
#: ``_TVP_CONTRIB_MAX_LEVERAGE`` times |nowcast|, and the netted-out remainder
#: is booked to an explicit ``"Offset"`` bar, kept separate from the genuine
#: intercept ``BASELINE_LABEL`` bar.
_TVP_CONTRIB_MAX_LEVERAGE = 5.0
_TVP_CONTRIB_OFFSET_CATEGORY = "Offset"
_TVP_CONTRIB_OFFSET_ID = "__offset__"


def _tvp_loading_matrix(
    result: Any,
    monthly_cols: list[str],
    k_factors: int,
) -> np.ndarray:
    """Stage-1 indicator->factor loadings as a (k_factors x n) matrix.

    Reads the EM-estimated ``loading.{f}->{col}`` parameters from the fitted
    ``DynamicFactorMQ`` result (same convention as
    ``run_factor_loading_figure._extract_loading_matrix``).
    """
    p = result.params
    L = np.zeros((k_factors, len(monthly_cols)))
    for j, col in enumerate(monthly_cols):
        for f in range(k_factors):
            key = f"loading.{f}->{col}"
            if key in p.index:
                L[f, j] = float(p[key])
    return L


def _tvp_contrib_frame(
    res: dict,
    meta: pd.DataFrame,
    k_factors: int,
) -> tuple[pd.DataFrame, float]:
    """Per-series factor-bridge contributions (pp) with category labels.

    The TVP nowcast is ``nowcast = a_q + sum_j lambda_j * f_jq``. The factor
    part ``C_j = lambda_j * f_jq`` is split across the selected indicators in
    proportion to the absolute Stage-1 loading of each indicator on factor j:

        contrib_i = sum_j C_j * |L[j, i]| / sum_i' |L[j, i']|

    Summing ``contrib_i`` over all indicators recovers ``sum_j C_j`` (the
    factor-driven part of the nowcast); adding the intercept ``a_q`` recovers
    the full nowcast. Category shares use absolute loadings, so they are
    invariant to the (arbitrary) factor sign/rotation.

    Returns
    -------
    (frame, intercept) where ``frame`` is indexed by series id with columns
    ``contrib_pp`` and ``category``, and ``intercept`` is the TVP bridge a_q.
    A synthetic ``_TVP_CONTRIB_OFFSET_ID`` row (category
    ``_TVP_CONTRIB_OFFSET_CATEGORY``) is appended when the leverage cap in
    :data:`_TVP_CONTRIB_MAX_LEVERAGE` binds -- see that constant's docstring.
    """
    result = res["result"]
    monthly_cols = list(res["selected_cols"])
    f_now = np.asarray(res["factors_now"], dtype=float)
    loadings = np.asarray(res["loadings"], dtype=float)
    intercept = float(res["intercept"])

    L = _tvp_loading_matrix(result, monthly_cols, k_factors)

    contrib = np.zeros(len(monthly_cols))
    for j in range(min(k_factors, len(loadings), len(f_now))):
        cj = float(loadings[j] * f_now[j])          # factor j contribution (pp)
        abs_row = np.abs(L[j])
        denom = float(np.nansum(abs_row))
        if not np.isfinite(denom) or denom < 1e-12:
            continue
        contrib += cj * (abs_row / denom)

    nowcast = intercept + float(np.nansum(contrib))
    gross = float(np.nansum(np.abs(contrib)))
    cap = _TVP_CONTRIB_MAX_LEVERAGE * abs(nowcast)
    offset = 0.0
    if np.isfinite(cap) and gross > cap:
        scale = cap / gross if gross > 1e-12 else 0.0
        factor_total_before = float(np.nansum(contrib))
        contrib = contrib * scale
        offset = factor_total_before - float(np.nansum(contrib))

    out = pd.Series(contrib, index=monthly_cols, name="contrib_pp").to_frame()
    out = out.join(meta[["category"]], how="left")
    out["category"] = out["category"].fillna("Unknown")
    if abs(offset) > 1e-9:
        offset_row = pd.DataFrame(
            {"contrib_pp": [offset], "category": [_TVP_CONTRIB_OFFSET_CATEGORY]},
            index=[_TVP_CONTRIB_OFFSET_ID],
        )
        out = pd.concat([out, offset_row])
    return out, intercept


def run_category_contrib_panel_tvp(
    selection_matrix: pd.DataFrame,
    X_monthly: pd.DataFrame,
    y_quarterly: pd.Series,
    meta: pd.DataFrame,
    quarterly_origins: Iterable[pd.Period | str],
    pub_lag_map: pd.Series | None = None,
    k_factors: int = 2,
    factor_order: int = 2,
    idiosyncratic_ar1: bool = True,
    maxiter: int = 200,
    q_ratio: float | None = None,
    q_ratio_bounds: tuple[float, float] = (1e-6, 1.0),
    covid_var_scale: float = COVID_VAR_SCALE,
    m_start: str = "2017-01",
    m_end: str = "2025-12",
    cache_path: str | Path | None = None,
    series_cache_path: str | Path | None = None,
    force_rerun: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fit the two-step TVP-DFM at each monthly origin and cache category
    contributions to the *TVP* nowcast (factor-bridge attribution).

    Schema-compatible with ``nowcast_plots.run_category_contrib_panel`` so the
    dashboard chart (``charts.contributions_stacked``) can consume it unchanged:
    long-format columns ``monthly_origin``, ``quarter``, ``month_in_quarter``,
    ``nowcast``, ``actual``, ``category``, ``contrib_pp``. A ``Baseline`` row per
    origin carries the TVP bridge intercept so the bars sum to the nowcast.
    """
    from ...visualization.nowcast_plots import (
        build_monthly_origins,
        needs_run_contrib_cache,
    )

    cache_path = Path(cache_path) if cache_path is not None else None
    series_cache_path = (
        Path(series_cache_path) if series_cache_path is not None else None
    )
    if cache_path is not None and not needs_run_contrib_cache(
        cache_path, force_rerun, series_cache_path=series_cache_path,
    ):
        if verbose:
            print(f"Loaded TVP category contributions from {cache_path.name}")
        return pd.read_parquet(cache_path)

    origins = build_monthly_origins(quarterly_origins, m_start=m_start, m_end=m_end)
    records: list[dict] = []
    series_records: list[dict] = []

    for origin_p, q, m_in_q in origins:
        origin_key = str(origin_p)
        if origin_key not in selection_matrix.index:
            if verbose:
                print(f"  {origin_key}: not in selection matrix — skipped.")
            continue
        sel_cols = selection_matrix.columns[
            selection_matrix.loc[origin_key].astype(bool)
        ].tolist()
        if not sel_cols:
            if verbose:
                print(f"  {origin_key}: no indicators — skipped.")
            continue
        if verbose:
            print(f"  {q} M{m_in_q} ({origin_key}): N={len(sel_cols)} ...",
                  end=" ", flush=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = nowcast_with_tvp(
                    X_monthly=X_monthly,
                    y_quarterly=y_quarterly,
                    selected_cols=sel_cols,
                    origin=origin_p,
                    k_factors=k_factors,
                    factor_order=factor_order,
                    idiosyncratic_ar1=idiosyncratic_ar1,
                    maxiter=maxiter,
                    pub_lag_map=pub_lag_map,
                    q_ratio=q_ratio,
                    q_ratio_bounds=q_ratio_bounds,
                    covid_var_scale=covid_var_scale,
                    return_details=True,
                )
            nc = float(res["nowcast"])
            if not (np.isfinite(nc) and abs(nc) <= NOWCAST_CAP):
                raise ValueError(f"implausible nowcast {nc}")
            actual = float(y_quarterly.get(q, np.nan))
            frame, intercept = _tvp_contrib_frame(res, meta, k_factors)

            grouped = frame.groupby("category")["contrib_pp"].sum()
            for cat, val in grouped.items():
                records.append({
                    "monthly_origin": origin_key,
                    "quarter": str(q),
                    "month_in_quarter": m_in_q,
                    "nowcast": nc,
                    "actual": actual,
                    "category": cat,
                    "contrib_pp": float(val),
                })
            # Neutral baseline bar carrying the bridge intercept a_q.
            records.append({
                "monthly_origin": origin_key,
                "quarter": str(q),
                "month_in_quarter": m_in_q,
                "nowcast": nc,
                "actual": actual,
                "category": BASELINE_LABEL,
                "contrib_pp": float(intercept),
            })
            for sid, srow in frame.iterrows():
                series_records.append({
                    "monthly_origin": origin_key,
                    "quarter": str(q),
                    "month_in_quarter": m_in_q,
                    "series": sid,
                    "category": srow["category"],
                    "contrib_pp": float(srow["contrib_pp"]),
                })
            if verbose:
                print(f"nowcast={nc:.3f}")
        except Exception as exc:
            if verbose:
                print(f"ERROR: {exc}")

    df = pd.DataFrame(records)
    if cache_path is not None and not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        if verbose:
            print(f"Saved {len(df)} rows -> {cache_path}")
    if series_cache_path is not None and series_records:
        series_cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(series_records).to_parquet(series_cache_path, index=False)
        if verbose:
            print(f"Saved {len(series_records)} rows -> {series_cache_path}")
    return df
