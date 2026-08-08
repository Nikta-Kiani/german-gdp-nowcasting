"""Integrated DFM with Stochastic Volatility (Path A: SV-scaled Kalman smoother).

Motivation
----------
The two-stage model in ``dfm_sv_bayes`` estimates the factors under a
*homoskedastic* state space and only fits stochastic volatility (SV) on the
factor VAR residuals afterwards, using it to rescale the prediction band. As a
result the SV point nowcast is identical to the plain EM-DFM point nowcast — the
volatility never reaches the Kalman gain, so it cannot reweight observations or
alter factor extraction.

This module implements the pragmatic, econometrically consistent fix (an
iterated two-step estimator in the spirit of Doz, Giannone & Reichlin 2011):

    Stage 1  EM-DFM (identical front-end, ``nowcast_utils.fit_dfm``) → smoothed
             factors F_t and fitted parameters (Lambda, A, R, Q).

    Stage 2  Bayesian SV on the factor VAR innovations (reuses
             ``dfm_sv_bayes.fit_sv_all_factors``) → per-factor log-volatility
             path h_{j,t}, hence a relative-variance path
                 r_{j,t} = e^{h_{j,t}} / mean_t(e^{h_{j,t}}),   mean_t r_{j,t}=1.

    Stage 3  Re-run *only* the Kalman smoother on the fitted parameters, but
             with a **time-varying factor-innovation covariance**
                 Q_t = D_t^{1/2} Q D_t^{1/2},   D_t = diag(r_{1,t}, ..., r_{k,t}),
             injected into the state-space representation. The one-step-ahead
             state covariance P_{t|t-1} = A P_{t-1|t-1} A' + Q_t and the Kalman
             gain K_t = P_{t|t-1} Z' (Z P_{t|t-1} Z' + R)^{-1} now depend on the
             estimated volatility, so in turbulent months the common factor
             becomes more responsive and the smoothed GDP cell (the nowcast)
             shifts. The point nowcast therefore *differs* from the plain DFM,
             concentrated in high-volatility episodes (2008-09, 2020), while the
             average scale is preserved because mean_t r_{j,t} = 1.

Optionally (``n_iter`` > 1) Stages 2-3 are iterated: the SV is re-fit on the
innovations of the SV-smoothed factors, giving an approximate joint estimate.

Because the re-smoothed state space carries the time-varying Q_t, the model's
own predictive standard deviation for the missing GDP cell (from
``get_prediction().se_mean``) already reflects the SV. The prediction interval is
therefore built directly from that SV-consistent sigma — no post-hoc rescaling.

Only the *common factor* innovation covariance is made time-varying, following
Marcellino, Porqueddu & Venditti (2016). Idiosyncratic variances are left
constant (as estimated by EM).

State-space mapping (statsmodels ``DynamicFactorMQ``)
----------------------------------------------------
The factor innovations occupy the leading ``k_factors`` block of ``state_cov``
(``state_cov[:k, :k]``); idiosyncratic variances occupy ``state_cov[k:, k:]``
(see ``DynamicFactorMQ.update``). We scale only the factor block. Injection is
done by wrapping the model's ``update`` so that, after the fitted matrices are
set, ``state_cov`` is replaced by the pre-computed time-varying array before the
smoother runs; the original 2-D covariance is restored afterwards so the fitted
result object is left untouched.

References
----------
Marcellino, M., Porqueddu, M. & Venditti, F. (2016). Short-Term GDP Forecasting
    with a Mixed-Frequency Dynamic Factor Model with Stochastic Volatility.
    Journal of Business & Economic Statistics, 34(1), 118-127.
Doz, C., Giannone, D. & Reichlin, L. (2011). A two-step estimator for large
    approximate dynamic factor models based on Kalman filtering. Journal of
    Econometrics, 164(1), 188-205.
Bańbura, M., Giannone, D., Modugno, M. & Reichlin, L. (2013). Now-casting and
    the real-time data flow. Handbook of Economic Forecasting, vol. 2A.
Kim, C., Shephard, N. & Chib, S. (1998). Stochastic Volatility: Likelihood
    Inference and Comparison with ARCH Models. Review of Economic Studies,
    65(3), 361-393.
"""

from __future__ import annotations

import types
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .dfm_sv_bayes import (
    extract_factor_innovations,
    fit_sv_all_factors,
    kalman_gdp_predictive_sd,
)


# ---------------------------------------------------------------------------
# 1. Relative-variance path from the SV posterior
# ---------------------------------------------------------------------------

def compute_relative_vol_path(
    sv_samples_list: list[dict[str, np.ndarray]],
    n_months: int,
    factor_order: int,
    r_clip: tuple[float, float] = (0.1, 20.0),
) -> np.ndarray:
    """Build the per-factor relative-variance path aligned to the monthly grid.

    For factor j the posterior-mean innovation variance at innovation time t is
    ``v_{j,t} = E_s[exp(h_{j,t})]``. The relative-variance multiplier is

        r_{j,t} = v_{j,t} / mean_t v_{j,t},

    which has time-average one (so the *level* of the factor-innovation
    covariance is unchanged on average — only its distribution over time is
    reshaped). The SV is estimated on standardised innovations, but the ratio is
    scale-free, so no back-transformation is needed.

    Alignment
    ---------
    The VAR(``factor_order``) innovations lose the first ``p = factor_order``
    monthly observations, so innovation index ``i`` corresponds to calendar month
    ``p + i``. The leading ``p`` months receive r = 1 (neutral).

    Parameters
    ----------
    sv_samples_list : list of SV posterior dicts (one per factor), each with an
                      ``h_all`` array of shape (S, T_inno).
    n_months        : number of monthly time points in the state space
                      (= ``result.model.nobs`` = length of the smoothed factors).
    factor_order    : VAR order p used when extracting the innovations.
    r_clip          : (min, max) clip bounds on the multiplier to keep the
                      re-smoothed filter numerically stable.

    Returns
    -------
    r_full : (n_months, k) array of relative-variance multipliers.
    """
    k = len(sv_samples_list)
    r_full = np.ones((n_months, k), dtype=float)
    if k == 0:
        return r_full

    p = int(factor_order)
    lo, hi = float(r_clip[0]), float(r_clip[1])

    for j, samps in enumerate(sv_samples_list):
        h_all = np.asarray(samps["h_all"], dtype=float)      # (S, T_inno)
        if h_all.ndim != 2 or h_all.shape[1] == 0:
            continue
        var_path = np.mean(np.exp(h_all), axis=0)            # (T_inno,)
        mean_var = float(np.mean(var_path))
        if not np.isfinite(mean_var) or mean_var <= 0:
            continue
        r = var_path / mean_var
        r = np.clip(r, lo, hi)
        if not np.all(np.isfinite(r)):
            continue

        t_inno = r.shape[0]
        start = min(p, n_months)
        end = min(n_months, start + t_inno)
        if end > start:
            r_full[start:end, j] = r[: end - start]

    return r_full


# ---------------------------------------------------------------------------
# 2. Re-smooth the fitted DFM with a time-varying factor-innovation covariance
# ---------------------------------------------------------------------------

def resmooth_with_sv_state_cov(result: Any, r_full: np.ndarray) -> object:
    """Re-run the Kalman smoother with a time-varying factor-innovation cov.

    The fitted parameters of ``result`` are held fixed. Only the leading
    ``k_factors`` block of ``state_cov`` is made time-varying, scaled month by
    month by ``r_full`` as ``Q_t[:k,:k] = D_t^{1/2} Q[:k,:k] D_t^{1/2}`` with
    ``D_t = diag(r_full[t])``. Idiosyncratic variances are unchanged.

    The fitted ``result`` object is left untouched: the model's original 2-D
    ``state_cov`` is restored after smoothing (the returned results object
    carries its own smoother output, so this does not affect it).

    Parameters
    ----------
    result : fitted ``DynamicFactorMQResults`` from ``nowcast_utils.fit_dfm``.
    r_full : (nobs, k) relative-variance path from ``compute_relative_vol_path``.

    Returns
    -------
    A new ``DynamicFactorMQResults`` from smoothing at the fitted parameters
    under the SV-scaled state covariance.
    """
    mod = result.model
    params = np.asarray(result.params, dtype=float)
    k = int(mod.k_factors)
    nobs = int(mod.nobs)

    # Defensive length/shape alignment of r_full to (nobs, k).
    rf = np.ones((nobs, k), dtype=float)
    n = min(nobs, r_full.shape[0])
    kk = min(k, r_full.shape[1])
    rf[:n, :kk] = np.asarray(r_full, dtype=float)[:n, :kk]

    cls_update = type(mod).update

    def _collapse_state_cov_2d() -> None:
        """Restore the covariance shape expected by statsmodels updates."""
        # ``DynamicFactorMQ.update`` assigns 2-D slices into ``state_cov`` (e.g.
        # the idiosyncratic block), which fails if the matrix is currently
        # time-varying (3-D). Collapse to the leading time slice first.
        sc = np.asarray(mod["state_cov"])
        if sc.ndim == 3:
            mod.ssm["state_cov"] = np.ascontiguousarray(sc[:, :, 0])

    # Reinstate the EM-estimated Kalman initialization. ``fit_em`` smooths the
    # returned result with a 'known' initialization derived from the EM
    # (constant = smoothed state at t=0), but then *restores the default*
    # stationary/diffuse initialization on the model. A fresh ``smooth`` would
    # therefore use the wrong initial condition and not reproduce the EM
    # nowcast. Re-applying the stored EM initialization makes the r=1 re-smooth
    # exactly reproduce the plain DFM point, so any change is due to the SV alone.
    em_init = None
    try:
        rv = result.mle_retvals
        if rv is not None and "inits" in rv and len(rv["inits"]) > 0:
            em_init = rv["inits"][-1]
    except Exception:
        em_init = None
    if em_init is not None:
        mod.ssm.initialization = em_init

    # Set the fitted matrices and capture the base (homoskedastic) state_cov.
    _collapse_state_cov_2d()
    cls_update(mod, params)
    Q0 = np.asarray(mod["state_cov"], dtype=float)
    if Q0.ndim == 3:
        Q0 = Q0[:, :, 0]
    Q0 = np.ascontiguousarray(Q0)
    Qf = Q0[:k, :k].copy()

    # Pre-compute the time-varying state covariance.
    d = np.sqrt(np.clip(rf, 1e-8, None))                     # (nobs, k)
    Q_tv = np.repeat(Q0[:, :, None], nobs, axis=2)           # (m, m, nobs)
    for t in range(nobs):
        dt = d[t]
        Q_tv[:k, :k, t] = (dt[:, None] * Qf) * dt[None, :]

    def _patched_update(
        self: Any,
        prms: Any,
        **kwargs: Any,
    ) -> None:
        """Apply fitted parameters, then restore the time-varying covariance."""
        # Reset to the 2-D base so the base update's slice assignments succeed
        # (update is called repeatedly: during results construction *and* on
        # every ``get_prediction`` / ``predict`` call), then re-impose the
        # time-varying factor-innovation covariance.
        self.ssm["state_cov"] = Q0
        cls_update(self, prms, **kwargs)
        self.ssm["state_cov"] = Q_tv

    # The override is left installed on this (per-origin, disposable) model
    # instance: ``get_prediction``/``predict`` on the returned results object
    # re-invoke ``update``, and must keep re-imposing the SV-scaled covariance so
    # the in-sample GDP prediction stays consistent with the re-smoothed states.
    # A subsequent ``resmooth`` on the same model collapses state_cov back to
    # 2-D at entry (``_collapse_state_cov_2d``) and re-installs a fresh override.
    mod.update = types.MethodType(_patched_update, mod)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_sv = mod.smooth(params)

    return res_sv


# ---------------------------------------------------------------------------
# 3. Aggregate relative volatility at the nowcast target (diagnostic)
# ---------------------------------------------------------------------------

def _rel_vol_at_target(
    sv_samples_list: list[dict[str, np.ndarray]],
    r_full: np.ndarray,
) -> float:
    """Variance-share weighted mean of the relative volatility at the last month.

    Weights are each factor's historical innovation variance (proxied by the
    squared fit-time ``_scale``), matching ``dfm_sv_bayes.compute_sv_relative_vol``.
    Returned on the volatility (sd) scale: sqrt of the relative-variance path.
    """
    if r_full.size == 0:
        return float("nan")
    r_target = np.asarray(r_full[-1], dtype=float)           # (k,)
    weights = []
    for samps in sv_samples_list:
        scale = float(samps.get("_scale", np.array([1.0]))[0])
        weights.append(scale ** 2)
    w = np.asarray(weights, dtype=float)
    if w.size != r_target.size or not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones_like(r_target)
    w = w / w.sum()
    return float(np.sqrt(np.sum(w * r_target)))


# ---------------------------------------------------------------------------
# 4. Single-origin integrated SV nowcast
# ---------------------------------------------------------------------------

def nowcast_with_sv_integrated(
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
    num_warmup: int = 500,
    num_samples: int = 1000,
    credibility: float = 0.9,
    sigma_floor: float = 0.05,
    rng_seed: int = 42,
    progress_bar: bool = False,
    robust: bool = False,
    r_clip: tuple[float, float] = (0.1, 20.0),
    n_iter: int = 1,
) -> dict:
    """Produce an SV-integrated DFM nowcast for a single origin (Path A).

    Unlike ``dfm_sv_bayes.nowcast_with_sv`` — where the SV only rescales the band
    and the point equals the plain EM nowcast — here the estimated volatility is
    fed back into the Kalman smoother via a time-varying factor-innovation
    covariance, so the **point nowcast itself responds to the SV**.

    Parameters
    ----------
    (shared with ``nowcast_with_sv``) plus:
    r_clip : (min, max) clip on the per-month relative-variance multiplier.
    n_iter : number of SV -> re-smooth iterations (1 = single pass; >1 re-fits
             the SV on the SV-smoothed factors, an approximate joint estimate).

    Returns
    -------
    dict with keys: origin, current_quarter, k_factors, n_indicators,
        nowcast (SV point), nowcast_baseline (plain EM point), point_shift,
        ci_lower, ci_upper, sigma_em (full SV predictive sd), rel_vol,
        rel_vol_target, n_iter, _sv_samples.
    """
    from scipy import stats
    from .nowcast_utils import (
        build_dfm_endog,
        fit_dfm,
        extract_nowcast,
        get_current_quarter,
        quarter_end_timestamp,
    )

    origin = pd.Period(origin)
    if len(selected_cols) == 0:
        raise ValueError(f"No selected columns for origin {origin}.")

    X_sel = X_monthly[selected_cols]

    # ── Stage 1: EM-DFM (identical front-end to every other model) ─────────
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

    # Plain EM point nowcast (the DFM / two-stage-SV point, for comparison).
    em_nowcast_baseline = extract_nowcast(result, origin)

    # ── Stages 2-3 (optionally iterated): SV → time-varying Q → re-smooth ──
    factors_source = result
    res_sv = result
    sv_samples_list: list[dict[str, np.ndarray]] = []
    r_full = np.ones((int(result.model.nobs), int(result.model.k_factors)))

    n_iter = max(1, int(n_iter))
    for _ in range(n_iter):
        _, innovations = extract_factor_innovations(
            factors_source, factor_order=factor_order
        )
        sv_samples_list = fit_sv_all_factors(
            innovations,
            num_warmup=num_warmup,
            num_samples=num_samples,
            rng_seed=rng_seed,
            progress_bar=progress_bar,
            robust=robust,
        )
        n_months = int(result.model.nobs)
        r_full = compute_relative_vol_path(
            sv_samples_list, n_months, factor_order, r_clip=r_clip
        )
        res_sv = resmooth_with_sv_state_cov(result, r_full)
        factors_source = res_sv

    sv_nowcast = extract_nowcast(res_sv, origin)

    # ── SV-consistent prediction interval ──────────────────────────────────
    # The re-smoothed state space carries the time-varying Q_t, so its own
    # predictive SD for the missing GDP cell already reflects the SV. No
    # post-hoc rescaling is applied.
    target_month = quarter_end_timestamp(get_current_quarter(origin))
    sigma_em = kalman_gdp_predictive_sd(res_sv, target_month)
    if not np.isfinite(sigma_em) or sigma_em < sigma_floor:
        sigma_em = float(sigma_floor)

    z = float(stats.norm.ppf((1.0 + credibility) / 2.0))
    lower = sv_nowcast - z * sigma_em
    upper = sv_nowcast + z * sigma_em

    rel_vol_target = _rel_vol_at_target(sv_samples_list, r_full)

    return {
        "origin":           str(origin),
        "current_quarter":  str(get_current_quarter(origin)),
        "k_factors":        k_factors,
        "n_indicators":     len(selected_cols),
        "nowcast":          float(sv_nowcast),
        "nowcast_baseline": float(em_nowcast_baseline),
        "point_shift":      float(sv_nowcast - em_nowcast_baseline),
        "ci_lower":         float(lower),
        "ci_upper":         float(upper),
        # sigma_em is the FULL SV-consistent predictive sd; rel_vol is set to 1
        # so downstream helpers (compute_crps_gaussian) reconstruct sigma =
        # sigma_em * sqrt(rel_vol) = sigma_em without double-counting the SV.
        "sigma_em":         float(sigma_em),
        "rel_vol":          1.0,
        "rel_vol_target":   float(rel_vol_target),
        "n_iter":           n_iter,
        "_sv_samples":      sv_samples_list,
    }


# ---------------------------------------------------------------------------
# 5. Expanding integrated-SV nowcast loop  (mirrors run_actpn_nowcast_loop_sv)
# ---------------------------------------------------------------------------

def run_actpn_nowcast_loop_sv_integrated(
    selection_matrix: pd.DataFrame,
    X_monthly: pd.DataFrame,
    y_quarterly: pd.Series,
    quarterly_origins: Iterable[pd.Period | str] | None = None,
    factor_order: int = 2,
    idiosyncratic_ar1: bool = True,
    maxiter: int = 200,
    k_factors: int = 2,
    pub_lag_map: pd.Series | None = None,
    fill_method: str = "ar_bic",
    ar_max_p: int = 4,
    ar_min_train: int = 24,
    num_warmup: int = 500,
    num_samples: int = 1000,
    credibility: float = 0.9,
    sigma_floor: float = 0.05,
    rng_seed: int = 42,
    r_clip: tuple[float, float] = (0.1, 20.0),
    n_iter: int = 1,
    save_path: str | Path | None = None,
    verbose: bool = True,
    robust: bool = False,
) -> pd.DataFrame:
    """Expanding A-CD-TPN loop with SV integrated into factor extraction.

    Schema-compatible with ``dfm_sv_bayes.run_actpn_nowcast_loop_sv`` (same
    ``ci_lower_{cov}`` / ``ci_upper_{cov}`` / ``sigma_em`` / ``rel_vol`` columns,
    consumable by ``nowcast_utils.build_interval_calibration_table`` and
    ``compute_crps_gaussian``), with extra diagnostic columns
    ``nowcast_baseline``, ``point_shift`` and ``rel_vol_target`` recording how
    far the SV moves the point relative to the plain EM-DFM nowcast.

    Returns
    -------
    pd.DataFrame indexed by ``monthly_origin``.
    """
    if pub_lag_map is None:
        warnings.warn(
            "pub_lag_map is None: integrated-SV loop uses balanced-panel masking. "
            "Pass pub_lag_map to match the A-CD-TPN real-time information set.",
            UserWarning,
            stacklevel=2,
        )

    cov_label = int(round(credibility * 100))

    if quarterly_origins is None:
        all_m = pd.PeriodIndex(selection_matrix.index, freq="M")
        quarterly_origins = pd.period_range(
            all_m[0].asfreq("Q"), all_m[-1].asfreq("Q"), freq="Q"
        )

    save_path = Path(save_path) if save_path is not None else None
    records: list[dict] = []
    global_idx = 0

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
                robust_tag = " [robust]" if robust else ""
                print(
                    f"  {q} M{m_in_q} ({origin_key}): N={len(selected_cols)}"
                    f" — EM + integrated-SV{robust_tag} ...",
                    end=" ", flush=True,
                )

            try:
                res = nowcast_with_sv_integrated(
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
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    credibility=credibility,
                    sigma_floor=sigma_floor,
                    rng_seed=rng_seed + global_idx,
                    progress_bar=False,
                    robust=robust,
                    r_clip=r_clip,
                    n_iter=n_iter,
                )

                error = res["nowcast"] - actual if not np.isnan(actual) else np.nan

                row = {
                    "quarter":               str(q),
                    "monthly_origin":        origin_key,
                    "month_in_quarter":      m_in_q,
                    "n_indicators":          res["n_indicators"],
                    "k_factors":             res["k_factors"],
                    "nowcast":               res["nowcast"],
                    "nowcast_baseline":      res["nowcast_baseline"],
                    "point_shift":           res["point_shift"],
                    "actual":                actual,
                    "error":                 error,
                    f"ci_lower_{cov_label}": res["ci_lower"],
                    f"ci_upper_{cov_label}": res["ci_upper"],
                    "rel_vol":               res["rel_vol"],
                    "rel_vol_target":        res["rel_vol_target"],
                    "sigma_em":              res.get("sigma_em", np.nan),
                }
                records.append(row)

                if verbose:
                    print(
                        f"k={res['k_factors']}  nowcast={res['nowcast']:.3f}  "
                        f"(EM={res['nowcast_baseline']:.3f}, "
                        f"Δ={res['point_shift']:+.3f})  actual={actual:.3f}  "
                        f"CI=[{res['ci_lower']:.3f}, {res['ci_upper']:.3f}]"
                    )

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
                    "nowcast_baseline":      np.nan,
                    "point_shift":           np.nan,
                    "actual":                actual,
                    "error":                 np.nan,
                    f"ci_lower_{cov_label}": np.nan,
                    f"ci_upper_{cov_label}": np.nan,
                    "rel_vol":               np.nan,
                    "rel_vol_target":        np.nan,
                    "sigma_em":              np.nan,
                })

            global_idx += 1

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.set_index("monthly_origin")
    return df
