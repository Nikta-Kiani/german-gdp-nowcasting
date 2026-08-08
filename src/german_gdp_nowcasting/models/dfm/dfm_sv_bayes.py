"""Bayesian DFM with Stochastic Volatility via numpyro.

.. note::
    RETIRED as the headline SV model (July 2026). This two-stage design only
    rescales the prediction band -- the point nowcast is identical to plain
    DFM-EN -- and has been fully replaced in the dashboard/evaluation surface
    by the *integrated* SV model in ``dfm_sv_integrated.py`` (k=2), which feeds
    SV back into the Kalman smoother so the point nowcast can actually change.
    This module is kept only because ``dfm_sv_integrated.py`` reuses its SV
    fitting primitives (``fit_sv_all_factors``, ``extract_factor_innovations``,
    ``kalman_gdp_predictive_sd``); ``run_actpn_nowcast_loop_sv`` itself (the
    two-stage nowcast loop) should not be re-run.

Two-stage approach
------------------
Stage 1 — EM (fast, handles mixed frequency):
    Fit DynamicFactorMQ via statsmodels EM/Kalman to obtain:
      • smoothed factors  F_t  (k × T)
      • EM point nowcast  ŷ_T

Stage 2 — Bayesian SV (numpyro NUTS):
    Fit an AR(1) log-volatility (Kim-Shephard-Chib) model on the VAR
    residuals of the smoothed factors to obtain:
      • time-varying log-volatility  h_t  per factor
      • relative volatility at the nowcast origin  r_T
      • SV-adjusted prediction interval  [ŷ_T ± z · σ_em · √r_T]

Two SV variants
---------------
Standard (_sv_model):
    y_t | h_t ~ N(0, exp(h_t / 2))
    Appropriate for normal regimes; may underestimate COVID-scale shocks.

Robust / outlier-tolerant (_sv_model_robust):
    y_t | h_t, λ_t ~ N(0, √λ_t · exp(h_t / 2))
    λ_t ~ InvGamma(ν/2, ν/2)  →  y_t | h_t ~ Student-t(ν, 0, exp(h_t/2))
    Marginalising over λ_t gives Student-t innovations. A single extreme
    observation (e.g. 2020Q2) is absorbed by λ_t rather than distorting the
    entire h_t path. Follows Lenza & Primiceri (2022).

Stochastic Volatility model specification
-----------------------------------------
    y_t  ~ N(0, exp(h_t / 2))                             observation
    h_t  = μ + ρ(h_{t-1} − μ) + σ_η ζ_t,  ζ_t ~ N(0,1) log-vol AR(1)
    h_0  ~ N(μ, σ_η / √(1 − ρ²))           stationary initialisation

Priors (calibrated for macro GDP nowcasting with COVID adaptability)
--------------------------------------------------------------------
    μ       ~ N(−0.5, 1.5)     flexible log-vol level
    ρ       ~ Beta(5, 2)       E[ρ] ≈ 0.71 — moderately persistent,
                                can adapt to sudden regime breaks
    σ_η     ~ HalfNormal(0.3)  allows large COVID-scale vol innovations

Non-centred parametrisation
----------------------------
    z_t ~ N(0, 1)  (sampled by NUTS)
    h_t = μ + ρ^t (h_0 − μ) + σ_η · cumulative_AR1(z)
    Avoids funnel geometry of centred form; better NUTS mixing
    (Papaspiliopoulos et al. 2007; Stan User Guide §13.3).

References
----------
Kim, C., Shephard, N. & Chib, S. (1998). Stochastic Volatility: Likelihood
    Inference and Comparison with ARCH Models. Review of Economic Studies,
    65(3), 361–393.
Cogley, T. & Sargent, T. J. (2005). Drifts and Volatilities: Monetary
    Policies and Outcomes in the Post WWII U.S. Review of Economic Dynamics,
    8(2), 262–302.
Primiceri, G. E. (2005). Time Varying Structural Vector Autoregressions and
    Monetary Policy. Review of Economic Studies, 72(3), 821–852.
Lenza, M. & Primiceri, G. E. (2022). How to Estimate a VAR after March 2020.
    Journal of Applied Econometrics, 37(4), 688–699.
Clark, T. E. (2011). Real-Time Density Forecasts from Bayesian Vector
    Autoregressions with Stochastic Volatility. Journal of Business &
    Economic Statistics, 29(3), 327–341.
Marcellino, M., Porqueddu, M. & Venditti, F. (2016). Short-Term GDP
    Forecasting with a Mixed-Frequency Dynamic Factor Model with Stochastic
    Volatility. Journal of Business & Economic Statistics, 34(1), 118–127.
Papaspiliopoulos, O., Roberts, G. O. & Sköld, M. (2007). A general framework
    for the parametrization of hierarchical models. Statistical Science, 22(1),
    59–73.
Hoffman, M. D. & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively Setting
    Path Lengths in Hamiltonian Monte Carlo. JMLR, 15(47), 1593–1623.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


# ---------------------------------------------------------------------------
# 1. Factor extraction from a fitted DynamicFactorMQResults
# ---------------------------------------------------------------------------

def extract_factor_innovations(
    result: Any,
    factor_order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract smoothed factors and VAR innovations from a fitted DFM.

    The smoothed factors are taken from ``result.factors.smoothed`` when
    available (statsmodels ≥ 0.14). A fallback reads directly from the
    Kalman smoother state array.

    A VAR(factor_order) is then estimated via OLS on the smoothed factors.
    The residuals are the "factor innovations" used as input to the SV model.

    Parameters
    ----------
    result       : fitted DynamicFactorMQResults (from nowcast_utils.fit_dfm).
    factor_order : VAR order used when fitting the DFM.

    Returns
    -------
    factors      : (T, k) array — smoothed factors.
    innovations  : (T − factor_order, k) array — VAR OLS residuals.
    """
    try:
        factors_df = result.factors.smoothed          # DataFrame (T × k)
        F = factors_df.values.astype(float)
    except AttributeError:
        k_factors = result.model.k_factors
        F = result.smoothed_state[:k_factors, :].T    # (T, k)

    T, k = F.shape

    if T <= factor_order + k + 1:
        raise ValueError(
            f"Too few time points ({T}) to fit VAR({factor_order}) on {k} factors."
        )

    p = factor_order
    n_obs = T - p
    Y = F[p:, :]                                               # (n_obs, k)
    X = np.hstack([F[p - j - 1: T - j - 1, :] for j in range(p)])  # (n_obs, k·p)

    # Column-wise standardisation stabilises the VAR OLS when smoothed factors
    # spike (late-sample EM fits); avoids lstsq / matmul overflow without
    # changing the economic ranking of factor innovations for the SV layer.
    x_mu, x_sig = X.mean(axis=0), X.std(axis=0)
    y_mu, y_sig = Y.mean(axis=0), Y.std(axis=0)
    x_sig = np.where(x_sig < 1e-8, 1.0, x_sig)
    y_sig = np.where(y_sig < 1e-8, 1.0, y_sig)
    Xs = (X - x_mu) / x_sig
    Ys = (Y - y_mu) / y_sig

    A_T, _, _, _ = np.linalg.lstsq(Xs, Ys, rcond=1e-10)     # (k·p, k)
    innovations = (Ys - Xs @ A_T).astype(float) * y_sig      # rescale to Y units

    if not np.all(np.isfinite(innovations)):
        # Last resort: first-difference of smoothed factors (preserves sign/timing).
        innovations = np.diff(F[p - 1:], axis=0).astype(float)

    return F, innovations


# ---------------------------------------------------------------------------
# 2. Numpyro SV models  (non-centred parametrisation)
# ---------------------------------------------------------------------------

def _sv_model(y: jnp.ndarray) -> None:
    """Numpyro model: univariate AR(1) stochastic volatility — standard Gaussian.

    Non-centred form: z_t ~ N(0,1) are the sampled variables; h_t is a
    deterministic function of (μ, ρ, σ_η, z) computed via jax.lax.scan.

    h_next is sampled from the one-step-ahead predictive distribution
    N(μ + ρ(h_T − μ), σ_η) rather than stored as a deterministic node,
    so that the posterior over h_{T+1} correctly propagates σ_η uncertainty.

    Parameters
    ----------
    y : 1-D JAX float64 array, zero-mean innovation series of length T.
    """
    T = y.shape[0]

    # Global SV parameters — priors calibrated for macro GDP with COVID adaptability
    mu        = numpyro.sample("mu",        dist.Normal(-0.5, 1.5))
    rho       = numpyro.sample("rho",       dist.Beta(5.0, 2.0))
    sigma_eta = numpyro.sample("sigma_eta", dist.HalfNormal(0.3))

    # Non-centred log-vol innovations
    z = numpyro.sample("z", dist.Normal(jnp.zeros(T), 1.0))

    # AR(1) recursion via jax.lax.scan
    sigma_stat = sigma_eta / jnp.sqrt(jnp.clip(1.0 - rho ** 2, min=1e-8))
    h0 = mu + sigma_stat * z[0]

    def _step(h_prev: jnp.ndarray, z_t: jnp.ndarray) -> tuple:
        """Advance the non-centred AR(1) log-volatility state."""
        h_t = mu + rho * (h_prev - mu) + sigma_eta * z_t
        return h_t, h_t

    h_final, h_rest = jax.lax.scan(_step, h0, z[1:])
    h = jnp.concatenate([h0[None], h_rest])                    # (T,)

    numpyro.deterministic("h_all", h)

    # Sample h_next from its predictive distribution (not deterministic mean),
    # so posterior draws of h_{T+1} correctly reflect σ_η uncertainty.
    h_next_mean = mu + rho * (h_final - mu)
    numpyro.sample("h_next", dist.Normal(h_next_mean, sigma_eta))

    # Vectorised observation likelihood
    numpyro.sample("obs", dist.Normal(0.0, jnp.exp(0.5 * h)), obs=y)


def _sv_model_robust(y: jnp.ndarray) -> None:
    """Numpyro model: univariate AR(1) SV with Student-t innovations.

    Implements the Lenza & Primiceri (2022) outlier-robust formulation:

        y_t | h_t, λ_t ~ N(0, √λ_t · exp(h_t / 2))
        λ_t ~ InvGamma(ν/2, ν/2)   →   y_t | h_t ~ Student-t(ν, 0, exp(h_t/2))
        ν   ~ Gamma(2, 0.1)         weakly informative; E[ν] = 20, heavy-tail mass

    Marginalising over λ_t gives Student-t innovations with ν degrees of
    freedom. A single extreme observation (COVID 2020Q2) is absorbed by its
    own λ_t rather than biasing the estimated h_t path for surrounding quarters.
    This is particularly important for a short evaluation window (2011–2025)
    where a single 10σ event can otherwise dominate the full-sample SV estimate.

    All log-volatility parameters share the same priors as ``_sv_model``.

    Parameters
    ----------
    y : 1-D JAX float64 array, zero-mean innovation series of length T.
    """
    T = y.shape[0]

    mu        = numpyro.sample("mu",        dist.Normal(-0.5, 1.5))
    rho       = numpyro.sample("rho",       dist.Beta(5.0, 2.0))
    sigma_eta = numpyro.sample("sigma_eta", dist.HalfNormal(0.3))

    # Degrees of freedom: Gamma(2, 0.1) has E[ν] ≈ 20; allows heavy tails
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))

    z = numpyro.sample("z", dist.Normal(jnp.zeros(T), 1.0))

    sigma_stat = sigma_eta / jnp.sqrt(jnp.clip(1.0 - rho ** 2, min=1e-8))
    h0 = mu + sigma_stat * z[0]

    def _step(h_prev: jnp.ndarray, z_t: jnp.ndarray) -> tuple:
        """Advance the robust model's AR(1) log-volatility state."""
        h_t = mu + rho * (h_prev - mu) + sigma_eta * z_t
        return h_t, h_t

    h_final, h_rest = jax.lax.scan(_step, h0, z[1:])
    h = jnp.concatenate([h0[None], h_rest])

    numpyro.deterministic("h_all", h)

    h_next_mean = mu + rho * (h_final - mu)
    numpyro.sample("h_next", dist.Normal(h_next_mean, sigma_eta))

    # Student-t variance scaling: λ_t ~ InvGamma(ν/2, ν/2)
    half_nu = nu / 2.0
    lam = numpyro.sample("lam", dist.InverseGamma(half_nu, half_nu).expand([T]))

    # Robust observation: y_t | h_t, λ_t ~ N(0, √λ_t · exp(h_t/2))
    numpyro.sample("obs", dist.Normal(0.0, jnp.sqrt(lam) * jnp.exp(0.5 * h)), obs=y)


# ---------------------------------------------------------------------------
# 3. MCMC fitting
# ---------------------------------------------------------------------------

def fit_sv_factor(
    innovations_1d: np.ndarray,
    num_warmup: int = 500,
    num_samples: int = 1000,
    rng_key: jax.Array | None = None,
    progress_bar: bool = False,
    robust: bool = False,
) -> dict[str, np.ndarray]:
    """Run NUTS on the SV model for a single factor's innovations.

    The innovation series is standardised internally so that the prior on
    μ (log-vol level) is scale-appropriate. The scale is stored in the
    returned dict under key ``_scale`` for back-transformation.

    Parameters
    ----------
    innovations_1d : 1-D zero-mean innovation series.
    num_warmup     : NUTS warm-up (burn-in) steps. Default 500.
    num_samples    : posterior draws to retain. Default 1000.
    rng_key        : JAX PRNG key; defaults to PRNGKey(0).
    progress_bar   : show tqdm bar during sampling.
    robust         : if True, use Student-t innovations (_sv_model_robust).
                     Recommended for samples that include the COVID period.

    Returns
    -------
    dict with posterior arrays: 'h_all' (S×T), 'h_next' (S,),
        'mu' (S,), 'rho' (S,), 'sigma_eta' (S,), '_scale' (scalar).
        When ``robust=True`` also includes 'nu' (S,) and 'lam' (S×T).
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    scale = float(np.std(innovations_1d))
    if scale < 1e-10:
        scale = 1.0

    # float64 for numerical precision — important near COVID extremes
    y_std = (innovations_1d / scale).astype(np.float64)

    model_fn = _sv_model_robust if robust else _sv_model

    kernel = NUTS(model_fn, target_accept_prob=0.85)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        progress_bar=progress_bar,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mcmc.run(rng_key, y=jnp.asarray(y_std))

    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    samples["_scale"] = np.array([scale])
    return samples


def fit_sv_all_factors(
    innovations: np.ndarray,
    num_warmup: int = 500,
    num_samples: int = 1000,
    rng_seed: int = 42,
    progress_bar: bool = False,
    robust: bool = False,
) -> list[dict[str, np.ndarray]]:
    """Fit the SV model independently for each factor (diagonal SV).

    Parameters
    ----------
    innovations  : (T_inno, k) array of factor VAR residuals.
    num_warmup   : NUTS warm-up steps per factor.
    num_samples  : posterior draws per factor.
    rng_seed     : base integer seed; factor j uses PRNGKey(rng_seed + j).
    progress_bar : show progress bar.
    robust       : if True, use Student-t innovations (_sv_model_robust).

    Returns
    -------
    List of sample dicts, one per factor column.
    """
    k = innovations.shape[1]
    samples_list: list[dict[str, np.ndarray]] = []
    for j in range(k):
        rng_key = jax.random.PRNGKey(rng_seed + j)
        samps = fit_sv_factor(
            innovations[:, j],
            num_warmup=num_warmup,
            num_samples=num_samples,
            rng_key=rng_key,
            progress_bar=progress_bar,
            robust=robust,
        )
        samples_list.append(samps)
    return samples_list


# ---------------------------------------------------------------------------
# 4. Relative volatility and prediction interval
# ---------------------------------------------------------------------------

def compute_sv_relative_vol(
    sv_samples_list: list[dict[str, np.ndarray]],
) -> np.ndarray:
    """Compute posterior distribution of the next-period relative volatility.

    The per-factor relative volatility is

        r_j^{(s)} = exp(h_{j,next}^{(s)} / 2) / E_t[exp(h_{j,t} / 2)]

    and the aggregate ``r`` returned here is a **variance-share weighted
    average** across factors, with weights equal to each factor's
    historical innovation variance (proxied by the squared posterior-mean
    scale ``_scale`` stored at fit time):

        w_j  ∝ scale_j²
        r^{(s)}  =  Σ_j  w_j · r_j^{(s)}     (Σ_j w_j = 1)

    Because h_next is now sampled from its predictive distribution
    N(μ + ρ(h_T − μ), σ_η), these draws correctly reflect the one-step-ahead
    uncertainty in log-volatility rather than just the conditional mean.

    Returns
    -------
    rel_vol_draws : 1-D array of length n_samples — posterior draws of r.
    """
    if not sv_samples_list:
        return np.array([])

    per_factor: list[np.ndarray] = []
    weights: list[float] = []
    for samps in sv_samples_list:
        h_all  = samps["h_all"]
        h_next = samps["h_next"]

        hist_avg_vol = np.mean(np.exp(0.5 * h_all), axis=1)
        next_vol     = np.exp(0.5 * h_next)
        rel          = next_vol / np.clip(hist_avg_vol, 1e-10, None)
        per_factor.append(rel)

        scale = float(samps.get("_scale", np.array([1.0]))[0])
        weights.append(scale ** 2)

    R = np.stack(per_factor, axis=0)                 # (k, S)
    w = np.asarray(weights, dtype=float)
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    return (w[:, None] * R).sum(axis=0)              # (S,)


# ---------------------------------------------------------------------------
# 4b. Model-consistent SV prediction interval
# ---------------------------------------------------------------------------

def kalman_gdp_predictive_sd(result: Any, target_month: pd.Timestamp) -> float:
    """Return the DFM's own predictive standard deviation for GDP at *target_month*.

    Uses ``DynamicFactorMQResults.get_prediction`` (statsmodels), which exposes
    the Kalman one-step-ahead / smoothed predictive variance for the missing
    GDP cell. This is the model-consistent equivalent of σ_em in Bańbura,
    Giannone, Modugno & Reichlin (2013) "Now-casting and the real-time data
    flow", Handbook of Economic Forecasting, vol. 2A, §3.

    Falls back to NaN if the variance cannot be retrieved (rare; very small
    panels at the start of the sample).
    """
    try:
        pred = result.get_prediction()
        gdp_col = pred.predicted_mean.columns[-1]
        se = pred.se_mean[gdp_col]
        return float(se.loc[target_month])
    except Exception:
        try:
            cov = result.predicted_state_cov  # (k_states, k_states, T)
            return float(np.sqrt(np.nanmean(np.diagonal(cov, axis1=0, axis2=1))))
        except Exception:
            return float("nan")


def build_sv_prediction_interval_v2(
    em_nowcast: float,
    result: Any,
    origin: pd.Period | str,
    sv_samples_list: list[dict[str, np.ndarray]],
    credibility: float = 0.9,
    sigma_floor: float = 0.05,
) -> tuple[float, float, float, float, float]:
    """Model-consistent SV-scaled prediction interval.

    Construction
    ------------
    1. σ_em = Kalman predictive standard deviation of GDP at the target month
       (from ``result.get_prediction().se_mean``). This is the model's *own*
       uncertainty about the missing-GDP cell, propagating factor uncertainty
       and idiosyncratic variance through the state-space.
    2. r* = posterior mean of the SV relative-volatility multiplier
       (``compute_sv_relative_vol``). r* = 1 in average regimes; > 1 in
       high-volatility regimes such as 2008-09 and 2020.
    3. σ_pred = σ_em · √r*. Interval = nowcast ± z_{α/2} · σ_pred.

    Returns
    -------
    (lower, em_nowcast, upper, rel_vol_mean, sigma_em)
    """
    from scipy import stats
    from .nowcast_utils import get_current_quarter, quarter_end_timestamp

    origin = pd.Period(origin)
    target_month = quarter_end_timestamp(get_current_quarter(origin))

    sigma_em = kalman_gdp_predictive_sd(result, target_month)
    if not np.isfinite(sigma_em) or sigma_em < sigma_floor:
        sigma_em = sigma_floor

    rel_vol_draws = compute_sv_relative_vol(sv_samples_list)
    rel_vol_mean = float(np.mean(rel_vol_draws))
    sigma_pred = sigma_em * np.sqrt(max(rel_vol_mean, 1e-8))

    z = float(stats.norm.ppf((1.0 + credibility) / 2.0))
    lower = em_nowcast - z * sigma_pred
    upper = em_nowcast + z * sigma_pred
    return lower, em_nowcast, upper, rel_vol_mean, float(sigma_em)


# ---------------------------------------------------------------------------
# 5. Volatility diagnostics  (for thesis plots)
# ---------------------------------------------------------------------------

def build_sv_volatility_df(
    sv_samples_list: list[dict[str, np.ndarray]],
    time_index: pd.DatetimeIndex | pd.PeriodIndex | None = None,
) -> pd.DataFrame:
    """Return a tidy DataFrame of posterior log-volatility per factor.

    Useful for plotting the time path of factor-level uncertainty.

    Parameters
    ----------
    sv_samples_list : list of MCMC sample dicts (one per factor).
    time_index      : optional time labels for the T_inno time points
                      (e.g. monthly DatetimeIndex). If None, integers are used.

    Returns
    -------
    pd.DataFrame with columns:
        factor, t, h_mean, h_lo5, h_hi95, vol_mean
    """
    records: list[dict] = []
    for j, samps in enumerate(sv_samples_list):
        h_all = samps["h_all"]              # (S, T_inno)
        T_inno = h_all.shape[1]

        h_mean = np.mean(h_all, axis=0)
        h_lo5  = np.percentile(h_all,  5, axis=0)
        h_hi95 = np.percentile(h_all, 95, axis=0)
        vol    = np.mean(np.exp(0.5 * h_all), axis=0)

        for t in range(T_inno):
            record: dict = {
                "factor":   j + 1,
                "t":        time_index[t] if time_index is not None else t,
                "h_mean":   float(h_mean[t]),
                "h_lo5":    float(h_lo5[t]),
                "h_hi95":   float(h_hi95[t]),
                "vol_mean": float(vol[t]),
            }
            records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 6. Single-origin SV nowcast
# ---------------------------------------------------------------------------

def nowcast_with_sv(
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
) -> dict:
    """Produce a Bayesian SV-enhanced DFM nowcast for a single origin.

    Runs the two-stage pipeline:
      1. EM-DFM (via nowcast_utils.fit_dfm) → point nowcast + smoothed factors.
         Real-time masking and AR(p) BIC fill are applied via build_dfm_endog
         when pub_lag_map is supplied (fill_method controls fill behaviour).
      2. Bayesian SV on factor VAR innovations → SV-adjusted credible interval.

    Parameters
    ----------
    X_monthly        : full monthly predictor panel.
    y_quarterly      : quarterly GDP first-release series.
    selected_cols    : indicator column names to use.
    origin           : monthly forecast origin.
    k_factors        : number of global factors in the EM-DFM (default 2).
        Main SV sensitivity runs use ``k_factors`` in {1, 2, 3} via separate
        calls to ``run_actpn_nowcast_loop_sv``.
    factor_order     : VAR order for factor dynamics.
    idiosyncratic_ar1: include AR(1) idiosyncratic components.
    maxiter          : EM iterations.
    pub_lag_map      : pd.Series mapping column id → publication lag in months.
    fill_method      : 'ar_bic' (default) — AR(p) BIC fill before DFM; 'none'.
    ar_max_p         : maximum AR order for BIC selection.
    ar_min_train     : minimum training observations for AR(1+).
    num_warmup       : NUTS warm-up steps per factor.
    num_samples      : posterior draws per factor.
    credibility      : nominal PI coverage (default 0.9).
    sigma_floor      : minimum σ_em in pp.
    rng_seed         : reproducibility seed.
    progress_bar     : show numpyro MCMC progress bar.
    robust           : if True, use Student-t SV (_sv_model_robust).

    Returns
    -------
    dict with keys: origin, current_quarter, k_factors, n_indicators,
        nowcast (EM point), ci_lower, ci_upper, rel_vol, sigma_em, _sv_samples.
    """
    from .nowcast_utils import (
        build_dfm_endog,
        fit_dfm,
        extract_nowcast,
        get_current_quarter,
    )

    origin = pd.Period(origin)

    if len(selected_cols) == 0:
        raise ValueError(f"No selected columns for origin {origin}.")

    X_sel = X_monthly[selected_cols]

    # ── Stage 1: EM-DFM with publication-lag-aware information set ──────────
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
    em_nowcast = extract_nowcast(result, origin)

    # ── Stage 2: Bayesian SV on factor innovations ───────────────────────────
    _, innovations = extract_factor_innovations(result, factor_order=factor_order)

    sv_samples_list = fit_sv_all_factors(
        innovations,
        num_warmup=num_warmup,
        num_samples=num_samples,
        rng_seed=rng_seed,
        progress_bar=progress_bar,
        robust=robust,
    )

    # ── Prediction interval (model-consistent Kalman PI scaled by SV) ───────
    lower, _, upper, rel_vol, sigma_em = build_sv_prediction_interval_v2(
        em_nowcast=em_nowcast,
        result=result,
        origin=origin,
        sv_samples_list=sv_samples_list,
        credibility=credibility,
        sigma_floor=sigma_floor,
    )

    return {
        "origin":          str(origin),
        "current_quarter": str(get_current_quarter(origin)),
        "k_factors":       k_factors,
        "n_indicators":    len(selected_cols),
        "nowcast":         em_nowcast,
        "ci_lower":        lower,
        "ci_upper":        upper,
        "rel_vol":         rel_vol,
        "sigma_em":        sigma_em,
        "_sv_samples":     sv_samples_list,
    }


# ---------------------------------------------------------------------------
# 7. Expanding nowcast loop with SV  (mirrors run_actpn_nowcast_loop)
# ---------------------------------------------------------------------------

def run_actpn_nowcast_loop_sv(
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
    save_path: str | Path | None = None,
    verbose: bool = True,
    robust: bool = False,
) -> pd.DataFrame:
    """Expanding A-CD-TPN nowcast loop with Bayesian SV — M1, M2, M3 per quarter.

    Mirrors ``nowcast_utils.run_actpn_nowcast_loop`` exactly, including the
    publication-lag masking and AR(p) BIC fill via ``pub_lag_map``.

    For each quarterly origin q and month-in-quarter m ∈ {1, 2, 3}:
      1. Load selected columns from selection_matrix at the monthly origin.
      2. Run Stage-1 EM-DFM with real-time masking + AR fill → point nowcast.
      3. Run Stage-2 Bayesian SV on factor innovations → CI.

    Parameters
    ----------
    selection_matrix  : binary DataFrame (monthly origins × series).
    X_monthly         : full monthly predictor panel.
    y_quarterly       : quarterly GDP first-release series.
    quarterly_origins : explicit quarterly PeriodIndex to evaluate.
    factor_order      : VAR order p for factor dynamics.
    idiosyncratic_ar1 : include AR(1) idiosyncratic components.
    maxiter           : EM iterations.
    k_factors         : fixed EM factor count (default 2). Pass 1, 2, or 3 for
                        the SV sensitivity specifications.
    pub_lag_map       : pd.Series mapping column id → publication lag (months).
    fill_method       : 'ar_bic' (default) — AR(p) BIC fill before DFM; 'none'.
    ar_max_p          : maximum AR order for BIC selection.
    ar_min_train      : minimum training observations for AR(1+).
    num_warmup        : NUTS warm-up steps per factor per origin (default 500).
    num_samples       : posterior draws per factor per origin (default 1000).
    credibility       : nominal PI coverage (default 0.9 → 90 %).
    sigma_floor       : minimum σ_em in pp.
    rng_seed          : base seed; each origin adds its sequential index.
    save_path         : if provided, intermediate results appended to CSV after
                        every successful origin (crash recovery).
    verbose           : print per-quarter progress.
    robust            : if True, use Student-t SV (_sv_model_robust).

    Returns
    -------
    pd.DataFrame indexed by quarter with columns:
        quarter, monthly_origin, month_in_quarter, n_indicators, k_factors,
        nowcast, actual, error, ci_lower_{cov}, ci_upper_{cov}, rel_vol, sigma_em.
    """
    from .nowcast_utils import quarter_end_timestamp

    if pub_lag_map is None:
        warnings.warn(
            "pub_lag_map is None: SV loop uses balanced-panel masking. "
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
                    f" — EM + SV{robust_tag} ...",
                    end=" ", flush=True,
                )

            try:
                res = nowcast_with_sv(
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
                )

                error = res["nowcast"] - actual if not np.isnan(actual) else np.nan

                row = {
                    "quarter":               str(q),
                    "monthly_origin":        origin_key,
                    "month_in_quarter":      m_in_q,
                    "n_indicators":          res["n_indicators"],
                    "k_factors":             res["k_factors"],
                    "nowcast":               res["nowcast"],
                    "actual":                actual,
                    "error":                 error,
                    f"ci_lower_{cov_label}": res["ci_lower"],
                    f"ci_upper_{cov_label}": res["ci_upper"],
                    "rel_vol":               res["rel_vol"],
                    "sigma_em":              res.get("sigma_em", np.nan),
                }
                records.append(row)

                if verbose:
                    print(
                        f"k={res['k_factors']}  nowcast={res['nowcast']:.3f}  "
                        f"actual={actual:.3f}  "
                        f"CI=[{res['ci_lower']:.3f}, {res['ci_upper']:.3f}]  "
                        f"rel_vol={res['rel_vol']:.2f}"
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
                    "actual":                actual,
                    "error":                 np.nan,
                    f"ci_lower_{cov_label}": np.nan,
                    f"ci_upper_{cov_label}": np.nan,
                    "rel_vol":               np.nan,
                    "sigma_em":              np.nan,
                })

            global_idx += 1

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.set_index("monthly_origin")
    return df
