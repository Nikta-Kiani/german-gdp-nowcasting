"""Post-COVID benchmark models and bias corrections for German GDP nowcasting.

Motivation
----------
The expanding-window evaluation anchors every model's level to the long-run
(1991-) mean QoQ growth (~0.33 pp). After 2022 German GDP stagnates near zero
(mean 0.03 pp, std 0.19 pp). A bias/variance decomposition of the M3 errors
shows that ~70% of the AR(1) post-COVID MSE is a *level* error, not a signal
error — and that every richer model (DFM, SV, XGB, MLP) lies *above* the
no-skill line RMSFE = std(GDP). The binding problem in the post-COVID regime is
therefore the stale intercept after a structural break, not signal extraction.

This module adds models/corrections that target exactly that, *all derived from
the GDP target and the already-saved nowcast CSVs* (no model re-runs), and all
strictly real-time (each quarter q uses information dated < q only):

  - rolling_mean_benchmark      Local-mean (recent unconditional mean) forecast.
  - local_level_uc              Unobserved-components local-level (Stock-Watson
                                2007) recursive 1-step trend forecast.
  - rolling_ar1                 AR(1) on a fixed rolling window (vs expanding).
  - intercept_correct           Clements-Hendry recursive intercept correction
                                applied to any existing model's nowcast path.
  - combine_forecasts           Equal-weight and inverse-MSE forecast
                                combinations of existing model paths.

References
----------
Clements, M. P. & Hendry, D. F. (1998, 1999). Forecasting Economic Time Series /
    Forecasting Non-stationary Economic Time Series. CUP / MIT Press.
    (intercept correction after structural breaks)
Stock, J. H. & Watson, M. W. (2007). Why has U.S. inflation become harder to
    forecast? Journal of Money, Credit and Banking, 39(s1), 3-33. (UC-SV)
Stock, J. H. & Watson, M. W. (2004). Combination forecasts of output growth in a
    seven-country data set. Journal of Forecasting, 23(6), 405-430.
Timmermann, A. (2006). Forecast combinations. Handbook of Economic Forecasting,
    vol. 1, 135-196. Elsevier.
Pesaran, M. H. & Timmermann, A. (2007). Selection of estimation window in the
    presence of breaks. Journal of Econometrics, 137(1), 134-161.
Muth, J. F. (1960). Optimal properties of exponentially weighted forecasts. JASA.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Target + existing-model loaders
# ---------------------------------------------------------------------------

def load_gdp_target(path: str | Path) -> pd.Series:
    """Load the first-release GDP QoQ log-growth target as a quarterly Series."""
    df = pd.read_csv(path)
    col = [c for c in df.columns if c != "quarter"][0]
    s = pd.Series(
        df[col].to_numpy(dtype=float),
        index=pd.PeriodIndex(df["quarter"].astype(str), freq="Q"),
        name="gdp",
    )
    return s.sort_index()


def load_model_m3(path: str | Path) -> pd.Series:
    """Load a saved nowcast CSV and return its M3 nowcast as a quarterly Series.

    Quarterly-only CSVs (AR1/RW) are returned as-is; CSVs with a
    ``month_in_quarter`` column are filtered to M3 (end-of-quarter information
    set, the headline nowcast).
    """
    df = pd.read_csv(path)
    if "month_in_quarter" in df.columns:
        df = df[df["month_in_quarter"] == 3]
    idx = pd.PeriodIndex(df["quarter"].astype(str), freq="Q")
    return pd.Series(df["nowcast"].to_numpy(dtype=float), index=idx, name="nowcast").sort_index()


# ---------------------------------------------------------------------------
# Real-time benchmarks (target-only)
# ---------------------------------------------------------------------------

def rolling_mean_benchmark(
    y: pd.Series,
    origins: pd.PeriodIndex,
    window: int | None = 8,
    min_obs: int = 4,
) -> pd.Series:
    """Recent-local-mean forecast: y_hat_q = mean(y over the last `window`
    quarters strictly before q).

    With ``window=None`` this is the expanding unconditional mean. A short window
    tracks regime shifts (post-COVID stagnation) without re-estimating any model
    (Pesaran & Timmermann 2007).
    """
    out = {}
    for q in origins:
        past = y.loc[y.index < q].dropna()
        if window is not None:
            past = past.iloc[-window:]
        out[q] = float(past.mean()) if len(past) >= min_obs else np.nan
    return pd.Series(out, name=f"local_mean_w{window}").sort_index()


def rolling_ar1(
    y: pd.Series,
    origins: pd.PeriodIndex,
    window: int = 40,
    min_obs: int = 16,
) -> pd.Series:
    """AR(1) estimated on a fixed rolling window of the last `window` quarters.

    Contrast with the expanding-window AR(1): a rolling window discards the
    pre-break high-growth sample, reducing the post-2022 upward level bias
    (Pesaran & Timmermann 2007; Giraitis, Kapetanios & Price 2013).
    """
    out = {}
    for q in origins:
        past = y.loc[y.index < q].dropna()
        past = past.iloc[-window:]
        if len(past) < min_obs:
            out[q] = np.nan
            continue
        yv = past.to_numpy()
        y_t, y_l = yv[1:], yv[:-1]
        X = np.column_stack([np.ones_like(y_l), y_l])
        beta, *_ = np.linalg.lstsq(X, y_t, rcond=None)
        out[q] = float(beta[0] + beta[1] * yv[-1])
    return pd.Series(out, name=f"rolling_ar1_w{window}").sort_index()


def local_level_uc(
    y: pd.Series,
    origins: pd.PeriodIndex,
    min_obs: int = 20,
) -> pd.Series:
    """Unobserved-components local-level (random-walk trend) 1-step forecast.

    y_t = tau_t + eps_t ,  tau_t = tau_{t-1} + eta_t  (Stock & Watson 2007).
    The signal-to-noise ratio is re-estimated by MLE on the expanding sample at
    each origin; the point forecast equals the filtered trend tau_{q-1|q-1}.
    This automatically re-centres to the recent regime (the optimal forecast is
    an EWMA of past growth, Muth 1960), which is the key device against the
    post-break level bias.
    """
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    out = {}
    for q in origins:
        past = y.loc[y.index < q].dropna()
        if len(past) < min_obs:
            out[q] = np.nan
            continue
        yv = past.to_numpy(dtype=float)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = UnobservedComponents(yv, level="local level")
                res = mod.fit(disp=False, maxiter=100)
                out[q] = float(np.asarray(res.forecast(1))[0])
        except Exception:
            out[q] = float(yv[-min(8, len(yv)):].mean())
    return pd.Series(out, name="local_level_uc").sort_index()


# ---------------------------------------------------------------------------
# Corrections / combinations of existing model paths
# ---------------------------------------------------------------------------

def intercept_correct(
    nowcast: pd.Series,
    y: pd.Series,
    window: int = 4,
    min_obs: int = 2,
) -> pd.Series:
    """Clements-Hendry recursive intercept correction of an existing nowcast.

    corrected_q = nowcast_q - mean( error_s : s in last `window` quarters < q )
    where error_s = nowcast_s - y_s (first-release actual, known by q in real
    time). Removes a slowly-moving systematic bias without re-estimating the
    model. Real-time valid: only past, already-released errors are used.
    """
    err = (nowcast - y.reindex(nowcast.index)).rename("error")
    out = {}
    for q in nowcast.index:
        if pd.isna(nowcast.get(q, np.nan)):
            out[q] = np.nan
            continue
        past_err = err.loc[err.index < q].dropna().iloc[-window:]
        bias = float(past_err.mean()) if len(past_err) >= min_obs else 0.0
        out[q] = float(nowcast[q] - bias)
    return pd.Series(out, name=f"{nowcast.name}_ic").sort_index()


def combine_forecasts(
    paths: dict[str, pd.Series],
    y: pd.Series,
    method: str = "equal",
    window: int = 12,
    min_obs: int = 6,
) -> pd.Series:
    """Combine several model nowcast paths.

    method='equal'   : equal-weight average (robust; Timmermann 2006).
    method='inv_mse' : weights proportional to 1 / recent MSE computed on the
                       last `window` quarters strictly before q (real-time;
                       Stock & Watson 2004). Falls back to equal weights until
                       `min_obs` past errors are available.
    """
    names = list(paths)
    common = None
    for s in paths.values():
        common = s.index if common is None else common.union(s.index)
    common = pd.PeriodIndex(sorted(common), freq="Q")
    M = pd.DataFrame({n: paths[n].reindex(common) for n in names})
    yv = y.reindex(common)

    out = {}
    for q in common:
        row = M.loc[q].dropna()
        if row.empty:
            out[q] = np.nan
            continue
        if method == "equal":
            out[q] = float(row.mean())
            continue
        w = {}
        for n in row.index:
            e = (M[n] - yv).loc[M.index < q].dropna().iloc[-window:]
            w[n] = 1.0 / float(np.mean(e.to_numpy() ** 2)) if len(e) >= min_obs else np.nan
        wser = pd.Series(w)
        if wser.notna().sum() == 0:
            out[q] = float(row.mean())
        else:
            wser = wser.reindex(row.index).fillna(wser.mean())
            wser = wser / wser.sum()
            out[q] = float((row * wser).sum())
    return pd.Series(out, name=f"combo_{method}").sort_index()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

REGIMES = {
    "pre-COVID": ("2011Q1", "2019Q4"),
    "COVID": ("2020Q1", "2021Q4"),
    "post-COVID": ("2022Q1", "2025Q4"),
}


def regime_metrics(nowcast: pd.Series, y: pd.Series) -> dict:
    """RMSFE and bias of a nowcast path per regime, plus full-sample RMSFE."""
    err = (nowcast - y.reindex(nowcast.index)).dropna()
    res = {}
    for lab, (a, b) in REGIMES.items():
        m = (err.index >= pd.Period(a)) & (err.index <= pd.Period(b))
        e = err[m]
        if len(e):
            res[f"{lab}_rmsfe"] = round(float(np.sqrt(np.mean(e.to_numpy() ** 2))), 3)
            res[f"{lab}_bias"] = round(float(e.mean()), 3)
    res["all_rmsfe"] = round(float(np.sqrt(np.mean(err.to_numpy() ** 2))), 3)
    return res


def no_skill_rmsfe(y: pd.Series, start: str, end: str) -> float:
    """Std of GDP over a window = RMSFE of the (infeasible) in-window mean
    forecast; the NSR=1 'no-skill' threshold (Lehmann et al. 2020)."""
    s = y.loc[(y.index >= pd.Period(start)) & (y.index <= pd.Period(end))].dropna()
    return float(s.std(ddof=1))
