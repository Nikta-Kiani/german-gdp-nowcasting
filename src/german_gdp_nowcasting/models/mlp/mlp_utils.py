"""Factor-augmented MLP nowcast for German GDP (non-linear factor->GDP benchmark).

Scientific question
-------------------
Once the 585-series panel is compressed to the DFM-EN's k=2 estimated factors,
does a non-linear MLP factor->GDP mapping improve on the linear DFM measurement
equation?  A tie or loss is equally publishable: it confirms that the
factor->GDP link is effectively linear at the available sample size and
signal-to-noise ratio.

Design (fixed-hyperparameter convention -- mirrors XGB-Full / DFM-EN)
--------------------------------------------------------------------
* Factor source: the headline DFM-EN spec (``en_only`` selection matrix, k=2,
  factor_order=2, idiosyncratic_ar1=True) -- imported read-only; DFM code is
  never modified.
* Real time: at each M3 quarterly origin q, rebuild the real-time masked +
  AR(p)-filled ``en_only`` panel via ``build_dfm_endog``, fit ``DynamicFactorMQ``,
  read ``result.factors.smoothed`` (monthly), aggregate monthly->quarterly
  (mean).  Training and prediction rows both come from the SAME DFM fit, so sign
  indeterminacy / cross-origin rotation flips are irrelevant.
* Features: F1_L0, F1_L1, F1_L2, F2_L0, F2_L1, F2_L2  (6 features).
  No GDP autoregressive lags -- keeps the test "factors only -> GDP" pure.
* Architecture: ``MLPRegressor``, 1 hidden layer, tanh, lbfgs, StandardScaler.
  Seed-averaged over 5 random initialisations to tame small-net variance.
* Tuning once on the pre-eval window (DFM fit at 2011-03, factors 1991Q1-2010Q4)
  via TimeSeriesSplit CV on a small HP grid; frozen for all 60 eval origins
  (same fixed-design convention as XGB-Full).
* Eval window: M3 only, 2011Q1-2025Q4  (same as every other model).

Outputs (canonical, under ``outputs/nowcasting/``)
--------------------------------------------------
  nowcast_results_mlp_factor.csv   (M3 format, same schema as XGB)
  mlp_factor_best_params.json      (frozen HP)
  mlp_factor_cache.parquet         (per-origin quarterly factors)

Usage
-----
  python -m german_gdp_nowcasting.models.mlp.mlp_utils
  python -m german_gdp_nowcasting.models.mlp.mlp_utils --force
"""

from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from ...config import paths as P
from ..dfm.nowcast_utils import build_dfm_endog, fit_dfm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_START   = pd.Period("2011Q1", "Q")
EVAL_END     = pd.Period("2025Q4", "Q")
TUNE_ORIGIN  = "2011-03"          # M3 of 2011Q1 -- first eval origin; factors
                                  # 1991Q1-2011Q1 available; tuning CV uses only
                                  # the pre-eval slice 1991Q1-2010Q4 (target_q=2011Q1)
K_FACTORS    = 2
FACTOR_ORDER = 2
N_LAGS       = 2                  # lags 0..2 -> 3 x 2 factors = 6 features
TRAIN_START  = pd.Period("1991Q1", "Q")
SEEDS        = [0, 1, 2, 3, 4]
HEADLINE_MIQ = 3
CV_SPLITS    = 5

# HP grid: 2 sizes x 4 regularisation strengths
HP_GRID: list[tuple[int, float]] = [
    (hidden, alpha)
    for hidden in (8, 16)
    for alpha in (1e-2, 1e-1, 1.0, 10.0)
]

FACTOR_CACHE = P.MLP_FACTOR_CACHE_PARQUET
RESULTS_CSV  = P.MLP_FACTOR_RESULTS_CSV
PARAMS_JSON  = P.MLP_FACTOR_BEST_PARAMS_JSON


# ===========================================================================
# 1.  Data loading
# ===========================================================================

def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Load monthly panel, pub-lag map, quarterly GDP, en_only selection matrix."""
    print("Loading data …")
    X = pd.read_csv(P.PANEL_TRANSFORMED_CSV, index_col=0, parse_dates=True)
    X.index = pd.DatetimeIndex(X.index, freq="MS")

    lag_raw = pd.read_csv(P.PUB_LAG_CSV, index_col=0).squeeze()
    pub_lag = lag_raw.rename_axis(None).astype(int)

    g = pd.read_csv(P.GDP_TARGET_CSV)
    g["quarter"] = pd.PeriodIndex(g["quarter"], freq="Q")
    y = g.set_index("quarter").iloc[:, 0]

    sel = pd.read_csv(P.EN_ONLY_MATRIX_CSV, index_col=0)
    print(
        f"  panel {X.shape}  pub_lag {pub_lag.shape}  "
        f"GDP {len(y)} quarters  sel_mat {sel.shape}"
    )
    return X, pub_lag, y, sel


# ===========================================================================
# 2.  Factor extraction  (wraps read-only DFM functions)
# ===========================================================================

def m3_of(q: pd.Period) -> str:
    """Return the M3 monthly-origin string for a quarterly Period.
    e.g. Period('2011Q1') -> '2011-03'
    """
    return str(q.asfreq("M", how="end"))


def cols_at(sel: pd.DataFrame, origin: str) -> list[str]:
    """Selected indicator columns for a given monthly origin string."""
    if origin not in sel.index:
        return []
    return sel.columns[sel.loc[origin].astype(bool)].tolist()


def extract_quarterly_factors(
    origin: str,
    selected: list[str],
    X: pd.DataFrame,
    y: pd.Series,
    pub_lag: pd.Series,
) -> pd.DataFrame | None:
    """Fit DFM-EN at one M3 origin; return quarterly-mean smoothed factors.

    Returns a DataFrame indexed by quarterly Period (freq='Q') with columns
    F1, F2 (or however many factors), or None if the fit fails.
    """
    if not selected:
        warnings.warn(f"{origin}: no indicators — skipped.", RuntimeWarning)
        return None

    op = pd.Period(origin, freq="M")
    sub_lag = pub_lag.reindex(selected).fillna(0).astype(int)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            endog, km = build_dfm_endog(
                X[selected], y, op,
                pub_lag_map=sub_lag,
                fill_method="ar_bic",
            )
            res = fit_dfm(
                endog, km,
                k_factors=K_FACTORS,
                factor_order=FACTOR_ORDER,
                idiosyncratic_ar1=True,
            )
    except Exception as exc:
        warnings.warn(f"{origin}: DFM fit failed — {exc}", RuntimeWarning)
        return None

    # Monthly smoothed factors -> quarterly mean
    # result.factors.smoothed: DatetimeIndex x k columns (string '0','1',...)
    fm = res.factors.smoothed.copy()
    fm.columns = [f"F{i + 1}" for i in range(fm.shape[1])]
    qi = pd.PeriodIndex(fm.index.to_period("Q"), freq="Q")
    fq = fm.groupby(qi).mean()
    fq.index.name = "quarter"
    return fq


# ---------------------------------------------------------------------------
# Cache I/O (single parquet; incremental: each origin is saved immediately)
# ---------------------------------------------------------------------------

def _load_cache() -> tuple[pd.DataFrame | None, set[str]]:
    """Return (raw_df, set_of_cached_origins) or (None, empty_set)."""
    if not FACTOR_CACHE.exists():
        return None, set()
    raw = pd.read_parquet(FACTOR_CACHE)
    return raw, set(raw["origin"].unique())


def _append_to_cache(
    existing: pd.DataFrame | None,
    origin: str,
    fq: pd.DataFrame | None,
    n_ind: int,
) -> pd.DataFrame:
    """Append one origin's factors to the accumulator DataFrame; return updated."""
    if fq is not None:
        tmp = fq.reset_index()                    # quarter | F1 | F2
        tmp["origin"] = origin
        tmp["n_indicators"] = n_ind
        tmp["quarter"] = tmp["quarter"].astype(str)
    else:
        # Sentinel row so the origin is marked 'processed but failed'
        tmp = pd.DataFrame([{
            "quarter": "__failed__",
            "F1": np.nan, "F2": np.nan,
            "origin": origin,
            "n_indicators": n_ind,
        }])
    combined = pd.concat([existing, tmp], ignore_index=True) if existing is not None else tmp
    combined.to_parquet(FACTOR_CACHE, index=False)
    return combined


def _cache_to_dict(raw: pd.DataFrame) -> dict[str, pd.DataFrame | None]:
    """Reconstruct cache dict: origin -> quarterly factor DataFrame (or None)."""
    cache: dict[str, pd.DataFrame | None] = {}
    for origin, grp in raw.groupby("origin"):
        if "__failed__" in grp["quarter"].values or grp["quarter"].eq("__failed__").all():
            cache[origin] = None
        else:
            sub = grp[grp["quarter"] != "__failed__"].copy()
            factor_cols = [c for c in sub.columns if c.startswith("F") and c[1:].isdigit()]
            fdf = sub[["quarter"] + factor_cols].set_index("quarter")
            fdf.index = pd.PeriodIndex(fdf.index, freq="Q")
            fdf.index.name = "quarter"
            cache[origin] = fdf
    return cache


def build_or_load_cache(
    all_origins: list[str],
    X: pd.DataFrame,
    y: pd.Series,
    sel: pd.DataFrame,
    pub_lag: pd.Series,
    force: bool = False,
) -> dict[str, pd.DataFrame | None]:
    """Build (or incrementally extend) the factor cache; return dict."""
    if force and FACTOR_CACHE.exists():
        FACTOR_CACHE.unlink()
        print("  --force: deleted existing cache.")

    raw, done = _load_cache()
    pending = [o for o in all_origins if o not in done]

    if not pending:
        print(f"Factor cache complete ({len(done)} origins). Loading …")
        return _cache_to_dict(raw)

    print(
        f"Factor cache: {len(done)}/{len(all_origins)} done. "
        f"Running {len(pending)} DFM fits …"
    )

    for i, origin in enumerate(pending, 1):
        selected = cols_at(sel, origin)
        tag = f"[{len(done) + i:3d}/{len(all_origins)}]"
        print(f"  {tag} {origin}  N={len(selected)} …", end=" ", flush=True)
        fq = extract_quarterly_factors(origin, selected, X, y, pub_lag)
        raw = _append_to_cache(raw, origin, fq, len(selected))
        if fq is not None:
            print(f"ok  ({len(fq)} quarters)")
        else:
            print("FAILED")

    print(f"Factor cache saved → {FACTOR_CACHE.name}")
    return _cache_to_dict(raw)


# ===========================================================================
# 3.  Feature engineering
# ===========================================================================

def make_lag_features(fq: pd.DataFrame, n_lags: int = N_LAGS) -> pd.DataFrame:
    """Build contemporaneous + lag-1 + lag-2 panel for each factor column.

    Returns DataFrame with columns F1_L0, F1_L1, F1_L2, F2_L0, F2_L1, F2_L2.
    """
    parts: dict[str, pd.Series] = {}
    for lag in range(n_lags + 1):
        for col in fq.columns:
            parts[f"{col}_L{lag}"] = fq[col].shift(lag)
    return pd.DataFrame(parts, index=fq.index)


def train_pred_arrays(
    fq: pd.DataFrame,
    y: pd.Series,
    target_q: pd.Period,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (X_train, y_train, x_pred, n_train).

    Training rows: [TRAIN_START, target_q - 1] with all features and GDP obs.
    Prediction row: single row at target_q (lag features from same DFM fit).
    """
    feat = make_lag_features(fq)

    mask = (feat.index >= TRAIN_START) & (feat.index < target_q)
    Xf = feat.loc[mask]
    yf = y.reindex(Xf.index)

    valid = Xf.notna().all(axis=1) & yf.notna()
    Xt = Xf.loc[valid].values.astype(float)
    yt = yf.loc[valid].values.astype(float)

    if target_q not in feat.index:
        raise KeyError(f"target_q {target_q} not in factor panel "
                       f"(max={feat.index.max()})")
    xp = feat.loc[[target_q]].values.astype(float)
    return Xt, yt, xp, int(valid.sum())


# ===========================================================================
# 4.  MLP helpers
# ===========================================================================

def _make_mlp(hidden: int, alpha: float, seed: int) -> MLPRegressor:
    """Construct the LBFGS MLP; this solver does not use early stopping."""
    return MLPRegressor(
        hidden_layer_sizes=(hidden,),
        activation="tanh",
        solver="lbfgs",      # optimal for small datasets; L2 via alpha
        alpha=alpha,
        max_iter=2000,
        random_state=seed,
    )


def cv_mse(Xt: np.ndarray, yt: np.ndarray, hidden: int, alpha: float) -> float:
    """Mean TimeSeriesSplit CV MSE with per-fold StandardScaling."""
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    mses: list[float] = []
    for tr, va in tscv.split(Xt):
        sc = StandardScaler().fit(Xt[tr])
        m = _make_mlp(hidden, alpha, seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(sc.transform(Xt[tr]), yt[tr])
        mses.append(float(np.mean((m.predict(sc.transform(Xt[va])) - yt[va]) ** 2)))
    return float(np.mean(mses))


def tune_mlp(Xt: np.ndarray, yt: np.ndarray) -> dict:
    """Grid search over HP_GRID; print progress; return best {hidden, alpha}."""
    print(f"  Grid search: {len(HP_GRID)} configurations …")
    best_mse: float = np.inf
    best: dict = {}
    for hidden, alpha in HP_GRID:
        score = cv_mse(Xt, yt, hidden, alpha)
        flag = " *" if score < best_mse else ""
        print(f"    hidden={hidden:2d}  alpha={alpha:6.3f}  CV-MSE={score:.5f}{flag}")
        if score < best_mse:
            best_mse, best = score, {"hidden": hidden, "alpha": alpha}
    print(f"  Best: {best}   CV-MSE={best_mse:.5f}")
    return best


def seed_avg_predict(
    Xt: np.ndarray,
    yt: np.ndarray,
    xp: np.ndarray,
    hidden: int,
    alpha: float,
) -> float:
    """Fit with each seed, return mean prediction (tames init variance)."""
    sc = StandardScaler().fit(Xt)
    X_sc = sc.transform(Xt)
    xp_sc = sc.transform(xp)
    preds: list[float] = []
    for seed in SEEDS:
        m = _make_mlp(hidden, alpha, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_sc, yt)
        preds.append(float(m.predict(xp_sc)[0]))
    return float(np.mean(preds))


# ===========================================================================
# 5.  Main
# ===========================================================================

def _nan_row(q: pd.Period, origin: str, n_ind: int) -> dict:
    """Build a schema-compatible failed-nowcast record."""
    return {
        "quarter": str(q), "monthly_origin": origin,
        "month_in_quarter": HEADLINE_MIQ,
        "n_indicators": n_ind, "n_features": np.nan,
        "nowcast": np.nan, "actual": np.nan, "error": np.nan,
    }


def main() -> None:
    """Run factor extraction, frozen tuning, and expanding MLP nowcasts."""
    force = "--force" in sys.argv

    # -------------------------------------------------------------------
    # Load shared data
    # -------------------------------------------------------------------
    X, pub_lag, y, sel = load_data()

    eval_quarters = pd.period_range(EVAL_START, EVAL_END, freq="Q")
    eval_origins  = [m3_of(q) for q in eval_quarters]
    # TUNE_ORIGIN is "2011-03" = first eval origin; no need to add twice
    all_origins   = list(dict.fromkeys([TUNE_ORIGIN] + eval_origins))

    # -------------------------------------------------------------------
    # Phase 1 — factor cache (61 DFM fits; incremental / resumable)
    # -------------------------------------------------------------------
    cache = build_or_load_cache(all_origins, X, y, sel, pub_lag, force=force)

    # -------------------------------------------------------------------
    # Phase 2 — tune MLP ONCE on pre-eval factors (origin 2011-03)
    # -------------------------------------------------------------------
    if PARAMS_JSON.exists() and not force:
        print(f"\nLoading frozen HP from {PARAMS_JSON.name} …")
        with open(PARAMS_JSON) as fh:
            best_hp = json.load(fh)
    else:
        print(
            f"\nTuning MLP on pre-eval slice "
            f"(DFM fit at {TUNE_ORIGIN}, factors 1991Q1–2011Q1, "
            f"CV uses pre-eval rows 1991Q1–2010Q4) …"
        )
        fq_tune = cache.get(TUNE_ORIGIN)
        if fq_tune is None:
            raise RuntimeError(
                f"DFM fit at {TUNE_ORIGIN} failed. "
                "Check data or re-run with --force."
            )
        # target_q=2011Q1 -> training rows are [TRAIN_START, 2010Q4] (strictly pre-eval)
        Xt, yt, _, n_tune = train_pred_arrays(
            fq_tune, y, target_q=pd.Period("2011Q1", "Q")
        )
        print(f"  Tune window: {n_tune} obs, {Xt.shape[1]} features")
        best_hp = tune_mlp(Xt, yt)
        with open(PARAMS_JSON, "w") as fh:
            json.dump(best_hp, fh, indent=2)
        print(f"  Frozen HP saved → {PARAMS_JSON.name}")

    hidden = int(best_hp["hidden"])
    alpha  = float(best_hp["alpha"])
    print(f"\nFrozen MLP: hidden_layer_sizes=({hidden},)  alpha={alpha}  seeds={SEEDS}")

    # -------------------------------------------------------------------
    # Phase 3 — expanding-window nowcast loop (M3, 2011Q1–2025Q4)
    # -------------------------------------------------------------------
    if RESULTS_CSV.exists() and not force:
        print(
            f"\n{RESULTS_CSV.name} already exists — skipping loop. "
            "Use --force to rerun."
        )
        return

    print(f"\nNowcast loop: {EVAL_START} → {EVAL_END} ({len(eval_quarters)} quarters) …")
    records: list[dict] = []

    for q, origin in zip(eval_quarters, eval_origins):
        n_ind = len(cols_at(sel, origin))
        fq = cache.get(origin)
        print(f"  {q} ({origin})  N={n_ind} …", end=" ", flush=True)

        if fq is None:
            print("no cached factors → NaN")
            records.append(_nan_row(q, origin, n_ind))
            continue

        try:
            Xt, yt, xp, n_train = train_pred_arrays(fq, y, q)
            if n_train < 16:
                print(f"too few training rows ({n_train}) → NaN")
                records.append(_nan_row(q, origin, n_ind))
                continue

            nc = seed_avg_predict(Xt, yt, xp, hidden, alpha)
            ac = float(y.get(q, np.nan))
            er = nc - ac if not np.isnan(ac) else np.nan

            print(
                f"feat={Xt.shape[1]}  n_train={n_train}  "
                f"nc={nc:.3f}  ac={ac:.3f}"
            )
            records.append({
                "quarter": str(q), "monthly_origin": origin,
                "month_in_quarter": HEADLINE_MIQ,
                "n_indicators": n_ind, "n_features": Xt.shape[1],
                "nowcast": nc, "actual": ac, "error": er,
            })
        except Exception as exc:
            print(f"ERROR: {exc}")
            records.append(_nan_row(q, origin, n_ind))

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_CSV, index=False)
    valid_n = int(df["error"].notna().sum())
    rmsfe_all = float(np.sqrt((df["error"] ** 2).mean(skipna=True)))
    print(
        f"\nSaved → {RESULTS_CSV.name}  "
        f"({valid_n}/{len(df)} valid observations  "
        f"full-window RMSFE={rmsfe_all:.4f} pp)"
    )
    print("Next: run mlp_factor_comparison.py for the combined evaluation table.")


if __name__ == "__main__":
    main()
