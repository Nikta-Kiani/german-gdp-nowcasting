#!/usr/bin/env python3
"""XGB-Full post-COVID sensitivity check.

Tests how robust the headline post-COVID RMSFE is to the tuning seed,
hyperparameters, and evaluation sample, using the same calling convention
as ``scripts/pipelines/orchestrators/05_xgb.py`` so results are directly comparable
to the headline CSV.

Four checks, in order:
  1. Seed sensitivity   -- retune + full 60-origin loop for several seeds.
  2. HP perturbation    -- one-at-a-time nudges around the seed-42 params.
  3. Jackknife           -- leave-one-quarter-out on the 16 post-COVID rows
                            of the seed-42 baseline (no retraining; free).
  4. DM test             -- XGB-Full (seed-42) vs. Rolling-AR(1) 40q,
                            post-COVID window (not computed anywhere else).

Run (from the repository root):
    python scripts/experiments/xgb_sensitivity.py

Runtime: ~35-45 min (11 full 60-origin loops at ~3.3 min each).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

for _parent in Path(__file__).resolve().parents:
    _src = _parent / "src"
    if (_src / "german_gdp_nowcasting").is_dir():
        sys.path.insert(0, str(_src))
        break
else:
    raise RuntimeError(
        "Could not locate src/german_gdp_nowcasting above this script. "
        "Run it from within the german-gdp-nowcasting repository."
    )

from german_gdp_nowcasting.config import paths as tp  # noqa: E402
from german_gdp_nowcasting.models.xgboost import xgb_utils as xu  # noqa: E402
from german_gdp_nowcasting.selection.core_utils import (  # noqa: E402
    load_monthly_panel, load_pub_lag_map, load_trafo_map,
    build_coverage_mask, make_monthly_forecast_origins,
)
from german_gdp_nowcasting.models.dfm.nowcast_utils import (  # noqa: E402
    align_forecast_errors,
    diebold_mariano_test,
)
from german_gdp_nowcasting.models.dfm.post_covid_benchmarks import (  # noqa: E402
    load_gdp_target,
    rolling_ar1,
)

EVAL_START, EVAL_END, TRAIN_START = "2011Q1", "2025Q4", "1991Q1"
REGIMES = {
    "pre": ("2011Q1", "2019Q4"),
    "COVID": ("2020Q1", "2021Q4"),
    "post": ("2022Q1", "2025Q4"),
}
GDP_LAGS = (1, 2)
BASELINE_SEED = 42
SEEDS = [0, 1, 7, BASELINE_SEED, 123]

# One-at-a-time nudges around the frozen seed-42 params (xgb_best_params.json).
HP_PERTURBATIONS: dict[str, dict] = {
    "max_depth-1": {"max_depth": 5},
    "max_depth+1": {"max_depth": 7},
    "lr_half": {"learning_rate": 0.0125},
    "lr_double": {"learning_rate": 0.05},
    "n_estimators-100": {"n_estimators": 400},
    "n_estimators+100": {"n_estimators": 600},
}

OUT_DIR = tp.OUT_NOWCASTING / "_scratch"


def load_common() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.PeriodIndex,
    pd.DataFrame,
]:
    """Load the shared real-time inputs for all sensitivity runs."""
    X_monthly = load_monthly_panel(tp.PANEL_TRANSFORMED_CSV)
    pub = load_pub_lag_map(tp.PUB_LAG_CSV)
    trafo = load_trafo_map(tp.DATA_DICT_ENRICHED_CSV)
    y_q = pd.read_csv(tp.GDP_TARGET_CSV)
    y_q["quarter"] = pd.PeriodIndex(y_q["quarter"], freq="Q")
    y_q = y_q.set_index("quarter").iloc[:, 0]
    y_q.name = "gdp"
    fo = make_monthly_forecast_origins("2011-01", "2025-12")
    q_origins = pd.period_range(EVAL_START, EVAL_END, freq="Q")
    coverage_mask = build_coverage_mask(X_monthly, fo, min_coverage=0.30)
    return X_monthly, pub, trafo, y_q, q_origins, coverage_mask


def tune_params(
    X_monthly: pd.DataFrame,
    pub: pd.Series,
    trafo: pd.Series,
    y_q: pd.Series,
    random_state: int,
) -> tuple[dict, float]:
    """Retune on the core set at 2010Q4, mirroring staged ``05_xgb.py``."""
    tune_sel = pd.read_csv(tp.CORE_MATRIX_CSV, index_col="forecast_origin").astype(int)
    tune_origin = pd.Period("2010-12", freq="M")
    m3_key = str(xu.quarter_to_m3_period(tune_origin.asfreq("Q")))
    if m3_key not in tune_sel.index:
        earlier = tune_sel.index[tune_sel.index <= m3_key]
        m3_key = earlier[-1] if len(earlier) > 0 else tune_sel.index[0]
    tune_cols = tune_sel.columns[tune_sel.loc[m3_key].astype(bool)].tolist()
    X_tr, y_tr, _ = xu.build_xgb_design_matrix(
        X_monthly=X_monthly, y_quarterly=y_q, origin=tune_origin,
        selected_cols=tune_cols, pub_lag_map=pub, lags=xu.DEFAULT_LAGS,
        train_start=TRAIN_START, trafo_map=trafo, fill_method="ar_bic",
        gdp_lags=GDP_LAGS,
    )
    hp = xu.tune_xgb(X_tr, y_tr, n_iter=40, cv_splits=5, random_state=random_state)
    return dict(hp.params), hp.cv_rmse


def run_full_loop(
    X_monthly: pd.DataFrame,
    pub: pd.Series,
    trafo: pd.Series,
    y_q: pd.Series,
    q_origins: pd.PeriodIndex,
    coverage_mask: pd.DataFrame,
    params: dict,
    random_state: int,
) -> pd.DataFrame:
    """Run one full expanding XGBoost evaluation specification."""
    df, _ = xu.run_xgb_nowcast_loop(
        selection_matrix=None, X_monthly=X_monthly, y_quarterly=y_q,
        params=params, quarterly_origins=q_origins, pub_lag_map=pub,
        lags=xu.DEFAULT_LAGS, train_start=TRAIN_START, trafo_map=trafo,
        fill_method="ar_bic", gdp_lags=GDP_LAGS, coverage_mask=coverage_mask,
        shap_pruning=True, shap_refit_every=4, shap_keep_frac=0.90,
        shap_min_features=20, verbose=False, random_state=random_state,
    )
    # run_xgb_nowcast_loop returns "quarter" as the index (only a plain
    # column after a CSV round-trip); normalise here so downstream helpers
    # can always rely on a "quarter" column.
    return df.reset_index()


def rmsfe_by_regime(df: pd.DataFrame) -> dict:
    """Compute M3 RMSFE for each regime and the full sample."""
    sub = df[df["month_in_quarter"] == 3].copy()
    sub["q"] = pd.PeriodIndex(sub["quarter"], freq="Q")
    out = {}
    for label, (q0, q1) in REGIMES.items():
        e = sub.loc[(sub["q"] >= pd.Period(q0, "Q")) & (sub["q"] <= pd.Period(q1, "Q")),
                    "error"].dropna()
        out[f"rmsfe_{label}"] = float(np.sqrt((e ** 2).mean())) if len(e) else np.nan
    e_full = sub["error"].dropna()
    out["rmsfe_full"] = float(np.sqrt((e_full ** 2).mean())) if len(e_full) else np.nan
    return out


def jackknife_post_covid(baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-quarter-out RMSFE on the 16 post-COVID rows (no retraining)."""
    sub = baseline_df[baseline_df["month_in_quarter"] == 3].copy()
    sub["q"] = pd.PeriodIndex(sub["quarter"], freq="Q")
    q0, q1 = REGIMES["post"]
    post = sub.loc[(sub["q"] >= pd.Period(q0, "Q")) & (sub["q"] <= pd.Period(q1, "Q"))]
    full_rmsfe = float(np.sqrt((post["error"].dropna() ** 2).mean()))
    rows = []
    for _, row in post.iterrows():
        kept = post.loc[post["quarter"] != row["quarter"], "error"].dropna()
        rows.append({
            "dropped_quarter": row["quarter"],
            "dropped_error": row["error"],
            "rmsfe_excl_quarter": float(np.sqrt((kept ** 2).mean())) if len(kept) else np.nan,
            "rmsfe_all_16": full_rmsfe,
        })
    return pd.DataFrame(rows)


def dm_vs_rolling_ar1(baseline_df: pd.DataFrame, y_q: pd.Series) -> dict:
    """DM test: XGB-Full (seed-42 baseline) vs. Rolling-AR(1) 40q, post-COVID."""
    from german_gdp_nowcasting.models.dfm.nowcast_utils import (
        expand_quarterly_nowcasts_to_monthly,
    )

    origins = pd.period_range(EVAL_START, EVAL_END, freq="Q")
    fc = rolling_ar1(y_q, origins, window=40)
    rar1 = pd.DataFrame({
        "quarter": fc.index.astype(str),
        "nowcast": fc.to_numpy(),
        "actual": y_q.reindex(fc.index).to_numpy(),
    })
    rar1["error"] = rar1["nowcast"] - rar1["actual"]
    rar1_m = expand_quarterly_nowcasts_to_monthly(rar1)

    q0, q1 = REGIMES["post"]
    ea, eb = align_forecast_errors(
        baseline_df, rar1_m, month_in_quarter=3, eval_start=q0, eval_end=q1,
    )
    return diebold_mariano_test(ea, eb)


def _run() -> None:
    """Execute all seed, hyperparameter, jackknife, and DM checks."""
    t0 = time.perf_counter()
    X_monthly, pub, trafo, y_q, q_origins, coverage_mask = load_common()

    results: list[dict] = []
    baseline_df: pd.DataFrame | None = None
    baseline_params: dict | None = None

    print("[1/4] Seed sensitivity ...", flush=True)
    for seed in SEEDS:
        ts = time.perf_counter()
        params, cv_rmse = tune_params(X_monthly, pub, trafo, y_q, random_state=seed)
        df = run_full_loop(X_monthly, pub, trafo, y_q, q_origins, coverage_mask,
                            params=params, random_state=seed)
        row = {"run": f"seed_{seed}", "seed": seed, "cv_rmse": cv_rmse}
        row.update(rmsfe_by_regime(df))
        row.update({f"param_{k}": v for k, v in params.items()})
        results.append(row)
        print(f"    seed={seed}  post-COVID RMSFE={row['rmsfe_post']:.3f}  "
              f"full RMSFE={row['rmsfe_full']:.3f}  ({time.perf_counter()-ts:.0f}s)",
              flush=True)
        if seed == BASELINE_SEED:
            baseline_df, baseline_params = df, params
            df.to_csv(OUT_DIR / "xgb_sensitivity_seed42_baseline.csv", index=False)

    assert baseline_df is not None and baseline_params is not None

    print("[2/4] Hyperparameter perturbations (seed=42 base params) ...", flush=True)
    for name, override in HP_PERTURBATIONS.items():
        ts = time.perf_counter()
        params = dict(baseline_params)
        params.update(override)
        df = run_full_loop(X_monthly, pub, trafo, y_q, q_origins, coverage_mask,
                            params=params, random_state=BASELINE_SEED)
        row = {"run": f"hp_{name}", "seed": BASELINE_SEED, "cv_rmse": np.nan}
        row.update(rmsfe_by_regime(df))
        row.update({f"param_{k}": v for k, v in params.items()})
        results.append(row)
        print(f"    {name}  post-COVID RMSFE={row['rmsfe_post']:.3f}  "
              f"full RMSFE={row['rmsfe_full']:.3f}  ({time.perf_counter()-ts:.0f}s)",
              flush=True)

    print("[3/4] Leave-one-quarter-out jackknife (post-COVID, seed-42 baseline) ...",
          flush=True)
    jk = jackknife_post_covid(baseline_df)
    jk_path = OUT_DIR / "xgb_sensitivity_jackknife_postcovid.csv"
    jk.to_csv(jk_path, index=False)
    print(f"    saved -> {jk_path.name}  "
          f"(RMSFE range excl. one quarter: "
          f"{jk['rmsfe_excl_quarter'].min():.3f}-{jk['rmsfe_excl_quarter'].max():.3f}, "
          f"all-16 = {jk['rmsfe_all_16'].iloc[0]:.3f})", flush=True)

    print("[4/4] DM test: XGB-Full (seed-42) vs Rolling-AR(1) 40q, post-COVID ...",
          flush=True)
    y_gdp = load_gdp_target(tp.GDP_TARGET_CSV)
    dm = dm_vs_rolling_ar1(baseline_df, y_gdp)
    print(f"    DM={dm['DM']:.3f}  p={dm['p_value']:.4f}  n={dm['n']}", flush=True)

    summary = pd.DataFrame(results)
    summary_path = OUT_DIR / "xgb_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)

    with open(OUT_DIR / "xgb_sensitivity_dm_vs_rolling_ar1.txt", "w") as fh:
        fh.write(
            f"DM test: XGB-Full (seed={BASELINE_SEED}) vs Rolling-AR(1) 40q, "
            f"post-COVID (2022Q1-2025Q4)\n"
            f"DM={dm['DM']:.4f}  p_value={dm['p_value']:.4f}  n={dm['n']}\n"
        )

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 10)
    print("\n=== Seed sensitivity + HP perturbation summary (RMSFE, pp) ===")
    print(summary[["run", "rmsfe_pre", "rmsfe_COVID", "rmsfe_post", "rmsfe_full"]]
          .round(3).to_string(index=False))
    print(f"\nSaved -> {summary_path.name}, {jk_path.name}, "
          f"xgb_sensitivity_dm_vs_rolling_ar1.txt")
    print(f"DONE xgb_sensitivity in {(time.perf_counter()-t0)/60:.1f} min", flush=True)


def main() -> None:
    """Run sensitivity checks with joblib's threading backend."""
    # Force joblib's threading backend: the default loky backend probes
    # SC_SEM_NSEMS_MAX via os.sysconf, which is blocked in the sandbox.
    import joblib

    with joblib.parallel_backend("threading"):
        _run()


if __name__ == "__main__":
    main()
