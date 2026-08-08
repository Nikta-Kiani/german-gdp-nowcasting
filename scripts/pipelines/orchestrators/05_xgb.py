#!/usr/bin/env python3
"""Rerun XGBoost nowcasts (retune + full SHAP-pruned loop) under new aggregation.

Mirrors notebook 05 headline calls. The quarterly feature aggregation changed
(raw-level bridge), so both tuning and the 60-origin loop are regenerated.
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
    build_rmsfe_table,
)

EVAL_START, EVAL_END, TRAIN_START = "2011Q1", "2025Q4", "1991Q1"
GDP_LAGS = (1, 2)


def _run() -> None:
    """Retune and run the headline SHAP-pruned XGBoost pipeline."""
    t0 = time.perf_counter()
    X_monthly = load_monthly_panel(tp.PANEL_TRANSFORMED_CSV)
    pub = load_pub_lag_map(tp.PUB_LAG_CSV)
    trafo = load_trafo_map(tp.DATA_DICT_ENRICHED_CSV)
    y_q = pd.read_csv(tp.GDP_TARGET_CSV)
    y_q["quarter"] = pd.PeriodIndex(y_q["quarter"], freq="Q")
    y_q = y_q.set_index("quarter").iloc[:, 0]; y_q.name = "gdp"
    fo = make_monthly_forecast_origins("2011-01", "2025-12")
    q_origins = pd.period_range(EVAL_START, EVAL_END, freq="Q")

    # Retune on core set at 2010-12 (pre-eval).
    print("[XGB] retuning ...", flush=True)
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
    hp = xu.tune_xgb(X_tr, y_tr, n_iter=40, cv_splits=5)
    xu.save_params(tp.XGB_BEST_PARAMS_JSON, hp)
    print(f"   best CV RMSE={hp.cv_rmse:.4f}", flush=True)

    # Full loop with SHAP pruning.
    print("[XGB] full loop (60 origins, SHAP prune every 4Q) ...", flush=True)
    coverage_mask = build_coverage_mask(X_monthly, fo, min_coverage=0.30)
    df_full, shap_log = xu.run_xgb_nowcast_loop(
        selection_matrix=None, X_monthly=X_monthly, y_quarterly=y_q,
        params=hp.params, quarterly_origins=q_origins, pub_lag_map=pub,
        lags=xu.DEFAULT_LAGS, train_start=TRAIN_START, trafo_map=trafo,
        fill_method="ar_bic", gdp_lags=GDP_LAGS, coverage_mask=coverage_mask,
        shap_pruning=True, shap_refit_every=4, shap_keep_frac=0.90,
        shap_min_features=20, shap_log_every=1, verbose=False,
    )
    df_full.to_csv(tp.xgb_results_csv("full"))
    if not shap_log.empty:
        shap_log.to_csv(tp.XGB_SHAP_IMPORTANCE_CSV)
    rm = float(np.sqrt(np.mean(df_full["error"].dropna() ** 2)))
    print(f"   saved {tp.xgb_results_csv('full').name}  RMSFE@M3={rm:.4f}", flush=True)

    # Focused RMSFE table (DFM-EN reference + baselines + XGB).
    def _load(p: Path) -> pd.DataFrame | None:
        """Load an optional saved benchmark result."""
        return pd.read_csv(p) if Path(p).exists() else None
    models = {}
    dfm = _load(tp.actpn_results_csv("en_only"))
    if dfm is not None:
        models["DFM-EN"] = dfm
    models["XGB-Full"] = df_full.reset_index()
    for nm, p in [("AR1", tp.AR1_RESULTS_CSV), ("RW", tp.RW_RESULTS_CSV)]:
        d = _load(p)
        if d is not None:
            models[nm] = d
    rmsfe = build_rmsfe_table(
        models, reference_key="DFM-EN" if "DFM-EN" in models else None,
        eval_start=EVAL_START, eval_end=EVAL_END, y_quarterly=y_q,
        month_in_quarter=3,
    )
    rmsfe.to_csv(tp.RMSFE_TABLE_XGB_CSV)
    print(rmsfe.to_string(), flush=True)
    print(f"DONE XGB in {(time.perf_counter()-t0)/60:.1f} min", flush=True)


def main() -> None:
    """Run XGBoost with joblib's threading backend."""
    # Force joblib's threading backend: the default loky backend probes
    # SC_SEM_NSEMS_MAX via os.sysconf, which is blocked in the sandbox.
    import joblib

    with joblib.parallel_backend("threading"):
        _run()


if __name__ == "__main__":
    main()
