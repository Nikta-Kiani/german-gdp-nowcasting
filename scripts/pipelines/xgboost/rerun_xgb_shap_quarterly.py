"""Recompute the XGB-Full SHAP importance log at **quarterly** cadence.

The original loop only logged SHAP at the pruning-refit quarters (every 4th
quarter -> annual). This script reruns the exact same XGB-Full specification but
with ``shap_log_every=1``, so mean |SHAP| is recorded for the *deployed* model at
every one of the 60 evaluation quarters — giving a per-origin importance trace
for the Part~I selection comparison. The pruning refit cadence is unchanged, so
the nowcasts themselves are identical; only the importance export is enriched.

Run (from the repository root):
    python scripts/pipelines/xgboost/rerun_xgb_shap_quarterly.py
"""

from __future__ import annotations

import sys
from pathlib import Path

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

from german_gdp_nowcasting.config import paths as _tp  # noqa: E402
from german_gdp_nowcasting.models.xgboost import xgb_utils as xu  # noqa: E402
from german_gdp_nowcasting.selection.core_utils import (  # noqa: E402
    load_monthly_panel, load_pub_lag_map, load_trafo_map,
    build_coverage_mask, make_monthly_forecast_origins,
)

EVAL_START, EVAL_END = "2011Q1", "2025Q4"
TRAIN_START = "1991Q1"
GDP_LAGS = (1, 2)


def main() -> None:
    """Recompute and save quarterly SHAP logs for XGB-Full."""
    print("Loading inputs ...")
    X_monthly = load_monthly_panel(_tp.PANEL_TRANSFORMED_CSV)
    pub_lag_map = load_pub_lag_map(_tp.PUB_LAG_CSV)
    trafo_map = load_trafo_map(_tp.DATA_DICT_ENRICHED_CSV)
    y_raw = pd.read_csv(_tp.GDP_TARGET_CSV)
    y_raw["quarter"] = pd.PeriodIndex(y_raw["quarter"], freq="Q")
    y_quarterly = y_raw.set_index("quarter").iloc[:, 0]
    y_quarterly.name = "gdp"

    forecast_origins = make_monthly_forecast_origins("2011-01", "2025-12")
    quarterly_origins = pd.period_range(EVAL_START, EVAL_END, freq="Q")
    coverage_mask = build_coverage_mask(X_monthly, forecast_origins,
                                        min_coverage=0.30)

    payload = xu.load_params(_tp.XGB_BEST_PARAMS_JSON)
    if payload is None or not isinstance(payload, dict) or not payload.get("params"):
        raise RuntimeError(
            f"No usable XGBoost parameters found in {_tp.XGB_BEST_PARAMS_JSON}. "
            "Run scripts/pipelines/orchestrators/05_xgb.py first to tune and save "
            "the required params payload."
        )
    params = payload["params"]
    print(f"BEST_PARAMS: {params}")

    print("Running XGB-Full loop with quarterly SHAP logging "
          "(shap_log_every=1) ...")
    df_full, shap_log = xu.run_xgb_nowcast_loop(
        selection_matrix=None,
        X_monthly=X_monthly,
        y_quarterly=y_quarterly,
        params=params,
        quarterly_origins=quarterly_origins,
        pub_lag_map=pub_lag_map,
        lags=xu.DEFAULT_LAGS,
        train_start=TRAIN_START,
        trafo_map=trafo_map,
        fill_method="ar_bic",
        gdp_lags=GDP_LAGS,
        coverage_mask=coverage_mask,
        shap_pruning=True,
        shap_refit_every=4,
        shap_keep_frac=0.90,
        shap_min_features=20,
        shap_log_every=1,
        verbose=True,
    )

    shap_csv = _tp.XGB_SHAP_IMPORTANCE_CSV
    out = shap_log.reset_index() if shap_log.index.names != [None] else shap_log
    n_q = out["quarter"].nunique() if not out.empty else 0
    out.to_csv(shap_csv, index=False)
    print(f"\n[saved] {shap_csv}")
    print(f"  rows={len(out)}  quarters={n_q}  features/quarter~{len(out)//max(n_q,1)}")

    # Sanity check: nowcasts must be unchanged vs the cached XGB-Full results.
    cached_path = _tp.xgb_results_csv("full")
    if cached_path.exists():
        cached = pd.read_csv(cached_path).set_index("quarter")["nowcast"]
        new = df_full["nowcast"]
        new.index = new.index.astype(str)
        cached.index = cached.index.astype(str)
        diff = (new - cached.reindex(new.index)).abs().max()
        print(f"  max |Δnowcast| vs cached XGB-Full: {diff:.2e} "
              f"({'unchanged' if diff < 1e-9 else 'CHANGED — investigate'})")


if __name__ == "__main__":
    main()
