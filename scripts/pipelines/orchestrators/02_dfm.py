#!/usr/bin/env python3
"""Rerun DFM-EN nowcasts (en_only / core / pls_only) + AR1/RW baselines + tables.

Mirrors notebook 04 headline calls. DFM consumes the monthly transformed panel
(unchanged) but the EN-derived selection matrices changed under the new
aggregation, so all EN/PLS/core DFM nowcasts are regenerated.
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
from german_gdp_nowcasting.selection.core_utils import (  # noqa: E402
    load_monthly_panel,
    load_pub_lag_map,
)
from german_gdp_nowcasting.models.dfm.nowcast_utils import (  # noqa: E402
    run_actpn_nowcast_loop, run_ar1_baseline, run_rw_baseline,
    build_rmsfe_table, diebold_mariano_test, align_forecast_errors,
)

EVAL_START, EVAL_END = "2011Q1", "2025Q4"
K_FACTORS, FACTOR_ORDER = 2, 2

INPUT_SETS = {
    "en_only": tp.EN_ONLY_MATRIX_CSV,
    "core": tp.CORE_MATRIX_CSV,
    "pls_only": tp.PLS_ONLY_MATRIX_CSV,
}


def main() -> None:
    """Regenerate DFM nowcasts, baselines, and evaluation tables."""
    t0 = time.perf_counter()
    X_monthly = load_monthly_panel(tp.PANEL_TRANSFORMED_CSV)
    pub = load_pub_lag_map(tp.PUB_LAG_CSV)
    y_q = pd.read_csv(tp.GDP_TARGET_CSV, index_col=0).iloc[:, 0]
    y_q.index = pd.PeriodIndex(y_q.index, freq="Q")
    q_origins = pd.period_range(EVAL_START, EVAL_END, freq="Q")

    results: dict[str, pd.DataFrame] = {}
    for key, path in INPUT_SETS.items():
        print(f"[DFM:{key}] ...", flush=True)
        mat = pd.read_csv(path, index_col=0)
        df = run_actpn_nowcast_loop(
            selection_matrix=mat, X_monthly=X_monthly, y_quarterly=y_q,
            quarterly_origins=q_origins, k_factors=K_FACTORS,
            factor_order=FACTOR_ORDER, idiosyncratic_ar1=True, maxiter=200,
            verbose=False, pub_lag_map=pub, fill_method="ar_bic",
            ar_max_p=4, ar_min_train=24,
        )
        out = tp.actpn_results_csv(key)
        df.to_csv(out)
        results[f"DFM-{key}"] = df
        rm = float(np.sqrt(np.mean(df[df["month_in_quarter"] == 3]["error"].dropna() ** 2)))
        print(f"   saved {out.name}  RMSFE@M3={rm:.4f}", flush=True)

    print("[baselines] AR1 / RW ...", flush=True)
    ar1 = run_ar1_baseline(y_q, q_origins, train_start_quarter="1991Q1")
    rw = run_rw_baseline(y_q, q_origins)
    ar1.to_csv(tp.AR1_RESULTS_CSV)
    rw.to_csv(tp.RW_RESULTS_CSV)

    # Headline RMSFE + DM tables (DFM-EN reference), matching notebook 04.
    print("[tables] rmsfe / DM ...", flush=True)
    tbl_models = {
        "DFM-EN": results["DFM-en_only"],
        "DFM-Core": results["DFM-core"],
        "DFM-PLS": results["DFM-pls_only"],
        "AR(1)": ar1, "RW": rw,
    }
    rmsfe = build_rmsfe_table(
        tbl_models, reference_key="DFM-EN",
        eval_start=EVAL_START, eval_end=EVAL_END, y_quarterly=y_q,
        month_in_quarter=3,
    )
    rmsfe.to_csv(tp.RMSFE_TABLE_CSV)
    print(rmsfe.to_string(), flush=True)

    dm_rows = []
    ref = results["DFM-en_only"]
    for name, df in tbl_models.items():
        if name == "DFM-EN":
            continue
        ea, eb = align_forecast_errors(ref, df, month_in_quarter=3,
                                       eval_start=EVAL_START, eval_end=EVAL_END)
        dm = diebold_mariano_test(ea, eb, h=1, loss="se")
        dm_rows.append({"model_vs_DFM-EN": name, **dm})
    pd.DataFrame(dm_rows).to_csv(tp.DM_TABLE_CSV, index=False)

    print(f"DONE DFM in {(time.perf_counter()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
