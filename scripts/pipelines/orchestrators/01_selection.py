#!/usr/bin/env python3
"""Rebuild the recursive selection matrices used in the thesis.

Reproduces the notebook 03–05 calls: capped elastic net, fixed-k, PLS,
frequency-smoothed EN, and the DFM input matrices. The GDP target is reused
from gdp_target.csv.
"""
from __future__ import annotations

import json
import sys
import time
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

from german_gdp_nowcasting.config import paths as tp  # noqa: E402
from german_gdp_nowcasting.selection.core_utils import (  # noqa: E402
    build_coverage_mask,
    load_monthly_panel,
    load_trafo_map,
    make_monthly_forecast_origins,
    save_selection_outputs,
)
from german_gdp_nowcasting.selection.elastic_net_selection import (  # noqa: E402
    covid_sample_weights,
    run_expanding_selection,
    run_expanding_selection_fixedk,
)
from german_gdp_nowcasting.selection.pls_selection import (  # noqa: E402
    run_expanding_selection_pls,
)
from german_gdp_nowcasting.selection.selection_postprocessing import (  # noqa: E402
    apply_frequency_smoothing,
)
from german_gdp_nowcasting.selection.dfm_input_builder import (  # noqa: E402
    build_dfm_input_sets,
)

FORECAST_START, FORECAST_END = "2011-01", "2025-12"
TRAIN_START = "1991Q1"
MIN_COVERAGE = 0.30
IMPUTER = "iterative"
MAX_SELECTED = 60
MIN_VOTES = 3


def main() -> None:
    """Regenerate all selection matrices and DFM input sets."""
    t0 = time.perf_counter()
    X_monthly = load_monthly_panel(tp.PANEL_TRANSFORMED_CSV)
    trafo_map = load_trafo_map(tp.DATA_DICT_ENRICHED_CSV)
    y_q = pd.read_csv(tp.GDP_TARGET_CSV, index_col=0).iloc[:, 0]
    y_q.index = pd.PeriodIndex(y_q.index, freq="Q")

    origins = make_monthly_forecast_origins(FORECAST_START, FORECAST_END)
    coverage_mask = build_coverage_mask(X_monthly, origins, min_coverage=MIN_COVERAGE)
    sw = covid_sample_weights(y_q, start="2020Q2", end="2021Q1", weight=0.25)

    print("[1/5] EN selection ...", flush=True)
    en_mat, en_res = run_expanding_selection(
        X_monthly=X_monthly, y_quarterly=y_q, trafo_map=trafo_map,
        forecast_origins=origins, coverage_mask=coverage_mask,
        train_start_quarter=TRAIN_START, min_selected=1,
        max_selected=MAX_SELECTED, n_splits=5, imputer_strategy=IMPUTER,
        tstat_prefilter=True, tstat_threshold=1.65, n_lags=0,
        sample_weight=sw,
    )
    save_selection_outputs(tp.OUT_INDICATOR_SELECTION, en_mat, en_res)
    print(f"   saved {tp.SELECTION_MATRIX_CSV.name}  {en_mat.shape}", flush=True)

    print("[2/5] Fixed-k selection ...", flush=True)
    fk_mat, fk_res = run_expanding_selection_fixedk(
        X_monthly=X_monthly, y_quarterly=y_q, trafo_map=trafo_map,
        forecast_origins=origins, coverage_mask=coverage_mask,
        train_start_quarter=TRAIN_START, k=30, l2_penalty=0.25,
        imputer_strategy=IMPUTER,
    )
    fk_mat.to_csv(tp.FIXEDK_MATRIX_CSV)
    with (tp.OUT_INDICATOR_SELECTION / "selection_results_fixedk.json").open("w") as f:
        json.dump(fk_res, f, indent=2)
    print(f"   saved {tp.FIXEDK_MATRIX_CSV.name}  {fk_mat.shape}", flush=True)

    print("[3/5] PLS selection ...", flush=True)
    pls_mat, pls_res = run_expanding_selection_pls(
        X_monthly=X_monthly, y_quarterly=y_q, trafo_map=trafo_map,
        forecast_origins=origins, coverage_mask=coverage_mask,
        train_start_quarter=TRAIN_START, n_components=5, top_k=30,
        imputer_strategy=IMPUTER,
    )
    pls_mat.to_csv(tp.PLS_MATRIX_CSV)
    with (tp.OUT_INDICATOR_SELECTION / "selection_results_pls.json").open("w") as f:
        json.dump(pls_res, f, indent=2)
    print(f"   saved {tp.PLS_MATRIX_CSV.name}  {pls_mat.shape}", flush=True)

    print("[4/5] EN frequency smoothing ...", flush=True)
    en_smoothed = apply_frequency_smoothing(en_mat, window_quarters=2, min_freq=0.5)
    en_smoothed.to_csv(tp.EN_SMOOTHED_MATRIX_CSV)
    print(f"   saved {tp.EN_SMOOTHED_MATRIX_CSV.name}  {en_smoothed.shape}", flush=True)

    print("[5/5] DFM input sets ...", flush=True)
    meta = pd.read_csv(
        tp.DATA_DICT_ENRICHED_CSV, usecols=["id", "name", "category"]
    ).set_index("id")
    matrices = {
        "EN raw": en_mat.astype(int),
        "EN smoothed": en_smoothed.astype(int),
        "PLS": pls_mat.astype(int),
        "fixed-k (k=30)": fk_mat.astype(int),
    }
    sets = build_dfm_input_sets(
        matrices=matrices, meta=meta, min_votes=MIN_VOTES,
        en_label="EN raw", pls_label="PLS",
    )
    sets.pop("_rate_table_diagnostic", None)
    tp.SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    sets["core"].to_csv(tp.CORE_MATRIX_CSV)
    sets["en_only"].to_csv(tp.EN_ONLY_MATRIX_CSV)
    sets["pls_only"].to_csv(tp.PLS_ONLY_MATRIX_CSV)
    for k, m in sets.items():
        print(f"   {k:9s}: {m.shape}  mean/origin={m.sum(axis=1).mean():.1f}", flush=True)

    print(f"DONE selection in {(time.perf_counter()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
