"""Regenerate the EN-only selection matrix with COVID down-weighting + N_max cap.

Root-cause fix for the post-COVID indicator explosion (n_selected jumping from
~50 to 130-222): the Elastic Net CV penalty collapses toward Ridge once the
COVID quarters enter the expanding training window, so it retains 100+ weakly
related series and never recovers.

This script mirrors the capped headline EN specification in
``scripts/pipelines/orchestrators/01_selection.py`` (same panel, imputer,
t-stat pre-filter, COVID sample weights and ``MAX_SELECTED``). It remains a
non-destructive diagnostic entry point: outputs are written to separate paths
so they can be compared with the canonical artifacts.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# The Elastic Net path + IterativeImputer emit large volumes of benign
# RuntimeWarning (matmul overflow) / ConvergenceWarning on this wide panel;
# left unsuppressed they flood the log by the megabyte and throttle the run.
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

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
)
from german_gdp_nowcasting.selection.elastic_net_selection import (  # noqa: E402
    covid_sample_weights,
    run_expanding_selection,
)

# --- Settings mirror the canonical 01_selection.py EN specification ----------
FORECAST_START, FORECAST_END = "2011-01", "2025-12"
TRAIN_START = "1991Q1"
MIN_COVERAGE = 0.30
# "iterative" matches the canonical pipeline exactly;
# "mean" is ~100x faster for a quick cap sanity-check. Override via env var.
IMPUTER = os.environ.get("REGEN_IMPUTER", "iterative").strip()
MAX_SELECTED = 60  # hard upper cap on indicators per origin

# --- New (non-destructive) output paths -------------------------------------
_SUFFIX = "" if IMPUTER == "iterative" else f"_{IMPUTER}"
OUT_MATRIX = tp.SELECTION_DIR / f"en_only_selection_matrix_capped60{_SUFFIX}.csv"
OUT_RESULTS = tp.OUT_INDICATOR_SELECTION / f"selection_results_capped60{_SUFFIX}.json"


def main() -> None:
    """Regenerate the capped EN selection path at non-canonical outputs."""
    t0 = time.perf_counter()
    X_monthly = load_monthly_panel(tp.PANEL_TRANSFORMED_CSV)
    trafo_map = load_trafo_map(tp.DATA_DICT_ENRICHED_CSV)
    y_q = pd.read_csv(tp.GDP_TARGET_CSV, index_col=0).iloc[:, 0]
    y_q.index = pd.PeriodIndex(y_q.index, freq="Q")

    origins = make_monthly_forecast_origins(FORECAST_START, FORECAST_END)
    coverage_mask = build_coverage_mask(X_monthly, origins, min_coverage=MIN_COVERAGE)
    sw = covid_sample_weights(y_q, start="2020Q2", end="2021Q1", weight=0.25)

    print(f"Regenerating EN selection with COVID weights + cap N_max={MAX_SELECTED} ...",
          flush=True)
    en_mat, en_res = run_expanding_selection(
        X_monthly=X_monthly, y_quarterly=y_q, trafo_map=trafo_map,
        forecast_origins=origins, coverage_mask=coverage_mask,
        train_start_quarter=TRAIN_START, min_selected=1, max_selected=MAX_SELECTED,
        n_splits=5, imputer_strategy=IMPUTER, tstat_prefilter=True,
        tstat_threshold=1.65, n_lags=0, sample_weight=sw,
    )

    tp.SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    en_mat.astype(int).to_csv(OUT_MATRIX)
    with OUT_RESULTS.open("w") as f:
        json.dump(en_res, f, indent=2, default=str)

    counts = en_mat.astype(int).sum(axis=1)
    print(f"\nSaved {OUT_MATRIX.name}  shape={en_mat.shape}")
    print(f"n_selected per origin: min={counts.min()} median={counts.median():.0f} "
          f"mean={counts.mean():.1f} max={counts.max()}")
    print(f"origins at the cap ({MAX_SELECTED}): {(counts >= MAX_SELECTED).sum()} / {len(counts)}")
    print(f"DONE in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
