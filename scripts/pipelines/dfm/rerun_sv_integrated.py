"""Re-run integrated DFM-SV nowcast (k=2) on the EN-only indicator set.

Unlike ``dfm_sv_bayes``, stochastic volatility is fed back into the Kalman
smoother via a time-varying factor-innovation covariance, so the point nowcast
can differ from plain DFM-EN. Writes ``nowcast_results_actpn_sv_integrated_k2.csv``.
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
from german_gdp_nowcasting.selection.core_utils import (  # noqa: E402
    load_monthly_panel,
    load_pub_lag_map,
)
from german_gdp_nowcasting.models.dfm.dfm_sv_integrated import (  # noqa: E402
    run_actpn_nowcast_loop_sv_integrated,
)
from german_gdp_nowcasting.models.dfm.nowcast_utils import compute_rmsfe  # noqa: E402

EVAL_START = "2011Q1"
EVAL_END = "2025Q4"
HEADLINE_MIQ = 3
FACTOR_ORDER = 2
K_FACTORS = 2

OUT_PATH = _tp.ACTPN_SV_RESULTS_K2_CSV


def main() -> None:
    """Run and save the headline integrated-SV DFM specification."""
    X_monthly = load_monthly_panel(_tp.PANEL_TRANSFORMED_CSV)
    pub_lag_map = load_pub_lag_map(_tp.PUB_LAG_CSV)
    y_quarterly = pd.read_csv(_tp.GDP_TARGET_CSV, index_col="quarter").squeeze("columns")
    y_quarterly.index = pd.PeriodIndex(y_quarterly.index, freq="Q")
    selection_matrix = pd.read_csv(
        _tp.EN_ONLY_MATRIX_CSV, index_col="forecast_origin",
    ).astype(int)
    quarterly_origins = pd.period_range(EVAL_START, EVAL_END, freq="Q")

    if OUT_PATH.exists():
        OUT_PATH.unlink()
        print(f"Removed previous {OUT_PATH.name}")

    print(f"\n=== Integrated SV k={K_FACTORS} on EN-only inputs ===")
    df = run_actpn_nowcast_loop_sv_integrated(
        selection_matrix=selection_matrix,
        X_monthly=X_monthly,
        y_quarterly=y_quarterly,
        quarterly_origins=quarterly_origins,
        factor_order=FACTOR_ORDER,
        idiosyncratic_ar1=True,
        maxiter=200,
        k_factors=K_FACTORS,
        pub_lag_map=pub_lag_map,
        num_warmup=500,
        num_samples=1000,
        credibility=0.9,
        rng_seed=42,
        n_iter=1,
        save_path=OUT_PATH,
        verbose=True,
    )
    df.to_csv(OUT_PATH)
    r3 = compute_rmsfe(
        df, eval_start=EVAL_START, eval_end=EVAL_END, month_in_quarter=HEADLINE_MIQ,
    )
    rp = compute_rmsfe(df, eval_start=EVAL_START, eval_end=EVAL_END)
    shift = df["point_shift"].abs().mean() if "point_shift" in df.columns else float("nan")
    print(f"Done: n={len(df)}  RMSFE M3={r3:.4f}  pooled={rp:.4f}  mean|Δpoint|={shift:.4f}")


if __name__ == "__main__":
    main()
