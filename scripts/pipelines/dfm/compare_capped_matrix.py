"""Compare DFM-EN and DFM-SV-integrated on OLD vs capped (N_max=60) EN matrix.

Non-destructive: reads the new capped matrix from its own path, writes results to
_capped60 CSVs, and does NOT touch the canonical EN_ONLY matrix. Run stages via
CLI arg: "en" (fast DFM-EN both matrices), "svint" (slow SV-integrated capped),
or "report" (print comparison from existing CSVs).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

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
    compute_rmsfe,
    run_actpn_nowcast_loop,
)
from german_gdp_nowcasting.models.dfm.dfm_sv_integrated import (  # noqa: E402
    run_actpn_nowcast_loop_sv_integrated,
)

EVAL_START, EVAL_END = "2011Q1", "2025Q4"
K_FACTORS, FACTOR_ORDER = 2, 2

CAPPED_MATRIX = tp.SELECTION_DIR / "en_only_selection_matrix_capped60.csv"
OUT_EN_OLD = tp.OUT_NOWCASTING / "nowcast_results_actpn_en_only_oldmatrix.csv"
OUT_EN_CAP = tp.OUT_NOWCASTING / "nowcast_results_actpn_en_only_capped60.csv"
OUT_SV_CAP = tp.OUT_NOWCASTING / "nowcast_results_actpn_sv_integrated_k2_capped60.csv"
OUT_SV_OLD = tp.OUT_NOWCASTING / "nowcast_results_actpn_sv_integrated_k2.csv"


def _load_inputs() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.PeriodIndex]:
    """Load panel, publication lags, GDP, and evaluation origins."""
    X = load_monthly_panel(tp.PANEL_TRANSFORMED_CSV)
    pl = load_pub_lag_map(tp.PUB_LAG_CSV)
    y = pd.read_csv(tp.GDP_TARGET_CSV, index_col="quarter").squeeze("columns")
    y.index = pd.PeriodIndex(y.index, freq="Q")
    qo = pd.period_range(EVAL_START, EVAL_END, freq="Q")
    return X, pl, y, qo


def _rmsfe(df: pd.DataFrame) -> tuple[float, float]:
    """Return headline M3 and pooled RMSFE."""
    return (compute_rmsfe(df, eval_start=EVAL_START, eval_end=EVAL_END, month_in_quarter=3),
            compute_rmsfe(df, eval_start=EVAL_START, eval_end=EVAL_END))


def run_en() -> None:
    """Run plain DFM-EN on the old and capped selection matrices."""
    X, pl, y, qo = _load_inputs()
    old = pd.read_csv(tp.EN_ONLY_MATRIX_CSV, index_col="forecast_origin").astype(int)
    cap = pd.read_csv(CAPPED_MATRIX, index_col=0).astype(int)
    for label, mat, out in [("OLD", old, OUT_EN_OLD), ("CAP60", cap, OUT_EN_CAP)]:
        print(f"\n=== DFM-EN on {label} matrix ===", flush=True)
        df = run_actpn_nowcast_loop(
            selection_matrix=mat, X_monthly=X, y_quarterly=y, quarterly_origins=qo,
            k_factors=K_FACTORS, factor_order=FACTOR_ORDER, idiosyncratic_ar1=True,
            maxiter=200, pub_lag_map=pl, verbose=False,
        )
        df.to_csv(out)
        m3, pooled = _rmsfe(df)
        print(f"  {label}: n={len(df)}  RMSFE M3={m3:.4f}  pooled={pooled:.4f}")


def run_svint() -> None:
    """Run integrated-SV DFM on the capped selection matrix."""
    X, pl, y, qo = _load_inputs()
    cap = pd.read_csv(CAPPED_MATRIX, index_col=0).astype(int)
    print("\n=== DFM-SV-integrated (k=2) on CAP60 matrix ===", flush=True)
    df = run_actpn_nowcast_loop_sv_integrated(
        selection_matrix=cap, X_monthly=X, y_quarterly=y, quarterly_origins=qo,
        k_factors=K_FACTORS, factor_order=FACTOR_ORDER, idiosyncratic_ar1=True,
        maxiter=200, pub_lag_map=pl, num_warmup=500, num_samples=1000,
        credibility=0.9, rng_seed=42, n_iter=1, save_path=OUT_SV_CAP, verbose=True,
    )
    df.to_csv(OUT_SV_CAP)
    m3, pooled = _rmsfe(df)
    print(f"  CAP60 SV-int: n={len(df)}  RMSFE M3={m3:.4f}  pooled={pooled:.4f}")


def report() -> None:
    """Print the capping comparison from saved result files."""

    def line(tag: str, path: Path) -> None:
        """Print one result file's RMSFE and optional point-shift summary."""
        if not Path(path).exists():
            print(f"  {tag}: (missing {Path(path).name})")
            return
        df = pd.read_csv(path, index_col=0)
        m3, pooled = _rmsfe(df)
        extra = ""
        if "point_shift" in df.columns:
            extra = f"  mean|Δpoint|={df['point_shift'].abs().mean():.4f}"
        print(f"  {tag:22s}: RMSFE M3={m3:.4f}  pooled={pooled:.4f}{extra}")
    print("\n================ CAPPING IMPACT (2011Q1-2025Q4) ================")
    print("DFM-EN:")
    line("OLD matrix", OUT_EN_OLD)
    line("CAP60 matrix", OUT_EN_CAP)
    print("DFM-SV-integrated (k=2):")
    line("OLD matrix", OUT_SV_OLD)
    line("CAP60 matrix", OUT_SV_CAP)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"en": run_en, "svint": run_svint, "report": report}[stage]()
