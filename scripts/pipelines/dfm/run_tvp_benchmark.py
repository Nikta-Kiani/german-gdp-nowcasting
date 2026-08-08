"""Two-step TVP-DFM benchmark for the thesis (random-walk factor->GDP loadings).

Motivation
----------
After 2022 German GDP enters a stagnation/energy-shock regime in which the
historical factor (ifo, hard data) -> GDP transmission can weaken permanently.
A fixed-loading DFM stays anchored to the pre-break estimate. Del Negro & Otrok
(2008) let the loadings drift as a random walk so the mapping adapts.

This script runs a two-step TVP-DFM on the **same** real-time information set
and the **same** EM-DFM front-end (r=2 factors, AR(2) factor dynamics, AR(1)
idiosyncratic, ragged-edge pub-lag masking, AR(p)-BIC fill) as DFM-EN, so the
only difference versus DFM-EN is the time-varying second-stage GDP equation.
It evaluates RMSFE (M3 + pooled, full sample + COVID regimes), NSR, and a
Diebold-Mariano test versus DFM-EN, DFM-ifoCAST and AR(1).

Run (from the repository root):
    python scripts/pipelines/dfm/run_tvp_benchmark.py
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

from german_gdp_nowcasting.config import paths as P  # noqa: E402
from german_gdp_nowcasting.selection.core_utils import (  # noqa: E402
    load_monthly_panel,
    load_pub_lag_map,
)
from german_gdp_nowcasting.models.dfm.nowcast_utils import (  # noqa: E402
    align_forecast_errors,
    compute_nsr,
    compute_rmsfe,
    diebold_mariano_test,
    expand_quarterly_nowcasts_to_monthly,
)
from german_gdp_nowcasting.models.dfm.tvp_dfm import (  # noqa: E402
    run_actpn_nowcast_loop_tvp,
)

EVAL_START = "2011Q1"
EVAL_END = "2025Q4"
HEADLINE_MIQ = 3
K_FACTORS = 2
FACTOR_ORDER = 2

REGIMES: dict[str, tuple[str, str]] = {
    "pre-COVID": ("2011Q1", "2019Q4"),
    "COVID": ("2020Q1", "2021Q4"),
    "post-COVID": ("2022Q1", "2025Q4"),
}


def regime_rmsfe(df: pd.DataFrame) -> dict[str, float]:
    """Compute headline M3 RMSFE within each evaluation regime."""
    return {reg: compute_rmsfe(df, eval_start=q0, eval_end=q1, month_in_quarter=HEADLINE_MIQ)
            for reg, (q0, q1) in REGIMES.items()}


def _load(path: Path) -> pd.DataFrame:
    """Load a result CSV and normalize it to monthly-origin rows."""
    d = pd.read_csv(path)
    if "month_in_quarter" not in d.columns:
        d = expand_quarterly_nowcasts_to_monthly(d)
    if "monthly_origin" in d.columns:
        d = d.set_index("monthly_origin", drop=False)
    return d


def main() -> None:
    """Run the TVP-DFM benchmark and comparison diagnostics."""
    print("=" * 78)
    print("Two-step TVP-DFM benchmark  (DFM: r=2, AR(2) factors; RW factor->GDP loadings)")
    print("=" * 78)

    X_monthly = load_monthly_panel(P.PANEL_TRANSFORMED_CSV)
    pub_lag_map = load_pub_lag_map(P.PUB_LAG_CSV)
    y_q = pd.read_csv(P.GDP_TARGET_CSV, index_col="quarter").squeeze("columns")
    y_q.index = pd.PeriodIndex(y_q.index, freq="Q")

    # Use the same EN real-time selection path as DFM-EN for a clean comparison.
    sel = pd.read_csv(P.EN_ONLY_MATRIX_CSV, index_col="forecast_origin")

    print("\n[1] Running expanding two-step TVP-DFM (this takes a few minutes) ...")
    quarterly_origins = pd.period_range(EVAL_START, EVAL_END, freq="Q")
    df_tvp = run_actpn_nowcast_loop_tvp(
        selection_matrix=sel,
        X_monthly=X_monthly,
        y_quarterly=y_q,
        quarterly_origins=quarterly_origins,
        k_factors=K_FACTORS,
        factor_order=FACTOR_ORDER,
        idiosyncratic_ar1=True,
        maxiter=200,
        pub_lag_map=pub_lag_map,
        fill_method="ar_bic",
        verbose=True,
    )
    P.OUT_NOWCASTING.mkdir(parents=True, exist_ok=True)
    df_tvp.to_csv(P.TVP_RESULTS_CSV)
    print(f"\nSaved nowcasts -> {P.TVP_RESULTS_CSV}")

    # Drift diagnostic: average q_ratio and the post-COVID loading path.
    if "q_ratio" in df_tvp.columns:
        print(f"  median q_ratio (drift speed): {df_tvp['q_ratio'].median():.2e}")

    print("\n[2] Evaluation")
    models = {"DFM-TVP": df_tvp, "DFM-EN": _load(P.actpn_results_csv("en_only"))}
    for label, path in [("DFM-ifoCAST", P.IFO_RESULTS_CSV), ("AR1", P.AR1_RESULTS_CSV)]:
        if path.exists():
            models[label] = _load(path)

    rows = []
    for name, d in models.items():
        rows.append({
            "model": name,
            "RMSFE_M3": compute_rmsfe(d, EVAL_START, EVAL_END, month_in_quarter=HEADLINE_MIQ),
            "RMSFE_pooled": compute_rmsfe(d, EVAL_START, EVAL_END, month_in_quarter=None),
            "NSR": compute_nsr(d, y_q, EVAL_START, EVAL_END),
            **{f"RMSFE_{k}": v for k, v in regime_rmsfe(d).items()},
        })
    eval_tbl = pd.DataFrame(rows).set_index("model").round(4)
    print("\n" + eval_tbl.to_string())
    eval_tbl.to_csv(P.OUT_NOWCASTING / "tvp_benchmark_rmsfe.csv")

    print("\n[3] Diebold-Mariano (TVP vs X; negative DM => TVP better):")
    for comp in [m for m in ("DFM-EN", "DFM-ifoCAST", "AR1") if m in models]:
        ea, eb = align_forecast_errors(df_tvp, models[comp], month_in_quarter=HEADLINE_MIQ,
                                       eval_start=EVAL_START, eval_end=EVAL_END)
        dm = diebold_mariano_test(ea, eb)
        print(f"    vs {comp:14s}: DM={dm['DM']:+.3f}  p={dm['p_value']:.3f}  n={dm['n']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
