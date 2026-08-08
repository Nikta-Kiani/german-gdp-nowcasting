"""Combined evaluation table: all headline models + MLP-Factor.

Reads only saved ``nowcast_results_*.csv`` files -- no model re-runs.
This is a self-contained diagnostic; the canonical cross-model tables are
written by ``scripts/pipelines/dfm/build_unified_evaluation.py``.

Metrics (M3 only, 2011Q1-2025Q4):
  RMSFE_full, RMSFE_pre-COVID, RMSFE_COVID, RMSFE_post-COVID
  NSR (noise-to-signal ratio)
  MZ_beta (Mincer-Zarnowitz slope)
  DM_p_vs_AR1, DM_p_vs_DFM-EN  (two-sided Harvey-Leybourne-Newbold DM test)

Run (from the repository root):
    python scripts/pipelines/mlp/mlp_factor_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
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
from german_gdp_nowcasting.models.dfm.nowcast_utils import (  # noqa: E402
    align_forecast_errors,
    compute_nsr,
    compute_rmsfe,
    diebold_mariano_test,
    expand_quarterly_nowcasts_to_monthly,
    mincer_zarnowitz_test,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
EVAL_START, EVAL_END = "2011Q1", "2025Q4"
M3 = 3

REGIMES: dict[str, tuple[str, str]] = {
    "pre-COVID":  ("2011Q1", "2019Q4"),
    "COVID":      ("2020Q1", "2021Q4"),
    "post-COVID": ("2022Q1", "2025Q4"),
}

MODELS: dict[str, Path] = {
    "DFM-ifoCAST":       P.IFO_RESULTS_CSV,
    "DFM-EN":            P.actpn_results_csv("en_only"),
    "DFM-BlockBalanced": P.BLOCKBALANCED_RESULTS_CSV,
    "DFM-SV-k2":         P.ACTPN_SV_RESULTS_K2_CSV,
    "combo_equal":       P.COMBO_EQUAL_PATH_CSV,
    "XGB-Full":          P.xgb_results_csv("full"),
    "MLP-Factor":        P.MLP_FACTOR_RESULTS_CSV,
    "AR1":               P.AR1_RESULTS_CSV,
    "RW":                P.RW_RESULTS_CSV,
}

OUT_TABLE = _HERE / "mlp_factor_comparison_table.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    """Load a saved nowcast and expand quarterly-only rows when needed."""
    df = pd.read_csv(path)
    if "month_in_quarter" not in df.columns:
        df = expand_quarterly_nowcasts_to_monthly(df)
    return df


def dm_p(
    a: pd.DataFrame,
    b: pd.DataFrame,
    q0: str = EVAL_START,
    q1: str = EVAL_END,
) -> float:
    """Return the M3 Diebold-Mariano p-value for two models."""
    ea, eb = align_forecast_errors(
        a, b, month_in_quarter=M3, eval_start=q0, eval_end=q1
    )
    return diebold_mariano_test(ea, eb)["p_value"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Build and print the combined MLP-factor benchmark table."""
    # Load GDP
    g = pd.read_csv(P.GDP_TARGET_CSV)
    g["quarter"] = pd.PeriodIndex(g["quarter"], freq="Q")
    y_q = g.set_index("quarter").iloc[:, 0]

    # Load all models
    loaded: dict[str, pd.DataFrame] = {}
    print("Loading nowcast CSVs …")
    for name, path in MODELS.items():
        if path.exists():
            loaded[name] = load_csv(path)
            n_m3 = (loaded[name]["month_in_quarter"] == M3).sum()
            print(f"  {name:20s}  {n_m3} M3 rows")
        else:
            print(f"  {'[skip]':20s}  {name} — {path.name} not found")

    if "MLP-Factor" not in loaded:
        print(
            "\nERROR: nowcast_results_mlp_factor.csv not found.\n"
            "Run mlp_utils.py first."
        )
        return

    # Compute metrics for each model
    rows: list[dict] = []
    for name, df in loaded.items():
        r_full = compute_rmsfe(df, EVAL_START, EVAL_END, month_in_quarter=M3)
        r_reg  = {
            k: compute_rmsfe(df, q0, q1, month_in_quarter=M3)
            for k, (q0, q1) in REGIMES.items()
        }
        nsr    = compute_nsr(df, y_q, EVAL_START, EVAL_END)
        mz     = mincer_zarnowitz_test(df, EVAL_START, EVAL_END, month_in_quarter=M3)
        dm_ar1 = (
            dm_p(df, loaded["AR1"])
            if "AR1" in loaded and name not in ("AR1", "RW")
            else np.nan
        )
        dm_en = (
            dm_p(df, loaded["DFM-EN"])
            if "DFM-EN" in loaded and name != "DFM-EN"
            else np.nan
        )
        rows.append({
            "model":            name,
            "RMSFE_full":       r_full,
            "RMSFE_pre-COVID":  r_reg["pre-COVID"],
            "RMSFE_COVID":      r_reg["COVID"],
            "RMSFE_post-COVID": r_reg["post-COVID"],
            "NSR":              nsr,
            "MZ_alpha":         mz["alpha"],
            "MZ_beta":          mz["beta"],
            "MZ_p_joint":       mz["p_joint_wald"],
            "DM_p_vs_AR1":      dm_ar1,
            "DM_p_vs_DFM-EN":   dm_en,
        })

    tbl = pd.DataFrame(rows).set_index("model").sort_values("RMSFE_full")
    tbl.round(4).to_csv(OUT_TABLE)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n" + "=" * 100)
    print("Combined nowcast evaluation  (M3, 2011Q1–2025Q4)")
    print("Regime cutoffs: pre-COVID 2011Q1–2019Q4 | COVID 2020Q1–2021Q4 | post-COVID 2022Q1–2025Q4")
    print("=" * 100)
    print(tbl.round(3).to_string())
    print("\nColumns: RMSFE in pp | NSR = RMSFE/SD(actual) | MZ_beta ≈ 1 → unbiased")
    print("DM p-values: two-sided Harvey-Leybourne-Newbold; < 0.10 → significant diff")
    print(f"\nSaved → {OUT_TABLE}")

    # -------------------------------------------------------------------
    # Highlight MLP-Factor row
    # -------------------------------------------------------------------
    if "MLP-Factor" in tbl.index:
        row = tbl.loc["MLP-Factor"]
        dfm_rmsfe = tbl.loc["DFM-EN", "RMSFE_full"] if "DFM-EN" in tbl.index else np.nan
        ar1_rmsfe = tbl.loc["AR1",    "RMSFE_full"] if "AR1"    in tbl.index else np.nan
        print("\n--- MLP-Factor summary ---")
        print(f"  RMSFE full-window : {row['RMSFE_full']:.4f} pp")
        if not np.isnan(dfm_rmsfe):
            print(f"  vs DFM-EN         : {row['RMSFE_full'] / dfm_rmsfe:.4f}  "
                  f"(DM p={row['DM_p_vs_DFM-EN']:.3f})")
        if not np.isnan(ar1_rmsfe):
            print(f"  vs AR1            : {row['RMSFE_full'] / ar1_rmsfe:.4f}  "
                  f"(DM p={row['DM_p_vs_AR1']:.3f})")
        print(f"  NSR               : {row['NSR']:.4f}")
        print(f"  MZ beta           : {row['MZ_beta']:.4f}  (p_joint={row['MZ_p_joint']:.3f})")
        print(f"  RMSFE pre-COVID   : {row['RMSFE_pre-COVID']:.4f}")
        print(f"  RMSFE post-COVID  : {row['RMSFE_post-COVID']:.4f}")


if __name__ == "__main__":
    main()
