"""Diagnose the post-COVID within-quarter reversal (M1 more accurate than M3).

Two complementary diagnostics, computed per regime and per DFM specification:

1. Bias-variance decomposition of RMSFE at M1, M2, M3
   RMSFE^2 = bias^2 + variance, evaluated on the (nowcast - actual) errors at
   each month-in-quarter. This isolates whether the M1->M3 change in RMSFE is
   driven by a shrinking/growing systematic error (bias) or by a
   shrinking/growing dispersion of errors around their mean (variance).

2. Revision informativeness: corr(nowcast_M3 - nowcast_M1, actual - nowcast_M1)
   The M1->M3 revision is mechanically driven by real hard-data releases
   overwriting AR(p)-bridge-filled cells (see ragged_edge.py) as the quarter
   progresses. This correlation measures whether that revision moves the
   nowcast *towards* the eventual outcome (positive, ideally close to +1) or
   is uninformative / harmful (near zero or negative).

Outputs
-------
  outputs/nowcasting/horizon_bias_variance_table.csv
  outputs/nowcasting/horizon_revision_informativeness_table.csv

Run (from the repository root):
    python scripts/pipelines/dfm/run_horizon_bias_variance.py
"""

from __future__ import annotations

import sys
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

from german_gdp_nowcasting.config import paths as P  # noqa: E402

REGIMES: dict[str, tuple[str, str]] = {
    "pre-COVID": ("2011Q1", "2019Q4"),
    "COVID": ("2020Q1", "2021Q4"),
    "post-COVID": ("2022Q1", "2025Q4"),
}

MODELS: dict[str, Path] = {
    "DFM-EN": P.actpn_results_csv("en_only"),
    "DFM-SV-k2": P.ACTPN_SV_RESULTS_K2_CSV,
    "DFM-ifoCAST": P.IFO_RESULTS_CSV,
    "DFM-BlockBalanced": P.BLOCKBALANCED_RESULTS_CSV,
    "DFM-TVP": P.TVP_RESULTS_CSV,
}


def _errors_by_month(df: pd.DataFrame, q0: str, q1: str) -> pd.DataFrame:
    """Wide (quarter x M1/M2/M3) DataFrame of forecast errors within [q0, q1]."""
    idx = pd.PeriodIndex(df["quarter"].astype(str), freq="Q")
    mask = (idx >= pd.Period(q0)) & (idx <= pd.Period(q1))
    sub = df.loc[np.asarray(mask)]
    piv = sub.pivot_table(index="quarter", columns="month_in_quarter", values="error")
    piv.columns = [f"M{c}" for c in piv.columns]
    return piv


def bias_variance_table() -> pd.DataFrame:
    """Bias/variance/RMSFE decomposition at M1, M2, M3 for every model x regime."""
    rows = []
    for name, path in MODELS.items():
        if not Path(path).exists():
            print(f"[skip] {name}: {path.name} not found")
            continue
        df = pd.read_csv(path)
        if "month_in_quarter" not in df.columns:
            continue
        for reg, (q0, q1) in REGIMES.items():
            piv = _errors_by_month(df, q0, q1)
            for m in ("M1", "M2", "M3"):
                if m not in piv.columns:
                    continue
                e = piv[m].dropna().to_numpy(dtype=float)
                if len(e) == 0:
                    continue
                bias = float(np.mean(e))
                var = float(np.var(e))
                rmsfe2 = float(np.mean(e ** 2))
                rows.append({
                    "model": name, "regime": reg, "month_in_quarter": m,
                    "n": len(e), "bias": round(bias, 4),
                    "bias_sq": round(bias ** 2, 4),
                    "variance": round(var, 4),
                    "RMSFE": round(float(np.sqrt(rmsfe2)), 4),
                    "bias_sq_share_pct": round(100.0 * bias ** 2 / rmsfe2, 1) if rmsfe2 > 0 else np.nan,
                })
    return pd.DataFrame(rows)


def revision_informativeness_table() -> pd.DataFrame:
    """corr(M3-M1 nowcast revision, actual-M1 nowcast) per model x regime.

    A value near +1 means the within-quarter revision (driven by real hard
    data overwriting AR-bridge-filled cells) moves the nowcast toward the
    eventual outcome; a value near 0 (or negative) means the revision is
    uninformative noise, or actively harmful, relative to the M1 nowcast.
    """
    rows = []
    for name, path in MODELS.items():
        if not Path(path).exists():
            continue
        df = pd.read_csv(path)
        if "month_in_quarter" not in df.columns:
            continue
        idx_all = pd.PeriodIndex(df["quarter"].astype(str), freq="Q")
        for reg, (q0, q1) in REGIMES.items():
            mask = (idx_all >= pd.Period(q0)) & (idx_all <= pd.Period(q1))
            sub = df.loc[np.asarray(mask)]
            nc = sub.pivot_table(index="quarter", columns="month_in_quarter", values="nowcast")
            act = sub.pivot_table(index="quarter", columns="month_in_quarter", values="actual")
            if 1 not in nc.columns or 3 not in nc.columns or 3 not in act.columns:
                continue
            common = nc.index.intersection(act.index)
            nc, act = nc.loc[common], act.loc[common]
            revision = nc[3] - nc[1]
            truth_surprise = act[3] - nc[1]
            valid = revision.notna() & truth_surprise.notna()
            if valid.sum() < 3:
                continue
            corr = float(np.corrcoef(revision[valid], truth_surprise[valid])[0, 1])
            rows.append({
                "model": name, "regime": reg, "n": int(valid.sum()),
                "corr_revision_vs_truth_surprise": round(corr, 3),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    bv = bias_variance_table()
    out_bv = P.OUT_NOWCASTING / "horizon_bias_variance_table.csv"
    bv.to_csv(out_bv, index=False)
    print(f"Saved: {out_bv}\n")
    wide_bias = bv.pivot_table(index=["regime", "model"], columns="month_in_quarter", values="bias")
    wide_var = bv.pivot_table(index=["regime", "model"], columns="month_in_quarter", values="variance")
    print("--- bias (signed mean error) ---")
    print(wide_bias.round(3).to_string())
    print("\n--- variance ---")
    print(wide_var.round(3).to_string())

    rev = revision_informativeness_table()
    out_rev = P.OUT_NOWCASTING / "horizon_revision_informativeness_table.csv"
    rev.to_csv(out_rev, index=False)
    print(f"\nSaved: {out_rev}\n")
    print(rev.pivot_table(index="model", columns="regime", values="corr_revision_vs_truth_surprise").round(3).to_string())
