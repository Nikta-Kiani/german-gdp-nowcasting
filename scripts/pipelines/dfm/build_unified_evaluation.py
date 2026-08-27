"""Build unified M3-only nowcasting evaluation tables and thesis diagnostics.

Reads saved nowcast CSVs (no model re-runs) and writes:

  - rmsfe_table_all_models.csv       headline cross-model comparison (M3)
  - rmsfe_table.csv                  DFM-focused table (M3 + pooled)
  - diebold_mariano_table.csv        pairwise DM (full sample)
  - diebold_mariano_table_all_models.csv  headline DM pairs incl. XGB/MLP
  - diebold_mariano_subwindows.csv   pre-COVID / post-COVID DM robustness
  - model_confidence_set_table.csv   90% MCS for headline models (M3)
  - mincer_zarnowitz_table.csv       bias/efficiency regressions
  - dfm_en_forecast_revision.csv     M1/M2/M3 nowcast evolution
  - figures/thesis_05_dfm_en_revision.png

COVID regime cutoffs (used in all sub-window tables):
  pre-COVID  : 2011Q1 – 2019Q4
  COVID      : 2020Q1 – 2021Q4  (excluded from DM sub-window table)
  post-COVID : 2022Q1 – 2025Q4

Run (from the repository root):
    python scripts/pipelines/dfm/build_unified_evaluation.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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
from german_gdp_nowcasting.models.dfm.nowcast_utils import (  # noqa: E402
    align_forecast_errors,
    build_forecast_loss_matrix,
    build_interval_calibration_table,
    compute_model_confidence_set,
    compute_rmsfe,
    compute_nsr,
    diebold_mariano_test,
    expand_quarterly_nowcasts_to_monthly,
    mincer_zarnowitz_test,
)

EVAL_START = "2011Q1"
EVAL_END = "2025Q4"
HEADLINE_MIQ = 3

REGIMES: dict[str, tuple[str, str]] = {
    "pre-COVID": ("2011Q1", "2019Q4"),
    "COVID": ("2020Q1", "2021Q4"),
    "post-COVID": ("2022Q1", "2025Q4"),
}

# Headline model set for the thesis (M3, 2011Q1–2025Q4).
HEADLINE_MODELS: dict[str, Path | None] = {
    "DFM-ifoCAST": P.IFO_RESULTS_CSV,
    "DFM-EN": P.actpn_results_csv("en_only"),
    "DFM-PLS": P.actpn_results_csv("pls_only"),
    "DFM-BlockBalanced": P.BLOCKBALANCED_RESULTS_CSV,
    "DFM-TVP": P.TVP_RESULTS_CSV,
    "DFM-SV-k2": P.ACTPN_SV_RESULTS_K2_CSV,
    "combo_equal": P.COMBO_EQUAL_PATH_CSV,
    "XGB-Full": P.xgb_results_csv("full"),
    "MLP-Factor": P.MLP_FACTOR_RESULTS_CSV,
    "AR1": P.AR1_RESULTS_CSV,
    "RW": P.RW_RESULTS_CSV,
}

DFM_TABLE_MODELS: dict[str, Path | None] = {
    "DFM-EN": P.actpn_results_csv("en_only"),
    "DFM-PLS": P.actpn_results_csv("pls_only"),
    "DFM-ifoCAST": P.IFO_RESULTS_CSV,
    "DFM-BlockBalanced": P.BLOCKBALANCED_RESULTS_CSV,
    "DFM-SV-k2": P.ACTPN_SV_RESULTS_K2_CSV,
    "combo_equal": P.COMBO_EQUAL_PATH_CSV,
    "AR1": P.AR1_RESULTS_CSV,
    "RW": P.RW_RESULTS_CSV,
}

SV_INTERVAL_MODELS: dict[str, Path] = {
    "DFM-SV-k2": P.ACTPN_SV_RESULTS_K2_CSV,
}

HEADLINE_DM_PAIRS: list[tuple[str, str]] = [
    ("DFM-EN", "DFM-ifoCAST"),
    ("DFM-EN", "AR1"),
    ("XGB-Full", "DFM-EN"),
    ("MLP-Factor", "DFM-EN"),
]


def load_nowcast_csv(path: Path) -> pd.DataFrame:
    """Load a saved nowcast CSV with DFM-compatible columns."""
    df = pd.read_csv(path)
    if "month_in_quarter" not in df.columns:
        df = expand_quarterly_nowcasts_to_monthly(df)
    if "monthly_origin" in df.columns and df.index.name != "monthly_origin":
        df = df.set_index("monthly_origin", drop=False)
    return df


def load_gdp_target() -> pd.Series:
    """Load the quarterly first-release GDP target."""
    raw = pd.read_csv(P.GDP_TARGET_CSV)
    raw["quarter"] = pd.PeriodIndex(raw["quarter"], freq="Q")
    return raw.set_index("quarter").iloc[:, 0]


def dm_pvalue(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    eval_start: str | None = None,
    eval_end: str | None = None,
) -> float:
    """Diebold-Mariano p-value (model_a vs model_b) at M3."""
    ea, eb = align_forecast_errors(
        df_a, df_b,
        month_in_quarter=HEADLINE_MIQ,
        eval_start=eval_start,
        eval_end=eval_end,
    )
    return diebold_mariano_test(ea, eb)["p_value"]


def build_headline_rmsfe_table(
    models: dict[str, pd.DataFrame],
    y_quarterly: pd.Series,
) -> pd.DataFrame:
    """M3-only RMSFE table with NSR, vs-AR1 ratio, and DM vs AR1."""
    ar1_df = models.get("AR1")
    ar1_rmsfe = compute_rmsfe(
        ar1_df, eval_start=EVAL_START, eval_end=EVAL_END, month_in_quarter=HEADLINE_MIQ,
    ) if ar1_df is not None else np.nan

    rows = []
    for name, df in models.items():
        if df is None:
            continue
        rmsfe = compute_rmsfe(
            df, eval_start=EVAL_START, eval_end=EVAL_END, month_in_quarter=HEADLINE_MIQ,
        )
        nsr = compute_nsr(df, y_quarterly, eval_start=EVAL_START, eval_end=EVAL_END)
        vs_ar1 = rmsfe / ar1_rmsfe if ar1_df is not None and not np.isnan(ar1_rmsfe) else np.nan
        dm_ar1 = (
            dm_pvalue(df, ar1_df)
            if ar1_df is not None and name not in ("AR1", "RW")
            else np.nan
        )
        rows.append({
            "model": name,
            "RMSFE_M3": round(rmsfe, 4),
            "NSR": round(nsr, 4) if not np.isnan(nsr) else np.nan,
            "vs_AR1": round(vs_ar1, 4) if not np.isnan(vs_ar1) else np.nan,
            "DM_p_vs_AR1": round(dm_ar1, 4) if not np.isnan(dm_ar1) else np.nan,
        })

    return pd.DataFrame(rows).set_index("model").sort_values("RMSFE_M3")


def build_dfm_rmsfe_table(
    models: dict[str, pd.DataFrame],
    y_quarterly: pd.Series,
) -> pd.DataFrame:
    """DFM-focused table: M3 + pooled M1–M3."""
    ar1_df = models["AR1"]
    ar1_rmsfe = compute_rmsfe(
        ar1_df, eval_start=EVAL_START, eval_end=EVAL_END, month_in_quarter=HEADLINE_MIQ,
    )

    rows = []
    for name, df in models.items():
        if df is None:
            continue
        r3 = compute_rmsfe(
            df, eval_start=EVAL_START, eval_end=EVAL_END, month_in_quarter=HEADLINE_MIQ,
        )
        rp = compute_rmsfe(df, eval_start=EVAL_START, eval_end=EVAL_END, month_in_quarter=None)
        nsr = compute_nsr(df, y_quarterly, eval_start=EVAL_START, eval_end=EVAL_END)
        vs_ar1 = r3 / ar1_rmsfe
        dm_ar1 = dm_pvalue(df, ar1_df) if name not in ("AR1", "RW") else np.nan
        rows.append({
            "model": name,
            "RMSFE_M3": round(r3, 4),
            "RMSFE_pooled": round(rp, 4),
            "vs AR1": round(vs_ar1, 3),
            "NSR": round(nsr, 3),
            "DM p (vs AR1)": round(dm_ar1, 3) if not np.isnan(dm_ar1) else np.nan,
        })

    return pd.DataFrame(rows).set_index("model").sort_values("RMSFE_M3")


def build_pairwise_dm_table(
    models: dict[str, pd.DataFrame],
    model_names: list[str],
    eval_start: str | None = None,
    eval_end: str | None = None,
) -> pd.DataFrame:
    """Symmetric pairwise DM p-value matrix."""
    idx = pd.Index(model_names, name="model")
    mat = pd.DataFrame(np.nan, index=idx, columns=idx.copy())
    for a, b in combinations(model_names, 2):
        if a not in models or b not in models:
            continue
        p = dm_pvalue(models[a], models[b], eval_start=eval_start, eval_end=eval_end)
        mat.loc[a, b] = p
        mat.loc[b, a] = p
    return mat.round(3)


def build_mz_table(models: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mincer-Zarnowitz regressions for all headline models."""
    rows = []
    for name, df in models.items():
        if df is None or name in ("AR1", "RW"):
            continue
        mz = mincer_zarnowitz_test(
            df, eval_start=EVAL_START, eval_end=EVAL_END, month_in_quarter=HEADLINE_MIQ,
        )
        rows.append({
            "model": name,
            "alpha": round(mz["alpha"], 4),
            "beta": round(mz["beta"], 4),
            "se_alpha": round(mz["se_alpha"], 4),
            "se_beta": round(mz["se_beta"], 4),
            "p_alpha_zero": round(mz["p_alpha_zero"], 4),
            "p_joint_H0_a0_b1": round(mz["p_joint_wald"], 4),
            "n": mz["n"],
        })
    return pd.DataFrame(rows).set_index("model")


def build_forecast_revision_table(df_en: pd.DataFrame) -> pd.DataFrame:
    """Pivot DFM-EN nowcasts to M1/M2/M3 columns per quarter."""
    sub = df_en[df_en["quarter"].between(EVAL_START, EVAL_END)].copy()
    wide = sub.pivot_table(
        index="quarter", columns="month_in_quarter",
        values=["nowcast", "error"], aggfunc="first",
    )
    wide.columns = [f"{v}_M{int(m)}" for v, m in wide.columns]
    wide = wide.reset_index()
    wide["actual"] = sub.groupby("quarter")["actual"].first().reindex(wide["quarter"]).values
    wide["revision_M1_to_M3"] = wide["nowcast_M3"] - wide["nowcast_M1"]
    wide["abs_revision_M1_to_M3"] = wide["revision_M1_to_M3"].abs()
    return wide.sort_values("quarter")


def plot_forecast_revision(rev: pd.DataFrame, save: Path) -> None:
    """Line plot of DFM-EN M1/M2/M3 nowcasts vs actual."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    q_ts = pd.PeriodIndex(rev["quarter"], freq="Q").to_timestamp(how="end")

    ax.plot(q_ts, rev["actual"], "k-", lw=1.8, label="Actual", zorder=5)
    ax.plot(q_ts, rev["nowcast_M1"], "--", color="#E07D96", lw=1.0, alpha=0.85, label="M1")
    ax.plot(q_ts, rev["nowcast_M2"], "-.", color="#C9617F", lw=1.0, alpha=0.85, label="M2")
    ax.plot(q_ts, rev["nowcast_M3"], "-", color="#8F3D58", lw=1.2, label="M3")

    covid_lo = float(mdates.date2num(pd.Timestamp("2020-01-01")))
    covid_hi = float(mdates.date2num(pd.Timestamp("2021-12-31")))
    ax.axvspan(covid_lo, covid_hi, color="grey", alpha=0.12, label="COVID (2020Q1–2021Q4)")
    ax.set_title(
        r"DFM-EN within-quarter nowcast revision ($M1 \to M2 \to M3$)" + "\n"
        f"Evaluation: {EVAL_START}–{EVAL_END}  |  "
        "pre-COVID 2011Q1–2019Q4  |  post-COVID 2022Q1–2025Q4"
    )
    ax.set_ylabel("QoQ GDP growth (pp)")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save.name}")


def main() -> None:
    """Build and persist all unified evaluation tables and diagnostics."""
    print("Loading saved nowcast CSVs …")
    y_q = load_gdp_target()

    loaded: dict[str, pd.DataFrame] = {}
    for name, path in HEADLINE_MODELS.items():
        if path is None:
            continue
        if not path.exists():
            print(f"  [skip] {name}: {path.name} not found")
            continue
        loaded[name] = load_nowcast_csv(path)
        print(f"  loaded {name}: {len(loaded[name])} rows")

    headline = build_headline_rmsfe_table(loaded, y_q)
    headline.to_csv(P.RMSFE_TABLE_ALL_CSV)
    print(f"\nWrote {P.RMSFE_TABLE_ALL_CSV.name}")
    print(headline.to_string())

    dfm_loaded: dict[str, pd.DataFrame] = {}
    for name, path in DFM_TABLE_MODELS.items():
        if path is None or not path.exists():
            continue
        dfm_loaded[name] = load_nowcast_csv(path)
    dfm_tbl = build_dfm_rmsfe_table(dfm_loaded, y_q)
    dfm_tbl.to_csv(P.RMSFE_TABLE_CSV)
    print(f"\nWrote {P.RMSFE_TABLE_CSV.name}")

    dm_dfm_names = list(dfm_loaded.keys())
    dm_dfm = build_pairwise_dm_table(dfm_loaded, dm_dfm_names)
    dm_dfm.to_csv(P.DM_TABLE_CSV)
    print(f"Wrote {P.DM_TABLE_CSV.name}")

    headline_names = list(loaded.keys())
    dm_all = build_pairwise_dm_table(loaded, headline_names)
    dm_all.to_csv(P.DM_TABLE_ALL_CSV)

    pair_rows = []
    for a, b in HEADLINE_DM_PAIRS:
        if a in loaded and b in loaded:
            pair_rows.append({
                "model_a": a, "model_b": b,
                "window": "full",
                "eval_start": EVAL_START, "eval_end": EVAL_END,
                "DM_p_value": round(dm_pvalue(loaded[a], loaded[b]), 4),
            })
    print(f"Wrote {P.DM_TABLE_ALL_CSV.name}")

    for reg, (q0, q1) in REGIMES.items():
        if reg == "COVID":
            continue
        for a, b in HEADLINE_DM_PAIRS + [("DFM-EN", "AR1"), ("DFM-BlockBalanced", "DFM-ifoCAST")]:
            if a not in loaded or b not in loaded:
                continue
            pair_rows.append({
                "model_a": a, "model_b": b,
                "window": reg,
                "eval_start": q0, "eval_end": q1,
                "DM_p_value": round(dm_pvalue(loaded[a], loaded[b], q0, q1), 4),
            })
    pd.DataFrame(pair_rows).to_csv(P.DM_SUBWINDOW_TABLE_CSV, index=False)
    print(f"Wrote {P.DM_SUBWINDOW_TABLE_CSV.name}")

    losses = build_forecast_loss_matrix(
        loaded,
        eval_start=EVAL_START,
        eval_end=EVAL_END,
        month_in_quarter=HEADLINE_MIQ,
        loss="se",
    )
    mcs = compute_model_confidence_set(losses)
    mcs.insert(0, "n", len(losses))
    mcs.insert(1, "RMSFE", np.sqrt(mcs["mean_loss"]))
    mcs.to_csv(P.MCS_TABLE_CSV)
    print(f"Wrote {P.MCS_TABLE_CSV.name}")

    mz_tbl = build_mz_table(loaded)
    mz_tbl.to_csv(P.MINCER_ZARNOWITZ_CSV)
    print(f"Wrote {P.MINCER_ZARNOWITZ_CSV.name}")

    if "DFM-EN" in loaded:
        rev = build_forecast_revision_table(loaded["DFM-EN"])
        rev.to_csv(P.FORECAST_REVISION_CSV, index=False)
        P.NOWCAST_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_forecast_revision(rev, P.FORECAST_REVISION_FIG)
        print(f"Wrote {P.FORECAST_REVISION_CSV.name}")
        print(
            f"  Mean |M1->M3 revision| = {rev['abs_revision_M1_to_M3'].mean():.3f} pp  "
            f"(post-COVID: {rev.loc[rev['quarter'] >= '2022Q1', 'abs_revision_M1_to_M3'].mean():.3f} pp)"
        )

    sv_loaded = {
        k: load_nowcast_csv(v) for k, v in SV_INTERVAL_MODELS.items() if v.exists()
    }
    if sv_loaded:
        sv_tbl = build_interval_calibration_table(
            sv_loaded, eval_start=EVAL_START, eval_end=EVAL_END,
        )
        sv_tbl.to_csv(P.SV_INTERVAL_TABLE_CSV)
        print(f"Wrote {P.SV_INTERVAL_TABLE_CSV.name}")

    xgb_compat = headline.rename(columns={
        "RMSFE_M3": "RMSFE",
        "DM_p_vs_AR1": "DM_vs_AR1_p",
    })
    xgb_compat.to_csv(P.RMSFE_TABLE_XGB_CSV)
    print(f"Wrote {P.RMSFE_TABLE_XGB_CSV.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
