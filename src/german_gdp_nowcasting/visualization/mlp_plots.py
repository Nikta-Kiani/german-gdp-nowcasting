"""Thesis-quality summary figures for the nowcasting comparison.

Two figures:
  1. thesis_01_rmsfe_by_regime_clean.png   -- headline model set (incl. MLP-Factor)
  2. thesis_02_mlp_linearity_diagnostic.png -- non-linearity test: the
     factor-augmented MLP and gradient boosting track the *linear* DFM-EN in
     calm periods and diverge only on the COVID outliers, evidence that the
     factor->GDP map is effectively linear at the available sample size.

Run with ``python -m german_gdp_nowcasting.visualization.mlp_plots``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import paths as _tp
from ..models.dfm.nowcast_utils import subset_eval_window
from . import xgb_plots as xp
from .nowcast_plots import _apply_year_axis, _q_to_ts, add_recession_bands

FIG = _tp.NOWCAST_FIGURES_DIR
OVERLEAF_FIG = _tp.THESIS_FIGURES

REGIMES: dict[str, tuple[str, str]] = {
    "pre-COVID":  ("2011Q1", "2019Q4"),
    "COVID":      ("2020Q1", "2021Q4"),
    "post-COVID": ("2022Q1", "2025Q4"),
}

MLP_PALETTE: dict[str, str] = {
    "MLP-Factor": "#5A3994",
}
MLP_LABELS: dict[str, str] = {
    "MLP-Factor": "MLP-Factor",
}
xp.XGB_MODEL_COLORS.update(MLP_PALETTE)
xp.XGB_MODEL_LABELS.update(MLP_LABELS)


def _load(path: Path, name: str) -> pd.DataFrame | None:
    """Load one result file, returning ``None`` when it is unavailable."""
    if not path.exists():
        print(f"  [skip] {name}: {path.name} not found")
        return None
    return pd.read_csv(path)


def load_all() -> dict[str, pd.DataFrame]:
    """Load every available headline-model result file."""
    candidates: dict[str, Path] = {
        "RW": _tp.RW_RESULTS_CSV,
        "AR1": _tp.AR1_RESULTS_CSV,
        "DFM-EN": _tp.actpn_results_csv("en_only"),
        "DFM-ifoCAST": _tp.IFO_RESULTS_CSV,
        "DFM-BlockBalanced": _tp.BLOCKBALANCED_RESULTS_CSV,
        "DFM-SV-k2": _tp.ACTPN_SV_RESULTS_K2_CSV,
        "combo_equal": _tp.COMBO_EQUAL_PATH_CSV,
        "XGB-Full": _tp.xgb_results_csv("full"),
        "MLP-Factor": _tp.MLP_FACTOR_RESULTS_CSV,
    }
    out: dict[str, pd.DataFrame] = {}
    for name, path in candidates.items():
        df = _load(path, name)
        if df is not None:
            out[name] = df
    print(f"Loaded {len(out)} model CSVs: {list(out.keys())}")
    return out


CLEAN_MODEL_ORDER: list[str] = [
    "RW", "AR1", "DFM-ifoCAST", "DFM-EN", "DFM-BlockBalanced",
    "DFM-SV-k2", "combo_equal", "XGB-Full", "MLP-Factor",
]

CLEAN_LABELS: dict[str, str] = {
    "RW": "RW",
    "AR1": "AR(1)",
    "DFM-ifoCAST": "DFM-ifoCAST",
    "DFM-EN": "DFM-EN",
    "DFM-BlockBalanced": "DFM-k20",
    "DFM-SV-k2": "DFM-SV (k=2, integrated)",
    "combo_equal": "Equal combo",
    "XGB-Full": "XGB-Full",
    "MLP-Factor": "MLP-Factor",
}

CLEAN_COLORS: dict[str, str] = {
    "RW": xp.XGB_MODEL_COLORS["RW"],
    "AR1": xp.XGB_MODEL_COLORS["AR1"],
    "DFM-ifoCAST": "#C9617F",
    "DFM-EN": xp.XGB_MODEL_COLORS["DFM-EN"],
    "DFM-BlockBalanced": "#8F3D58",
    "DFM-SV-k2": xp.XGB_MODEL_COLORS["DFM-SV-k2"],
    "combo_equal": "#D4A574",
    "XGB-Full": xp.XGB_MODEL_COLORS["XGB-Full"],
    "MLP-Factor": MLP_PALETTE["MLP-Factor"],
}


def _regime_rmsfe(
    results: dict[str, pd.DataFrame],
    models: list[str],
    regimes: dict[str, tuple[str, str]],
    miq: int = 3,
) -> pd.DataFrame:
    """Pivot table: regimes x models (M3 RMSFE). Uses shared eval-window helper."""
    rows = []
    for m in models:
        if m not in results:
            continue
        df = results[m]
        for label, (q0, q1) in regimes.items():
            sub = subset_eval_window(
                df, eval_start=q0, eval_end=q1, month_in_quarter=miq,
            )
            errs = sub["error"].dropna()
            val = (
                float(np.sqrt(np.mean(errs.values ** 2)))
                if len(errs) else np.nan
            )
            rows.append({"model": m, "regime": label, "RMSFE": val})
    tbl = pd.DataFrame(rows).pivot(index="regime", columns="model", values="RMSFE")
    return tbl.reindex(list(regimes.keys()))


def plot_clean_comparison(
    results: dict[str, pd.DataFrame],
    save: Path | None = None,
) -> plt.Figure:
    """Plot headline M3 RMSFE by economic regime."""
    xp.setup_style()
    models_present = [m for m in CLEAN_MODEL_ORDER if m in results]
    tbl = _regime_rmsfe(results, models_present, REGIMES)

    n_m = len(models_present)
    width = 0.78 / max(n_m, 1)
    x = np.arange(len(REGIMES))

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, m in enumerate(models_present):
        vals = tbl[m].values if m in tbl.columns else np.full(len(REGIMES), np.nan)
        ax.bar(
            x + (i - (n_m - 1) / 2) * width,
            vals,
            width,
            color=CLEAN_COLORS.get(m, "#94A3B8"),
            edgecolor="white",
            linewidth=0.7,
            label=CLEAN_LABELS.get(m, m),
            zorder=3,
        )

    for i, m in enumerate(models_present):
        if m not in tbl.columns:
            continue
        for j, v in enumerate(tbl[m].values):
            if np.isnan(v):
                continue
            ax.text(
                x[j] + (i - (n_m - 1) / 2) * width,
                v + 0.04,
                f"{v:.2f}",
                ha="center", va="bottom",
                fontsize=5.8, color="#374151",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(list(REGIMES.keys()), ha="center")
    ax.set_ylabel("RMSFE (pp)")
    ax.set_title(
        "Predictive accuracy by economic regime — headline models (RMSFE, M3)",
        fontsize=11, fontweight="600",
    )
    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.98))
    if save:
        fig.savefig(save, dpi=220, bbox_inches="tight")
        print(f"Saved: {save}")
    return fig


def _m3_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return one M3-indexed row per quarter."""
    if "month_in_quarter" in df.columns:
        df = df.loc[df["month_in_quarter"] == 3].copy()
    if "quarter" in df.columns:
        df = df.set_index("quarter")
    return df


def _ts(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Convert a quarterly result frame's index to timestamps."""
    return _q_to_ts(df.index)


def plot_mlp_linearity_diagnostic(
    results: dict[str, pd.DataFrame],
    save: Path | None = None,
) -> plt.Figure:
    """Linearity test: DFM-EN vs XGB-Full vs MLP-Factor on calm vs COVID quarters."""
    xp.setup_style()

    dfm_en_df = _m3_df(results["DFM-EN"]) if "DFM-EN" in results else None
    xgb_df = _m3_df(results["XGB-Full"]) if "XGB-Full" in results else None
    mlp_df = _m3_df(results["MLP-Factor"]) if "MLP-Factor" in results else None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.2), sharex=False)
    fig.subplots_adjust(hspace=0.46)

    ax1.axhline(0, color="#CBD5E1", lw=0.7, zorder=1)
    add_recession_bands(ax1, annotate=True)

    _ref = dfm_en_df if dfm_en_df is not None else mlp_df
    if _ref is not None:
        x_act = _ts(_ref)
        ax1.plot(
            x_act, _ref["actual"].values,
            color="#1A2332", lw=2.0, label="Actual GDP", zorder=4,
        )

    if dfm_en_df is not None:
        ax1.plot(
            _ts(dfm_en_df), dfm_en_df["nowcast"].values,
            color=CLEAN_COLORS["DFM-EN"], lw=1.7, label="DFM-EN", zorder=3,
        )

    if xgb_df is not None:
        ax1.plot(
            _ts(xgb_df), xgb_df["nowcast"].values,
            color=CLEAN_COLORS["XGB-Full"], lw=1.5, ls="--",
            label="XGB-Full", zorder=3,
        )

    if mlp_df is not None:
        ax1.plot(
            _ts(mlp_df), mlp_df["nowcast"].values,
            color=MLP_PALETTE["MLP-Factor"], lw=1.5,
            label="MLP-Factor", zorder=3,
        )

    ax1.set_ylabel("Q/Q log-growth (pp)")
    ax1.set_title(
        "Nowcast vs actual GDP — non-linear learners track the linear DFM in calm quarters",
        fontsize=10.5,
    )
    ax1.legend(loc="upper left", framealpha=0.88, fontsize=8, ncol=2)
    _apply_year_axis(ax1, base=2)

    ax2.axhline(0, color="#CBD5E1", lw=0.7, zorder=1)
    add_recession_bands(ax2, annotate=False)

    try:
        ax2.axvspan(
            pd.Timestamp("2020-04-01"), pd.Timestamp("2021-12-31"),
            color="#FEF3C7", alpha=0.55, zorder=0,
            label="COVID shock in training data (2020Q2–2021Q4)",
        )
    except Exception:
        pass

    if dfm_en_df is not None:
        ax2.plot(
            _ts(dfm_en_df), dfm_en_df["error"].values,
            color=CLEAN_COLORS["DFM-EN"], lw=1.4,
            label="DFM-EN error", zorder=4,
        )

    if xgb_df is not None:
        ax2.plot(
            _ts(xgb_df), xgb_df["error"].values,
            color=CLEAN_COLORS["XGB-Full"], lw=1.4, ls="--",
            label="XGB-Full error", zorder=4,
        )

    if mlp_df is not None:
        ax2.plot(
            _ts(mlp_df), mlp_df["error"].values,
            color=MLP_PALETTE["MLP-Factor"], lw=1.4,
            label="MLP-Factor error", zorder=4,
        )

    ax2.set_ylabel("Forecast error (pp)")
    ax2.set_title(
        "Signed errors — the gap opens only on the 2020Q2/Q3 outliers",
        fontsize=10.5,
    )
    ax2.legend(loc="upper left", framealpha=0.88, fontsize=8, ncol=2)
    _apply_year_axis(ax2, base=2)

    fig.text(
        0.5, 0.01,
        "With the panel already summarised by two DFM factors, a non-linear "
        "factor-to-GDP map does not improve on the linear measurement equation; "
        "flexible learners over-fit the pandemic tail once those outliers enter "
        "the expanding training set.",
        ha="center", va="bottom", fontsize=7.5, color="#475569",
        wrap=True,
    )
    if save:
        fig.savefig(save, dpi=220, bbox_inches="tight")
        print(f"Saved: {save}")
    return fig


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    OVERLEAF_FIG.mkdir(parents=True, exist_ok=True)
    xp.setup_style()
    results = load_all()

    if len(results) < 2:
        print("Too few CSVs available — run the backtests first.")
        sys.exit(1)

    for fname, plot_fn in [
        ("thesis_01_rmsfe_by_regime_clean.png", plot_clean_comparison),
        ("thesis_02_mlp_linearity_diagnostic.png", plot_mlp_linearity_diagnostic),
    ]:
        fig = plot_fn(results, save=FIG / fname)
        plt.close(fig)
        overleaf_path = OVERLEAF_FIG / fname
        overleaf_path.write_bytes((FIG / fname).read_bytes())
        print(f"Copied to thesis figures: {overleaf_path}")

    print("Done.")
