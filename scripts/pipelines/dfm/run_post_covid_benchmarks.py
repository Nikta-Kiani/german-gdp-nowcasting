"""Generate post-COVID benchmark results table and figure.

Writes:
  outputs/nowcasting/post_covid_benchmarks_table.csv
  outputs/nowcasting/figures/thesis_03_post_covid_improvement.png

Run (from the repository root):
    python scripts/pipelines/dfm/run_post_covid_benchmarks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

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
from german_gdp_nowcasting.models.dfm import post_covid_benchmarks as B  # noqa: E402

FIG = P.NOWCAST_FIGURES_DIR


def _load_m3_optional(path: Path, label: str) -> pd.Series | None:
    """Load an optional M3 result series, logging absent artifacts."""
    if not path.exists():
        print(f"[skip] {label}: {path.name} not found")
        return None
    return B.load_model_m3(path)


def build_models() -> tuple[dict[str, pd.Series], pd.Series]:
    """Headline models for the post-COVID figure and table."""
    y = B.load_gdp_target(P.GDP_TARGET_CSV)
    origins = pd.period_range("2011Q1", "2025Q4", freq="Q")

    ar1 = B.load_model_m3(P.AR1_RESULTS_CSV)
    dfm = B.load_model_m3(P.actpn_results_csv("en_only"))

    models: dict[str, pd.Series] = {
        "AR(1) expanding": ar1,
        "Rolling-AR(1) 40q": B.rolling_ar1(y, origins, window=40),
        "AR(1) + IC": B.intercept_correct(ar1, y, window=4),
        "DFM-EN": dfm,
    }

    for label, path in [
        ("DFM-ifoCAST", P.IFO_RESULTS_CSV),
        ("DFM-BlockBalanced", P.BLOCKBALANCED_RESULTS_CSV),
        ("DFM-TVP", P.TVP_RESULTS_CSV),
        ("XGB-Full", P.xgb_results_csv("full")),
        ("MLP-Factor", P.MLP_FACTOR_RESULTS_CSV),
    ]:
        s = _load_m3_optional(path, label)
        if s is not None:
            models[label] = s

    combo = _load_m3_optional(P.COMBO_EQUAL_PATH_CSV, "combo_equal")
    if combo is not None:
        models["combo_equal"] = combo

    return models, y


def results_table(models: dict[str, pd.Series], y: pd.Series) -> pd.DataFrame:
    """Compute regime metrics for each available headline model."""
    rows = []
    for n, s in models.items():
        m = B.regime_metrics(s, y)
        rows.append({"model": n, **m})
    return pd.DataFrame(rows).set_index("model")


FIG_MODELS = [
    "AR(1) expanding",
    "Rolling-AR(1) 40q",
    "AR(1) + IC",
    "DFM-ifoCAST",
    "DFM-EN",
    "DFM-BlockBalanced",
    "DFM-TVP",
    "combo_equal",
    "XGB-Full",
    "MLP-Factor",
]

AR_MODEL_COLORS: dict[str, str] = {
    "AR(1) expanding": "#C8D6E5",
    "Rolling-AR(1) 40q": "#93AFC8",
    "AR(1) + IC": "#6E92B4",
}

THESIS_MODEL_COLORS: dict[str, str] = {
    **AR_MODEL_COLORS,
    "DFM-ifoCAST": "#C9617F",
    "DFM-EN": "#F0B4C4",
    "DFM-BlockBalanced": "#E07D96",
    "DFM-TVP": "#B07AA1",
    "combo_equal": "#D4A574",
    "XGB-Full": "#9DCFBF",
    "MLP-Factor": "#C4B5E0",
}

_BAR_EDGE = "#D8DEE6"


def _bar_color(model: str) -> str:
    """Return the stable comparison color for a model."""
    return THESIS_MODEL_COLORS.get(model, "#C5CED8")


def make_figure(tbl: pd.DataFrame, save: Path) -> None:
    """Plot horizontal RMSFE comparisons for the three regimes."""
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9,
    })
    regimes = list(B.REGIMES)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    models = [m for m in FIG_MODELS if m in tbl.index]
    y_pos = np.arange(len(models))

    for ax, reg in zip(axes, regimes):
        a, b = B.REGIMES[reg]
        vals = [tbl.loc[m, f"{reg}_rmsfe"] for m in models]
        colors = [_bar_color(m) for m in models]
        ax.barh(
            y_pos, vals, color=colors, edgecolor=_BAR_EDGE,
            linewidth=0.6, height=0.7, zorder=3,
        )
        for yi, v in zip(y_pos, vals):
            if not np.isnan(v):
                ax.text(v, yi, f" {v:.2f}", va="center", fontsize=7, color="#1F2937")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_title(f"{reg}\n{a}-{b}", fontsize=9.5, fontweight="600")
        ax.set_xlabel("RMSFE (pp)")
        ax.grid(axis="x", ls=":", color="#D1D5DB", zorder=0)

    fig.suptitle(
        "Post-COVID benchmarks — RMSFE by regime (M3)",
        fontsize=11.5, fontweight="700",
    )
    from matplotlib.patches import Patch
    leg = [Patch(fc=_bar_color(m), label=m) for m in models]
    fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(save, dpi=220, bbox_inches="tight")
    print(f"Saved: {save}")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    P.OUT_NOWCASTING.mkdir(parents=True, exist_ok=True)
    models, y = build_models()
    tbl = results_table(models, y)
    tbl = tbl.reindex([m for m in FIG_MODELS if m in tbl.index])
    out_csv = P.OUT_NOWCASTING / "post_covid_benchmarks_table.csv"
    tbl.to_csv(out_csv)
    print(f"Saved: {out_csv}\n")
    show_cols = ["pre-COVID_rmsfe", "COVID_rmsfe", "post-COVID_rmsfe",
                 "post-COVID_bias", "all_rmsfe"]
    print(tbl.loc[:, show_cols].sort_values(by="post-COVID_rmsfe").to_string())
    make_figure(tbl, FIG / "thesis_03_post_covid_improvement.png")
