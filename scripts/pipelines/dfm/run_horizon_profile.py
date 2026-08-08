"""Within-quarter information-accrual profile (M1 -> M2 -> M3) by regime.

Outputs
-------
  outputs/nowcasting/horizon_profile_table.csv
  outputs/nowcasting/figures/thesis_04_horizon_profile.png

Run (from the repository root):
    python scripts/pipelines/dfm/run_horizon_profile.py
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

FIG = P.NOWCAST_FIGURES_DIR

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

MODEL_COLORS: dict[str, str] = {
    "DFM-EN": "#E07D96",
    "DFM-SV-k2": "#3D5FAE",
    "DFM-ifoCAST": "#C9617F",
    "DFM-BlockBalanced": "#8F3D58",
    "DFM-TVP": "#B07AA1",
}


def _rmsfe(df: pd.DataFrame, q0: str, q1: str, m: int) -> float:
    """RMSFE for a (regime, month-in-quarter) cell, dropping NaN error pairs."""
    idx = pd.PeriodIndex(df["quarter"].astype(str), freq="Q")
    mask = np.asarray(
        (idx >= pd.Period(q0)) & (idx <= pd.Period(q1)) & (df["month_in_quarter"] == m)
    )
    e = df["error"].to_numpy()[mask]
    e = e[~np.isnan(e)]
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan


def build_table() -> pd.DataFrame:
    """Long table: model x regime x month-in-quarter -> RMSFE."""
    rows = []
    for name, path in MODELS.items():
        if not Path(path).exists():
            print(f"[skip] {name}: {path.name} not found")
            continue
        df = pd.read_csv(path)
        if "month_in_quarter" not in df.columns:
            print(f"[skip] {name}: no month_in_quarter column")
            continue
        for reg, (q0, q1) in REGIMES.items():
            for m in (1, 2, 3):
                rows.append({
                    "model": name, "regime": reg, "month_in_quarter": m,
                    "RMSFE": round(_rmsfe(df, q0, q1, m), 4),
                })
    return pd.DataFrame(rows)


def make_figure(tbl: pd.DataFrame, save: Path) -> None:
    """Plot within-quarter RMSFE profiles across regimes."""
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    })
    regimes = list(REGIMES)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    models = [m for m in MODELS if m in set(tbl["model"])]

    for ax, reg in zip(axes, regimes):
        sub = tbl[tbl["regime"] == reg]
        for m in models:
            s = sub[sub["model"] == m].sort_values("month_in_quarter")
            ax.plot([1, 2, 3], s["RMSFE"].to_numpy(), marker="o", lw=1.8,
                    color=MODEL_COLORS.get(m, "#888"), label=m, zorder=3)
        ax.set_title(f"{reg}\n{REGIMES[reg][0]}–{REGIMES[reg][1]}",
                     fontsize=9.5, fontweight="600")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["M1", "M2", "M3"])
        ax.set_xlabel("information set (month in quarter)")
        if ax is axes[0]:
            ax.set_ylabel("RMSFE (pp)")
        ax.grid(axis="y", ls=":", color="#D1D5DB", zorder=0)
        s0 = sub[sub["model"] == models[0]].sort_values("month_in_quarter")["RMSFE"].to_numpy()
        if len(s0) == 3 and np.isfinite(s0).all():
            better = s0[2] < s0[0]
            ax.text(
                0.5, 0.94,
                r"more data $\Rightarrow$ better" if better else r"more data $\Rightarrow$ worse",
                transform=ax.transAxes, ha="center", fontsize=8,
                color="#15803D" if better else "#B91C1C", fontweight="600",
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models),
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        r"Within-quarter information accrual ($M1 \to M3$) by regime",
        fontsize=11.5, fontweight="700",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(save, dpi=220, bbox_inches="tight")
    print(f"Saved: {save}")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    P.OUT_NOWCASTING.mkdir(parents=True, exist_ok=True)
    tbl = build_table()
    out_csv = P.OUT_NOWCASTING / "horizon_profile_table.csv"
    tbl.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}\n")
    wide = tbl.pivot_table(index=["regime", "model"], columns="month_in_quarter",
                           values="RMSFE")
    wide.columns = [f"M{c}" for c in wide.columns]
    print(wide.to_string())
    make_figure(tbl, FIG / "thesis_04_horizon_profile.png")
