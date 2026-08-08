"""Figure: DFM-TVP time-varying factor->GDP loadings (loading-drift plot).

Reads the DFM-TVP nowcast CSV (M3 origins) and plots the filtered random-walk
bridge coefficients over time, with the down-weighted COVID window shaded and
the 2022 stagnation onset marked. Saves the figure into the thesis figures
directory (07_tvp_loading_drift.png) and the dashboard outputs figures dir.

Run (from the repository root):
    python scripts/pipelines/dfm/run_tvp_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
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

THESIS_FIG = P.ROOT / "Overleaf-Thesis" / "figures" / "07_tvp_loading_drift.png"
DASH_FIG = P.NOWCAST_FIGURES_DIR / "07_tvp_loading_drift.png"

LOADING_COLORS = ["#B07AA1", "#6E92B4"]
COVID0, COVID1 = pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")
BREAK = pd.Timestamp("2022-01-01")


def main() -> None:
    """Plot and save the M3 time-varying GDP bridge loadings."""
    df = pd.read_csv(P.TVP_RESULTS_CSV)
    df = df[df["month_in_quarter"] == 3].copy()
    df["date"] = pd.PeriodIndex(df["quarter"].astype(str), freq="Q").to_timestamp(how="end")
    df = df.sort_values("date")

    load_cols = [c for c in df.columns if c.startswith("tvp_loading_")]

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10,
    })
    fig, ax = plt.subplots(figsize=(11, 4.6))

    ax.axvspan(COVID0, COVID1, color="#E2899B", alpha=0.12, lw=0,
               label="COVID-19 (down-weighted)")
    ax.axvline(BREAK, color="#6B7280", ls="--", lw=1.1)
    ax.axhline(0.0, color="#C5CED8", lw=0.8, zorder=0)

    for j, col in enumerate(load_cols):
        ax.plot(df["date"], df[col], color=LOADING_COLORS[j % len(LOADING_COLORS)],
                lw=2.0, marker="o", ms=3.2,
                label=f"Loading on factor {j + 1}  ($\\lambda_{{{j + 1}}}$)")

    ax.annotate("2022 stagnation onset", xy=(BREAK, ax.get_ylim()[1]),
                xytext=(6, -12), textcoords="offset points",
                fontsize=8.5, color="#6B7280")
    ax.set_xlabel("Quarter (M3 origin)")
    ax.set_ylabel("Filtered bridge loading")
    ax.set_title("DFM-TVP: time-varying factor\u2192GDP loadings",
                 fontsize=12, fontweight="600")
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()

    for path in (THESIS_FIG, DASH_FIG):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
