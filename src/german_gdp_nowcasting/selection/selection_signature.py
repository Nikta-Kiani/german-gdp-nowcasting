"""Clean thesis figure: the Elastic Net *selection signature*.

Contrasts the data-driven hard-data core (what the Elastic Net selects most
often) against the forward-looking survey indicators in ifo's expert set that
the Elastic Net systematically never selects. This is the single figure that
makes the central Part-I result legible: regularised real-time selection
concentrates on contemporaneous hard data and discards the leading surveys on
which the expert benchmark relies.

Run with ``python -m german_gdp_nowcasting.selection.selection_signature``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import paths as P

FIG_DIR = P.OUT_INDICATOR_SELECTION / "figures"

# Hard (real-activity / quantity) vs soft (survey, expectational) blocks.
SOFT_CATEGORIES = {"Surveys"}

# Palette: muted academic tones, hard = slate blue, soft/survey = rose.
COL_HARD = "#3B6CA8"
COL_SOFT = "#C2506E"
COL_GRID = "#E2E8F0"
COL_TEXT = "#1F2933"


def _short_name(raw: str, max_len: int = 34) -> str:
    """Compress a verbose Macrobond series name to a legible stub."""
    s = str(raw)
    if s.startswith("Germany, "):
        s = s[len("Germany, "):]
    s = s.split(", Calendar")[0].split(", Constant")[0].split(", SA")[0]
    s = s.replace("Production Sales, Turnover", "Turnover")
    s = s.replace("Business Surveys, Ifo, Business Survey", "ifo")
    s = s.replace("Economic Surveys, ZEW", "ZEW")
    s = s.replace("Domestic Trade, ", "").replace("Foreign Trade, ", "")
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "\u2026"
    return s


def load_metadata() -> pd.DataFrame:
    """Load names, categories, and publication lags by indicator id."""
    dd = pd.read_csv(P.DATA_DICT_ENRICHED_CSV)
    dd = dd.set_index("id")[["name", "category", "pub_lag"]]
    return dd


def en_selection_frequency() -> pd.Series:
    """Per-indicator share of origins selected by the Elastic Net (EN-only)."""
    m = pd.read_csv(P.EN_ONLY_MATRIX_CSV, index_col=0).astype(float)
    return m.mean(axis=0)


def load_ifocast_overlap() -> pd.DataFrame:
    """Load ifoCAST indicator overlap frequencies."""
    df = pd.read_csv(P.OUT_NOWCASTING / "ifocast_selection_frequency.csv", index_col=0)
    return df


def _is_soft(category: str) -> bool:
    """Return whether a category represents forward-looking soft data."""
    return str(category) in SOFT_CATEGORIES


def build_figure(save: Path) -> plt.Figure:
    """Build and save the Elastic Net selection-signature figure."""
    meta = load_metadata()
    en_freq = en_selection_frequency()
    overlap = load_ifocast_overlap()

    # --- Panel A: data-driven hard-data core (top EN frequencies) ---
    freq = en_freq.reindex(meta.index).dropna()
    cats = meta["category"].reindex(freq.index)
    core = (
        pd.DataFrame({"freq": freq, "category": cats})
        .sort_values("freq", ascending=False)
        .head(15)
        .iloc[::-1]
    )
    core_names = [_short_name(meta.loc[i, "name"]) for i in core.index]
    core_colors = [COL_SOFT if _is_soft(c) else COL_HARD for c in core["category"]]

    # --- Panel B: ifo expert set, EN selection frequency, hard vs survey ---
    exp = overlap.copy()
    exp["category"] = meta["category"].reindex(exp.index)
    exp["soft"] = exp["category"].map(_is_soft)
    exp = exp.sort_values(["soft", "sel_freq_EN"], ascending=[True, True])
    exp_names = [_short_name(n) for n in exp["name"]]
    exp_colors = [COL_SOFT if s else COL_HARD for s in exp["soft"]]

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(13.2, 6.4), gridspec_kw={"width_ratios": [1.0, 1.05]}
    )

    # Panel A
    yA = np.arange(len(core))
    axA.barh(yA, core["freq"].values, color=core_colors, edgecolor="white",
             linewidth=0.6, zorder=3)
    axA.set_yticks(yA)
    axA.set_yticklabels(core_names, fontsize=8)
    axA.set_xlim(0, 1.02)
    axA.set_xlabel("Selection frequency (share of 180 origins)", fontsize=9)
    axA.set_title("(a) The data-driven hard-data core",
                  fontsize=11, fontweight="600", loc="left")
    for y, v in zip(yA, core["freq"].values):
        axA.text(v + 0.012, y, f"{v:.0%}", va="center", fontsize=7,
                 color=COL_TEXT)
    axA.grid(axis="x", color=COL_GRID, linewidth=0.8, zorder=0)
    axA.set_axisbelow(True)

    # Panel B
    yB = np.arange(len(exp))
    axB.barh(yB, exp["sel_freq_EN"].values, color=exp_colors, edgecolor="white",
             linewidth=0.6, zorder=3)
    axB.set_yticks(yB)
    axB.set_yticklabels(exp_names, fontsize=8)
    axB.set_xlim(0, 1.02)
    axB.set_xlabel("EN selection frequency", fontsize=9)
    axB.set_title("(b) ifo expert set: what the Elastic Net keeps and discards",
                  fontsize=11, fontweight="600", loc="left")
    for y, v in zip(yB, exp["sel_freq_EN"].values):
        axB.text(v + 0.012, y, f"{v:.0%}", va="center", fontsize=7,
                 color=COL_TEXT)
    axB.grid(axis="x", color=COL_GRID, linewidth=0.8, zorder=0)
    axB.set_axisbelow(True)

    # Shade the survey block in panel B to underline the "never selected" zone.
    # Surveys are sorted last, so they render at the TOP of the horizontal axis.
    n_soft = int(exp["soft"].sum())
    n_tot = len(exp)
    if n_soft:
        lo, hi = n_tot - n_soft - 0.5, n_tot - 0.5
        axB.axhspan(lo, hi, color=COL_SOFT, alpha=0.07, zorder=0)
        axB.text(
            0.97, (lo + hi) / 2,
            "Surveys (lag-0,\nforward-looking)\nnever selected",
            ha="right", va="center", fontsize=7.5, color=COL_SOFT,
            style="italic",
        )

    for ax in (axA, axB):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COL_HARD),
        plt.Rectangle((0, 0), 1, 1, color=COL_SOFT),
    ]
    fig.legend(
        handles, ["Hard data (real activity, lag 1\u20132m)",
                  "Soft data (surveys, lag 0)"],
        loc="lower center", ncol=2, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Elastic Net selection signature: a hard-data core, surveys excluded",
        fontsize=12.5, fontweight="700", x=0.02, ha="left",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save, dpi=220, bbox_inches="tight")
    print(f"Saved: {save}")
    return fig


def main() -> None:
    """Generate the Elastic Net selection-signature figure."""
    build_figure(FIG_DIR / "selcmp_task4_en_signature.png")


if __name__ == "__main__":
    main()
