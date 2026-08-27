"""Figure: DFM indicator→factor loadings (economic interpretation).

At each M3 forecast origin the Elastic Net selects a *different* indicator
subset; the DFM is re-fitted on that subset only and extracts r=2 latent
factors from the co-movement within the active panel.  This script:

  1. Fits (or loads cached) DFM-EN models at every M3 origin (2011Q1–2025Q4).
  2. Aligns factors across origins (F1 = real-activity composite, F2 = surveys).
  3. Produces two publication figures:
       08_factor_loading_categories.png — category shares over time (stacked areas)
       08_factor_loading_snapshot.png   — top indicators at a representative origin
       08_factor_interpretation_integrated.png — all three views in one figure

Run (from the repository root):
    python scripts/pipelines/dfm/run_factor_loading_figure.py
    python scripts/pipelines/dfm/run_factor_loading_figure.py --force
"""

from __future__ import annotations

import argparse
import gc
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Defer matplotlib until figures are built (keeps the DFM loop lighter).
plt = None
gridspec = None

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
from german_gdp_nowcasting.visualization import nowcast_plots as npl  # noqa: E402
from german_gdp_nowcasting.selection.core_utils import (  # noqa: E402
    load_monthly_panel,
    load_pub_lag_map,
)
from german_gdp_nowcasting.models.dfm.nowcast_utils import (  # noqa: E402
    build_dfm_endog,
    fit_dfm,
)

EVAL_START, EVAL_END = "2011Q1", "2025Q4"
SNAPSHOT_ORIGIN = "2024-12"
K_FACTORS = 2

HARD_CATEGORIES = frozenset({"Production", "Turnover", "Orders", "Trade"})
MAIN_CATEGORIES = [
    "Production", "Turnover", "Orders", "Surveys", "Trade", "Global", "Other",
]

FACTOR_COLORS = ["#8E44AD", "#4A6FA5"]
FACTOR_SHORT = ["Factor 1: real activity", "Factor 2: mixed"]
FACTOR_LABELS = [
    "Factor 1 — real activity",
    "Factor 2 — mixed (surveys largest single category)",
]

CACHE_CSV = P.OUT_NOWCASTING / "factor_loading_m3_panel.csv"
THESIS_CAT = P.THESIS_FIGURES / "08_factor_loading_categories.png"
THESIS_SNAP = P.THESIS_FIGURES / "08_factor_loading_snapshot.png"
THESIS_INT = P.THESIS_FIGURES / "08_factor_interpretation_integrated.png"
DASH_CAT = P.NOWCAST_FIGURES_DIR / "08_factor_loading_categories.png"
DASH_SNAP = P.NOWCAST_FIGURES_DIR / "08_factor_loading_snapshot.png"
DASH_INT = P.NOWCAST_FIGURES_DIR / "08_factor_interpretation_integrated.png"

INTEGRATED_CAPTION = P.OUT_NOWCASTING / "figures" / "08_factor_interpretation_integrated_caption.txt"

COVID0, COVID1 = pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")
BREAK = pd.Timestamp("2022-01-01")


def _short_name(raw: str, max_len: int = 38) -> str:
    """Return a compact indicator label for plotting."""
    s = str(raw)
    if s.startswith("Germany, "):
        s = s[len("Germany, "):]
    s = s.split(", Calendar")[0].split(", Constant")[0]
    s = s.replace("Production Sales, Turnover", "Turnover")
    s = s.replace("Business Surveys, Ifo, Business Survey", "ifo")
    s = s.replace("Business Surveys, DG ECFIN", "DG ECFIN")
    s = s.replace("Industrial Production", "IP")
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "\u2026"
    return s


def _display_name_clear(raw: str) -> str:
    """Plain-language indicator label for snapshot bars (no truncation)."""
    s = str(raw)
    if s.startswith("Germany, "):
        s = s[len("Germany, "):]
    for cut in (", Calendar Adjusted", ", Constant Prices", ", SA", " (X13", ", By Industry",
                ", Balance", ", SA (X-13 ARIMA)"):
        if cut in s:
            s = s.split(cut)[0]
    low = s.lower()

    if "expectations with regard to export" in low:
        if "machinery" in low:
            return "ifo export expectations (machinery)"
        if "motor vehicles" in low:
            return "ifo export expectations (motor vehicles)"
        if "manufacturing industry, total" in low:
            return "ifo export expectations (manufacturing total)"
        return "ifo export expectations"
    if "expectations with regard to employees" in low:
        if "motor vehicles" in low:
            return "ifo employment expectations (motor vehicles)"
        return "ifo employment expectations"
    if "employment expectations" in low:
        if "investment goods" in low:
            return "DG ECFIN employment expectations (investment goods)"
        if "motor vehicles" in low:
            return "DG ECFIN employment expectations (motor vehicles)"
        return "DG ECFIN employment expectations"
    if "dg ecfin" in low and "main industrial" in low:
        return "DG ECFIN industrial confidence (aggregate)"
    if "dg ecfin" in low and "industrial confidence" in low:
        return "DG ECFIN industrial confidence (subsector)"
    if "ifo" in low and "expectations" in low and "manufacturing industry, total" in low:
        return "ifo manufacturing expectations (total)"
    if "ifo" in low and "services" in low:
        return "ifo services confidence"
    if "ifo" in low and "consumer goods" in low:
        return "ifo consumer-goods orders"
    if "ifo" in low and "manufacturing by sectors" in low:
        if "manufacture of " in low:
            sector = s.split("Manufacture of ")[-1].split(",")[0].strip()
            if len(sector) > 24:
                sector = sector[:24].rstrip()
            return f"ifo sector: {sector}"
        return "ifo manufacturing (by sector)"
    if "ifo" in low and "manufacturing industry" in low:
        return "ifo manufacturing climate"

    if s.startswith("Industrial Production, Total, Excluding Construction"):
        return "Industrial production (excl. construction)"
    if s.startswith("Industrial Production, Total"):
        return "Industrial production (total)"
    if "New Orders, Manufacturing, Total (Excluding" in s:
        return "Manufacturing new orders (total)"
    if s.startswith("New Orders, Manufacturing, Total"):
        return "Manufacturing new orders (total)"
    if "Turnover, Manufacturing, Domestic Markets" in s:
        return "Manufacturing turnover (domestic market)"
    if "Turnover, Manufacturing & Mining" in s and "Domestic" in s:
        return "Manufacturing turnover (domestic, excl. energy)"
    if "Turnover, Manufacture of Fabricated Metal" in s:
        return "Fabricated-metal turnover"
    if "Turnover, Intermediate Goods, Total" in s:
        return "Intermediate-goods turnover"
    if "Manufacturing, Fabricated Metal Products" in s:
        return "Fabricated-metal production"
    if "Manufacturing, Manufacture of Basic Metals" in s:
        return "Basic-metals production"

    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) >= 3:
        return f"{parts[0]}: {parts[1]}, {parts[2]}"
    if len(parts) >= 2:
        return f"{parts[0]}: {parts[1]}"
    return parts[0] if parts else s


def _plot_snapshot_bars(
    ax: Any,
    snap_df: pd.DataFrame,
    meta: pd.DataFrame,
    factor: int,
    *,
    top_n: int = 6,
) -> None:
    """Full-width horizontal bar chart for one factor's top loadings."""
    sub = snap_df[snap_df["factor"] == factor].sort_values("rank").head(top_n).copy()
    labels = [
        _display_name_clear(meta.loc[sid, "name"]) if sid in meta.index else sid
        for sid in sub["id"]
    ]
    colors = [npl.CATEGORY_COLORS.get(c, "#9CA3AF") for c in sub["category"]]
    y_pos = np.arange(len(sub))
    vals = sub["loading"].values
    ax.barh(y_pos, vals, color=colors, edgecolor="white", linewidth=0.6, height=0.62)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="#374151", lw=0.9, zorder=2)
    ax.set_xlabel("Indicator-to-factor loading")
    f_idx = factor - 1
    ax.set_title(
        f"{FACTOR_SHORT[f_idx]} — top {top_n} indicators by loading strength",
        fontsize=9.5, fontweight="600", color=FACTOR_COLORS[f_idx], loc="left", pad=4,
    )
    ax.invert_yaxis()
    ax.tick_params(axis="y", pad=4)


def _m3_origins(semiannual: bool = True) -> list[tuple[pd.Period, str]]:
    """Quarterly M3 (end-of-quarter) monthly origins.

    Default ``semiannual=True`` keeps Q2 and Q4 only (30 points) so the
    panel build stays fast and memory-safe while still showing drift.
    """
    out: list[tuple[pd.Period, str]] = []
    for q in pd.period_range(EVAL_START, EVAL_END, freq="Q"):
        if semiannual and q.quarter not in (2, 4):
            continue
        m = q.asfreq("M", how="end")
        out.append((q, f"{m.year}-{m.month:02d}"))
    return out


def _extract_loading_matrix(result: Any, monthly_cols: list[str]) -> np.ndarray:
    """Extract the fitted indicator-to-factor loading matrix."""
    p = result.params
    L = np.zeros((K_FACTORS, len(monthly_cols)))
    for j, col in enumerate(monthly_cols):
        for f in range(K_FACTORS):
            key = f"loading.{f}->{col}"
            if key in p.index:
                L[f, j] = p[key]
    return L


def _align_factors(
    L: np.ndarray,
    monthly_cols: list[str],
    meta: pd.DataFrame,
    endog: pd.DataFrame,
    result: Any,
) -> np.ndarray:
    """Label F1 as the hard-activity factor and stabilise signs."""
    cats = meta.reindex(monthly_cols)["category"].fillna("Unknown")

    def _hard_score(row: int) -> float:
        """Score a factor by mean absolute loading on hard-data series."""
        mask = cats.isin(HARD_CATEGORIES).values
        if not mask.any():
            return 0.0
        return float(np.mean(np.abs(L[row, mask])))

    if _hard_score(1) > _hard_score(0):
        L = L[[1, 0], :]

    # Sign: F1 positively correlated with total IP when available.
    ip_id = next((c for c in monthly_cols if c == "deprod1404"), None)
    if ip_id is not None:
        fm = result.factors.smoothed.copy()
        aligned = endog[[ip_id]].join(fm, how="inner").dropna()
        if len(aligned) > 12:
            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            if corr < 0:
                L[0, :] *= -1

    # Sign: F2 positively correlated with first survey in the panel.
    surv_ids = [
        c for c in monthly_cols
        if c in meta.index and meta.loc[c, "category"] == "Surveys"
    ]
    if surv_ids:
        sid = surv_ids[0]
        fm = result.factors.smoothed.copy()
        aligned = endog[[sid]].join(fm, how="inner").dropna()
        if len(aligned) > 12 and aligned.shape[1] >= 3:
            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 2])
            if corr < 0:
                L[1, :] *= -1

    return L


def _category_shares(
    L_row: np.ndarray,
    monthly_cols: list[str],
    meta: pd.DataFrame,
) -> pd.Series:
    """Aggregate one factor's absolute loadings to category shares."""
    df = pd.DataFrame({"id": monthly_cols, "absL": np.abs(L_row)})
    df["category"] = meta.reindex(df["id"])["category"].fillna("Unknown").values
    df.loc[~df["category"].isin(MAIN_CATEGORIES[:-1]), "category"] = "Other"
    g = df.groupby("category", observed=True)["absL"].sum()
    total = g.sum()
    if total <= 0:
        return pd.Series(0.0, index=MAIN_CATEGORIES)
    shares = (g / total).reindex(MAIN_CATEGORIES, fill_value=0.0)
    return shares


def _top_indicators(
    L_row: np.ndarray,
    monthly_cols: list[str],
    meta: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """Return the strongest indicators for one factor."""
    df = pd.DataFrame({
        "id": monthly_cols,
        "loading": L_row,
        "absL": np.abs(L_row),
    })
    df["name"] = [_short_name(meta.loc[i, "name"]) if i in meta.index else i
                  for i in df["id"]]
    df["category"] = meta.reindex(df["id"])["category"].fillna("Unknown").values
    return df.nlargest(n, "absL")


def build_loading_panel(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (category_shares_long, snapshot_top_long)."""
    if CACHE_CSV.exists() and not force:
        cached = pd.read_csv(CACHE_CSV, parse_dates=["date"])
        cat_df = cached[cached["record_type"] == "category"].copy()
        snap_df = cached[cached["record_type"] == "snapshot"].copy()
        if not cat_df.empty:
            print(f"Loaded factor-loading cache ({len(cat_df)} category rows)")
            return cat_df, snap_df

    X = load_monthly_panel(P.PANEL_TRANSFORMED_CSV)
    pub = load_pub_lag_map(P.PUB_LAG_CSV)
    y = pd.read_csv(P.GDP_TARGET_CSV, index_col="quarter").squeeze("columns")
    y.index = pd.PeriodIndex(y.index, freq="Q")
    sel = pd.read_csv(P.EN_ONLY_MATRIX_CSV, index_col="forecast_origin")
    meta = pd.read_csv(
        P.DATA_DICT_ENRICHED_CSV, usecols=["id", "name", "category"],
    ).set_index("id")

    cat_records: list[dict] = []
    snap_records: list[dict] = []

    snapshot_L: tuple[np.ndarray, list[str], str, str, int] | None = None

    for q, origin in _m3_origins():
        if origin not in sel.index:
            continue
        cols = sel.columns[sel.loc[origin].astype(bool)].tolist()
        if len(cols) < 4:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                endog, km = build_dfm_endog(
                    X[cols], y, origin,
                    pub_lag_map=pub, fill_method="ar_bic",
                )
                res = fit_dfm(
                    endog, km, k_factors=K_FACTORS,
                    factor_order=2, idiosyncratic_ar1=True,
                )
            monthly_cols = list(endog.columns[:-1])
            L = _extract_loading_matrix(res, monthly_cols)
            L = _align_factors(L, monthly_cols, meta, endog, res)

            date = q.to_timestamp(how="end")
            for f in range(K_FACTORS):
                shares = _category_shares(L[f], monthly_cols, meta)
                for cat, share in shares.items():
                    cat_records.append({
                        "record_type": "category",
                        "quarter": str(q),
                        "date": date,
                        "origin": origin,
                        "n_indicators": len(cols),
                        "factor": f + 1,
                        "category": cat,
                        "share": float(share),
                    })

            if origin == SNAPSHOT_ORIGIN:
                snapshot_L = (L.copy(), list(monthly_cols), origin, str(q), len(cols))
            elif snapshot_L is None:
                snapshot_L = (L.copy(), list(monthly_cols), origin, str(q), len(cols))
            print(f"  {origin}  N={len(cols):3d}  OK", flush=True)
        except Exception as exc:
            print(f"  {origin}  SKIP ({exc})", flush=True)
        finally:
            gc.collect()

    if snapshot_L is not None:
        L_snap, mcols, s_origin, s_q, s_n = snapshot_L
        for f in range(K_FACTORS):
            top = _top_indicators(L_snap[f], mcols, meta, n=10)
            for rank, row in enumerate(top.itertuples(), start=1):
                snap_records.append({
                    "record_type": "snapshot",
                    "origin": s_origin,
                    "quarter": s_q,
                    "factor": f + 1,
                    "rank": rank,
                    "id": row.id,
                    "name": row.name,
                    "category": row.category,
                    "loading": float(row.loading),
                    "abs_loading": float(row.absL),
                    "n_indicators": s_n,
                })

    cat_df = pd.DataFrame(cat_records)
    snap_df = pd.DataFrame(snap_records)
    if not cat_df.empty:
        CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([cat_df, snap_df], ignore_index=True).to_csv(CACHE_CSV, index=False)
        print(f"Saved cache -> {CACHE_CSV}")
    return cat_df, snap_df


def _import_mpl() -> None:
    """Import matplotlib lazily for the figure-building phase."""
    global plt, gridspec
    if plt is None:
        import matplotlib.pyplot as _plt
        import matplotlib.gridspec as _gridspec
        plt = _plt
        gridspec = _gridspec


def _setup_axes_style() -> None:
    """Apply shared thesis styling to lazily imported matplotlib."""
    _import_mpl()
    npl.setup_style()
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })


def _load_tvp_m3() -> pd.DataFrame:
    """M3 TVP bridge loadings (factor -> GDP)."""
    df = pd.read_csv(P.TVP_RESULTS_CSV)
    df = df[df["month_in_quarter"] == 3].copy()
    df["date"] = pd.PeriodIndex(df["quarter"].astype(str), freq="Q").to_timestamp(how="end")
    return df.sort_values("date")


def _regime_shading(ax: Any, *, zorder: int = 0) -> None:
    """Mark the COVID window and post-2022 regime break."""
    ax.axvspan(COVID0, COVID1, color="#E2899B", alpha=0.10, lw=0, zorder=zorder)
    ax.axvline(BREAK, color="#6B7280", ls="--", lw=1.0, zorder=zorder + 1)


def _panel_label(ax: Any, letter: str) -> None:
    """Add a publication-style panel letter to an axis."""
    ax.text(
        -0.10, 1.06, f"({letter})",
        transform=ax.transAxes, fontsize=11, fontweight="600", color="#1F2937",
    )


def _plot_category_stack(
    ax: Any,
    cat_df: pd.DataFrame,
    factor: int,
    *,
    show_legend: bool,
) -> None:
    """One factor's category-share stacked area."""
    pivot_cols = [c for c in MAIN_CATEGORIES if c in cat_df["category"].unique()]
    sub = cat_df[cat_df["factor"] == factor]
    wide = sub.pivot_table(
        index="date", columns="category", values="share", aggfunc="first",
    ).reindex(columns=pivot_cols, fill_value=0.0).sort_index()

    colors = [npl.CATEGORY_COLORS.get(c, "#9CA3AF") for c in wide.columns]
    _regime_shading(ax)
    ax.stackplot(
        wide.index, *[wide[c].values for c in wide.columns],
        labels=wide.columns, colors=colors, alpha=0.88, linewidth=0,
    )
    f_idx = factor - 1
    ax.set_title(FACTOR_SHORT[f_idx], fontsize=10, fontweight="600",
                 color=FACTOR_COLORS[f_idx], loc="left", pad=6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of |loading|")
    if show_legend:
        ax.legend(
            loc="center left", bbox_to_anchor=(1.02, 0.5),
            frameon=False, fontsize=8, title="Category",
        )
    ax.tick_params(labelbottom=False)


def fig_integrated(cat_df: pd.DataFrame, snap_df: pd.DataFrame) -> "plt.Figure":
    """Three-panel figure: factor content, TVP GDP transmission, snapshot detail."""
    _import_mpl()
    from matplotlib.patches import Patch
    _setup_axes_style()
    cat_df = cat_df.sort_values("date")
    tvp_df = _load_tvp_m3()
    if snap_df.empty:
        raise ValueError("Snapshot rows required for integrated figure.")

    meta = pd.read_csv(
        P.DATA_DICT_ENRICHED_CSV, usecols=["id", "name", "category"],
    ).set_index("id")

    n_ind = int(snap_df["n_indicators"].iloc[0])
    quarter = snap_df["quarter"].iloc[0]
    x_min = min(cat_df["date"].min(), tvp_df["date"].min())
    x_max = max(cat_df["date"].max(), tvp_df["date"].max())
    top_n = 6

    fig = plt.figure(figsize=(11.5, 12.5))
    gs = gridspec.GridSpec(
        4, 2, figure=fig,
        height_ratios=[1.0, 0.92, 0.72, 0.72],
        hspace=0.55, wspace=0.26,
    )
    fig.suptitle(
        "DFM factor interpretation: what the factors are, and how they map to GDP",
        fontsize=12, fontweight="600", y=0.97,
    )

    # (a) Stage 1 — indicator -> factor content
    ax_f1 = fig.add_subplot(gs[0, 0])
    ax_f2 = fig.add_subplot(gs[0, 1])
    _plot_category_stack(ax_f1, cat_df, factor=1, show_legend=False)
    _plot_category_stack(ax_f2, cat_df, factor=2, show_legend=True)
    for ax in (ax_f1, ax_f2):
        ax.set_xlim(x_min, x_max)
        _panel_label(ax, "a")
    ax_f1.set_title(
        f"{FACTOR_SHORT[0]} — which data categories drive this factor?",
        fontsize=9.5, fontweight="600", color=FACTOR_COLORS[0], loc="left", pad=6,
    )
    ax_f2.set_title(
        f"{FACTOR_SHORT[1]} — which data categories drive this factor?",
        fontsize=9.5, fontweight="600", color=FACTOR_COLORS[1], loc="left", pad=6,
    )

    # (b) Stage 2 — factor -> GDP bridge (TVP)
    ax_tvp = fig.add_subplot(gs[1, :])
    _regime_shading(ax_tvp)
    ax_tvp.axhline(0.0, color="#C5CED8", lw=0.8, zorder=0)
    for j, col in enumerate(["tvp_loading_1", "tvp_loading_2"]):
        ax_tvp.plot(
            tvp_df["date"], tvp_df[col],
            color=FACTOR_COLORS[j], lw=2.1, marker="o", ms=3.0, zorder=3,
            label=rf"$\lambda_{{{j + 1}}}$  {FACTOR_SHORT[j]}",
        )
    ax_tvp.set_xlim(x_min, x_max)
    ax_tvp.set_ylabel("Loading on GDP (pp per unit of factor)")
    ax_tvp.set_xlabel("Quarter (M3 forecast origin)")
    ax_tvp.set_title(
        "How strongly does each factor transmit to GDP?  (DFM-TVP bridge, Stage 2)",
        fontsize=9.5, fontweight="600", loc="left", pad=6,
    )
    covid_patch = Patch(facecolor="#E2899B", alpha=0.15, edgecolor="none",
                        label="COVID-19 quarters (down-weighted)")
    handles, labels = ax_tvp.get_legend_handles_labels()
    ax_tvp.legend(
        handles + [covid_patch], labels + [covid_patch.get_label()],
        frameon=False, fontsize=8.5, loc="upper center",
        bbox_to_anchor=(0.5, -0.20), ncol=3,
    )
    ax_tvp.annotate(
        "2022 stagnation onset",
        xy=(BREAK, 0.0), xycoords=("data", "axes fraction"),
        xytext=(5, 8), textcoords="offset points",
        fontsize=8.5, color="#6B7280", va="bottom",
    )
    _panel_label(ax_tvp, "b")

    # (c) Snapshot — full-width bars, one factor per row (avoids label overlap)
    ax_s1 = fig.add_subplot(gs[2, :])
    ax_s2 = fig.add_subplot(gs[3, :])
    _plot_snapshot_bars(ax_s1, snap_df, meta, factor=1, top_n=top_n)
    _plot_snapshot_bars(ax_s2, snap_df, meta, factor=2, top_n=top_n)
    _panel_label(ax_s1, "c")

    fig.text(
        0.5, 0.015,
        f"Panel (c): strongest loadings at {quarter}, M3 information set "
        f"({n_ind} Elastic-Net indicators).  "
        "Bar colour = economic category (same as panel a).  "
        "Positive loading = indicator moves with the factor.",
        ha="center", fontsize=8.5, color="#4B5563",
    )
    fig.subplots_adjust(left=0.22, right=0.94, top=0.94, bottom=0.07, hspace=0.55)

    caption = f"""DFM factor interpretation: economic content and GDP transmission (2011Q1–2025Q4, M3 origins, DFM-EN inputs).

Panel (a) summarises Stage 1 of the two-step DFM. At each forecast origin the model is re-fitted on the Elastic-Net-selected indicator set; the stacked areas show the share of each economic category in the absolute indicator-to-factor loadings. Factor 1 is a stable real-activity composite: production, turnover, orders and trade carry about 90% of its loading mass before and after COVID. Factor 2 is mixed — surveys are its largest single category (about 31%), but production, turnover and orders still account for about two-thirds of its mass. Category shares can spike when selection keeps few series at a given origin.

Panel (b) plots the Stage-2 DFM-TVP bridge coefficients: how many percentage points of GDP growth each factor implies per unit increase in the latent factor. The loadings are allowed to drift over time (random walk) and COVID quarters are down-weighted. Before 2020, Factor 1 carries most of the GDP transmission; the post-2022 period shows a weaker but positive Factor-1 link consistent with the stagnation regime.

Panel (c) lists the six indicators with the largest absolute loadings on each factor at {quarter} (N={n_ind} active series), displayed as two full-width bar charts so that series names remain readable. Each bar is the EM-estimated Stage-1 loading; colour matches the category legend in panel (a). A positive loading means the indicator co-moves with the factor; negative loadings indicate an inverse relationship within the estimated factor rotation.

Notes: Factors are identified only up to rotation and sign. Panels (a) and (b) use a rotation realigned at every origin, fixing Factor 1 to co-move positively with industrial production and Factor 2 with the first survey series. That alignment is a labelling convention, not a claim that Factor 2 is a survey factor. Factor colours are held constant across all three panels.
"""
    INTEGRATED_CAPTION.parent.mkdir(parents=True, exist_ok=True)
    INTEGRATED_CAPTION.write_text(caption.strip() + "\n", encoding="utf-8")

    return fig


def fig_category_profile(cat_df: pd.DataFrame) -> "plt.Figure":
    """Stacked area: economic category shares of |loading| per factor over time."""
    _import_mpl()
    _setup_axes_style()
    cat_df = cat_df.sort_values("date")
    pivot_cols = [c for c in MAIN_CATEGORIES if c in cat_df["category"].unique()]

    fig = plt.figure(figsize=(11.5, 5.2))
    gs = gridspec.GridSpec(1, 2, wspace=0.22)
    fig.suptitle(
        "DFM-EN: what do the two factors load on?  (category shares, M3 origins)",
        fontsize=12, fontweight="600", y=1.02,
    )

    for f, ax in enumerate([fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]):
        sub = cat_df[cat_df["factor"] == f + 1]
        wide = sub.pivot_table(
            index="date", columns="category", values="share", aggfunc="first",
        ).reindex(columns=pivot_cols, fill_value=0.0).sort_index()

        colors = [npl.CATEGORY_COLORS.get(c, "#9CA3AF") for c in wide.columns]
        ax.stackplot(
            wide.index, *[wide[c].values for c in wide.columns],
            labels=wide.columns, colors=colors, alpha=0.88, linewidth=0,
        )
        ax.axvspan(COVID0, COVID1, color="#E2899B", alpha=0.10, lw=0)
        ax.axvline(BREAK, color="#6B7280", ls="--", lw=1.0)
        ax.set_title(FACTOR_LABELS[f], fontsize=10.5, fontweight="600",
                     color=FACTOR_COLORS[f], loc="left")
        ax.set_xlim(wide.index.min(), wide.index.max())
        ax.set_ylim(0, 1)
        ax.set_ylabel("Share of |loading|" if f == 0 else "")
        if f == 1:
            ax.legend(
                loc="center left", bbox_to_anchor=(1.02, 0.5),
                frameon=False, fontsize=8.5, title="Category",
            )

    fig.text(
        0.5, -0.02,
        "At each origin the DFM is re-fitted on the Elastic-Net-selected indicators only; "
        "shares sum to 1 within each factor.  Shaded: COVID window.",
        ha="center", fontsize=8.5, color="#6B7280",
    )
    fig.subplots_adjust(bottom=0.14, top=0.90, wspace=0.22)
    return fig


def fig_snapshot(snap_df: pd.DataFrame) -> "plt.Figure":
    """Horizontal bars: top indicators at the representative snapshot origin."""
    _import_mpl()
    _setup_axes_style()
    if snap_df.empty:
        raise ValueError(f"No snapshot rows for origin {SNAPSHOT_ORIGIN}")

    n_ind = int(snap_df["n_indicators"].iloc[0])
    quarter = snap_df["quarter"].iloc[0]
    meta = pd.read_csv(
        P.DATA_DICT_ENRICHED_CSV, usecols=["id", "name", "category"],
    ).set_index("id")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=False)
    fig.suptitle(
        f"Top indicator loadings at {quarter}  (M3 origin, N={n_ind} selected series)",
        fontsize=12, fontweight="600", y=1.02,
    )

    for f, ax in enumerate(axes):
        sub = snap_df[snap_df["factor"] == f + 1].sort_values("rank")
        labels = [
            _display_name_clear(meta.loc[sid, "name"]) if sid in meta.index else sid
            for sid in sub["id"]
        ]
        colors = [npl.CATEGORY_COLORS.get(c, "#9CA3AF") for c in sub["category"]]
        y_pos = np.arange(len(sub))
        vals = sub["loading"].values
        ax.barh(y_pos, vals, color=colors, edgecolor="white", linewidth=0.6, height=0.72)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.axvline(0, color="#C5CED8", lw=0.8)
        ax.set_xlabel("Loading (standardised indicators)")
        ax.set_title(FACTOR_LABELS[f], fontsize=10.5, fontweight="600",
                     color=FACTOR_COLORS[f], loc="left")
        ax.invert_yaxis()

    fig.text(
        0.5, -0.02,
        "Bars show the EM-estimated indicator-to-factor loadings from Stage 1. "
        "Colour = economic category (same palette as elsewhere in the thesis).",
        ha="center", fontsize=8.5, color="#6B7280",
    )
    fig.subplots_adjust(bottom=0.12, top=0.90, wspace=0.28)
    return fig


def _save(fig: "plt.Figure", thesis_path: Path, dash_path: Path) -> None:
    """Save a figure to both thesis and dashboard destinations."""
    for path in (thesis_path, dash_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(f"Saved: {path}")


def main() -> None:
    """Build cached loading data and all factor-interpretation figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-fit all origins")
    args = parser.parse_args()

    print("Building factor-loading panel (DFM-EN, r=2, M3 origins) ...")
    cat_df, snap_df = build_loading_panel(force=args.force)

    if cat_df.empty:
        raise RuntimeError("No factor-loading rows produced.")

    if snap_df.empty:
        raise RuntimeError("Snapshot panel is empty — re-run with --force.")

    fig1 = fig_category_profile(cat_df)
    _save(fig1, THESIS_CAT, DASH_CAT)

    fig2 = fig_snapshot(snap_df)
    _save(fig2, THESIS_SNAP, DASH_SNAP)

    fig3 = fig_integrated(cat_df, snap_df)
    _save(fig3, THESIS_INT, DASH_INT)
    _import_mpl()
    plt.close("all")


if __name__ == "__main__":
    main()
