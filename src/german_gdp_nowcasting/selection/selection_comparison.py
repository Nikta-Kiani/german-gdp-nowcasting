r"""Cross-model indicator-selection comparison for German GDP nowcasting.

This module turns the heterogeneous selection / attribution artefacts produced
by the thesis pipeline into a *single, comparable* view of **which slices of the
German macro data each model leans on, and how that focus moves through time
and across the COVID structural break**.

It harmonises several very different notions of "a variable matters":

==================  ============================================  ===================================
Model family        Native signal                                 Source artefact
==================  ============================================  ===================================
Elastic Net (EN)    binary inclusion 0/1 per origin               ``selection_matrix.csv``
Block-balanced k=20 binary inclusion 0/1 per origin               ``selection_matrix_blockbalanced_k20.csv``
PLS (comparison)    binary inclusion 0/1 per origin               ``selection_matrix_pls.csv``
DFM EN input        binary inclusion 0/1 per origin               ``dfm_input_sets/en_only_selection_matrix.csv``
XGBoost             mean |SHAP| per (feature, lag) per quarter    ``xgb_shap_importance.csv``
==================  ============================================  ===================================

The bridge concept is the **category mass share** \(s_{c,t}\): the fraction of a
model's total "selection mass" at time *t* that is allocated to economic
category *c* (Surveys, Production, Orders, ...). For binary selectors this is the
share of the *selected set*; for XGBoost it is the share of total |SHAP|. Both are
dimensionless, sum to one across categories, and are therefore directly comparable.

Run as a script to (re)generate every figure and the interpretation guide::

    python -m german_gdp_nowcasting.selection.selection_comparison

Author: thesis pipeline (Part I — indicator selection diagnostics).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import paths as P

# Plotting back-ends are imported lazily inside the plotting functions so the
# IO/transform layer stays importable in headless / test contexts.


# =============================================================================
# 0. Configuration
# =============================================================================

#: Output directory for all comparison artefacts (figures + interpretation md).
FIG_DIR: Path = P.OUT_INDICATOR_SELECTION / "figures"

#: Binary selection masks (180 monthly origins x 585 series, values in {0, 1}).
#: Keys are the human-facing model labels used throughout the figures.
BINARY_MATRIX_PATHS: dict[str, Path] = {
    "EN (raw)": P.SELECTION_MATRIX_CSV,
    "PLS": P.PLS_MATRIX_CSV,
    "Block-balanced (k=20)": P.BLOCKBALANCED_MATRIX_CSV,
    "DFM EN-only": P.EN_ONLY_MATRIX_CSV,
}

#: Canonical ordering of the economic categories (most → least populous), with
#: a "hard" (real-activity) vs "soft" (survey/sentiment) tag used for the
#: soft-vs-hard narrative. ``Misc`` is renamed ``Other`` for display.
CATEGORY_ORDER: tuple[str, ...] = (
    "Surveys",
    "Orders",
    "Turnover",
    "Production",
    "Construction",
    "Trade",
    "Prices",
    "Commodities",
    "Financial",
    "Global",
    "Misc",
)

#: Soft (expectational / sentiment) vs hard (realised real-activity & price) split.
SOFT_CATEGORIES: frozenset[str] = frozenset({"Surveys"})
HARD_CATEGORIES: frozenset[str] = frozenset(
    {"Orders", "Turnover", "Production", "Construction", "Trade"}
)

#: Luminous pastel / "sorbet" palette — one hue per category, deliberately
#: avoiding harsh primaries and neon. Ordered to match ``CATEGORY_ORDER``.
SORBET: dict[str, str] = {
    "Surveys": "#F2A6B3",      # rose sorbet (soft data, the headline series)
    "Orders": "#F7C59F",       # peach
    "Turnover": "#F9DFA6",     # mango cream
    "Production": "#A8D8C9",   # mint
    "Construction": "#BFE0D2", # pale jade
    "Trade": "#9FC8E8",        # sky sorbet
    "Prices": "#C9B8E8",       # lilac
    "Commodities": "#E0C2A8",  # caramel
    "Financial": "#A8C5E0",    # periwinkle
    "Global": "#D7C2D9",       # mauve
    "Misc": "#D9D2C5",         # stone
}

#: Display names (only where they differ from the raw category code).
CATEGORY_DISPLAY: dict[str, str] = {"Misc": "Other", "Surveys": "Surveys (soft)"}

#: Regime windows (inclusive), matching the thesis design.
REGIME_ORDER: tuple[str, ...] = ("pre-COVID", "COVID", "post-COVID")
REGIME_COLORS: dict[str, str] = {
    "pre-COVID": "#A8D8C9",   # calm mint  (2011–2019)
    "COVID": "#F2A6B3",       # alert rose (2020–2021)
    "post-COVID": "#C9B8E8",  # lilac      (2022–2025)
}

#: Task 3 regime-bar comparison — method families side by side per regime.
TASK3_METHOD_ORDER: tuple[str, ...] = ("EN (raw)", "XGBoost (SHAP)")
TASK3_METHOD_COLORS: dict[str, str] = {
    "EN (raw)": "#539BC1",     # clear blue
    "XGBoost (SHAP)": "#E89C4B",    # warm amber
}
TASK3_METHOD_LABELS: dict[str, str] = {
    "EN (raw)": "EN",
    "XGBoost (SHAP)": "XGBoost (SHAP)",
}

#: Publication-lag buckets (months after the reference month) for the Task-1
#: companion panel. Disentangles *timeliness/availability* from *informativeness*:
#: lag-0 series (surveys, sentiment) are available in real time at every origin,
#: whereas lag-1/2 hard-activity series (production, orders, turnover) are not yet
#: released at short horizons. A soft-data tilt that merely tracks the lag-0
#: availability baseline reflects the information set, not a genuine preference
#: (Bańbura & Rünstler 2011; Giannone, Reichlin & Small 2008).
PUBLAG_ORDER: tuple[str, ...] = ("0", "1", "2", "n/a")
PUBLAG_LABELS: dict[str, str] = {
    "0": "lag 0 (same month — surveys, timely)",
    "1": "lag 1 month",
    "2": "lag 2 months (hard data)",
    "n/a": "unknown lag",
}
PUBLAG_COLORS: dict[str, str] = {
    "0": "#F2A6B3",   # rose — timely soft data
    "1": "#F9DFA6",   # mango — one-month lag
    "2": "#9FC8E8",   # sky — delayed hard data
    "n/a": "#D9D2C5",  # stone — unknown
}


# =============================================================================
# 1. IO layer
# =============================================================================

def load_metadata(path: Path = P.DATA_DICT_ENRICHED_CSV) -> pd.DataFrame:
    """Load the enriched data dictionary indexed by series ``id``.

    Returns a frame with at least ``name``, ``category`` and ``pub_lag`` for the
    585 modelled series. ``pub_lag`` is coerced to a nullable integer (0/1/2
    months); ``category`` is left as the raw code (mapped for display elsewhere).
    """
    md = pd.read_csv(path)
    if "id" not in md.columns:
        raise ValueError(f"{path} is missing the required 'id' column.")
    md = md.drop_duplicates(subset="id").set_index("id")
    md["pub_lag"] = pd.to_numeric(md.get("pub_lag"), errors="coerce").astype("Int64")
    if "category" not in md.columns:
        raise ValueError(f"{path} is missing the required 'category' column.")
    return md


def load_binary_matrix(path: Path) -> pd.DataFrame:
    """Load one binary selection mask as a (origins x series) 0/1 DataFrame.

    The index (``forecast_origin``) is parsed to a monthly :class:`~pandas.Period`
    so that all models share a single, sortable time axis. Values are validated
    to be in {0, 1}.
    """
    mat = pd.read_csv(path, index_col=0)
    mat.index = pd.PeriodIndex(mat.index, freq="M")
    mat.index.name = "origin"
    bad = set(np.unique(mat.to_numpy())) - {0, 1}
    if bad:
        raise ValueError(f"{path.name} contains non-binary values {bad}.")
    return mat.astype(np.int8)


def load_all_binary_matrices(
    paths: Mapping[str, Path] = BINARY_MATRIX_PATHS,
) -> dict[str, pd.DataFrame]:
    """Load every configured binary selection mask, skipping any absent file."""
    out: dict[str, pd.DataFrame] = {}
    for label, path in paths.items():
        if Path(path).exists():
            out[label] = load_binary_matrix(Path(path))
        else:
            print(f"[warn] missing binary matrix for '{label}': {path}")
    return out


# Autoregressive target features in XGB — not macro indicators; exclude from
# category / publag mass shares so they do not pollute selection diagnostics.
_NON_INDICATOR_IDS: frozenset[str] = frozenset({"gdp"})


def load_shap(path: Path = P.XGB_SHAP_IMPORTANCE_CSV) -> pd.DataFrame:
    """Load the XGBoost SHAP log and resolve each feature to its base series id.

    The raw log is long-format ``(quarter, feature, mean_abs_shap)`` where
    ``feature`` carries a lag suffix (``deprod1404__L0``). Lags are stripped to
    recover the economic series id so SHAP mass can be aggregated to the same
    585-series universe as the selection masks. The quarter string is parsed to
    a quarterly :class:`~pandas.Period`.
    """
    shap = pd.read_csv(path)
    shap["series_id"] = shap["feature"].str.rsplit("__L", n=1).str[0]
    shap["quarter"] = pd.PeriodIndex(shap["quarter"], freq="Q")
    return shap.loc[~shap["series_id"].isin(_NON_INDICATOR_IDS)].copy()


# =============================================================================
# 2. Transform layer — time, categories, shares, rankings
# =============================================================================

def regime_of(period: pd.Period) -> str:
    """Map a monthly/quarterly :class:`~pandas.Period` to a COVID regime label.

    Boundaries follow the thesis design exactly:
    pre-COVID = ≤2019, COVID = 2020–2021, post-COVID = ≥2022.
    """
    year = period.year
    if year <= 2019:
        return "pre-COVID"
    if year <= 2021:
        return "COVID"
    return "post-COVID"


def attach_category(
    ids: Sequence[str], metadata: pd.DataFrame
) -> pd.Series:
    """Return a Series mapping each series id → economic category.

    Ids absent from the metadata are labelled ``"Misc"`` so they never silently
    vanish from category aggregates (data-handling principle: never drop data
    without trace).
    """
    cat = metadata["category"].reindex(ids)
    missing_ids = [i for i in pd.Index(ids).unique() if i not in metadata.index]
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        suffix = "..." if len(missing_ids) > 5 else ""
        print(
            f"[warn] {len(missing_ids)} id(s) missing from metadata "
            f"(labelled 'Misc'): {preview}{suffix}"
        )
    return cat.fillna("Misc")


def _normalise_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Row-normalise a non-negative frame to composition shares (rows sum to 1).

    Rows whose total mass is zero are left as zeros (no selection that period).
    """
    totals = df.sum(axis=1)
    safe = totals.replace(0, np.nan)
    return df.div(safe, axis=0).fillna(0.0)


def category_share_from_binary(
    matrix: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Composition share of the *selected set* by category, per origin.

    For each origin *t*, returns the fraction of the variables selected at *t*
    that belong to each category (rows sum to 1 when anything is selected). This
    is the binary-model analogue of the SHAP mass share.
    """
    cats = attach_category(matrix.columns, metadata)
    counts = matrix.T.groupby(cats).sum().T  # origins x categories (#selected)
    counts = counts.reindex(columns=list(CATEGORY_ORDER), fill_value=0)
    return _normalise_rows(counts)


def category_inclusion_probability(
    matrix: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Inclusion probability by category, per origin.

    Returns, for each origin, the fraction of *available* variables within a
    category that were selected (#selected_c / #available_c). Unlike the
    composition share this does **not** sum to one; it measures how aggressively
    a model taps each block irrespective of block size.
    """
    cats = attach_category(matrix.columns, metadata)
    selected = matrix.T.groupby(cats).sum().T
    available = cats.value_counts()
    selected = selected.reindex(columns=list(CATEGORY_ORDER), fill_value=0)
    available = available.reindex(list(CATEGORY_ORDER)).replace(0, np.nan)
    return selected.div(available, axis=1).fillna(0.0)


def category_share_from_shap(
    shap: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Quarterly category share of total |SHAP| mass (XGBoost).

    Lags are already aggregated into ``series_id``. For each quarter, |SHAP| is
    summed within category and normalised by the quarter total → rows sum to 1.
    """
    df = shap.copy()
    df["category"] = attach_category(df["series_id"].values, metadata).values
    grouped = (
        df.groupby(["quarter", "category"])["mean_abs_shap"].sum().unstack("category")
    )
    grouped = grouped.reindex(columns=list(CATEGORY_ORDER), fill_value=0.0).fillna(0.0)
    return _normalise_rows(grouped)


# ---- Publication-lag mass shares (Task-1 companion) -------------------------

def publag_bucket(metadata: pd.DataFrame) -> pd.Series:
    """Map each series id → publication-lag bucket label in ``PUBLAG_ORDER``.

    pub_lag values 0/1/2 are kept as their own buckets; any other or missing
    value is routed to ``"n/a"`` so no mass is silently dropped.
    """
    lag = pd.to_numeric(metadata["pub_lag"], errors="coerce")

    def _lab(v: object) -> str:
        """Map one raw publication lag to a supported display bucket."""
        if pd.isna(v):
            return "n/a"
        iv = int(v)
        return str(iv) if str(iv) in PUBLAG_ORDER else "n/a"

    return lag.map(_lab).rename("publag")


def publag_share_from_binary(
    matrix: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Composition share of the selected set by publication-lag bucket, per origin."""
    buckets = publag_bucket(metadata).reindex(matrix.columns).fillna("n/a")
    counts = matrix.T.groupby(buckets).sum().T
    counts = counts.reindex(columns=list(PUBLAG_ORDER), fill_value=0)
    return _normalise_rows(counts)


def publag_share_from_long(
    df: pd.DataFrame, metadata: pd.DataFrame, value_col: str
) -> pd.DataFrame:
    """Quarterly share of total mass (``value_col``) by publication-lag bucket.

    Generic over the long-format SHAP (|SHAP|) log, which exposes
    ``series_id`` and ``quarter`` columns.
    """
    work = df.copy()
    work["publag"] = (
        publag_bucket(metadata).reindex(work["series_id"].values).fillna("n/a").values
    )
    grouped = work.groupby(["quarter", "publag"])[value_col].sum().unstack("publag")
    grouped = grouped.reindex(columns=list(PUBLAG_ORDER), fill_value=0.0).fillna(0.0)
    return _normalise_rows(grouped)


def universe_publag_share(metadata: pd.DataFrame) -> pd.Series:
    """Population share of the modelled universe in each publication-lag bucket.

    This is the *availability baseline*: the lag-bucket composition a model would
    reproduce if it selected indicators at random irrespective of timeliness.
    """
    buckets = publag_bucket(metadata)
    share = buckets.value_counts(normalize=True)
    return share.reindex(list(PUBLAG_ORDER)).fillna(0.0)


def to_quarterly(monthly_share: pd.DataFrame) -> pd.DataFrame:
    """Average a monthly category-share panel to a quarterly one (common axis)."""
    out = monthly_share.copy()
    out.index = out.index.asfreq("Q")
    return out.groupby(level=0).mean()


def regime_category_share(
    quarterly_share: pd.DataFrame, weights: pd.Series | None = None
) -> pd.DataFrame:
    """Average a quarterly category-share panel within each COVID regime.

    Parameters
    ----------
    quarterly_share
        Index of quarterly Periods, columns of categories, rows ~ sum to 1.
    weights
        Optional per-quarter weights (e.g. number of features); defaults to equal
        weighting of the quarters observed in each regime.

    Returns a (regime x category) frame, regimes ordered ``REGIME_ORDER``.
    """
    reg = quarterly_share.copy()
    reg["regime"] = [regime_of(p) for p in reg.index]
    if weights is not None:
        reg["__w"] = weights.reindex(quarterly_share.index).fillna(0.0).values
        out = (
            reg.drop(columns="regime")
            .mul(reg["__w"], axis=0)
            .groupby(reg["regime"])
            .sum()
        )
        wsum = reg.groupby("regime")["__w"].sum()
        out = out.div(wsum, axis=0).drop(columns="__w")
    else:
        out = reg.groupby("regime").mean(numeric_only=True)
    return out.reindex(REGIME_ORDER).reindex(columns=list(CATEGORY_ORDER))


# ---- Ranking scores (for the consensus task) --------------------------------

def ranking_binary(matrix: pd.DataFrame) -> pd.Series:
    """Mean selection rate per series over all origins (∈ [0, 1])."""
    return matrix.mean(axis=0).rename("score")


def ranking_shap(shap: pd.DataFrame) -> pd.Series:
    """Global mean |SHAP| per series (lags aggregated, averaged over quarters)."""
    per_q = shap.groupby(["quarter", "series_id"])["mean_abs_shap"].sum()
    return per_q.groupby("series_id").mean().rename("score")


def load_ifocast_membership() -> list[str]:
    """Active ifoCAST predictor ids (``in_panel == True``; GDP target excluded)."""
    path = P.OUT_NOWCASTING / "ifocast_indicator_mapping.csv"
    if not path.exists():
        return []
    m = pd.read_csv(path)
    pred = m[~m["my_id"].isin(["(target)"]) & m["my_id"].notna()].copy()
    if "in_panel" in pred.columns:
        pred = pred[pred["in_panel"].astype(bool)]
    ids = pred["my_id"].astype(str)
    ids = ids[~ids.isin(["nan", ""])]
    return list(dict.fromkeys(ids))


def ranking_fixed(ids: Sequence[str]) -> pd.Series:
    """Equal weight for each member of a fixed expert set."""
    return pd.Series(1.0, index=list(ids), name="score")


def top_n(score: pd.Series, n: int = 15) -> pd.Index:
    """Return the index of the top-``n`` series by score (descending)."""
    return score.sort_values(ascending=False).head(n).index


def spearman_method_matrix(scores: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Spearman rank-correlation matrix across methods over the union universe.

    Each method's score Series is reindexed onto the union of all series ids and
    missing values filled with 0 (a not-selected / zero-importance series ranks
    at the bottom). Spearman ρ is then computed pairwise.

    Note: 0-filling introduces ties at the bottom; this is the standard, robust
    convention for comparing sparse selection/attribution signals and biases ρ
    toward agreement only insofar as methods jointly ignore the same series.
    """
    from scipy.stats import spearmanr

    universe = sorted(set().union(*[s.index for s in scores.values()]))
    aligned = pd.DataFrame(
        {m: s.reindex(universe).fillna(0.0) for m, s in scores.items()}
    )
    methods = list(scores.keys())
    out = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for a in methods:
        for b in methods:
            rho, _ = spearmanr(aligned[a], aligned[b])
            out.loc[a, b] = rho
    return out.astype(float)


def consensus_membership(
    scores: Mapping[str, pd.Series], metadata: pd.DataFrame, n: int = 15
) -> pd.DataFrame:
    """Union of each method's top-``n`` series with per-method percentile ranks.

    Returns a frame indexed by series id (plus a ``name`` column) whose method
    columns hold the within-method percentile rank (0–1, higher = more important)
    of each series. Cells where a series is not scored by a method are 0.
    This is the data behind the cross-model consensus heatmap.
    """
    keep: set[str] = set()
    for s in scores.values():
        keep |= set(top_n(s, n))
    keep_idx = sorted(keep)

    cols: dict[str, pd.Series] = {}
    for m, s in scores.items():
        pct = s.rank(pct=True)  # within-method percentile
        cols[m] = pct.reindex(keep_idx).fillna(0.0)
    out = pd.DataFrame(cols, index=keep_idx)
    out.insert(0, "name", metadata["name"].reindex(keep_idx).values)
    out.insert(1, "category", attach_category(keep_idx, metadata).values)
    # Order rows by mean consensus rank (most broadly agreed-upon at the top).
    out = out.assign(_consensus=out[list(scores)].mean(axis=1)).sort_values(
        "_consensus", ascending=False
    )
    return out.drop(columns="_consensus")


# =============================================================================
# 3. Aesthetics helpers
# =============================================================================

def _apply_mpl_style() -> None:
    """Apply a clean, thesis-grade matplotlib style (soft grid, light frame)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#9aa0a6",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#e7e7ea",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
        }
    )


def _cat_color(cat: str) -> str:
    """Return the stable plot color for an economic category."""
    return SORBET.get(cat, "#D9D2C5")


def _cat_label(cat: str) -> str:
    """Return the publication label for an economic category."""
    return CATEGORY_DISPLAY.get(cat, cat)


def save_fig(fig: Any, path: Path) -> None:
    """Save a matplotlib figure, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"[saved] {path}")


def _plotly_layout(fig: Any, title: str, height: int = 600) -> None:
    """Apply the shared luminous-pastel layout to a plotly figure."""
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=18)),
        font=dict(family="DejaVu Sans, Arial", size=12, color="#33373d"),
        height=height,
        margin=dict(l=70, r=40, t=70, b=60),
        legend=dict(bgcolor="rgba(255,255,255,0.6)"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#ececf0", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#ececf0", zeroline=False)


# =============================================================================
# 4. TASK 1 — Macro-category structural shift through time
# =============================================================================

def _present_categories(panels: Mapping[str, pd.DataFrame]) -> list[str]:
    """Categories with non-trivial mass in at least one model (keeps legends lean)."""
    present: list[str] = []
    for c in CATEGORY_ORDER:
        if any((p.get(c, pd.Series(dtype=float)).abs().sum() > 1e-9) for p in panels.values()):
            present.append(c)
    return present


def plot_task1_structural_shift(
    panels: Mapping[str, pd.DataFrame], out_path: Path
) -> Any:
    """Multi-panel stacked-area chart of category mass share through time.

    Parameters
    ----------
    panels
        Ordered mapping ``model label -> quarterly category-share frame`` (index
        of quarterly Periods, columns ⊆ ``CATEGORY_ORDER``, rows ~ sum to 1).
    out_path
        PNG destination.

    Economic reading: a rising rose band (Surveys) at the expense of the
    green/blue real-activity bands signals a rotation **from hard to soft data**,
    typically around turning points where survey signals lead realised activity.
    """
    import matplotlib.pyplot as plt

    _apply_mpl_style()
    cats = _present_categories(panels)
    labels = list(panels.keys())
    n = len(labels)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(13, 3.0 * nrow), sharex=True, squeeze=False
    )
    axes_flat = axes.ravel()

    for ax, label in zip(axes_flat, labels):
        df = panels[label].reindex(columns=cats).fillna(0.0)
        x = df.index.to_timestamp()
        ax.stackplot(
            x,
            [df[c].values for c in cats],
            colors=[_cat_color(c) for c in cats],
            labels=[_cat_label(c) for c in cats],
            edgecolor="white",
            linewidth=0.3,
        )
        ax.set_title(label)
        ax.set_ylim(0, 1)
        ax.set_ylabel("category mass share")
        ax.margins(x=0)
        # COVID shading 2020–2021 for visual anchoring.
        ax.axvspan(
            pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"),
            color="#9aa0a6", alpha=0.10, lw=0,
        )

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    handles, leg_labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, leg_labels, loc="lower center", ncol=min(len(cats), 6),
        bbox_to_anchor=(0.5, -0.02), fontsize=9,
    )
    fig.suptitle(
        "Structural shift in selection focus — category mass share, 2011–2025",
        fontsize=14, fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_fig(fig, out_path)
    plt.close(fig)
    return fig


def plot_task1_publag_companion(
    panels: Mapping[str, pd.DataFrame],
    baseline: pd.Series,
    out_path: Path,
) -> Any:
    """Companion panel: selection/attention mass share by **publication lag**.

    Disentangles *timeliness/availability* from *informativeness*. lag-0 series
    (surveys, sentiment) are released the same month and are therefore available
    at every forecast origin, while lag-1/2 hard-activity series are not yet
    published at short horizons. The dashed line marks the **universe lag-0
    availability baseline** (the lag-0 share a model would reproduce by selecting
    at random): a rose band whose top edge sits *above* the baseline indicates a
    genuine tilt toward timely soft data beyond what availability alone implies;
    a band that merely tracks the baseline reflects the information set, not a
    preference (Bańbura & Rünstler 2011).

    Parameters
    ----------
    panels
        Ordered mapping ``model label -> quarterly publication-lag-share frame``
        (index of quarterly Periods, columns ⊆ ``PUBLAG_ORDER``, rows ~ sum to 1).
    baseline
        Universe lag-bucket shares from :func:`universe_publag_share`.
    out_path
        PNG destination.
    """
    import matplotlib.pyplot as plt

    _apply_mpl_style()
    buckets = [b for b in PUBLAG_ORDER
               if any((p.get(b, pd.Series(dtype=float)).abs().sum() > 1e-9)
                      for p in panels.values())]
    labels = list(panels.keys())
    n = len(labels)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(13, 3.0 * nrow), sharex=True, squeeze=False
    )
    axes_flat = axes.ravel()
    lag0_base = float(baseline.get("0", np.nan))

    for ax, label in zip(axes_flat, labels):
        df = panels[label].reindex(columns=buckets).fillna(0.0)
        x = df.index.to_timestamp()
        ax.stackplot(
            x,
            [df[b].values for b in buckets],
            colors=[PUBLAG_COLORS.get(b, "#D9D2C5") for b in buckets],
            labels=[PUBLAG_LABELS.get(b, b) for b in buckets],
            edgecolor="white",
            linewidth=0.3,
        )
        if np.isfinite(lag0_base):
            ax.axhline(lag0_base, color="#33373d", lw=1.1, ls="--", zorder=5)
        ax.set_title(label)
        ax.set_ylim(0, 1)
        ax.set_ylabel("mass share by pub. lag")
        ax.margins(x=0)
        ax.axvspan(
            pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"),
            color="#9aa0a6", alpha=0.10, lw=0,
        )

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    handles, leg_labels = axes_flat[0].get_legend_handles_labels()
    if np.isfinite(lag0_base):
        from matplotlib.lines import Line2D
        handles = handles + [Line2D([0], [0], color="#33373d", lw=1.1, ls="--")]
        leg_labels = leg_labels + [
            f"universe lag-0 baseline = {lag0_base:.0%}"
        ]
    fig.legend(
        handles, leg_labels, loc="lower center", ncol=min(len(leg_labels), 5),
        bbox_to_anchor=(0.5, -0.02), fontsize=9,
    )
    fig.suptitle(
        "Availability vs informativeness — selection mass by publication lag, "
        "2011–2025",
        fontsize=14, fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_fig(fig, out_path)
    plt.close(fig)
    return fig


def plot_task1_interactive(
    panels: Mapping[str, pd.DataFrame], out_html: Path
) -> Any:
    """Interactive (plotly) faceted stacked-area version of Task 1."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    cats = _present_categories(panels)
    labels = list(panels.keys())
    ncol = 2
    nrow = int(np.ceil(len(labels) / ncol))
    fig = make_subplots(
        rows=nrow, cols=ncol, subplot_titles=labels,
        shared_xaxes=False, vertical_spacing=0.10, horizontal_spacing=0.08,
    )
    for i, label in enumerate(labels):
        r, c = i // ncol + 1, i % ncol + 1
        df = panels[label].reindex(columns=cats).fillna(0.0)
        x = df.index.to_timestamp()
        for cat in cats:
            fig.add_trace(
                go.Scatter(
                    x=x, y=df[cat].values, name=_cat_label(cat),
                    mode="lines", stackgroup=f"g{i}",
                    line=dict(width=0.5, color=_cat_color(cat)),
                    fillcolor=_cat_color(cat),
                    legendgroup=cat, showlegend=(i == 0),
                    hovertemplate=f"{_cat_label(cat)}: "+"%{y:.1%}<extra></extra>",
                ),
                row=r, col=c,
            )
        fig.update_yaxes(range=[0, 1], tickformat=".0%", row=r, col=c)
    _plotly_layout(
        fig,
        "Category mass share through time — EN / PLS / XGBoost SHAP",
        height=320 * nrow,
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"[saved] {out_html}")
    return fig


# =============================================================================
# 5. TASK 2 — Cross-model selection consensus
# =============================================================================

def plot_task2_spearman(rho: pd.DataFrame, out_path: Path) -> Any:
    """Heatmap of the Spearman rank-correlation between methods' importance scores."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    _apply_mpl_style()
    cmap = LinearSegmentedColormap.from_list(
        "sorbet_div", ["#9FC8E8", "#FBF7EF", "#F2A6B3"]
    )
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(rho), 0.9 + 0.8 * len(rho)))
    im = ax.imshow(rho.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(rho)))
    ax.set_yticks(range(len(rho)))
    ax.set_xticklabels(rho.columns, rotation=40, ha="right")
    ax.set_yticklabels(rho.index)
    for i in range(len(rho)):
        for j in range(len(rho)):
            v = rho.values[i, j]
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center",
                color="#33373d" if abs(v) < 0.7 else "white", fontsize=9,
            )
    ax.set_title("Cross-model agreement on indicator importance (Spearman ρ)")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("rank correlation")
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    return fig


def plot_task2_consensus_heatmap(
    membership: pd.DataFrame, score_cols: Sequence[str], out_path: Path
) -> Any:
    """Aligned heatmap of top-indicator percentile ranks across methods.

    Rows are the union of every method's top indicators (labelled by readable
    name + category colour chip); columns are methods; cell intensity is the
    within-method percentile rank. Broad horizontal bands of colour flag the
    indicators every model agrees on (the German "core" — e.g. Ifo climate,
    industrial production).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    _apply_mpl_style()
    data = membership[list(score_cols)].astype(float)
    cmap = LinearSegmentedColormap.from_list(
        "sorbet_seq", ["#FBF7EF", "#F9DFA6", "#F7C59F", "#F2A6B3"]
    )
    fig, ax = plt.subplots(
        figsize=(1.6 + 0.95 * len(score_cols), 1.2 + 0.34 * len(data))
    )
    im = ax.imshow(data.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(score_cols)))
    ax.set_xticklabels(score_cols, rotation=40, ha="right")

    # Readable row labels: truncated name + category colour chip on the left.
    short = membership["name"].str.replace(
        r"^Germany,\s*", "", regex=True
    ).str.slice(0, 46)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(
        [f"{sid}  ·  {nm}" for sid, nm in zip(membership.index, short)], fontsize=7.5
    )
    for i, cat in enumerate(membership["category"]):
        ax.add_patch(
            plt.Rectangle((-0.7, i - 0.5), 0.18, 1, color=_cat_color(cat),
                          clip_on=False, transform=ax.transData)
        )
    for i in range(len(data)):
        for j in range(len(score_cols)):
            v = data.values[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color="#33373d")
    ax.set_title("Top-indicator consensus across methods (within-method percentile rank)")
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("percentile rank")
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    return fig


# =============================================================================
# 6. TASK 3 — Regime-switching analysis
# =============================================================================

def plot_task3_regime_bars(
    regime_panels: Mapping[str, pd.DataFrame], out_path: Path
) -> Any:
    """Grouped bars: one panel per COVID regime, bars per category (EN / SHAP).

    Parameters
    ----------
    regime_panels
        ``model label -> (regime x category)`` frame (rows ~ sum to 1).

    Layout: pre-COVID | COVID | post-COVID subplots; within each subplot, grouped
    bars compare EN (smoothed) and XGBoost (SHAP) category mass shares.

    Economic reading: comparing the rose (Surveys) and green (Production) bars
    across methods *within* a regime shows whether linear screening and tree-based
    attribution agree on the macro slice that matters in that phase of the cycle.
    """
    import matplotlib.pyplot as plt

    _apply_mpl_style()
    cats = _present_categories(regime_panels)
    methods = [m for m in TASK3_METHOD_ORDER if m in regime_panels]
    if not methods:
        raise ValueError("plot_task3_regime_bars: no Task 3 methods in regime_panels.")

    n_reg = len(REGIME_ORDER)
    fig, axes = plt.subplots(
        1, n_reg, figsize=(4.8 * n_reg, 4.0), squeeze=False, sharey=True
    )
    axes_flat = axes.ravel()
    x = np.arange(len(cats))
    n_m = len(methods)
    width = 0.8 / max(n_m, 1)

    for ax, reg in zip(axes_flat, REGIME_ORDER):
        for k, method in enumerate(methods):
            row = (
                regime_panels[method]
                .reindex(index=REGIME_ORDER, columns=cats)
                .fillna(0.0)
                .loc[reg]
            )
            offset = (k - (n_m - 1) / 2) * width
            ax.bar(
                x + offset,
                row.values,
                width,
                label=TASK3_METHOD_LABELS.get(method, method),
                color=TASK3_METHOD_COLORS.get(method, "#D9D2C5"),
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_title(reg, fontweight="semibold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_cat_label(c) for c in cats], rotation=40, ha="right", fontsize=8
        )
        ax.set_ylabel("category mass share")

    handles, leg = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, leg, loc="upper center", ncol=n_m, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Regime-switching focus: category mass share by COVID regime "
        "(EN vs XGBoost SHAP)",
        y=1.06,
        fontsize=14,
        fontweight="semibold",
    )
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    return fig


def plot_task3_soft_hard(
    regime_panels: Mapping[str, pd.DataFrame], out_path: Path
) -> Any:
    """Soft-vs-hard summary: stacked share of survey vs real-activity mass by regime.

    Collapses the category panels into a soft (Surveys) / hard (real activity) /
    other split and shows, per model, how the soft share moves across regimes —
    the single clearest test of the "dump hard data for soft data" hypothesis.
    """
    import matplotlib.pyplot as plt

    _apply_mpl_style()
    labels = list(regime_panels.keys())
    soft = pd.DataFrame(index=REGIME_ORDER, columns=labels, dtype=float)
    for label, df in regime_panels.items():
        d = df.reindex(index=REGIME_ORDER, columns=list(CATEGORY_ORDER)).fillna(0.0)
        soft[label] = d[list(SOFT_CATEGORIES & set(d.columns))].sum(axis=1)

    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(1.4 + 1.05 * len(labels), 4.2))
    for k, reg in enumerate(REGIME_ORDER):
        ax.bar(
            x + (k - 1) * width, soft.loc[reg].values, width,
            label=reg, color=REGIME_COLORS[reg], edgecolor="white", linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("soft (survey) mass share")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="#9aa0a6", lw=0.8, ls="--")
    ax.set_title("Survey mass share by regime (universe survey share is 0.67)", pad=34)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    return fig


# =============================================================================
# 7. Interpretation framework
# =============================================================================

INTERPRETATION_GUIDE = """\
# Reading the cross-method selection diagnostics

These figures support Part I of the thesis. They do not identify a unique
German nowcasting set.

## Common scale: category mass share
Each method has a native signal (binary inclusion for EN / PLS / block-balanced;
mean |SHAP| for XGBoost). All of them are mapped onto the share of that method's
own selected mass in each economic category. Shares sum to one within a method,
so a rise in one category is a fall elsewhere. Compare mix, not absolute
importance, across methods.

## What the thesis finds
- **Category tilt.** Every data-driven method places 65–100% of selected mass on
  delayed hard activity (production, turnover, orders, trade, construction)
  against a 29% universe share, and under-weights lag-0 series relative to the
  panel's 70% lag-0 share. That rejects a soft-data-dominance reading of this
  completed-quarter selection problem.
- **Series disagreement.** Spearman rank correlations among the four methods are
  0.28–0.46. Only two series are selected by the elastic net at every origin.
  Mean Jaccard overlap with the frozen ifoCAST set is 0.11. Agreement is about
  *kind* of data, not about a shared list.
- **COVID rotation is estimator-sensitive.** The pooled EN survey share rises
  during COVID, but three of sixty refits carry it and it reverses quickly.
  XGBoost moves the other way (survey share falls as production importance
  rises). Do not read a method-independent shift towards sentiment.

## Task 1 — category mix through time
Read the stacked areas for the *level* of concentration first, then for
movement. Turnover, orders and production dominate every panel. Isolated survey
spikes on the EN path are substitutions under the 60-series cap, not a new
regime. Part I scores completed-quarter association, so publication timing
confers no advantage; surveys can still matter in Part II before hard releases
arrive.

## Task 2 — cross-method ranks
Spearman ρ is computed over the union of ever-weighted series, with absent
weights set to zero. Modest ρ is the expected result when many collinear
indicators measure the same activity block. Bright cells in only one column of
the consensus heatmap are method-specific stand-ins for that block, not a
reason to drop the category.

## Task 3 — regime bars
Hard-activity majorities in every window are the finding. A COVID rise in
survey mass that appears in one method and not in the others is not a
structural shift. PLS remains essentially 100% hard in every window.

## Rules of thumb
1. Cite a category pattern only when it appears in more than one estimator
   family.
2. Keep availability (publication lag) separate from completed-quarter
   association.
3. Do not promote the vote set, the ifoCAST list, or the elastic-net list as
   *the* German nowcasting set. Part II tests those inputs downstream and does
   not reject equal accuracy among them.
"""


def write_interpretation(out_path: Path) -> None:
    """Persist the interpretation framework next to the figures."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(INTERPRETATION_GUIDE, encoding="utf-8")
    print(f"[saved] {out_path}")


# =============================================================================
# 8. Orchestration
# =============================================================================

@dataclass
class Inputs:
    """Container for all loaded inputs (kept together for reuse / testing)."""

    metadata: pd.DataFrame
    binaries: dict[str, pd.DataFrame]
    shap: pd.DataFrame


def load_inputs() -> Inputs:
    """Load every artefact required by the three tasks."""
    return Inputs(
        metadata=load_metadata(),
        binaries=load_all_binary_matrices(),
        shap=load_shap(),
    )


def build_time_panels(inp: Inputs) -> dict[str, pd.DataFrame]:
    """Assemble the quarterly category-share panels used by Task 1.

    Selects a representative, non-redundant subset of models for the time-series
    comparison: EN (smoothed), PLS and XGBoost SHAP.
    """
    panels: dict[str, pd.DataFrame] = {}
    if "EN (raw)" in inp.binaries:
        panels["EN (raw)"] = to_quarterly(
            category_share_from_binary(inp.binaries["EN (raw)"], inp.metadata)
        )
    if "PLS" in inp.binaries:
        panels["PLS"] = to_quarterly(
            category_share_from_binary(inp.binaries["PLS"], inp.metadata)
        )
    panels["XGBoost (SHAP mass)"] = category_share_from_shap(inp.shap, inp.metadata)
    return panels


def build_publag_panels(inp: Inputs) -> dict[str, pd.DataFrame]:
    """Assemble the quarterly publication-lag-share panels for the Task-1 companion.

    Mirrors the model subset of :func:`build_time_panels` so the companion panel
    aligns 1:1 with the category structural-shift figure.
    """
    panels: dict[str, pd.DataFrame] = {}
    if "EN (raw)" in inp.binaries:
        panels["EN (raw)"] = to_quarterly(
            publag_share_from_binary(inp.binaries["EN (raw)"], inp.metadata)
        )
    if "PLS" in inp.binaries:
        panels["PLS"] = to_quarterly(
            publag_share_from_binary(inp.binaries["PLS"], inp.metadata)
        )
    panels["XGBoost (SHAP mass)"] = publag_share_from_long(
        inp.shap, inp.metadata, "mean_abs_shap"
    )
    return panels


def build_regime_panels(inp: Inputs) -> dict[str, pd.DataFrame]:
    """Assemble (regime x category) share panels used by Task 3."""
    panels: dict[str, pd.DataFrame] = {}
    if "EN (raw)" in inp.binaries:
        en_q = to_quarterly(
            category_share_from_binary(inp.binaries["EN (raw)"], inp.metadata)
        )
        panels["EN (raw)"] = regime_category_share(en_q)
    panels["XGBoost (SHAP)"] = regime_category_share(
        category_share_from_shap(inp.shap, inp.metadata)
    )
    return panels


def build_ranking_scores(inp: Inputs) -> dict[str, pd.Series]:
    """Per-method global importance scores used by the consensus task."""
    scores: dict[str, pd.Series] = {}
    for label in ("EN (raw)", "PLS", "Block-balanced (k=20)"):
        if label in inp.binaries:
            scores[label] = ranking_binary(inp.binaries[label])
    scores["XGBoost"] = ranking_shap(inp.shap)
    ifocast_ids = load_ifocast_membership()
    if ifocast_ids:
        scores["ifoCAST (fixed)"] = ranking_fixed(ifocast_ids)
    return scores


def main() -> None:
    """Run all three tasks end-to-end and write every artefact to ``FIG_DIR``."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading inputs ...")
    inp = load_inputs()

    # ---- Task 1 -------------------------------------------------------------
    print("\n[Task 1] Macro-category structural shift")
    time_panels = build_time_panels(inp)
    plot_task1_structural_shift(time_panels, FIG_DIR / "selcmp_task1_structural_shift.png")
    plot_task1_interactive(time_panels, FIG_DIR / "selcmp_task1_structural_shift.html")
    # Companion panel: decompose the same selection mass by publication lag to
    # separate availability/timeliness from genuine soft-vs-hard preference.
    publag_panels = build_publag_panels(inp)
    plot_task1_publag_companion(
        publag_panels,
        universe_publag_share(inp.metadata),
        FIG_DIR / "selcmp_task1_publag_companion.png",
    )

    # ---- Task 2 -------------------------------------------------------------
    print("\n[Task 2] Cross-model selection consensus")
    scores = build_ranking_scores(inp)
    rho = spearman_method_matrix(scores)
    rho.to_csv(FIG_DIR / "selcmp_task2_spearman.csv")
    plot_task2_spearman(rho, FIG_DIR / "selcmp_task2_spearman.png")

    # Consensus heatmap: compare the headline method families.
    consensus_methods = [
        m for m in (
            "EN (raw)", "PLS", "Block-balanced (k=20)", "XGBoost",
            "ifoCAST (fixed)",
        )
        if m in scores
    ]
    membership = consensus_membership(
        {m: scores[m] for m in consensus_methods}, inp.metadata, n=15
    )
    membership.to_csv(FIG_DIR / "selcmp_task2_consensus.csv")
    plot_task2_consensus_heatmap(
        membership, consensus_methods, FIG_DIR / "selcmp_task2_consensus.png"
    )

    # ---- Task 3 -------------------------------------------------------------
    print("\n[Task 3] Regime-switching analysis")
    regime_panels = build_regime_panels(inp)
    plot_task3_regime_bars(regime_panels, FIG_DIR / "selcmp_task3_regime_bars.png")
    plot_task3_soft_hard(regime_panels, FIG_DIR / "selcmp_task3_soft_hard.png")

    # ---- Interpretation -----------------------------------------------------
    write_interpretation(FIG_DIR / "selcmp_interpretation.md")
    print(f"\nDone. All artefacts in: {FIG_DIR}")


if __name__ == "__main__":
    main()
