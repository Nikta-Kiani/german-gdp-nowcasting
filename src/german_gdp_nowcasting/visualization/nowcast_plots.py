"""Economics-focused plots for the DFM nowcasting notebook.

Styling and palette stay consistent across the thesis figures: white
background, no top/right spines, a light dashed y-grid, a single COVID
band (2020Q1–2021Q4), percentage-point y-units, and a colour-blind-aware
palette across DFM input sets and benchmarks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

MODEL_COLORS: dict[str, str] = {
    # --- Baselines (soft blue-slate, light → anchor) ---
    "RW":         "#DCE4F0",
    "AR1":        "#9AADC4",
    "ARp":        "#5C7291",

    # --- DFM / A-CD-TPN (blush–rose) ---
    "en_only":    "#E07D96",
    "ifoCAST":    "#C9617F",
    "blockbalanced": "#8F3D58",

    # --- Stochastic volatility ---
    "SV_k2":      "#3D5FAE",

    # --- Ground truth & context ---
    "actual":     "#1A2332",
    "recession":  "#EEF2F8",
}

MODEL_LABELS: dict[str, str] = {
    "en_only":    "DFM-EN",
    "ifoCAST":    "DFM-ifoCAST",
    "blockbalanced": "DFM-k20",
    "AR1":        "AR(1)",
    "RW":         "RW",
    "ARp":        "AR(p)-AIC",
    "SV_k2":      "DFM-SV (k=2, integrated)",
}

# Only the COVID episode is highlighted (2020Q1–2021Q4). The GFC and
# euro-crisis bands are omitted so the evaluation windows stay readable.
DEFAULT_RECESSIONS: list[tuple[str, str]] = [
    ("2020Q1", "2021Q4"),
]


# Curated qualitative palette for the indicator categories in the configured
# enriched metadata. Order matches the 11 categories used in this thesis
# (Surveys, Orders, Turnover, Production,
# Prices, Global, Construction, Financial, Trade, Misc, Commodities) so
# that any one chart picks colours consistently with neighbouring ones.
CATEGORY_COLORS: dict[str, str] = {
    "Surveys":      "#0B2545",
    "Orders":       "#E07A1F",
    "Turnover":     "#1FA489",
    "Production":   "#8E44AD",
    "Prices":       "#C2185B",
    "Global":       "#3F7CAC",
    "Construction": "#B7791F",
    "Financial":    "#11827C",
    "Trade":        "#9B2C2C",
    "Misc":         "#6B7280",
    "Commodities":  "#65A30D",
    "Unknown":      "#9CA3AF",
}

# Fallback ordered palette (used for categories not in CATEGORY_COLORS).
_CATEGORY_PALETTE: list[str] = list(CATEGORY_COLORS.values())


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def setup_style() -> None:
    """Apply consistent matplotlib rcParams across all figures.

    Aims for a clean, publication-grade look: sans-serif typography with
    Inter/Source Sans 3/Helvetica preferred, soft slate spines, very light
    dashed gridlines, and a tonal default colour cycle aligned with
    ``MODEL_COLORS``.
    """
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "#475569",
        "axes.linewidth": 0.9,
        "axes.labelcolor": "#1F2937",
        "axes.titlecolor": "#0F172A",
        "axes.titleweight": "600",
        "axes.titlesize": 12,
        "axes.titlepad": 10,
        "axes.labelsize": 9.5,
        "axes.labelpad": 6,

        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.linestyle": (0, (3, 3)),
        "grid.color": "#E2E8F0",
        "grid.alpha": 0.9,
        "grid.linewidth": 0.6,

        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.borderaxespad": 0.4,

        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica Neue", "Avenir Next", "Avenir", "SF Pro Display",
            "Helvetica", "Arial", "Inter", "Source Sans 3", "DejaVu Sans",
        ],
        "font.size": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype":  42,

        "axes.prop_cycle": cycler(color=[
            MODEL_COLORS["en_only"],
            MODEL_COLORS["ifoCAST"],
            MODEL_COLORS["blockbalanced"],
            MODEL_COLORS["SV_k2"],
            MODEL_COLORS["ARp"],
        ]),
    })


def add_recession_bands(
    ax: plt.Axes,
    periods: Sequence[tuple[str, str]] = DEFAULT_RECESSIONS,
    color: str = MODEL_COLORS["recession"],
    label: str | None = "COVID-19",
    annotate: bool = False,
) -> None:
    """Shade recession periods on a quarterly time axis.

    The shading is drawn *without* expanding the axis limits: only the
    portion of each band that intersects the current ``ax.get_xlim()`` is
    rendered. This avoids the empty-space artefact that occurred when
    pre-2011 bands forced matplotlib to widen the x-axis.
    """
    x_lo, x_hi = ax.get_xlim()
    for q0, q1 in periods:
        x0 = mdates.date2num(pd.Period(q0, freq="Q").to_timestamp(how="start"))
        x1 = mdates.date2num(pd.Period(q1, freq="Q").to_timestamp(how="end"))
        # Clip to current limits; skip if the band lies entirely outside.
        x0c, x1c = max(x0, x_lo), min(x1, x_hi)
        if x1c <= x0c:
            continue
        ax.axvspan(
            mdates.num2date(x0c), mdates.num2date(x1c),
            color=color, alpha=0.55, zorder=0, linewidth=0,
        )
        if annotate and label:
            y_lo, y_hi = ax.get_ylim()
            ax.text(
                mdates.num2date((x0c + x1c) / 2.0),
                y_hi - (y_hi - y_lo) * 0.04,
                label,
                ha="center", va="top",
                fontsize=7.5, color="#9D2C2C", alpha=0.85,
            )


def _q_to_ts(idx: pd.Index) -> pd.DatetimeIndex:
    """Quarter strings -> quarter-end timestamps."""
    return pd.PeriodIndex(idx, freq="Q").to_timestamp(how="end")


def _save(fig: plt.Figure, save: str | Path | None) -> None:
    """Save a figure when an output path is supplied."""
    if save is not None:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches="tight")


def _apply_year_axis(ax: plt.Axes, base: int = 2) -> None:
    """Format a quarter-end time axis with a clean year locator."""
    ax.xaxis.set_major_locator(mdates.YearLocator(base=base))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(0)
        lbl.set_ha("center")


# ---------------------------------------------------------------------------
# 1. Nowcast time-series grid
# ---------------------------------------------------------------------------

def fig_nowcast_grid(
    results_by_set: Mapping[str, pd.DataFrame],
    save: str | Path | None = None,
) -> plt.Figure:
    """2x2 panel: nowcast vs actual per input set, with the COVID band.

    All four panels share the same x-range and y-range. Year labels are
    shown on every panel (not just the bottom row). The COVID-19 recession
    band is annotated consistently on every panel.
    """
    keys = list(results_by_set.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True, sharey=True)
    for i, (ax, key) in enumerate(zip(axes.flat, keys)):
        df = results_by_set[key]
        x = _q_to_ts(df.index)
        ax.plot(x, df["actual"], color=MODEL_COLORS["actual"], lw=1.4,
                label="Actual GDP")
        ax.plot(x, df["nowcast"], color=MODEL_COLORS.get(key, "#1d4ed8"),
                lw=1.7, label=f"Nowcast ({MODEL_LABELS.get(key, key)})")
        ax.axhline(0, color="#CBD5E1", lw=0.8)
        ax.set_xlim(x[0], x[-1])
        add_recession_bands(ax, annotate=True)
        ax.set_title(MODEL_LABELS.get(key, key), fontsize=11)
        # y-label only on the left column
        if i % 2 == 0:
            ax.set_ylabel("Q/Q log-growth (pp)")
        ax.legend(loc="lower left", framealpha=0.85, fontsize=8)
        _apply_year_axis(ax, base=2)
        # Force year labels to show on all panels, not just the bottom row
        ax.tick_params(axis="x", labelbottom=True)
        plt.setp(ax.get_xticklabels(), visible=True)

    fig.suptitle(
        "Mixed-Frequency DFM Nowcasts of German GDP Growth by Indicator Set",
        fontsize=13, y=1.01, fontweight="600",
    )
    fig.tight_layout()
    _save(fig, save)
    return fig


# ---------------------------------------------------------------------------
# 2. Error panel
# ---------------------------------------------------------------------------

def fig_error_panel(
    results_by_set: Mapping[str, pd.DataFrame],
    save: str | Path | None = None,
    rolling_window: int = 4,
) -> plt.Figure:
    """Forecast errors (left) and rolling MAE (right) -- two-panel layout.

    The right panel is given more horizontal real estate and a year-based
    locator so quarterly tick labels no longer overlap.
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(15, 5.0),
        gridspec_kw={"width_ratios": [2.2, 1.6], "wspace": 0.22},
    )

    ax = axes[0]
    all_x: list[pd.DatetimeIndex] = []
    for key, df in results_by_set.items():
        x = _q_to_ts(df.index)
        all_x.append(x)
        ax.plot(x, df["error"], lw=1.2, color=MODEL_COLORS.get(key, "#2F5D8A"),
                label=MODEL_LABELS.get(key, key))
    ax.axhline(0, color=MODEL_COLORS["actual"], lw=0.8)
    if all_x:
        xmin = min(x[0] for x in all_x)
        xmax = max(x[-1] for x in all_x)
        ax.set_xlim(xmin, xmax)
    ax.set_ylabel("Nowcast - actual (pp)")
    ax.set_title("Forecast errors over time")
    ax.legend(ncol=min(len(results_by_set), 4), loc="lower left", framealpha=0.85)
    add_recession_bands(ax, annotate=True)
    _apply_year_axis(ax, base=2)

    ax2 = axes[1]
    all_x2: list[pd.DatetimeIndex] = []
    for key, df in results_by_set.items():
        rolling = df["error"].abs().rolling(rolling_window, min_periods=2).mean()
        x = _q_to_ts(rolling.index)
        all_x2.append(x)
        ax2.plot(x, rolling,
                 color=MODEL_COLORS.get(key, "#2F5D8A"), lw=1.1,
                 label=MODEL_LABELS.get(key, key))
    if all_x2:
        xmin = min(x[0] for x in all_x2)
        xmax = max(x[-1] for x in all_x2)
        ax2.set_xlim(xmin, xmax)
    ax2.set_title(f"Rolling {rolling_window}-quarter MAE")
    ax2.set_ylabel("MAE (pp)")
    add_recession_bands(ax2)
    _apply_year_axis(ax2, base=2)

    fig.tight_layout()
    _save(fig, save)
    return fig


# ---------------------------------------------------------------------------
# 3. RMSFE bar chart with DM significance markers
# ---------------------------------------------------------------------------

# Footer spacing for ``fig_rmsfe_bar`` (03_rmsfe_bar.png). Tune here if the gap
# below the plot looks too large or the legend overlaps the x-axis label.
_RMSFE_BAR_FOOTER = {
    "fig_extra_height": 1.1,   # added to 0.38 * n_models (figure inches)
    "legend_anchor_y": -0.2,   # axes coords; more negative → legend lower
    "margin_bottom": 0.11,      # tight_layout rect bottom; smaller → less white gap
}


def fig_rmsfe_bar(
    rmsfe: pd.Series,
    dm_pvals_vs_ar1: pd.Series | None = None,
    save: str | Path | None = None,
    alpha: float = 0.05,
    nsr_threshold: float | None = None,
) -> plt.Figure:
    """Compact horizontal bar chart of full-sample RMSFE.

    Models are sorted best (lowest) at top to worst at bottom.
    A dashed vertical line marks the AR(1) benchmark. Labels appended
    with * denote models for which the HLN-corrected Diebold–Mariano
    test rejects equal squared-error accuracy versus the expanding AR(1)
    at ``alpha``. Over the thesis full sample no model rejects.

    A solid vertical line at ``nsr_threshold`` (std of GDP actuals) marks the
    practical-relevance threshold from Lehmann et al. (2020): models to the
    left of this line have NSR < 1 (RMSFE < GDP variability).

    Parameters
    ----------
    rmsfe           : pd.Series mapping model name → RMSFE.
    dm_pvals_vs_ar1 : pd.Series mapping model name → DM p-value vs AR(1).
    alpha           : significance level for DM star annotation.
    nsr_threshold   : std of GDP actuals (= RMSFE at which NSR = 1).
                      Pass ``y_quarterly.std()`` to enable the NSR line.
    """
    order = rmsfe.sort_values(ascending=False)   # worst→best, top→bottom
    labels = []
    for k in order.index:
        lbl = MODEL_LABELS.get(k, k)
        if dm_pvals_vs_ar1 is not None:
            p = dm_pvals_vs_ar1.get(k, np.nan)
            if pd.notna(p) and p < alpha:
                lbl = lbl + "  *"
        labels.append(lbl)

    colors = [MODEL_COLORS.get(k, "#1d4ed8") for k in order.index]

    _f = _RMSFE_BAR_FOOTER
    fig, ax = plt.subplots(figsize=(7.0, 0.38 * len(order) + _f["fig_extra_height"]))
    bars = ax.barh(
        np.arange(len(order)), order.values,
        color=colors, edgecolor="white", linewidth=0.5,
        height=0.62, alpha=0.92,
    )

    # Numeric labels just past the bar end
    x_pad = order.max() * 0.012
    for bar, val in zip(bars, order.values):
        ax.text(val + x_pad, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8.2, color="#1F2937")

    # AR(1) reference line
    ar1_rmsfe = rmsfe.get("AR1", np.nan)
    legend_handles = []
    if pd.notna(ar1_rmsfe):
        ln1 = ax.axvline(ar1_rmsfe, color="#64748B", lw=1.1, ls="--", alpha=0.75,
                         label=f"AR(1)  {ar1_rmsfe:.3f}")
        legend_handles.append(ln1)

    # NSR = 1 threshold: RMSFE equals GDP standard deviation
    if nsr_threshold is not None and np.isfinite(nsr_threshold):
        ln2 = ax.axvline(
            nsr_threshold, color="#B7791F", lw=1.3, ls=":",
            label=f"NSR = 1 threshold  ({nsr_threshold:.3f} pp)",
        )
        legend_handles.append(ln2)

    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(labels, fontsize=8.8)
    ax.set_xlabel("RMSFE (pp)")
    ax.set_title("Out-of-Sample Forecast Accuracy — 2011Q1 to 2025Q4", fontsize=11)
    ax.set_xlim(0, order.max() * 1.13)
    ax.grid(axis="x", linestyle=(0, (3, 3)), color="#E2E8F0", alpha=0.9, lw=0.6)
    ax.set_axisbelow(True)

    footnotes = []
    if dm_pvals_vs_ar1 is not None:
        footnotes.append(
            f"*  HLN-corrected Diebold–Mariano test rejects equal squared-error"
            f" accuracy versus the expanding AR(1) at the {int(alpha*100)}% level."
            " Over the thesis full sample (2011Q1–2025Q4) no model rejects."
        )
    if nsr_threshold is not None:
        footnotes.append(
            "Dotted line: NSR = 1 threshold (RMSFE = std of German GDP growth)."
            " Models to the left have practical relevance (Lehmann et al. 2020)."
        )

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, _f["legend_anchor_y"]),
            ncol=len(legend_handles),
            frameon=False,
            fontsize=8.5,
            borderaxespad=0.0,
        )

    fig.tight_layout(rect=(0, _f["margin_bottom"], 1, 0.98))
    _save(fig, save)
    return fig


# ---------------------------------------------------------------------------
# 4. RMSFE / MAE by economic regime
# ---------------------------------------------------------------------------

DEFAULT_REGIMES: dict[str, tuple[str, str]] = {
    "Pre-GFC + Eurocrisis (2011-2014)": ("2011Q1", "2014Q4"),
    "Expansion (2015-2019)":            ("2015Q1", "2019Q4"),
    "COVID shock (2020-2021)":          ("2020Q1", "2021Q4"),
    "Post-COVID (2022-2025)":           ("2022Q1", "2025Q4"),
}


def _regime_grouped_bars(
    results_by_model: Mapping[str, pd.DataFrame],
    regimes: Mapping[str, tuple[str, str]],
    metric: str,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Shared workhorse for regime-grouped bar charts (``metric`` in {rmsfe, mae})."""
    rows = []
    for model, df in results_by_model.items():
        for label, (q0, q1) in regimes.items():
            sub = df.loc[(df.index >= q0) & (df.index <= q1), "error"].dropna()
            if not len(sub):
                value = np.nan
            elif metric == "rmsfe":
                value = float(np.sqrt(np.mean(sub.values ** 2)))
            elif metric == "mae":
                value = float(np.mean(np.abs(sub.values)))
            else:
                raise ValueError(f"unknown metric: {metric}")
            rows.append({"model": model, "regime": label, "value": value})
    tbl = pd.DataFrame(rows).pivot(index="regime", columns="model", values="value")
    tbl = tbl.reindex(regimes.keys())

    models = list(results_by_model.keys())
    n_m = len(models)
    width = 0.8 / max(n_m, 1)
    x = np.arange(len(regimes))
    # Taller canvas so grouped bars and the below-plot legend read clearly in print.
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    for i, m in enumerate(models):
        ax.bar(x + (i - (n_m - 1) / 2) * width, tbl[m].values, width,
               color=MODEL_COLORS.get(m, "#1d4ed8"),
               edgecolor="white", linewidth=0.6,
               label=MODEL_LABELS.get(m, m))
    ax.set_xticks(x)
    ax.set_xticklabels(list(regimes.keys()), rotation=0, ha="center")
    # Legend below the plot, no frame -- avoids overlap with the tallest bars
    # (a common convention in published bar charts).
    ax.legend(
        ncol=min(n_m, 6), loc="upper center",
        bbox_to_anchor=(0.5, -0.12), frameon=False, borderaxespad=0.0,
        fontsize=8.5,
    )
    return fig, ax, tbl


def fig_rmsfe_by_regime(
    results_by_model: Mapping[str, pd.DataFrame],
    regimes: Mapping[str, tuple[str, str]] = DEFAULT_REGIMES,
    save: str | Path | None = None,
) -> plt.Figure:
    """Grouped bar chart: RMSFE per model and per economic regime."""
    fig, ax, _ = _regime_grouped_bars(results_by_model, regimes, metric="rmsfe")
    ax.set_ylabel("RMSFE (pp)")
    ax.set_title("Predictive accuracy by economic regime (RMSFE)")
    fig.tight_layout(rect=(0, 0.14, 1, 0.98))
    _save(fig, save)
    return fig


def fig_mae_by_regime(
    results_by_model: Mapping[str, pd.DataFrame],
    regimes: Mapping[str, tuple[str, str]] = DEFAULT_REGIMES,
    save: str | Path | None = None,
) -> plt.Figure:
    """Grouped bar chart: mean absolute error per model and per economic regime."""
    fig, ax, _ = _regime_grouped_bars(results_by_model, regimes, metric="mae")
    ax.set_ylabel("MAE (pp)")
    ax.set_title("Predictive accuracy by economic regime (MAE)")
    fig.tight_layout(rect=(0, 0.14, 1, 0.98))
    _save(fig, save)
    return fig


# ---------------------------------------------------------------------------
# 5. SV calibration plot
# ---------------------------------------------------------------------------

def fig_sv_calibration(
    sv_df: pd.DataFrame,
    credibility_grid: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
    save: str | Path | None = None,
    model_label: str = "SV-scaled Kalman PI",
) -> plt.Figure:
    """Nominal vs empirical coverage of SV prediction intervals, plus CRPS bar.

    Two-panel figure:
      Left  — calibration plot: nominal coverage (x) vs empirical coverage (y).
               A perfectly calibrated model lies on the 45° diagonal.
      Right — CRPS by credibility level: mean CRPS of the Gaussian predictive
               at each nominal coverage (computed from the implied σ at that
               level). Lower CRPS is better; rewards both sharpness and
               calibration (Gneiting & Raftery 2007).

    Parameters
    ----------
    sv_df           : DataFrame from run_actpn_nowcast_loop_sv. Must contain
                      'nowcast', 'actual', 'sigma_em', 'rel_vol' columns.
    credibility_grid: sequence of nominal coverage levels to evaluate.
    model_label     : legend label for the model curve.
    """
    from scipy import stats as _st

    valid = sv_df.dropna(subset=["nowcast", "actual", "sigma_em", "rel_vol"])
    if valid.empty:
        raise ValueError("sv_df has no rows with sigma_em and rel_vol.")

    sigma_pred = valid["sigma_em"].values * np.sqrt(
        np.clip(valid["rel_vol"].values, 1e-8, None)
    )
    mu  = valid["nowcast"].values
    y   = valid["actual"].values
    err = y - mu

    empirical: list[float] = []
    crps_vals: list[float] = []

    for c in credibility_grid:
        z = _st.norm.ppf((1.0 + c) / 2.0)
        inside = float((np.abs(err) <= z * sigma_pred).mean())
        empirical.append(inside)

        # CRPS for Gaussian N(mu, sigma_pred^2) — Gneiting & Raftery (2007)
        z_std = err / np.clip(sigma_pred, 1e-10, None)
        crps_i = float(np.mean(
            sigma_pred * (
                z_std * (2.0 * _st.norm.cdf(z_std) - 1.0)
                + 2.0 * _st.norm.pdf(z_std)
                - 1.0 / np.sqrt(np.pi)
            )
        ))
        crps_vals.append(crps_i)

    fig, axes = plt.subplots(
        1, 2, figsize=(11.0, 5.0), layout="constrained",
    )

    # Left panel — calibration
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="#94A3B8", ls="--", lw=1, label="Perfect calibration")
    ax.plot(credibility_grid, empirical, marker="o",
            color=MODEL_COLORS["en_only"], lw=1.7, label=model_label)
    for c, e in zip(credibility_grid, empirical):
        ax.annotate(f"{e:.2f}", (c, e),
                    textcoords="offset points", xytext=(4, 4), fontsize=8.5)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Interval calibration")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower right", frameon=False)

    # Right panel — CRPS by nominal level
    ax2 = axes[1]
    bar_colors = [MODEL_COLORS["en_only"]] * len(credibility_grid)
    bars = ax2.bar(
        np.arange(len(credibility_grid)), crps_vals,
        color=bar_colors, edgecolor="white", linewidth=0.5,
        alpha=0.88, width=0.6,
    )
    x_labels = [f"{int(c * 100)}%" for c in credibility_grid]
    ax2.set_xticks(np.arange(len(credibility_grid)))
    ax2.set_xticklabels(x_labels)
    ax2.set_xlabel("Nominal coverage (defines σ_pred)")
    ax2.set_ylabel("Mean CRPS (pp)")
    ax2.set_title("CRPS of Gaussian predictive density")
    for bar, val in zip(bars, crps_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val + ax2.get_ylim()[1] * 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8.2)

    fig.suptitle(
        "SV predictive distribution — calibration and scoring",
        fontsize=12, fontweight="600",
    )
    _save(fig, save)
    return fig


# ---------------------------------------------------------------------------
# 6. SV volatility panel
# ---------------------------------------------------------------------------

def fig_sv_volatility(
    sv_df: pd.DataFrame,
    save: str | Path | None = None,
    credibility: float = 0.9,
) -> plt.Figure:
    """Two-panel diagnostic: posterior rel_vol + PI width vs |error|."""
    cov_label = int(round(credibility * 100))
    lo_col = f"ci_lower_{cov_label}"
    hi_col = f"ci_upper_{cov_label}"

    df = sv_df.copy()
    if "month_in_quarter" in df.columns:
        df = df.loc[df["month_in_quarter"] == 3]
    df = df.dropna(subset=["nowcast", "actual"])
    if "quarter" in df.columns:
        x = _q_to_ts(pd.Index(df["quarter"].astype(str)))
    else:
        x = _q_to_ts(df.index)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.8), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})

    ax = axes[0]
    ax.plot(x, df["rel_vol"], color=MODEL_COLORS["en_only"], lw=1.6,
            label="Posterior mean rel_vol")
    ax.axhline(1, color="#111827", lw=0.8, ls=":")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylabel("Relative volatility")
    ax.set_title("SV posterior multiplier r_t and prediction-interval response")
    ax.legend(loc="upper left")
    add_recession_bands(ax, annotate=True)

    ax = axes[1]
    width = df[hi_col] - df[lo_col]
    ax.plot(x, width, color=MODEL_COLORS["en_only"], lw=1.4, label=f"PI width ({cov_label}%)")
    ax.plot(x, df["error"].abs(), color=MODEL_COLORS["en_only"], lw=1.2, ls="--",
            label="|Nowcast error|")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylabel("pp")
    ax.set_xlabel("Quarter")
    ax.legend(loc="upper left")
    add_recession_bands(ax)
    _apply_year_axis(ax, base=2)

    fig.tight_layout()
    _save(fig, save)
    return fig


# ---------------------------------------------------------------------------
# 6b. Fan chart: nowcast + time-varying 90 % prediction interval (SV)
# ---------------------------------------------------------------------------

def fig_nowcast_fan(
    sv_df: pd.DataFrame,
    credibility: float = 0.90,
    save: str | Path | None = None,
) -> plt.Figure:
    """Fan chart: integrated DFM-SV nowcast with a time-varying interval.

    The shaded band is the ``credibility``-level prediction interval from
    the integrated SV specification (``ci_lower_90`` / ``ci_upper_90``).
    Width tracks the estimated volatility state: it widens in 2020 and
    narrows only partly after 2022. Full-sample 90% coverage is 88.3%
    (53/60); six of seven misses fall in the eight pandemic quarters.

    Parameters
    ----------
    sv_df:
        DataFrame produced by the integrated SV nowcasting loop. Expected
        columns: ``nowcast``, ``actual``, ``ci_lower_<cov>``,
        ``ci_upper_<cov>``, ``rel_vol``. Index is quarterly.
    credibility:
        Nominal coverage of the shaded interval (default 0.90).
    """
    cov_label = int(round(credibility * 100))
    lo_col = f"ci_lower_{cov_label}"
    hi_col = f"ci_upper_{cov_label}"

    df = sv_df.copy()
    if "month_in_quarter" in df.columns:
        df = df.loc[df["month_in_quarter"] == 3]
    df = df.dropna(subset=["nowcast", "actual"])
    if "quarter" in df.columns:
        x = _q_to_ts(pd.Index(df["quarter"].astype(str)))
    else:
        x = _q_to_ts(df.index)

    fig, ax = plt.subplots(figsize=(13, 5.0))

    # 90 % shaded band
    if lo_col in df.columns and hi_col in df.columns:
        ax.fill_between(
            x,
            df[lo_col].values,
            df[hi_col].values,
            color=MODEL_COLORS["en_only"],
            alpha=0.13,
            label=f"{cov_label}% prediction interval (SV)",
            zorder=1,
        )
        # Dashed interval boundary lines for visual clarity
        ax.plot(x, df[lo_col].values, color=MODEL_COLORS["en_only"],
                lw=0.7, ls="--", alpha=0.45, zorder=2)
        ax.plot(x, df[hi_col].values, color=MODEL_COLORS["en_only"],
                lw=0.7, ls="--", alpha=0.45, zorder=2)

    # Nowcast line
    ax.plot(x, df["nowcast"].values,
            color=MODEL_COLORS["en_only"], lw=2.0,
            label="Core DFM nowcast (SV-scaled)", zorder=4)

    # Actual GDP
    ax.plot(x, df["actual"].values,
            color=MODEL_COLORS["actual"], lw=1.4, ls="-",
            marker="o", markersize=3.5,
            label="Actual GDP growth", zorder=5)

    ax.axhline(0, color="#CBD5E1", lw=0.8, zorder=0)
    ax.set_xlim(x[0], x[-1])
    add_recession_bands(ax, annotate=True)

    ax.set_ylabel("Quarter-on-Quarter Log-Growth (pp)")
    ax.set_xlabel("Quarter")
    ax.set_title(
        f"Integrated DFM-SV nowcast of German GDP growth with {cov_label}% "
        "prediction interval",
        fontsize=12,
    )
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8.5)
    _apply_year_axis(ax, base=2)

    fig.tight_layout()
    _save(fig, save)
    return fig


# ---------------------------------------------------------------------------
# 7. Indicator-set composition heatmap
# ---------------------------------------------------------------------------

def fig_composition_heatmap(
    matrices: Mapping[str, pd.DataFrame],
    meta: pd.DataFrame,
    top_n_categories: int = 10,
    save: str | Path | None = None,
) -> plt.Figure:
    """Heatmap: rows = top categories, columns = input sets, cell = share (%)."""
    cats = meta["category"].fillna("Unknown")
    rows = []
    for key, m in matrices.items():
        counts = m.astype(int).sum(axis=0)
        joined = counts.rename("n").to_frame().join(cats.rename("category"))
        share = joined.groupby("category")["n"].sum()
        share = share / share.sum() * 100
        rows.append(share.rename(key))
    tbl = pd.concat(rows, axis=1).fillna(0)
    top = tbl.sum(axis=1).sort_values(ascending=False).head(top_n_categories).index
    tbl = tbl.loc[top]

    fig, ax = plt.subplots(figsize=(7.4, 0.45 * len(top) + 1.7))
    im = ax.imshow(tbl.values, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(tbl.shape[1]))
    ax.set_xticklabels([MODEL_LABELS.get(c, c) for c in tbl.columns], rotation=0, ha="center")
    ax.set_yticks(range(tbl.shape[0]))
    ax.set_yticklabels(tbl.index)
    for i in range(tbl.shape[0]):
        for j in range(tbl.shape[1]):
            ax.text(j, i, f"{tbl.values[i, j]:.0f}%", ha="center", va="center", fontsize=8.5,
                    color="white" if tbl.values[i, j] > tbl.values.max() * 0.55 else "#111827")
    ax.set_title("Economic composition of each input set (% of total selections)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="% of selections")
    fig.tight_layout()
    _save(fig, save)
    return fig


# ---------------------------------------------------------------------------
# 8. Contribution decomposition at a single origin
# ---------------------------------------------------------------------------

def fig_contribution_decomp(
    result: Any,
    selected_ids: Sequence[str],
    meta: pd.DataFrame,
    target_quarter: pd.Period,
    save: str | Path | None = None,
) -> plt.Figure:
    """Bar chart: category-level contribution to the nowcast at one origin.

    Uses :func:`compute_category_contributions` so that bars are expressed
    in GDP nowcast percentage points (see that function's docstring for
    details on the proportional decomposition).
    """
    grouped = compute_category_contributions(result, selected_ids, meta, target_quarter)
    grouped = grouped.sort_values()

    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(grouped) + 1.6))
    colors = ["#9D174D" if v < 0 else MODEL_COLORS["en_only"] for v in grouped.values]
    ax.barh(grouped.index, grouped.values, color=colors,
            edgecolor="white", linewidth=0.7, alpha=0.95)
    ax.axvline(0, color="#111827", lw=0.8)
    ax.set_xlabel("Contribution to GDP nowcast (pp)")
    ax.set_title(f"Indicator-block contributions at {target_quarter}")
    fig.tight_layout()
    _save(fig, save)
    return fig


# ---------------------------------------------------------------------------
# 9. ifoCAST-style stacked category-contribution bars (recent quarters)
# ---------------------------------------------------------------------------

def fig_category_contrib_stacked(
    contributions_by_quarter: Mapping[str, pd.Series],
    actuals: pd.Series | None = None,
    nowcasts: pd.Series | None = None,
    title: str = "Category Contributions to the Core DFM Nowcast",
    save: str | Path | None = None,
) -> plt.Figure:
    """ifoCAST-style stacked bar chart of category-level contributions.

    Each column is a quarter; bars stack signed per-category contributions
    in **GDP nowcast pp**. The actual GDP value is overlaid as a black
    diamond and the point nowcast as a coloured circle.

    Returns the matplotlib Figure. For an interactive version use
    :func:`fig_category_contrib_interactive`.
    """
    quarters = list(contributions_by_quarter.keys())
    all_cats: list[str] = []
    for s in contributions_by_quarter.values():
        for c in s.index:
            if c not in all_cats:
                all_cats.append(c)

    # Sort categories so the most important ones are at the base of each bar.
    total_abs = {
        c: sum(abs(float(contributions_by_quarter[q].get(c, 0.0)))
               for q in quarters)
        for c in all_cats
    }
    all_cats = sorted(all_cats, key=lambda c: total_abs[c], reverse=True)

    cat_color = {
        c: CATEGORY_COLORS.get(c, _CATEGORY_PALETTE[i % len(_CATEGORY_PALETTE)])
        for i, c in enumerate(all_cats)
    }

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(quarters) + 3), 6.0))
    x = np.arange(len(quarters))
    bar_width = 0.72

    pos_stack = np.zeros(len(quarters))
    neg_stack = np.zeros(len(quarters))
    plotted_cats: list[str] = []
    for cat in all_cats:
        vals = np.array([
            float(contributions_by_quarter[q].get(cat, 0.0)) for q in quarters
        ])
        if np.all(vals == 0):
            continue
        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)
        ax.bar(x, pos, bar_width, bottom=pos_stack, color=cat_color[cat],
               edgecolor="white", linewidth=0.5, label=cat, zorder=2)
        ax.bar(x, neg, bar_width, bottom=neg_stack, color=cat_color[cat],
               edgecolor="white", linewidth=0.5, zorder=2)
        pos_stack += pos
        neg_stack += neg
        plotted_cats.append(cat)

    # Net labels above/below each column.
    nets = pos_stack + neg_stack
    for xi, top, bot, net in zip(x, pos_stack, neg_stack, nets):
        if abs(net) < 0.02:
            continue
        ypos = top + 0.05 if net > 0 else bot - 0.05
        va = "bottom" if net > 0 else "top"
        ax.text(xi, ypos, f"{net:+.2f}", ha="center", va=va,
                fontsize=8, color="#1F2937", fontweight="600")

    if nowcasts is not None:
        nc = np.array([nowcasts.get(q, np.nan) for q in quarters], dtype=float)
        ax.plot(x, nc, "o", color=MODEL_COLORS["en_only"], markersize=8,
                markeredgecolor="white", markeredgewidth=1.2,
                label="Core DFM nowcast", zorder=5)
    if actuals is not None:
        ac = np.array([actuals.get(q, np.nan) for q in quarters], dtype=float)
        ax.plot(x, ac, "D", color=MODEL_COLORS["actual"], markersize=7.5,
                markeredgecolor="white", markeredgewidth=1.2,
                label="Actual GDP", zorder=6)

    ax.axhline(0, color=MODEL_COLORS["actual"], lw=0.9, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(quarters, rotation=0, ha="center")
    ax.set_ylabel("Contribution to GDP Nowcast (pp)")
    ax.set_xlabel("Quarter")
    ax.set_title(title, fontsize=12)

    ncol = min(7, max(1, len(plotted_cats) + 2))
    ax.legend(
        ncol=ncol, fontsize=8.5, loc="upper center",
        bbox_to_anchor=(0.5, -0.12), borderaxespad=0.0, frameon=False,
    )

    fig.tight_layout()
    _save(fig, save)
    return fig


def fig_category_contrib_interactive(
    contributions_by_quarter: Mapping[str, pd.Series],
    actuals: pd.Series | None = None,
    nowcasts: pd.Series | None = None,
    title: str = "Category Contributions to the Core DFM Nowcast",
    save_html: str | Path | None = None,
    xaxis_title: str = "Quarter",
    hover_meta: Mapping[str, Mapping[str, str]] | None = None,
) -> Any:
    """Interactive Plotly chart of category-level GDP nowcast contributions.

    Two views are available via toggle buttons:

    * **Absolute (pp)** — signed stacked bars in percentage-point units.
      One category may dominate if its indicators carry the largest
      z-score signal (e.g. ifo survey indicators in Germany).
    * **Relative (%)** — each bar is re-scaled to 100 % so the
      *proportional* contribution of every category is visible even when
      one dominates in absolute terms.

    Hovering over any bar shows the category, x-axis label, and value.
    When ``hover_meta`` is provided (keys = x labels), hovers also show
    target quarter and month-in-quarter (M1/M2/M3).

    Parameters
    ----------
    contributions_by_quarter:
        Mapping from x-axis label (quarter or monthly origin) to per-category
        contribution Series in pp.
    save_html:
        Optional path to write a stand-alone HTML file.
    xaxis_title:
        X-axis label (e.g. ``"Forecast origin (month)"`` for monthly panels).
    hover_meta:
        Optional dict ``{x_label: {"quarter": "2024Q1", "miq": "M2"}}``.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "plotly is required for the interactive chart. "
            "Install it with: pip install plotly"
        )

    quarters = list(contributions_by_quarter.keys())
    all_cats: list[str] = []
    for s in contributions_by_quarter.values():
        for c in s.index:
            if c not in all_cats:
                all_cats.append(c)

    total_abs = {
        c: sum(abs(float(contributions_by_quarter[q].get(c, 0.0)))
               for q in quarters)
        for c in all_cats
    }
    all_cats = sorted(all_cats, key=lambda c: total_abs[c], reverse=False)

    # Absolute values per category (pp)
    abs_vals: dict[str, list[float]] = {}
    for cat in all_cats:
        vals = [float(contributions_by_quarter[q].get(cat, 0.0)) for q in quarters]
        if not all(v == 0 for v in vals):
            abs_vals[cat] = vals

    active_cats = list(abs_vals.keys())

    # Normalised values: each quarter scaled to ±100 %
    # We split positive and negative stacks separately so that the relative
    # chart faithfully represents the signed decomposition.
    pos_totals = np.array([
        sum(max(0.0, abs_vals[c][qi]) for c in active_cats)
        for qi in range(len(quarters))
    ])
    neg_totals = np.array([
        sum(min(0.0, abs_vals[c][qi]) for c in active_cats)
        for qi in range(len(quarters))
    ])

    norm_vals: dict[str, list[float]] = {}
    for cat in active_cats:
        nv = []
        for qi, v in enumerate(abs_vals[cat]):
            if v >= 0:
                denom = pos_totals[qi] if pos_totals[qi] > 1e-12 else 1.0
                nv.append(v / denom * 100.0)
            else:
                denom = abs(neg_totals[qi]) if abs(neg_totals[qi]) > 1e-12 else 1.0
                nv.append(v / denom * 100.0)
        norm_vals[cat] = nv

    n_bar_traces = len(active_cats)

    # Build traces: bar traces for absolute view, then bar traces for relative
    # view (initially invisible), then scatter traces for nowcast / actuals.
    def _hover_pp(cat: str) -> str:
        """Build absolute-contribution hover text for one category."""
        if hover_meta:
            return (
                f"<b>{cat}</b><br>Origin: %{{x}}<br>"
                "Target: %{customdata[0]}<br>MIQ: %{customdata[1]}<br>"
                "Contribution: %{y:.3f} pp<extra></extra>"
            )
        return (
            f"<b>{cat}</b><br>{xaxis_title}: %{{x}}<br>"
            "Contribution: %{y:.3f} pp<extra></extra>"
        )

    def _hover_pct(cat: str) -> str:
        """Build relative-share hover text for one category."""
        if hover_meta:
            return (
                f"<b>{cat}</b><br>Origin: %{{x}}<br>"
                "Target: %{customdata[0]}<br>MIQ: %{customdata[1]}<br>"
                "Share: %{y:.1f}%<extra></extra>"
            )
        return (
            f"<b>{cat}</b><br>{xaxis_title}: %{{x}}<br>"
            "Share: %{y:.1f}%<extra></extra>"
        )

    customdata = None
    if hover_meta:
        customdata = [
            [hover_meta.get(q, {}).get("quarter", ""),
             hover_meta.get(q, {}).get("miq", "")]
            for q in quarters
        ]

    traces: list = []
    for cat in active_cats:
        color = CATEGORY_COLORS.get(cat, "#9CA3AF")
        traces.append(go.Bar(
            name=cat, x=quarters, y=abs_vals[cat],
            marker_color=color, marker_line_color="white",
            marker_line_width=0.5, visible=True,
            legendgroup=cat,
            customdata=customdata,
            hovertemplate=_hover_pp(cat),
        ))
    for cat in active_cats:
        color = CATEGORY_COLORS.get(cat, "#9CA3AF")
        traces.append(go.Bar(
            name=cat, x=quarters, y=norm_vals[cat],
            marker_color=color, marker_line_color="white",
            marker_line_width=0.5, visible=False,
            legendgroup=cat, showlegend=False,
            customdata=customdata,
            hovertemplate=_hover_pct(cat),
        ))

    scatter_traces: list = []
    if nowcasts is not None:
        nc = [float(nowcasts.get(q, float("nan"))) for q in quarters]
        scatter_traces.append(go.Scatter(
            name="Core DFM nowcast", x=quarters, y=nc, mode="markers",
            marker=dict(symbol="circle", size=10, color=MODEL_COLORS["en_only"],
                        line=dict(color="white", width=1.5)),
            hovertemplate="Nowcast: %{y:.3f} pp<extra></extra>",
        ))
    if actuals is not None:
        ac = [float(actuals.get(q, float("nan"))) for q in quarters]
        scatter_traces.append(go.Scatter(
            name="Actual GDP", x=quarters, y=ac, mode="markers",
            marker=dict(symbol="diamond", size=10, color=MODEL_COLORS["actual"],
                        line=dict(color="white", width=1.5)),
            hovertemplate="Actual GDP: %{y:.3f} pp<extra></extra>",
        ))

    all_traces = traces + scatter_traces

    # Visibility masks for buttons
    abs_vis  = ([True]  * n_bar_traces + [False] * n_bar_traces
                + [True] * len(scatter_traces))
    norm_vis = ([False] * n_bar_traces + [True]  * n_bar_traces
                + [False] * len(scatter_traces))

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        barmode="relative",
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(
            title=xaxis_title,
            tickfont=dict(size=10 if len(quarters) > 20 else 11),
            tickangle=-45 if len(quarters) > 20 else 0,
        ),
        yaxis=dict(
            title="Contribution to GDP Nowcast (pp)",
            tickfont=dict(size=11),
            zeroline=True, zerolinecolor="#111827", zerolinewidth=1,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.32,
            xanchor="center", x=0.5, font=dict(size=10),
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif"),
        margin=dict(l=60, r=20, t=80, b=140),
        height=560 if len(quarters) <= 24 else 620,
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0.5, xanchor="center", y=1.08, yanchor="top",
            buttons=[
                dict(
                    label="Absolute (pp)",
                    method="update",
                    args=[
                        {"visible": abs_vis},
                        {"yaxis.title.text": "Contribution to GDP Nowcast (pp)"},
                    ],
                ),
                dict(
                    label="Relative (%)",
                    method="update",
                    args=[
                        {"visible": norm_vis},
                        {"yaxis.title.text": "Share of total contribution (%)"},
                    ],
                ),
            ],
            pad={"r": 10, "t": 10},
            showactive=True,
            bgcolor="#F8FAFC",
            bordercolor="#CBD5E1",
            font=dict(size=11),
        )],
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E2E8F0", gridwidth=0.6, showgrid=True)

    if save_html is not None:
        Path(save_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_html))

    return fig


def build_monthly_origins(
    quarterly_origins: Iterable[pd.Period | str],
    m_start: str = "2017-01",
    m_end: str = "2025-12",
) -> list[tuple[pd.Period, pd.Period, int]]:
    """Monthly forecast origins (M1/M2/M3) within a month range.

    Returns a list of ``(origin_period, target_quarter, month_in_quarter)``.
    """
    m_lo = pd.Period(m_start, freq="M")
    m_hi = pd.Period(m_end, freq="M")
    out: list[tuple[pd.Period, pd.Period, int]] = []
    for q in quarterly_origins:
        q = pd.Period(q, freq="Q")
        q_m1 = q.asfreq("M", how="start")
        for m_in_q in (1, 2, 3):
            origin_p = q_m1 + (m_in_q - 1)
            if m_lo <= origin_p <= m_hi:
                out.append((origin_p, q, m_in_q))
    return out


def needs_run_contrib_cache(
    cache_path: str | Path,
    force_rerun: bool = False,
    series_cache_path: str | Path | None = None,
) -> bool:
    """Return True if the category-contribution cache should be rebuilt."""
    p = Path(cache_path)
    if force_rerun or not p.exists():
        return True
    if series_cache_path is not None and not Path(series_cache_path).exists():
        return True
    try:
        df = pd.read_parquet(p)
        required = {"monthly_origin", "quarter", "month_in_quarter", "category", "contrib_pp"}
        return not required.issubset(df.columns) or len(df) < 50
    except Exception:
        return True


def run_category_contrib_panel(
    selection_matrix: pd.DataFrame,
    X_monthly: pd.DataFrame,
    y_quarterly: pd.Series,
    meta: pd.DataFrame,
    quarterly_origins: Iterable[pd.Period | str],
    pub_lag_map: pd.Series | None = None,
    k_factors: int = 2,
    factor_order: int = 2,
    idiosyncratic_ar1: bool = True,
    maxiter: int = 200,
    m_start: str = "2017-01",
    m_end: str = "2025-12",
    cache_path: str | Path | None = None,
    series_cache_path: str | Path | None = None,
    force_rerun: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fit Core DFM at each monthly origin and cache category contributions.

    Returns a long-format DataFrame with columns
    ``monthly_origin``, ``quarter``, ``month_in_quarter``, ``nowcast``,
    ``actual``, ``category``, ``contrib_pp``.
    """
    from ..models.dfm.nowcast_utils import build_dfm_endog, extract_nowcast, fit_dfm

    cache_path = Path(cache_path) if cache_path is not None else None
    series_cache_path = Path(series_cache_path) if series_cache_path is not None else None
    if cache_path is not None and not needs_run_contrib_cache(
        cache_path, force_rerun, series_cache_path=series_cache_path,
    ):
        if verbose:
            print(f"Loaded category contributions from {cache_path.name}")
        return pd.read_parquet(cache_path)

    origins = build_monthly_origins(quarterly_origins, m_start=m_start, m_end=m_end)
    records: list[dict] = []
    series_records: list[dict] = []

    for origin_p, q, m_in_q in origins:
        origin_key = str(origin_p)
        if origin_key not in selection_matrix.index:
            if verbose:
                print(f"  {origin_key}: not in selection matrix — skipped.")
            continue
        sel_cols = selection_matrix.columns[
            selection_matrix.loc[origin_key].astype(bool)
        ].tolist()
        if not sel_cols:
            if verbose:
                print(f"  {origin_key}: no indicators — skipped.")
            continue
        if verbose:
            print(f"  {q} M{m_in_q} ({origin_key}): N={len(sel_cols)} ...", end=" ", flush=True)
        try:
            X_sel = X_monthly[sel_cols]
            endog, k_endog_M = build_dfm_endog(
                X_sel, y_quarterly, origin_p, pub_lag_map=pub_lag_map,
            )
            result = fit_dfm(
                endog,
                k_endog_M=k_endog_M,
                k_factors=k_factors,
                factor_order=factor_order,
                idiosyncratic_ar1=idiosyncratic_ar1,
                maxiter=maxiter,
            )
            nc = extract_nowcast(result, origin_p)
            actual = float(y_quarterly.get(q, np.nan))
            frame = _contrib_frame(
                result,
                selected_ids=sel_cols,
                meta=meta,
                target_quarter=q,
            )
            grouped = frame.groupby("category")["contrib_pp"].sum()
            for cat, val in grouped.items():
                records.append({
                    "monthly_origin": origin_key,
                    "quarter": str(q),
                    "month_in_quarter": m_in_q,
                    "nowcast": nc,
                    "actual": actual,
                    "category": cat,
                    "contrib_pp": float(val),
                })
            for sid, row in frame.iterrows():
                series_records.append({
                    "monthly_origin": origin_key,
                    "quarter": str(q),
                    "month_in_quarter": m_in_q,
                    "series": sid,
                    "category": row["category"],
                    "contrib_pp": float(row["contrib_pp"]),
                })
            if verbose:
                print(f"nowcast={nc:.3f}")
        except Exception as exc:
            if verbose:
                print(f"ERROR: {exc}")

    df = pd.DataFrame(records)
    if cache_path is not None and not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        if verbose:
            print(f"Saved {len(df)} rows -> {cache_path}")
    if series_cache_path is not None and series_records:
        series_cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(series_records).to_parquet(series_cache_path, index=False)
        if verbose:
            print(f"Saved {len(series_records)} rows -> {series_cache_path}")
    return df


def contrib_cache_to_plot_dicts(
    cache_df: pd.DataFrame,
    m_start: str,
    m_end: str,
) -> tuple[
    dict[str, pd.Series],
    pd.Series,
    pd.Series,
    dict[str, dict[str, str]],
]:
    """Filter cache to a month window and build plot inputs."""
    m_lo, m_hi = pd.Period(m_start, freq="M"), pd.Period(m_end, freq="M")
    sub = cache_df.copy()
    sub["_origin_p"] = sub["monthly_origin"].map(lambda s: pd.Period(s, freq="M"))
    sub = sub.loc[(sub["_origin_p"] >= m_lo) & (sub["_origin_p"] <= m_hi)]
    sub = sub.sort_values(["_origin_p", "category"])

    contribs: dict[str, pd.Series] = {}
    nowcasts: dict[str, float] = {}
    actuals: dict[str, float] = {}
    hover_meta: dict[str, dict[str, str]] = {}

    for origin_key, grp in sub.groupby("monthly_origin", sort=True):
        x_label = origin_key
        contribs[x_label] = grp.set_index("category")["contrib_pp"]
        row0 = grp.iloc[0]
        nowcasts[x_label] = float(row0["nowcast"])
        actuals[x_label] = float(row0["actual"])
        hover_meta[x_label] = {
            "quarter": str(row0["quarter"]),
            "miq": f"M{int(row0['month_in_quarter'])}",
        }

    return (
        contribs,
        pd.Series(nowcasts),
        pd.Series(actuals),
        hover_meta,
    )


def fig_category_contrib_period_panels(
    cache_df: pd.DataFrame,
    periods: Sequence[tuple[str, str, str]],
    fig_dir: str | Path,
    base_title: str = "Core DFM: Category Contributions by Forecast Origin",
) -> list:
    """Write one interactive HTML per period; return list of Plotly figures."""
    fig_dir = Path(fig_dir)
    figures = []
    for m_start, m_end, slug in periods:
        contribs, nowcasts, actuals, hover_meta = contrib_cache_to_plot_dicts(
            cache_df, m_start, m_end,
        )
        if not contribs:
            continue
        title = f"{base_title} ({m_start} – {m_end})"
        fig = fig_category_contrib_interactive(
            contributions_by_quarter=contribs,
            actuals=actuals,
            nowcasts=nowcasts,
            title=title,
            save_html=fig_dir / f"08d_category_contrib_{slug}_interactive.html",
            xaxis_title="Forecast origin (month)",
            hover_meta=hover_meta,
        )
        figures.append(fig)
    return figures


#: Cap on |category contribution| as a multiple of |nowcast g|. The
#: predict()-based attribution below is a proportional split
#: ``contrib_i = g * z_i / sum_i(z_i)`` of the nowcast across standardized
#: indicator predictions ``z_i``. This ratio estimator reconciles exactly
#: (``sum_i contrib_i == g``) and is well-behaved when the indicators broadly
#: agree in sign, but it is a classic *proportional/return-attribution*
#: instability (cf. Karnosky & Singer, 1994, on the analogous problem in
#: portfolio performance attribution): when selected indicators pull in
#: strongly offsetting directions -- plausible in real economies during
#: sharp V-shaped swings, when some series have already rebounded while
#: others remain depressed -- ``sum_i(z_i)`` collapses toward zero even
#: though individual ``z_i`` stay large, and every bar is inflated by the
#: same explosive factor while still summing to the (small) nowcast.
#: Empirically this ratio is well below 5x for the vast majority of monthly
#: origins across the headline DFM specifications (2017-2025); we therefore
#: floor the denominator so no single category can carry more than
#: ``_CONTRIB_MAX_LEVERAGE`` times the nowcast, and book the reconciliation
#: gap this creates -- which is exactly the part of the swing that cannot be
#: attributed to any one category because the underlying indicators are
#: cancelling each other out -- as an explicit ``"Offset"`` category rather
#: than silently inflating a real one.
_CONTRIB_MAX_LEVERAGE = 5.0
_CONTRIB_OFFSET_CATEGORY = "Offset"
_CONTRIB_OFFSET_ID = "__offset__"


def _contrib_frame(
    result: Any,
    selected_ids: Sequence[str],
    meta: pd.DataFrame,
    target_quarter: pd.Period,
) -> pd.DataFrame:
    """Per-series contributions (pp) with category labels.

    See :data:`_CONTRIB_MAX_LEVERAGE` for the leverage cap that keeps this
    proportional attribution from exploding when indicators offset each
    other; any reconciliation gap is returned as a synthetic
    ``_CONTRIB_OFFSET_ID`` row with category ``_CONTRIB_OFFSET_CATEGORY``.
    """
    contrib: np.ndarray
    used_ids: list[str]
    offset = 0.0
    try:
        pred = result.predict()
        target_ts = target_quarter.asfreq("M", how="end").to_timestamp()
        if target_ts not in pred.index:
            target_ts = pred.index[pred.index.get_indexer([target_ts], method="nearest")[0]]
        row = pred.loc[target_ts]

        gdp_col = pred.columns[-1]
        g = float(row[gdp_col])

        used_ids = [c for c in selected_ids if c in pred.columns and c != gdp_col]
        if not used_ids:
            raise ValueError("none of selected_ids found in predict() columns")
        s = row[used_ids].astype(float).values

        try:
            endog_names = list(result.model.endog_names)
            endog_std   = np.array(result.model._endog_std)
            std_map     = dict(zip(endog_names, endog_std))
            sigmas      = np.array([std_map.get(uid, 1.0) for uid in used_ids])
            sigmas      = np.where(sigmas < 1e-12, 1.0, sigmas)
            z = s / sigmas
        except Exception:
            z = s

        gross = np.nansum(np.abs(z))
        net = np.nansum(z)
        if not np.isfinite(gross) or gross < 1e-10:
            n = len(used_ids)
            w = np.full(n, 1.0 / n) if n else np.zeros(0)
            contrib = g * w
        else:
            floor = gross / _CONTRIB_MAX_LEVERAGE
            if abs(net) >= floor:
                effective_denom = net
            else:
                effective_denom = floor if net >= 0 else -floor
            contrib = g * z / effective_denom
            offset = g - float(np.nansum(contrib))
    except Exception:
        used_ids = list(selected_ids)
        contrib = np.zeros(len(used_ids))
        offset = 0.0

    out = pd.Series(contrib, index=used_ids, name="contrib_pp").to_frame()
    out = out.join(meta[["category"]], how="left")
    out["category"] = out["category"].fillna("Unknown")
    if abs(offset) > 1e-9:
        offset_row = pd.DataFrame(
            {"contrib_pp": [offset], "category": [_CONTRIB_OFFSET_CATEGORY]},
            index=[_CONTRIB_OFFSET_ID],
        )
        out = pd.concat([out, offset_row])
    return out


def compute_category_contributions(
    result: Any,
    selected_ids: Sequence[str],
    meta: pd.DataFrame,
    target_quarter: pd.Period,
    origin: pd.Period | str | None = None,
) -> pd.Series:
    """Category-level contributions to the GDP nowcast at ``target_quarter``.

    The function returns contributions in the same units as the GDP
    nowcast itself (percentage points), so that summing the returned
    Series across categories approximately recovers the point nowcast.
    This is intended for the ifoCAST-style waterfall in
    :func:`fig_category_contrib_stacked`.

    Method
    ------
    The fitted ``DynamicFactorMQ`` model's ``predict()`` method returns,
    in original units, a (T x k_endog) panel of one-step-ahead/smoothed
    predicted values for every endogenous variable -- the monthly
    indicators *and* GDP. At the target quarter-end month :math:`t^\\star`,
    let :math:`\\hat x_{i,t^\\star}` be the predicted value of indicator
    :math:`i` and :math:`g = \\hat y_{t^\\star}` the predicted GDP value
    (the nowcast). We attribute :math:`g` across the *selected* monthly
    indicators using the signed shares
    :math:`w_i = \\hat x_{i,t^\\star} / \\sum_j \\hat x_{j,t^\\star}`
    (with a small-epsilon guard, falling back to equal weights when the
    denominator is degenerate). Per-indicator contributions are then
    :math:`c_i = g \\cdot w_i` in GDP pp, and we aggregate them by
    ``meta['category']``.

    This is deliberately based on ``predict()`` rather than on
    ``result.model.design`` because for ``DynamicFactorMQ`` the GDP row
    of the design matrix is built via Mariano-Murasawa cumulator
    restrictions on an augmented state vector, so direct
    ``Lambda . f_t`` indexing yields values in the wrong units (the
    reason the bars were invisible in the previous version of this
    chart).

    The result is *not* a Banbura-Modugno news decomposition (which
    would require successive-vintage Kalman updates), but it preserves
    the accounting identity ``sum(c_i) == g`` and is the standard
    starting-point approximation used in ifoCAST-style charts; cf.
    Lehmann, Reif & Wollmershauser (2020).

    Category labels come from the configured enriched metadata (a
    project-curated mapping). The label ``Turnover`` corresponds to the
    standard English term for Destatis "Umsatz / Production Sales" --
    every series under that category is of the form
    "Germany, Production Sales, Turnover, ..." -- so no renaming is
    needed.
    """
    return _contrib_frame(
        result, selected_ids, meta, target_quarter,
    ).groupby("category")["contrib_pp"].sum()


#: Plausibility bound for a single nowcast (pp QoQ log-growth), mirroring the
#: guard in ``run_blockbalanced_benchmark.py``.
_GUARD_NOWCAST_CAP = 20.0

#: Divergence-guard fallback ladder (progressively simpler state-space
#: structure), identical to ``run_blockbalanced_benchmark.guarded_nowcast``.
_GUARD_LADDER = [
    dict(k_factors=2, idiosyncratic_ar1=True),
    dict(k_factors=2, idiosyncratic_ar1=False),
    dict(k_factors=1, idiosyncratic_ar1=True),
    dict(k_factors=1, idiosyncratic_ar1=False),
]


def run_category_contrib_panel_guarded(
    selection_matrix: pd.DataFrame,
    X_monthly: pd.DataFrame,
    y_quarterly: pd.Series,
    meta: pd.DataFrame,
    quarterly_origins: Iterable[pd.Period | str],
    pub_lag_map: pd.Series | None = None,
    factor_order: int = 2,
    maxiter: int = 200,
    m_start: str = "2017-01",
    m_end: str = "2025-12",
    cache_path: str | Path | None = None,
    series_cache_path: str | Path | None = None,
    force_rerun: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fit Core DFM at each monthly origin with a divergence-guard fallback
    ladder, and cache category contributions.

    Intended for small, structurally constrained selection sets (e.g. the
    block-balanced k=20 matrix) that occasionally hit an explosive EM optimum
    at ill-conditioned post-2022 windows -- the same guard ladder used by
    ``run_blockbalanced_benchmark.guarded_nowcast`` (drop idiosyncratic AR(1),
    then reduce to one factor; accept the first finite, plausible nowcast).
    Attribution reuses ``_contrib_frame`` unchanged, so the resulting chart has
    the identical ifoCAST-style predict()-based attribution as DFM-EN -- no
    intercept/baseline row is needed because this model has no bridge stage.

    Returns a long-format DataFrame with the same schema as
    :func:`run_category_contrib_panel`: ``monthly_origin``, ``quarter``,
    ``month_in_quarter``, ``nowcast``, ``actual``, ``category``, ``contrib_pp``.
    """
    from ..models.dfm.nowcast_utils import build_dfm_endog, extract_nowcast, fit_dfm

    cache_path = Path(cache_path) if cache_path is not None else None
    series_cache_path = Path(series_cache_path) if series_cache_path is not None else None
    if cache_path is not None and not needs_run_contrib_cache(
        cache_path, force_rerun, series_cache_path=series_cache_path,
    ):
        if verbose:
            print(f"Loaded category contributions from {cache_path.name}")
        return pd.read_parquet(cache_path)

    origins = build_monthly_origins(quarterly_origins, m_start=m_start, m_end=m_end)
    records: list[dict] = []
    series_records: list[dict] = []

    for origin_p, q, m_in_q in origins:
        origin_key = str(origin_p)
        if origin_key not in selection_matrix.index:
            if verbose:
                print(f"  {origin_key}: not in selection matrix — skipped.")
            continue
        sel_cols = selection_matrix.columns[
            selection_matrix.loc[origin_key].astype(bool)
        ].tolist()
        if not sel_cols:
            if verbose:
                print(f"  {origin_key}: no indicators — skipped.")
            continue
        if verbose:
            print(f"  {q} M{m_in_q} ({origin_key}): N={len(sel_cols)} ...", end=" ", flush=True)

        nc = np.nan
        frame = None
        fit_tag = "failed"
        for i, cfg in enumerate(_GUARD_LADDER):
            try:
                X_sel = X_monthly[sel_cols]
                endog, k_endog_M = build_dfm_endog(
                    X_sel, y_quarterly, origin_p, pub_lag_map=pub_lag_map,
                )
                result = fit_dfm(
                    endog,
                    k_endog_M=k_endog_M,
                    factor_order=factor_order,
                    maxiter=maxiter,
                    **cfg,
                )
                cand_nc = extract_nowcast(result, origin_p)
                if not (np.isfinite(cand_nc) and abs(cand_nc) <= _GUARD_NOWCAST_CAP):
                    raise ValueError(f"implausible nowcast {cand_nc}")
                nc = cand_nc
                frame = _contrib_frame(
                    result, selected_ids=sel_cols, meta=meta, target_quarter=q,
                )
                fit_tag = "base" if i == 0 else f"fallback{i}"
                break
            except Exception:
                continue

        if frame is None:
            if verbose:
                print("ERROR: all fallback configs diverged")
            continue

        actual = float(y_quarterly.get(q, np.nan))
        grouped = frame.groupby("category")["contrib_pp"].sum()
        for cat, val in grouped.items():
            records.append({
                "monthly_origin": origin_key,
                "quarter": str(q),
                "month_in_quarter": m_in_q,
                "nowcast": nc,
                "actual": actual,
                "category": cat,
                "contrib_pp": float(val),
            })
        for sid, row in frame.iterrows():
            series_records.append({
                "monthly_origin": origin_key,
                "quarter": str(q),
                "month_in_quarter": m_in_q,
                "series": sid,
                "category": row["category"],
                "contrib_pp": float(row["contrib_pp"]),
            })
        if verbose:
            print(f"nowcast={nc:.3f}  fit_tag={fit_tag}")

    df = pd.DataFrame(records)
    if cache_path is not None and not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        if verbose:
            print(f"Saved {len(df)} rows -> {cache_path}")
    if series_cache_path is not None and series_records:
        series_cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(series_records).to_parquet(series_cache_path, index=False)
        if verbose:
            print(f"Saved {len(series_records)} rows -> {series_cache_path}")
    return df
