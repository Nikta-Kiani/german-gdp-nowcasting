"""Plotting helpers for the XGBoost nowcasting notebook (stage 05).

Reuses the publication style from ``nowcast_plots`` (stage 04) so thesis
figures are visually consistent across DFM and XGB benchmarks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .nowcast_plots import (
    MODEL_COLORS,
    MODEL_LABELS,
    add_recession_bands,
    setup_style as _dfm_setup_style,
    _apply_year_axis,
    _q_to_ts,
    _save,
)

# Family palettes: baselines/DFM/SV follow stage-04 MODEL_COLORS; XGB uses sage–mint greens.
BASELINE_PALETTE: dict[str, str] = {
    k: MODEL_COLORS[k] for k in ("RW", "AR1") if k in MODEL_COLORS
}

XGB_GREEN_PALETTE: dict[str, str] = {
    "XGB-Full": "#358F6A",
}

DFM_ROSE_PALETTE: dict[str, str] = {
    "en_only": MODEL_COLORS["en_only"],
    "DFM-EN": MODEL_COLORS["en_only"],
    "DFM-ifoCAST": MODEL_COLORS["ifoCAST"],
    "DFM-BlockBalanced": MODEL_COLORS["blockbalanced"],
    "combo_equal": "#D4A574",
}

SV_BLUE_PALETTE: dict[str, str] = {
    "DFM-SV-k2": MODEL_COLORS["SV_k2"],
}

# Legend / bar order: baselines → DFM → ensemble → ML
DEFAULT_MODEL_ORDER: list[str] = [
    "RW", "AR1",
    "DFM-ifoCAST", "DFM-EN", "DFM-BlockBalanced",
    "DFM-SV-k2", "combo_equal",
    "XGB-Full",
]

XGB_MODEL_COLORS: dict[str, str] = {
    **MODEL_COLORS,
    **XGB_GREEN_PALETTE,
    **DFM_ROSE_PALETTE,
    **SV_BLUE_PALETTE,
}

XGB_MODEL_LABELS: dict[str, str] = {
    **MODEL_LABELS,
    "XGB-Full": "XGBoost (Full + SHAP)",
    "DFM-EN": "DFM-EN",
    "DFM-ifoCAST": "DFM-ifoCAST",
    "DFM-BlockBalanced": "DFM-k20",
    "DFM-SV-k2": "DFM-SV (k=2, integrated)",
    "combo_equal": "Equal combo",
}


def setup_style() -> None:
    """Apply stage-04 matplotlib rcParams."""
    _dfm_setup_style()


def sort_models_for_plot(
    models: list[str],
    order: list[str] | None = None,
) -> list[str]:
    """Order series for grouped bars: RW/AR1 → XGB → DFM → DFM-SV."""
    order = order or DEFAULT_MODEL_ORDER
    known = [m for m in order if m in models]
    extra = sorted(m for m in models if m not in known)
    return known + extra


def _color_for(name: str) -> str:
    """Return the stable model-family color for a result label."""
    if name in XGB_MODEL_COLORS:
        return XGB_MODEL_COLORS[name]
    if name.startswith("DFM-SV-k"):
        return SV_BLUE_PALETTE.get(
            name, MODEL_COLORS.get(name.replace("DFM-SV-", "SV_"), MODEL_COLORS["SV_k2"]),
        )
    if name in DFM_ROSE_PALETTE:
        return DFM_ROSE_PALETTE[name]
    if name.startswith("DFM-"):
        return DFM_ROSE_PALETTE["core"]
    return MODEL_COLORS.get(name, "#94A3B8")


def _label_for(name: str) -> str:
    """Return the publication label for a model name."""
    return XGB_MODEL_LABELS.get(name, name)


def _m3_series(df: pd.DataFrame) -> pd.DataFrame:
    """Headline M3 nowcasts; one row per quarter for plotting."""
    out = df.copy()
    if "month_in_quarter" in out.columns:
        out = out.loc[out["month_in_quarter"] == 3]
    if out.index.name != "quarter" and "quarter" in out.columns:
        out = out.set_index("quarter")
    return out


def fig_nowcast_vs_actual(
    results: Mapping[str, pd.DataFrame],
    title: str = "XGBoost nowcasts vs first-release GDP growth (M3)",
    interval_model: str | None = None,
    interval_alpha: int = 90,
    save: str | Path | None = None,
) -> plt.Figure:
    """Line chart of M3 point nowcasts + actual GDP."""
    fig, ax = plt.subplots(figsize=(12, 4.8))
    actual_drawn = False
    all_x: list[pd.DatetimeIndex] = []

    for name, df in results.items():
        sub = _m3_series(df)
        x = _q_to_ts(sub.index)
        all_x.append(x)
        if not actual_drawn:
            ax.plot(
                x, sub["actual"].values,
                color=MODEL_COLORS["actual"], lw=2.0, label="Actual GDP",
            )
            actual_drawn = True
        ax.plot(
            x, sub["nowcast"].values,
            lw=1.6, color=_color_for(name), label=_label_for(name),
        )

    if interval_model and interval_model in results:
        sub = _m3_series(results[interval_model])
        lo, hi = f"ci_lower_{interval_alpha}", f"ci_upper_{interval_alpha}"
        if lo in sub.columns and hi in sub.columns:
            x = _q_to_ts(sub.index)
            ax.fill_between(
                x, sub[lo].values, sub[hi].values,
                alpha=0.18, color=_color_for(interval_model),
                label=f"{_label_for(interval_model)} {interval_alpha}% PI",
            )

    if all_x:
        ax.set_xlim(min(x[0] for x in all_x), max(x[-1] for x in all_x))
    ax.axhline(0, color="#CBD5E1", lw=0.8)
    add_recession_bands(ax, annotate=True)
    ax.set_title(title, fontsize=12, fontweight="600")
    ax.set_ylabel("Q/Q log-growth (pp)")
    ax.legend(loc="lower left", framealpha=0.85, ncol=2)
    _apply_year_axis(ax, base=2)
    fig.tight_layout()
    _save(fig, save)
    return fig


def fig_xgb_dfm_overlay(
    xgb_df: pd.DataFrame,
    dfm_df: pd.DataFrame,
    xgb_name: str = "XGB-core",
    dfm_name: str = "DFM-EN",
    title: str = "XGB vs DFM (Core) — M3 nowcasts",
    save: str | Path | None = None,
) -> plt.Figure:
    """Overlay XGB-Full and DFM-EN against actual GDP."""
    fig, ax = plt.subplots(figsize=(12, 4.8))
    xgb = _m3_series(xgb_df)
    dfm = _m3_series(dfm_df)
    x = _q_to_ts(xgb.index)
    ax.plot(x, xgb["actual"].values, color=MODEL_COLORS["actual"],
            lw=2.0, label="Actual GDP")
    ax.plot(x, xgb["nowcast"].values, color=_color_for(xgb_name), lw=1.7,
            label=_label_for(xgb_name))
    ax.plot(_q_to_ts(dfm.index), dfm["nowcast"].values,
            color=_color_for(dfm_name), lw=1.7, ls="--",
            label=_label_for(dfm_name))
    ax.axhline(0, color="#CBD5E1", lw=0.8)
    add_recession_bands(ax, annotate=True)
    ax.set_title(title, fontsize=12, fontweight="600")
    ax.set_ylabel("Q/Q log-growth (pp)")
    ax.legend(loc="lower left", framealpha=0.85)
    _apply_year_axis(ax, base=2)
    fig.tight_layout()
    _save(fig, save)
    return fig


def fig_rmsfe_bar(
    rmsfe_table: pd.DataFrame,
    metric: str = "RMSFE",
    highlight_prefix: str = "XGB",
    title: str | None = None,
    save: str | Path | None = None,
) -> plt.Figure:
    """Horizontal RMSFE / NSR bars (DFM-style colours)."""
    sub = rmsfe_table[[metric]].dropna().sort_values(metric)
    colors = [_color_for(m) for m in sub.index]
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(sub) + 1.2))
    ax.barh(sub.index.astype(str), sub[metric].values, color=colors, height=0.72)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(title or f"{metric} (M3, {sub.index[0]}…)", fontsize=11)
    for i, v in enumerate(sub[metric].values):
        ax.text(v, i, f"  {v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    _save(fig, save)
    return fig


DEFAULT_REGIMES: dict[str, tuple[str, str]] = {
    "pre-COVID":  ("2011Q1", "2019Q4"),
    "COVID":      ("2020Q1", "2021Q4"),
    "post-COVID": ("2022Q1", "2025Q4"),
}


def _regime_rmsfe_table(
    results_by_model: Mapping[str, pd.DataFrame],
    regimes: Mapping[str, tuple[str, str]],
    month_in_quarter: int | None = 3,
) -> pd.DataFrame:
    """Pivot table: regimes × models (RMSFE values)."""
    from ..models.dfm.nowcast_utils import _subset_eval_window

    rows: list[dict] = []
    for model, df in results_by_model.items():
        for label, (q0, q1) in regimes.items():
            sub = _subset_eval_window(
                df, eval_start=q0, eval_end=q1, month_in_quarter=month_in_quarter,
            )
            errs = sub["error"].dropna()
            value = (
                float(np.sqrt(np.mean(errs.values ** 2)))
                if len(errs) else np.nan
            )
            rows.append({"model": model, "regime": label, "RMSFE": value})
    tbl = pd.DataFrame(rows).pivot(index="regime", columns="model", values="RMSFE")
    return tbl.reindex(list(regimes.keys()))


def fig_rmsfe_by_regime(
    results_by_model: Mapping[str, pd.DataFrame],
    regimes: Mapping[str, tuple[str, str]] | None = None,
    month_in_quarter: int | None = 3,
    models: list[str] | None = None,
    title: str = "Predictive accuracy by economic regime (RMSFE, M3)",
    save: str | Path | None = None,
) -> plt.Figure:
    """Grouped bar chart matching stage-04 layout: regimes on the x-axis,
    one bar per model within each regime group (pre-COVID | COVID | post-COVID).
    """
    regimes = regimes or DEFAULT_REGIMES
    source = results_by_model
    if models is not None:
        source = {k: v for k, v in results_by_model.items() if k in models}

    tbl = _regime_rmsfe_table(source, regimes, month_in_quarter=month_in_quarter)
    model_list = sort_models_for_plot(list(tbl.columns))
    tbl = tbl[model_list]
    n_m = len(model_list)
    width = 0.8 / max(n_m, 1)
    x = np.arange(len(regimes))

    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    for i, m in enumerate(model_list):
        ax.bar(
            x + (i - (n_m - 1) / 2) * width,
            tbl[m].values,
            width,
            color=_color_for(m),
            edgecolor="white",
            linewidth=0.6,
            label=_label_for(m),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(list(regimes.keys()), rotation=0, ha="center")
    ax.set_ylabel("RMSFE (pp)")
    ax.set_title(title, fontsize=12, fontweight="600")
    ax.legend(
        ncol=min(n_m, 6),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
        borderaxespad=0.0,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.98))
    _save(fig, save)
    return fig


def _strip_lag(col: str) -> str:
    """Remove an XGBoost lag suffix from a feature name."""
    return col.rsplit("__L", 1)[0] if "__L" in col else col


def fig_shap_bar(
    shap_log: pd.DataFrame,
    top_n: int = 15,
    aggregate_lags: bool = True,
    title: str = "SHAP mean |φ| (SHAP-refit quarters, XGB-Full)",
    save: str | Path | None = None,
) -> plt.Figure:
    """Top-N SHAP importance with cross-quarter error bars."""
    if shap_log.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No SHAP log available", ha="center", va="center")
        return fig

    df = shap_log.reset_index()
    if aggregate_lags:
        df["feature_base"] = df["feature"].map(_strip_lag)
        agg = (
            df.groupby(["quarter", "feature_base"])["mean_abs_shap"]
            .sum().reset_index()
        )
        stats = (
            agg.groupby("feature_base")["mean_abs_shap"]
            .agg(["mean", "std"])
            .sort_values("mean", ascending=False)
        )
    else:
        stats = (
            df.groupby("feature")["mean_abs_shap"]
            .agg(["mean", "std"])
            .sort_values("mean", ascending=False)
        )
    stats = stats.head(top_n)
    fig, ax = plt.subplots(figsize=(8.5, 0.38 * len(stats) + 1.0))
    ax.barh(
        stats.index.astype(str), stats["mean"].values,
        xerr=stats["std"].fillna(0).values,
        color=MODEL_COLORS["SV_k2"], alpha=0.88,
        error_kw=dict(ecolor="#334155", lw=0.7, capsize=2),
    )
    ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, save)
    return fig


def show_plotly(fig: Any) -> None:
    """Display a Plotly figure inline in Jupyter / VS Code / Cursor notebooks."""
    try:
        from IPython.display import display
        display(fig)
    except ImportError:
        import plotly.io as pio
        fig.show(renderer=pio.renderers.default)


def fig_feature_count_evolution(
    results: Mapping[str, pd.DataFrame],
    title: str = "Feature count per M3 origin",
    save: str | Path | None = None,
) -> plt.Figure:
    """Plot the deployed feature count at each M3 origin."""
    fig, ax = plt.subplots(figsize=(12, 3.8))
    for name, df in results.items():
        sub = _m3_series(df)
        if "n_features" not in sub.columns:
            continue
        ax.plot(
            _q_to_ts(sub.index), sub["n_features"].values,
            lw=1.5, color=_color_for(name), label=_label_for(name),
        )
    ax.set_ylabel("# features")
    ax.set_title(title)
    add_recession_bands(ax, annotate=False)
    ax.legend(frameon=False)
    _apply_year_axis(ax, base=2)
    fig.tight_layout()
    _save(fig, save)
    return fig


def fig_xgb_nowcast_interactive(
    xgb_df: pd.DataFrame,
    dfm_df: pd.DataFrame | None = None,
    xgb_name: str = "XGB-Core",
    dfm_name: str = "DFM-EN",
    save_html: str | Path | None = None,
    show: bool = True,
) -> Any:
    """Plotly line chart: actual, XGB, optional DFM (hover = quarter / error)."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for interactive charts. pip install plotly"
        ) from exc

    xgb = _m3_series(xgb_df)
    x = _q_to_ts(xgb.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=xgb["actual"], name="Actual GDP",
        line=dict(color=MODEL_COLORS["actual"], width=2.5),
        hovertemplate="%{x|%Y-Q%q}<br>Actual: %{y:.3f} pp<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=xgb["nowcast"], name=_label_for(xgb_name),
        line=dict(color=_color_for(xgb_name), width=2),
        customdata=np.column_stack([
            xgb["error"].values,
            xgb.get("monthly_origin", pd.Series(index=xgb.index)).values,
        ]),
        hovertemplate=(
            "%{x|%Y-Q%q}<br>Nowcast: %{y:.3f} pp<br>"
            "Error: %{customdata[0]:.3f}<br>Origin: %{customdata[1]}<extra></extra>"
        ),
    ))
    if dfm_df is not None:
        dfm = _m3_series(dfm_df)
        fig.add_trace(go.Scatter(
            x=_q_to_ts(dfm.index), y=dfm["nowcast"], name=_label_for(dfm_name),
            line=dict(color=_color_for(dfm_name), width=2, dash="dash"),
            hovertemplate="%{x|%Y-Q%q}<br>DFM: %{y:.3f} pp<extra></extra>",
        ))
    fig.update_layout(
        title="German GDP growth — XGB vs DFM (M3)",
        xaxis_title="", yaxis_title="Q/Q log-growth (pp)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=480,
    )
    if save_html is not None:
        p = Path(save_html)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(p))
    if show:
        show_plotly(fig)
    return fig


def fig_xgb_shap_category_interactive(
    shap_log: pd.DataFrame,
    category_map: pd.Series,
    save_html: str | Path | None = None,
    show: bool = True,
) -> Any:
    """Stacked area: SHAP mass by macro category over evaluation quarters."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plotly is required.") from exc

    if shap_log.empty:
        raise ValueError("shap_log is empty")

    df = shap_log.reset_index()
    df["series_id"] = df["feature"].map(_strip_lag)
    df["category"] = df["series_id"].map(category_map).fillna("Unknown")
    by_q = (
        df.groupby(["quarter", "category"])["mean_abs_shap"]
        .sum().unstack(fill_value=0).sort_index()
    )
    quarters = by_q.index.astype(str)
    fig = go.Figure()
    for cat in by_q.columns:
        fig.add_trace(go.Scatter(
            x=quarters, y=by_q[cat].values, name=cat,
            stackgroup="one", mode="lines",
            hovertemplate=f"{cat}<br>%{{x}}: %{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        title="XGB-Full SHAP mass by indicator category",
        xaxis_title="Quarter", yaxis_title="Σ mean |SHAP|",
        template="plotly_white", height=440,
    )
    if save_html is not None:
        p = Path(save_html)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(p))
    if show:
        show_plotly(fig)
    return fig


def save_fig(fig: plt.Figure, path: str | Path) -> None:
    """Save matplotlib figure (compat wrapper)."""
    _save(fig, path)
