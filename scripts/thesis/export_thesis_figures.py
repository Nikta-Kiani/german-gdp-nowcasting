#!/usr/bin/env python3
"""Export all thesis figures as vector PDFs from the canonical dashboard data.

Reuses the companion dashboard's chart builders and data layer so every
figure is generated from the same stored result cut. Figures are written
under ``THESIS_ROOT/figures/`` with stable filenames. In-figure titles are
stripped: captions live in the thesis document.

Requires the companion dashboard on ``DASHBOARD_SRC``. Write the figures
to ``THESIS_ROOT``.

Run from the repository root:

    python scripts/thesis/export_thesis_figures.py

Kaleido cannot render the dashboard's quoted Inter font stack, so a thesis
template with an unquoted Arial/Helvetica family is registered instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

WORKSPACE = Path(__file__).resolve().parents[3]
DASH_SRC = Path(os.environ.get(
    "DASHBOARD_SRC", WORKSPACE / "german-gdp-nowcast-dashboard" / "src",
))
sys.path.insert(0, str(DASH_SRC))

from dashboard import charts, config as C, data  # noqa: E402

FIG_DIR = Path(os.environ.get(
    "THESIS_ROOT", WORKSPACE / "Overleaf-Thesis",
)) / "figures"
PREVIEW_DIR = Path(__file__).resolve().parent / "figure_previews"

TEXT_WIDTH_PX = 531  # 398.33862 pt at 96 css-px/in
FONT_FAMILY = "Arial, Helvetica, sans-serif"
# Thesis PDFs are printed at ~textwidth; dashboard defaults (12–14 pt) read
# oversized once embedded. Keep body/tick fonts one step below caption size.
AXIS_TITLE_SIZE = 10
TICK_SIZE = 9
BODY_SIZE = 11
LEGEND_SIZE = 10


def register_thesis_template() -> None:
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        font=dict(family=FONT_FAMILY, size=BODY_SIZE, color=C.INK),
        paper_bgcolor=C.PAPER,
        plot_bgcolor=C.PAPER,
        title=dict(font=dict(size=13, color=C.INK), x=0.0, xanchor="left"),
        margin=dict(t=70, b=60, l=70, r=30),
        colorway=[C.CATEGORY_COLORS[c] for c in C.CATEGORY_ORDER],
        xaxis=dict(
            showgrid=False, zeroline=False, linecolor=C.GRID,
            ticks="outside", tickcolor=C.GRID,
            tickfont=dict(size=TICK_SIZE, family=FONT_FAMILY, color=C.INK),
            title=dict(font=dict(size=AXIS_TITLE_SIZE, family=FONT_FAMILY,
                                 color=C.INK)),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=C.GRID, gridwidth=1, zeroline=False,
            tickfont=dict(size=TICK_SIZE, family=FONT_FAMILY, color=C.INK),
            title=dict(font=dict(size=AXIS_TITLE_SIZE, family=FONT_FAMILY,
                                 color=C.INK)),
        ),
        legend=dict(bgcolor="rgba(255,255,255,0.7)", bordercolor=C.GRID,
                    borderwidth=0, font=dict(size=LEGEND_SIZE)),
        hovermode="closest",
    )
    pio.templates["thesis"] = tpl
    pio.templates.default = "plotly_white+thesis"


def _apply_thesis_fonts(fig: go.Figure) -> None:
    """Force compact axis/legend fonts after chart builders may set larger ones."""
    fig.update_layout(
        font=dict(family=FONT_FAMILY, size=BODY_SIZE, color=C.INK),
        legend_font=dict(size=LEGEND_SIZE, family=FONT_FAMILY),
    )
    fig.update_xaxes(
        tickfont=dict(size=TICK_SIZE, family=FONT_FAMILY, color=C.INK),
        title_font=dict(size=AXIS_TITLE_SIZE, family=FONT_FAMILY, color=C.INK),
    )
    fig.update_yaxes(
        tickfont=dict(size=TICK_SIZE, family=FONT_FAMILY, color=C.INK),
        title_font=dict(size=AXIS_TITLE_SIZE, family=FONT_FAMILY, color=C.INK),
    )


def export(fig: go.Figure, name: str, *, width: int = TEXT_WIDTH_PX,
           height: int = 380, strip_title: bool = True,
           margin: dict | None = None) -> None:
    if strip_title:
        fig.update_layout(title_text="")
    _apply_thesis_fonts(fig)
    fig.update_layout(width=width, height=height)
    if margin:
        fig.update_layout(margin=margin)
    out = FIG_DIR / f"{name}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Round-trip through plotly's JSON encoder: kaleido's orjson cannot
    # serialize pandas Timestamps inside vrect/vline shapes.
    fig = pio.from_json(pio.to_json(fig))
    pio.write_image(fig, out, format="pdf", width=width, height=height, scale=1)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pio.write_image(fig, PREVIEW_DIR / f"{name}.png", format="png",
                    width=width, height=height, scale=2)
    print(f"exported {name}.pdf ({width}x{height})")


# ------------------------------------------------------------------ Part I --
def fig_publag_composition() -> None:
    mat = data.publag_category_matrix()
    fig = charts.publag_composition(mat)
    export(fig, "selection_publag_composition", height=330,
           margin=dict(t=20, b=55, l=165, r=150))


def fig_soft_hard_by_regime() -> None:
    long = data.regime_soft_hard()
    uni = data.universe_soft_hard()
    fig = charts.regime_rotation_bars(long, uni)
    export(fig, "selection_soft_hard_by_regime", height=360,
           margin=dict(t=35, b=80, l=70, r=30))


def fig_agreement_heatmap() -> None:
    df = data.cross_method_agreement()
    fig = charts.agreement_heatmap(
        df, title="", label="Spearman rho", height=420)
    export(fig, "selection_agreement_spearman", height=400,
           margin=dict(t=20, b=110, l=150, r=30))


# Single-method stacked-area exports (selection_*_category_mass.pdf) are
# superseded by ``export_selection_grid.py`` → selection_category_mass_grid.pdf.


# ----------------------------------------------------------------- Part II --
# All models with a saved nowcast path, DFM-PLS included: it is part of the
# headline horse race in the thesis (fig nowcast_rmsfe_by_regime).
NOWCAST_MODELS = data.accuracy_models()


def fig_rmsfe_by_regime() -> None:
    rmsfe_long = data.rmsfe_by_regime()
    # full_window_accuracy() covers every saved path (incl. DFM-PLS), unlike
    # the summary CSV; the row order follows the full-sample RMSFE ranking.
    acc = data.full_window_accuracy().sort_values("rmse")
    order = acc["model"].tolist()
    fig = charts.rmsfe_regime_bars(
        rmsfe_long, NOWCAST_MODELS, list(C.REGIMES), order)
    # Lift the two-line panel titles off the first bar; the extra top
    # margin keeps them inside the canvas after the shift.
    n_reg = len(C.REGIMES)
    for ann in fig.layout.annotations[:n_reg]:
        ann.update(yshift=14)
    export(fig, "nowcast_rmsfe_by_regime", width=800, height=480,
           margin=dict(t=72, b=54, l=156, r=54))


def fig_postcovid_benchmarks() -> None:
    df = data.load_post_covid()
    fig = charts.post_covid_bars(df, "post-COVID_rmsfe")
    export(fig, "nowcast_postcovid_benchmarks", height=360,
           margin=dict(t=20, b=50, l=160, r=55))


# Four information sets of the post-COVID release-block experiment (Fig. 8.5).
# Two visual groups at each origin: hard-frozen (left pair) vs hard-updates
# (right pair). A gap between the pairs is the 2×2 design; equal heights
# inside a pair are the empirical finding.
_RELEASE_BLOCK_ORDER = (
    "both_frozen", "other_only", "hard_only", "full",
)
# Offsets around each origin tick: left pair, gap, right pair.
_RELEASE_BLOCK_OFFSETS = {
    "both_frozen": -0.30,
    "other_only": -0.12,
    "hard_only": 0.12,
    "full": 0.30,
}


def fig_release_block() -> None:
    """Paired-bar RMSFE chart for the four post-COVID release-block sets.

    M1 is drawn as a reference line (the frozen information set). At M2 and
    M3 the left pair holds hard activity frozen; the right pair lets it
    update. Matching heights inside a pair are the result.
    """
    df = data.load_release_block_states()
    if df.empty:
        raise FileNotFoundError(
            f"Missing {C.RELEASE_BLOCK_STATES_CSV}. Stage "
            "release_block_counterfactual_states.csv into the dashboard data."
        )
    rmsfe = df.pivot(index="horizon", columns="state", values="RMSFE")
    m1 = float(rmsfe.loc["M2", "both_frozen"])
    origins = ["M2", "M3"]
    x = list(range(len(origins)))
    bar_w = 0.16

    fig = go.Figure()
    fig.add_hline(
        y=m1, line=dict(color=C.RELEASE_BLOCK_COLORS["both_frozen"],
                        width=1.2, dash="dot"),
    )
    fig.add_annotation(
        x=1.48, y=m1, xref="x", yref="y", xanchor="left",
        yanchor="bottom", yshift=4, showarrow=False,
        text=f"M1  {m1:.3f}",
        font=dict(size=9, color=C.RELEASE_BLOCK_COLORS["both_frozen"],
                  family=FONT_FAMILY),
    )
    for key in _RELEASE_BLOCK_ORDER:
        color = C.RELEASE_BLOCK_COLORS[key]
        xs = [xi + _RELEASE_BLOCK_OFFSETS[key] for xi in x]
        ys = [float(rmsfe.loc[h, key]) for h in origins]
        fig.add_trace(go.Bar(
            x=xs, y=ys, width=bar_w, name=C.RELEASE_BLOCK_LABELS[key],
            marker=dict(color=color, line=dict(width=0)),
            text=[f"{v:.3f}" for v in ys],
            textposition="outside",
            textfont=dict(size=8.5, color=C.INK, family=FONT_FAMILY),
            cliponaxis=False,
        ))

    # Group captions under the x ticks (hard frozen | hard updates).
    for xi in x:
        fig.add_annotation(
            x=xi + (_RELEASE_BLOCK_OFFSETS["both_frozen"]
                    + _RELEASE_BLOCK_OFFSETS["other_only"]) / 2,
            y=0, yref="y", yanchor="top", yshift=-18, showarrow=False,
            text="hard frozen",
            font=dict(size=8.5, color=C.SUBTLE, family=FONT_FAMILY),
        )
        fig.add_annotation(
            x=xi + (_RELEASE_BLOCK_OFFSETS["hard_only"]
                    + _RELEASE_BLOCK_OFFSETS["full"]) / 2,
            y=0, yref="y", yanchor="top", yshift=-18, showarrow=False,
            text="hard updates",
            font=dict(size=8.5, color=C.SUBTLE, family=FONT_FAMILY),
        )

    fig.update_xaxes(
        tickmode="array", tickvals=x, ticktext=origins,
        title_text="Information set", title_standoff=22,
        range=[-0.55, 1.75],
    )
    fig.update_yaxes(
        title_text="RMSFE (pp)", title_standoff=6,
        rangemode="tozero", range=[0, 0.56],
    )
    fig.update_layout(
        barmode="overlay", bargap=0,
        legend=dict(
            orientation="h", y=-0.28, x=0.5, xanchor="center",
            font=dict(size=9.5),
        ),
    )
    export(fig, "nowcast_release_block_counterfactual", width=800, height=360,
           strip_title=True, margin=dict(t=28, b=88, l=58, r=22))


# Simple coloured lines. The dashboard rose-to-plum arc is too close for six
# overlapping paths, so this figure uses a muted qualitative palette instead.
_HORIZON_SPECS = [
    ("DFM-ifoCAST", "DFM-ifoCAST", "#E07A5F"),
    ("DFM-EN", "DFM-EN", "#C44569"),
    ("DFM-PLS", "DFM-PLS", "#2A9D8F"),
    ("DFM-BlockBalanced", "DFM-block-balanced", "#5C4B8A"),
    ("DFM-TVP", "DFM-TVP", "#9E77C0"),
    ("DFM-SV-k2", "DFM-SV", "#3D6FA8"),
]
_REGIME_N = {"pre-COVID": 36, "COVID": 8, "post-COVID": 16}


def fig_horizon_profiles() -> None:
    df = data.load_horizon_profile()
    present = set(df["model"])
    specs = [spec for spec in _HORIZON_SPECS if spec[0] in present]
    regimes = list(C.REGIMES)
    fig = make_subplots(
        rows=1, cols=3, shared_yaxes=False, horizontal_spacing=0.08,
        subplot_titles=[
            (
                f"<b>{r}</b><br>"
                f"<span style='font-size:10px;color:{C.SUBTLE}'>"
                f"{C.REGIMES[r][0]}–{C.REGIMES[r][1]}"
                f"  ·  N = {_REGIME_N[r]}</span>"
            )
            for r in regimes
        ],
    )
    miq_label = {1: "M1", 2: "M2", 3: "M3"}
    for col, regime in enumerate(regimes, start=1):
        sub = df[df["regime"] == regime]
        for key, label, color in specs:
            s = sub[sub["model"] == key].sort_values("month_in_quarter")
            if s.empty:
                continue
            fig.add_trace(go.Scatter(
                x=[miq_label[i] for i in s["month_in_quarter"]],
                y=s["RMSFE"], name=label, mode="lines+markers",
                line=dict(color=color, width=2.2, shape="linear"),
                marker=dict(size=6, symbol="circle", color=color),
                showlegend=(col == 1), legendgroup=key,
            ), row=1, col=col)
        fig.update_yaxes(rangemode="tozero", row=1, col=col)
        fig.update_xaxes(
            title_text="Information set" if col == 2 else None,
            row=1, col=col,
        )
    fig.update_yaxes(title_text="RMSFE (pp)", title_standoff=6, row=1, col=1)
    fig.update_layout(
        legend=dict(
            orientation="h", y=-0.24, x=0.5, xanchor="center",
            font=dict(size=9.5),
        ),
    )
    for ann in fig.layout.annotations[:3]:
        ann.update(font=dict(size=11, color=C.INK, family=FONT_FAMILY))
    export(fig, "nowcast_horizon_profiles", width=800, height=340,
           strip_title=True, margin=dict(t=52, b=80, l=62, r=22))


_BV_MODELS = ["DFM-EN", "DFM-ifoCAST", "DFM-PLS", "DFM-BlockBalanced",
              "DFM-TVP", "DFM-SV-k2"]
_BV_TITLES = {key: label for key, label, _ in _HORIZON_SPECS}
_BV_REGIME_FILES = {
    "pre-COVID": "nowcast_bias_variance_pre_covid",
    "COVID": "nowcast_bias_variance_covid",
    "post-COVID": "nowcast_bias_variance_postcovid",
}


def fig_bias_variance_by_regime() -> None:
    """Cross-model bias–variance panels, one PDF per evaluation regime.

    Same dashboard design as the headline post-COVID figure (fig:bias-variance);
    pre-COVID and COVID companions are for the appendix.
    """
    df = data.load_horizon_bias_variance()
    models = [m for m in _BV_MODELS if m in set(df["model"])]
    for regime, name in _BV_REGIME_FILES.items():
        fig = charts.bias_variance_decomposition(df, models, regime)
        for ann in fig.layout.annotations[:len(models)]:
            key = ann.text.replace("<b>", "").replace("</b>", "")
            ann.update(
                text=f"<b>{_BV_TITLES.get(key, key)}</b>",
                font=dict(size=9, color=C.INK, family=FONT_FAMILY),
            )
        export(fig, name, width=860, height=340,
               margin=dict(t=48, b=64, l=58, r=18))


def fig_mz_forest() -> None:
    mz = data.load_mincer_zarnowitz().copy()
    # Display labels in the forest plot while keeping the model-key colors.
    labels = {m: C.model_label(m) for m in mz["model"]}
    charts._DFM_TABLE_EXTRA.update(
        {label: key for key, label in labels.items()})
    mz["model"] = mz["model"].map(labels)
    fig = charts.mz_forest(mz)
    export(fig, "nowcast_mincer_zarnowitz", width=700, height=400,
           margin=dict(t=42, b=52, l=140, r=30))


def fig_xgb_sensitivity() -> None:
    df = data.load_xgb_sensitivity()
    post = data.load_post_covid().set_index("model")["post-COVID_rmsfe"]
    dfm_rows = [m for m in post.index if m.startswith("DFM")]
    dfm_vals = post.loc[dfm_rows].astype(float)
    dfm_range = (float(dfm_vals.min()), float(dfm_vals.max()))
    best_dfm = (dfm_vals.idxmin(), float(dfm_vals.min()))
    rolling = float(post.loc["Rolling-AR(1) 40q"])
    fig = charts.xgb_sensitivity_bars(df, dfm_range, best_dfm, rolling)
    export(fig, "nowcast_xgb_postcovid_sensitivity", width=700, height=420,
           margin=dict(t=52, b=50, l=170, r=60))


def fig_factor_loadings() -> None:
    """Combined factor-interpretation panel for sec:now-inputsets.

    One vector PDF with three tagged panels: (a)/(b) category shares of
    |indicator->factor loadings| for the two DFM-EN factors at semiannual M3
    origins (Stage 1, as before), and (c) the DFM-TVP bridge coefficients
    beta_1(q), beta_2(q) of eq:tvp-obs at quarterly M3 origins (Stage 2, the
    dashboard's lambda plot). The bridge panel is descriptive: the pipeline
    aligns factor sign/permutation across origins, and the thesis text says
    so. Bridge lines are end-labelled directly, so the single legend can stay
    with the category palette of row 1.
    """
    cat_df = data.load_factor_loading_categories()
    tvp = data.load_tvp_m3_bridge()

    x0 = str(min(cat_df["date"].min(), tvp["date"].min()).date())
    x1 = str(max(cat_df["date"].max(), tvp["date"].max()).date())

    fig = make_subplots(
        rows=2, cols=2, horizontal_spacing=0.07, vertical_spacing=0.11,
        row_heights=[0.50, 0.50],
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=[
            (f"<b>(a)</b> <span style='color:{C.FACTOR_COLORS[0]}'>"
             f"{C.FACTOR_SHORT[0]}</span> — category shares of |loading|"),
            (f"<b>(b)</b> <span style='color:{C.FACTOR_COLORS[1]}'>"
             f"{C.FACTOR_SHORT[1]}</span> — category shares of |loading|"),
            ("<b>(c)</b> DFM-TVP bridge coefficients on the quarterly "
             "factor averages (M3 origins)"),
        ],
    )

    for f in (1, 2):
        sub = cat_df[cat_df["factor"] == f]
        wide = (
            sub.pivot_table(index="date", columns="category", values="share",
                            aggfunc="first")
            .reindex(columns=C.FACTOR_LOADING_CATEGORIES, fill_value=0.0)
            .sort_index()
        )
        charts._stacked_share_areas(fig, wide, row=1, col=f,
                                    show_legend=(f == 2))
        fig.add_vrect(
            x0="2020-01-01", x1="2021-12-31", row=1, col=f,
            fillcolor="rgba(226,137,155,0.12)", line_width=0, layer="below",
        )
        fig.update_yaxes(range=[0, 1], row=1, col=f,
                         showticklabels=(f == 1))
        fig.update_xaxes(
            showticklabels=True, tickformat="%Y", tick0="2012-01-01",
            dtick="M24", ticks="outside", ticklen=3, range=[x0, x1],
            row=1, col=f,
        )
    fig.update_yaxes(title_text="Share of |loading|", row=1, col=1)

    # (c) Stage-2 bridge: beta_1, beta_2 over M3 origins, end-labelled.
    fig.add_vrect(
        x0="2020-01-01", x1="2021-12-31", row=2, col=1,
        fillcolor="rgba(226,137,155,0.12)", line_width=0, layer="below",
    )
    fig.add_hline(y=0.0, line=dict(color="#C5CED8", width=0.8), row=2, col=1)
    beta_cols = ["tvp_loading_1", "tvp_loading_2"]
    beta_names = ["β₁  Factor 1", "β₂  Factor 2"]
    ends: list[float] = []
    for j, col in enumerate(beta_cols):
        fig.add_trace(go.Scatter(
            x=tvp["date"], y=tvp[col], mode="lines",
            line=dict(color=C.FACTOR_COLORS[j], width=2.0),
            showlegend=False, hoverinfo="skip",
        ), row=2, col=1)
        ends.append(float(tvp[col].iloc[-1]))
    if abs(ends[0] - ends[1]) < 0.06:
        ends[0] += 0.03
        ends[1] -= 0.03
    # End labels sit inside the plot, just left of the final point: the
    # higher-ending line is labelled above its end, the lower one below.
    hi = int(np.argmax(ends))
    for j, val in enumerate(ends):
        fig.add_annotation(
            x=tvp["date"].iloc[-1], y=val, xref="x3", yref="y3",
            text=beta_names[j], showarrow=False, xanchor="right", xshift=-6,
            yanchor="bottom" if j == hi else "top",
            yshift=5 if j == hi else -5,
            font=dict(size=9, color=C.FACTOR_COLORS[j], family=FONT_FAMILY),
        )
    fig.add_annotation(
        x="2020-01-15", y=1.0, yref="y3 domain", xref="x3",
        text="COVID (down-weighted)", showarrow=False,
        xanchor="left", yanchor="top", yshift=-4,
        font=dict(size=9, color=C.SUBTLE, family=FONT_FAMILY),
    )
    fig.update_yaxes(title_text="pp of GDP growth per factor unit",
                     zeroline=False, row=2, col=1)
    fig.update_xaxes(
        title_text="Forecast origin", tickformat="%Y",
        tick0="2012-01-01", dtick="M24", range=[x0, x1],
        title_standoff=6, ticks="outside", ticklen=3,
        row=2, col=1,
    )

    for ann in fig.layout.annotations[:3]:
        ann.update(font=dict(size=11, family=FONT_FAMILY), yshift=0)
    fig.update_layout(
        legend=dict(orientation="v", x=1.02, y=0.90, yanchor="top",
                    xanchor="left", font=dict(size=9.5),
                    title=dict(text="Category")),
    )
    fig.update_yaxes(title_standoff=6)
    export(fig, "dfm_factor_interpretation", width=800, height=510,
           margin=dict(t=32, b=40, l=56, r=118))


def fig_contributions() -> None:
    stacked = [
        (data.load_contributions, "DFM-EN", "dfm_en_postcovid_contributions"),
        (data.load_contributions_blockbalanced, "DFM-BlockBalanced",
         "dfm_blockbalanced_postcovid_contributions"),
    ]
    for loader, label, name in stacked:
        df = loader()
        fig = charts.contributions_stacked(
            df, "2022-01-01", "2025-12-31", model_label=label)
        export(fig, name, width=800, height=380,
               margin=dict(t=25, b=88, l=65, r=25))

    tvp = data.load_contributions_tvp()
    fig = charts.contributions_tvp_bridge(
        tvp, "2022-01-01", "2025-12-31")
    export(fig, "dfm_tvp_postcovid_contributions", width=800, height=500,
           margin=dict(t=12, b=100, l=70, r=25))


def fig_sv_fanchart() -> None:
    df = data.load_nowcast("DFM-SV-k2")
    sub = data.m3_slice(df, True).copy()
    sub["date"] = data.quarter_to_ts(sub["quarter"])
    sub = sub.sort_values("date")
    covered = ((sub["actual"] >= sub["ci_lower_90"])
               & (sub["actual"] <= sub["ci_upper_90"]))
    print(f"SV coverage check: {int(covered.sum())}/{len(sub)} covered; "
          f"mean width {float((sub['ci_upper_90']-sub['ci_lower_90']).mean()):.4f}")
    fig = go.Figure()
    fig.add_vrect(x0="2020-01-01", x1="2021-12-31",
                  fillcolor=C.REGIME_COLORS["COVID"], opacity=0.10,
                  line_width=0, layer="below")
    fig.add_trace(go.Scatter(
        x=sub["date"], y=sub["ci_lower_90"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=sub["date"], y=sub["ci_upper_90"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(103,150,203,0.30)", name="90% interval"))
    fig.add_trace(go.Scatter(
        x=sub["date"], y=sub["nowcast"], mode="lines",
        line=dict(color=C.model_color("DFM-SV-k2"), width=2.0),
        name="SV nowcast (M3)"))
    miss = ~covered
    fig.add_trace(go.Scatter(
        x=sub["date"], y=sub["actual"], mode="markers",
        marker=dict(size=5, color=np.where(miss, "#B3403A", C.INK)),
        name="Realised GDP (first release)"))
    fig.update_layout(
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
    )
    fig.update_yaxes(title_text="GDP growth (pp)")
    export(fig, "nowcast_sv_fanchart", height=330,
           margin=dict(t=20, b=70, l=60, r=20))


def fig_revision_band() -> None:
    df = data.load_revision_path()
    fig = charts.revision_band(df)
    export(fig, "nowcast_revision_band", width=800, height=360,
           margin=dict(t=25, b=80, l=65, r=25))


def fig_gdp_regimes() -> None:
    gdp = data.load_gdp_target()
    g = gdp[gdp["date"] >= "2011-01-01"].copy()
    fig = go.Figure()
    for regime, (q0, q1) in C.REGIMES.items():
        x0 = data.quarter_to_ts(pd.Series([q0]))[0]
        x1 = data.quarter_to_ts(pd.Series([q1]))[0] + pd.offsets.QuarterEnd(0)
        fig.add_vrect(x0=x0, x1=x1, fillcolor=C.REGIME_COLORS[regime],
                      opacity=0.14, line_width=0, layer="below",
                      annotation_text=regime, annotation_position="top left",
                      annotation_font=dict(size=10, color=C.SUBTLE,
                                           family=FONT_FAMILY))
    fig.add_hline(y=0, line=dict(color=C.GRID, width=1))
    fig.add_trace(go.Scatter(
        x=g["date"], y=g["gdp"], mode="lines+markers",
        line=dict(color=C.INK, width=1.8), marker=dict(size=4),
        showlegend=False))
    fig.update_yaxes(title_text="GDP growth (pp, qoq, first release)")
    export(fig, "data_gdp_regimes", height=280,
           margin=dict(t=30, b=45, l=60, r=20))


def main() -> None:
    register_thesis_template()
    fig_publag_composition()
    fig_soft_hard_by_regime()
    fig_agreement_heatmap()
    fig_rmsfe_by_regime()
    fig_postcovid_benchmarks()
    fig_horizon_profiles()
    fig_release_block()
    fig_bias_variance_by_regime()
    fig_mz_forest()
    fig_xgb_sensitivity()
    fig_factor_loadings()
    fig_contributions()
    fig_sv_fanchart()
    fig_revision_band()
    fig_gdp_regimes()
    print("ALL_FIGURES_DONE")


if __name__ == "__main__":
    main()
