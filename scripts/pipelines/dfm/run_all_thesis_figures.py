"""Regenerate every thesis figure under outputs/nowcasting/figures/.

Reads saved nowcast CSVs and selection matrices — no model re-runs except
category-contribution cache when ``--rebuild-contrib`` is passed.

Run (from the repository root):
    python scripts/pipelines/dfm/run_all_thesis_figures.py
    python scripts/pipelines/dfm/run_all_thesis_figures.py --rebuild-contrib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
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
from german_gdp_nowcasting.models.dfm import nowcast_utils as nu  # noqa: E402
from german_gdp_nowcasting.visualization import xgb_plots as xp  # noqa: E402

FIG = P.NOWCAST_FIGURES_DIR

EVAL_START, EVAL_END = "2011Q1", "2025Q4"
HEADLINE_MIQ = 3

REGIMES = {
    "pre-COVID": ("2011Q1", "2019Q4"),
    "COVID": ("2020Q1", "2021Q4"),
    "post-COVID": ("2022Q1", "2025Q4"),
}

# Headline model keys for comparison figures
HEADLINE_MODELS: dict[str, Path] = {
    "RW": P.RW_RESULTS_CSV,
    "AR1": P.AR1_RESULTS_CSV,
    "DFM-ifoCAST": P.IFO_RESULTS_CSV,
    "DFM-EN": P.actpn_results_csv("en_only"),
    "DFM-BlockBalanced": P.BLOCKBALANCED_RESULTS_CSV,
    "DFM-SV-k2": P.ACTPN_SV_RESULTS_K2_CSV,
    "combo_equal": P.COMBO_EQUAL_PATH_CSV,
    "XGB-Full": P.xgb_results_csv("full"),
    "MLP-Factor": P.MLP_FACTOR_RESULTS_CSV,
}


def _load(path: Path) -> pd.DataFrame:
    """Load a result CSV and normalize it to monthly-origin rows."""
    df = pd.read_csv(path)
    if "month_in_quarter" not in df.columns:
        df = nu.expand_quarterly_nowcasts_to_monthly(df)
    return df


def load_results() -> dict[str, pd.DataFrame]:
    """Load all available headline model result files."""
    out: dict[str, pd.DataFrame] = {}
    for name, path in HEADLINE_MODELS.items():
        if path.exists():
            out[name] = _load(path)
            print(f"  loaded {name}")
        else:
            print(f"  [skip] {name}: {path.name} missing")
    return out


def fig_composition(save: Path) -> None:
    """EN-only vs block-balanced economic composition."""
    meta = pd.read_csv(P.DATA_DICT_ENRICHED_CSV, usecols=["id", "category"]).set_index("id")
    matrices: dict[str, pd.DataFrame] = {}
    for key, path in [
        ("en_only", P.EN_ONLY_MATRIX_CSV),
        ("blockbalanced", P.BLOCKBALANCED_MATRIX_CSV),
    ]:
        if path.exists():
            matrices[key] = pd.read_csv(path, index_col=0)
    if matrices:
        npl.fig_composition_heatmap(matrices, meta, save=save)
        print(f"  saved {save.name}")


def fig_dfm_rmsfe(results: dict[str, pd.DataFrame], save: Path) -> None:
    """Write the headline regime-specific RMSFE comparison."""
    xp.setup_style()
    order = [m for m in xp.DEFAULT_MODEL_ORDER if m in results]
    order += [m for m in results if m not in order]
    fig = xp.fig_rmsfe_by_regime(
        results,
        regimes=REGIMES,
        month_in_quarter=HEADLINE_MIQ,
        models=order,
        title="RMSFE by regime — headline models (M3)",
        save=save,
    )
    if fig:
        print(f"  saved {save.name}")


def fig_sv_plots(sv_df: pd.DataFrame) -> None:
    """Write volatility, fan, and interval-calibration figures."""
    npl.fig_sv_volatility(sv_df, save=FIG / "06_sv_volatility.png")
    npl.fig_nowcast_fan(sv_df, credibility=0.90, save=FIG / "11_nowcast_fan_sv90.png")
    npl.fig_sv_calibration(
        sv_df, model_label="DFM-SV-k2 (integrated, EN inputs)",
        save=FIG / "09_sv_comparison.png",
    )
    print("  saved 06_sv_volatility, 11_nowcast_fan_sv90, 09_sv_comparison")


def fig_category_contrib(rebuild: bool) -> None:
    """Build or load DFM-EN contribution caches and figures."""
    from german_gdp_nowcasting.selection.core_utils import (
        load_monthly_panel,
        load_pub_lag_map,
    )

    X_monthly = load_monthly_panel(P.PANEL_TRANSFORMED_CSV)
    pub_lag_map = load_pub_lag_map(P.PUB_LAG_CSV)
    y_q = pd.read_csv(P.GDP_TARGET_CSV, index_col="quarter").squeeze("columns")
    y_q.index = pd.PeriodIndex(y_q.index, freq="Q")
    sel = pd.read_csv(P.EN_ONLY_MATRIX_CSV, index_col="forecast_origin").astype(int)
    meta = pd.read_csv(P.DATA_DICT_ENRICHED_CSV, usecols=["id", "category"]).set_index("id")
    origins = pd.period_range(EVAL_START, EVAL_END, freq="Q")

    df = npl.run_category_contrib_panel(
        selection_matrix=sel,
        X_monthly=X_monthly,
        y_quarterly=y_q,
        meta=meta,
        quarterly_origins=origins,
        pub_lag_map=pub_lag_map,
        k_factors=2,
        factor_order=2,
        m_start="2017-01",
        m_end="2025-12",
        cache_path=P.CATEGORY_CONTRIB_CACHE_PARQUET,
        series_cache_path=P.SERIES_CONTRIB_CACHE_PARQUET,
        force_rerun=rebuild,
        verbose=True,
    )
    base_title = "DFM-EN: category contributions by forecast origin"
    periods = [
        ("2017-01", "2019-12", "2017_2019"),
        ("2020-01", "2021-12", "2020_2021"),
        ("2022-01", "2023-12", "2022_2023"),
        ("2024-01", "2025-12", "2024_2025"),
    ]
    npl.fig_category_contrib_period_panels(
        df, periods=periods, fig_dir=FIG, base_title=base_title,
    )
    # Full-window and COVID-only panels (legacy filenames used in thesis drafts)
    for m_start, m_end, fname in [
        ("2017-01", "2025-12", "08b_category_contrib_interactive.html"),
        ("2020-01", "2021-12", "08c_category_contrib_covid_interactive.html"),
    ]:
        contribs, nowcasts, actuals, hover_meta = npl.contrib_cache_to_plot_dicts(
            df, m_start, m_end,
        )
        if contribs:
            npl.fig_category_contrib_interactive(
                contributions_by_quarter=contribs,
                actuals=actuals,
                nowcasts=nowcasts,
                title=f"{base_title} ({m_start} – {m_end})",
                save_html=FIG / fname,
                xaxis_title="Forecast origin (month)",
                hover_meta=hover_meta,
            )
    print("  saved category contrib panels (08b, 08c, 08d*)")


def fig_category_contrib_tvp(rebuild: bool) -> None:
    """Build the DFM-TVP category-contribution cache (factor-bridge attribution).

    Uses the *same* EN selection matrix and information set as the DFM-EN
    decomposition, but attributes the TVP nowcast. Produces a cache that the
    dashboard renders with the identical stacked-bar design.
    """
    from german_gdp_nowcasting.selection.core_utils import (
        load_monthly_panel,
        load_pub_lag_map,
    )
    from german_gdp_nowcasting.models.dfm import tvp_dfm as tvp

    X_monthly = load_monthly_panel(P.PANEL_TRANSFORMED_CSV)
    pub_lag_map = load_pub_lag_map(P.PUB_LAG_CSV)
    y_q = pd.read_csv(P.GDP_TARGET_CSV, index_col="quarter").squeeze("columns")
    y_q.index = pd.PeriodIndex(y_q.index, freq="Q")
    sel = pd.read_csv(P.EN_ONLY_MATRIX_CSV, index_col="forecast_origin").astype(int)
    meta = pd.read_csv(P.DATA_DICT_ENRICHED_CSV, usecols=["id", "category"]).set_index("id")
    origins = pd.period_range(EVAL_START, EVAL_END, freq="Q")

    tvp.run_category_contrib_panel_tvp(
        selection_matrix=sel,
        X_monthly=X_monthly,
        y_quarterly=y_q,
        meta=meta,
        quarterly_origins=origins,
        pub_lag_map=pub_lag_map,
        k_factors=2,
        factor_order=2,
        m_start="2017-01",
        m_end="2025-12",
        cache_path=P.CATEGORY_CONTRIB_TVP_CACHE_PARQUET,
        series_cache_path=P.SERIES_CONTRIB_TVP_CACHE_PARQUET,
        force_rerun=rebuild,
        verbose=True,
    )
    print("  saved TVP category contrib cache "
          f"({P.CATEGORY_CONTRIB_TVP_CACHE_PARQUET.name})")


def fig_category_contrib_blockbalanced(rebuild: bool) -> None:
    """Build the DFM-BlockBalanced category-contribution cache.

    Same DynamicFactorMQ machinery and ifoCAST-style predict()-based
    attribution as DFM-EN (``_contrib_frame``), just on the block-balanced
    k=20 selection matrix, with the same divergence-guard fallback ladder
    used by the headline block-balanced nowcast (no intercept/baseline row).
    """
    from german_gdp_nowcasting.selection.core_utils import (
        load_monthly_panel,
        load_pub_lag_map,
    )

    X_monthly = load_monthly_panel(P.PANEL_TRANSFORMED_CSV)
    pub_lag_map = load_pub_lag_map(P.PUB_LAG_CSV)
    y_q = pd.read_csv(P.GDP_TARGET_CSV, index_col="quarter").squeeze("columns")
    y_q.index = pd.PeriodIndex(y_q.index, freq="Q")
    sel = pd.read_csv(P.BLOCKBALANCED_MATRIX_CSV, index_col="forecast_origin").astype(int)
    meta = pd.read_csv(P.DATA_DICT_ENRICHED_CSV, usecols=["id", "category"]).set_index("id")
    origins = pd.period_range(EVAL_START, EVAL_END, freq="Q")

    npl.run_category_contrib_panel_guarded(
        selection_matrix=sel,
        X_monthly=X_monthly,
        y_quarterly=y_q,
        meta=meta,
        quarterly_origins=origins,
        pub_lag_map=pub_lag_map,
        factor_order=2,
        m_start="2017-01",
        m_end="2025-12",
        cache_path=P.CATEGORY_CONTRIB_BLOCKBALANCED_CACHE_PARQUET,
        series_cache_path=P.SERIES_CONTRIB_BLOCKBALANCED_CACHE_PARQUET,
        force_rerun=rebuild,
        verbose=True,
    )
    print("  saved BlockBalanced category contrib cache "
          f"({P.CATEGORY_CONTRIB_BLOCKBALANCED_CACHE_PARQUET.name})")


def fig_xgb() -> None:
    """Write XGBoost SHAP importance figures when data are available."""
    shap_path = P.XGB_SHAP_IMPORTANCE_CSV
    if not shap_path.exists():
        print("  [skip] XGB SHAP — file missing")
        return
    shap = pd.read_csv(shap_path)
    if shap.empty:
        return
    xp.setup_style()
    fig = xp.fig_shap_bar(shap, top_n=15, aggregate_lags=True)
    xp.save_fig(fig, FIG / "xgb_05_shap_bar.png")
    plt.close(fig)
    dd = pd.read_csv(P.DATA_DICT_ENRICHED_CSV)
    cat_map = dd.set_index("id")["category"]
    xp.fig_xgb_shap_category_interactive(
        shap, cat_map,
        save_html=FIG / "xgb_shap_category_interactive.html",
        show=False,
    )
    print("  saved xgb_05_shap_bar, xgb_shap_category_interactive.html")


def fig_xgb_nowcast_compare(results: dict[str, pd.DataFrame]) -> None:
    """Write the interactive XGBoost-versus-DFM comparison."""
    if "XGB-Full" not in results:
        return
    xp.fig_xgb_nowcast_interactive(
        results["XGB-Full"],
        dfm_df=results.get("DFM-EN"),
        xgb_name="XGB-Full",
        dfm_name="DFM-EN",
        save_html=FIG / "xgb_nowcast_compare_interactive.html",
        show=False,
    )
    print("  saved xgb_nowcast_compare_interactive.html")


def fig_mlp_nowcast_compare(results: dict[str, pd.DataFrame]) -> None:
    """All-headline-models RMSFE-by-regime bar chart (includes MLP-Factor)."""
    xp.setup_style()
    order = [m for m in HEADLINE_MODELS if m in results]
    fig = xp.fig_rmsfe_by_regime(
        results,
        regimes=REGIMES,
        month_in_quarter=HEADLINE_MIQ,
        models=order,
        title="RMSFE by regime — all headline models",
        save=FIG / "10b_rmsfe_all_models.png",
    )
    if fig:
        print("  saved 10b_rmsfe_all_models.png")


def main() -> None:
    """Regenerate the complete thesis figure suite."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-contrib", action="store_true",
                        help="Force rebuild category-contribution cache")
    parser.add_argument("--rebuild-contrib-tvp", action="store_true",
                        help="Force rebuild DFM-TVP category-contribution cache")
    parser.add_argument("--rebuild-contrib-bb", action="store_true",
                        help="Force rebuild DFM-BlockBalanced category-contribution cache")
    args = parser.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    npl.setup_style()
    print("=== Thesis figure regeneration ===\n")
    results = load_results()

    print("\n[1] Composition heatmap")
    fig_composition(FIG / "05_composition_heatmap.png")

    print("\n[2] DFM RMSFE by regime")
    fig_dfm_rmsfe(results, FIG / "03_rmsfe_by_regime_headline.png")

    if "DFM-SV-k2" in results:
        print("\n[3] SV figures")
        fig_sv_plots(results["DFM-SV-k2"])

    print("\n[4] Category contributions")
    fig_category_contrib(rebuild=args.rebuild_contrib)

    print("\n[4b] DFM-TVP category contributions")
    fig_category_contrib_tvp(rebuild=args.rebuild_contrib_tvp)

    print("\n[4c] DFM-BlockBalanced category contributions")
    fig_category_contrib_blockbalanced(rebuild=args.rebuild_contrib_bb)

    print("\n[5] XGBoost figures")
    fig_xgb()
    fig_xgb_nowcast_compare(results)

    print("\n[6] All-models comparison figure")
    fig_mlp_nowcast_compare(results)

    print("\n[7] Evaluation scripts (thesis_01–05)")
    import build_unified_evaluation  # noqa: E402
    import run_horizon_profile  # noqa: E402
    import run_post_covid_benchmarks  # noqa: E402
    build_unified_evaluation.main()
    run_horizon_profile.build_table().to_csv(P.OUT_NOWCASTING / "horizon_profile_table.csv", index=False)
    run_horizon_profile.make_figure(
        pd.read_csv(P.OUT_NOWCASTING / "horizon_profile_table.csv"),
        FIG / "thesis_04_horizon_profile.png",
    )
    models, y = run_post_covid_benchmarks.build_models()
    tbl = run_post_covid_benchmarks.results_table(models, y)
    tbl.to_csv(P.OUT_NOWCASTING / "post_covid_benchmarks_table.csv")
    run_post_covid_benchmarks.make_figure(tbl, FIG / "thesis_03_post_covid_improvement.png")

    from german_gdp_nowcasting.visualization import mlp_plots as pts
    pts_results = pts.load_all()
    pts.plot_clean_comparison(pts_results, save=FIG / "thesis_01_rmsfe_by_regime_clean.png")
    pts.plot_mlp_linearity_diagnostic(pts_results, save=FIG / "thesis_02_mlp_linearity_diagnostic.png")

    print("\n[8] Part I selection comparison figures")
    from german_gdp_nowcasting.selection import selection_comparison
    selection_comparison.main()

    print("\n[9] Factor loading interpretation figures")
    import run_factor_loading_figure  # noqa: E402
    run_factor_loading_figure.main()

    print(f"\nDone. Figures in {FIG}")


if __name__ == "__main__":
    main()
