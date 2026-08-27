"""Canonical filesystem locations for data and generated pipeline artifacts.

The package defaults to the clean repository layout, with inputs under
``<repo>/data`` and generated files under ``<repo>/outputs``. If those trees
are empty, it also looks one directory up (the original thesis layout:
``<repo>/../Dataset/ifoCAST_DATA.xlsx`` and ``<repo>/../Project_files/``).
Private data can remain elsewhere by setting these environment variables:

``GERMAN_GDP_NOWCASTING_DATA_DIR``
    Directory containing ``panel/``, ``metadata/``, and ``qa/``.
``GERMAN_GDP_NOWCASTING_OUTPUTS_DIR``
    Directory containing stage output subdirectories.
``GERMAN_GDP_NOWCASTING_DATASET_XLSX``
    Path to the source ``ifoCAST_DATA.xlsx`` workbook.
``GERMAN_GDP_NOWCASTING_SUPERVISOR_EXPORTS_DIR``
    Directory containing optional supervisor-review workbooks.
``GERMAN_GDP_NOWCASTING_SUPERVISOR_KEPT_XLSX``
    Optional path to the workbook listing retained indicators.
``GERMAN_GDP_NOWCASTING_SUPERVISOR_DROPPED_XLSX``
    Optional path to the workbook listing excluded indicators.
``GERMAN_GDP_NOWCASTING_THESIS_DIR``
    Optional path to the separate LaTeX thesis repository.
"""

from __future__ import annotations

import os
from pathlib import Path


def _environment_path(name: str, default: Path) -> Path:
    """Return an absolute path from an environment override or a default."""
    raw = os.environ.get(name)
    path = Path(raw).expanduser() if raw else default
    return path.resolve()


def _first_existing(candidates: list[Path], default: Path) -> Path:
    """Return the first path that exists, otherwise ``default``."""
    for path in candidates:
        if path.exists():
            return path.resolve()
    return default.resolve()


def _first_tree(
    default: Path,
    fallbacks: list[Path],
    markers: list[str],
) -> Path:
    """Return the first directory that already contains a marker file."""
    for directory in (default, *fallbacks):
        if any((directory / marker).exists() for marker in markers):
            return directory.resolve()
    return default.resolve()


# paths.py is <repo>/src/german_gdp_nowcasting/config/paths.py.
REPO_ROOT = Path(__file__).resolve().parents[3]

# Sibling trees from the original working copy (not in the public repo).
_LEGACY_ROOT = REPO_ROOT.parent
_LEGACY_PROJECT = _LEGACY_ROOT / "Project_files"
_LEGACY_CURSOR_EXPORTS = _LEGACY_ROOT / ".cursor" / "supervisor_exports"

# --- Structured data (panels, metadata, quality assurance) ---
DATA = _environment_path(
    "GERMAN_GDP_NOWCASTING_DATA_DIR",
    _first_tree(
        REPO_ROOT / "data",
        [_LEGACY_PROJECT / "data"],
        ["panel/data_df.csv", "panel/data_transformed.csv", "metadata/data_dict_catalog.csv"],
    ),
)
DATA_PANEL = DATA / "panel"
DATA_METADATA = DATA / "metadata"
DATA_QA = DATA / "qa"

PANEL_RAW_CSV = DATA_PANEL / "data_df.csv"
PANEL_TRANSFORMED_CSV = DATA_PANEL / "data_transformed.csv"
DATA_DICT_CATALOG_CSV = DATA_METADATA / "data_dict_catalog.csv"
DATA_DICT_ENRICHED_CSV = DATA_METADATA / "data_dict_enriched.csv"
STATIONARITY_REPORT_CSV = DATA_QA / "stationarity_report.csv"
DEDUPLICATION_DECISIONS_CSV = DATA_QA / "deduplication_decisions.csv"
NEAR_DUPLICATE_PAIRS_CSV = DATA_QA / "near_duplicate_pairs.csv"

# --- Publication lag metadata ---
PUB_LAG_CSV = DATA_METADATA / "pub_lag_map.csv"

# --- Stage outputs (not under versioned `data/`) ---
OUTPUTS = _environment_path(
    "GERMAN_GDP_NOWCASTING_OUTPUTS_DIR",
    _first_tree(
        REPO_ROOT / "outputs",
        [_LEGACY_PROJECT / "outputs"],
        [
            "indicator_selection/selection_matrix.csv",
            "nowcasting/nowcast_results_ar1.csv",
            "nowcasting/nowcast_results_actpn_en_only.csv",
        ],
    ),
)
NOTEBOOK_FIGURES = OUTPUTS / "notebooks"
OUT_INDICATOR_SELECTION = OUTPUTS / "indicator_selection"
GDP_TARGET_CSV = OUT_INDICATOR_SELECTION / "gdp_target.csv"
SELECTION_MATRIX_CSV = OUT_INDICATOR_SELECTION / "selection_matrix.csv"
SELECTION_RESULTS_JSON = OUT_INDICATOR_SELECTION / "selection_results.json"
PLS_MATRIX_CSV = OUT_INDICATOR_SELECTION / "selection_matrix_pls.csv"
BLOCKBALANCED_MATRIX_CSV = (
    OUT_INDICATOR_SELECTION / "selection_matrix_blockbalanced_k20.csv"
)
# Fixed-k EN-path benchmark (notebook 03) and frequency-smoothed EN (notebook 04)
FIXEDK_MATRIX_CSV = OUT_INDICATOR_SELECTION / "selection_matrix_fixedk.csv"
FIXEDK_RESULTS_JSON = OUT_INDICATOR_SELECTION / "selection_results_fixedk.json"
EN_SMOOTHED_MATRIX_CSV = (
    OUT_INDICATOR_SELECTION / "selection_matrix_en_smoothed.csv"
)

# --- DFM-ready selection matrices (notebook 05) ---
SELECTION_DIR = OUT_INDICATOR_SELECTION / "dfm_input_sets"
EN_ONLY_MATRIX_CSV = SELECTION_DIR / "en_only_selection_matrix.csv"
CORE_MATRIX_CSV = SELECTION_DIR / "core_selection_matrix.csv"
PLS_ONLY_MATRIX_CSV = SELECTION_DIR / "pls_only_selection_matrix.csv"

OUT_NOWCASTING = OUTPUTS / "nowcasting"
NOWCAST_FIGURES_DIR = OUT_NOWCASTING / "figures"
RMSFE_TABLE_CSV = OUT_NOWCASTING / "rmsfe_table.csv"
DM_TABLE_CSV = OUT_NOWCASTING / "diebold_mariano_table.csv"
AR1_RESULTS_CSV = OUT_NOWCASTING / "nowcast_results_ar1.csv"
# Canonical headline SV model: integrated DFM-SV (k=2), where stochastic
# volatility is fed back into the Kalman smoother via a time-varying
# factor-innovation covariance (see dfm_sv_integrated.py). Point nowcast can
# differ from plain DFM-EN. Produced by scripts/rerun_sv_integrated.py.
ACTPN_SV_RESULTS_K2_CSV = OUT_NOWCASTING / "nowcast_results_actpn_sv_integrated_k2.csv"
RW_RESULTS_CSV = OUT_NOWCASTING / "nowcast_results_rw.csv"
CATEGORY_CONTRIB_CACHE_PARQUET = (
    OUT_NOWCASTING / "category_contribs_en_2017_2025.parquet"
)
SERIES_CONTRIB_CACHE_PARQUET = (
    OUT_NOWCASTING / "series_contribs_en_2017_2025.parquet"
)

# Two-step TVP-DFM track (random-walk factor->GDP loadings, Del Negro & Otrok 2008)
TVP_RESULTS_CSV = OUT_NOWCASTING / "nowcast_results_dfm_tvp.csv"
CATEGORY_CONTRIB_TVP_CACHE_PARQUET = (
    OUT_NOWCASTING / "category_contribs_tvp_2017_2025.parquet"
)
SERIES_CONTRIB_TVP_CACHE_PARQUET = (
    OUT_NOWCASTING / "series_contribs_tvp_2017_2025.parquet"
)
CATEGORY_CONTRIB_BLOCKBALANCED_CACHE_PARQUET = (
    OUT_NOWCASTING / "category_contribs_blockbalanced_2017_2025.parquet"
)
SERIES_CONTRIB_BLOCKBALANCED_CACHE_PARQUET = (
    OUT_NOWCASTING / "series_contribs_blockbalanced_2017_2025.parquet"
)

# ifoCAST benchmark track (fixed expert set + block-balanced k=20)
IFO_RESULTS_CSV = OUT_NOWCASTING / "nowcast_results_dfm_ifocast.csv"
BLOCKBALANCED_RESULTS_CSV = OUT_NOWCASTING / "nowcast_results_dfm_blockbalanced.csv"
COMBO_EQUAL_PATH_CSV = OUT_NOWCASTING / "nowcast_path_combo_equal.csv"

# Ragged-edge diagnostics (per-origin information-set summary CSVs)
RAGGED_EDGE_DIAG_DIR = OUT_NOWCASTING / "ragged_edge_diagnostics"
RAGGED_EDGE_DIAG_CSV = RAGGED_EDGE_DIAG_DIR / "info_set_summary.csv"


def actpn_results_csv(input_set: str) -> Path:
    """Path for the A-CD-TPN nowcast CSV of a given input set ('en_only')."""
    return OUT_NOWCASTING / f"nowcast_results_actpn_{input_set}.csv"


def xgb_results_csv(input_set: str) -> Path:
    """Path for the XGBoost nowcast CSV ('full' is the headline benchmark)."""
    return OUT_NOWCASTING / f"nowcast_results_xgb_{input_set}.csv"


# --- Factor-augmented MLP (non-linear factor->GDP benchmark, notebook 06) ---
MLP_FACTOR_RESULTS_CSV = OUT_NOWCASTING / "nowcast_results_mlp_factor.csv"
MLP_FACTOR_BEST_PARAMS_JSON = OUT_NOWCASTING / "mlp_factor_best_params.json"
MLP_FACTOR_CACHE_PARQUET = OUT_NOWCASTING / "mlp_factor_cache.parquet"


def mlp_results_csv(variant: str = "factor") -> Path:
    """Path for an MLP nowcast CSV ('factor' is the headline benchmark)."""
    if variant == "factor":
        return MLP_FACTOR_RESULTS_CSV
    return OUT_NOWCASTING / f"nowcast_results_mlp_{variant}.csv"


XGB_SHAP_IMPORTANCE_CSV = OUT_NOWCASTING / "xgb_shap_importance.csv"
XGB_BEST_PARAMS_JSON = OUT_NOWCASTING / "xgb_best_params.json"
RMSFE_TABLE_XGB_CSV = OUT_NOWCASTING / "rmsfe_table_with_xgb.csv"
DM_TABLE_XGB_CSV = OUT_NOWCASTING / "diebold_mariano_table_with_xgb.csv"
RMSFE_TABLE_ALL_CSV = OUT_NOWCASTING / "rmsfe_table_all_models.csv"
DM_TABLE_ALL_CSV = OUT_NOWCASTING / "diebold_mariano_table_all_models.csv"
DM_SUBWINDOW_TABLE_CSV = OUT_NOWCASTING / "diebold_mariano_subwindows.csv"
MCS_TABLE_CSV = OUT_NOWCASTING / "model_confidence_set_table.csv"
MINCER_ZARNOWITZ_CSV = OUT_NOWCASTING / "mincer_zarnowitz_table.csv"
FORECAST_REVISION_CSV = OUT_NOWCASTING / "dfm_en_forecast_revision.csv"
FORECAST_REVISION_FIG = NOWCAST_FIGURES_DIR / "thesis_05_dfm_en_revision.png"
SV_INTERVAL_TABLE_CSV = OUT_NOWCASTING / "sv_interval_calibration_table.csv"
RELEASE_BLOCK_RESULTS_CSV = (
    OUT_NOWCASTING / "release_block_counterfactual_results.csv"
)
RELEASE_BLOCK_STATES_CSV = (
    OUT_NOWCASTING / "release_block_counterfactual_states.csv"
)
RELEASE_BLOCK_DECOMPOSITION_CSV = (
    OUT_NOWCASTING / "release_block_counterfactual_mean_decomposition.csv"
)
RELEASE_BLOCK_FIG = NOWCAST_FIGURES_DIR / "release_block_counterfactual.pdf"

# --- Optional thesis exports ---
THESIS_DIR = _environment_path(
    "GERMAN_GDP_NOWCASTING_THESIS_DIR",
    _first_existing(
        [
            REPO_ROOT.parent / "Overleaf-Thesis",
            REPO_ROOT / "Overleaf-Thesis",
        ],
        REPO_ROOT.parent / "Overleaf-Thesis",
    ),
)
THESIS_FIGURES = THESIS_DIR / "figures"

# --- Source Excel (Macrobond / ifo pull) ---
_DATASET_DEFAULT = REPO_ROOT / "Dataset" / "ifoCAST_DATA.xlsx"
DATASET_XLSX = _environment_path(
    "GERMAN_GDP_NOWCASTING_DATASET_XLSX",
    _first_existing(
        [
            REPO_ROOT / "Dataset" / "ifoCAST_DATA.xlsx",
            _LEGACY_ROOT / "Dataset" / "ifoCAST_DATA.xlsx",
        ],
        _DATASET_DEFAULT,
    ),
)

# --- Supervisor review exports ---
SUPERVISOR_EXPORTS = _environment_path(
    "GERMAN_GDP_NOWCASTING_SUPERVISOR_EXPORTS_DIR",
    _first_existing(
        [OUTPUTS / "supervisor_exports", _LEGACY_CURSOR_EXPORTS],
        OUTPUTS / "supervisor_exports",
    ),
)
SUPERVISOR_KEPT_XLSX = _environment_path(
    "GERMAN_GDP_NOWCASTING_SUPERVISOR_KEPT_XLSX",
    _first_existing(
        [
            SUPERVISOR_EXPORTS / "indicators_kept_review.xlsx",
            SUPERVISOR_EXPORTS / "indicators_kept_robert.xlsx",
            _LEGACY_CURSOR_EXPORTS / "indicators_kept_robert.xlsx",
            _LEGACY_CURSOR_EXPORTS / "indicators_kept_review.xlsx",
        ],
        SUPERVISOR_EXPORTS / "indicators_kept_review.xlsx",
    ),
)
SUPERVISOR_DROPPED_XLSX = _environment_path(
    "GERMAN_GDP_NOWCASTING_SUPERVISOR_DROPPED_XLSX",
    _first_existing(
        [
            SUPERVISOR_EXPORTS / "indicators_dropped_review.xlsx",
            SUPERVISOR_EXPORTS / "indicators_dropped_robert.xlsx",
            _LEGACY_CURSOR_EXPORTS / "indicators_dropped_robert.xlsx",
            _LEGACY_CURSOR_EXPORTS / "indicators_dropped_review.xlsx",
        ],
        SUPERVISOR_EXPORTS / "indicators_dropped_review.xlsx",
    ),
)
