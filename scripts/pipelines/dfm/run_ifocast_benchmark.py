"""ifoCAST fixed-indicator benchmark for the thesis DFM.

Motivation
----------
The ifo Institute's ifoCAST (Lehmann, Reif & Wollmershäuser 2020) is a
mixed-frequency dynamic factor model whose indicator set is **fixed**: a
machine-learning screen on a ~300-variable database is condensed *once* into a
panel of 21 indicators (incl. GDP), grouped into seven blocks. This thesis
instead performs **time-varying** real-time indicator selection (Elastic Net,
PLS, a fixed-k "core") with an expanding window.

The supervisor's question: does the time-varying selection improve on a faithful
replication of the ifoCAST **fixed** set, holding the *same* DFM machinery
(r = 2 factors, AR(2) factor dynamics, AR(1) idiosyncratic, ragged-edge masking,
AR(p)-BIC fill) constant?

This script:
  1. Maps the 21 ifoCAST indicators (supervisor's specification, Spec_GDP_2020.xlsx)
     to this dataset's IDs — 20 unique predictors (no deduplication required).
  2. Quantifies overlap with the EN / core real-time selection paths.
  3. Builds a *constant* selection matrix (ifoCAST IDs = 1 at every origin) and
     runs the identical A-CD-TPN DFM loop.
  4. Reports RMSFE (M3 + pooled, full sample + COVID regimes), NSR, and a
     Diebold-Mariano test versus DFM-EN, DFM-core, and AR(1).

Run (from the repository root):
    python scripts/pipelines/dfm/run_ifocast_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

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
from german_gdp_nowcasting.selection.core_utils import (  # noqa: E402
    load_monthly_panel,
    load_pub_lag_map,
)
from german_gdp_nowcasting.models.dfm.nowcast_utils import (  # noqa: E402
    align_forecast_errors,
    compute_nsr,
    compute_rmsfe,
    diebold_mariano_test,
    run_actpn_nowcast_loop,
)

EVAL_START = "2011Q1"
EVAL_END = "2025Q4"
HEADLINE_MIQ = 3
K_FACTORS = 2
FACTOR_ORDER = 2

REGIMES: dict[str, tuple[str, str]] = {
    "pre-COVID": ("2011Q1", "2019Q4"),
    "COVID": ("2020Q1", "2021Q4"),
    "post-COVID": ("2022Q1", "2025Q4"),
}

# ---------------------------------------------------------------------------
# ifoCAST 21-indicator mapping — supervisor's exact specification (Spec_GDP_2020.xlsx).
# `my_id` is the series identifier from the supervisor's panel; `conf` flags
# match quality (exact / close). GDP is the DFM target, not a predictor.
# All 20 predictors are unique — no deduplication required.
# ---------------------------------------------------------------------------
IFOCAST_MAP: list[dict] = [
    {"ifo": "Price-adjusted gross domestic product", "group": "National accounts",
     "my_id": None, "conf": "target",
     "note": "DFM dependent variable (gdp_qoq_log_growth_first_release)."},

    {"ifo": "Production in manufacturing industry", "group": "Production",
     "my_id": "deprod1404", "conf": "close",
     "note": "Industrial Production, Total, CA+SA. "
             "Closest available match to 'Verarbeitendes Gewerbe'; "
             "supervisor-confirmed ID (Spec_GDP_2020.xlsx)."},

    {"ifo": "Sales in manufacturing", "group": "Turnover",
     "my_id": "detrad1853", "conf": "close",
     "note": "Turnover, Manufacturing & Mining excl. Energy, Total, CA+SA. "
             "Supervisor-confirmed ID (Spec_GDP_2020.xlsx)."},

    {"ifo": "Sales in industry (excluding construction)", "group": "Turnover",
     "my_id": "detrad3414", "conf": "exact",
     "note": "Turnover, Manufacturing, Total, CA+SA. "
             "Supervisor-confirmed ID (Spec_GDP_2020.xlsx). "
             "NOTE: this series is NOT present in the thesis panel; "
             "it is dropped before DFM estimation, leaving 19 active predictors."},

    {"ifo": "Sales in the energy sector", "group": "Turnover",
     "my_id": "detrad3364", "conf": "exact",
     "note": "Turnover, Energy, Total, CA+SA."},

    {"ifo": "Sales in retail trade (without car dealership)", "group": "Turnover",
     "my_id": "detrad1360", "conf": "exact",
     "note": "Retail trade turnover, Total, excl. vehicle trade, real."},

    {"ifo": "Sales in the hospitality industry", "group": "Turnover",
     "my_id": "detrad3877", "conf": "exact",
     "note": "Services turnover, accommodation & food services, Total."},

    {"ifo": "Sales in wholesale trade", "group": "Turnover",
     "my_id": "detrad1045", "conf": "exact",
     "note": "Wholesale trade turnover, Total, excl. vehicle trade, real."},

    {"ifo": "Order intake in manufacturing, total", "group": "Orders",
     "my_id": "deprod2112", "conf": "exact",
     "note": "New orders, manufacturing, Total."},

    {"ifo": "Order intake in manufacturing, domestic", "group": "Orders",
     "my_id": "deprod2832", "conf": "exact",
     "note": "New orders, manufacturing, domestic, Total (excl. other vehicle "
             "construction). Supervisor-confirmed ID (Spec_GDP_2020.xlsx); "
             "replaces former deprod2113."},

    {"ifo": "ifo Business Expectations, industry and trade", "group": "Survey",
     "my_id": "desurv0006", "conf": "exact",
     "note": "ifo, Trade & Industry, expectations next 6 months, balance."},

    {"ifo": "ifo Business Expectations, manufacturing", "group": "Survey",
     "my_id": "desurv1146", "conf": "exact",
     "note": "ifo, Manufacturing, expectations next 6 months, balance."},

    {"ifo": "ifo Export Expectations", "group": "Survey",
     "my_id": "desurv1142", "conf": "close",
     "note": "ifo, Manufacturing, export business expectations next 3 months."},

    {"ifo": "ifo order change vs previous month, manufacturing", "group": "Survey",
     "my_id": "desurv1132", "conf": "exact",
     "note": "ifo, Manufacturing, orders on hand vs previous month, balance."},

    {"ifo": "ifo Business Expectations, wholesale", "group": "Survey",
     "my_id": "deifo_g460ong0_ges_bds", "conf": "close",
     "note": "ifo, Wholesale (excl. food/bev/tobacco), expectations next 6m."},

    {"ifo": "ZEW Financial Market Indicator", "group": "Survey",
     "my_id": "desurv0076", "conf": "close",
     "note": "ZEW Financial Market Report, Services, Balance. "
             "Supervisor-confirmed ID (Spec_GDP_2020.xlsx); "
             "nearest panel analogue to the ZEW Indicator of Economic Sentiment."},

    {"ifo": "Notified Vacancies", "group": "Labor market",
     "my_id": "delama1501", "conf": "exact",
     "note": "Labour-turnover, unfilled vacancies, Total, CA+SA (BA)."},

    {"ifo": "Imports (special trade)", "group": "International",
     "my_id": "detrad0689", "conf": "exact",
     "note": "Foreign trade, Total, import, EUR, CA+SA."},

    {"ifo": "Exports (special trade)", "group": "International",
     "my_id": "detrad0692", "conf": "exact",
     "note": "Foreign trade, Total, export, EUR, CA+SA."},

    {"ifo": "Global industrial production", "group": "International",
     "my_id": "worldprod0003", "conf": "exact",
     "note": "CPB World Trade Monitor, industrial production excl. construction."},

    {"ifo": "Global trade volume", "group": "International",
     "my_id": "worldtrad0001", "conf": "exact",
     "note": "CPB World Trade Monitor, total trade volume, SA, index."},
]


def load_data_dict() -> pd.DataFrame:
    """Load enriched indicator metadata indexed by series ID."""
    return pd.read_csv(P.DATA_DICT_ENRICHED_CSV).set_index("id")


def build_mapping_table(panel_cols: set[str], dd: pd.DataFrame) -> pd.DataFrame:
    """Map the published ifoCAST specification to available panel series."""
    rows = []
    for m in IFOCAST_MAP:
        mid = m["my_id"]
        in_panel = mid in panel_cols if mid else None
        rows.append({
            "ifoCAST_indicator": m["ifo"],
            "ifoCAST_group": m["group"],
            "my_id": mid if mid else "(target)",
            "my_name": dd.loc[mid, "name"] if (mid and mid in dd.index) else "",
            "my_category": dd.loc[mid, "category"] if (mid and mid in dd.index) else "",
            "pub_lag": dd.loc[mid, "pub_lag"] if (mid and mid in dd.index) else np.nan,
            "match": m["conf"],
            "in_panel": in_panel,
            "note": m["note"],
        })
    return pd.DataFrame(rows)


def ifocast_predictor_ids() -> list[str]:
    """Unique predictor IDs (GDP target excluded, duplicates removed)."""
    seen: list[str] = []
    for m in IFOCAST_MAP:
        mid = m["my_id"]
        if mid and mid not in seen:
            seen.append(mid)
    return seen


def overlap_with_selection(
    ifo_ids: list[str], matrix_csv: Path, label: str,
) -> pd.DataFrame:
    """Selection frequency of each ifoCAST ID along a real-time selection path."""
    mat = pd.read_csv(matrix_csv, index_col="forecast_origin").astype(int)
    freq = {}
    for cid in ifo_ids:
        freq[cid] = float(mat[cid].mean()) if cid in mat.columns else np.nan
    out = pd.Series(freq, name=f"sel_freq_{label}")
    return out.to_frame()


def jaccard_per_origin(ifo_ids: set[str], matrix_csv: Path) -> pd.Series:
    """Per-origin Jaccard similarity between the ifoCAST set and the selected set."""
    mat = pd.read_csv(matrix_csv, index_col="forecast_origin").astype(int)
    ifo_present = ifo_ids.intersection(mat.columns)
    sims = {}
    for origin, row in mat.iterrows():
        sel = set(mat.columns[row.astype(bool)])
        inter = len(sel & ifo_present)
        union = len(sel | ifo_present)
        sims[origin] = inter / union if union else np.nan
    return pd.Series(sims, name="jaccard")


def build_fixed_selection_matrix(
    ifo_ids: list[str], reference_matrix_csv: Path,
) -> pd.DataFrame:
    """Constant selection matrix: ifoCAST IDs = 1 at every origin in the grid."""
    ref = pd.read_csv(reference_matrix_csv, index_col="forecast_origin")
    cols = [c for c in ifo_ids if c in ref.columns]
    mat = pd.DataFrame(0, index=ref.index, columns=cols, dtype=int)
    mat.loc[:, :] = 1
    return mat


def regime_rmsfe(df: pd.DataFrame) -> dict[str, float]:
    """Compute headline M3 RMSFE within each evaluation regime."""
    out = {}
    for reg, (q0, q1) in REGIMES.items():
        out[reg] = compute_rmsfe(df, eval_start=q0, eval_end=q1, month_in_quarter=HEADLINE_MIQ)
    return out


def main() -> None:
    """Build the ifoCAST mapping, nowcasts, and benchmark evaluation."""
    print("=" * 78)
    print("ifoCAST fixed-indicator benchmark  (DFM: r=2, AR(2) factors)")
    print("=" * 78)

    dd = load_data_dict()
    X_monthly = load_monthly_panel(P.PANEL_TRANSFORMED_CSV)
    pub_lag_map = load_pub_lag_map(P.PUB_LAG_CSV)
    y_q = pd.read_csv(P.GDP_TARGET_CSV, index_col="quarter").squeeze("columns")
    y_q.index = pd.PeriodIndex(y_q.index, freq="Q")

    panel_cols = set(X_monthly.columns)

    # --- 1. Mapping table -------------------------------------------------
    map_tbl = build_mapping_table(panel_cols, dd)
    print("\n[1] ifoCAST -> thesis-dataset mapping")
    print(map_tbl.to_string(index=False, max_colwidth=46))
    out_dir = P.OUT_NOWCASTING
    out_dir.mkdir(parents=True, exist_ok=True)
    map_tbl.to_csv(out_dir / "ifocast_indicator_mapping.csv", index=False)

    ifo_ids = ifocast_predictor_ids()
    missing = [c for c in ifo_ids if c not in panel_cols]
    if missing:
        print(f"\n  WARNING: not in panel -> {missing}")
    ifo_ids = [c for c in ifo_ids if c in panel_cols]
    print(f"\n  Unique ifoCAST predictors available in panel: {len(ifo_ids)}")

    # --- 2. Overlap with EN / core selections -----------------------------
    print("\n[2] Selection frequency of ifoCAST IDs along the real-time path")
    en_freq = overlap_with_selection(ifo_ids, P.EN_ONLY_MATRIX_CSV, "EN")
    core_freq = overlap_with_selection(ifo_ids, P.CORE_MATRIX_CSV, "core")
    freq_tbl = en_freq.join(core_freq)
    freq_tbl["name"] = [dd.loc[i, "name"][:40] if i in dd.index else "" for i in freq_tbl.index]
    print(freq_tbl.round(3).to_string())
    freq_tbl.to_csv(out_dir / "ifocast_selection_frequency.csv")

    jac_en = jaccard_per_origin(set(ifo_ids), P.EN_ONLY_MATRIX_CSV)
    jac_core = jaccard_per_origin(set(ifo_ids), P.CORE_MATRIX_CSV)
    print(f"\n  Mean per-origin Jaccard(ifoCAST, EN)   = {jac_en.mean():.3f}")
    print(f"  Mean per-origin Jaccard(ifoCAST, core) = {jac_core.mean():.3f}")
    print(f"  ifoCAST IDs ever picked by EN   : "
          f"{int((en_freq['sel_freq_EN'] > 0).sum())}/{len(ifo_ids)}")
    print(f"  ifoCAST IDs ever picked by core : "
          f"{int((core_freq['sel_freq_core'] > 0).sum())}/{len(ifo_ids)}")

    # --- 3. DFM on the fixed ifoCAST set ----------------------------------
    print("\n[3] Running DFM on the fixed ifoCAST set (this takes a few minutes) ...")
    fixed_mat = build_fixed_selection_matrix(ifo_ids, P.CORE_MATRIX_CSV)
    quarterly_origins = pd.period_range(EVAL_START, EVAL_END, freq="Q")
    df_ifo = run_actpn_nowcast_loop(
        selection_matrix=fixed_mat,
        X_monthly=X_monthly,
        y_quarterly=y_q,
        quarterly_origins=quarterly_origins,
        k_factors=K_FACTORS,
        factor_order=FACTOR_ORDER,
        idiosyncratic_ar1=True,
        maxiter=200,
        pub_lag_map=pub_lag_map,
        fill_method="ar_bic",
        verbose=False,
    )
    df_ifo.to_csv(out_dir / "nowcast_results_dfm_ifocast.csv")

    # --- 4. Evaluation vs EN / core / AR1 ---------------------------------
    print("\n[4] Evaluation")

    def _load(path: Path) -> pd.DataFrame:
        """Load a saved result and restore its monthly-origin index."""
        d = pd.read_csv(path)
        if "monthly_origin" in d.columns:
            d = d.set_index("monthly_origin", drop=False)
        return d

    df_en = _load(P.actpn_results_csv("en_only"))
    df_core = _load(P.actpn_results_csv("core"))
    df_ar1 = _load(P.AR1_RESULTS_CSV)
    if "month_in_quarter" not in df_ar1.columns:
        from german_gdp_nowcasting.models.dfm.nowcast_utils import (
            expand_quarterly_nowcasts_to_monthly,
        )
        df_ar1 = expand_quarterly_nowcasts_to_monthly(df_ar1)

    models = {
        "DFM-ifoCAST(fixed)": df_ifo,
        "DFM-EN(time-varying)": df_en,
        "DFM-core(time-varying)": df_core,
        "AR1": df_ar1,
    }

    rows = []
    for name, d in models.items():
        r3 = compute_rmsfe(d, EVAL_START, EVAL_END, month_in_quarter=HEADLINE_MIQ)
        rp = compute_rmsfe(d, EVAL_START, EVAL_END, month_in_quarter=None)
        nsr = compute_nsr(d, y_q, EVAL_START, EVAL_END)
        reg = regime_rmsfe(d)
        rows.append({
            "model": name, "RMSFE_M3": r3, "RMSFE_pooled": rp, "NSR": nsr,
            **{f"RMSFE_{k}": v for k, v in reg.items()},
        })
    eval_tbl = pd.DataFrame(rows).set_index("model").round(4)
    print("\n" + eval_tbl.to_string())
    eval_tbl.to_csv(out_dir / "ifocast_benchmark_rmsfe.csv")

    # Diebold-Mariano: ifoCAST vs each competitor (M3, full sample)
    print("\n  Diebold-Mariano (ifoCAST vs X; negative DM => ifoCAST better):")
    for comp in ("DFM-EN(time-varying)", "DFM-core(time-varying)", "AR1"):
        ea, eb = align_forecast_errors(df_ifo, models[comp], month_in_quarter=HEADLINE_MIQ,
                                       eval_start=EVAL_START, eval_end=EVAL_END)
        dm = diebold_mariano_test(ea, eb)
        print(f"    vs {comp:24s}: DM={dm['DM']:+.3f}  p={dm['p_value']:.3f}  n={dm['n']}")

    print("\nArtefacts written to outputs/nowcasting/:")
    for f in ("ifocast_indicator_mapping.csv", "ifocast_selection_frequency.csv",
              "nowcast_results_dfm_ifocast.csv", "ifocast_benchmark_rmsfe.csv"):
        print(f"  - {f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
