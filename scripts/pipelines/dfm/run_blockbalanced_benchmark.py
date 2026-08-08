"""Block-balanced targeted Elastic Net (k=20) — a parsimonious, structured rival
to the fixed ifoCAST set and the unconstrained EN.

Motivation
----------
The fixed ifoCAST set (19 indicators, balanced across 7 groups, survey-heavy)
matched or beat the unconstrained time-varying EN (~53 indicators, hard-data
heavy) in run_ifocast_benchmark.py. Forcing ifoCAST's surveys onto the full EN
set helped only marginally — the decisive factor was **parsimony + structure**,
not the surveys per se.

This model operationalizes that lesson while *keeping selection data-driven and
time-varying*:

  At each expanding-window origin,
    1. fit the Elastic Net (CV-tuned) on all coverage-passing candidates;
    2. rank candidates by |coefficient| (primary) and marginal |t-stat|
       (secondary, Bai & Ng 2008) — the latter breaks ties and represents
       categories the EN shrank to exactly zero;
    3. select a **block-balanced top-k=20**: guarantee >=1 indicator from every
       economic category that has a candidate (mirrors ifoCAST's "7 groups"),
       then fill the remaining slots by |coefficient|.

Because category balance forces breadth, the Commodities block (OPEC/Brent oil)
and the Global block enter whenever they carry signal — economically motivated
by the energy-price channel emphasised in the ifo Spring 2026 forecast.

The DFM machinery is held identical to every other model: r=2 factors, AR(2)
factor dynamics, AR(1) idiosyncratic, ragged-edge pub-lag masking, AR(p)-BIC
fill.

Run (from the repository root):
    python scripts/pipelines/dfm/run_blockbalanced_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

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
    align_quarterly_xy,
    build_coverage_mask,
    load_monthly_panel,
    load_pub_lag_map,
    monthly_to_quarterly,
    training_end_quarter,
)
from german_gdp_nowcasting.selection.elastic_net_selection import (  # noqa: E402
    ts_elastic_net,
)
from german_gdp_nowcasting.models.dfm.nowcast_utils import (  # noqa: E402
    align_forecast_errors,
    compute_nsr,
    compute_rmsfe,
    diebold_mariano_test,
    expand_quarterly_nowcasts_to_monthly,
    nowcast_for_origin,
)
import scipy.stats as stats  # noqa: E402

#: Plausibility bound for a single nowcast (pp QoQ log-growth). German quarterly
#: GDP growth never left [-11, +9] even in 2020; |nowcast| beyond this signals a
#: diverged EM optimum, not a forecast.
NOWCAST_CAP = 20.0

EVAL_START, EVAL_END, MIQ = "2011Q1", "2025Q4", 3
K_TOTAL = 20            # target indicators per origin (mirror ifoCAST's ~20 predictors)
MIN_PER_GROUP = 1       # floor per economic category (structural breadth)
TRAIN_START = "1991Q1"
REGIMES = {"pre-COVID": ("2011Q1", "2019Q4"),
           "COVID": ("2020Q1", "2021Q4"),
           "post-COVID": ("2022Q1", "2025Q4")}


def marginal_tstats(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """|OLS t-stat| of each column against y (mean-imputed), for tie-breaking."""
    valid = y.notna()
    yv = y.loc[valid].to_numpy(float)
    Xv = X.loc[valid].fillna(X.loc[valid].mean())
    out = {}
    for c in Xv.columns:
        x = Xv[c].to_numpy(float)
        if x.std() == 0:
            out[c] = 0.0
            continue
        slope, _i, _r, _p, se = stats.linregress(x, yv)
        out[c] = abs(slope / se) if se and se > 0 else 0.0
    return pd.Series(out)


def block_balanced_pick(
    coef_abs: pd.Series,
    tstat: pd.Series,
    group_map: pd.Series,
    k_total: int = K_TOTAL,
    min_per_group: int = MIN_PER_GROUP,
) -> list[str]:
    """Pick a parsimonious, block-balanced top-k indicator set.

    Balance is imposed **only over categories the Elastic Net deems relevant**
    (non-zero coefficient). Forcing a representative from an EN-zeroed category
    injects series the model should not lean on — during 2022-24 this pulled in
    energy-crisis-driven Prices/Commodities series whose extreme moves
    destabilised the DFM Kalman smoother (explosive post-COVID nowcasts).

    Procedure:
      1. Pool = EN-non-zero candidates (ranked by |coefficient|).
      2. Group floor: >=``min_per_group`` from each category *present in the
         pool*; never forces an EN-rejected category.
      3. Fill remaining slots by |coefficient| within the pool.
      4. If the pool has < k_total members, top up with the highest marginal
         |t-stat| candidates (Bai & Ng 2008 targeted predictors) — a relevance
         measure, not a forced-category injection.
    """
    nonzero = coef_abs[coef_abs > 1e-10].sort_values(ascending=False)
    pool = nonzero.index.tolist()
    grp = group_map.reindex(pool).fillna("Unknown")
    selected: list[str] = []

    # 1-2. group floor over EN-relevant categories only
    for g in grp.unique():
        members = nonzero.loc[grp == g]
        for sid in members.index[:min_per_group]:
            if sid not in selected:
                selected.append(sid)

    # 3. fill remaining by |coefficient| within the EN-relevant pool
    for sid in pool:
        if len(selected) >= k_total:
            break
        if sid not in selected:
            selected.append(sid)

    # 4. if EN was too sparse, top up by marginal |t-stat| (targeted predictors)
    if len(selected) < k_total:
        rest = tstat.drop(index=selected, errors="ignore").sort_values(ascending=False)
        for sid in rest.index:
            if len(selected) >= k_total:
                break
            selected.append(sid)

    if len(selected) > k_total:
        selected = coef_abs.reindex(selected).sort_values(
            ascending=False).index[:k_total].tolist()
    return selected


def guarded_nowcast(
    X_monthly: pd.DataFrame,
    y_q: pd.Series,
    sel: list[str],
    origin: pd.Period | str,
    pub_lag: pd.Series,
) -> tuple[float, str]:
    """Single-origin nowcast with a divergence-guard fallback ladder.

    A small (k=20) DFM occasionally lands on an explosive EM optimum at
    ill-conditioned post-2022 windows, returning a non-finite or absurd value.
    We retry with progressively more robust state-space structure (drop
    idiosyncratic AR(1), then reduce to one factor) and accept the first finite
    nowcast within the plausibility bound. Mirrors the spirit of the existing
    ``fit_dfm`` fallback ladder, extended to converged-but-explosive output.
    """
    ladder = [
        dict(k_factors=2, idiosyncratic_ar1=True),
        dict(k_factors=2, idiosyncratic_ar1=False),
        dict(k_factors=1, idiosyncratic_ar1=True),
        dict(k_factors=1, idiosyncratic_ar1=False),
    ]
    for i, cfg in enumerate(ladder):
        try:
            r = nowcast_for_origin(
                X_monthly=X_monthly, y_quarterly=y_q, selected_cols=sel,
                origin=origin, factor_order=2, maxiter=200,
                pub_lag_map=pub_lag, fill_method="ar_bic", **cfg,
            )
            nc = r["nowcast"]
            if np.isfinite(nc) and abs(nc) <= NOWCAST_CAP:
                return float(nc), ("base" if i == 0 else f"fallback{i}")
        except Exception:
            continue
    return np.nan, "failed"


def guarded_nowcast_loop(
    mat: pd.DataFrame,
    X_monthly: pd.DataFrame,
    y_q: pd.Series,
    pub_lag: pd.Series,
    origins_q: Iterable[pd.Period | str],
) -> pd.DataFrame:
    """A-CD-TPN loop (M1-M3) using the divergence-guarded single-origin nowcast."""
    rec = []
    for q in origins_q:
        q = pd.Period(q, freq="Q")
        actual = float(y_q.get(q, np.nan))
        q_m1 = q.asfreq("M", how="start")
        for m in (1, 2, 3):
            op = q_m1 + (m - 1)
            ok = str(op)
            if ok not in mat.index:
                continue
            sel = mat.columns[mat.loc[ok].astype(bool)].tolist()
            nc, tag = guarded_nowcast(X_monthly, y_q, sel, op, pub_lag)
            rec.append({
                "quarter": str(q), "monthly_origin": ok, "month_in_quarter": m,
                "n_indicators": len(sel), "fit_tag": tag, "nowcast": nc,
                "actual": actual,
                "error": nc - actual if np.isfinite(nc) and not np.isnan(actual) else np.nan,
            })
    return pd.DataFrame(rec).set_index("monthly_origin")


def run_selection(
    X_monthly: pd.DataFrame,
    y_q: pd.Series,
    trafo_map: pd.Series,
    group_map: pd.Series,
    origins: Iterable[pd.Period | str],
) -> pd.DataFrame:
    """Build the expanding block-balanced Elastic Net selection path."""
    Xq = monthly_to_quarterly(X_monthly, trafo_map)
    Xq, y_q = align_quarterly_xy(Xq, y_q)
    cov = build_coverage_mask(X_monthly, [pd.Period(o) for o in origins])
    train_start = pd.Period(TRAIN_START, freq="Q")

    rows: dict[str, pd.Series] = {}
    cache: dict[str, list[str]] = {}
    for origin in origins:
        ok = str(origin)
        end_q = training_end_quarter(origin)
        ck = str(end_q)
        if ck in cache:
            sel = cache[ck]
        else:
            tr_idx = Xq.index[(Xq.index >= train_start) & (Xq.index <= end_q)]
            y_tr_full = y_q.reindex(tr_idx)
            valid = y_tr_full.index[y_tr_full.notna()]
            cands = cov.columns[cov.loc[ok]].tolist()
            Xtr = Xq.loc[valid, cands]
            Xtr = Xtr.loc[:, Xtr.notna().any()]
            ytr = y_q.loc[valid]
            fit = ts_elastic_net(Xtr, ytr)
            coef_abs = fit.coefficients.abs()
            tstat = marginal_tstats(Xtr, ytr)
            sel = block_balanced_pick(coef_abs, tstat, group_map)
            cache[ck] = sel
        r = pd.Series(0, index=X_monthly.columns, dtype=int)
        r.loc[[s for s in sel if s in r.index]] = 1
        rows[ok] = r
    mat = pd.DataFrame.from_dict(rows, orient="index").astype(int)
    mat.index.name = "forecast_origin"
    return mat


def main() -> None:
    """Run selection, guarded DFM nowcasts, and benchmark evaluation."""
    print("Block-balanced targeted EN (k=20, >=1/category) — DFM r=2, AR(2)\n")
    X = load_monthly_panel(P.PANEL_TRANSFORMED_CSV)
    pub_lag = load_pub_lag_map(P.PUB_LAG_CSV)
    y = pd.read_csv(P.GDP_TARGET_CSV, index_col="quarter").squeeze("columns")
    y.index = pd.PeriodIndex(y.index, freq="Q")
    dd = pd.read_csv(P.DATA_DICT_ENRICHED_CSV)
    trafo_map = pd.to_numeric(dd.set_index("id")["trafo_applied"], errors="raise")
    group_map = dd.set_index("id")["category"]

    # EN selection matrix already defines the monthly origin grid; reuse it.
    en = pd.read_csv(P.EN_ONLY_MATRIX_CSV, index_col="forecast_origin")
    origins = pd.PeriodIndex(en.index, freq="M")

    print("[1] Running expanding block-balanced selection ...")
    mat = run_selection(X, y, trafo_map, group_map, origins)
    mat.to_csv(P.OUT_INDICATOR_SELECTION / "selection_matrix_blockbalanced_k20.csv")
    # category composition diagnostic
    comp = {}
    for ok, row in mat.iterrows():
        for sid in mat.columns[row.astype(bool)]:
            g = group_map.get(sid, "Unknown")
            comp[g] = comp.get(g, 0) + 1
    comp = pd.Series(comp).sort_values(ascending=False)
    print(f"  mean N per origin = {mat.sum(axis=1).mean():.1f}")
    print("  category share of selections (all origins):")
    print((comp / comp.sum()).round(3).to_string())

    print("\n[2] Running DFM on the block-balanced set (divergence-guarded) ...")
    df_bb = guarded_nowcast_loop(
        mat, X, y, pub_lag, pd.period_range(EVAL_START, EVAL_END, freq="Q"),
    )
    df_bb.to_csv(P.OUT_NOWCASTING / "nowcast_results_dfm_blockbalanced.csv")
    n_fb = int((df_bb["fit_tag"].astype(str) != "base").sum())
    print(f"  origins needing fallback/guard: {n_fb} of {len(df_bb)} "
          f"(tags: {df_bb['fit_tag'].value_counts().to_dict()})")

    def _load(path: Path) -> pd.DataFrame:
        """Load a benchmark result and normalize its monthly-origin index."""
        d = pd.read_csv(path)
        if "month_in_quarter" not in d.columns:
            d = expand_quarterly_nowcasts_to_monthly(d)
        if "monthly_origin" in d.columns:
            d = d.set_index("monthly_origin", drop=False)
        return d

    models = {
        "DFM-BlockBalanced(k20)": df_bb,
        "DFM-ifoCAST(fixed)": _load(P.OUT_NOWCASTING / "nowcast_results_dfm_ifocast.csv"),
        "DFM-EN(time-varying)": _load(P.actpn_results_csv("en_only")),
        "DFM-core(time-varying)": _load(P.actpn_results_csv("core")),
        "AR1": _load(P.AR1_RESULTS_CSV),
    }
    rows = []
    for name, d in models.items():
        rows.append({
            "model": name,
            "RMSFE_M3": compute_rmsfe(d, EVAL_START, EVAL_END, month_in_quarter=MIQ),
            "RMSFE_pooled": compute_rmsfe(d, EVAL_START, EVAL_END, month_in_quarter=None),
            "NSR": compute_nsr(d, y, EVAL_START, EVAL_END),
            **{f"RMSFE_{r}": compute_rmsfe(d, q0, q1, month_in_quarter=MIQ)
               for r, (q0, q1) in REGIMES.items()},
        })
    tbl = pd.DataFrame(rows).set_index("model").round(4)
    print("\n[3] Evaluation\n" + tbl.to_string())
    tbl.to_csv(P.OUT_NOWCASTING / "blockbalanced_benchmark_rmsfe.csv")

    print("\nDiebold-Mariano (block-balanced vs X; negative => block-balanced better):")
    for comp_name in ("DFM-ifoCAST(fixed)", "DFM-EN(time-varying)", "DFM-core(time-varying)", "AR1"):
        ea, eb = align_forecast_errors(df_bb, models[comp_name], month_in_quarter=MIQ,
                                       eval_start=EVAL_START, eval_end=EVAL_END)
        dm = diebold_mariano_test(ea, eb)
        print(f"  vs {comp_name:24s}: DM={dm['DM']:+.3f}  p={dm['p_value']:.3f}  n={dm['n']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
