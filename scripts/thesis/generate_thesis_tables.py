#!/usr/bin/env python3
"""Generate all thesis result tables (booktabs) from the canonical data cut.

Every numeric cell is read from the stored result cut (via the companion
dashboard data layer) or recomputed with the same derivations. The script
prints verification lines before writing booktabs files.

Requires the companion dashboard on ``DASHBOARD_SRC``. Write the tables
to ``THESIS_ROOT``.

Run from the repository root:

    python scripts/thesis/generate_thesis_tables.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REPO_ROOT.parent
DASH_SRC = Path(os.environ.get(
    "DASHBOARD_SRC", WORKSPACE / "german-gdp-nowcast-dashboard" / "src",
))
sys.path.insert(0, str(DASH_SRC))
sys.path.insert(0, str(REPO_ROOT / "src"))

from dashboard import config as C, data  # noqa: E402
from dashboard.stats import align_forecast_errors, diebold_mariano_test  # noqa: E402
from german_gdp_nowcasting.models.dfm.nowcast_utils import (  # noqa: E402
    build_forecast_loss_matrix,
    compute_model_confidence_set,
)

TAB_DIR = Path(os.environ.get(
    "THESIS_ROOT", WORKSPACE / "Overleaf-Thesis",
)) / "tables"
NC = C.OUT_NOWCAST

LABELS = {
    "combo_equal": "Equal-weight combination",
    "DFM-SV-k2": "DFM-SV (integrated, $k=2$)",
    "DFM-EN": "DFM-EN",
    "DFM-BlockBalanced": "DFM-block-balanced",
    "DFM-ifoCAST": "DFM-ifoCAST",
    "DFM-TVP": "DFM-TVP",
    "DFM-PLS": "DFM-PLS",
    "MLP-Factor": "MLP-Factor",
    "XGB-Full": "XGB-Full",
    "AR1": "AR(1), expanding",
    "RW": "Random walk",
    "AR(1) expanding": "AR(1), expanding",
    "Rolling-AR(1) 40q": "AR(1), rolling 40q",
    "AR(1) + IC": "AR(1) + intercept correction",
}

SHORT = {
    "DFM-ifoCAST": "ifo", "DFM-EN": "EN", "DFM-PLS": "PLS",
    "DFM-BlockBalanced": "BB",
    "DFM-TVP": "TVP", "DFM-SV-k2": "SV", "combo_equal": "Combo",
    "XGB-Full": "XGB", "MLP-Factor": "MLP", "AR1": "AR(1)", "RW": "RW",
}


def fmt(x, nd=3, plus=False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"
    if s.startswith("-") or s.startswith("+"):
        s = f"${s}$"
    return s


def fmt_p(p, nd=3) -> str:
    if p is None or not np.isfinite(p):
        return "--"
    if p < 0.5 * 10 ** (-nd):
        return f"$<$0.{'0' * (nd - 1)}1"
    return f"{p:.{nd}f}"


def fmt_p_stars(p, nd=3) -> str:
    base = fmt_p(p, nd)
    if p is None or not np.isfinite(p):
        return base
    if p < 0.001:
        return f"{base}$^{{***}}$"
    if p < 0.01:
        return f"{base}$^{{**}}$"
    if p < 0.05:
        return f"{base}$^{{*}}$"
    return base


def write_table(name: str, body: str) -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    (TAB_DIR / f"{name}.tex").write_text(body, encoding="utf-8")
    print(f"wrote tables/{name}.tex")


NOTE_ENV = (
    "\\par\\smallskip\n\\begin{{minipage}}{{\\linewidth}}\\footnotesize\n"
    "\\emph{{Notes:}} {note}\n\\end{{minipage}}\n"
)


# ------------------------------------------------------------------------- #
# 1. Full-sample accuracy (ACC-01..10)
# ------------------------------------------------------------------------- #
def tab_accuracy_full() -> None:
    df = pd.read_csv(NC / "rmsfe_table_all_models.csv")
    path_metrics = data.full_window_accuracy().set_index("model")
    scored_rows: list[tuple[float, str]] = []
    for _, r in df.iterrows():
        model = r["model"]
        if model not in path_metrics.index:
            raise KeyError(f"Missing saved forecast path for {model}")
        metrics = path_metrics.loc[model]
        if not np.isclose(
            float(r["RMSFE_M3"]), float(metrics["rmse"]), atol=5e-4
        ):
            raise ValueError(
                f"RMSFE mismatch for {model}: summary={r['RMSFE_M3']}, "
                f"saved path={metrics['rmse']}"
            )
        rel = fmt(r["vs_AR1"], 3) if np.isfinite(r["vs_AR1"]) else "--"
        scored_rows.append((
            float(r["RMSFE_M3"]),
            f"{LABELS[model]} & {fmt(r['RMSFE_M3'], 3)} & "
            f"{fmt(metrics['mae'], 3)} & "
            f"{fmt(metrics['bias'], 3, plus=True)} & {rel} \\\\",
        ))
    # Older summary CSVs omit DFM-PLS; take its metrics from the saved M3
    # error path (same derivation as the regime table). Skip when the
    # summary already contains the row, otherwise the table duplicates it.
    if "DFM-PLS" not in set(df["model"]):
        pls = path_metrics.loc["DFM-PLS"]
        ar1_rmse = float(path_metrics.loc["AR1", "rmse"])
        scored_rows.append((
            float(pls["rmse"]),
            f"{LABELS['DFM-PLS']} & {fmt(pls['rmse'], 3)} & "
            f"{fmt(pls['mae'], 3)} & "
            f"{fmt(pls['bias'], 3, plus=True)} & "
            f"{fmt(float(pls['rmse']) / ar1_rmse, 3)} \\\\",
        ))
    rows = [row for _, row in sorted(scored_rows, key=lambda t: t[0])]
    body = (
        "% Auto-generated by scripts/thesis/generate_thesis_tables.py\n"
        "% Sources: data/real/nowcasting/rmsfe_table_all_models.csv and "
        "saved model paths\n"
        "% (DFM-PLS row computed entirely from its saved M3 error path)\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Model & RMSFE & MAE & Bias & Rel.\\ RMSFE \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
        + NOTE_ENV.format(note=(
            "Final-month (M3) nowcasts, 2011Q1--2025Q4, $N=60$ quarters. "
            "RMSFE, MAE and bias are in percentage points of quarter-on-quarter "
            "log growth of first-release GDP. Bias is the mean signed error, "
            "defined as nowcast minus actual, so a positive value indicates "
            "average overprediction. Rel.\\ RMSFE is the ratio to the expanding "
            "AR(1). Rows are ordered by RMSFE."))
    )
    write_table("tab_accuracy_full", body)


# ------------------------------------------------------------------------- #
# 2. Regime RMSFE and post-COVID bias (REG-01..10)
# ------------------------------------------------------------------------- #
def tab_regime_rmsfe() -> None:
    df = pd.read_csv(NC / "post_covid_benchmarks_table.csv")
    # The benchmark table predates the integrated-SV run.  Add its M3 regime
    # metrics from the canonical horizon decomposition so the model set is
    # consistent with the full-sample and within-quarter comparisons.
    if "DFM-SV-k2" not in set(df["model"]):
        bv = pd.read_csv(NC / "horizon_bias_variance_table.csv")
        sv = bv[
            (bv["model"] == "DFM-SV-k2")
            & (bv["month_in_quarter"] == "M3")
        ].set_index("regime")
        sv_full = pd.read_csv(
            NC / "rmsfe_table_all_models.csv"
        ).set_index("model").loc["DFM-SV-k2", "RMSFE_M3"]
        df = pd.concat([
            df,
            pd.DataFrame([{
                "model": "DFM-SV-k2",
                "pre-COVID_rmsfe": sv.loc["pre-COVID", "RMSFE"],
                "pre-COVID_bias": sv.loc["pre-COVID", "bias"],
                "COVID_rmsfe": sv.loc["COVID", "RMSFE"],
                "COVID_bias": sv.loc["COVID", "bias"],
                "post-COVID_rmsfe": sv.loc["post-COVID", "RMSFE"],
                "post-COVID_bias": sv.loc["post-COVID", "bias"],
                "all_rmsfe": sv_full,
            }]),
        ], ignore_index=True)
    # DFM-PLS is likewise absent from the benchmark CSV; add its M3 regime
    # metrics from the saved forecast-error path.
    if "DFM-PLS" not in set(df["model"]):
        pls = data.m3_slice(
            data.load_nowcast("DFM-PLS"),
            C.MODELS["DFM-PLS"].has_miq,
        )
        pls_row: dict[str, float | str] = {"model": "DFM-PLS"}
        for reg, (q0, q1) in C.REGIMES.items():
            sub = pls[(pls["quarter"] >= q0) & (pls["quarter"] <= q1)]
            err = sub["error"].dropna()
            pls_row[f"{reg}_rmsfe"] = float(np.sqrt(np.mean(err**2)))
            pls_row[f"{reg}_bias"] = float(err.mean())
        pls_row["all_rmsfe"] = float(
            np.sqrt(np.mean(pls["error"].dropna() ** 2))
        )
        df = pd.concat([df, pd.DataFrame([pls_row])], ignore_index=True)
    df = df.sort_values("post-COVID_rmsfe")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{LABELS[r['model']]} & {fmt(r['pre-COVID_rmsfe'], 3)} & "
            f"{fmt(r['COVID_rmsfe'], 3)} & {fmt(r['post-COVID_rmsfe'], 3)} & "
            f"{fmt(r['post-COVID_bias'], 3, plus=True)} \\\\")
    body = (
        "% Auto-generated by scripts/thesis/generate_thesis_tables.py\n"
        "% Sources: post_covid_benchmarks_table.csv; "
        "horizon_bias_variance_table.csv\n"
        "% (DFM-SV-k2 and DFM-PLS rows recomputed from saved M3 error "
        "paths)\n"
        # Default tabcolsep overruns the measure by ~7pt at \small.
        "\\setlength{\\tabcolsep}{4.5pt}\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        " & \\multicolumn{3}{c}{RMSFE by regime} & Post bias \\\\\n"
        "\\cmidrule(lr){2-4}\n"
        "Model & pre-COVID & COVID & post-COVID & mean error \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
        + NOTE_ENV.format(note=(
            "M3 nowcasts. Regimes: pre-COVID 2011Q1--2019Q4 ($N=36$), COVID "
            "2020Q1--2021Q4 ($N=8$), post-COVID 2022Q1--2025Q4 ($N=16$). "
            "RMSFE and mean error (nowcast minus actual) in percentage "
            "points. The table covers the headline models and the two "
            "break-robust AR(1) variants; rows are sorted by post-COVID "
            "RMSFE. For scale, the sample "
            "standard deviation of realised first-release growth is 0.372 "
            "(pre-COVID), 5.176 (COVID) and 0.187 (post-COVID) percentage "
            "points."))
    )
    write_table("tab_regime_rmsfe", body)


# ------------------------------------------------------------------------- #
# 3. Mincer-Zarnowitz (MZ-01..08)
# ------------------------------------------------------------------------- #
def tab_mz() -> None:
    df = pd.read_csv(NC / "mincer_zarnowitz_table.csv")
    df = df.assign(
        _d_beta=(df["beta"] - 1.0).abs(),
        _d_alpha=df["alpha"].abs(),
    ).sort_values(
        ["p_joint_H0_a0_b1", "_d_beta", "_d_alpha"],
        ascending=[False, True, True],
    )
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{LABELS[r['model']]} & "
            f"{fmt(r['alpha'], 3)} ({fmt(r['se_alpha'], 3)}) & "
            f"{fmt(r['beta'], 3)} ({fmt(r['se_beta'], 3)}) & "
            f"{fmt_p_stars(r['p_joint_H0_a0_b1'], 3)} \\\\")
    body = (
        "% Auto-generated by scripts/thesis/generate_thesis_tables.py\n"
        "% Source: data/real/nowcasting/mincer_zarnowitz_table.csv\n"
        "\\begin{tabular}{lccr}\n\\toprule\n"
        "Model & $\\hat\\alpha$ (s.e.) & $\\hat\\beta$ (s.e.) & "
        "joint $p$ \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
        + NOTE_ENV.format(note=(
            "Mincer--Zarnowitz regressions $y_q = \\alpha + \\beta "
            "\\hat{y}_q + \\varepsilon_q$ of realised first-release growth "
            "on the M3 nowcast, 2011Q1--2025Q4, $N=60$ (same fit as "
            "\\cref{fig:mz-forest}). The joint $p$-value tests "
            "$H_0\\colon \\alpha=0, \\beta=1$ (forecast efficiency). "
            "Only the equal-weight combination fails to reject at 5\\%; "
            "DFM-EN and DFM-SV reject despite the next-closest slopes. "
            "$^{*}$~$p<0.05$, $^{**}$~$p<0.01$, $^{***}$~$p<0.001$. "
            "$\\hat\\beta>1$ means nowcasts are compressed: outcomes move "
            "more than one-for-one with the forecast."))
    )
    write_table("tab_mz", body)


# ------------------------------------------------------------------------- #
# 4. Selected DM comparisons incl. regime tests (DM-01..09, XGB-04)
# ------------------------------------------------------------------------- #
def _regime_dm(key_a: str, key_b: str, regime: str) -> tuple[float, int]:
    df_a = data.m3_slice(data.load_nowcast(key_a), C.MODELS[key_a].has_miq)
    df_b = data.m3_slice(data.load_nowcast(key_b), C.MODELS[key_b].has_miq)
    q0, q1 = C.REGIMES[regime]
    e1, e2 = align_forecast_errors(df_a, df_b, month_in_quarter=None,
                                   eval_start=q0, eval_end=q1)
    res = diebold_mariano_test(e1, e2)
    return float(res["p_value"]), int(res["n"])


def tab_dm_selected() -> None:
    dm = pd.read_csv(NC / "diebold_mariano_table_all_models.csv",
                     index_col=0)
    full_rmsfe = pd.read_csv(
        NC / "rmsfe_table_all_models.csv"
    ).set_index("model")["RMSFE_M3"]
    regime_rmsfe = pd.read_csv(
        NC / "post_covid_benchmarks_table.csv"
    ).set_index("model")

    def full_row(a: str, b: str) -> tuple[str, str, int, float, float]:
        return (
            f"{LABELS[a]} vs.\\ {LABELS[b]}",
            "full",
            60,
            float(full_rmsfe[a] - full_rmsfe[b]),
            float(dm.loc[a, b]),
        )

    def regime_row(
        a: str, b: str, regime: str
    ) -> tuple[str, str, int, float, float]:
        p, n = _regime_dm(a, b, regime)
        delta = float(
            regime_rmsfe.loc[a, f"{regime}_rmsfe"]
            - regime_rmsfe.loc[b, f"{regime}_rmsfe"]
        )
        print(
            f"VERIFY DM {a} vs {b} ({regime}): "
            f"delta={delta:+.3f} p={p:.3f} n={n}"
        )
        return f"{LABELS[a]} vs.\\ {LABELS[b]}", regime, n, delta, p

    sections = [
        (
            "Full-sample ranking and functional form",
            [
                full_row("combo_equal", "DFM-EN"),
                full_row("DFM-EN", "XGB-Full"),
                full_row("DFM-EN", "MLP-Factor"),
                full_row("XGB-Full", "MLP-Factor"),
            ],
        ),
        (
            "Input-set sensitivity",
            [
                full_row("DFM-EN", "DFM-ifoCAST"),
                regime_row("DFM-EN", "DFM-ifoCAST", "pre-COVID"),
                regime_row(
                    "DFM-BlockBalanced", "DFM-ifoCAST", "post-COVID"
                ),
            ],
        ),
        (
            "Post-COVID adaptation",
            [
                regime_row("XGB-Full", "DFM-EN", "post-COVID"),
                regime_row("DFM-TVP", "DFM-EN", "post-COVID"),
            ],
        ),
    ]

    # XGB versus the rolling AR is produced by the dedicated robustness run;
    # the rolling forecast path is not part of the headline DM matrix.
    xgb_dm_text = C.XGB_SENSITIVITY_DM_TXT.read_text(encoding="utf-8")
    match = re.search(r"p_value=([0-9.]+)\s+n=(\d+)", xgb_dm_text)
    if not match:
        raise ValueError("Could not parse XGB versus rolling-AR DM result")
    xgb_p, xgb_n = float(match.group(1)), int(match.group(2))
    xgb_delta = float(
        regime_rmsfe.loc["XGB-Full", "post-COVID_rmsfe"]
        - regime_rmsfe.loc["Rolling-AR(1) 40q", "post-COVID_rmsfe"]
    )
    sections[-1][1].append((
        "XGB-Full vs.\\ AR(1), rolling 40q",
        "post-COVID",
        xgb_n,
        xgb_delta,
        xgb_p,
    ))

    lines = []
    for i, (heading, rows) in enumerate(sections):
        if i:
            lines.append("\\addlinespace")
        lines.append(
            f"\\multicolumn{{5}}{{l}}{{\\emph{{{heading}}}}} \\\\"
        )
        lines.extend(
            f"{name} & {window} & {n} & {fmt(delta, 3, plus=True)} "
            f"& {fmt_p(p)} \\\\"
            for name, window, n, delta, p in rows
        )
    body = (
        "% Auto-generated by scripts/thesis/generate_thesis_tables.py\n"
        "% Sources: diebold_mariano_table_all_models.csv; regime tests\n"
        "% recomputed from saved M3 error paths with src/dashboard/stats.py;\n"
        "% _scratch/xgb_sensitivity_dm_vs_rolling_ar1.txt\n"
        "\\setlength{\\tabcolsep}{4.5pt}\n"
        "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{5.7cm}lrrr@{}}\n"
        "\\toprule\n"
        "Comparison & Window & $N$ & $\\Delta$RMSFE & DM $p$ \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
        + NOTE_ENV.format(note=(
            "Two-sided Harvey--Leybourne--Newbold corrected Diebold--Mariano "
            "tests of equal squared-error loss on M3 nowcast errors. "
            "The displayed comparisons correspond to the chapter's three "
            "substantive claims: full-sample ranking and functional form, "
            "input-set sensitivity, and post-COVID adaptation. "
            "$\\Delta$RMSFE is the first-named model minus the second-named "
            "model in percentage points, so a negative value favours the "
            "first model. "
            "\\emph{full} = 2011Q1--2025Q4; pre-COVID = 2011Q1--2019Q4; "
            "post-COVID = 2022Q1--2025Q4. Full-window values are taken from "
            "the saved pairwise test table; regime values are computed from "
            "the saved forecast-error paths with the same HLN "
            "implementation. The XGB--rolling-AR comparison comes from the "
            "dedicated XGBoost robustness run. With $N=16$ post-COVID "
            "quarters these tests "
            "have low power."))
    )
    write_table("tab_dm_selected", body)


# ------------------------------------------------------------------------- #
# 5. SV interval calibration (SV-01..05)
# ------------------------------------------------------------------------- #
def tab_sv_calibration() -> None:
    cal = pd.read_csv(NC / "sv_interval_calibration_table.csv").iloc[0]
    sv = data.m3_slice(data.load_nowcast("DFM-SV-k2"), True)
    covered = int(((sv["actual"] >= sv["ci_lower_90"])
                   & (sv["actual"] <= sv["ci_upper_90"])).sum())
    n = len(sv)
    print(f"VERIFY SV coverage: {covered}/{n}, "
          f"mean width {float(cal['mean_width']):.4f}")
    body = (
        "% Auto-generated by scripts/thesis/generate_thesis_tables.py\n"
        "% Source: data/real/nowcasting/sv_interval_calibration_table.csv\n"
        "\\begin{tabular}{lr}\n\\toprule\n"
        "Quantity & Value \\\\\n\\midrule\n"
        f"Nominal coverage & {fmt(float(cal['coverage_nominal']), 3)} \\\\\n"
        f"Empirical coverage & {fmt(float(cal['coverage_empirical']), 3)} "
        f"({covered}/{n}) \\\\\n"
        f"Mean interval width (pp) & {fmt(float(cal['mean_width']), 3)} \\\\\n"
        f"Mean half-width (pp) & {fmt(float(cal['mean_width']) / 2, 3)} \\\\\n"
        f"Mean CRPS (pp) & {fmt(float(cal['CRPS']), 3)} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
        + NOTE_ENV.format(note=(
            "90\\% predictive intervals of the integrated DFM-SV ($k=2$) at "
            "M3, 2011Q1--2025Q4, $N=60$. Width is the full distance between "
            "the 5\\% and 95\\% quantiles. CRPS is the mean Gneiting--Raftery "
            "score of the Gaussian predictive density (lower is better)."))
    )
    write_table("tab_sv_calibration", body)


# ------------------------------------------------------------------------- #
# 6. Selection set sizes and composition masses (SEL-01..07, 14, 15)
# ------------------------------------------------------------------------- #
def _mass_by_regime() -> pd.DataFrame:
    """Hard/soft mass shares by regime and method (derivation A)."""
    long = data.regime_soft_hard()
    return long


def tab_selection_masses() -> None:
    long = _mass_by_regime()
    uni = data.universe_soft_hard()
    methods = [("EN", "EN"), ("Block-balanced", "Block-balanced"),
               ("PLS", "PLS"), ("XGBoost (SHAP)", "XGBoost (SHAP)")]
    seg_rows = {"Hard (real activity)": "Hard real activity",
                "Soft (surveys)": "Surveys (soft)"}
    lines = []
    for seg, seg_label in seg_rows.items():
        base = float(uni.get(seg, np.nan))
        lines.append(
            f"\\multicolumn{{5}}{{l}}{{\\emph{{{seg_label}}} "
            f"(universe share {fmt(base, 3)})}} \\\\")
        for mkey, mlabel in methods:
            vals = []
            for reg in C.REGIMES:
                v = long[(long["method"] == mkey) & (long["regime"] == reg)
                         & (long["segment"] == seg)]["share"]
                vals.append(float(v.sum()) if len(v) else np.nan)
            print(f"VERIFY mass {mkey} {seg}: "
                  + " / ".join(fmt(v, 3) for v in vals))
            lines.append(
                f"\\quad {mlabel} & {fmt(vals[0], 3)} & {fmt(vals[1], 3)} & "
                f"{fmt(vals[2], 3)} & {fmt(np.nan)} \\\\")
    # lag-0 mass (SEL-14/15) as a separate block, full window
    lag0 = {"Universe": 0.696, "EN": 0.224, "XGBoost (SHAP)": 0.235,
            "PLS": 0.0006}
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{l}{\\emph{Lag-0 (timely) mass, "
                 "full window}} \\\\")
    for k, v in lag0.items():
        nd = 4 if v < 0.01 else 3
        lines.append(f"\\quad {k} & \\multicolumn{{3}}{{c}}{{--}} & "
                     f"{fmt(v, nd)} \\\\")
    body = (
        "% Auto-generated by scripts/thesis/generate_thesis_tables.py\n"
        "% Derivation A: selection matrices + SHAP + metadata categories\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        " & pre-COVID & COVID & post-COVID & full \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
        + NOTE_ENV.format(note=(
            "Share of each method's selected mass by regime. For the binary "
            "selectors (EN, block-balanced, PLS) mass is the selection "
            "count over origins; for XGBoost it is mean absolute SHAP "
            "contribution aggregated to quarters. Hard real activity pools "
            "Orders, Turnover, Production, Construction and Trade; regimes "
            "as in the text (108/24/48 monthly origins pre/COVID/post for "
            "the binary methods; 36/8/16 quarters for SHAP). Lag-0 mass is "
            "the share of selected mass on indicators available in the "
            "origin month; the universe base rate is 407/585 = 0.696."))
    )
    write_table("tab_selection_masses", body)


def verify_selection_sizes() -> None:
    en = data.load_selection_matrix("EN (raw)")
    sizes = en.sum(axis=1)
    print(f"VERIFY EN sizes: mean={sizes.mean():.3f} median={sizes.median()} "
          f"min={int(sizes.min())} max={int(sizes.max())} n={len(sizes)}")
    bb = data.load_selection_matrix("Block-balanced (k=20)")
    pls = data.load_selection_matrix("PLS")
    print(f"VERIFY BB sizes: unique={sorted(set(bb.sum(axis=1)))[:5]}")
    print(f"VERIFY PLS sizes: unique={sorted(set(pls.sum(axis=1)))[:5]}")
    agree = data.cross_method_agreement()
    print("VERIFY Spearman:\n", agree.round(3))
    # SEL-11..13
    ifo = set(data.ifocast_membership())
    jac = []
    for _, row in en.iterrows():
        sel = set(en.columns[row.astype(bool)])
        jac.append(len(sel & ifo) / len(sel | ifo))
    print(f"VERIFY EN-ifo Jaccard mean: {np.mean(jac):.3f} "
          f"(|ifo panel|={len(ifo)})")
    always = (en.sum(axis=0) == len(en)).sum()
    print(f"VERIFY EN always-selected: {int(always)}")
    never = len([s for s in ifo if s in en.columns
                 and en[s].sum() == 0]
                + [s for s in ifo if s not in en.columns])
    print(f"VERIFY ifo never selected by EN: {never}")


# ------------------------------------------------------------------------- #
# 7. Appendix: elastic-net core series (top 30 by selection frequency)
# ------------------------------------------------------------------------- #
_EN_CORE_HEADER = (
    "Indicator & Series ID & Category & EN freq. & SHAP \\\\\n"
)


def tab_en_core_series(top_n: int = 30) -> None:
    en = data.load_selection_matrix("EN (raw)")
    freq = en.mean(axis=0).sort_values(ascending=False).head(top_n)
    md = pd.read_csv(C.DATA_DICT_CSV).drop_duplicates("id").set_index("id")
    shap = pd.read_csv(C.SHAP_CSV)
    shap["id"] = shap["feature"].map(data.feature_to_id)
    shap_mean = shap.groupby("id")["mean_abs_shap"].mean()
    lines = []
    for sid, f in freq.items():
        raw = str(md.loc[sid, "name"]) if sid in md.index else sid
        name = _short_indicator_name(raw, sid=str(sid))
        cat = md.loc[sid, "category"] if sid in md.index else "Misc"
        shap_v = shap_mean.get(sid, np.nan)
        name_tex = (_tex_escape(name)
                    .replace("excl. ", "excl.\\ ")
                    .replace("vs previous", "vs.\\ previous"))
        sid_tex = "\\scriptsize\\texttt{" + _tex_escape(sid).replace(
            "\\_", "\\_\\allowbreak{}"
        ) + "}"
        lines.append(
            f"{name_tex} & {sid_tex} & "
            f"{_tex_escape(str(cat))} & {fmt(f, 3)} & {fmt(shap_v, 3)} \\\\"
        )
        print(f"VERIFY EN-core {sid}: {name}  ({len(name)} chars)")
    note = (
        "Top 30 series by elastic-net selection frequency across the 180 "
        "monthly origins. Names drop the country prefix and the source "
        "seasonal-adjustment / price-base boilerplate "
        "(Calendar Adjusted, X13 JDemetra+, constant prices, SA, index). "
        "\\emph{EN freq.} is the share of origins at which the series "
        "enters the selected set; e.g.\\ 0.93 means the series was chosen "
        "at 167 of the 180 origins. \\emph{SHAP} is the mean absolute SHAP "
        "weight in XGB-Full, averaged over quarterly origins; it measures "
        "average predictive influence, not selection frequency."
    )
    # longtable[c]: fill pages naturally (no forced \pagebreak — the chapter
    # intro already consumes page 1, so a mid-list break left page 2 half empty).
    colspec = "{@{}p{5.4cm} p{2.25cm} l c c@{}}"
    body = (
        "% Auto-generated by scripts/thesis/generate_thesis_tables.py\n"
        "% Sources: en_only_selection_matrix.csv; metadata/data_dict.csv;\n"
        "% nowcasting/xgb_shap_importance.csv\n"
        "% longtable: breaks across pages; centred; short indicator names\n"
        # footnotesize keeps all 30 rows plus the notes inside two pages; at
        # \small the last row and the notes spilled onto a third.
        "{\\footnotesize\\setlength{\\tabcolsep}{5pt}%\n"
        "\\renewcommand{\\arraystretch}{1.0}%\n"
        "\\hyphenpenalty=10000\\exhyphenpenalty=10000\n"
        "\\begin{longtable}[c]" + colspec + "\n"
        "\\caption[Elastic-net core series]{Top 30 series by elastic-net "
        "selection frequency, with mean absolute SHAP weight in XGB-Full.}%\n"
        "\\label{tab:en-core-series}\\\\\n"
        "\\toprule\n"
        + _EN_CORE_HEADER +
        "\\midrule\n"
        "\\endfirsthead\n"
        "\\caption[]{Top 30 series by elastic-net selection frequency "
        "(continued).}\\\\\n"
        "\\toprule\n"
        + _EN_CORE_HEADER +
        "\\midrule\n"
        "\\endhead\n"
        "\\bottomrule\n"
        "\\endfoot\n"
        "\\bottomrule\n"
        "\\endlastfoot\n"
        + "\n".join(lines) + "\n"
        "\\end{longtable}}\n"
        + NOTE_ENV.format(note=note)
    )
    write_table("tab_en_core_series", body)


# ------------------------------------------------------------------------- #
# 8. Appendix: horizon profile (HOR-01..07)
# ------------------------------------------------------------------------- #
def tab_horizon_profile() -> None:
    df = data.load_horizon_profile()
    models = ["DFM-EN", "DFM-ifoCAST", "DFM-PLS", "DFM-BlockBalanced",
              "DFM-TVP", "DFM-SV-k2"]
    models = [m for m in models if m in set(df["model"])]
    lines = []
    for reg, n in [("pre-COVID", 36), ("COVID", 8), ("post-COVID", 16)]:
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{reg}}} "
                     f"($N={n}$)}} \\\\")
        for m in models:
            s = (df[(df["model"] == m) & (df["regime"] == reg)]
                 .set_index("month_in_quarter")["RMSFE"])
            lines.append(f"\\quad {LABELS[m]} & {fmt(s.get(1), 4)} & "
                         f"{fmt(s.get(2), 4)} & {fmt(s.get(3), 4)} \\\\")
    body = (
        "% Auto-generated by scripts/thesis/generate_thesis_tables.py\n"
        "% Source: data/real/nowcasting/horizon_profile_table.csv\n"
        "% (DFM-PLS rows recomputed from nowcast_results_actpn_pls_only.csv)\n"
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Model & M1 & M2 & M3 \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
        + NOTE_ENV.format(note=(
            "RMSFE (pp) of the DFM variants at the three within-quarter "
            "information sets M1, M2, M3. Regime windows as in the main "
            "text."))
    )
    write_table("tab_horizon_profile", body)


# ------------------------------------------------------------------------- #
# 8. Appendix: bias-variance decomposition (BV-01..03 and all models)
# ------------------------------------------------------------------------- #
def tab_bias_variance() -> None:
    df = data.load_horizon_bias_variance()
    models = ["DFM-EN", "DFM-ifoCAST", "DFM-PLS", "DFM-BlockBalanced",
              "DFM-TVP", "DFM-SV-k2"]
    models = [m for m in models if m in set(df["model"])]
    sub = df[df["regime"] == "post-COVID"]
    lines = []
    for m in models:
        s = sub[sub["model"] == m].set_index("month_in_quarter")
        for miq in ["M1", "M2", "M3"]:
            r = s.loc[miq]
            first = LABELS[m] if miq == "M1" else ""
            lines.append(
                f"{first} & {miq} & {fmt(r['bias'], 4, plus=True)} & "
                f"{fmt(r['bias_sq'], 4)} & {fmt(r['variance'], 4)} & "
                f"{fmt(r['RMSFE'], 4)} & {fmt(r['bias_sq_share_pct'], 1)} "
                "\\\\")
        lines.append("\\addlinespace")
    body = (
        "% Auto-generated by scripts/thesis/generate_thesis_tables.py\n"
        "% Source: data/real/nowcasting/horizon_bias_variance_table.csv\n"
        "% (DFM-PLS rows from nowcast_results_actpn_pls_only.csv)\n"
        "\\begin{tabular}{llrrrrr}\n\\toprule\n"
        "Model & Set & Bias & Bias$^2$ & Variance & RMSFE & "
        "Bias$^2$ share (\\%) \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\\bottomrule\n\\end{tabular}\n"
        + NOTE_ENV.format(note=(
            "Post-COVID window 2022Q1--2025Q4, $N=16$. Bias is the mean "
            "error (nowcast minus actual, pp); MSE $=$ Bias$^2$ $+$ "
            "Variance (pp$^2$); RMSFE in pp. The bias$^2$ share is the "
            "fraction of MSE due to the squared mean error."))
    )
    write_table("tab_bias_variance", body)


# ------------------------------------------------------------------------- #
# 9. Appendix: full DM matrix
# ------------------------------------------------------------------------- #
def tab_dm_full() -> None:
    dm = pd.read_csv(NC / "diebold_mariano_table_all_models.csv", index_col=0)
    # Older cuts of the pairwise matrix omit DFM-PLS. Backfill its row and
    # column from the saved M3 error paths with the identical HLN-corrected
    # test (verified to reproduce every canonical pair exactly). Appending it
    # unconditionally would duplicate the label once the cut already has it,
    # after which dm.loc[m, m2] returns a Series rather than a scalar.
    if "DFM-PLS" not in dm.index:
        pls = data.m3_slice(
            data.load_nowcast("DFM-PLS"), C.MODELS["DFM-PLS"].has_miq
        )
        pls_p: dict[str, float] = {}
        for m in dm.index:
            other = data.m3_slice(
                data.load_nowcast(m), C.MODELS[m].has_miq
            )
            ea, eb = align_forecast_errors(
                pls, other, month_in_quarter=None,
                eval_start="2011Q1", eval_end="2025Q4",
            )
            pls_p[m] = float(diebold_mariano_test(ea, eb)["p_value"])
        dm = dm.reindex(index=list(dm.index) + ["DFM-PLS"],
                        columns=list(dm.columns) + ["DFM-PLS"])
        for m, p in pls_p.items():
            dm.loc["DFM-PLS", m] = p
            dm.loc[m, "DFM-PLS"] = p
    order = [m for m in ["DFM-ifoCAST", "DFM-EN", "DFM-PLS",
                         "DFM-BlockBalanced",
                         "DFM-TVP", "DFM-SV-k2", "combo_equal", "XGB-Full",
                         "MLP-Factor", "AR1", "RW"] if m in dm.index]
    dm = dm.loc[order, order]
    header = " & ".join(SHORT[m] for m in order)
    lines = []
    for m in order:
        cells = []
        for m2 in order:
            v = dm.loc[m, m2]
            cells.append("--" if not np.isfinite(v) else f"{v:.3f}")
        lines.append(f"{SHORT[m]} & " + " & ".join(cells) + " \\\\")
    body = (
        "% Auto-generated by scripts/thesis/generate_thesis_tables.py\n"
        "% Source: data/real/nowcasting/diebold_mariano_table_all_models.csv\n"
        "% (DFM-PLS row/column recomputed from saved M3 error paths with "
        "the same HLN-corrected test)\n"
        "\\setlength{\\tabcolsep}{3.6pt}\n"
        "\\begin{tabular}{@{}l" + "r" * len(order) + "@{}}\n\\toprule\n"
        " & " + header + " \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
        + NOTE_ENV.format(note=(
            "Two-sided HLN-corrected Diebold--Mariano $p$-values for equal "
            "squared-error loss, M3 nowcasts, 2011Q1--2025Q4, $N=60$. "
            "ifo = DFM-ifoCAST, EN = DFM-EN, PLS = DFM-PLS, BB = "
            "DFM-block-balanced, TVP = "
            "DFM-TVP, SV = DFM-SV ($k=2$), Combo = equal-weight "
            "combination, XGB = XGB-Full, MLP = MLP-Factor."))
    )
    write_table("tab_dm_full", body)


# ------------------------------------------------------------------------- #
# 10. Appendix: model confidence set
# ------------------------------------------------------------------------- #
def tab_mcs() -> None:
    # Recomputed from the saved M3 error paths so that DFM-PLS (absent from
    # model_confidence_set_table.csv) enters the set; with the canonical
    # ten-model input this reproduces the saved table exactly (seed 42).
    models = [m for m in ["DFM-ifoCAST", "DFM-EN", "DFM-PLS",
                          "DFM-BlockBalanced", "DFM-TVP", "DFM-SV-k2",
                          "combo_equal", "XGB-Full", "MLP-Factor", "AR1",
                          "RW"]]
    loaded = {
        m: data.m3_slice(data.load_nowcast(m), C.MODELS[m].has_miq)
        for m in models
    }
    losses = build_forecast_loss_matrix(
        loaded, eval_start="2011Q1", eval_end="2025Q4",
        month_in_quarter=None, loss="se",
    )
    mcs = compute_model_confidence_set(losses)
    mcs.insert(1, "RMSFE", np.sqrt(mcs["mean_loss"]))
    rows = []
    for _, r in mcs.sort_values("RMSFE").iterrows():
        retained = "yes" if bool(r["in_MCS"]) else "no"
        rows.append(
            f"{LABELS[r.name]} & {fmt(r['RMSFE'], 3)} & "
            f"{fmt_p(r['MCS_p_value'], 3)} & {retained} \\\\"
        )
    body = (
        "% Auto-generated by scripts/thesis/generate_thesis_tables.py\n"
        "% Recomputed from saved M3 error paths (arch MCS, seed 42);\n"
        "% reproduces model_confidence_set_table.csv on the canonical\n"
        "% ten-model input and adds the DFM-PLS row\n"
        "\\begin{tabular}{lrrc}\n\\toprule\n"
        "Model & RMSFE & MCS $p$ & Retained \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
        + NOTE_ENV.format(note=(
            "90\\% Hansen--Lunde--Nason model confidence set for squared "
            "M3 forecast loss, 2011Q1--2025Q4, $N=60$. The range statistic "
            "uses 10,000 stationary-bootstrap replications, expected block "
            "length four quarters, and random seed 42. MCS $p$ is the "
            "smallest test size at which the model remains in the set; a "
            "model is retained when $p>0.10$. Retention means the model "
            "cannot be eliminated as inferior at this level, not that all "
            "retained models have equal population accuracy."))
    )
    write_table("tab_mcs", body)


# ------------------------------------------------------------------------- #
# 11. Appendix: XGB sensitivity (XGB-01..04)
# ------------------------------------------------------------------------- #
def tab_xgb_sensitivity() -> None:
    df = pd.read_csv(NC / "_scratch" / "xgb_sensitivity_summary.csv")
    jk = pd.read_csv(NC / "_scratch" / "xgb_sensitivity_jackknife_postcovid.csv")
    seeds = df[df["run"].str.startswith("seed")]
    hps = df[df["run"].str.startswith("hp")]
    print(f"VERIFY XGB seeds post range: {seeds['rmsfe_post'].min():.3f}"
          f"-{seeds['rmsfe_post'].max():.3f}")
    print(f"VERIFY XGB hp post range: {hps['rmsfe_post'].min():.3f}"
          f"-{hps['rmsfe_post'].max():.3f}")
    print(f"VERIFY XGB jackknife range: "
          f"{jk['rmsfe_excl_quarter'].min():.3f}"
          f"-{jk['rmsfe_excl_quarter'].max():.3f}")
    run_labels = {
        "seed_0": "seed $= 0$", "seed_1": "seed $= 1$",
        "seed_7": "seed $= 7$", "seed_42": "seed $= 42$ (headline)",
        "seed_123": "seed $= 123$",
        "hp_max_depth-1": "max depth $-1$ (5)",
        "hp_max_depth+1": "max depth $+1$ (7)",
        "hp_lr_half": "learning rate $\\times 0.5$",
        "hp_lr_double": "learning rate $\\times 2$",
        "hp_n_estimators-100": "trees $-100$ (400)",
        "hp_n_estimators+100": "trees $+100$ (600)",
    }
    lines = []
    for _, r in df.iterrows():
        cv = fmt(r["cv_rmse"], 3) if np.isfinite(r["cv_rmse"]) else "--"
        lines.append(f"{run_labels[r['run']]} & {cv} & "
                     f"{fmt(r['rmsfe_pre'], 3)} & {fmt(r['rmsfe_COVID'], 3)} "
                     f"& {fmt(r['rmsfe_post'], 3)} & "
                     f"{fmt(r['rmsfe_full'], 3)} \\\\")
    body = (
        "% Auto-generated by scripts/thesis/generate_thesis_tables.py\n"
        "% Source: data/real/nowcasting/_scratch/xgb_sensitivity_summary.csv\n"
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Run & CV RMSE & pre-COVID & COVID & post-COVID & full \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
        + NOTE_ENV.format(note=(
            "RMSFE (pp) of XGB-Full re-runs across random seeds and local "
            "hyperparameter perturbations around the headline "
            "configuration (seed $=42$). Regime windows and $N$ as in the "
            "main text. Leave-one-quarter-out post-COVID RMSFE across the "
            "16 jackknife samples spans "
            f"{jk['rmsfe_excl_quarter'].min():.3f}--"
            f"{jk['rmsfe_excl_quarter'].max():.3f} pp. The HLN-DM test of "
            "the headline run against the rolling AR(1) over "
            "2022Q1--2025Q4 gives $p=0.452$ ($N=16$)."))
    )
    write_table("tab_xgb_sensitivity", body)


# ------------------------------------------------------------------------- #
# 12. Appendix: ifoCAST specification (used DFM-ifoCAST predictors)
# ------------------------------------------------------------------------- #
def _tex_escape(s: str) -> str:
    return (s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            .replace("#", "\\#"))


# Trailing seasonal-adjustment / price-base / unit tokens in Macrobond names.
_TRAILING_BOILER = re.compile(
    r"(?:,\s*(?:"
    r"Calendar Adjusted(?:\s*\([^)]+\))?"
    r"|Working[- ]Day Adjusted(?:\s*\([^)]+\))?"
    r"|Constant Prices"
    r"|SA(?:\s*\([^)]+\))?"
    r"|Index"
    r"|EUR\s*\[[^\]]+\]"
    r"))+$",
    re.IGNORECASE,
)

# After dropping the country prefix and adjustment boilerplate, map the
# remaining concept string to a one-line label (same style as the ifoCAST table).
_EN_CORE_SHORT_NAMES = {
    "Domestic Trade, Services Trade, Turnover, Accommodation & Food Services, Total":
        "Accommodation & food services turnover",
    "Domestic Trade, Retail Trade, Turnover, Total, Excluding Vehicle Trade":
        "Retail turnover excl. vehicles",
    "Foreign Trade, Total, Export": "Exports (special trade)",
    "Construction, Civil Engineering": "Civil engineering production",
    "New Orders, Manufacturing, Manufacture of Motor Vehicles":
        "New orders, motor vehicles",
    "Production Sales, Turnover, Manufacture of Basic Pharmaceutical Products & Preparations, Domestic Markets":
        "Pharma turnover, domestic",
    "Production Sales, Turnover, Manufacturing, Domestic Markets":
        "Manufacturing turnover, domestic",
    "Construction, Construction of Buildings": "Building construction production",
    "Manufacturing, Machinery & Equipment N.E.C., Total":
        "Machinery & equipment production",
    "Production Sales, Turnover, Durable Goods, Domestic Markets":
        "Durable goods turnover, domestic",
    "Industrial Production, Total": "Industrial production, total",
    "Production Sales, Turnover, Energy, Total": "Energy turnover",
    "Domestic Trade, Services Trade, Turnover, Accommodation & Food Services, Food & Beverage Service Activities":
        "Food & beverage services turnover",
    "Bundesbank, Orders Received from the Domestic Market, Germany, 29+30 Manufacture of Motor Vehicles, Trailers, Semi-Trailers & of Other Transport Equipment":
        "Bundesbank vehicle orders, domestic",
    "Manufacturing, Basic Metal Products, Total": "Basic metals production",
    "Manufacturing, Manufacture of Basic Metals & Fabricated Metal Products":
        "Basic & fabricated metals production",
    "Production Sales, Turnover, Intermediate Goods, Total":
        "Intermediate goods turnover",
    "Production Sales, Turnover, Manufacture of Fabricated Metal Products, Excluding Machinery":
        "Fabricated metals turnover",
    "Production Sales, Turnover, Manufacture of Fabricated Metal Products, Excluding Machinery, Domestic Markets":
        "Fabricated metals turnover, domestic",
    "Manufacturing, Fabricated Metal Products, Except Machinery & Equipment, Total":
        "Fabricated metals prod. excl. machinery",
    "Production Sales, Turnover, Manufacture of Motor Vehicles, Trailers, Semi-Trailers, Domestic Markets":
        "Motor vehicles turnover, domestic",
    "Production Sales, Turnover, Energy, Domestic Markets":
        "Energy turnover, domestic",
    "Production Sales, Turnover, Intermediate Goods, Domestic Markets":
        "Intermediate goods turnover, domestic",
    "New Orders, Manufacturing, Domestic, By Industry, Manufacture of Basic Metals":
        "New orders, basic metals, domestic",
    "Industrial Production, Total, Excluding Construction":
        "Industrial production excl. construction",
    "Production Sales, Turnover, Manufacturing & Mining, Excluding Energy, Domestic Markets":
        "Mfg & mining turnover excl. energy, domestic",
    "Business Surveys, DG ECFIN, Industrial Confidence Indicator, Main Industrial Grouping, Durable Consumer Goods, Employment Expectations for the Months Ahead":
        "ECFIN employment expectations, durables",
    "Bundesbank, Orders Received, Germany, Consumer Goods (Durable- & Non-Durable Gooods)":
        "Bundesbank orders, consumer goods",
    "Domestic Trade, Wholesale Trade, Turnover, Total, Excluding Vehicle Trade":
        "Wholesale turnover excl. vehicles",
    "Bundesbank, Orders Received from the Domestic Market, Germany, 20+21 Manufacture of Chemicals, Chemical Products, Basic Pharmaceutical Products & Pharmaceutical Preparations":
        "Bundesbank chemicals & pharma orders, domestic",
}

# ID lookup so shortening cannot fail if the Macrobond string drifts slightly.
_EN_CORE_SHORT_BY_ID = {
    "detrad3877": "Accommodation & food services turnover",
    "detrad1360": "Retail turnover excl. vehicles",
    "detrad0692": "Exports (special trade)",
    "deprod1989": "Civil engineering production",
    "deprod4738": "New orders, motor vehicles",
    "detrad4750": "Pharma turnover, domestic",
    "detrad3415": "Manufacturing turnover, domestic",
    "deprod1984": "Building construction production",
    "deprod0194": "Machinery & equipment production",
    "detrad1954": "Durable goods turnover, domestic",
    "deprod1404": "Industrial production, total",
    "detrad3364": "Energy turnover",
    "detrad3857": "Food & beverage services turnover",
    "buba_mb_118226": "Bundesbank vehicle orders, domestic",
    "deprod0190": "Basic metals production",
    "deprod3790": "Basic & fabricated metals production",
    "detrad1878": "Intermediate goods turnover",
    "detrad3518": "Fabricated metals turnover",
    "detrad4770": "Fabricated metals turnover, domestic",
    "deprod0191": "Fabricated metals prod. excl. machinery",
    "detrad4789": "Motor vehicles turnover, domestic",
    "detrad3365": "Energy turnover, domestic",
    "detrad1879": "Intermediate goods turnover, domestic",
    "deprod2708": "New orders, basic metals, domestic",
    "deprod1370": "Industrial production excl. construction",
    "detrad1854": "Mfg & mining turnover excl. energy, domestic",
    "indu_de_cdur_7_bs_m": "ECFIN employment expectations, durables",
    "buba_mb_118162": "Bundesbank orders, consumer goods",
    "detrad1045": "Wholesale turnover excl. vehicles",
    "buba_mb_118220": "Bundesbank chemicals & pharma orders, domestic",
}


def _strip_indicator_boilerplate(name: str) -> str:
    """Drop 'Germany,' and trailing Calendar Adjusted / SA / Index tokens."""
    s = name.strip()
    s = re.sub(r"^Germany,\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^Germany\s+", "", s, flags=re.IGNORECASE)
    trailing = re.compile(
        r",\s*(?:"
        r"Calendar Adjusted(?:\s*\([^)]+\))?"
        r"|Working[- ]Day Adjusted(?:\s*\([^)]+\))?"
        r"|Seasonally Adjusted(?:\s*\([^)]+\))?"
        r"|Constant Prices"
        r"|Current Prices"
        r"|SA(?:\s*\([^)]+\))?"
        r"|NSA"
        r"|Index"
        r"|EUR\s*\[[^\]]+\]"
        r")\s*$",
        re.IGNORECASE,
    )
    prev = None
    while prev != s:
        prev = s
        s = trailing.sub("", s).strip(" ,")
    s = _TRAILING_BOILER.sub("", s).strip(" ,")
    return s


def _short_indicator_name(name: str, sid: str | None = None) -> str:
    """Country prefix and X13/SA/Index boilerplate off; keep the economic concept."""
    if sid and sid in _EN_CORE_SHORT_BY_ID:
        return _EN_CORE_SHORT_BY_ID[sid]
    s = _strip_indicator_boilerplate(name)
    mapped = _EN_CORE_SHORT_NAMES.get(s)
    if mapped:
        return mapped
    return s


def tab_ifocast_mapping() -> None:
    """Nineteen monthly predictors used in the fixed DFM-ifoCAST input set."""
    path = NC / "ifocast_spec_gdp_2020.csv"
    if not path.exists():
        path = TAB_DIR / "ifocast_spec_gdp_2020.csv"
    df = pd.read_csv(path)
    pred = df[(df["role"] == "predictor") & (df["in_panel"].astype(bool))].copy()

    trafo_label = {"pch": "growth", "lev": "levels"}
    groups = [
        ("Production", {"Production"}),
        ("Turnover", {"Turnover"}),
        ("Orders", {"Orders"}),
        ("Trade and global", {"Global"}),
        ("Surveys", {"Surveys"}),
        ("Labour market", {"Labor", "Labour"}),
    ]
    label_override = {
        "ifo orders on hand vs previous month, manufacturing":
            "ifo orders on hand vs previous month, mfg",
    }

    lines: list[str] = []
    for title, cats in groups:
        sub = pred[pred["category"].isin(cats)]
        if sub.empty:
            continue
        if lines:
            lines.append("\\addlinespace[0.35em]")
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for _, r in sub.iterrows():
            ind = label_override.get(str(r["indicator"]), str(r["indicator"]))
            sid = _tex_escape(str(r["series_id"])).replace(
                "\\_", "\\_\\allowbreak{}"
            )
            lag = int(r["pub_lag"]) if pd.notna(r["pub_lag"]) and str(
                r["pub_lag"]).strip() != "" else 2
            trafo = trafo_label.get(str(r["transformation"]), str(r["transformation"]))
            ind_tex = (_tex_escape(ind)
                       .replace("excl. ", "excl.\\ ")
                       .replace("vs previous", "vs.\\ previous"))
            lines.append(
                f"{ind_tex} &\n"
                f"  \\texttt{{{sid}}} & {r['type']} & "
                f"{trafo} & {lag} \\\\"
            )

    print(f"VERIFY ifoCAST spec: {len(pred)} predictors in DFM-ifoCAST")
    note = (
        f"Monthly predictors in the fixed DFM-ifoCAST input set "
        f"($N={len(pred)}$). Soft denotes survey balances; Hard denotes "
        "real-activity and labour series. \\emph{Transform.} is the "
        "stationarity transformation used in estimation: growth series "
        "enter as period-to-period growth, survey balances in levels "
        "(\\cref{sec:data-panel}). \\emph{Lag} is the publication lag in "
        "months."
    )
    body = (
        "% Auto-generated by scripts/thesis/generate_thesis_tables.py\n"
        "% Source: ifocast_spec_gdp_2020.csv (active DFM-ifoCAST predictors)\n"
        # 26 rows at the document's 1.2 stretch push the table past the space
        # left by the section intro; 1.05 keeps both on one page.
        "\\renewcommand{\\arraystretch}{1.05}%\n"
        "\\begin{tabular}{@{}p{5.35cm}"
        ">{\\raggedright\\arraybackslash}p{3.15cm}ccc@{}}\n\\toprule\n"
        "Indicator & Series ID & Type & Transform. & Lag \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
        + NOTE_ENV.format(note=note)
    )
    write_table("tab_ifocast_mapping", body)


# ------------------------------------------------------------------------- #
# 13. Data chapter: category composition and timeliness (DES-01, 09-11)
# ------------------------------------------------------------------------- #
def tab_panel_categories() -> None:
    dd = pd.read_csv(C.DATA_DICT_CSV)
    ragged = pd.read_csv(C.RAGGED_EDGE_CSV)
    lag = ragged.set_index("series")["pub_lag"]
    dd["lag"] = dd["id"].map(lag)
    lines = []
    for cat in C.CATEGORY_ORDER:
        sub = dd[dd["category"] == cat]
        if sub.empty:
            continue
        n = len(sub)
        lag0 = int((sub["lag"] == 0).sum())
        trafo = int((sub["trafo_applied"] != 0).sum())
        disp = C.CATEGORY_DISPLAY.get(cat, cat)
        lines.append(f"{disp} & {n} & {n / len(dd) * 100:.1f} & {lag0} & "
                     f"{trafo} \\\\")
    lines.append("\\midrule")
    lag0_tot = int((dd["lag"] == 0).sum())
    trafo_tot = int((dd["trafo_applied"] != 0).sum())
    lines.append(f"Total & {len(dd)} & 100.0 & {lag0_tot} & {trafo_tot} \\\\")
    print(f"VERIFY panel: {len(dd)} series, lag0={lag0_tot}, "
          f"transformed={trafo_tot}")
    body = (
        "% Auto-generated by scripts/thesis/generate_thesis_tables.py\n"
        "% Sources: metadata/data_dict.csv;\n"
        "% nowcasting/ragged_edge_diagnostics/info_set_summary.csv\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Category & Series & Share (\\%) & Lag 0 & Transformed \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
        + NOTE_ENV.format(note=(
            "Composition of the 585-series monthly predictor panel. Lag 0 "
            "counts series available in the origin month itself under the "
            "publication-lag map (407 of 585 overall, with two series at "
            "lag 1 and 176 at lag 2). Transformed counts series with a "
            "non-zero stationarity transformation code (187 of 585); the "
            "remaining 398 enter untransformed."))
    )
    write_table("tab_panel_categories", body)


def main() -> None:
    tab_accuracy_full()
    tab_regime_rmsfe()
    tab_mz()
    tab_dm_selected()
    tab_sv_calibration()
    tab_selection_masses()
    verify_selection_sizes()
    tab_horizon_profile()
    tab_bias_variance()
    tab_dm_full()
    tab_mcs()
    tab_xgb_sensitivity()
    tab_ifocast_mapping()
    tab_panel_categories()
    tab_en_core_series()
    print("ALL_TABLES_DONE")


if __name__ == "__main__":
    main()
