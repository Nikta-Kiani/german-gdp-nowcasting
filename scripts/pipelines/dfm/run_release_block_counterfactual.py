"""Post-COVID counterfactual release-block experiment for DFM-EN.

At M2 and M3, evaluate the exhaustive 2x2 information-set design:

    both_frozen   all monthly predictors held at their M1 information set
    other_only    non-hard complement updated; hard activity held at M1
    hard_only     hard-activity categories updated; other series held at M1
    full          both blocks updated as in the observed data flow

The two blocks partition the selected panel. Order-averaged Shapley
contributions therefore add exactly to the observed M1-to-M2/M3 change in
squared forecast error, including any DFM interaction between the blocks.

Run from the repository root after configuring the private data paths:
    python scripts/pipelines/dfm/run_release_block_counterfactual.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for _parent in Path(__file__).resolve().parents:
    _src = _parent / "src"
    if (_src / "german_gdp_nowcasting").is_dir():
        sys.path.insert(0, str(_src))
        break
else:
    raise RuntimeError("Could not locate src/german_gdp_nowcasting above this script.")

from german_gdp_nowcasting.config import paths as P  # noqa: E402  # pyright: ignore[reportMissingImports]
from german_gdp_nowcasting.models.dfm.nowcast_utils import nowcast_for_origin  # noqa: E402  # pyright: ignore[reportMissingImports]
from german_gdp_nowcasting.models.dfm.ragged_edge import freeze_release_block  # noqa: E402  # pyright: ignore[reportMissingImports]
from german_gdp_nowcasting.selection.core_utils import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    load_monthly_panel,
    load_pub_lag_map,
)
from german_gdp_nowcasting.visualization.nowcast_plots import setup_style  # noqa: E402  # pyright: ignore[reportMissingImports]

EVAL_START, EVAL_END = "2022Q1", "2025Q4"
HARD_ACTIVITY_CATEGORIES = {
    "Production",
    "Turnover",
    "Trade",
    "Orders",
    "Construction",
}
STATE_ORDER = ["both_frozen", "other_only", "hard_only", "full"]
STATE_LABELS = {
    "both_frozen": "Both blocks frozen",
    "other_only": "Non-hard block only",
    "hard_only": "Hard-activity block only",
    "full": "Observed full update",
}
# Thesis Fig. 8.5 palette: steel-blue = frozen M1 set; amber = non-hard
# complement; teal = hard-activity treatment; ink = observed full update.
COL_FROZEN = "#3B6FA0"
COL_SOFT = "#D4940A"
COL_HARD = "#1A8A6C"
COL_FULL = "#1A2332"
STATE_COLORS = {
    "both_frozen": COL_FROZEN,
    "other_only": COL_SOFT,
    "hard_only": COL_HARD,
    "full": COL_FULL,
}
BOOTSTRAP_REPS = 20_000
BOOTSTRAP_MEAN_BLOCK = 4
SEED = 20260813


def _load_saved_results(path: Path) -> pd.DataFrame:
    """Load the observed DFM-EN path with one unique row per monthly origin."""
    df = pd.read_csv(path)
    required = {"monthly_origin", "quarter", "month_in_quarter", "nowcast", "actual"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if df["monthly_origin"].duplicated().any():
        raise ValueError("Observed DFM-EN results contain duplicate monthly origins")
    return df.set_index("monthly_origin", drop=False)


def _selected_columns_for_quarter(
    selection_matrix: pd.DataFrame,
    quarter: pd.Period,
) -> list[str]:
    """Return the fixed within-quarter indicator set and verify invariance."""
    m1 = quarter.asfreq("M", how="start")
    keys = [str(m1 + i) for i in range(3)]
    missing = [key for key in keys if key not in selection_matrix.index]
    if missing:
        raise KeyError(f"Selection matrix lacks origins: {missing}")
    rows = selection_matrix.loc[keys].astype(bool)
    if not (rows.iloc[0].equals(rows.iloc[1]) and rows.iloc[0].equals(rows.iloc[2])):
        raise ValueError(f"Indicator selection changes within {quarter}; blocks not held fixed")
    return rows.columns[rows.iloc[0]].tolist()


def _fit_hybrid(
    X_monthly: pd.DataFrame,
    y_quarterly: pd.Series,
    selected_cols: list[str],
    origin: pd.Period,
    pub_lag_map: pd.Series,
) -> float:
    """Fit one DFM-EN counterfactual using an adjusted release calendar."""
    result = nowcast_for_origin(
        X_monthly=X_monthly,
        y_quarterly=y_quarterly,
        selected_cols=selected_cols,
        origin=origin,
        k_factors=2,
        factor_order=2,
        idiosyncratic_ar1=True,
        maxiter=200,
        pub_lag_map=pub_lag_map,
        fill_method="ar_bic",
    )
    return float(result["nowcast"])


def run_experiment(
    selection_matrix: pd.DataFrame,
    X_monthly: pd.DataFrame,
    y_quarterly: pd.Series,
    pub_lag_map: pd.Series,
    category_map: pd.Series,
    observed: pd.DataFrame,
) -> pd.DataFrame:
    """Run the two hybrid cells and combine them with saved M1/full forecasts."""
    records: list[dict[str, object]] = []
    quarters = pd.period_range(EVAL_START, EVAL_END, freq="Q")

    for quarter in quarters:
        selected = _selected_columns_for_quarter(selection_matrix, quarter)
        hard = [sid for sid in selected if category_map.get(sid) in HARD_ACTIVITY_CATEGORIES]
        other = [sid for sid in selected if sid not in set(hard)]
        if not hard or not other:
            raise ValueError(f"{quarter} does not contain both release blocks")

        m1 = quarter.asfreq("M", how="start")
        m1_row = observed.loc[str(m1)]
        actual = float(m1_row["actual"])

        for month_in_quarter in (2, 3):
            origin = m1 + (month_in_quarter - 1)
            full_row = observed.loc[str(origin)]

            lag_other_only = freeze_release_block(
                pub_lag_map, hard, origin=origin, freeze_origin=m1
            )
            lag_hard_only = freeze_release_block(
                pub_lag_map, other, origin=origin, freeze_origin=m1
            )
            print(
                f"{quarter} M{month_in_quarter}: "
                f"N={len(selected)} (hard={len(hard)}, other={len(other)})",
                flush=True,
            )
            hybrid_nowcasts = {
                "other_only": _fit_hybrid(
                    X_monthly, y_quarterly, selected, origin, lag_other_only
                ),
                "hard_only": _fit_hybrid(
                    X_monthly, y_quarterly, selected, origin, lag_hard_only
                ),
            }
            state_nowcasts = {
                "both_frozen": float(m1_row["nowcast"]),
                **hybrid_nowcasts,
                "full": float(full_row["nowcast"]),
            }
            for state, nowcast in state_nowcasts.items():
                records.append(
                    {
                        "quarter": str(quarter),
                        "monthly_origin": str(origin),
                        "month_in_quarter": month_in_quarter,
                        "state": state,
                        "nowcast": nowcast,
                        "actual": actual,
                        "error": nowcast - actual,
                        "squared_error": (nowcast - actual) ** 2,
                        "n_indicators": len(selected),
                        "n_hard_activity": len(hard),
                        "n_non_hard_other": len(other),
                    }
                )
    return pd.DataFrame(records)


def build_state_summary(results: pd.DataFrame) -> pd.DataFrame:
    """RMSFE and bias-variance moments for every counterfactual state."""
    rows = []
    for (horizon, state), group in results.groupby(["month_in_quarter", "state"]):
        errors = group["error"].to_numpy(float)
        rows.append(
            {
                "horizon": f"M{horizon}",
                "state": state,
                "N": len(errors),
                "RMSFE": np.sqrt(np.mean(errors**2)),
                "MSE": np.mean(errors**2),
                "bias": np.mean(errors),
                "error_variance": np.var(errors, ddof=0),
            }
        )
    out = pd.DataFrame(rows)
    out["state"] = pd.Categorical(out["state"], STATE_ORDER, ordered=True)
    return out.sort_values(["horizon", "state"]).reset_index(drop=True)


def _bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float]:
    """Stationary-block bootstrap interval for a mean loss contribution.

    The resampling unit is the complete quarter-level counterfactual vector
    before this function is called, so state comparisons remain paired.
    Circular blocks with expected length four preserve short-run dependence.
    """
    rng = np.random.default_rng(SEED)
    n = len(values)
    restart_probability = 1.0 / BOOTSTRAP_MEAN_BLOCK
    draws = np.empty(BOOTSTRAP_REPS)
    for rep in range(BOOTSTRAP_REPS):
        indices = np.empty(n, dtype=int)
        current = 0
        for position in range(n):
            if position == 0 or rng.random() < restart_probability:
                current = int(rng.integers(n))
            else:
                current = (current + 1) % n
            indices[position] = current
        draws[rep] = values[indices].mean()
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def build_loss_decomposition(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute quarter-level and mean Shapley contributions to squared error."""
    wide = results.pivot(
        index=["quarter", "month_in_quarter"],
        columns="state",
        values="squared_error",
    )
    missing = set(STATE_ORDER).difference(wide.columns)
    if missing:
        raise ValueError(f"Counterfactual states missing from results: {sorted(missing)}")

    quarter = wide.reset_index()
    quarter["hard_given_frozen"] = quarter["hard_only"] - quarter["both_frozen"]
    quarter["hard_given_other"] = quarter["full"] - quarter["other_only"]
    quarter["other_given_frozen"] = quarter["other_only"] - quarter["both_frozen"]
    quarter["other_given_hard"] = quarter["full"] - quarter["hard_only"]
    quarter["hard_shapley"] = (
        quarter["hard_given_frozen"] + quarter["hard_given_other"]
    ) / 2
    quarter["other_shapley"] = (
        quarter["other_given_frozen"] + quarter["other_given_hard"]
    ) / 2
    quarter["interaction"] = (
        quarter["full"]
        - quarter["hard_only"]
        - quarter["other_only"]
        + quarter["both_frozen"]
    )
    quarter["total_mse_change"] = quarter["full"] - quarter["both_frozen"]
    np.testing.assert_allclose(
        quarter["hard_shapley"] + quarter["other_shapley"],
        quarter["total_mse_change"],
        rtol=1e-10,
        atol=1e-12,
    )

    rows = []
    for horizon, group in quarter.groupby("month_in_quarter"):
        row: dict[str, object] = {"horizon": f"M{horizon}", "N": len(group)}
        for column in (
            "hard_given_frozen",
            "hard_given_other",
            "other_given_frozen",
            "other_given_hard",
            "hard_shapley",
            "other_shapley",
            "interaction",
            "total_mse_change",
        ):
            values = group[column].to_numpy(float)
            low, high = _bootstrap_mean_ci(values)
            row[column] = float(values.mean())
            row[f"{column}_ci_low"] = low
            row[f"{column}_ci_high"] = high
        rows.append(row)
    return quarter, pd.DataFrame(rows)


def make_figure(
    state_summary: pd.DataFrame,
    decomposition: pd.DataFrame,
    save: Path,
) -> None:
    """Paired-bar RMSFE chart for the four counterfactual information sets.

    At each origin the left pair holds hard activity frozen and the right
    pair lets it update. Matching heights inside a pair are the result.
    The Shapley allocation is reported in the thesis table, not here.
    ``decomposition`` is accepted for call-site compatibility and unused.
    """
    del decomposition
    setup_style()
    rmsfe = state_summary.pivot(index="horizon", columns="state", values="RMSFE")
    m1 = float(rmsfe.loc["M2", "both_frozen"])
    origins = ["M2", "M3"]
    offsets = {
        "both_frozen": -0.30,
        "other_only": -0.12,
        "hard_only": 0.12,
        "full": 0.30,
    }
    bar_w = 0.16
    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    ax.axhline(m1, color=COL_FROZEN, linewidth=1.1, linestyle=":", zorder=2)
    ax.text(1.48, m1 + 0.012, f"M1  {m1:.3f}", ha="left", va="bottom",
            fontsize=8.0, color=COL_FROZEN)
    for i, horizon in enumerate(origins):
        for state in STATE_ORDER:
            xi = i + offsets[state]
            value = float(rmsfe.loc[horizon, state])
            ax.bar(xi, value, width=bar_w, color=STATE_COLORS[state],
                   edgecolor="white", linewidth=0.4, zorder=3,
                   label=STATE_LABELS[state] if i == 0 else None)
            ax.text(xi, value + 0.008, f"{value:.3f}", ha="center",
                    va="bottom", fontsize=7.2, color="#1F2937")
        ax.text(i + (offsets["both_frozen"] + offsets["other_only"]) / 2,
                -0.035, "hard frozen", ha="center", va="top",
                fontsize=8.0, color="#64748B")
        ax.text(i + (offsets["hard_only"] + offsets["full"]) / 2,
                -0.035, "hard updates", ha="center", va="top",
                fontsize=8.0, color="#64748B")
    ax.set_xticks([0, 1], origins)
    ax.set_xlim(-0.55, 1.72)
    ax.set_ylim(0.0, 0.56)
    ax.set_xlabel("Information set")
    ax.set_ylabel("RMSFE (percentage points)")
    ax.legend(frameon=False, fontsize=8.0, loc="upper left", ncol=2)
    fig.tight_layout()
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the experiment and write reproducible result artifacts."""
    X_monthly = load_monthly_panel(P.PANEL_TRANSFORMED_CSV)
    pub_lag_map = load_pub_lag_map(P.PUB_LAG_CSV)
    y_quarterly = pd.read_csv(P.GDP_TARGET_CSV, index_col="quarter").squeeze("columns")
    y_quarterly.index = pd.PeriodIndex(y_quarterly.index, freq="Q")
    selection = pd.read_csv(P.EN_ONLY_MATRIX_CSV, index_col="forecast_origin").astype(int)
    metadata = pd.read_csv(P.DATA_DICT_ENRICHED_CSV, usecols=["id", "category"])
    category_map = metadata.set_index("id")["category"]
    observed = _load_saved_results(P.actpn_results_csv("en_only"))

    results = run_experiment(
        selection, X_monthly, y_quarterly, pub_lag_map, category_map, observed
    )
    state_summary = build_state_summary(results)
    quarter_decomp, mean_decomp = build_loss_decomposition(results)

    out = P.OUT_NOWCASTING
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(P.RELEASE_BLOCK_RESULTS_CSV, index=False)
    state_summary.to_csv(P.RELEASE_BLOCK_STATES_CSV, index=False)
    quarter_decomp.to_csv(
        out / "release_block_counterfactual_quarterly_decomposition.csv", index=False
    )
    mean_decomp.to_csv(P.RELEASE_BLOCK_DECOMPOSITION_CSV, index=False)
    make_figure(state_summary, mean_decomp, P.RELEASE_BLOCK_FIG)

    print("\nCounterfactual state summary:")
    print(state_summary.round(4).to_string(index=False))
    print("\nShapley loss decomposition:")
    print(mean_decomp.round(4).to_string(index=False))
    print(f"\nSaved figure: {P.RELEASE_BLOCK_FIG}")


if __name__ == "__main__":
    main()
