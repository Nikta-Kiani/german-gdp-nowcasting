"""Core utilities for indicator selection: I/O, validation, GDP parsing, aggregation.

Shared helpers used by elastic_net_selection.py, pls_selection.py, and
selection_postprocessing.py.  Do not import from selection-method modules
here to avoid circular dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default hyperparameters (shared across EN and fixed-k)
# ---------------------------------------------------------------------------

# 40-point log grid from 0.001 to 10 covers weak-to-strong regularisation.
DEFAULT_ALPHAS: np.ndarray = np.logspace(-3, 1, 40)

# Mix of mostly-Ridge (0.1), balanced (0.5), mostly-LASSO (0.9), pure-LASSO (1.0).
DEFAULT_L1_RATIOS: tuple[float, ...] = (0.1, 0.5, 0.9, 1.0)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ElasticNetFit:
    """Container for a fitted impute-scale-ElasticNetCV result."""

    estimator: object
    alpha: float
    l1_ratio: float
    cv_mse: float
    selected_variables: list[str]
    coefficients: pd.Series


@dataclass(frozen=True)
class FixedKFit:
    """Container for a fixed-k ElasticNet result via regularisation path."""

    alpha: float
    selected_variables: list[str]
    n_selected: int


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_monthly_panel(path: str | Path) -> pd.DataFrame:
    """Load monthly predictor panel and validate its DatetimeIndex."""
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    validate_monthly_index(df.index, name=Path(path).name)
    if not df.columns.is_unique:
        raise ValueError(f"{path} contains duplicate column names.")
    return df.apply(pd.to_numeric, errors="coerce")


def load_trafo_map(path: str | Path) -> pd.Series:
    """Load trafo_applied by series id from the enriched data dictionary.

    Expects the CSV to have an 'id' column and a 'trafo_applied' column.
    """
    data_dict = pd.read_csv(path)
    required = {"id", "trafo_applied"}
    missing = required.difference(data_dict.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if data_dict["id"].duplicated().any():
        dups = data_dict.loc[data_dict["id"].duplicated(), "id"].tolist()
        raise ValueError(f"{path} contains duplicate ids: {dups[:5]}")
    return pd.to_numeric(data_dict.set_index("id")["trafo_applied"], errors="raise")


def load_pub_lag_map(path: str | Path) -> pd.Series:
    """Load publication lags (in months) by series id.

    - pub_lag counts _whole calendar months elapsed_ after the reference month ends.
    - If a statistic for T=2023-01 has pub_lag=2:
      * End of reference month: 2023-01-31
      * Add 2 whole months: February (1), March (2)
      * First day available: 2023-04-01 ("start of T+3" = 3rd month after T)
    - This is **standard in official statistics and macroeconomic real-time data**
      (see ECB SDW, FRED, Bundesbank).
    - So: Value for month T with pub_lag=P is available at the start of month T+P+1.

    Returns
    -------
    pd.Series : index = series id, values = integer publication lag.
    """
    df = pd.read_csv(path)
    required = {"id", "pub_lag"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if df["id"].duplicated().any():
        dups = df.loc[df["id"].duplicated(), "id"].tolist()
        raise ValueError(f"{path} contains duplicate ids: {dups[:5]}")
    return pd.to_numeric(df.set_index("id")["pub_lag"], errors="raise").astype(int)


# ---------------------------------------------------------------------------
# Index validators
# ---------------------------------------------------------------------------

def validate_monthly_index(index: pd.DatetimeIndex, name: str = "index") -> None:
    """Raise if the index is not a sorted, unique, monthly DatetimeIndex."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{name} must be a DatetimeIndex.")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} must be sorted.")
    if not index.is_unique:
        raise ValueError(f"{name} must not contain duplicates.")
    if index.inferred_freq not in {"MS", "M"}:
        raise ValueError(
            f"{name} must have monthly frequency; got {index.inferred_freq!r}."
        )


def validate_quarterly_index(index: pd.PeriodIndex, name: str = "index") -> None:
    """Raise if the index is not a sorted, unique, quarterly PeriodIndex."""
    if not isinstance(index, pd.PeriodIndex):
        raise TypeError(f"{name} must be a PeriodIndex.")
    if not index.freqstr.startswith("Q"):
        raise ValueError(
            f"{name} must have quarterly frequency; got {index.freqstr!r}."
        )
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} must be sorted.")
    if not index.is_unique:
        raise ValueError(f"{name} must not contain duplicates.")


# ---------------------------------------------------------------------------
# GDP vintage parsing
# ---------------------------------------------------------------------------

def parse_gdp_realtime(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Parse the GDP_realtime Excel sheet into a quarterly vintage matrix.

    Expected layout (header=None):
      Row 0      : vintage release dates in columns 1+
      Rows 11+   : col 0 = quarter string (e.g. '1991-Q1'), cols 1+ = GDP index
    """
    DATA_START = 11
    vintage_dates = pd.to_datetime(df_raw.iloc[0, 1:], errors="coerce")
    valid_vintages = vintage_dates.notna()

    raw_quarters = df_raw.iloc[DATA_START:, 0].astype(str).str.strip()
    raw_quarters = raw_quarters[
        ~raw_quarters.str.lower().isin({"nan", "", "none", "nat"})
    ]
    period_index = pd.PeriodIndex(
        raw_quarters.str.replace("-", "", regex=False),
        freq="Q",
    )
    values = (
        df_raw.iloc[DATA_START:, 1:]
        .loc[raw_quarters.index, valid_vintages]
        .apply(pd.to_numeric, errors="coerce")
    )
    values.index = period_index
    values.columns = pd.DatetimeIndex(vintage_dates[valid_vintages])
    values.index.name = "quarter"
    values.columns.name = "vintage_date"
    return values


def extract_gdp_target(
    excel_path: str | Path,
    sheet_name: str = "GDP_realtime",
) -> pd.Series:
    """Extract quarter-on-quarter log GDP growth from the vintage matrix.

    For each quarter q (rows are processed in calendar order):
      - Take the **leftmost** vintage column (earliest publication date in the
        Excel sheet) where row q is non-missing.
      - Compute 100 × ln(GDP_q / GDP_{q-1}) using GDP index levels from **that
        same** vintage column for both q and q−1.

    Hence growth rates are **internally consistent** within one vintage (no
    mixing revisions across columns).

    Returns
    -------
    pd.Series with PeriodIndex (quarterly) and name
    'gdp_qoq_log_growth_first_release', in percentage points.
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"GDP Excel file not found: {excel_path}")

    df_raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    vintages = parse_gdp_realtime(df_raw)

    growth: dict[pd.Period, float] = {}
    for quarter in vintages.index[1:]:
        row = vintages.loc[quarter]
        valid_vintage_dates = row.index[row.notna()]
        if valid_vintage_dates.empty:
            continue
        first_vintage = valid_vintage_dates[0]
        prev_quarter = quarter - 1
        if prev_quarter not in vintages.index:
            continue
        current = vintages.at[quarter, first_vintage]
        previous = vintages.at[prev_quarter, first_vintage]
        if pd.notna(current) and pd.notna(previous) and current > 0 and previous > 0:
            growth[quarter] = 100.0 * np.log(current / previous)

    target = pd.Series(
        growth, name="gdp_qoq_log_growth_first_release"
    ).sort_index()
    target.index.name = "quarter"
    validate_quarterly_index(target.index, name="GDP target index")
    return target


# ---------------------------------------------------------------------------
# Monthly → Quarterly aggregation
# ---------------------------------------------------------------------------

def monthly_to_quarterly(
    df_monthly: pd.DataFrame,
    trafo_map: pd.Series,
) -> pd.DataFrame:
    """Aggregate predictors to quarterly frequency (raw-level bridge).

    Delegates to the central :mod:`aggregation` module, which implements the
    economically-careful **raw-level → quarterly → re-transform** bridge:

      1. Aggregate the *raw monthly levels* (``data_df.csv``) to a quarterly
         level using ``AGGREGATION_RULES`` (quarterly **mean** of the three
         monthly levels for every economic category — raw index levels are
         never summed).
      2. Re-transform the quarterly level into the stationary quarterly series:
         identity for level-stationary series (``trafo_applied == 0``) and
         quarterly log-growth ``Δln`` for growth/diff series
         (``trafo_applied != 0``), with a simple-difference fallback for
         non-positive levels.

    For growth series, ``Δln(mean(raw monthly levels))`` matches the
    quarterly-average log-growth rate (a constant scale factor cancels in the
    difference).  ``df_monthly`` (the transformed panel) supplies the series set
    and monthly sample to align on; the raw panel is loaded internally.  Used
    by the Elastic-Net selection, which by design ignores publication lags and
    operates on the full history.
    """
    validate_monthly_index(df_monthly.index, name="monthly predictor index")
    missing = sorted(set(df_monthly.columns).difference(trafo_map.index))
    if missing:
        raise ValueError(
            f"trafo_applied missing for {len(missing)} series: {missing[:5]}"
        )
    from .aggregation import monthly_to_quarterly_raw

    quarterly = monthly_to_quarterly_raw(
        df_monthly.columns, trafo_map, index=df_monthly.index
    )
    quarterly.index.name = "quarter"
    validate_quarterly_index(quarterly.index, name="quarterly predictor index")
    return quarterly


# ---------------------------------------------------------------------------
# Publication-lag helpers
# ---------------------------------------------------------------------------

def pub_lag_adjusted_end_quarter(
    origin: pd.Period | str,
    pub_lag: int,
) -> pd.Period:
    """Return the last complete quarter whose data are available at *origin*
    for a series with the given publication lag.

    Parameters
    ----------
    origin : monthly forecast origin (e.g. ``pd.Period('2011-03', 'M')``).
    pub_lag : publication lag in months (0 = same-month release).

    Returns
    -------
    pd.Period (quarterly) – the last completed quarter available.

    Examples
    --------
    >>> pub_lag_adjusted_end_quarter('2011-01', 0)  # Jan 2011, lag=0
    Period('2010Q4', 'Q-DEC')
    >>> pub_lag_adjusted_end_quarter('2011-01', 2)  # Jan 2011, lag=2
    Period('2010Q3', 'Q-DEC')
    """
    origin_p = pd.Period(origin, freq="M")
    last_available_month = origin_p - pub_lag
    last_q = last_available_month.asfreq("Q")
    last_q_end_month = last_q.asfreq("M", how="end")
    if last_available_month < last_q_end_month:
        return last_q - 1
    return last_q


def available_series_at_origin(
    origin: pd.Period | str,
    series_ids: Iterable[str],
    pub_lag_map: pd.Series,
    month_in_quarter_threshold: int | None = None,
) -> list[str]:
    """Return the subset of series available at *origin* given publication lags."""
    origin_p = pd.Period(origin, freq="M")
    month_in_q = ((origin_p.month - 1) % 3) + 1

    result = []
    for sid in series_ids:
        lag = int(pub_lag_map.get(sid, 0))
        if month_in_quarter_threshold is not None and lag > month_in_quarter_threshold:
            continue
        last_q = pub_lag_adjusted_end_quarter(origin_p, lag)
        if last_q is not None:
            result.append(sid)
    return result


# ---------------------------------------------------------------------------
# Forecast-origin utilities
# ---------------------------------------------------------------------------

def make_monthly_forecast_origins(
    start: str = "2011-01",
    end: str = "2025-12",
) -> pd.PeriodIndex:
    """Return monthly forecast origins for the real-time evaluation window."""
    origins = pd.period_range(start=start, end=end, freq="M")
    if origins.empty:
        raise ValueError("Forecast-origin range is empty.")
    return origins


def training_end_quarter(origin: pd.Period | str) -> pd.Period:
    """Return the last completed quarter strictly before a monthly or quarterly origin.

    Examples
    --------
    2011-01 (M1 of 2011Q1) → 2010Q4
    2011-03 (M3 of 2011Q1) → 2010Q4
    2011Q1                 → 2010Q4
    """
    p = pd.Period(origin)
    if p.freqstr.startswith("Q"):
        return p - 1
    if p.freqstr.startswith("M"):
        return p.asfreq("Q") - 1
    raise ValueError(f"Unsupported origin frequency: {p.freqstr!r}")


# ---------------------------------------------------------------------------
# Coverage mask
# ---------------------------------------------------------------------------

def build_coverage_mask(
    X_monthly: pd.DataFrame,
    forecast_origins: Iterable[pd.Period | str],
    min_coverage: float = 0.30,
    train_start: str = "1991-01",
) -> pd.DataFrame:
    """Build a forecast-origin × series boolean data-coverage mask.

    A series passes at origin t if at least `min_coverage` of its monthly
    observations are non-missing in the expanding training window
    [train_start, training_end_quarter(t)].
    """
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must lie in (0, 1].")
    validate_monthly_index(X_monthly.index, name="monthly predictor index")
    train_start_ts = pd.Period(train_start, freq="M").to_timestamp()

    rows: dict[str, pd.Series] = {}
    for origin in forecast_origins:
        origin = pd.Period(origin)
        end_ts = training_end_quarter(origin).asfreq("M", how="end").to_timestamp()
        window = X_monthly.loc[train_start_ts:end_ts]
        if window.empty:
            raise ValueError(f"Empty training window for origin {origin}.")
        rows[str(origin)] = window.notna().mean().ge(min_coverage)

    mask = pd.DataFrame.from_dict(rows, orient="index").astype(bool)
    mask.index.name = "forecast_origin"
    return mask.reindex(columns=X_monthly.columns)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_quarterly_xy(
    X_quarterly: pd.DataFrame,
    y_quarterly: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align quarterly predictors and target on their common period index."""
    validate_quarterly_index(X_quarterly.index, name="quarterly predictor index")
    validate_quarterly_index(y_quarterly.index, name="GDP target index")
    common = X_quarterly.index.intersection(y_quarterly.index).sort_values()
    if common.empty:
        raise ValueError("No overlapping quarters between predictors and GDP target.")
    return X_quarterly.loc[common], y_quarterly.loc[common]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_selection_outputs(
    output_dir: str | Path,
    selection_matrix: pd.DataFrame,
    selection_results: dict[str, dict],
) -> None:
    """Save indicator-selection outputs for the DFM notebook.

    Files written
    -------------
    selection_matrix.csv  : forecast_origin × series_id binary matrix.
    selection_results.json: per-origin diagnostics and selected variable ids.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_matrix.to_csv(output_dir / "selection_matrix.csv")
    with (output_dir / "selection_results.json").open("w", encoding="utf-8") as f:
        json.dump(selection_results, f, indent=2)
