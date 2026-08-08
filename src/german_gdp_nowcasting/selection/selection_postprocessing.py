"""Post-processing utilities for indicator selection results.

Functions
---------
compute_selection_stability : per-series selection frequency and timing metrics.
apply_frequency_smoothing   : persistence filter over pooled monthly vintages
                               in the last ``window_quarters`` training-end
                               quarters (mean raw selection ≥ ``min_freq``);
                               smoothed rows need not be subsets of raw rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Selection stability analysis
# ---------------------------------------------------------------------------

def compute_selection_stability(
    selection_matrix: pd.DataFrame,
    data_dict: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute per-series selection frequency and stability metrics.

    Parameters
    ----------
    selection_matrix : binary DataFrame (origins × series).
    data_dict : optional enriched data dictionary for category labels.

    Returns
    -------
    pd.DataFrame indexed by series id with columns:
        selection_freq  : fraction of origins the series was selected
        n_selected      : count of origins where selected
        first_origin    : first origin where selected (or NaN)
        last_origin     : last origin where selected (or NaN)
        category        : category label (if data_dict provided)
    """
    freq = selection_matrix.mean(axis=0).rename("selection_freq")
    count = selection_matrix.sum(axis=0).rename("n_selected")

    def _first_origin(col: pd.Series) -> str | float:
        """Return the first origin selecting a series, or NaN."""
        idx = col[col == 1].index
        return idx[0] if len(idx) > 0 else np.nan

    def _last_origin(col: pd.Series) -> str | float:
        """Return the last origin selecting a series, or NaN."""
        idx = col[col == 1].index
        return idx[-1] if len(idx) > 0 else np.nan

    first_sel = selection_matrix.apply(_first_origin, axis=0).rename("first_origin")
    last_sel = selection_matrix.apply(_last_origin, axis=0).rename("last_origin")

    stability = pd.concat([freq, count, first_sel, last_sel], axis=1)

    if data_dict is not None:
        if "id" in data_dict.columns and "category" in data_dict.columns:
            cat = data_dict.set_index("id")["category"]
        elif "category" in data_dict.columns:
            cat = data_dict["category"]
        else:
            cat = None
        if cat is not None:
            stability = stability.join(cat.rename("category"), how="left")

    return stability.sort_values("selection_freq", ascending=False)


# ---------------------------------------------------------------------------
# Frequency smoothing post-processor
# ---------------------------------------------------------------------------

def apply_frequency_smoothing(
    selection_matrix: pd.DataFrame,
    window_quarters: int = 2,
    min_freq: float = 0.5,
) -> pd.DataFrame:
    """Smooth a binary selection matrix by requiring consistent recent selection.

    Elastic Net selections can be "jumpy" across consecutive quarterly
    windows — a variable selected in 2011Q1 may drop out in 2011Q2 due to a
    small data update.  For the DFM, a stable factor structure is preferable.

    For each **monthly** forecast origin, the routine pools every raw EN row
    whose training-end quarter lies in the last ``window_quarters`` distinct
    end-quarters, then sets series *j* to 1 iff the **column mean** of raw
    selections over those pooled rows is ≥ ``min_freq``.  Because the pool
    contains **many vintages** (not only the current row), a series can be 1
    in the smoothed matrix while 0 in raw at the same origin if it was
    selected often enough on other months in the window—and vice versa.  The
    smoothed row is therefore **not** a subset of the current raw row; the
    **count of selected indicators per origin can increase or decrease**.  The
    operation is purely backward-looking and introduces no future information.

    Parameters
    ----------
    selection_matrix : pd.DataFrame
        Binary matrix (forecast_origins × series_ids), index strings like
        "2011-01" (monthly PeriodIndex as strings).
    window_quarters : int
        Number of distinct training-end-quarters to look back (default 2 = half
        year).  A window of 2 quarters (~6 pooled monthly origins) keeps the
        model agile enough to respond to structural breaks within one to two
        quarters (e.g. the 2020 pandemic, the 2022 energy shock) while still
        filtering out single-month noise.  The previous default of 4 (one full
        year) was too inertial for rapid regime changes.
    min_freq : float
        Threshold on the **mean raw binary value** over all pooled monthly
        rows in the look-back window (default ``0.5``).  Larger values require
        more persistent selection across vintages in the pool.

    Returns
    -------
    pd.DataFrame of the same shape and dtype as selection_matrix (int 0/1).
    """
    if not 0 < min_freq <= 1:
        raise ValueError("min_freq must lie in (0, 1].")
    if window_quarters < 1:
        raise ValueError("window_quarters must be >= 1.")

    origins = pd.PeriodIndex(selection_matrix.index, freq="M")

    # Map each origin string → its training-end-quarter string
    end_quarters = pd.Series(
        [str(pd.Period(o).asfreq("Q") - 1) for o in origins],
        index=selection_matrix.index,
    )

    # Unique training-end-quarters in calendar order (preserving insertion order)
    unique_end_qs: list[str] = list(dict.fromkeys(end_quarters.values))

    smoothed_rows: dict[str, pd.Series] = {}
    for origin_str, row in selection_matrix.iterrows():
        end_q_str = end_quarters.loc[origin_str]
        end_q_pos = unique_end_qs.index(end_q_str)

        # Look back window_quarters distinct end-quarters (inclusive)
        start_pos = max(0, end_q_pos - window_quarters + 1)
        window_qs = set(unique_end_qs[start_pos: end_q_pos + 1])

        # Collect all origin rows whose training-end-quarter falls in the window
        window_origin_mask = end_quarters.isin(window_qs)
        window_slice = selection_matrix.loc[window_origin_mask]

        # A series passes if its mean selection across those rows ≥ min_freq
        smoothed_rows[origin_str] = window_slice.mean(axis=0).ge(min_freq).astype(int)

    smoothed = pd.DataFrame.from_dict(smoothed_rows, orient="index")
    smoothed.index.name = selection_matrix.index.name
    return smoothed.reindex(columns=selection_matrix.columns).fillna(0).astype(int)
