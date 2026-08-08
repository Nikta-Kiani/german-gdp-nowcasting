"""Build DFM-ready indicator selection matrices.

Produces three binary (origin x series) matrices used by the DFM model package:

  * ``core``     - >= ``min_votes`` consensus across EN / EN-smooth / PLS / fixed-k
  * ``en_only``  - Elastic Net raw selections (sensitivity baseline)
  * ``pls_only`` - PLS+VIP top-30 selections (sensitivity baseline)

Design choices
--------------
* Selection matrices encode **predictive relevance** from expanding-window
  Part~I procedures (training ends at the last completed quarter). They do
  **not** apply a matrix-level publication-lag gate: that would drop soft-data
  series at M1/M2 origins and break cross-method comparability (PLS selects
  mostly lag-2 series). Real-time feasibility is enforced in Part~II by
  ``ragged_edge.apply_pub_lag_mask`` plus AR fill before the DFM
  (Giannone et al., 2008; Bańbura et al., 2013).

* ``apply_release_filter`` remains available for optional robustness exercises
  (``intra_quarter_gate=True``); it is not used in the main pipeline.

* The ``topup_by_block`` function is preserved in this module for reference
  but is no longer called by the main pipeline.

References
----------
Bai, J. & Ng, S. (2008). Forecasting economic time series using targeted
    predictors. *Journal of Econometrics*, 146(2), 304-317.
Lenza, M. & Primiceri, G. E. (2022). How to estimate a vector autoregression
    after March 2020. *Journal of Applied Econometrics*, 37(4), 688-699.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .core_utils import available_series_at_origin


# ---------------------------------------------------------------------------
# Rate table across methods
# ---------------------------------------------------------------------------

def selection_rates(matrix: pd.DataFrame) -> pd.Series:
    """Share of origins in which each column is selected (1/non-zero)."""
    return matrix.astype(bool).sum(axis=0) / len(matrix)


def build_rate_table(
    matrices: Mapping[str, pd.DataFrame],
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """Per-series mean selection frequency across methods, joined with metadata.

    Parameters
    ----------
    matrices : mapping method-label -> selection matrix
    meta : DataFrame indexed by series id with at least a ``category`` column
    """
    rates = {label: selection_rates(m) for label, m in matrices.items()}
    table = pd.DataFrame(rates)
    table["mean_across_methods"] = table.mean(axis=1)
    table = table.join(meta[["category"]], how="left")
    table["category"] = table["category"].fillna("Unknown")
    return table


# ---------------------------------------------------------------------------
# Algorithm-robust core (>= min_votes of N methods)
# ---------------------------------------------------------------------------

def build_core_matrix(
    matrices: Mapping[str, pd.DataFrame],
    min_votes: int = 3,
) -> pd.DataFrame:
    """Per origin, keep series selected by >= ``min_votes`` methods."""
    aligned = _align_matrices(matrices)
    stack = np.stack([m.astype(bool).to_numpy() for m in aligned.values()])
    votes = stack.sum(axis=0)
    core = (votes >= min_votes).astype(int)
    template = next(iter(aligned.values()))
    return pd.DataFrame(core, index=template.index, columns=template.columns)


def _align_matrices(matrices: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Reindex all matrices to the union of origins and the union of columns."""
    common_idx = sorted(set().union(*[m.index for m in matrices.values()]))
    common_cols = sorted(set().union(*[m.columns for m in matrices.values()]))
    return {
        k: m.reindex(index=common_idx, columns=common_cols, fill_value=0).astype(int)
        for k, m in matrices.items()
    }


# ---------------------------------------------------------------------------
# Economic-block top-up
# ---------------------------------------------------------------------------

def derive_block_quotas(
    rate_table: pd.DataFrame,
    target_topup_size: int = 20,
    min_share_for_floor: float = 0.02,
) -> pd.Series:
    """Data-driven quotas per category, summing to roughly ``target_topup_size``.

    Each category's quota is proportional to its cumulative mean selection
    frequency across methods. Categories with share > ``min_share_for_floor``
    receive at least 1 slot. The result is a pd.Series indexed by category.
    """
    cat_mass = rate_table.groupby("category")["mean_across_methods"].sum()
    cat_share = cat_mass / cat_mass.sum()
    quota = (cat_share * target_topup_size).round().astype(int)
    floor_mask = cat_share > min_share_for_floor
    quota.loc[floor_mask & (quota < 1)] = 1
    return quota.sort_values(ascending=False)


def topup_by_block(
    core_matrix: pd.DataFrame,
    matrices: Mapping[str, pd.DataFrame],
    meta: pd.DataFrame,
    coverage_mask: pd.DataFrame,
    target_topup_size: int,
    pub_lag_map: pd.Series,
    month_in_quarter_threshold: int | None = None,
) -> pd.DataFrame:
    """Fill per-category quotas using an expanding-window rate table and a
    release-lag gate, so quota slots are never wasted on unavailable series.

    For each origin the function:

    1. Computes a rate table from the expanding window of past origins only
       (``<= origin``), eliminating look-ahead bias in selection frequencies.
    2. Derives category quotas from that origin-specific rate table.
    3. Pre-computes the joint availability set: series that pass both the
       coverage mask *and* the publication-lag check at this origin.
    4. Fills each category's quota from the top-ranked available candidates
       that are not already in the core for this origin.

    Parameters
    ----------
    core_matrix : pd.DataFrame
        Binary (origins × series) algorithm-robust core.
    matrices : Mapping[str, pd.DataFrame]
        Raw per-method selection matrices (used for expanding rate table).
    meta : pd.DataFrame
        Series metadata with at least a ``category`` column.
    coverage_mask : pd.DataFrame
        Binary mask indicating sufficient data coverage per origin/series.
    target_topup_size : int
        Approximate total number of top-up slots across all categories.
    pub_lag_map : pd.Series
        Publication lag in months, indexed by series id.
    month_in_quarter_threshold : int or None
        Passed through to ``available_series_at_origin``; None = no intra-quarter
        restriction.
    """
    out = core_matrix.copy()
    for origin in out.index:
        # --- 1. Expanding-window rate table (no look-ahead) -------------------
        past = {k: v.loc[v.index <= origin] for k, v in matrices.items()}
        rate_t = build_rate_table(past, meta)
        quotas_t = derive_block_quotas(rate_t, target_topup_size=target_topup_size)

        meta_cat = rate_t["category"]
        rank = rate_t["mean_across_methods"]

        # --- 2. Joint availability: coverage mask ∩ release filter ------------
        if origin not in coverage_mask.index:
            continue
        cov_avail = set(coverage_mask.columns[coverage_mask.loc[origin].astype(bool)])
        rel_avail = set(
            available_series_at_origin(
                origin,
                list(cov_avail),
                pub_lag_map,
                month_in_quarter_threshold=month_in_quarter_threshold,
            )
        )

        # --- 3. Fill quotas using only truly available candidates -------------
        selected = set(out.columns[out.loc[origin].astype(bool)])
        for cat, quota in quotas_t.items():
            candidates = (
                rank.loc[meta_cat == cat]
                .sort_values(ascending=False)
                .index.tolist()
            )
            in_cat = [sid for sid in selected if meta_cat.get(sid) == cat]
            need = int(quota) - len(in_cat)
            if need <= 0:
                continue
            for sid in candidates:
                if need == 0:
                    break
                if sid in selected or sid not in rel_avail or sid not in out.columns:
                    continue
                out.at[origin, sid] = 1
                selected.add(sid)
                need -= 1
    return out


# ---------------------------------------------------------------------------
# Real-time release-lag filter (reuses core_utils helper)
# ---------------------------------------------------------------------------

def apply_release_filter(
    matrix: pd.DataFrame,
    pub_lag_map: pd.Series,
    *,
    intra_quarter_gate: bool = False,
    month_in_quarter_threshold: int | None = None,
) -> pd.DataFrame:
    """Zero-out entries whose publication lag makes them unavailable at origin.

    **Not used in the main DFM input pipeline.** Post-hoc application with
    ``intra_quarter_gate=True`` removes soft-data series at M1/M2 monthly
    origins and can drive median Jaccard overlap to zero across methods that
    select different lag profiles (e.g. PLS lag-2 vs EN hard data).

    Parameters
    ----------
    intra_quarter_gate : if True, at M1/M2/M3 only retain series with
        pub_lag <= month-in-quarter cap (hard/soft split). If False, only the
        quarterly ``pub_lag_adjusted_end_quarter`` check applies.
    month_in_quarter_threshold : fixed pub_lag cap when intra_quarter_gate is
        True; if None, derived per origin as ``((month - 1) % 3)``.
    """
    out = matrix.copy()
    for origin in matrix.index:
        sel = matrix.columns[matrix.loc[origin].astype(bool)].tolist()
        if not sel:
            continue

        lag_cap: int | None = None
        if intra_quarter_gate:
            origin_p = pd.Period(origin, freq="M")
            if month_in_quarter_threshold is None:
                lag_cap = ((origin_p.month - 1) % 3)
            else:
                lag_cap = month_in_quarter_threshold

        keep = set(
            available_series_at_origin(
                origin,
                sel,
                pub_lag_map,
                month_in_quarter_threshold=lag_cap,
            )
        )
        drop = [c for c in sel if c not in keep]
        if drop:
            out.loc[origin, drop] = 0
    return out


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_dfm_input_sets(
    matrices: Mapping[str, pd.DataFrame],
    meta: pd.DataFrame,
    coverage_mask: pd.DataFrame,
    pub_lag_map: pd.Series,
    min_votes: int = 3,
    en_label: str = "EN raw",
    pls_label: str = "PLS",
) -> dict[str, pd.DataFrame]:
    """Build all three DFM input matrices in one call.

    Returns a dict with keys ``core``, ``en_only``, ``pls_only``.

    * ``core``    - >=``min_votes`` consensus (no matrix-level release filter).
    * ``en_only`` - raw EN selection matrix.
    * ``pls_only``- raw PLS+VIP matrix.

    ``pub_lag_map`` is accepted for API compatibility (e.g. optional top-up)
    but is not applied here; Part~II nowcasting uses it via ``ragged_edge``.

    Additional diagnostic keys
    --------------------------
    ``_rate_table_diagnostic`` : pd.DataFrame
        Full-sample rate table (all origins). Present only for post-hoc
        inspection; not used in model construction.
    """
    core = build_core_matrix(matrices, min_votes=min_votes)
    rate_table_diag = build_rate_table(matrices, meta)
    aligned = _align_matrices(matrices)
    return {
        "core": core,
        "en_only": aligned[en_label],
        "pls_only": aligned[pls_label],
        "_rate_table_diagnostic": rate_table_diag,
    }
