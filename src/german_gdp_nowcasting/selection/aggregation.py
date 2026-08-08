"""Central monthly→quarterly aggregation for the thesis nowcasting pipeline.

Economically-careful **raw-level → quarterly → re-transform** bridge used by
Elastic Net, XGBoost, the factor-augmented MLP, and any model that needs
quarterly predictors from the monthly panel.

Rationale
---------
Official quarterly index and volume statistics are built from monthly data as
the growth of the **quarterly average (or quarter-end) level** relative to the
previous quarter.  Summing *raw* index levels is not meaningful, so the
pipeline is:

    1.  Aggregate the **raw monthly levels** to a quarterly level using an
        economically-motivated rule (``AGGREGATION_RULES`` keyed by metadata):
          * level / index / volume / price / survey balances  → quarterly MEAN
          * point-in-time stocks (end-of-period)              → quarterly LAST
        Default = MEAN.  Raw levels are never summed.
    2.  **Re-transform** the quarterly level into the stationary quarterly
        series used by the models:
          * ``trafo_applied == 0`` (level-stationary: surveys, rates) → identity
          * ``trafo_applied != 0`` (growth / diff series)             → Δln
            (quarterly log-growth), with a simple first difference fallback
            when the quarterly level contains non-positive values — mirroring
            the offline transform in ``02_data_preparation``.

Real-time use (publication lags, ragged edge)
---------------------------------------------
At a forecast origin the within-quarter months that are not yet released are
completed by the existing AR(p)-BIC fill **on the transformed series**
(``ragged_edge.fill_ragged_edge_ar``).  Those filled *transformed* values are
**back-transformed** to raw levels — chaining from the last observed raw level
of each series — before the raw→quarterly→re-transform bridge is applied.  No
look-ahead is introduced: observed raw levels are used only through
``origin − pub_lag``; later within-quarter months come exclusively from the AR
fill.

The per-series offline transform (log vs. simple difference, and the number of
differences, including the rare extra-difference applied to a handful of
non-stationary series in notebook 02) is **detected empirically** by comparing
``data_df.csv`` (raw) against ``data_transformed.csv`` (transformed), so the
back-transform is the exact inverse of whatever was actually applied.
"""

from __future__ import annotations

from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import paths as _tp


# ---------------------------------------------------------------------------
# Aggregation rules (raw monthly level → quarterly level)
# ---------------------------------------------------------------------------

#: Default temporal-aggregation method when a category is not listed below.
DEFAULT_AGG_METHOD: str = "mean"

#: Economically-motivated quarterly aggregation of the *raw monthly levels*,
#: keyed by the ``category`` field of ``data_dict_enriched.csv``.
#:
#: ``"mean"`` — quarterly average of the three monthly levels.  This is the
#:   standard temporal aggregation of a within-quarter flow proxy or of a
#:   stock observed repeatedly through the quarter, and matches how official
#:   quarterly index/turnover/production/price figures are built from monthly
#:   data (quarterly average of the monthly series).
#: ``"last"`` — quarter-end (third-month) level; reserved for stocks measured
#:   at a single point in time (end-of-period).  Not used by any current
#:   category: the panel's monetary/financial series are reported as monthly
#:   averages, so a quarterly mean is the consistent aggregate.
#:
#: Every economic category in the panel aggregates by MEAN — summing raw index
#: levels is not meaningful, and a quarterly average is the economically
#: defensible default for flows (orders, turnover, production, trade),
#: prices/commodities (quarterly average price), and level-stationary surveys
#: and rates (average reading within the quarter).
AGGREGATION_RULES: dict[str, str] = {
    "Surveys": "mean",       # diffusion / balance indices — average reading
    "Orders": "mean",        # order-volume indices — flow, quarterly average
    "Turnover": "mean",      # turnover-volume indices — flow, quarterly average
    "Production": "mean",    # industrial-production indices — flow average
    "Prices": "mean",        # price indices — quarterly average price
    "Construction": "mean",  # construction output / orders — flow average
    "Trade": "mean",         # exports / imports values — flow average
    "Global": "mean",        # global activity / uncertainty indices — average
    "Commodities": "mean",   # oil & commodity prices — quarterly average price
    "Financial": "mean",     # rates / market levels reported as monthly avgs
    "Misc": "mean",          # uncertainty / residual indices — average
}


def build_aggregation_rules(
    metadata: pd.DataFrame | str | Path | None = None,
) -> pd.Series:
    """Return a per-series Series mapping ``id → aggregation method``.

    The method is looked up from :data:`AGGREGATION_RULES` by the series
    ``category`` and falls back to :data:`DEFAULT_AGG_METHOD` for unknown
    categories.
    """
    if metadata is None:
        metadata = _tp.DATA_DICT_ENRICHED_CSV
    if not isinstance(metadata, pd.DataFrame):
        metadata = pd.read_csv(metadata)
    if "id" in metadata.columns:
        metadata = metadata.set_index("id")
    cats = metadata["category"].astype(str)
    methods = cats.map(lambda c: AGGREGATION_RULES.get(c, DEFAULT_AGG_METHOD))
    methods.name = "agg_method"
    return methods


# ---------------------------------------------------------------------------
# Cached loaders (raw panel, trafo map, effective transforms)
# ---------------------------------------------------------------------------

_RAW_CACHE: pd.DataFrame | None = None
_EFF_CACHE: pd.DataFrame | None = None
_RULES_CACHE: pd.Series | None = None


def load_raw_panel() -> pd.DataFrame:
    """Load and cache the raw monthly level panel (``data_df.csv``)."""
    global _RAW_CACHE
    if _RAW_CACHE is None:
        raw = pd.read_csv(_tp.PANEL_RAW_CSV, index_col=0, parse_dates=True)
        raw.index.name = "date"
        _RAW_CACHE = raw.apply(pd.to_numeric, errors="coerce")
    return _RAW_CACHE


def get_aggregation_methods() -> pd.Series:
    """Cached per-series aggregation-method map."""
    global _RULES_CACHE
    if _RULES_CACHE is None:
        _RULES_CACHE = build_aggregation_rules()
    return _RULES_CACHE


def _detect_one(raw: pd.Series, transformed: pd.Series) -> tuple[bool, int]:
    """Detect (is_log, diff_order) that maps ``raw`` onto ``transformed``.

    Tries level vs. log base and 0–3 differences, returning the combination
    whose maximum absolute reconstruction error is smallest.  Falls back to
    ``(False, 0)`` (identity) when detection is ambiguous.
    """
    r = raw.dropna()
    bases: list[tuple[bool, pd.Series]] = [(False, r)]
    if (r > 0).all() and len(r) > 0:
        bases.append((True, np.log(r)))

    best: tuple[float, bool, int] | None = None
    for is_log, base in bases:
        x = base.copy()
        for d in range(0, 4):
            aligned = pd.concat([x, transformed], axis=1).dropna()
            if len(aligned) >= 10:
                err = float((aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs().max())
                if best is None or err < best[0]:
                    best = (err, is_log, d)
            x = x.diff()
    if best is None or best[0] > 1e-6:
        return (False, 0)
    return (best[1], best[2])


def get_effective_transforms() -> pd.DataFrame:
    """Per-series effective offline transform, detected empirically.

    Returns a DataFrame indexed by series id with boolean ``is_log`` and
    integer ``diff_order`` columns, computed once by comparing the raw and
    transformed panels.  This captures the exact transform applied in
    ``02_data_preparation`` — including log vs. simple-difference fallbacks and
    the extra differencing applied to a few non-stationary series — so the
    back-transform is its precise inverse.
    """
    global _EFF_CACHE
    if _EFF_CACHE is None:
        raw = load_raw_panel()
        transformed = pd.read_csv(
            _tp.PANEL_TRANSFORMED_CSV, index_col="date", parse_dates=True
        )
        common = [c for c in transformed.columns if c in raw.columns]
        rows: dict[str, dict] = {}
        for col in common:
            is_log, d = _detect_one(raw[col], transformed[col])
            rows[col] = {"is_log": is_log, "diff_order": d}
        _EFF_CACHE = pd.DataFrame.from_dict(rows, orient="index")
    return _EFF_CACHE


# ---------------------------------------------------------------------------
# Re-transform: quarterly raw level → stationary quarterly series
# ---------------------------------------------------------------------------

def retransform_quarterly(
    q_levels: pd.DataFrame,
    trafo_map: pd.Series,
) -> pd.DataFrame:
    """Re-transform a quarterly *raw level* matrix into the stationary series.

    * ``trafo_applied == 0`` (level-stationary)  → identity (level kept).
    * ``trafo_applied != 0`` (growth / diff)     → quarterly log-growth
      ``Δln`` (``ln x_q − ln x_{q-1}``); a simple first difference is used as a
      fallback for series whose quarterly level contains non-positive values
      (mirrors the offline transform's positivity guard).
    """
    trafo = trafo_map.reindex(q_levels.columns)
    level_cols = trafo.index[trafo.fillna(0).eq(0)].tolist()
    growth_cols = trafo.index[~trafo.fillna(0).eq(0)].tolist()

    parts: list[pd.DataFrame] = []
    if level_cols:
        parts.append(q_levels[level_cols])
    if growth_cols:
        block = q_levels[growth_cols]
        out = pd.DataFrame(index=block.index, columns=growth_cols, dtype=float)
        for col in growth_cols:
            s = block[col]
            if s.dropna().empty:
                continue
            if (s.dropna() <= 0).any():
                out[col] = s.diff()
            else:
                out[col] = np.log(s).diff()
        parts.append(out)

    quarterly = pd.concat(parts, axis=1).reindex(columns=q_levels.columns)
    quarterly.index.name = "quarter"
    return quarterly


# ---------------------------------------------------------------------------
# Raw monthly level → quarterly level
# ---------------------------------------------------------------------------

def raw_to_quarterly_levels(
    df_raw_levels: pd.DataFrame,
    methods: pd.Series | None = None,
) -> pd.DataFrame:
    """Aggregate raw monthly levels to quarterly levels via :data:`AGGREGATION_RULES`."""
    if methods is None:
        methods = get_aggregation_methods()
    methods = methods.reindex(df_raw_levels.columns).fillna(DEFAULT_AGG_METHOD)
    q_idx = df_raw_levels.index.to_period("Q")

    mean_cols = methods.index[methods.eq("mean")].tolist()
    last_cols = methods.index[methods.eq("last")].tolist()

    parts: list[pd.DataFrame] = []
    if mean_cols:
        parts.append(df_raw_levels[mean_cols].groupby(q_idx).mean())
    if last_cols:
        parts.append(df_raw_levels[last_cols].groupby(q_idx).last())

    q_levels = pd.concat(parts, axis=1).reindex(columns=df_raw_levels.columns)
    q_levels.index.name = "quarter"
    return q_levels


def monthly_to_quarterly_raw(
    columns: pd.Index | list[str],
    trafo_map: pd.Series,
    index: pd.DatetimeIndex | None = None,
    raw_panel: pd.DataFrame | None = None,
    methods: pd.Series | None = None,
) -> pd.DataFrame:
    """Full-panel raw→quarterly→re-transform bridge (no publication lags).

    Used by the Elastic-Net indicator selection, which by design ignores
    publication lags and operates on the complete history.  ``columns`` (and
    optionally ``index``) come from the transformed panel so the output is
    aligned to the same series and sample.
    """
    if raw_panel is None:
        raw_panel = load_raw_panel()
    cols = [c for c in columns if c in raw_panel.columns]
    raw = raw_panel[cols].copy()
    if index is not None:
        raw = raw.reindex(index)

    q_levels = raw_to_quarterly_levels(raw, methods=methods)
    quarterly = retransform_quarterly(q_levels, trafo_map)
    return quarterly.reindex(columns=list(columns))


# ---------------------------------------------------------------------------
# Real-time back-transform: filled transformed months → raw levels
# ---------------------------------------------------------------------------

def _invert_diffs(
    base: pd.Series,
    transformed: pd.Series,
    fill_dates: pd.DatetimeIndex,
    d: int,
) -> pd.Series:
    """In-place chained inversion of a ``d``-th difference for ``fill_dates``.

    ``Δ^d base_t = transformed_t`` ⇒
    ``base_t = transformed_t + Σ_{j=1}^{d} (−1)^{j+1} C(d,j) base_{t−j}``.
    """
    base = base.copy()
    idx = base.index
    for dt in fill_dates:
        pos = idx.get_loc(dt)
        if pos < d:
            continue
        t_val = transformed.get(dt, np.nan)
        prev = base.iloc[pos - d:pos]
        if pd.isna(t_val) or prev.isna().any():
            continue
        bt = float(t_val)
        for j in range(1, d + 1):
            bt += ((-1) ** (j + 1)) * comb(d, j) * float(base.iloc[pos - j])
        base.iloc[pos] = bt
    return base


def reconstruct_raw_levels(
    X_filled_transformed: pd.DataFrame,
    origin: pd.Period | str,
    pub_lag_map: pd.Series,
    raw_panel: pd.DataFrame | None = None,
    eff: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Reconstruct completed raw monthly levels at a forecast origin.

    For each series:
      * months ``≤ origin − pub_lag`` use the observed raw level (real-time
        information set);
      * later within-quarter months (the AR-filled ragged edge in
        ``X_filled_transformed``) are back-transformed from the filled
        transformed values, chaining from the last observed raw levels.

    ``X_filled_transformed`` is the publication-lag-masked, AR(p)-BIC-filled
    *transformed* panel (output of ``apply_pub_lag_mask`` →
    ``fill_ragged_edge_ar``).  No future raw information is used.
    """
    if raw_panel is None:
        raw_panel = load_raw_panel()
    if eff is None:
        eff = get_effective_transforms()

    origin_p = pd.Period(origin, freq="M")
    idx = X_filled_transformed.index
    out = pd.DataFrame(index=idx, columns=X_filled_transformed.columns, dtype=float)

    for col in X_filled_transformed.columns:
        lag = int(pub_lag_map.get(col, 0)) if pub_lag_map is not None else 0
        last_obs_ts = (origin_p - lag).to_timestamp()

        # Observed raw levels through the real-time horizon.
        if col in raw_panel.columns:
            raw_obs = raw_panel[col].reindex(idx)
        else:
            raw_obs = pd.Series(np.nan, index=idx)
        obs_mask = idx <= last_obs_ts
        out.loc[obs_mask, col] = raw_obs[obs_mask]

        filled = X_filled_transformed[col]
        fill_dates = idx[(idx > last_obs_ts) & filled.notna().values]
        if len(fill_dates) == 0:
            continue

        if col in eff.index:
            is_log = bool(eff.at[col, "is_log"])
            d = int(eff.at[col, "diff_order"])
        else:
            is_log, d = (False, 0)

        if d == 0:
            # transformed == raw level (log/identity with no differencing);
            # the filled transformed value *is* the (log-)level estimate.
            vals = filled.reindex(fill_dates)
            out.loc[fill_dates, col] = np.exp(vals) if is_log else vals
            continue

        base = np.log(out[col]) if is_log else out[col].copy()
        base = _invert_diffs(base, filled, fill_dates, d)
        recon = np.exp(base) if is_log else base
        out.loc[fill_dates, col] = recon.reindex(fill_dates)

    return out


def quarterly_block_realtime(
    X_filled_transformed: pd.DataFrame,
    origin: pd.Period | str,
    pub_lag_map: pd.Series,
    trafo_map: pd.Series,
    raw_panel: pd.DataFrame | None = None,
    methods: pd.Series | None = None,
    eff: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Real-time raw→quarterly→re-transform bridge for one forecast origin.

    Back-transforms the masked + AR-filled transformed panel to completed raw
    levels, aggregates them to quarterly levels (:data:`AGGREGATION_RULES`),
    and re-transforms to the stationary quarterly series consumed by the
    models.
    """
    raw_levels = reconstruct_raw_levels(
        X_filled_transformed, origin, pub_lag_map,
        raw_panel=raw_panel, eff=eff,
    )
    q_levels = raw_to_quarterly_levels(raw_levels, methods=methods)
    quarterly = retransform_quarterly(q_levels, trafo_map)
    return quarterly.reindex(columns=X_filled_transformed.columns)
