"""Equal-weight combination of DFM-EN, DFM-block-balanced and DFM-ifoCAST.

Averages the three saved M3 paths — no DFM re-runs. Writes
``nowcast_path_combo_equal.csv``. This is the thesis combination, not a
two-model average of ifoCAST and block-balanced alone.

Run (from the repository root):
    python scripts/pipelines/dfm/run_ifocast_regime_combo.py
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
from german_gdp_nowcasting.models.dfm import post_covid_benchmarks as B  # noqa: E402

EVAL_START, EVAL_END = "2011Q1", "2025Q4"
REGIMES = {
    "pre-COVID": ("2011Q1", "2019Q4"),
    "COVID": ("2020Q1", "2021Q4"),
    "post-COVID": ("2022Q1", "2025Q4"),
}


def m3_series(path: Path) -> pd.Series:
    """Load one model's headline M3 nowcast series."""
    return B.load_model_m3(path)


def to_eval_df(nowcast: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Align a quarterly nowcast series with actuals and errors."""
    q = pd.PeriodIndex(nowcast.index, freq="Q")
    df = pd.DataFrame({
        "quarter": q.astype(str),
        "month_in_quarter": 3,
        "nowcast": nowcast.to_numpy(float),
    })
    df["actual"] = [float(y.get(p, np.nan)) for p in q]
    df["error"] = df["nowcast"] - df["actual"]
    return df


def rmsfe(df: pd.DataFrame, q0: str, q1: str) -> float:
    """Compute RMSFE over an inclusive quarterly window."""
    idx = pd.PeriodIndex(df["quarter"], freq="Q")
    e = df["error"][(idx >= pd.Period(q0)) & (idx <= pd.Period(q1))].dropna()
    return float(np.sqrt((e ** 2).mean())) if len(e) else np.nan


def main() -> None:
    """Build and evaluate the equal-weight DFM combination."""
    y = pd.read_csv(P.GDP_TARGET_CSV, index_col="quarter").squeeze("columns")
    y.index = pd.PeriodIndex(y.index, freq="Q")

    ifo = m3_series(P.IFO_RESULTS_CSV)
    bb = m3_series(P.BLOCKBALANCED_RESULTS_CSV)
    en = m3_series(P.actpn_results_csv("en_only"))
    common = ifo.index.intersection(bb.index).intersection(en.index)
    ifo, bb, en = ifo.reindex(common), bb.reindex(common), en.reindex(common)

    combo_eq = (ifo + bb + en) / 3.0
    combo_eq.name = "combo_equal"

    out = P.COMBO_EQUAL_PATH_CSV
    df = to_eval_df(combo_eq, y)
    df.to_csv(out, index=False)
    print(f"Wrote {out.name} ({len(df)} rows)")

    rows = []
    for name, s in [("combo_equal", combo_eq), ("DFM-ifoCAST", ifo),
                    ("DFM-EN", en), ("DFM-BlockBalanced", bb)]:
        d = to_eval_df(s, y)
        row = {"model": name, "RMSFE_M3": round(rmsfe(d, EVAL_START, EVAL_END), 4)}
        for reg, (q0, q1) in REGIMES.items():
            row[f"RMSFE_{reg}"] = round(rmsfe(d, q0, q1), 4)
        rows.append(row)
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
