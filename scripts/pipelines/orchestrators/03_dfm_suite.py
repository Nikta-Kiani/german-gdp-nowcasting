#!/usr/bin/env python3
"""Wave 2b: DFM benchmark variants affected by the new aggregation/selection.

Runs sequentially (all CPU-heavy DFM/EN loops):
  1. rerun_sv_integrated.py      -> nowcast_results_actpn_sv_integrated_k2.csv
  2. run_blockbalanced_benchmark.py -> blockbalanced results + rmsfe
  3. run_ifocast_regime_combo.py -> nowcast_path_combo_equal.csv

ifoCAST itself is NOT rerun (fixed indicator set, monthly DFM -> unaffected by
the quarterly aggregation change).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
DFM = Path(__file__).resolve().parent.parent / "dfm"
SCRIPTS = [
    "rerun_sv_integrated.py",
    "run_blockbalanced_benchmark.py",
    "run_ifocast_regime_combo.py",
]


def main() -> None:
    """Run the sequential DFM benchmark-variant suite."""
    for s in SCRIPTS:
        t0 = time.perf_counter()
        print(f"\n===== RUN {s} =====", flush=True)
        subprocess.run([PY, s], cwd=str(DFM), check=True)
        print(f"===== DONE {s} in {(time.perf_counter()-t0)/60:.1f} min =====",
              flush=True)
    print("\nDFM SUITE DONE", flush=True)


if __name__ == "__main__":
    main()
