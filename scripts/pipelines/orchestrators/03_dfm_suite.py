#!/usr/bin/env python3
"""Run the complete DFM benchmark suite.

Runs sequentially (all CPU-heavy):
  1. run_ifocast_benchmark.py       -> fixed ifoCAST benchmark
  2. rerun_sv_integrated.py         -> integrated stochastic-volatility DFM
  3. run_blockbalanced_benchmark.py -> block-balanced benchmark
  4. run_ifocast_regime_combo.py    -> equal-weight combination
  5. run_tvp_benchmark.py           -> time-varying-parameter benchmark
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
DFM = Path(__file__).resolve().parent.parent / "dfm"
SCRIPTS = [
    "run_ifocast_benchmark.py",
    "rerun_sv_integrated.py",
    "run_blockbalanced_benchmark.py",
    "run_ifocast_regime_combo.py",
    "run_tvp_benchmark.py",
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
