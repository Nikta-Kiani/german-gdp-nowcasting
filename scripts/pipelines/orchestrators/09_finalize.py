#!/usr/bin/env python3
"""Wave 3: rebuild all evaluation tables, figures and contribution caches.

Read-only over the regenerated nowcast CSVs (except the contribution cache,
which is rebuilt with --rebuild-contrib). Run only after ALL nowcasts exist
(DFM-EN/core/pls, SV, block-balanced, combo, XGB, MLP-Factor).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
DFM = Path(__file__).resolve().parent.parent / "dfm"

STEPS = [
    (DFM, ["build_unified_evaluation.py"]),
    (DFM, ["run_horizon_profile.py"]),
    (DFM, ["run_post_covid_benchmarks.py"]),
    (DFM, ["run_all_thesis_figures.py", "--rebuild-contrib"]),
]


def main() -> None:
    """Run final evaluation, figure, and contribution-cache steps."""
    for cwd, cmd in STEPS:
        t0 = time.perf_counter()
        print(f"\n===== RUN {' '.join(cmd)} =====", flush=True)
        subprocess.run([PY, *cmd], cwd=str(cwd), check=True)
        print(f"===== DONE {cmd[0]} in {(time.perf_counter()-t0)/60:.1f} min =====",
              flush=True)
    print("\nFINALIZE DONE", flush=True)


if __name__ == "__main__":
    main()
