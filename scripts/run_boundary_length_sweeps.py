"""
Compute length-ratio spiral fidelity F at coupling points other than the
anatomical one, to verify that the length-ratio insensitivity reported in
Section 4.4 is robust across the viable region of coupling space.

The paper reports results at three coupling points:
  (k1, k2) = (1.00, 0.67)  — anatomical, see data/F60_lengths.npz
  (k1, k2) = (1.25, 0.67)  — near the upper k1 boundary at anatomical k2
  (k1, k2) = (1.00, 1.00)  — near the upper k2 boundary at anatomical k1

The latter two are produced by this script and stored as:
  data/F60_boundary_k1_1p25_k2_0p67.npz
  data/F60_boundary_k1_1p00_k2_1p00.npz

Run:  python scripts/run_boundary_length_sweeps.py
"""

import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

from grids import compute_F_length_grid  # noqa: E402

DATA_DIR = os.path.join(HERE, '..', 'data')

BOUNDARY_POINTS = [
    (1.25, 0.67, 'F60_boundary_k1_1p25_k2_0p67.npz',
     'near upper k1 boundary at anatomical k2'),
    (1.00, 1.00, 'F60_boundary_k1_1p00_k2_1p00.npz',
     'near upper k2 boundary at anatomical k1'),
]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for k1, k2, fname, descr in BOUNDARY_POINTS:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            print(f"[{fname}] exists — skipping")
            continue
        print(f"\nComputing (k1={k1}, k2={k2}) — {descr}")
        print(f"  Estimated time: ~20 min on a single core")
        t0 = time.time()
        rpm, rmd, F = compute_F_length_grid(
            rpm_range=(0.8, 3.5), rmd_range=(0.5, 3.0),
            n_rpm=60, n_rmd=60, k1=k1, k2=k2,
        )
        np.savez(path, F_grid=F, rpm_vals=rpm, rmd_vals=rmd, k1=k1, k2=k2)
        print(f"  Saved {path}   (wall time {time.time() - t0:.0f}s)")

        # Summary statistics, matching Table in §4.4 of the paper
        viable_pct = (F >= 0.999).mean() * 100
        i_anat = np.argmin(np.abs(rmd - 1.0))
        j_anat = np.argmin(np.abs(rpm - 2.0))
        i_gold = np.argmin(np.abs(rmd - 1.618))
        j_gold = np.argmin(np.abs(rpm - 1.618))
        print(f"  Stats: min F = {F.min():.4f}, mean F = {F.mean():.4f}, "
              f"viable {viable_pct:.1f}%")
        print(f"  F at 2:1:1 = {F[i_anat, j_anat]:.4f}, "
              f"F at golden = {F[i_gold, j_gold]:.4f}")


if __name__ == "__main__":
    main()
