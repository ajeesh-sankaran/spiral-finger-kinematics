"""
Wide-domain length-ratio sweep at the anatomical coupling point.

Tests whether the viable region has any boundary at all in phalangeal
length-ratio space, by evaluating F on a logarithmically spaced grid spanning
two orders of magnitude in each ratio:

    Lp/Lm in [0.1, 10],  Lm/Ld in [0.1, 10],  at (k1, k2) = (1, 0.67)

Result reported in Section 4.4: F >= 0.9995 at every point, so there is no
length-ratio boundary of V(0.999) at anatomical coupling. This domain extends
far beyond any anatomically realisable finger and is included as a robustness
check, not as a biologically meaningful range.

Output: data/F_lengths_wide.npz

Run:  python scripts/run_wide_length_sweep.py
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

from kinematics import spiral_fidelity, K1_ANAT, K2_ANAT  # noqa: E402

OUT = os.path.join(HERE, '..', 'data', 'F_lengths_wide.npz')
N = 20
LO, HI = 0.1, 10.0


def main():
    if os.path.exists(OUT):
        print(f"{OUT} exists - skipping")
        return

    rpm = np.logspace(np.log10(LO), np.log10(HI), N)
    rmd = np.logspace(np.log10(LO), np.log10(HI), N)
    F = np.zeros((N, N))

    t0 = time.time()
    for i, b in enumerate(rmd):
        for j, a in enumerate(rpm):
            # recover link lengths from the two ratios, total length 1
            Lm = 1.0 / (a + 1.0 + 1.0 / b)
            Lp = a * Lm
            Ld = Lm / b
            F[i, j] = spiral_fidelity(K1_ANAT, K2_ANAT, Lp, Lm, Ld)
        if (i + 1) % 5 == 0:
            print(f"  row {i+1}/{N}, t = {time.time() - t0:.0f}s")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, rpm_vals=rpm, rmd_vals=rmd, F_grid=F, k1=K1_ANAT, k2=K2_ANAT)

    i, j = np.unravel_index(np.argmin(F), F.shape)
    print(f"\nF range: [{F.min():.6f}, {F.max():.6f}]")
    print(f"viable at tau=0.999 : {(F >= 0.999).mean() * 100:.1f}%")
    print(f"viable at tau=0.9995: {(F >= 0.9995).mean() * 100:.1f}%")
    print(f"worst case: Lp/Lm = {rpm[j]:.3f}, Lm/Ld = {rmd[i]:.3f}, F = {F.min():.6f}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
