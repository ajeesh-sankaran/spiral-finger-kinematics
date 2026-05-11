"""
Grid computation utilities for the spiral kinematics analysis.

Provides:
  - compute_F_coupling_grid: F over (k1, k2) at fixed link lengths
  - compute_F_length_grid:    F over (Lp/Lm, Lm/Ld) at fixed coupling
  - load_or_compute:          cache wrapper that saves/loads .npz files

Each grid takes a few minutes to compute on a single core; the precomputed
results are stored in ../data/ and are loaded by default.
"""

from __future__ import annotations

import os
import time
import numpy as np

from kinematics import spiral_fidelity, LP_DEFAULT, LM_DEFAULT, LD_DEFAULT, K1_ANAT, K2_ANAT


def compute_F_coupling_grid(
    k1_range=(0.4, 1.5), k2_range=(0.3, 1.2),
    n_k1: int = 80, n_k2: int = 80,
    Lp: float = LP_DEFAULT, Lm: float = LM_DEFAULT, Ld: float = LD_DEFAULT,
    verbose: bool = True,
):
    """
    Evaluate F(k1, k2, pi/2) on an n_k1 x n_k2 grid.

    Returns
    -------
    k1_vals : (n_k1,)
    k2_vals : (n_k2,)
    F_grid  : (n_k2, n_k1) with F_grid[i, j] = F(k1_vals[j], k2_vals[i], pi/2)
    pole_x, pole_y : optimised pole positions at each grid cell
    """
    k1_vals = np.linspace(*k1_range, n_k1)
    k2_vals = np.linspace(*k2_range, n_k2)
    F_grid = np.zeros((n_k2, n_k1))
    pole_x = np.zeros((n_k2, n_k1))
    pole_y = np.zeros((n_k2, n_k1))

    t0 = time.time()
    for i, k2 in enumerate(k2_vals):
        for j, k1 in enumerate(k1_vals):
            F, P = spiral_fidelity(k1, k2, Lp, Lm, Ld, return_pole=True)
            F_grid[i, j] = F
            pole_x[i, j] = P[0]
            pole_y[i, j] = P[1]
        if verbose and (i + 1) % 10 == 0:
            done = (i + 1) * n_k1
            total = n_k1 * n_k2
            print(f"  coupling grid: {done}/{total} cells, t = {time.time() - t0:.1f}s")
    return k1_vals, k2_vals, F_grid, pole_x, pole_y


def compute_F_length_grid(
    rpm_range=(0.8, 3.5), rmd_range=(0.5, 3.0),
    n_rpm: int = 60, n_rmd: int = 60,
    k1: float = K1_ANAT, k2: float = K2_ANAT,
    verbose: bool = True,
):
    """
    Evaluate F over the phalangeal length-ratio space (Lp/Lm, Lm/Ld) at fixed
    coupling (k1, k2), with total finger length normalised to unity.

    Returns
    -------
    rpm_vals : (n_rpm,)   — Lp/Lm grid values
    rmd_vals : (n_rmd,)   — Lm/Ld grid values
    F_grid   : (n_rmd, n_rpm)
    """
    rpm_vals = np.linspace(*rpm_range, n_rpm)
    rmd_vals = np.linspace(*rmd_range, n_rmd)
    F_grid = np.zeros((n_rmd, n_rpm))

    t0 = time.time()
    for i, rmd in enumerate(rmd_vals):
        for j, rpm in enumerate(rpm_vals):
            # Recover (Lp, Lm, Ld) from ratios with total length 1
            Lm = 1.0 / (rpm + 1.0 + 1.0 / rmd)
            Lp = rpm * Lm
            Ld = Lm / rmd
            F_grid[i, j] = spiral_fidelity(k1, k2, Lp, Lm, Ld)
        if verbose and (i + 1) % 10 == 0:
            done = (i + 1) * n_rpm
            total = n_rpm * n_rmd
            print(f"  length-ratio grid: {done}/{total} cells, t = {time.time() - t0:.1f}s")
    return rpm_vals, rmd_vals, F_grid


def load_or_compute_coupling(path: str, **kwargs):
    """Load coupling-space F-grid from .npz, or compute and cache."""
    if os.path.exists(path):
        z = np.load(path)
        return (z['k1_vals'], z['k2_vals'], z['F_grid'],
                z.get('pole_x', np.zeros_like(z['F_grid'])),
                z.get('pole_y', np.zeros_like(z['F_grid'])))
    k1_vals, k2_vals, F_grid, pole_x, pole_y = compute_F_coupling_grid(**kwargs)
    np.savez(path, k1_vals=k1_vals, k2_vals=k2_vals,
             F_grid=F_grid, pole_x=pole_x, pole_y=pole_y)
    return k1_vals, k2_vals, F_grid, pole_x, pole_y


def load_or_compute_length(path: str, **kwargs):
    """Load length-ratio F-grid from .npz, or compute and cache."""
    if os.path.exists(path):
        z = np.load(path)
        return z['rpm_vals'], z['rmd_vals'], z['F_grid']
    rpm_vals, rmd_vals, F_grid = compute_F_length_grid(**kwargs)
    np.savez(path, rpm_vals=rpm_vals, rmd_vals=rmd_vals, F_grid=F_grid)
    return rpm_vals, rmd_vals, F_grid
