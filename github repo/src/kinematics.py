"""
Core kinematics functions for the coupled three-link finger model.

Implements:
  - Forward kinematics of the planar three-link chain with linear joint coupling
  - Spiral fit (R^2 of ln r vs psi about an optimised pole)
  - Pole optimisation by coarse grid search + Nelder-Mead refinement
  - Spiral fidelity F(k1, k2, theta_max) as defined in eq. (2.11) of the paper

Sanity checkpoint
-----------------
At the anatomical operating point (k1 = 1.0, k2 = 0.67, 2:1:1 phalangeal lengths):
  F = 0.9997
  optimised pole P = (0.0505, 0.1776)
  tangent angle alpha = 75.6 degrees

A working installation should reproduce these values to four decimal places.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

# ── Default anatomical constants ───────────────────────────────────────────
LP_DEFAULT = 0.5    # proximal phalanx (normalised, 2:1:1)
LM_DEFAULT = 0.25   # middle phalanx
LD_DEFAULT = 0.25   # distal phalanx
THETA_MAX_DEFAULT = np.pi / 2   # physiological MCP flexion range, 90 degrees
K1_ANAT = 1.0       # MCP-PIP coupling at the anatomical operating point
K2_ANAT = 0.67      # PIP-DIP coupling at the anatomical operating point


# ── Forward kinematics ────────────────────────────────────────────────────

def trajectory(k1: float, k2: float,
               Lp: float = LP_DEFAULT, Lm: float = LM_DEFAULT, Ld: float = LD_DEFAULT,
               theta_max: float = THETA_MAX_DEFAULT, N: int = 200):
    """
    Compute the fingertip trajectory of the coupled three-link chain.

    The MCP is at the origin with the finger initially extended along +x.
    Cumulative joint angles are phi_1 = theta_p, phi_2 = (1+k1) theta_p,
    phi_3 = (1+k1+k1*k2) theta_p.

    Parameters
    ----------
    k1, k2 : coupling coefficients (theta_m = k1 * theta_p; theta_d = k2 * theta_m)
    Lp, Lm, Ld : phalangeal lengths (must sum to 1)
    theta_max : maximum MCP flexion angle in radians
    N : number of trajectory sample points

    Returns
    -------
    x, y : np.ndarray of shape (N,) — fingertip coordinates
    """
    tp = np.linspace(1e-3, theta_max, N)
    w2 = 1.0 + k1
    w3 = 1.0 + k1 + k1 * k2
    x = Lp * np.cos(tp) + Lm * np.cos(w2 * tp) + Ld * np.cos(w3 * tp)
    y = Lp * np.sin(tp) + Lm * np.sin(w2 * tp) + Ld * np.sin(w3 * tp)
    return x, y


# ── Spiral fit at a fixed pole ────────────────────────────────────────────

def fit_at_pole(px: float, py: float, x: np.ndarray, y: np.ndarray):
    """
    Fit ln r = mu * psi + a0 over the trajectory, with pole at (px, py).

    Returns
    -------
    R2 : float — coefficient of determination of the log-linear fit
    mu : float — spiral cotangent parameter (slope of ln r vs psi)
    a0 : float — spiral scale parameter (intercept)

    Returns (-inf, nan, nan) if the trajectory passes too close to the pole.
    """
    u = x - px
    v = y - py
    r2 = u * u + v * v
    if np.any(r2 < 1e-10):
        return -np.inf, np.nan, np.nan
    lnr = 0.5 * np.log(r2)
    psi = np.unwrap(np.arctan2(v, u))      # remove 2*pi jumps
    A = np.vstack([psi, np.ones_like(psi)]).T
    coeffs, *_ = np.linalg.lstsq(A, lnr, rcond=None)
    mu, a0 = coeffs
    pred = A @ coeffs
    ss_res = np.sum((lnr - pred) ** 2)
    ss_tot = np.sum((lnr - lnr.mean()) ** 2)
    R2 = 1.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return R2, mu, a0


def _R2_grid_at(gx: np.ndarray, gy: np.ndarray, x: np.ndarray, y: np.ndarray):
    """
    Vectorised R^2 evaluation over a grid of candidate pole positions.

    Returns a 2D array R2[i, j] for poles (gx[j], gy[i]).
    """
    GX, GY = np.meshgrid(gx, gy)
    u = x[None, None, :] - GX[:, :, None]
    v = y[None, None, :] - GY[:, :, None]
    r2 = u * u + v * v
    invalid = np.any(r2 < 1e-10, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        lnr = 0.5 * np.log(r2)
        psi = np.unwrap(np.arctan2(v, u), axis=-1)
        psi_mean = psi.mean(axis=-1, keepdims=True)
        lnr_mean = lnr.mean(axis=-1, keepdims=True)
        dpsi = psi - psi_mean
        dlnr = lnr - lnr_mean
        Sxx = np.sum(dpsi * dpsi, axis=-1)
        Sxy = np.sum(dpsi * dlnr, axis=-1)
        Syy = np.sum(dlnr * dlnr, axis=-1)
        R = np.where(Syy > 0, (Sxy * Sxy) / (Sxx * Syy + 1e-30), 1.0)
    R[invalid] = -np.inf
    return R


# ── Spiral fidelity (pole-optimised R^2) ──────────────────────────────────

def spiral_fidelity(k1: float, k2: float,
                    Lp: float = LP_DEFAULT, Lm: float = LM_DEFAULT, Ld: float = LD_DEFAULT,
                    theta_max: float = THETA_MAX_DEFAULT, N: int = 200,
                    pole_search_radius: float = 2.0,
                    pole_search_resolution: int = 81,
                    return_pole: bool = False,
                    return_full: bool = False):
    """
    Compute F(k1, k2, theta_max) = sup over P of R^2(ln r vs psi about pole P).

    Two-stage pole optimisation:

      1. Coarse grid search over (px, py) in [-R, R]^2 with R = pole_search_radius
         and pole_search_resolution^2 candidates (default: 81 x 81, step 0.05).
         This domain is several times the finger length and ensures the global
         optimum lies in its interior for all biologically relevant (k1, k2).

      2. Nelder-Mead refinement initialised at the best coarse-grid pole,
         with convergence tolerances xatol = 1e-10, fatol = 1e-12.

    Parameters
    ----------
    return_pole : if True, also return the optimised pole position (px, py)
    return_full : if True, also return (mu, a0) and the trajectory arrays

    Returns
    -------
    F : float (or tuple if return_pole / return_full)
    """
    x, y = trajectory(k1, k2, Lp, Lm, Ld, theta_max, N)

    gx = np.linspace(-pole_search_radius, pole_search_radius, pole_search_resolution)
    gy = np.linspace(-pole_search_radius, pole_search_radius, pole_search_resolution)
    R2_grid = _R2_grid_at(gx, gy, x, y)
    idx = np.unravel_index(np.argmax(R2_grid), R2_grid.shape)
    best_R2 = R2_grid[idx]
    best_P = (gx[idx[1]], gy[idx[0]])

    try:
        res = minimize(
            lambda P: -fit_at_pole(P[0], P[1], x, y)[0],
            x0=np.array(best_P),
            method='Nelder-Mead',
            options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 5000},
        )
        F = -res.fun
        P_opt = (float(res.x[0]), float(res.x[1]))
        if F < best_R2:
            F, P_opt = best_R2, best_P
    except Exception:
        F, P_opt = best_R2, best_P

    if return_full:
        R2, mu, a0 = fit_at_pole(P_opt[0], P_opt[1], x, y)
        return F, P_opt, mu, a0, x, y
    if return_pole:
        return F, P_opt
    return F


# ── Sanity check ──────────────────────────────────────────────────────────

def sanity_check(verbose: bool = True) -> bool:
    """
    Verify the fitter reproduces the anatomical-point checkpoint.

    Returns True on success.
    """
    F, P, mu, a0, _, _ = spiral_fidelity(K1_ANAT, K2_ANAT, return_full=True)
    alpha = np.degrees(np.arctan(1.0 / abs(mu)))

    ok_F  = abs(F - 0.9997) < 5e-4
    ok_Px = abs(P[0] - 0.0505) < 5e-3
    ok_Py = abs(P[1] - 0.1776) < 5e-3
    ok_a  = abs(alpha - 75.6) < 0.5
    ok = all([ok_F, ok_Px, ok_Py, ok_a])

    if verbose:
        print("Sanity check at anatomical (k1 = 1.0, k2 = 0.67):")
        print(f"  F             = {F:.6f}   (expected 0.9997)   {'OK' if ok_F else 'FAIL'}")
        print(f"  pole P        = ({P[0]:.4f}, {P[1]:.4f})   (expected (0.0505, 0.1776))   "
              f"{'OK' if ok_Px and ok_Py else 'FAIL'}")
        print(f"  alpha         = {alpha:.2f} deg   (expected 75.6 deg)   {'OK' if ok_a else 'FAIL'}")
        print(f"  Overall: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    sanity_check()
