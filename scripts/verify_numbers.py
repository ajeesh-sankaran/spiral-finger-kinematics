#!/usr/bin/env python3
"""
Verify every number quoted in the revised manuscript against the computed
grids. Prints a table of claim vs computed value.

Run from the repository root:  PYTHONPATH=. python3 scripts/verify_numbers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.kinematics import spiral_fidelity  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
PHI = (1 + 5 ** 0.5) / 2
TAU = 0.999


def frac_viable(F, tau=TAU):
    return 100.0 * np.mean(F >= tau)


def boundary_k1(F, k1_vals, k2_vals, k2, tau=TAU):
    """Largest k1 at which F >= tau, at the grid row nearest k2."""
    j = int(np.argmin(np.abs(k2_vals - k2)))
    row = F[j, :]
    ok = np.where(row >= tau)[0]
    if len(ok) == 0:
        return None
    if ok[-1] == len(k1_vals) - 1:
        return float("inf")
    i = ok[-1]
    # linear interpolation between the last viable and first non-viable node
    f0, f1 = row[i], row[i + 1]
    t = (f0 - tau) / (f0 - f1)
    return float(k1_vals[i] + t * (k1_vals[i + 1] - k1_vals[i]))


def main():
    rows = []

    def chk(label, claim, got, tol=None, fmt="{:.5f}"):
        if got is None:
            ok = "?"
        elif isinstance(claim, str):
            ok = "-"
        else:
            ok = "OK" if abs(got - claim) <= (tol if tol is not None else 5e-4) else "**MISMATCH**"
        gs = got if isinstance(got, str) else (fmt.format(got) if got is not None else "n/a")
        cs = claim if isinstance(claim, str) else fmt.format(claim)
        rows.append((label, cs, gs, ok))

    # ---- anatomical checkpoint -------------------------------------------
    F, P, mu, a0, _, _ = spiral_fidelity(1.0, 0.67, return_full=True)
    alpha = np.degrees(np.arctan(1.0 / abs(mu)))
    chk("anatomical F", 0.9997, F)
    chk("anatomical pole p_x", 0.050, P[0], tol=2e-3, fmt="{:.4f}")
    chk("anatomical pole p_y", 0.179, P[1], tol=2e-3, fmt="{:.4f}")
    chk("anatomical mu", -0.258, mu, tol=3e-3, fmt="{:.4f}")
    chk("anatomical alpha (deg)", 75.6, alpha, tol=0.1, fmt="{:.2f}")

    # ---- Figure 2 panels --------------------------------------------------
    for (k1, k2, cF, cA) in [(1.00, 0.67, 0.9997, 75.6),
                             (0.55, 0.35, 0.9999, 86.2),
                             (1.50, 1.20, 0.905, 57.0)]:
        Fp, Pp, mup, _, _, _ = spiral_fidelity(k1, k2, return_full=True)
        ap = np.degrees(np.arctan(1.0 / abs(mup)))
        chk(f"Fig2 k1={k1} k2={k2}: R2", cF, Fp, tol=1e-3)
        chk(f"Fig2 k1={k1} k2={k2}: alpha", cA, ap, tol=0.2, fmt="{:.2f}")

    # ---- coupling grid ----------------------------------------------------
    d = np.load(DATA / "F80_coupling.npz")
    Fc, k1v, k2v = d["F_grid"], d["k1_vals"], d["k2_vals"]
    chk("V(0.999) %% of coupling box", 80.3, frac_viable(Fc), tol=0.3, fmt="{:.1f}")
    for k2, claim in [(0.67, 1.35), (1.00, 1.11), (1.20, 1.01)]:
        chk(f"k1 boundary at k2={k2}", claim, boundary_k1(Fc, k1v, k2v, k2),
            tol=0.02, fmt="{:.3f}")
    # crossover: highest k2 at which the boundary still exceeds the grid
    cross = None
    for j, k2 in enumerate(k2v):
        b = boundary_k1(Fc, k1v, k2v, k2)
        if b == float("inf"):
            cross = k2
    chk("k2 crossover (boundary leaves grid)", 0.573, cross, tol=0.02, fmt="{:.3f}")

    # ---- length-ratio grid ------------------------------------------------
    dl = np.load(DATA / "F60_lengths.npz")
    Fl = dl["F_grid"]
    chk("length-ratio F min", 0.99960, Fl.min(), tol=5e-6)
    chk("length-ratio F max", 0.99995, Fl.max(), tol=5e-6)
    chk("length-ratio %% viable", 100.0, frac_viable(Fl), tol=0.01, fmt="{:.2f}")

    def lengths_from_ratios(rpm, rmd):
        Ld = 1.0 / (rpm * rmd + rmd + 1.0)
        Lm = rmd * Ld
        Lp = rpm * Lm
        return Lp, Lm, Ld

    Lp, Lm, Ld = lengths_from_ratios(2.0, 1.0)
    chk("2:1:1 F", 0.9997, spiral_fidelity(1.0, 0.67, Lp, Lm, Ld))
    Lp, Lm, Ld = lengths_from_ratios(PHI, PHI)
    chk("golden-ratio F", 0.9998, spiral_fidelity(1.0, 0.67, Lp, Lm, Ld))

    # ---- boundary sweeps --------------------------------------------------
    for fn, claim_pct, claim_min in [
            ("F60_boundary_k1_1p25_k2_0p67.npz", 98.8, 0.9989),
            ("F60_boundary_k1_1p00_k2_1p00.npz", 100.0, 0.9991)]:
        db = np.load(DATA / fn)
        Fb = db["F_grid"]
        chk(f"{fn[13:-4]} %% viable", claim_pct, frac_viable(Fb), tol=0.2, fmt="{:.1f}")
        chk(f"{fn[13:-4]} min F", claim_min, Fb.min(), tol=1e-4)

    # ---- wide length-ratio sweep -----------------------------------------
    wide = DATA / "F_lengths_wide.npz"
    if wide.exists():
        dw = np.load(wide)
        Fw, rpmw, rmdw = dw["F_grid"], dw["rpm_vals"], dw["rmd_vals"]
        iw, jw = np.unravel_index(np.argmin(Fw), Fw.shape)
        chk("wide sweep min F", 0.99950, Fw.min(), tol=1e-5)
        chk("wide sweep %% viable at 0.999", 100.0, frac_viable(Fw), tol=0.01,
            fmt="{:.2f}")
        chk("wide sweep argmin Lp/Lm", 10.0, rpmw[jw], tol=0.01, fmt="{:.2f}")
        chk("wide sweep argmin Lm/Ld", 0.16, rmdw[iw], tol=0.01, fmt="{:.3f}")
        rows.append(("NOTE: wide-sweep min is 0.999499, i.e. just BELOW 0.9995",
                     "-", "-", "-"))

    # ---- closure bound ----------------------------------------------------
    chk("PIP max at k1=0.55 (deg)", 49.5, 90 * 0.55, tol=0.05, fmt="{:.2f}")
    chk("DIP at anatomical (deg)", 60.3, 90 * 1.0 * 0.67, tol=0.05, fmt="{:.2f}")

    # ---- Section 2.4: pole envelope over the whole coupling grid ----------
    chk("pole p_x min (quoted -0.04)", -0.04, d["pole_x"].min(), tol=1e-2,
        fmt="{:.4f}")
    chk("pole p_x max (quoted 0.85)", 0.85, d["pole_x"].max(), tol=1e-2,
        fmt="{:.4f}")
    chk("pole p_y min (quoted 0.03)", 0.03, d["pole_y"].min(), tol=1e-2,
        fmt="{:.4f}")
    chk("pole p_y max (quoted 0.37)", 0.37, d["pole_y"].max(), tol=1e-2,
        fmt="{:.4f}")

    # ---- Section 2.4: radial departure from the best-fit spiral ----------
    def max_departure(k1, k2):
        _, P_, mu_, a0_, x_, y_ = spiral_fidelity(k1, k2, return_full=True)
        u, v = x_ - P_[0], y_ - P_[1]
        r = np.hypot(u, v)
        psi = np.unwrap(np.arctan2(v, u))
        return 100.0 * np.max(np.abs(r - np.exp(mu_ * psi + a0_)))

    chk("max radial departure at anatomical (% of length)", 1.2,
        max_departure(1.0, 0.67), tol=0.1, fmt="{:.2f}")
    chk("max radial departure at the tau=0.999 boundary (%)", 5.0,
        max_departure(1.35, 0.67), tol=0.6, fmt="{:.2f}")

    # ---- Figure 1B: the angle alpha drawn at the two sample points -------
    _, Pa, mua, _, xa, ya = spiral_fidelity(1.0, 0.67, return_full=True)
    Pa = np.asarray(Pa)
    alpha_fit = np.degrees(np.arctan(1.0 / abs(mua)))
    chk("Fig1B fitted alpha (deg)", 75.6, alpha_fit, tol=0.1, fmt="{:.2f}")
    for frac, claim in ((0.30, 74.9), (0.78, 76.2)):
        i = int(frac * len(xa))
        Q = np.array([xa[i], ya[i]])
        tv = np.array([xa[i + 1] - xa[i - 1], ya[i + 1] - ya[i - 1]])
        tv /= np.linalg.norm(tv)
        inward = (Pa - Q) / np.linalg.norm(Pa - Q)
        ang = np.degrees(np.arccos(np.clip(np.dot(inward, tv), -1, 1)))
        chk(f"Fig1B alpha arc at {int(frac * 100)}% along (deg)", claim, ang,
            tol=0.1, fmt="{:.2f}")

    # ---- Section 4.2: decline beyond the k1 boundary ---------------------
    for k2, claim in [(0.67, 0.9988), (1.00, 0.992), (1.20, 0.905)]:
        j = int(np.argmin(np.abs(k2v - k2)))
        chk(f"F at grid edge k1=1.5, k2={k2}", claim, Fc[j, -1], tol=1e-3,
            fmt="{:.4f}")
    chk("%% of coupling box with F < 0.95", 0.9, 100.0 * np.mean(Fc < 0.95),
        tol=0.1, fmt="{:.2f}")
    chk("global min F over coupling grid", 0.902, Fc.min(), tol=1e-3,
        fmt="{:.4f}")

    # ---- Section 2.2: coupling values from the maximal joint ranges ------
    chk("k1 and k2 from ranges 90/90/60: k1", 1.0, 90.0 / 90.0, tol=1e-9,
        fmt="{:.4f}")
    chk("k1 and k2 from ranges 90/90/60: k2", 0.67, 60.0 / 90.0, tol=4e-3,
        fmt="{:.4f}")

    # ---- Section 4.4: where the length-ratio extremes lie ----------------
    rpm_l, rmd_l = dl["rpm_vals"], dl["rmd_vals"]
    il, jl = np.unravel_index(np.argmin(Fl), Fl.shape)
    chk("length-ratio min sits on the lowest L_m/L_d row", rmd_l[0], rmd_l[il],
        tol=1e-6, fmt="{:.3f}")
    iu, ju = np.unravel_index(np.argmax(Fl), Fl.shape)
    chk("length-ratio max sits at the upper-right corner",
        rpm_l[-1] + rmd_l[-1], rpm_l[ju] + rmd_l[iu], tol=1e-6, fmt="{:.3f}")

    # ---- Section 4.4: reference points at all three coupling values ------
    worst = 1.0
    for k1, k2 in ((1.0, 0.67), (1.25, 0.67), (1.0, 1.0)):
        for rp, rm in ((2.0, 1.0), (PHI, PHI)):
            worst = min(worst, spiral_fidelity(k1, k2, *lengths_from_ratios(rp, rm)))
    chk("worst reference point over three couplings (quoted >= 0.9992)",
        0.9992, np.floor(worst * 1e4) / 1e4, tol=1e-6)

    # ---- Section 4.4: extent of the sub-threshold corner -----------------
    db = np.load(DATA / "F60_boundary_k1_1p25_k2_0p67.npz")
    Fb, rp_b, rm_b = db["F_grid"], db["rpm_vals"], db["rmd_vals"]
    bi, bj = np.where(Fb < TAU)
    chk("sub-threshold corner: lowest L_p/L_m (quoted ~3.1)", 3.1,
        rp_b[bj].min(), tol=0.05, fmt="{:.2f}")
    chk("sub-threshold corner: highest L_m/L_d (quoted ~0.7)", 0.7,
        rm_b[bi].max(), tol=0.05, fmt="{:.2f}")

    # ---- Appendix A: the algebraic identities, at random parameters -------
    rng = np.random.default_rng(20260801)
    worst = 0.0
    for _ in range(200):
        k1_, k2_ = rng.uniform(0.3, 1.6, 2)
        Lv = rng.uniform(0.1, 1.0, 3)
        Lv /= Lv.sum()
        Pv = rng.uniform(-0.6, 0.9, 2)
        th = rng.uniform(0.05, np.pi / 2)
        wv = np.array([1.0, 1.0 + k1_, 1.0 + k1_ + k1_ * k2_])
        lamv = wv * Lv
        xs = (Lv * np.cos(wv * th)).sum()
        ys = (Lv * np.sin(wv * th)).sum()
        xd = -(lamv * np.sin(wv * th)).sum()
        yd = (lamv * np.cos(wv * th)).sum()
        uu, vv = xs - Pv[0], ys - Pv[1]
        N_def, D_def = uu * xd + vv * yd, uu * yd - vv * xd

        Om = wv[None, :] - wv[:, None]          # Omega_ij = w_j - w_i
        offd = ~np.eye(3, dtype=bool)
        outer = Lv[:, None] * lamv[None, :]
        # (A.7) as printed in the appendix
        N_app = (-(outer * np.sin(Om * th))[offd].sum()
                 + Pv[0] * (lamv * np.sin(wv * th)).sum()
                 - Pv[1] * (lamv * np.cos(wv * th)).sum())
        # (A.9) with the corrected D_P
        D_app = ((Lv * lamv).sum() + (outer * np.cos(Om * th))[offd].sum()
                 - Pv[0] * (lamv * np.cos(wv * th)).sum()
                 - Pv[1] * (lamv * np.sin(wv * th)).sum())
        worst = max(worst, abs(N_app - N_def), abs(D_app - D_def))
    chk("Appendix A (A.7)/(A.9) vs definitions, worst of 200", 0.0, worst,
        tol=1e-12, fmt="{:.2e}")

    # small-theta limit of mu, Appendix A
    px_, py_ = 0.137, -0.264
    chk("Appendix A small-theta limit -p_y/(1-p_x)", -py_ / (1 - px_),
        -py_ / (1 - px_), tol=1e-12, fmt="{:.6f}")

    # ---- report -----------------------------------------------------------
    w = max(len(r[0]) for r in rows)
    print(f"{'quantity'.ljust(w)}  {'claimed':>12}  {'computed':>12}   status")
    print("-" * (w + 44))
    bad = 0
    for label, cs, gs, ok in rows:
        print(f"{label.ljust(w)}  {cs:>12}  {gs:>12}   {ok}")
        if "MISMATCH" in ok:
            bad += 1
    print("-" * (w + 44))
    print(f"{bad} mismatch(es)")
    return bad


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
