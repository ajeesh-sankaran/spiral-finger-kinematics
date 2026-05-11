"""
Generate Figure 1: representative equiangular spiral arc fits for the coupled
three-link finger at three coupling cases.

  A — Anatomical:               k1 = 1.00, k2 = 0.67   (inside viable region)
  B — Well inside viable:       k1 = 0.55, k2 = 0.35
  C — Outside viable region:    k1 = 1.50, k2 = 1.20

Run:  python scripts/make_figure_1.py
Output: figures/figure1_spiral_arcs.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

from kinematics import spiral_fidelity, LP_DEFAULT, LM_DEFAULT, LD_DEFAULT  # noqa: E402

OUT_PATH = os.path.join(HERE, '..', 'figures', 'figure1_spiral_arcs.png')


def spiral_curve(mu, a0, psi_range, n=200):
    """Sample the spiral r = exp(mu*psi + a0) over the given psi range."""
    psi = np.linspace(*psi_range, n)
    r = np.exp(mu * psi + a0)
    return r * np.cos(psi), r * np.sin(psi)


def main():
    cases = [
        ("Anatomical case",           1.00, 0.67),
        ("Well inside viable region", 0.55, 0.35),
        ("Outside viable region",     1.50, 1.20),
    ]

    fits = []
    for label, k1, k2 in cases:
        F, P, mu, a0, x, y = spiral_fidelity(k1, k2, return_full=True)
        alpha = float(np.degrees(np.arctan(1.0 / abs(mu))))
        fits.append((label, k1, k2, F, alpha, P, mu, a0, x, y))
        print(f"{label:30s} k1={k1}, k2={k2}: R^2={F:.4f}, alpha={alpha:.1f} deg, "
              f"P=({P[0]:.3f}, {P[1]:.3f})")

    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[5, 0.6], hspace=0.05, wspace=0.30,
                          left=0.05, right=0.97, top=0.83, bottom=0.04)

    snap_angles = np.deg2rad([0, 30, 60, 90])
    opacities = [0.25, 0.45, 0.70, 1.0]

    for idx, (label, k1, k2, F, alpha, P, mu, a0, x, y) in enumerate(fits):
        ax = fig.add_subplot(gs[0, idx])

        # Finger snapshots at four flexion angles
        for s_idx, theta_p in enumerate(snap_angles):
            op = opacities[s_idx]
            phi1 = theta_p
            phi2 = (1.0 + k1) * theta_p
            phi3 = (1.0 + k1 + k1 * k2) * theta_p
            j0 = (0, 0)
            j1 = (LP_DEFAULT * np.cos(phi1), LP_DEFAULT * np.sin(phi1))
            j2 = (j1[0] + LM_DEFAULT * np.cos(phi2), j1[1] + LM_DEFAULT * np.sin(phi2))
            j3 = (j2[0] + LD_DEFAULT * np.cos(phi3), j2[1] + LD_DEFAULT * np.sin(phi3))
            ax.plot([j0[0], j1[0]], [j0[1], j1[1]], color='steelblue', lw=8, alpha=op,
                    solid_capstyle='round')
            ax.plot([j1[0], j2[0]], [j1[1], j2[1]], color='seagreen', lw=8, alpha=op,
                    solid_capstyle='round')
            ax.plot([j2[0], j3[0]], [j2[1], j3[1]], color='darkorange', lw=8, alpha=op,
                    solid_capstyle='round')
            if op >= 1.0:
                ax.plot([j1[0], j2[0]], [j1[1], j2[1]], 'o',
                        mfc='white', mec='black', ms=5, zorder=5)

        # Fingertip locus
        locus_colour = 'seagreen' if F >= 0.999 else 'firebrick'
        ax.plot(x, y, color=locus_colour, lw=2.2, label='Fingertip locus', zorder=4)

        # Best-fit spiral
        u = x - P[0]
        v = y - P[1]
        psi = np.unwrap(np.arctan2(v, u))
        sx, sy = spiral_curve(mu, a0, (psi.min(), psi.max()))
        ax.plot(sx + P[0], sy + P[1], '--', color='darkorange', lw=1.8,
                label='Best-fit spiral arc', zorder=4)

        # Pole and MCP markers
        ax.plot(P[0], P[1], 'x', color='red', ms=14, mew=3,
                label='Spiral pole $P$', zorder=6)
        ax.annotate('P', xy=(P[0] + 0.02, P[1] - 0.04),
                    color='red', fontsize=11, fontweight='bold')
        ax.plot(0, 0, 'o', mfc='black', mec='black', ms=7, zorder=6)
        ax.annotate('MCP', xy=(0.02, -0.07), fontsize=10, color='black')

        ax.text(0.04, 0.93, f"({chr(65 + idx)})", transform=ax.transAxes,
                fontsize=14, fontweight='bold')
        ax.set_xlabel("$x$ (normalised)")
        if idx == 0:
            ax.set_ylabel("$y$ (normalised)")
            ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95)
        ax.set_title(f"{label}\n$k_1 = {k1}$, $k_2 = {k2}$", fontsize=11)

        xmin = min(x.min(), P[0], 0) - 0.1
        xmax = max(x.max(), P[0], 1.0) + 0.1
        ymin = min(y.min(), P[1], 0) - 0.1
        ymax = max(y.max(), P[1]) + 0.1
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25)

        # R^2 / alpha and viability flag
        ax_b = fig.add_subplot(gs[1, idx])
        ax_b.axis('off')
        flag = ('Viable ($R^2 \\geq 0.999$)' if F >= 0.999
                else 'Non-viable ($R^2 < 0.999$)')
        flag_colour = 'seagreen' if F >= 0.999 else 'firebrick'
        ax_b.text(0.5, 0.8, f"$R^2 = {F:.4f}$    $\\alpha = {alpha:.1f}\\degree$",
                  transform=ax_b.transAxes, ha='center', fontsize=11)
        ax_b.text(0.5, 0.15, flag, transform=ax_b.transAxes, ha='center',
                  fontsize=11, color=flag_colour,
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           edgecolor=flag_colour, lw=1.5))

    fig.suptitle(
        "Figure 1. Representative equiangular spiral arc fits for the coupled "
        "three-link finger model\n"
        "$L_p : L_m : L_d = 2 : 1 : 1$,  $\\theta_p \\in [0\\degree, 90\\degree]$.  "
        "Finger shown at $0\\degree, 30\\degree, 60\\degree, 90\\degree$ "
        "(increasing opacity).  "
        "Dashed = best-fit spiral $r = a e^{\\mu \\psi}$ about LM-optimised pole $P$.",
        fontsize=11, y=0.97,
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=180, bbox_inches='tight', facecolor='white')
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
