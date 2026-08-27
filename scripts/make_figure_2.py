"""
Generate Figure 2: representative equiangular spiral arc fits for the coupled
three-link finger at three coupling cases.

  A — Anatomical:               k1 = 1.00, k2 = 0.67   (inside viable region)
  B — Well inside viable:       k1 = 0.55, k2 = 0.35
  C — Outside viable region:    k1 = 1.50, k2 = 1.20

Colours follow the Okabe-Ito qualitative palette, which is distinguishable
under the common forms of colour vision deficiency.

Run:  python scripts/make_figure_2.py
Output: figures/figure2_spiral_arcs.{pdf,png}
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

from kinematics import spiral_fidelity, LP_DEFAULT, LM_DEFAULT, LD_DEFAULT  # noqa: E402

OUT_STEM = os.path.join(HERE, '..', 'figures', 'figure2_spiral_arcs')

# ── Okabe-Ito palette ─────────────────────────────────────────────────────
C_PROX = '#0072B2'   # blue
C_MID = '#009E73'    # bluish green
C_DIST = '#E69F00'   # orange
C_SPIRAL = '#CC79A7'  # reddish purple
C_VIABLE = '#0072B2'
C_NONVIABLE = '#D55E00'  # vermillion


def spiral_curve(mu, a0, psi_range, n=400):
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

    fig = plt.figure(figsize=(14.5, 5.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[6, 0.8], hspace=0.42, wspace=0.26,
                          left=0.05, right=0.98, top=0.92, bottom=0.04)

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
            j0 = (0.0, 0.0)
            j1 = (LP_DEFAULT * np.cos(phi1), LP_DEFAULT * np.sin(phi1))
            j2 = (j1[0] + LM_DEFAULT * np.cos(phi2), j1[1] + LM_DEFAULT * np.sin(phi2))
            j3 = (j2[0] + LD_DEFAULT * np.cos(phi3), j2[1] + LD_DEFAULT * np.sin(phi3))
            for (a, b, colour) in ((j0, j1, C_PROX), (j1, j2, C_MID), (j2, j3, C_DIST)):
                ax.plot([a[0], b[0]], [a[1], b[1]], color=colour, lw=7.5, alpha=op,
                        solid_capstyle='round', zorder=2)
            if op >= 1.0:
                ax.plot([j1[0], j2[0]], [j1[1], j2[1]], 'o',
                        mfc='white', mec='black', ms=4.5, mew=1.0, zorder=5)

        # Fingertip locus
        ax.plot(x, y, color='black', lw=2.0, label='Fingertip locus', zorder=4)

        # Best-fit spiral arc
        u = x - P[0]
        v = y - P[1]
        psi = np.unwrap(np.arctan2(v, u))
        sx, sy = spiral_curve(mu, a0, (psi.min(), psi.max()))
        ax.plot(sx + P[0], sy + P[1], '--', color=C_SPIRAL, lw=2.0,
                label='Best-fit spiral arc', zorder=4)

        # Pole and MCP markers
        ax.plot(P[0], P[1], 'x', color='black', ms=11, mew=2.2,
                label='Spiral pole $P$', zorder=6)
        ax.annotate('$P$', xy=(P[0] + 0.025, P[1] - 0.055), fontsize=11)
        ax.plot(0, 0, 'o', mfc='black', mec='black', ms=6, zorder=6)
        ax.annotate('MCP', xy=(0.025, -0.085), fontsize=9.5)

        ax.text(0.03, 0.94, f"({chr(65 + idx)})", transform=ax.transAxes,
                fontsize=14, fontweight='bold')
        ax.set_xlabel("$x$ (normalised)")
        if idx == 0:
            ax.set_ylabel("$y$ (normalised)")
            ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95)
        ax.set_title(f"{label}\n$k_1 = {k1:.2f}$, $k_2 = {k2:.2f}$", fontsize=11)

        ax.set_xlim(min(x.min(), P[0], 0) - 0.12, max(x.max(), P[0], 1.0) + 0.12)
        ax.set_ylim(min(y.min(), P[1], 0) - 0.14, max(y.max(), P[1]) + 0.12)
        ax.set_aspect('equal')
        ax.set_anchor('S')      # sit at the bottom of the cell, no dead space
        ax.grid(True, alpha=0.22)

        # Annotation strip: R^2, alpha, viability
        ax_b = fig.add_subplot(gs[1, idx])
        ax_b.axis('off')
        viable = F >= 0.999
        flag = ('Viable ($R^2 \\geq 0.999$)' if viable
                else 'Non-viable ($R^2 < 0.999$)')
        colour = C_VIABLE if viable else C_NONVIABLE
        ax_b.text(0.5, 0.95, f"$R^2 = {F:.4f}$    $\\alpha = {alpha:.1f}\\degree$",
                  transform=ax_b.transAxes, ha='center', va='top', fontsize=11)
        ax_b.text(0.5, 0.30, flag, transform=ax_b.transAxes, ha='center', va='top',
                  fontsize=10.5, color=colour,
                  bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                            edgecolor=colour, lw=1.4))

    os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
    for ext, kw in (('pdf', {}), ('png', {'dpi': 300})):
        plt.savefig(f"{OUT_STEM}.{ext}", bbox_inches='tight', facecolor='white', **kw)
        print(f"Saved {OUT_STEM}.{ext}")


if __name__ == "__main__":
    main()
