"""
Generate Figure 1: schematic of the kinematic model and the spiral geometry,
defining the variables introduced in Sections 2.1--2.3.

  (A) Forward kinematics: link lengths, joint angles, cumulative angles.
  (B) Spiral geometry: pole, polar coordinates, tangent angle.

Added at the reviewer's suggestion that a figure showing the variables
discussed in Section 2.1 would be useful.

Run:  python scripts/make_figure_1_schematic.py
Output: figures/figure1_model_schematic.{pdf,png}
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

from kinematics import spiral_fidelity, LP_DEFAULT, LM_DEFAULT, LD_DEFAULT  # noqa: E402

OUT_STEM = os.path.join(HERE, '..', 'figures', 'figure1_model_schematic')

C_PROX = '#0072B2'
C_MID = '#009E73'
C_DIST = '#E69F00'
C_SPIRAL = '#CC79A7'
C_GUIDE = '#999999'

K1, K2 = 1.0, 0.67


def _bisector_label(ax, centre, r, a1_deg, a2_deg, text, colour='black',
                    fontsize=11, frac=0.72):
    """Place a label on the bisector of an arc, at `frac` of its radius."""
    mid = np.deg2rad(0.5 * (a1_deg + a2_deg))
    pos = np.asarray(centre) + frac * r * np.array([np.cos(mid), np.sin(mid)])
    ax.text(*pos, text, fontsize=fontsize, color=colour, ha='center',
            va='center')


def _outward_label(ax, a, b, text, colour, offset=0.095):
    """Label a segment on its convex side, i.e. the side away from the MCP."""
    a, b = np.asarray(a), np.asarray(b)
    mid = 0.5 * (a + b)
    d = b - a
    n = np.array([d[1], -d[0]])
    n = n / np.linalg.norm(n) * offset
    if np.dot(n, mid) < 0:          # MCP is at the origin
        n = -n
    ax.text(*(mid + n), text, fontsize=12, color=colour, ha='center',
            va='center', fontweight='bold')


def joints(theta_p, k1=K1, k2=K2):
    phi1 = theta_p
    phi2 = (1.0 + k1) * theta_p
    phi3 = (1.0 + k1 + k1 * k2) * theta_p
    j0 = np.array([0.0, 0.0])
    j1 = j0 + LP_DEFAULT * np.array([np.cos(phi1), np.sin(phi1)])
    j2 = j1 + LM_DEFAULT * np.array([np.cos(phi2), np.sin(phi2)])
    j3 = j2 + LD_DEFAULT * np.array([np.cos(phi3), np.sin(phi3)])
    return j0, j1, j2, j3, (phi1, phi2, phi3)


def panel_a(ax):
    theta_p = np.deg2rad(30)
    j0, j1, j2, j3, (phi1, phi2, phi3) = joints(theta_p)

    # reference axes
    ax.annotate('', xy=(1.12, 0), xytext=(-0.12, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.0))
    ax.annotate('', xy=(0, 1.05), xytext=(0, -0.12),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.0))
    ax.text(1.13, -0.055, '$x$', fontsize=11)
    ax.text(-0.065, 1.06, '$y$', fontsize=11)

    # extension of each link direction, for the joint-angle arcs
    for base, prev_phi, this_phi, L in ((j1, phi1, phi2, 0.30),
                                        (j2, phi2, phi3, 0.24)):
        tip = base + L * np.array([np.cos(prev_phi), np.sin(prev_phi)])
        ax.plot([base[0], tip[0]], [base[1], tip[1]], ':', color=C_GUIDE, lw=1.2)

    # links
    for a, b, colour, lw in ((j0, j1, C_PROX, 9), (j1, j2, C_MID, 9),
                             (j2, j3, C_DIST, 9)):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=colour, lw=lw,
                solid_capstyle='round', zorder=3)

    # joint markers
    for p in (j0, j1, j2):
        ax.plot(*p, 'o', mfc='white', mec='black', ms=7, mew=1.2, zorder=5)
    ax.plot(*j3, 'o', mfc='black', mec='black', ms=6, zorder=5)

    # link-length labels, placed on the convex side of each segment
    for a, b, lab, colour in ((j0, j1, '$L_p$', C_PROX),
                              (j1, j2, '$L_m$', C_MID),
                              (j2, j3, '$L_d$', C_DIST)):
        _outward_label(ax, a, b, lab, colour,
                       offset=0.095 if lab != '$L_d$' else 0.115)

    # joint-angle arcs, labelled on their bisectors
    for centre, a1, a2, rad, frac, lab in (
            (j0, 0.0, np.degrees(phi1), 0.30, 1.30, r'$\theta_p$'),
            (j1, np.degrees(phi1), np.degrees(phi2), 0.17, 1.18, r'$\theta_m$'),
            (j2, np.degrees(phi2), np.degrees(phi3), 0.20, 1.22, r'$\theta_d$')):
        ax.add_patch(Arc(centre, 2 * rad, 2 * rad, theta1=a1, theta2=a2,
                         color='black', lw=1.2))
        _bisector_label(ax, centre, rad, a1, a2, lab, frac=frac)

    # cumulative angle phi_3 measured from the x direction
    ax.plot([j2[0], j2[0] + 0.42], [j2[1], j2[1]], ':', color=C_GUIDE, lw=1.2)
    ax.add_patch(Arc(j2, 0.66, 0.66, theta1=0, theta2=np.degrees(phi3),
                     color=C_GUIDE, lw=1.2, linestyle='--'))
    _bisector_label(ax, j2, 0.33, 0.0, np.degrees(phi3), r'$\phi_3$',
                    colour='#666666', frac=1.18)

    # joint names, offset below/right of each joint marker
    ax.text(j0[0] - 0.01, j0[1] - 0.11, 'MCP', fontsize=9.5, ha='center')
    ax.text(j1[0] + 0.02, j1[1] - 0.11, 'PIP', fontsize=9.5, ha='center')
    ax.text(j2[0] + 0.15, j2[1] - 0.05, 'DIP', fontsize=9.5, ha='center')
    tip_dir = (j3 - j2) / np.linalg.norm(j3 - j2)
    lab = j3 + 0.07 * tip_dir
    ax.text(lab[0], lab[1], 'fingertip', fontsize=9.5, ha='center', va='bottom')

    ax.text(0.62, 0.97,
            'Coupling:  $\\theta_m = k_1\\theta_p$,  $\\theta_d = k_2\\theta_m$\n'
            '$\\phi_1 = \\theta_p$,  $\\phi_2 = \\theta_p+\\theta_m$,  '
            '$\\phi_3 = \\theta_p+\\theta_m+\\theta_d$',
            transform=ax.transAxes, ha='center', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                      edgecolor='#cccccc'))

    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(-0.32, 1.16)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(A)  Forward kinematics and the coupling constraint',
                 fontsize=11)


def _tangent_angle(P, Q, t):
    """
    Arc limits (degrees) for the equiangular angle alpha at the point Q.

    alpha is defined in the paper as arctan(1/|mu|), the acute angle between
    the tangent and the radial line through P. Since mu < 0 here the
    trajectory spirals inward, so the angle to the *outward* radius is obtuse
    and it is the angle to the inward radial direction, Q -> P, that equals
    alpha. Drawing the arc against the inward direction also puts it in the
    wedge between the two lines already on the plot.
    """
    inward = (P - Q) / np.linalg.norm(P - Q)
    a_rad = np.degrees(np.arctan2(inward[1], inward[0]))
    a_tan = np.degrees(np.arctan2(t[1], t[0]))
    delta = ((a_tan - a_rad + 180.0) % 360.0) - 180.0   # signed, in (-180, 180]
    lo, hi = (a_rad, a_rad + delta) if delta > 0 else (a_rad + delta, a_rad)
    return lo, hi, abs(delta)


def panel_b(ax):
    F, P, mu, a0, x, y = spiral_fidelity(K1, K2, return_full=True)
    P = np.asarray(P)
    alpha = np.degrees(np.arctan(1.0 / abs(mu)))

    ax.plot(x, y, color='black', lw=2.0, zorder=4)

    # pole
    ax.plot(*P, 'x', color='black', ms=11, mew=2.2, zorder=6)
    ax.text(P[0] - 0.02, P[1] - 0.10, '$P=(p_x,p_y)$', fontsize=10.5,
            ha='center')

    # reference ray from the pole, for the polar angle psi
    ax.plot([P[0], P[0] + 0.62], [P[1], P[1]], ':', color=C_GUIDE, lw=1.2)

    # alpha is shown at two points, to make its constancy the visible point
    for k, (frac, show_polar) in enumerate(((0.30, True), (0.78, False))):
        i = int(frac * len(x))
        Q = np.array([x[i], y[i]])
        t = np.array([x[i + 1] - x[i - 1], y[i + 1] - y[i - 1]])
        t = t / np.linalg.norm(t)

        # radial line P -> Q
        ax.plot([P[0], Q[0]], [P[1], Q[1]], '-', color=C_SPIRAL,
                lw=1.8 if show_polar else 1.2,
                alpha=1.0 if show_polar else 0.65, zorder=3)
        ax.plot(*Q, 'o', mfc=C_SPIRAL, mec='black', ms=7, mew=1.0, zorder=7)

        # tangent, drawn as an arrow in the direction of increasing theta_p
        ax.add_patch(FancyArrowPatch(Q - 0.17 * t, Q + 0.26 * t,
                                     arrowstyle='-|>', mutation_scale=13,
                                     color=C_DIST, lw=1.8, zorder=5))

        # the equiangular angle itself
        lo, hi, span = _tangent_angle(P, Q, t)
        ax.add_patch(Arc(Q, 0.30, 0.30, theta1=lo, theta2=hi,
                         color=C_DIST, lw=1.5, zorder=6))
        _bisector_label(ax, Q, 0.15, lo, hi, r'$\alpha$', colour=C_DIST,
                        fontsize=13, frac=1.55)
        print(f"  panel B, point {k + 1}: arc spans {span:.2f} deg "
              f"(alpha = {alpha:.2f} deg)")

        if show_polar:
            mid = (P + Q) / 2
            ax.text(mid[0] - 0.075, mid[1] + 0.015, '$r$', fontsize=13,
                    color=C_SPIRAL)
            psi = np.degrees(np.arctan2(Q[1] - P[1], Q[0] - P[0]))
            ax.add_patch(Arc(P, 0.60, 0.60, theta1=0, theta2=psi,
                             color=C_GUIDE, lw=1.2))
            _bisector_label(ax, P, 0.30, 0.0, psi, r'$\psi$',
                            colour='#666666', fontsize=13, frac=1.22)

    # direction-of-travel note
    ax.annotate('direction of\nincreasing $\\theta_p$',
                xy=(0.985, 0.60), xycoords='axes fraction', ha='right',
                va='center', fontsize=8.5, color=C_DIST)

    ax.text(0.5, 0.02,
            'Equiangular spiral:  $r = a\\,e^{\\mu\\psi}$,  '
            '$\\mu = \\cot\\alpha$\n'
            '$\\alpha$ constant along the curve  $\\Leftrightarrow$  '
            'exact spiral',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                      edgecolor='#cccccc'))

    ax.set_xlim(-0.55, 1.15)
    ax.set_ylim(-0.28, 1.02)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(B)  Spiral geometry about the pole $P$', fontsize=11)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    panel_a(axes[0])
    panel_b(axes[1])
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
    for ext, kw in (('pdf', {}), ('png', {'dpi': 300})):
        fig.savefig(f"{OUT_STEM}.{ext}", bbox_inches='tight', facecolor='white', **kw)
        print(f"Saved {OUT_STEM}.{ext}")


if __name__ == "__main__":
    main()
