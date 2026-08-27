"""
Generate Figure 4: spiral fidelity over phalangeal length-ratio space at the
anatomical coupling values (k1 = 1, k2 = 0.67), as a two-panel figure.

  (A) On the banded colour scale of Figure 3, so the reader can see that the
      entire length-ratio space sits in the two highest bands.
  (B) The same data resolved on the range it actually occupies, with contours,
      so the structure is visible.

The two panels together answer the reviewer's request for visible colour
variation while preventing the misreading that the gradient in (B) indicates
proximity to a viability boundary -- it does not; the whole plotted range is
viable.

Run:  python scripts/make_figure_4.py
Output: figures/figure4_length_ratios.{pdf,png}
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from scipy.ndimage import gaussian_filter, zoom

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data', 'F60_lengths.npz')
OUT_STEM = os.path.join(HERE, '..', 'figures', 'figure4_length_ratios')

TAUS = [0.99, 0.995, 0.999, 0.9995, 0.9999]
PANEL_B_RANGE = (0.9996, 0.99995)
PANEL_B_CONTOURS = [0.9997, 0.9998, 0.9999]
PHI = (1 + 5 ** 0.5) / 2
SIGMA = 0.5
ZOOM = 6


def main():
    d = np.load(DATA)
    F, rpm, rmd = d['F_grid'], d['rpm_vals'], d['rmd_vals']
    print(f"grid {F.shape}, F in [{F.min():.6f}, {F.max():.6f}], "
          f"{100 * np.mean(F >= 0.999):.2f}% viable at tau=0.999")

    Fs = zoom(gaussian_filter(F, sigma=SIGMA, mode='nearest'), ZOOM, order=3)
    xs = np.linspace(rpm[0], rpm[-1], Fs.shape[1])
    ys = np.linspace(rmd[0], rmd[-1], Fs.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

    # ── Panel A: banded scale of Figure 2 ────────────────────────────────
    axA = axes[0]
    boundaries = [0.90] + TAUS + [1.0]
    cmapA = plt.get_cmap('viridis', len(boundaries) - 1)
    normA = BoundaryNorm(boundaries, cmapA.N)
    mA = axA.pcolormesh(xs, ys, Fs, cmap=cmapA, norm=normA, shading='gouraud', rasterized=True)
    cbA = fig.colorbar(mA, ax=axA, boundaries=boundaries, ticks=boundaries,
                       spacing='uniform', pad=0.025)
    cbA.ax.set_yticklabels([f"{b:g}" for b in boundaries])
    cbA.set_label('Spiral fidelity $F$  (colour scale of Figure 3)', fontsize=10)

    # ── Panel B: resolved on the occupied range ──────────────────────────
    axB = axes[1]
    mB = axB.pcolormesh(xs, ys, Fs, cmap='viridis', shading='gouraud',
                        vmin=PANEL_B_RANGE[0], vmax=PANEL_B_RANGE[1],
                        rasterized=True)
    cs = axB.contour(xs, ys, Fs, levels=PANEL_B_CONTOURS, colors='white',
                     linewidths=1.3)
    axB.clabel(cs, fmt=lambda v: f"{v:.4f}", fontsize=8.5, inline=True,
               manual=[(3.05, 0.82), (3.00, 1.55), (2.55, 2.72)])
    cbB = fig.colorbar(mB, ax=axB, pad=0.025)
    cbB.set_label('Spiral fidelity $F$  (resolved range)', fontsize=10)
    cbB.formatter.set_useOffset(False)
    cbB.update_ticks()

    titles = [
        '(A)  On the colour scale of Figure 3',
        '(B)  Resolved on $[0.9996,\\,0.99995]$',
    ]
    for ax, title in zip(axes, titles):
        ax.plot(2.0, 1.0, marker='*', ms=19, mfc='white', mec='black', mew=1.3,
                linestyle='none', zorder=6,
                label='Anatomical 2:1:1 ($R^2 = 0.9997$)')
        ax.plot(PHI, PHI, marker='D', ms=9, mfc='white', mec='black', mew=1.3,
                linestyle='none', zorder=6,
                label='Golden ratio $\\Phi \\approx 1.618$ ($R^2 = 0.9998$)')
        ax.set_xlabel('$L_p / L_m$', fontsize=11)
        ax.set_ylabel('$L_m / L_d$', fontsize=11)
        ax.set_xlim(rpm[0], rpm[-1])
        ax.set_ylim(rmd[0], rmd[-1])
        ax.set_title(title, fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, -0.015))

    fig.tight_layout(rect=(0, 0.045, 1, 1))
    os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
    for ext, kw in (('pdf', {'dpi': 400}), ('png', {'dpi': 300})):
        fig.savefig(f"{OUT_STEM}.{ext}", bbox_inches='tight', facecolor='white', **kw)
        print(f"Saved {OUT_STEM}.{ext}")


if __name__ == "__main__":
    main()
