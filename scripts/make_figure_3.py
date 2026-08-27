"""
Generate Figure 3: spiral fidelity F(k1, k2, pi/2) and the nested viable
regions V(tau) in coupling parameter space, for 2:1:1 phalangeal lengths.

This figure consolidates the Figures 2 and 4 of the submitted
manuscript into a single panel, following the reviewer's suggestion. The
colour scale is viridis, banded at the threshold values
tau in {0.99, 0.995, 0.999, 0.9995, 0.9999}; the bands are drawn with uniform
width on the colourbar so that every contour is visible.

Contours and band edges are computed on a bicubic interpolant of the 80x80
grid with light Gaussian pre-smoothing (sigma = 0.5 grid cells) for display
only. Quoted boundary positions are computed from the unsmoothed grid by
scripts/verify_numbers.py.

Run:  python scripts/make_figure_3.py
Output: figures/figure3_coupling_space.{pdf,png}
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from scipy.ndimage import gaussian_filter, zoom

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data', 'F80_coupling.npz')
OUT_STEM = os.path.join(HERE, '..', 'figures', 'figure3_coupling_space')

TAUS = [0.99, 0.995, 0.999, 0.9995, 0.9999]
PRIMARY = 0.999
SIGMA = 0.5      # display-only Gaussian pre-smoothing, in grid cells
ZOOM = 6         # bicubic upsampling factor for display


def main():
    d = np.load(DATA)
    F, k1v, k2v = d['F_grid'], d['k1_vals'], d['k2_vals']
    print(f"grid {F.shape}, F in [{F.min():.6f}, {F.max():.6f}], "
          f"{100 * np.mean(F >= PRIMARY):.1f}% viable at tau={PRIMARY}")

    # Display-only smoothing and bicubic upsampling
    Fs = zoom(gaussian_filter(F, sigma=SIGMA, mode='nearest'), ZOOM, order=3)
    k1s = np.linspace(k1v[0], k1v[-1], Fs.shape[1])
    k2s = np.linspace(k2v[0], k2v[-1], Fs.shape[0])

    boundaries = [float(np.floor(F.min() * 100) / 100)] + TAUS + [1.0]
    cmap = plt.get_cmap('viridis', len(boundaries) - 1)
    norm = BoundaryNorm(boundaries, cmap.N)

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    mesh = ax.pcolormesh(k1s, k2s, Fs, cmap=cmap, norm=norm, shading='gouraud', rasterized=True)

    for tau in TAUS:
        thick = tau == PRIMARY
        cs = ax.contour(k1s, k2s, Fs, levels=[tau], colors='white',
                        linewidths=2.6 if thick else 1.1,
                        linestyles='solid' if thick else 'dashed')
        ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=8.5, inline=True,
                  inline_spacing=6)

    ax.plot(1.0, 0.67, marker='*', ms=20, mfc='white', mec='black', mew=1.4,
            linestyle='none', zorder=6,
            label='Anatomical point ($k_1=1$, $k_2=0.67$, $R^2=0.9997$)')

    ax.set_xlabel('$k_1$  (MCP--PIP coupling)', fontsize=11)
    ax.set_ylabel('$k_2$  (PIP--DIP coupling)', fontsize=11)
    ax.set_xlim(k1v[0], k1v[-1])
    ax.set_ylim(k2v[0], k2v[-1])
    ax.legend(loc='lower left', fontsize=9, framealpha=0.92)

    cbar = fig.colorbar(mesh, ax=ax, boundaries=boundaries, ticks=boundaries,
                        spacing='uniform', pad=0.025)
    cbar.ax.set_yticklabels([f"{b:g}" for b in boundaries])
    cbar.set_label('Spiral fidelity $F(k_1, k_2, \\pi/2)$', fontsize=10.5)
    cbar.ax.set_title('bands of\nequal width,\nunequal $R^2$',
                      fontsize=8, style='italic', pad=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
    for ext, kw in (('pdf', {'dpi': 400}), ('png', {'dpi': 300})):
        fig.savefig(f"{OUT_STEM}.{ext}", bbox_inches='tight', facecolor='white', **kw)
        print(f"Saved {OUT_STEM}.{ext}")


if __name__ == "__main__":
    main()
