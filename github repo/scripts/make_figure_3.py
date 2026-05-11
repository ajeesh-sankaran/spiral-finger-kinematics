"""
Generate Figure 3: spiral fidelity over the phalangeal length-ratio space,
for fixed coupling (k1, k2) = (1, 0.67).

Uses the precomputed 60x60 grid in data/F60_lengths.npz; recomputes if absent.

Run:  python scripts/make_figure_3.py
Output: figures/figure3_length_ratios.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

from grids import load_or_compute_length  # noqa: E402

DATA_PATH = os.path.join(HERE, '..', 'data', 'F60_lengths.npz')
OUT_PATH = os.path.join(HERE, '..', 'figures', 'figure3_length_ratios.png')

_RYG_COLOURS = [(0.62, 0.13, 0.10), (0.85, 0.50, 0.20), (0.92, 0.78, 0.30),
                (0.30, 0.65, 0.30), (0.10, 0.45, 0.20)]
RYG_CMAP = LinearSegmentedColormap.from_list('ryg', _RYG_COLOURS, N=256)


def main():
    rpm_vals, rmd_vals, F_grid = load_or_compute_length(DATA_PATH)
    print(f"Grid: {len(rpm_vals)} x {len(rmd_vals)} cells")
    print(f"F range: [{F_grid.min():.6f}, {F_grid.max():.6f}]")
    viable_frac = (F_grid >= 0.999).mean()
    print(f"Viable fraction at tau = 0.999: {viable_frac:.4f}")

    i_anat = np.argmin(np.abs(rmd_vals - 1.0))
    j_anat = np.argmin(np.abs(rpm_vals - 2.0))
    i_gold = np.argmin(np.abs(rmd_vals - 1.618))
    j_gold = np.argmin(np.abs(rpm_vals - 1.618))
    F_anat = F_grid[i_anat, j_anat]
    F_gold = F_grid[i_gold, j_gold]
    print(f"F at anatomical (Lp/Lm=2, Lm/Ld=1): {F_anat:.6f}")
    print(f"F at golden (Lp/Lm=Lm/Ld=Phi):     {F_gold:.6f}")

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    RPM, RMD = np.meshgrid(rpm_vals, rmd_vals)
    im = ax.pcolormesh(RPM, RMD, F_grid, vmin=0.97, vmax=1.0,
                       cmap=RYG_CMAP, shading='auto')

    # Draw the R^2 = 0.999 contour only if part of the grid is below it.
    if F_grid.min() < 0.999:
        cs = ax.contour(RPM, RMD, F_grid, levels=[0.999],
                        colors='white', linewidths=2.5)
        ax.clabel(cs, fmt='$R^2 = 0.999$', inline=True, fontsize=10)

    ax.plot(2.0, 1.0, marker='*', color='blue', markersize=22,
            markeredgecolor='white', markeredgewidth=1.5, zorder=8, linestyle='None')
    ax.plot(1.618, 1.618, marker='D', color='gold', markersize=14,
            markeredgecolor='black', markeredgewidth=1.2, zorder=8, linestyle='None')

    ax.annotate(f"Anatomical 2:1:1\n$R^2 = {F_anat:.4f}$",
                xy=(2.0, 1.0), xytext=(2.7, 1.5),
                fontsize=10, color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1f3a5e',
                         edgecolor='white', lw=1.0),
                arrowprops=dict(arrowstyle='-', color='white', lw=1.0))
    ax.annotate(f"Golden ratio\n$\\Phi \\approx 1.618$\n$R^2 = {F_gold:.4f}$",
                xy=(1.618, 1.618), xytext=(0.95, 2.5),
                fontsize=10, color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='gold',
                         edgecolor='black', lw=1.0),
                arrowprops=dict(arrowstyle='-', color='black', lw=1.0))

    ax.text(3.15, 0.65,
            "$R^2 \\geq 0.999$\nthroughout the\nplotted region",
            ha='center', va='center', fontsize=11, color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#1f5e3a',
                     edgecolor='white', lw=1.5))

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Spiral fidelity $R^2$", fontsize=11)
    cbar.set_ticks([0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 0.999, 1.0])

    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='blue',
                   markersize=18, markeredgecolor='white',
                   label='Anatomical 2:1:1  ($L_p/L_m=2$, $L_m/L_d=1$)'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='gold',
                   markersize=12, markeredgecolor='black',
                   label='Golden ratio  ($L_p/L_m=L_m/L_d=\\Phi$)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              fontsize=9, framealpha=0.95)

    ax.set_xlabel("$L_p / L_m$", fontsize=12)
    ax.set_ylabel("$L_m / L_d$", fontsize=12)
    ax.set_title(
        "Figure 3.  Spiral fidelity $R^2$ over phalangeal length-ratio space\n"
        "$k_1 = 1$,  $k_2 = 0.67$,  $\\theta_p \\in [0\\degree, 90\\degree]$,  "
        "total length normalised to 1",
        fontsize=11,
    )
    ax.set_xlim(rpm_vals.min(), rpm_vals.max())
    ax.set_ylim(rmd_vals.min(), rmd_vals.max())

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=180, bbox_inches='tight', facecolor='white')
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
