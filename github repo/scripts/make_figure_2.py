"""
Generate Figure 2: spiral fidelity F(k1, k2, pi/2) heatmap over the coupling
parameter space, with viable-region contour V(0.999).

Uses the precomputed 80x80 grid in data/F80_coupling.npz; recomputes if absent.

Run:  python scripts/make_figure_2.py
Output: figures/figure2_coupling_space.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

from grids import load_or_compute_coupling  # noqa: E402

DATA_PATH = os.path.join(HERE, '..', 'data', 'F80_coupling.npz')
OUT_PATH = os.path.join(HERE, '..', 'figures', 'figure2_coupling_space.png')

# Red-yellow-green colourmap matching the paper's figure style
_RYG_COLOURS = [(0.62, 0.13, 0.10), (0.85, 0.50, 0.20), (0.92, 0.78, 0.30),
                (0.30, 0.65, 0.30), (0.10, 0.45, 0.20)]
RYG_CMAP = LinearSegmentedColormap.from_list('ryg', _RYG_COLOURS, N=256)


def main():
    k1v, k2v, F_grid, _, _ = load_or_compute_coupling(DATA_PATH)
    print(f"Grid: {len(k1v)} x {len(k2v)} cells")
    print(f"F range: [{F_grid.min():.6f}, {F_grid.max():.6f}]")
    j_anat = np.argmin(np.abs(k1v - 1.0))
    i_anat = np.argmin(np.abs(k2v - 0.67))
    print(f"F at anatomical (k1={k1v[j_anat]:.3f}, k2={k2v[i_anat]:.3f}): "
          f"{F_grid[i_anat, j_anat]:.4f}")

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    K1, K2 = np.meshgrid(k1v, k2v)
    im = ax.pcolormesh(K1, K2, F_grid, vmin=0.97, vmax=1.0,
                       cmap=RYG_CMAP, shading='auto')

    cs999 = ax.contour(K1, K2, F_grid, levels=[0.999], colors='white', linewidths=2.5)
    ax.clabel(cs999, fmt='$R^2 = 0.999$', inline=True, fontsize=10)
    ax.contour(K1, K2, F_grid, levels=[0.998], colors='white',
               linewidths=1.2, linestyles='--')

    ax.plot(1.0, 0.67, marker='*', color='blue', markersize=22,
            markeredgecolor='white', markeredgewidth=1.5, zorder=8, linestyle='None')
    ax.axhline(0.67, color='gray', lw=0.7, ls=':', alpha=0.6, zorder=2)
    ax.axvline(1.0, color='gray', lw=0.7, ls=':', alpha=0.6, zorder=2)

    ax.text(0.55, 0.42, "Viable\n($R^2 \\geq 0.999$)",
            ha='center', va='center', fontsize=12, color='white',
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#1f5e3a',
                     edgecolor='white', lw=1.5))
    ax.text(1.40, 1.05, "Non-viable\n($R^2 < 0.999$)",
            ha='center', va='center', fontsize=12, color='white',
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#7a1f10',
                     edgecolor='white', lw=1.5))

    ax.annotate("Anatomical\n($k_1=1$, $k_2=0.67$)\n$R^2 = 0.9997$",
                xy=(1.0, 0.67), xytext=(0.55, 0.95),
                fontsize=10, color='white', ha='center',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#1f3a5e',
                         edgecolor='white', lw=1.2),
                arrowprops=dict(arrowstyle='-|>', color='white', lw=1.2))

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Spiral fidelity $R^2$", fontsize=11)
    cbar.set_ticks([0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 0.999, 1.0])

    ax.set_xlabel("$k_1$  (MCP–PIP coupling coefficient)", fontsize=12)
    ax.set_ylabel("$k_2$  (PIP–DIP coupling coefficient)", fontsize=12)
    ax.set_title(
        "Figure 2.  Viable approximation region $\\mathcal{V}(0.999)$ "
        "in coupling parameter space\n"
        "$L_p : L_m : L_d = 2 : 1 : 1$,  $\\theta_p \\in [0\\degree, 90\\degree]$,  "
        "pole optimised by LM at each point",
        fontsize=11,
    )
    ax.set_xlim(k1v.min(), k1v.max())
    ax.set_ylim(k2v.min(), k2v.max())

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=180, bbox_inches='tight', facecolor='white')
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
