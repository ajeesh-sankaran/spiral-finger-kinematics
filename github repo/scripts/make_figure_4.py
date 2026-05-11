"""
Generate Figure 4: threshold-robustness analysis. Two panels:
  Left  — F(k1, k2, pi/2) heatmap with nested V(tau) contours for tau in
          {0.99, 0.995, 0.999, 0.9995, 0.9999}
  Right — same nested regions shaded from light to dark grey

Uses the precomputed 80x80 coupling grid in data/F80_coupling.npz.

Run:  python scripts/make_figure_4.py
Output: figures/figure4_threshold_robustness.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

from grids import load_or_compute_coupling  # noqa: E402

DATA_PATH = os.path.join(HERE, '..', 'data', 'F80_coupling.npz')
OUT_PATH = os.path.join(HERE, '..', 'figures', 'figure4_threshold_robustness.png')

THRESHOLDS = [0.99, 0.995, 0.999, 0.9995, 0.9999]


def main():
    k1v, k2v, F_grid, _, _ = load_or_compute_coupling(DATA_PATH)
    K1, K2 = np.meshgrid(k1v, k2v)

    i_anat = np.argmin(np.abs(k2v - 0.67))
    j_anat = np.argmin(np.abs(k1v - 1.0))
    F_anat = F_grid[i_anat, j_anat]
    print(f"F at anatomical: {F_anat:.6f}")

    dk1 = k1v[1] - k1v[0]
    dk2 = k2v[1] - k2v[0]
    print("Viable region areas:")
    for tau in THRESHOLDS:
        mask = F_grid >= tau
        area = mask.sum() * dk1 * dk2
        ins = "IN" if F_anat >= tau else "OUT"
        print(f"  tau = {tau:.4f}: area = {area:.4f}, anatomical {ins}")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    # ── Left panel: heatmap with overlaid threshold contours ────────────
    ax = axes[0]
    im = ax.pcolormesh(K1, K2, F_grid, vmin=0.95, vmax=1.0,
                       cmap='viridis', shading='auto')
    plt.colorbar(im, ax=ax, label=r'$F(k_1, k_2, \pi/2)$')

    colours = ['white', 'cyan', 'yellow', 'orange', 'red']
    for tau, col in zip(THRESHOLDS, colours):
        cs = ax.contour(K1, K2, F_grid, levels=[tau], colors=col, linewidths=2)
        ax.clabel(cs, fmt=f'$\\tau={tau}$', inline=True, fontsize=8)
    ax.plot(1.0, 0.67, marker='*', color='red', markersize=15,
            markeredgecolor='white', linestyle='None',
            label=f'anatomical ($k_1=1.0$, $k_2=0.67$)')
    ax.set_xlabel(r'$k_1$ (MCP–PIP coupling)')
    ax.set_ylabel(r'$k_2$ (PIP–DIP coupling)')
    ax.set_title('Spiral fidelity $F$ with viable-region contours')
    ax.legend(loc='lower right', fontsize=9)

    # ── Right panel: nested filled viable regions ────────────────────────
    ax = axes[1]
    fills = [
        (0.99,    '#cccccc', r'$\tau = 0.99$'),
        (0.995,   '#999999', r'$\tau = 0.995$'),
        (0.999,   '#666666', r'$\tau = 0.999$ (primary)'),
        (0.9995,  '#333333', r'$\tau = 0.9995$'),
    ]
    for tau, col, _ in fills:
        mask = F_grid >= tau
        ax.contourf(K1, K2, mask.astype(float), levels=[0.5, 1.5],
                    colors=[col], alpha=0.85)
        ax.contour(K1, K2, F_grid, levels=[tau], colors='black', linewidths=0.6)
    handles = [Patch(facecolor=col, edgecolor='black', label=lbl)
               for _, col, lbl in fills]
    handles.append(plt.Line2D([], [], marker='*', color='red', linestyle='None',
                              markersize=12, markeredgecolor='white',
                              label='anatomical (1, 0.67)'))
    ax.plot(1.0, 0.67, marker='*', color='red', markersize=15,
            markeredgecolor='white', linestyle='None')
    ax.set_xlabel(r'$k_1$ (MCP–PIP coupling)')
    ax.set_ylabel(r'$k_2$ (PIP–DIP coupling)')
    ax.set_title(r'Nested viable regions $\mathcal{V}(\tau)$')
    ax.legend(handles=handles, loc='lower right', fontsize=9)
    ax.set_xlim(k1v.min(), k1v.max())
    ax.set_ylim(k2v.min(), k2v.max())

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=180, bbox_inches='tight', facecolor='white')
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
