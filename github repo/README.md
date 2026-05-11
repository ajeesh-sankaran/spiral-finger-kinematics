# Spiral Finger Kinematics

Code accompanying the paper *Modelling equiangular spiral arc kinematics of the human finger: exact conditions, viable parameter regions, and the role of phalangeal proportions* (Mathematical Medicine and Biology, submitted).

## Overview

This repository contains the Python code that produces every computational figure in the paper. The model is a planar three-link kinematic chain with linear joint coupling — a standard abstraction of the tendon-driven human finger:

- The three links represent the proximal, middle, and distal phalanges.
- Two coupling coefficients (k₁, k₂) tie the joint angles together: θₘ = k₁ θₚ and θ_d = k₂ θₘ.
- The fingertip trajectory is fit to an equiangular spiral r = a e^(μψ) about an optimised pole; the spiral fidelity is the R² of the log-linear fit.

The viable approximation region 𝒱(τ) is the set of coupling coefficients (k₁, k₂) for which the spiral fidelity F(k₁, k₂, π/2) meets or exceeds threshold τ. The paper characterises 𝒱(0.999) in both coupling space and phalangeal length-ratio space and analyses its sensitivity to the threshold choice.

## Repository layout

```
spiral-finger-kinematics/
├── README.md
├── requirements.txt
├── src/
│   ├── kinematics.py        # core model: trajectory, pole fit, spiral fidelity
│   └── grids.py             # batched grid computation utilities
├── scripts/
│   ├── make_figure_1.py     # three representative spiral arcs
│   ├── make_figure_2.py     # F over coupling space, V(0.999) contour
│   ├── make_figure_3.py     # F over length-ratio space
│   └── make_figure_4.py     # threshold-robustness analysis
├── data/
│   ├── F80_coupling.npz     # precomputed F over (k1, k2), 80x80 grid
│   └── F60_lengths.npz      # precomputed F over (Lp/Lm, Lm/Ld), 60x60 grid
└── figures/                 # generated PNGs (created on first run)
```

## Requirements

- Python 3.9 or later
- numpy
- scipy
- matplotlib

Install with `pip install -r requirements.txt`.

## Quick start

Verify that the fitter reproduces the paper's anatomical-point checkpoint:

```bash
python src/kinematics.py
```

Expected output:

```
Sanity check at anatomical (k1 = 1.0, k2 = 0.67):
  F             = 0.999705   (expected 0.9997)   OK
  pole P        = (0.0505, 0.1784)   (expected (0.0505, 0.1776))   OK
  alpha         = 75.60 deg   (expected 75.6 deg)   OK
  Overall: PASS
```

Generate any figure:

```bash
python scripts/make_figure_1.py
python scripts/make_figure_2.py
python scripts/make_figure_3.py
python scripts/make_figure_4.py
```

Figures are written to `figures/`. Figures 2, 3, and 4 use precomputed grids in `data/`; if the `.npz` files are absent, the scripts will recompute them (3–5 minutes each on a single core) and cache the result.

## Numerical procedure

The pole optimisation in `spiral_fidelity` is a two-stage search:

1. **Coarse grid.** R² is evaluated on an 81 × 81 grid of candidate poles spanning [−2, 2]² in normalised coordinates, with step size 0.05. This domain is several times the finger length and ensures the global optimum lies in its interior for every biologically relevant (k₁, k₂).
2. **Local refinement.** The best coarse-grid pole is refined by Nelder–Mead simplex with tolerances xatol = 10⁻¹⁰ and fatol = 10⁻¹².

The trajectory is sampled at N = 200 uniformly spaced values of θₚ ∈ [0, π/2]. Phase unwrapping is applied to ψ before the linear regression to remove the 2π discontinuity of arctan2.

These choices are described in §2.4 of the paper.

## Verification checkpoint

At the anatomical operating point (k₁ = 1, k₂ = 0.67) with 2:1:1 phalangeal lengths:

| Quantity | Value |
|---|---|
| F(1, 0.67, π/2) | 0.9997 |
| Optimised pole P | (0.0505, 0.1776) |
| Spiral cotangent μ | −0.258 |
| Tangent angle α | 75.6° |

A working installation should reproduce these to four decimal places. `sanity_check()` in `src/kinematics.py` confirms this automatically.

## Citation

If you use this code, please cite the paper.

## License

MIT — see [LICENSE](LICENSE).
