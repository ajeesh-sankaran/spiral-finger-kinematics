# Spiral Finger Kinematics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20127758.svg)](https://doi.org/10.5281/zenodo.20127758)

Code accompanying the paper *Modelling equiangular spiral arc kinematics of the human finger: exact conditions, viable parameter regions, and the role of phalangeal proportions* (Mathematical Medicine and Biology, under revision).

The badge above is the Zenodo **concept** DOI and always resolves to the most recent archived release.

## Overview

This repository contains the Python code that produces every computational figure in the paper. The model is a planar three-link kinematic chain with linear joint coupling — a standard abstraction of the tendon-driven human finger:

- The three links represent the proximal, middle, and distal phalanges.
- Two coupling coefficients (k₁, k₂) tie the joint angles together: θₘ = k₁ θₚ and θ_d = k₂ θₘ.
- The fingertip trajectory is fitted to an equiangular spiral r = a e^(μψ) about an optimised pole; the spiral fidelity is the R² of the log-linear fit.

The viable approximation region 𝒱(τ) is the set of coupling coefficients (k₁, k₂) for which the spiral fidelity F(k₁, k₂, π/2) meets or exceeds threshold τ. The paper characterises 𝒱(0.999) in both coupling space and phalangeal length-ratio space, and analyses its sensitivity to the threshold choice and to phalangeal proportions.

## Repository layout

```
spiral-finger-kinematics/
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── kinematics.py                    # core model: trajectory, pole fit, spiral fidelity
│   └── grids.py                         # batched grid computation utilities
├── scripts/
│   ├── make_figure_1.py                 # model + spiral geometry schematic
│   ├── make_figure_2.py                 # three representative spiral arc fits
│   ├── make_figure_3.py                 # F over coupling space, banded at all five thresholds
│   ├── make_figure_4.py                 # F over length-ratio space, two panels
│   ├── run_boundary_length_sweeps.py    # length-ratio sweeps at boundary coupling points
│   ├── run_wide_length_sweep.py         # wide-domain length-ratio sweep, two orders of magnitude
│   └── verify_numbers.py                # recompute every number quoted in the paper
├── data/
│   ├── F80_coupling.npz                 # F over (k1, k2), 80x80 grid, with optimised poles
│   ├── F60_lengths.npz                  # F over length ratios at anatomical coupling
│   ├── F60_boundary_k1_1p25_k2_0p67.npz # length-ratio sweep near the k1 boundary
│   ├── F60_boundary_k1_1p00_k2_1p00.npz # length-ratio sweep near the k2 boundary
│   └── F_lengths_wide.npz               # wide-domain sweep, Lp/Lm and Lm/Ld in [0.1, 10]
└── figures/                             # generated figures, PDF and PNG
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
  F             = 0.999696   (expected 0.9997)   OK
  pole P        = (0.0503, 0.1794)   (expected (0.0505, 0.1776))   OK
  alpha         = 75.56 deg   (expected 75.6 deg)   OK
  Overall: PASS
```

Digits beyond the fourth decimal place depend on the BLAS and SciPy build; the checkpoint tolerances are set accordingly.

Generate the figures:

```bash
python scripts/make_figure_1.py
python scripts/make_figure_2.py
python scripts/make_figure_3.py
python scripts/make_figure_4.py
```

Figures are written to `figures/` in both PDF and PNG. Figures 3 and 4 read the precomputed grids in `data/`.

## Verifying the paper's numbers

Every numerical claim in the paper — the anatomical checkpoint, the viable-region area, the boundary positions in k₁, the length-ratio range, the boundary and wide sweeps, and the closure bounds — can be recomputed from the archived grids:

```bash
PYTHONPATH=. python scripts/verify_numbers.py
```

The script prints a table of claimed against computed values and exits non-zero if any disagree.

## Numerical procedure

The spiral fit has a nested structure. For a fixed pole, the equiangular spiral r = a e^(μψ) is equivalent to the linear relationship ln r = μψ + ln a, so the cotangent μ and the log-scale are obtained in **closed form** by ordinary least squares at that pole — there is no iterative optimisation at this stage. Only the two-dimensional pole position is determined numerically, in two stages:

1. **Coarse grid.** R² is evaluated on an 81 × 81 grid of candidate poles spanning [−2, 2]² in normalised coordinates, with step size 0.05. This domain is several times the finger length and ensures the global optimum lies in its interior for every biologically relevant (k₁, k₂).
2. **Local refinement.** The best coarse-grid pole is refined by Nelder–Mead simplex with tolerances xatol = 10⁻¹⁰ and fatol = 10⁻¹².

The trajectory is sampled at N = 200 uniformly spaced values of θₚ ∈ [0, π/2]. Phase unwrapping is applied to ψ before the linear regression to remove the 2π discontinuity of arctan2.

These choices are described in §2.4 of the paper.

## Verification checkpoint

At the anatomical operating point (k₁ = 1, k₂ = 0.67) with 2:1:1 phalangeal lengths:

| Quantity | Value |
|---|---|
| F(1, 0.67, π/2) | 0.9997 |
| Optimised pole P | (0.050, 0.179) |
| Spiral cotangent μ | −0.258 |
| Tangent angle α | 75.6° |

`sanity_check()` in `src/kinematics.py` confirms this automatically.

## Supplementary length-ratio sweeps

Section 4.4 of the paper establishes length-ratio insensitivity. Three supporting sweeps are archived here.

**At two coupling points near the viable-region boundary**, to check that the insensitivity is not specific to the anatomical coupling:

- **(k₁ = 1.25, k₂ = 0.67)** — near the upper k₁ boundary at anatomical k₂. F ≥ 0.999 across 98.8% of the length-ratio space; minimum F = 0.9989 in the corner Lp/Lm > 3.2, Lm/Ld < 0.6.
- **(k₁ = 1.00, k₂ = 1.00)** — near the upper k₂ boundary at anatomical k₁. F ≥ 0.999 across the entire length-ratio space; minimum F = 0.9991.

Regenerate with (≈ 40 min total on a single core):

```bash
python scripts/run_boundary_length_sweeps.py
```

**Over a wide domain**, spanning two orders of magnitude in each ratio, to test whether the viable region has any length-ratio boundary at all:

- **Lp/Lm ∈ [0.1, 10], Lm/Ld ∈ [0.1, 10]** at anatomical coupling. F ≥ 0.999 at every point; minimum F = 0.99950 at Lp/Lm = 10, Lm/Ld = 0.16.

Regenerate with (≈ 1 min):

```bash
python scripts/run_wide_length_sweep.py
```

Both scripts skip any file that already exists, so they can be run safely against a populated `data/` directory.

## Citation

If you use this code, please cite the paper. The archived releases of this repository are on Zenodo under the concept DOI [10.5281/zenodo.20127758](https://doi.org/10.5281/zenodo.20127758), which resolves to the most recent version.

## License

Released under the MIT License, covering both code and data. See the `LICENSE` file for the full text.
