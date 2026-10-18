# Exact Error Laws and Distance-Independent Resummation for Quasinormal-Mode Perturbation by Exterior Scatterers

**Author:** Hitesh Kumar Singh · [ORCID: 0009-0008-1583-4848](https://orcid.org/0009-0008-1583-4848)
*Department of Physics, Kurukshetra University, Kurukshetra, Haryana, India*

---

## Abstract

First-order quasinormal-mode (QNM) perturbation theory predicts exponentially increasing frequency shifts when a perturber is placed progressively farther outside an open resonator, but the accuracy of this prediction and the errors of proposed corrections remain insufficiently quantified. This work derives parameter-free error laws and a distance-independent resummation for a one-dimensional dielectric slab cavity coupled to a thin exterior layer. By expressing the exact pole condition in Fabry-Pérot form and separating the cavity reflection into its QNM pole and a non-resonant background, the complete second-order error of first-order perturbation theory is obtained. Exact poles in 40-digit arithmetic verify the law to a maximum relative deviation of 5.0 × 10⁻⁷. Across cavities spanning a factor of 14 in radiative loss, the exact shift grows at the spatial rate of the QNM field to within 0.2%. A resummed approximation removes the distance-dependent error growth, leaving a predicted relative error of 3.86δ. At δ = 10⁻³ and a separation of 30 cavity lengths, the naive, regularized, self-consistent and resummed schemes have relative errors of 37%, 96%, 11% and 0.4%.

## Core Objectives

- Quantify, against exact poles, the accuracy of the QNM perturbation schemes used for exterior scatterers.
- Explain analytically why each scheme fails, and whether the exponential growth of the shift is physical.
- Construct a remedy that stays within perturbation theory.

## Scientific Significance

QNM perturbation theory is a standard tool in nanophotonics, yet the field of a QNM grows exponentially outside the resonator and published treatments disagree on whether the resulting growth of the shift is an artefact. This project turns that qualitative disagreement into quantitative error laws and a prescription for when each scheme can be trusted. The same tension between leaky-mode response and distant perturbations appears in other open systems such as black-hole QNMs, noted in the manuscript only as an analogy.

## Methodological Approach

- **Analytical:** the exact pole condition is written in Fabry-Pérot form; the cavity reflection is split into its QNM pole and a non-resonant background; the expansion to second order gives the error law.
- **Exact reference:** transfer-matrix poles continued in the perturbation strength, verified with 50-digit Muller iteration and 40-digit arithmetic.
- **Benchmarks:** four perturbative schemes, a Lorentzian fit of the transmission and an embedding-basis resonant-state expansion (RSE), over four decades of strength and gaps up to 30 cavity lengths (572 grid cells).
- **Scope tests:** three cavities, five layer thicknesses and both signs of the permittivity change, designed after the law was derived. A written analysis plan fixed expectations in advance, and the unmet expectations are reported.

## Contributions

1. The complete second-order error of first-order QNM theory for an exterior layer: a term linear in the gap, a constant set by the ratio of non-resonant background to pole residue, and a decaying second-Born term.
2. A demonstration that the exponential growth of the exact shift is physical (field rate to within 0.2%), while the error of the naive formula grows at twice that rate.
3. An analytic floor, 1 − c·exp(−γ<sub>f</sub> d), for the regularized scheme, showing it discards the physical growth.
4. A distance-independent resummation whose error is set by the perturber alone.
5. Validity maps and an RSE benchmark that define the regimes of each scheme.

6. **Positioning relative to prior work.** The project builds on the regularized-mode treatment of Franke, Ren and Hughes and on the argument of Wu, Jaramillo and Lalanne that the exponential growth is physical (arXiv:2401.15112).

## Key Findings

- The second-order coefficient agrees with the 40-digit exact result to 5.0 × 10⁻⁷ (maximum relative deviation).
- Naive error: 37% at 30 cavity lengths (δ = 10⁻³); regularized: 96%; self-consistent: 11%; resummed: 0.4%, and 1.5 × 10⁻⁵ (median) when the second Born term is included.
- Fraction of the 572 grid cells with error below 10%: naive 47%, regularized 12%, self-consistent 52%, resummed 63% (73% with the second Born term).
- The embedding-basis RSE meets the 10⁻⁶ error target at N = 801 even for the four hardest configurations (gap 3 L).

## Scope and Limitations

The study is one-dimensional, transverse-electric and nondispersive; extension to three-dimensional vectorial resonators remains to be shown. The resummed scheme retains the second Born term of the perturber, and all single-mode schemes fail when the perturbed pole hybridizes with resonances between cavity and perturber.

## Roles

Hitesh Kumar Singh is the sole author. Hitesh Kumar Singh conceived the study, derived the error laws and resummation, wrote the software, designed the benchmark and scope campaigns, carried out the analysis and wrote the manuscript.

## Reproducibility and Availability

Running `notebook.ipynb` from top to bottom (Python 3.11, packages pinned in `requirements.txt`) reproduces every number and figure in about 100 s on two cores. The code is released under the MIT License.
