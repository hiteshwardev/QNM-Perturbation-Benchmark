# Quasinormal-mode perturbation theory for exterior perturbers

Exact error laws and a distance-independent resummation, computed for a one-dimensional
dielectric slab cavity perturbed by a thin layer placed outside it.

First-order quasinormal-mode (QNM) perturbation theory predicts frequency shifts that grow
exponentially with the distance to an exterior perturber, because the QNM field grows
exponentially outside the cavity. This repository compares naive, regularized, self-consistent
and resummed perturbative schemes, an embedding-basis resonant-state expansion and a Lorentzian
fit of the transmission with exact transfer-matrix poles. It derives the second-order error of
first-order theory from the Fabry–Pérot form of the pole condition and verifies it in 40-digit
arithmetic.

## Main results

* The exact shift grows at the spatial rate of the QNM field, so the exponential growth is physical.
* The error of first-order theory has three parts. One grows linearly with the gap, one is set by
  the background-to-residue ratio α of the cavity reflection, and one is the second Born term of
  the layer. The law contains no fitted parameter and agrees with 40-digit exact poles to 5×10⁻⁷.
* Evaluating the overlap at the shifted frequency removes the growing term. Resumming the
  Fabry–Pérot series with α removes the rest, leaving a relative error of about 3.9δ that does not
  depend on the distance.
* The error of the regularized scheme follows the closed-form floor |1 − c e^(−γ_f d)|.

## Reproducing the study

The whole study runs from `notebook.ipynb`, which computes every number and figure. It takes about
two minutes on two cores of a 2.1 GHz processor.

```bash
python -m venv .venv
source .venv/bin/activate            # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

With conda, `conda env create -f environment.yml` followed by `conda activate qnm-exterior-perturbation`
sets up the same environment. You can also open the notebook in Jupyter and run all cells in
order. The notebook must be run from the repository root, because it imports the modules in `src/`.

## Repository layout

```
notebook.ipynb       complete study: theory, validation, computations, figures, summary
src/
  tmm.py             transfer matrices, Newton pole search, pole continuation
  qnm.py             closed-form slab quasinormal modes and normalized fields
  perturbation.py    exterior overlap and the naive, regularized, self-consistent and resummed shifts
  rse.py             resonant-state expansion with a homogeneous basis slab
  lorentzian.py      Lorentzian fit of the exact transmission
  stats.py           straight-line fits with Newey-West standard errors
  plotting.py        fixed-size single-panel figures with an automatic layout check
figures/             fig01 ... fig21 as PDF and 600 dpi PNG, and manifest.csv with the captions
results/             data tables written by the notebook and summary.json with all quoted numbers
requirements.txt     pinned Python dependencies
environment.yml      conda environment
```

## Outputs

| file in `results/` | content |
|---|---|
| `benchmarks.csv` | known-answer validation benchmarks and their pass criteria |
| `interior_control.csv` | first-order theory for a layer inside the cavity |
| `tracking.csv` | comparison of pole-continuation conventions |
| `identity.csv` | minimal-loss mode identification at large gaps |
| `shifts_and_errors.csv` | exact poles, all scheme shifts and relative errors on the full grid |
| `line_scans.csv` | 600-gap scans used for display |
| `lorentzian.csv` | Lorentzian fits of the exact transmission |
| `naive_error_rates.csv` | growth-rate fits of the naive error |
| `second_order_coefficient_40digit.csv` | second-order coefficient from 40-digit exact poles |
| `regularized_floor.csv` | small-strength error of the regularized scheme and the floor law |
| `crossover.csv` | gap at which each scheme first exceeds 10% error |
| `cavities.csv` | scope campaign across three cavities |
| `geometry.csv` | scope campaign across perturber thickness and sign |
| `rse.csv`, `rse_exponents.csv`, `rse_deep.csv` | resonant-state-expansion convergence |
| `scheme_summary.csv` | summary of the perturbative schemes |
| `summary.json` | every number quoted in the manuscript |

## Conventions

Time dependence exp(−iωt), c = 1, lengths in units of the cavity length L and wavenumbers in
units of 1/L. The production cavity has refractive index 2π in vacuum, and the perturbing layer
has thickness 0.1 L and permittivity change δ(n_R² − 1).

## License

The code is released under the MIT License (see `LICENSE`). The figures and data files in
`figures/` and `results/` are released under the Creative Commons Attribution 4.0 International
license (CC BY 4.0).

## Citation

H. K. Singh, *Quasinormal-mode perturbation theory for exterior perturbers: exact error laws and a
distance-independent resummation*, manuscript in preparation (2026).
