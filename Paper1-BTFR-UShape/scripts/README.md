# Reproducing the published results

All figures and statistical results of the paper can be regenerated from this
directory. No external downloads are required: the derived data tables are
included in `../data/`.

## Requirements

```
pip install -r ../../requirements.txt
```
Python 3 with NumPy, SciPy, pandas, statsmodels, scikit-learn, Matplotlib.

## Scripts

Run from within this `scripts/` directory.

### `multivariate_tng_analysis.py`
Supplementary Section S4 — nested multivariate regression on the TNG100-1
catalogue (N = 53,363), with environment as the primary independent variable and
gas fraction and stellar mass as statistical controls. Standardised predictors,
HC3 heteroskedasticity-consistent standard errors.

Reads `../data/tng_galaxies.csv`.
Writes `multivariate_results.txt` (full regression tables) and
`multivariate_tng_analysis.png` (Supplementary Figure S4).

Expected output — the quadratic environmental coefficient across the model
hierarchy:

| Model | `env2` coefficient | Significance |
|---|---|---|
| M0 (environment only) | +0.0790 ± 0.0019 | 42.6σ |
| M1 (+ controls) | +0.0285 ± 0.0015 | 19.0σ |
| M2 (+ interactions) | +0.0095 ± 0.0017 | 5.7σ |

Interaction `env2:fgas` = +0.0467 ± 0.0019 (24.6σ).

Note the attenuation: controlling for internal galaxy properties reduces the
curvature by roughly a factor of eight. This is discussed explicitly in the
paper and in Supplementary Section S4.

### `regenerate_figure2_inversion.py`
Figure 2 — sign inversion of the environmental response between gas-rich and
gas-poor populations in TNG300.

Reads `../data/tng300_stratified_results.csv` (1,089,994 galaxies, stratified
results). Writes `../figures/figure2_inversion.png`.

### `regenerate_figure3_robustness.py`
Figure 3 — statistical robustness of the SPARC U-shape detection: Monte Carlo
perturbation, bootstrap, jackknife by environment, cross-validation, permutation.

Reads `../data/sparc_with_environment.csv`. Writes
`../figures/figure3_robustness.png`.

Expected console output (these are the values quoted in the paper):

```
MC survival  : 100.0%
Bootstrap CI : [0.9308, 1.7312], 100.0% U-shape
Jackknife    : 3/4 stable
Cross-val    : RMSE lin=0.1336, quad=0.1230 (7.9% improvement)
Permutation  : p = 0.0000
```

These tests assess **internal statistical robustness only**. They do not
propagate observational systematics (environmental misclassification, distance
and inclination uncertainties, baryonic mass errors), which remain unquantified.

## If your numbers differ

The values above are those reported in the published paper. Any deviation
indicates a change in the input data or the software environment, and should be
investigated before the output is used.
