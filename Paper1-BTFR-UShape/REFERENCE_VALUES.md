# Reference values

**Single source of truth for every number reported in this work.**

Any document in this repository — the accepted article, the narrative research
document, the Supplementary Information, figure captions, README files — must
agree with this table. When a value is corrected, it is corrected here first,
then propagated.

This file exists because inconsistencies between documents are the failure mode
that costs the most credibility, and because it happened: see *Known
discrepancies* at the end.

Last verified: 14 August 2026, by re-running the scripts in `scripts/` from a
clean checkout.

---

## A note on sample sizes — read this first

The SPARC sample size differs between analyses, and the two figures this
produces are **both correct**:

| Sample | N | What it is |
|---|---|---|
| Full quality-cut sample | 181 | All galaxies with Q ≤ 2 and 30° < i < 80° |
| Environmentally classified | 175 | Those additionally having a documented environmental class |
| 2MRS cross-matched | 169 | Those additionally matched to 2MRS with complete coordinates |

Consequently the SPARC curvature coefficient appears as **+1.33** (N = 175, the
headline value of the article, since the analysis is environmental) and as
**+1.36** (N = 181, the full-sample fit used in the robustness suite). Neither is
an error. Any document quoting one of them must state which sample it refers to.

---

## SPARC

| Quantity | Value | N | Source |
|---|---|---|---|
| Curvature coefficient, environmental classification | +1.329 ± 0.253 (5.26σ, p < 10⁻⁶) | 175 | Supplementary Table S1, row REF |
| Curvature coefficient, full-sample fit | +1.361 | 181 | `regenerate_figure3_robustness.py` |
| BTFR fit intercept / slope | a = −1.058, b = 0.344 | 181 | Methods |
| Monte Carlo: survival | 100.0 % | 181 | `regenerate_figure3_robustness.py`, panel A |
| Monte Carlo: mean coefficient | **1.356** | 181 | idem |
| Bootstrap: 95 % CI | **[0.9308, 1.7312]** | 181 | idem, panel B |
| Bootstrap: fraction U-shaped | 100.0 % | 181 | idem |
| Jackknife by environment | 3 / 4 stable | 181 | idem, panel C |
| Cross-validation: RMSE linear | 0.1336 | 181 | idem, panel D |
| Cross-validation: RMSE quadratic | 0.1230 | 181 | idem |
| Cross-validation: improvement | 7.9 % | 181 | idem |
| Permutation test | p < 0.0001 | 181 | idem, panel E |

### Environmental proxy sensitivity (Supplementary S1)

| Proxy | a | σ_a | σ | p |
|---|---|---|---|---|
| SPARC catalog (REF, 2 Mpc NED) | +1.329 | 0.253 | 5.26 | < 10⁻⁶ |
| 2MRS quartiles, R = 1 Mpc | — | — | — | degenerate |
| 2MRS quartiles, R = 2 Mpc | +0.263 | 0.289 | 0.91 | 0.36 |
| 2MRS quartiles, R = 3 Mpc | +0.449 | 0.263 | 1.71 | 0.09 |
| 2MRS quartiles, R = 5 Mpc | −0.062 | 0.261 | 0.24 | 0.81 |
| 2MRS quartiles, R = 7 Mpc | −0.062 | 0.257 | 0.24 | 0.81 |
| 2MRS quartiles, R = 10 Mpc | −0.788 | 0.246 | 3.20 | 0.002 |

### Model comparison (Supplementary S2, N = 181)

| Model | k | ΔAIC | ΔBIC | w_i (AIC) | R² |
|---|---|---|---|---|---|
| Cubic spline | 5 | 0.00 | 2.36 | 0.602 | 0.287 |
| Broken-line | 4 | 0.84 | 0.00 | 0.396 | 0.276 |
| Quadratic | 3 | 11.14 | 7.11 | 0.002 | 0.225 |
| Linear | 2 | 39.95 | 32.71 | 1.3×10⁻⁹ | 0.081 |
| Flat | 1 | 53.24 | 42.81 | 1.7×10⁻¹² | 0.000 |

---

## ALFALFA and Little THINGS

| Quantity | Value | N | Source |
|---|---|---|---|
| ALFALFA curvature | +0.070 ± 0.028, p = 0.0065 | 21 834 | Article, Detection section |
| Little THINGS curvature | +0.29 ± 0.32, p = 0.39 | 40 | idem (not statistically meaningful) |

---

## IllustrisTNG — stratified analysis (TNG300)

Source: `data/tng300_stratified_results.csv`, rendered by
`scripts/regenerate_figure2_inversion.py`.

| Category | N | a | σ_a |
|---|---|---|---|
| Gas-rich | 444 374 | +0.01728 | 0.00808 |
| Intermediate | 28 810 | +0.01001 | 0.00389 |
| Gas-poor | 150 425 | +0.02939 | 0.01404 |
| Gas-poor + low mass | 115 980 | +0.05156 | 0.01420 |
| Gas-poor + mid mass | 25 665 | −0.02165 | 0.00354 |
| Gas-poor + high mass | 8 779 | −0.01380 | 0.00106 |
| True R-dom (gas-poor + quiescent) | 149 519 | +0.02941 | 0.01400 |
| True R-dom + low mass | 115 813 | +0.05115 | 0.01417 |
| True R-dom + mid mass | 25 019 | −0.02111 | 0.00352 |
| True R-dom + high mass | 8 686 | −0.01342 | 0.00086 |
| Extreme R-dom | 16 924 | −0.01872 | 0.00278 |

Total across all stratified categories: 1 089 994 galaxies.

### Isolation-controlled test (Supplementary S3)

| Stratification | N | a | σ_a | σ | p |
|---|---|---|---|---|---|
| Full R-dom | 9 386 | −0.0211 | 0.0040 | 5.3 | 5×10⁻⁴ |
| Q1 (most dense) | 2 347 | −0.0476 | 0.0092 | 5.2 | 6×10⁻⁴ |
| Q2 (dense) | 2 346 | +0.0089 | 0.0150 | 0.6 | 0.57 |
| Q3 (sparse) | 2 346 | −0.0083 | 0.0222 | 0.4 | 0.72 |
| Q4 (most isolated) | 2 347 | +0.0030 | 0.0081 | 0.4 | 0.72 |

---

## IllustrisTNG — multivariate analysis (TNG100-1, Supplementary S4)

Source: `scripts/multivariate_results.txt`, produced by
`scripts/multivariate_tng_analysis.py`. N = 53 363, standardised predictors,
HC3 robust standard errors.

| Term | M0 | M1 | M2 |
|---|---|---|---|
| Intercept | −0.0790 ± 0.0023 | −0.0285 ± 0.0016 | −0.0683 ± 0.0018 |
| env | −0.3092 ± 0.0016 | −0.0226 ± 0.0018 | +0.0323 ± 0.0030 |
| **env²** | **+0.0790 ± 0.0019** | **+0.0285 ± 0.0015** | **+0.0095 ± 0.0017** |
| f_gas | — | +0.3509 ± 0.0018 | +0.3680 ± 0.0022 |
| log M⋆ | — | +0.0103 ± 0.0010 | +0.0280 ± 0.0011 |
| env × f_gas | — | — | −0.1089 ± 0.0033 |
| env × log M⋆ | — | — | −0.0305 ± 0.0013 |
| env² × f_gas | — | — | +0.0467 ± 0.0019 |
| R² | 0.3750 | 0.6793 | 0.6977 |
| AIC | 37313 | 1716 | −1426 |
| BIC | 37339 | 1761 | −1355 |
| env² significance | 42.6σ | 19.0σ | 5.7σ |

**Attenuation.** The curvature coefficient in the fully controlled model is
about **eight times smaller** than in the environment-only fit. This is stated
explicitly wherever the multivariate result is reported.

### Marginal environmental slope (model M2, at mean log M⋆)

| Population | ∂Δ/∂ρ | Significance |
|---|---|---|
| Gas-rich (f_gas at +1σ) | −0.0765 ± 0.0024 | 31.7σ |
| Star-dominated (f_gas at −1σ) | +0.1412 ± 0.0059 | 24.1σ |
| Interaction env × f_gas | −0.1089 ± 0.0033 | 33.0σ |

---

## Screening

| Quantity | Value | Status |
|---|---|---|
| Characteristic screening density ρ₀ | ~10⁻²⁴ g cm⁻³ | **Adopted phenomenological parameter, not a prediction.** Fixed a posteriori. |
| Globular clusters (null test) | r = +0.12, p = 0.30 | Baumgardt & Hilker catalogue |
| Solar System (Cassini) | \|γ_PPN − 1\| < 2.3×10⁻⁵ | Bertotti et al. 2003 |

---

## Values that are NOT part of this work

The following appear in the December 2025 documents preserved under `legacy/`
and are **withdrawn**. They must not be quoted from any current document:

| Withdrawn claim | Why |
|---|---|
| λ_QR = 1.23 ± 0.35, universal across 14 orders of magnitude | Universality rejected: χ²/dof = 4.5, p = 0.001. Internally inconsistent across documents (1.23 vs 31.4). |
| λ_QR ~ O(1) derived from KKLT compactification | First-principles computation gives 10⁻¹¹ to 10⁻³. |
| Sign inversion at 26σ | No environmental control. Isolation-controlled values are 5.3σ / 5.2σ. |
| ρ·G_eff = const ("conservation law") | Required SO(2) symmetry absent from the Lagrangian. |
| Theory-of-Everything status | No General Relativity limit. |
| Clinical biomarker results (NHANES) | Exploratory, never validated. |

---

## Two quantities that have both been called C_Q and C_R

Earlier versions of the narrative document reported "C_Q = +2.82 +/- 0.15, C_R =
-0.72 +/- 0.08" as coupling constants of the Lagrangian. They are not coupling
constants, and the uncertainties quoted alongside them do not appear in any
source file. The confusion is worth recording, since two distinct quantities
carry the same symbols across the project.

| Symbol as used | What it actually is | Value | Source |
|---|---|---|---|
| Curvature against the Q proxy | Curvature of the residual when environment is measured by gas content | +2.8156 | `BTFR/Test-MicrophysicsQHI/test_qhi_refined_results.csv`, column `curvature_Q_proxy` |
| Curvature against the R proxy | Same, measured by stellar dominance | -0.7201 | idem, column `curvature_R_proxy` |
| C_Q (calibration) | Fitted coefficient of the environmental response, Method 1 | +0.6082 +/- 0.1264 | `tests/04_calibration/calibrate_params.py` |
| C_R (calibration) | idem | +0.7524 +/- 0.1153 | idem |

The opposition of sign between the first two is the sign inversion measured on
SPARC directly, and it is a real result. The two calibration coefficients are
both positive because that fit parameterises the environmental response
differently, and in that parameterisation two positive coefficients are what
produces a U-shape. Neither pair measures a coupling of the Lagrangian: the
framework contains no forward model that would allow such a conversion.

The uncertainties +/-0.15 and +/-0.08 previously attached to +2.82 and -0.72 have
no source and have been removed.

---

## Known discrepancies

**1. Figure 3 caption in the accepted article (open).**
The caption of Figure 3 states "mean coefficient a = 1.33" and a bootstrap
interval of "[0.94, 1.75]". The figure itself, generated by the deterministic
script (`np.random.seed(42)`), shows 1.356 and [0.931, 1.731]. The figure is
correct; the caption carries values from an earlier revision.
→ **To be corrected at the proof stage.**

**2. Narrative research document v3.1 (being addressed).**
Quotes a = 1.36 ± 0.24 without stating that this is the N = 181 fit, which
invites confusion with the N = 175 headline value. Being corrected in v4.0 of
that document, together with the withdrawn claims listed above.

---

## How to verify this table

```bash
cd scripts
python multivariate_tng_analysis.py     # multivariate block
python regenerate_figure2_inversion.py  # TNG300 stratified block
python regenerate_figure3_robustness.py # SPARC robustness block
```

Values printed to the console and written to `multivariate_results.txt` must
match this file exactly. Any deviation means the input data or the software
environment has changed, and must be investigated before any output is used.
