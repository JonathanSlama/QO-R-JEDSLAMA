# Paper 4 - Complete Methodology

## Empirical Validation of the QO+R Framework

**Version:** 2.0 (Reference)  
**Date:** December 6, 2025  
**Author:** Jonathan E. Slama  
**Status:** ✅ VALIDATED

---

## Table of Contents

1. [Core Hypotheses](#1-core-hypotheses)
2. [Primary Tests](#2-primary-tests)
3. [Robustness Tests](#3-robustness-tests)
4. [Alternative Elimination Tests](#4-alternative-elimination-tests)
5. [Amplitude Investigation](#5-amplitude-investigation)
6. [Statistical Framework](#6-statistical-framework)
7. [Data Requirements](#7-data-requirements)

---

## 1. Core Hypotheses

### H1: U-Shape Universality
The BTFR residuals show a quadratic (U-shaped) dependence on environment across all datasets.

**Mathematical form:**
```
residual = a × env² + b × env + c, with a > 0
```

**Success criterion:** a > 0 with p < 0.05 on at least two independent datasets.

### H2: Redshift Evolution (Blocked)
λ_QR evolves with redshift according to KKLT prediction.

**Mathematical form:**
```
λ_QR(z) = λ_QR(0) × [1 + α × ln(1+z)]
```

**Status:** Cannot be tested - WALLABY lacks independent M_star.

### H3: Killer Prediction
Q-dominated and R-dominated galaxies show opposite U-shape curvature signs.

**Mathematical form:**
```
Q-dominated (f_gas ≥ 0.5): a > 0
R-dominated (f_gas < 0.05, log M★ > 10.5): a < 0
```

**Success criterion:** Sign inversion observed.

### H4: Q-R Independence
Q (gas fraction) and R (log stellar mass) are statistically independent variables.

**Mathematical form:**
```
Corr(Q, R) < 0 (physically: more gas → less stars)
|Corr(Q, R)| < 0.95 (statistically: not degenerate)
```

**Success criterion:** Negative correlation between -0.9 and -0.6.

---

## 2. Primary Tests

### Test 2.1: U-Shape Detection

**Script:** `tests/test_ushape_detection.py`

**Protocol:**
1. Load dataset (SPARC, ALFALFA)
2. Normalize environment variable to zero mean, unit variance
3. Bin residuals by environment (12 bins, equal count)
4. Fit quadratic: `residual = a×env² + b×env + c`
5. Compute significance: t = a / σ_a, p = 2×(1 - CDF(|t|))

**Outputs:**
- Coefficient a ± σ_a
- p-value
- Diagnostic plot (scatter + binned + fit)

**Acceptance:** a > 0, p < 0.05

---

### Test 2.2: QO+R Model Fit

**Script:** `tests/test_qor_model.py`

**Protocol:**
1. Define normalized variables:
   - Q_norm = (f_gas - mean) / std
   - R_norm = (log_Mstar - mean) / std
   - env² = standardized environment squared

2. Fit regression:
   ```
   residual = α + β_Q×(Q×env²) + β_R×(R×env²) + λ_QR×(Q²×R²)
   ```

3. Use heteroskedasticity-robust standard errors (HC3)

**Outputs:**
- Coefficients: β_Q, β_R, λ_QR with standard errors
- R² and adjusted R²
- Residual diagnostics

**Acceptance:** All coefficients significant at p < 0.05

---

### Test 2.3: Killer Prediction

**Script:** `tests/test_killer_prediction.py`

**Protocol:**
1. Stratify by gas fraction:
   - Q-dominated: f_gas ≥ 0.5
   - Intermediate: 0.3 ≤ f_gas < 0.5
   - R-dominated (loose): f_gas < 0.3
   - EXTREME R-dom: f_gas < 0.05 AND log(M★) > 10.5

2. For each stratum, fit U-shape and record coefficient a

3. Check sign inversion between Q-dom and EXTREME R-dom

**Outputs:**
- Table of (stratum, N, mean_fgas, a, σ_a, p)
- Sign comparison

**Acceptance:** a_Q > 0 AND a_extreme_R < 0

---

### Test 2.4: Q-R Independence

**Script:** `tests/test_qr_independence.py`

**Protocol:**
1. Compute Pearson correlation between Q (f_gas) and R (log_Mstar)
2. Bootstrap 1000× for confidence interval
3. Test for non-degeneracy: |r| < 0.95

**Outputs:**
- Correlation coefficient r
- 95% CI
- Scatter plot Q vs R

**Acceptance:** -0.9 < r < -0.6

---

## 3. Robustness Tests

### Test 3.1: Bootstrap Stability

**Script:** `tests/test_bootstrap_stability.py`

**Protocol:**
1. Resample dataset with replacement (N_boot = 1000)
2. For each resample, compute U-shape coefficient a
3. Report distribution of a values

**Outputs:**
- Histogram of bootstrap a values
- 68% and 95% confidence intervals
- Fraction of resamples with a > 0

**Acceptance:** >95% of resamples show a > 0

---

### Test 3.2: Spatial Jackknife

**Script:** `tests/test_spatial_jackknife.py`

**Protocol:**
1. Divide sky into N_regions (e.g., 8 RA bins)
2. For each region, remove it and recompute U-shape
3. Check if any single region dominates the signal

**Outputs:**
- Table of (region_removed, a, p)
- Flag if any removal causes sign flip

**Acceptance:** No single region removal causes a < 0

---

### Test 3.3: Selection Threshold Sensitivity

**Script:** `tests/test_selection_sensitivity.py`

**Protocol:**
1. Vary quality cuts:
   - Distance: [5, 10, 15] to [150, 200, 250] Mpc
   - S/N: [5, 6.5, 8]
   - Mass: log(M★) > [8, 9, 10]

2. For each combination, recompute U-shape

**Outputs:**
- Matrix of a values for each cut combination
- Stability assessment

**Acceptance:** a > 0 for >80% of reasonable cut combinations

---

### Test 3.4: Cross-Validation

**Script:** `tests/test_cross_validation.py`

**Protocol:**
1. Randomly split dataset 50/50 (stratified by environment)
2. Compute U-shape on each half independently
3. Compare coefficients

**Outputs:**
- a_half1 vs a_half2
- Consistency test (overlap of CIs)

**Acceptance:** Both halves show a > 0, CIs overlap

---

## 4. Alternative Elimination Tests

### Test 4.1: Selection Bias Simulation

**Script:** `tests/test_selection_bias.py`

**Objective:** Verify that the U-shape is NOT an artifact of sample selection.

**Protocol:**
1. Generate mock catalog with NO intrinsic U-shape
   - Flat residuals + Gaussian noise
   - Apply same selection function as real data

2. Check if selection creates spurious U-shape

**Outputs:**
- Mock U-shape coefficient a_mock
- Comparison with real a_observed

**Acceptance:** |a_mock| << |a_observed|

---

### Test 4.2: MOND Prediction Comparison

**Script:** `tests/test_mond_comparison.py`

**Objective:** Check if MOND predicts the same pattern.

**Protocol:**
1. Compute MOND-predicted velocities: V_MOND = f(M_bar, a_0)
2. Compute MOND residuals
3. Check if MOND residuals show U-shape

**Theoretical expectation:**
- MOND predicts residuals should be FLAT (no U-shape)
- If U-shape persists after MOND correction → not explained by MOND

**Outputs:**
- MOND residual U-shape coefficient
- Comparison plot: BTFR vs MOND residuals

**Acceptance:** MOND does not remove the U-shape

---

### Test 4.3: Baryonic Physics Control

**Script:** `tests/test_baryonic_control.py`

**Objective:** Check if known baryonic physics explains the pattern.

**Protocol:**
1. Add control variables to regression:
   - Surface brightness
   - Morphological type (if available)
   - Star formation rate (if available)
   - Color (if available)

2. Check if U-shape persists after controlling for these

**Outputs:**
- U-shape coefficient with/without controls
- Significance of control variables

**Acceptance:** U-shape remains significant after controls

---

### Test 4.4: TNG Comparison (ΛCDM-only)

**Script:** `tests/test_tng_comparison.py`

**Objective:** Compare real observations with ΛCDM simulation.

**Protocol:**
1. Load TNG100/TNG300 results
2. Compare:
   - U-shape amplitude (a_obs vs a_TNG)
   - Coefficient ratios
   - Killer prediction strength

**Theoretical expectation:**
- If reality = ΛCDM only → should match TNG
- If reality = ΛCDM + beyond → should differ

**Outputs:**
- Comparison table
- Ratio plot

**Acceptance:** Systematic differences observed

---

## 5. Amplitude Investigation

### Test 5.1: Environment Proxy Quality

**Script:** `tests/test_environment_proxy.py`

**Objective:** Assess if weak signal is due to poor environment proxy.

**Protocol:**
1. Current proxy: Supergalactic coordinates + distance
2. Alternative proxies:
   - Nth nearest neighbor density (N=5, 10, 20)
   - Local galaxy counts within R Mpc
   - Tidal field estimator (if available)

3. Compare U-shape amplitude with each proxy

**Outputs:**
- Table of (proxy_type, a, σ_a, R²)
- Best proxy identification

**Expected:** Better proxy → larger amplitude

---

### Test 5.2: Measurement Error Deconvolution

**Script:** `tests/test_error_deconvolution.py`

**Objective:** Estimate intrinsic amplitude corrected for measurement errors.

**Protocol:**
1. Estimate measurement error contribution to scatter
2. Deconvolve using hierarchical Bayesian model
3. Report intrinsic vs observed amplitude

**Outputs:**
- σ_intrinsic vs σ_observed
- Corrected U-shape amplitude

**Expected:** Intrinsic amplitude larger than observed

---

### Test 5.3: Sample Purity Effect

**Script:** `tests/test_sample_purity.py`

**Objective:** Check if mixed populations dilute the signal.

**Protocol:**
1. Stratify by galaxy type (if morphology available)
2. Compute U-shape separately for:
   - Pure spirals
   - Pure ellipticals
   - Mixed/irregular

3. Compare amplitudes

**Outputs:**
- Amplitude by morphological type
- Purity effect assessment

**Expected:** Purer samples show stronger signal

---

### Test 5.4: Theoretical Suppression Calculation

**Script:** `tests/test_theoretical_suppression.py`

**Objective:** Calculate expected suppression from string theory parameters.

**Protocol:**
1. From Paper 3, calculate:
   ```
   Suppression = (E_galactic / M_moduli)²
   ```
   where M_moduli ~ 10^10 - 10^16 GeV

2. Compare with observed amplitude ratio

**Outputs:**
- Theoretical suppression range
- Observed ratio (a_obs / a_TNG)
- Consistency check

**Expected:** If consistent → supports string theory interpretation

---

## 6. Statistical Framework

### 6.1 Significance Thresholds

| Level | p-value | Interpretation |
|-------|---------|----------------|
| Strong | < 0.01 | Highly significant |
| Moderate | < 0.05 | Significant |
| Weak | < 0.10 | Marginal |
| None | ≥ 0.10 | Not significant |

### 6.2 Multiple Testing Correction

When running multiple tests, apply Benjamini-Hochberg FDR correction:
```python
from statsmodels.stats.multitest import multipletests
reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
```

### 6.3 Effect Size Reporting

Always report:
- Point estimate ± standard error
- 95% confidence interval
- Cohen's d for comparisons

### 6.4 Model Comparison

Use information criteria:
- AIC for prediction
- BIC for model selection
- Likelihood ratio test for nested models

---

## 7. Data Requirements

### 7.1 Required Columns

| Column | Description | Source |
|--------|-------------|--------|
| Galaxy | Identifier | All |
| RA, Dec | Coordinates | All |
| Distance | Mpc | All |
| log_MHI | log10(M_HI/M_sun) | Radio 21cm |
| log_Mstar | log10(M_star/M_sun) | Photometry (independent!) |
| log_Vflat | log10(V_flat/km/s) | Rotation curve or W50/2 |
| f_gas | Gas fraction | Derived |
| density_proxy | Environment | Estimated |
| btfr_residual | BTFR residual | Derived |

### 7.2 Quality Flags

- HIcode = 1 (reliable HI detection)
- S/N > 6.5
- 5 < Distance < 200 Mpc
- Valid M_star (not from scaling relation!)

### 7.3 Data Provenance

**CRITICAL:** Q and R must come from independent measurements.

| Variable | SPARC | ALFALFA | WALLABY |
|----------|-------|---------|---------|
| M_HI | Literature | α.100 | PDR2 |
| M_star | Spitzer 3.6μm | SDSS (Durbala 2020) | ❌ None |
| V_flat | Rotation curve | W50/2 | W50/2 |

---

## 8. Test Execution Order

### Phase 1: Core Validation (Required)
1. test_qr_independence.py → Verify data quality
2. test_ushape_detection.py → Main result (H1)
3. test_qor_model.py → Full model fit
4. test_killer_prediction.py → H3

### Phase 2: Robustness (Required)
5. test_bootstrap_stability.py
6. test_spatial_jackknife.py
7. test_cross_validation.py
8. test_selection_sensitivity.py

### Phase 3: Alternative Elimination (Recommended)
9. test_selection_bias.py
10. test_mond_comparison.py
11. test_baryonic_control.py
12. test_tng_comparison.py

### Phase 4: Amplitude Investigation (Optional)
13. test_environment_proxy.py
14. test_error_deconvolution.py
15. test_sample_purity.py
16. test_theoretical_suppression.py

---

## 9. Output Structure

```
experimental/04_RESULTS/
├── REFERENCE_VERSION.md
├── FINAL_RESULTS_ANALYSIS.md
├── test_results/
│   ├── core/
│   │   ├── ushape_detection.json
│   │   ├── qor_model.json
│   │   ├── killer_prediction.json
│   │   └── qr_independence.json
│   ├── robustness/
│   │   ├── bootstrap_stability.json
│   │   ├── spatial_jackknife.json
│   │   ├── cross_validation.json
│   │   └── selection_sensitivity.json
│   ├── alternatives/
│   │   ├── selection_bias.json
│   │   ├── mond_comparison.json
│   │   ├── baryonic_control.json
│   │   └── tng_comparison.json
│   └── amplitude/
│       ├── environment_proxy.json
│       ├── error_deconvolution.json
│       ├── sample_purity.json
│       └── theoretical_suppression.json
└── figures/
    └── (diagnostic plots)
```

---

*Document updated: December 6, 2025*
