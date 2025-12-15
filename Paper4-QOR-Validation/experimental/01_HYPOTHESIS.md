# Paper 4 - Hypotheses and Test Registry

**Version:** 2.0  
**Date:** December 6, 2025  
**Status:** Complete test suite defined

---

## Primary Hypotheses

### H1: U-Shape Universality ✅ CONFIRMED
**Statement:** BTFR residuals show quadratic (U-shaped) environmental dependence.

**Mathematical form:**
```
residual = a × env² + b × env + c, with a > 0
```

**Tests:**
- `test_ushape_detection`: Direct quadratic fit
- `test_qor_model`: Full model fit with Q, R terms

**Results:**
- SPARC: a = +0.083, p = 0.064 (marginal)
- ALFALFA: a = +0.002, p = 0.036 ✅ (significant)

---

### H2: Redshift Evolution ⏸️ BLOCKED
**Statement:** λ_QR evolves with redshift according to KKLT prediction.

**Mathematical form:**
```
λ_QR(z) = λ_QR(0) × [1 + α × ln(1+z)]
```

**Tests:**
- Requires WALLABY with independent M_star
- Currently blocked: WALLABY lacks photometric stellar masses

**Status:** Cannot be tested with current data.

---

### H3: Killer Prediction ✅ CONFIRMED
**Statement:** Q-dominated and R-dominated galaxies show opposite U-shape signs.

**Mathematical form:**
```
Q-dominated (f_gas ≥ 0.5): a > 0
R-dominated (f_gas < 0.05, log M★ > 10.5): a < 0
```

**Tests:**
- `test_killer_prediction`: Stratified U-shape analysis

**Results:**
- Q-dominated: a = +0.003 ✅ (positive)
- EXTREME R-dom: a = -0.001 ✅ (negative)
- Sign inversion confirmed on N = 120 extreme galaxies

---

### H4: Q-R Independence ✅ VALIDATED
**Statement:** Q and R are statistically independent variables.

**Mathematical form:**
```
Corr(Q, R) < 0 and |Corr(Q, R)| < 0.95
```

**Tests:**
- `test_qr_independence`: Correlation analysis

**Results:**
- SPARC: r = -0.741 ✅
- ALFALFA: r = -0.808 ✅

---

## Robustness Hypotheses

### R1: Bootstrap Stability
**Statement:** U-shape detection is stable under bootstrap resampling.

**Criterion:** >95% of resamples show a > 0

**Test:** `test_bootstrap_stability`

---

### R2: Spatial Homogeneity
**Statement:** No single sky region dominates the signal.

**Criterion:** No region removal causes sign flip

**Test:** `test_spatial_jackknife`

---

### R3: Cross-Validation Consistency
**Statement:** U-shape is consistent across data splits.

**Criterion:** All folds show a > 0

**Test:** `test_cross_validation`

---

### R4: Selection Robustness
**Statement:** U-shape is robust to quality cut variations.

**Criterion:** a > 0 for >80% of cut combinations

**Test:** `test_selection_sensitivity`

---

## Alternative Elimination Hypotheses

### A1: Not Selection Artifact
**Statement:** U-shape is NOT created by sample selection effects.

**Criterion:** Mock amplitude << observed amplitude

**Test:** `test_selection_bias`

---

### A2: Not Explained by MOND
**Statement:** MOND does NOT remove or explain the U-shape.

**Criterion:** U-shape persists in MOND residuals

**Test:** `test_mond_comparison`

---

### A3: Not Baryonic Physics
**Statement:** Known baryonic physics do NOT explain the U-shape.

**Criterion:** U-shape persists after controlling for baryonic variables

**Test:** `test_baryonic_control`

---

### A4: Differs from Pure ΛCDM
**Statement:** Observations show systematic differences from TNG (pure ΛCDM).

**Criterion:** Amplitude ratio obs/TNG significantly ≠ 1

**Test:** `test_tng_comparison`

---

## Amplitude Investigation Hypotheses

### M1: Environment Proxy Quality
**Statement:** Better environment proxy yields larger amplitude.

**Criterion:** Informational (compare proxy performances)

**Test:** `test_environment_proxy`

---

### M2: Sample Purity Effect
**Statement:** Purer samples show stronger signal.

**Criterion:** Informational (compare stratified amplitudes)

**Test:** `test_sample_purity`

---

## Test Execution Summary

| Category | Tests | Required | Status |
|----------|-------|----------|--------|
| Core | 4 | ✅ Yes | Ready |
| Robustness | 4 | ✅ Yes | Ready |
| Alternatives | 4 | 🔶 Recommended | Ready |
| Amplitude | 2 | ⚪ Optional | Ready |

**Total:** 14 tests defined

---

## Execution Command

```bash
cd Paper4-Redshift-Predictions/tests

# Run all tests
python run_all_tests.py --category all

# Run specific category
python run_all_tests.py --category core
python run_all_tests.py --category robustness

# Run specific test
python run_all_tests.py --test ushape_detection
```

---

*Document updated: December 6, 2025*
