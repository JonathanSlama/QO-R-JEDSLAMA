# Next Steps: Immediate Actions

## What Can Be Done NOW (No New Data Required)

This document outlines tests and analyses that can be performed immediately using existing data.

---

## Priority 1: Improve Environment Proxy

### Current Status
- Using Supergalactic coordinates as proxy
- U-shape amplitude: a ≈ 0.003
- TNG prediction: a ≈ 0.017
- **Gap: ~5× smaller than expected**

### Hypothesis
The weak amplitude may be due to poor environment estimation, not missing physics.

### Tests to Run
1. `test_environment_proxy.py` - Compare proxy methods (DONE)
2. `test_nth_neighbor_density.py` - Systematic N-th neighbor analysis
3. `test_group_membership.py` - Use galaxy group catalogs

### Data Sources
- ALFALFA positions (already have)
- SDSS group catalogs (Yang et al. 2007, Tempel et al. 2017)
- 2MRS density field

### Expected Outcome
If proxy is the issue: amplitude should increase 2-5× with better estimator.

---

## Priority 2: Mass Scaling Test

### Theoretical Prediction
String theory predicts mass-dependent coupling:
```
a(M_star) ∝ (M_star / M_Planck)^α × f(Calabi-Yau)
```

### Test Protocol
1. Bin ALFALFA by stellar mass (4-5 bins)
2. Fit U-shape in each bin
3. Measure scaling: a = a_0 × (M_star)^α
4. Compare α with string theory prediction

### Data
Already have: ALFALFA with SDSS M_star (Durbala 2020)

### Script
`test_mass_scaling.py`

---

## Priority 3: Complete Alternative Elimination

### Alternatives to Test

| Theory | Prediction | How to Test |
|--------|-----------|-------------|
| WDM | Suppressed small-scale power | Check mass dependence |
| SIDM | Core formation in halos | Different environmental signature |
| f(R) | Scale-dependent growth | Different z-evolution |
| Fuzzy DM | Soliton cores | Mass-dependent cutoff |

### Method
1. Derive each theory's prediction for BTFR residuals
2. Compare with observed pattern
3. Document incompatibilities

### Script
`test_alternative_theories.py`

---

## Priority 4: Gravitational Lensing Pilot

### Goal
Detect same Q²R² pattern in weak lensing residuals.

### Data Sources (Public)
- DES Y3 shear catalog
- KiDS-1000 shear catalog
- HSC-SSP Y3

### Challenge
Need to define Q (gas fraction) for lens galaxies:
- Cross-match with HI surveys (limited)
- Use color as proxy for gas content
- Use morphology (spiral vs elliptical)

### Script
`test_lensing_pilot.py`

---

## Priority 5: Galaxy Cluster Test

### Goal
Detect U-shape in cluster M-T scaling relation residuals.

### Data Sources (Public)
- Planck SZ catalog (1,653 clusters)
- MCXC meta-catalog (1,743 clusters)
- eROSITA eFEDS (542 clusters)

### Method
1. Compute M-T relation: M_500 = A × T^α
2. Calculate residuals
3. Define Q (gas fraction from X-ray) and R (stellar fraction from BCG)
4. Test for environmental dependence

### Script
`test_cluster_scaling.py`

---

## File Checklist

| File | Status | Priority |
|------|--------|----------|
| `test_mass_scaling.py` | To create | High |
| `test_nth_neighbor_density.py` | To create | High |
| `test_alternative_theories.py` | To create | Medium |
| `test_lensing_pilot.py` | To create | Medium |
| `test_cluster_scaling.py` | To create | Medium |
| `test_group_membership.py` | To create | Low |

---

## Timeline

| Week | Action |
|------|--------|
| 1 | Mass scaling test |
| 2 | N-th neighbor density optimization |
| 3-4 | Alternative theories elimination |
| 5-6 | Lensing pilot preparation |
| 7-8 | Cluster test preparation |

---

*Created: December 7, 2025*
