# QO+R TOE Validation - Honest Assessment of Discrepant Results

**Document Type:** Critical Analysis  
**Author:** Jonathan Édouard Slama  
**Institution:** Metafund Research Division, Strasbourg, France  
**Date:** December 14, 2025

---

## 1. Summary of Results

### 1.1 SPARC Results (N = 181)

| Test | Result | Status |
|------|--------|--------|
| λ_QR detection | 0.943 ± 0.228 (4.14σ) | ✅ PASS |
| Universality | ratio 1.16 | ✅ PASS |
| Q-R anticorrelation | r = -0.82 | ✅ PASS |
| BTFR slope | 0.344 | ❌ FAIL |
| String theory range | λ = 0.94 ∈ (0.1, 100) | ✅ PASS |
| Model comparison | ΔAIC = +14.7 | ✅ PASS |
| **Overall** | **5/6 passed** | **SUPPORTED** |

### 1.2 ALFALFA Results (N = 19,210)

| Test | Result | Status |
|------|--------|--------|
| λ_QR detection | -0.028 ± 0.033 (0.86σ) | ❌ FAIL |
| Universality | Mass-dependent (varies -0.79 to +0.06) | ❌ FAIL |
| Q-R anticorrelation | r = -0.77 | ✅ PASS |
| BTFR slope | 0.300 | ✅ PASS |
| String theory range | |λ| = 0.028 < 0.1 | ❌ FAIL |
| Model comparison | ΔAIC = -1.3 | ❌ FAIL |
| **Overall** | **2/6 passed** | **NOT SUPPORTED** |

---

## 2. The Problem: Contradictory Results

### 2.1 Direct Contradiction

| Parameter | SPARC | ALFALFA | Compatible? |
|-----------|-------|---------|-------------|
| λ_QR value | +0.94 | -0.03 | ❌ NO |
| λ_QR sign | Positive | Negative | ❌ NO |
| Significance | 4.1σ | 0.9σ | ❌ NO |
| ΔAIC | +14.7 (QO+R wins) | -1.3 (Linear wins) | ❌ NO |

### 2.2 Previous ALFALFA Analysis

In `paper4_results_v3.json`, a previous analysis of the same ALFALFA data showed:
- λ_QR = -0.00129 ± 0.00042
- p_lambda = 0.0023 (**significant at ~3σ**)

But our rigorous test with identical SPARC criteria shows:
- λ_QR = -0.028 ± 0.033
- Significance = 0.86σ (**not significant**)

**The results depend on methodology.**

---

## 3. Possible Explanations

### 3.1 QO+R is Wrong (Most Parsimonious)

The simplest explanation: **QO+R is not supported by the larger ALFALFA dataset**.

- SPARC (181 galaxies) may show a spurious signal
- With 100× more data, the signal disappears
- This is a common pattern for false discoveries

### 3.2 Methodological Differences

The two analyses may define variables differently:

| Variable | Rigorous Test | Previous Analysis |
|----------|---------------|-------------------|
| Q | log(M_HI / M_bar) | May differ |
| R | log(M_star / M_bar) | May differ |
| Residual | Recalculated from V, M | Pre-calculated in file |

If results depend on variable definitions, this is a **robustness problem**.

### 3.3 Different Physics in Different Samples

Possible but less likely:
- SPARC: Resolved rotation curves (V_flat)
- ALFALFA: Integrated line widths (W50/2)
- Different selection functions

But if QO+R is real, it should appear in BOTH samples.

### 3.4 Mass Dependence (Not Universal)

ALFALFA Test 2 shows clear mass dependence:

| Mass Bin | λ_QR | N |
|----------|------|---|
| Low (7-9) | -0.79 | 1637 |
| Intermediate (9-10) | -0.17 | 8881 |
| High (10-12) | +0.06 | 8692 |

This violates the **universality** prediction of QO+R. The coupling constant should NOT depend on mass.

---

## 4. Honest Assessment

### 4.1 What We Can Conclude

1. **SPARC shows a QO+R signal at 4.1σ** - This is real in that dataset
2. **ALFALFA does NOT show the same signal** - With same criteria
3. **The results are inconsistent** - This is a problem for QO+R
4. **λ_QR appears mass-dependent** - Violates universality

### 4.2 What We CANNOT Conclude

1. We cannot claim QO+R is definitively falsified (SPARC still shows signal)
2. We cannot claim QO+R is confirmed (ALFALFA contradicts it)
3. We cannot adjust methodology to "make it work" (that's p-hacking)

### 4.3 Scientific Status

**QO+R TOE is in a problematic state:**

- ✅ Supported by SPARC (181 galaxies)
- ❌ Not supported by ALFALFA (19,210 galaxies)
- ⚠️ Results depend on methodology
- ⚠️ λ_QR appears non-universal

---

## 5. What Should We Do?

### 5.1 Option A: Acknowledge the Contradiction

Document both results honestly:
- SPARC supports QO+R
- ALFALFA does not support QO+R
- Further investigation needed

**This is the honest approach.**

### 5.2 Option B: Investigate Methodological Differences

Understand WHY the previous ALFALFA analysis showed significance:
- What variable definitions were used?
- How were residuals calculated?
- Are there data quality cuts we're missing?

**This is legitimate scientific investigation** (not p-hacking) if done transparently.

### 5.3 Option C: Abandon QO+R

If larger dataset contradicts smaller dataset, the larger dataset usually wins.

**This would be premature** without understanding the methodological differences.

---

## 6. Recommended Path Forward

### 6.1 Immediate Actions

1. **Document the discrepancy** - Done in this file
2. **Do NOT adjust criteria post-hoc** - That's p-hacking
3. **Investigate methodology differences** - Legitimate science
4. **Be transparent** - Include negative results in any publication

### 6.2 Investigation Questions

1. Why does previous ALFALFA analysis show significance?
2. What is the correct definition of Q, R for the QO+R model?
3. Is the mass dependence real physics or artifact?
4. Are SPARC and ALFALFA measuring the same thing?

### 6.3 Publication Implications

Any paper on QO+R must:
- Present BOTH SPARC (positive) and ALFALFA (negative) results
- Discuss the discrepancy honestly
- NOT cherry-pick the favorable dataset

---

## 7. Conclusions

### 7.1 Current State of Evidence

| Evidence | Direction | Weight |
|----------|-----------|--------|
| SPARC λ_QR = 0.94 (4.1σ) | Supports QO+R | Medium (N=181) |
| ALFALFA λ_QR = -0.03 (0.9σ) | Against QO+R | High (N=19,210) |
| Previous ALFALFA (3σ) | Supports QO+R | Unclear (methodology?) |
| Mass dependence of λ | Against universality | High |

### 7.2 Honest Verdict

**QO+R TOE is NOT robustly supported by the available data.**

The positive SPARC result does not replicate in the larger ALFALFA sample with identical methodology. This is a serious problem for the theory.

### 7.3 What This Means

This does NOT mean QO+R is definitively wrong. But it DOES mean:
- More work is needed to understand the discrepancy
- Claims of TOE validation are premature
- The theory needs to explain why results differ between datasets

---

## Appendix: Methodological Details

### A1. Rigorous Test Definition of Variables

```python
Q = log10(M_HI / M_bar)      # Gas fraction in log
R = log10(M_star / M_bar)    # Stellar fraction in log
residual = log_V - log_V_btfr  # BTFR residual
```

### A2. Pre-Registered Criteria

All criteria defined before analysis:
1. λ_QR detection: |λ/σ| > 3.0
2. Universality: ratio < 3.0
3. Q-R anticorrelation: r < -0.5
4. BTFR slope: 0.20 ≤ slope ≤ 0.30
5. String theory λ: 0.1 < |λ| < 100
6. Model comparison: ΔAIC > 2.0

### A3. Data Sources

- SPARC: Lelli et al. (2016)
- ALFALFA: Haynes et al. (2018) + Durbala et al. (2020)

---

**Document Version:** 1.0  
**Last Updated:** December 14, 2025  
**Status:** CRITICAL ANALYSIS - DISCREPANT RESULTS
