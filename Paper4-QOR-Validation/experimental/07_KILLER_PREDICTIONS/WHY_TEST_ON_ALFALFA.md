# Why Test QO+R TOE on ALFALFA?

**Document Type:** Scientific Justification  
**Author:** Jonathan Édouard Slama  
**Institution:** Metafund Research Division, Strasbourg, France  
**Date:** December 2025

---

## 1. The Scientific Question

**Question:** Does the QO+R Theory of Everything make correct predictions about galaxy dynamics?

**Specific prediction:** The BTFR residuals should depend on a Q×R interaction term:

```
Δlog(V) = β_Q × Q + β_R × R + λ_QR × Q × R
```

Where:
- Q = log(f_gas) = gas fraction proxy
- R = log(1 - f_gas) = stellar fraction proxy  
- λ_QR = universal coupling constant (~1 from string theory)

---

## 2. What Data Do We Need?

To test this prediction, we need galaxies with:

| Requirement | Why |
|-------------|-----|
| **HI mass (M_HI)** | To compute Q (gas fraction) |
| **Stellar mass (M_star)** | To compute R (stellar fraction) |
| **Rotation velocity (V)** | To measure BTFR residuals |
| **Large sample** | Statistical power to detect λ_QR |
| **Independent measurements** | Avoid artificial correlations |

---

## 3. Why ALFALFA Satisfies These Requirements

### 3.1 Provides HI Masses

ALFALFA is the **largest blind HI survey ever conducted**:
- 31,502 sources
- Direct measurement of neutral hydrogen via 21cm emission
- No selection bias based on optical properties

### 3.2 Independent Stellar Masses Available

The Durbala et al. (2020) cross-match provides:
- Stellar masses from SDSS photometry
- **Completely independent** of HI measurements
- ~19,000 galaxies with both M_HI and M_star

### 3.3 Velocity Proxy Available

W50 (line width at 50%) provides rotation velocity proxy:
- V ≈ W50 / 2
- Not as precise as resolved rotation curves (SPARC)
- But sufficient for statistical analysis with large N

### 3.4 Large Sample Size

| Dataset | N galaxies | Statistical power |
|---------|------------|-------------------|
| SPARC | 181 | Detects 4σ signal |
| ALFALFA | ~19,000 | **100× more power** |

With 100× more galaxies, we can:
- Detect smaller effects
- Test universality across mass bins
- Reduce random errors

---

## 4. Scientific Advantages of ALFALFA

### 4.1 HI-Selected = Gas-Rich Galaxies

ALFALFA preferentially selects **gas-rich galaxies** (high Q):
- These are "Q-dominated" systems in QO+R framework
- Ideal for testing Q×R interaction term
- Complements optical surveys (which favor R-dominated)

### 4.2 Uniform Selection Function

ALFALFA is **flux-limited** with well-understood selection:
- S/N > 6.5 threshold
- Known completeness as function of mass and distance
- Can correct for selection effects if needed

### 4.3 Local Universe

ALFALFA covers z < 0.06:
- Minimal cosmological corrections
- Well-calibrated distances (Hubble flow)
- Direct comparison with local physics

---

## 5. What ALFALFA Tests vs What SPARC Tests

### 5.1 SPARC Strengths

- **Precise rotation curves** → accurate V_flat
- **Resolved kinematics** → can test radial dependence
- **High-quality sample** → minimal systematics

### 5.2 ALFALFA Strengths

- **Large N** → statistical power
- **Blind HI selection** → unbiased gas-rich sample
- **Independent M_star** → avoids Q-R artifacts

### 5.3 Complementarity

| Test | SPARC | ALFALFA |
|------|-------|---------|
| λ_QR detection | 4.1σ | ? (expect higher) |
| Universality | 2 mass bins | 3+ mass bins possible |
| Q-R correlation | -0.82 | Should be similar |
| Sample bias | Rotation curve quality | HI flux limit |

**Both tests are needed** - if QO+R is real, it should appear in BOTH datasets despite different selection effects.

---

## 6. Potential Issues and Mitigations

### 6.1 W50 vs V_flat

**Issue:** W50 is not the same as V_flat
- Includes turbulence
- Inclination-dependent
- Less precise

**Mitigation:** 
- Large N compensates for increased scatter
- W50 is still correlated with V_flat
- Any QO+R signal should still be detectable

### 6.2 SDSS Stellar Masses

**Issue:** SDSS M_star has ~0.2 dex uncertainty
- IMF assumptions
- Dust corrections

**Mitigation:**
- Systematic errors affect mean, not scatter
- Durbala 2020 provides multiple M_star estimates
- Can check consistency

### 6.3 Selection Bias

**Issue:** ALFALFA under-represents gas-poor galaxies

**Mitigation:**
- This is actually an advantage for testing Q-dominated systems
- Combine with optical surveys (KiDS) for full picture
- Sign inversion test uses both ALFALFA (Q-dominated) and KiDS (R-dominated)

---

## 7. Pre-Registration of Tests

Before running ALFALFA analysis, we define **identical criteria** as SPARC:

| Test | Criterion | Rationale |
|------|-----------|-----------|
| λ_QR detection | \|λ/σ\| > 3.0 | Standard discovery threshold |
| Universality | ratio < 3.0 | Universal constant check |
| Q-R correlation | r < -0.5 | Mass conservation |
| BTFR slope | [0.20, 0.30] | MOND limit |
| String theory λ | (0.1, 100) | KKLT prediction |
| Model comparison | ΔAIC > 2 | Standard criterion |

**No post-hoc adjustments** - same criteria, different dataset.

---

## 8. Expected Outcomes

### 8.1 If QO+R is Correct

- λ_QR should be detected at high significance (>>3σ with N=19,000)
- λ_QR value should be consistent with SPARC (~0.94)
- Universality should hold across mass bins
- ΔAIC should strongly favor QO+R model

### 8.2 If QO+R is Wrong

- λ_QR not significant, or
- λ_QR inconsistent with SPARC, or
- Strong mass dependence (non-universal)

### 8.3 Falsification Criterion

QO+R is **falsified** if:
- ALFALFA shows λ_QR consistent with zero at 3σ
- OR λ_QR has opposite sign from SPARC
- OR λ_QR varies by >10× across mass bins

---

## 9. Conclusion

ALFALFA is the **ideal dataset** for QO+R TOE validation because:

1. ✅ Provides both M_HI and M_star (via Durbala cross-match)
2. ✅ Large sample (~19,000) for high statistical power
3. ✅ Independent measurements avoid Q-R artifacts
4. ✅ HI-selected samples complement optical surveys
5. ✅ Well-documented systematics and selection function
6. ✅ 100% real observational data

If QO+R passes on both SPARC (N=181) and ALFALFA (N=19,000), this provides **strong evidence** for the theory across different selection functions and measurement methods.

---

**Document Version:** 1.0  
**Last Updated:** December 14, 2025
