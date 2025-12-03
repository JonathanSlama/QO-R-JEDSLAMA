# Theoretical Framework: The Revelatory Division

**Document Version:** 1.0  
**Last Updated:** December 2025  
**Author:** Jonathan Édouard Slama

---

## 1. The Philosophy of Division

### 1.1 The Standard View

In mathematics and applied sciences, division is typically viewed as an operation that extracts a ratio—a quotient that captures the essential relationship between two quantities:

```
Dividend ÷ Divisor = Quotient
```

The quotient is the prize; it reveals proportionality, normalizes for confounders, and enables comparison across scales.

### 1.2 The Overlooked Remainder

But division yields more than a quotient. In integer arithmetic:

```
Dividend ÷ Divisor = Quotient + Remainder
```

The remainder is what's "left over"—the part that doesn't fit neatly into the quotient. In real-number division, the remainder manifests as the residual: the difference between observed values and model predictions.

**Standard practice:** Treat residuals as noise. Minimize them. Ignore them.

**Our hypothesis:** Residuals contain structured information.

---

## 2. The Revelatory Division Concept

### 2.1 Two Sources of Information

We propose that any ratio-based analysis generates two complementary information streams:

| Component | What It Captures | Typical Treatment |
|-----------|------------------|-------------------|
| **Quotient (Q)** | Average/expected relationship | Analyzed, published |
| **Reliquat (R)** | Individual deviation from expected | Discarded as "noise" |

The term "Reliquat" (French for "remainder" or "residue") emphasizes that this is not mere statistical noise, but a substantive remainder worthy of investigation.

### 2.2 The Core Insight

In domains ranging from galaxy dynamics (Paper 1) to clinical diagnostics (this work), we hypothesize that:

1. The quotient captures **population-level** relationships
2. The reliquat captures **individual-level** deviations
3. Systematic patterns in the reliquat reveal **hidden variables** or **distinct subpopulations**

---

## 3. Application to Clinical Biomarkers

### 3.1 The Ratio Paradigm in Medicine

Clinical medicine relies heavily on ratios:

| Ratio | Formula | Purpose |
|-------|---------|---------|
| HOMA-IR | (Glucose × Insulin) / 405 | Insulin resistance index |
| BMI | Weight / Height² | Body composition proxy |
| NLR | Neutrophils / Lymphocytes | Inflammatory state |
| PSA Density | PSA / Prostate Volume | Cancer risk adjustment |
| eGFR | f(Creatinine, Age, Sex) | Kidney function |

These ratios are treated as **sufficient statistics**—the ratio value tells you what you need to know.

### 3.2 Our Hypothesis

**Hypothesis H1:** Residuals from ratio-based models contain diagnostic information not captured by the ratio itself.

**Hypothesis H2:** The pattern of residuals (not just their magnitude) differs systematically between health states.

**Hypothesis H3:** Residual patterns may exhibit non-linear structure (e.g., U-shapes) that reveals competing biological mechanisms.

---

## 4. Mechanistic Speculation

### 4.1 Why Might Residuals Carry Information?

Consider HOMA-IR = (Glucose × Insulin) / 405. This assumes:
- A fixed relationship between glucose and insulin
- That this relationship is the same for everyone
- That deviations are random measurement error

But biology is complex. Deviations might reflect:
- **Genetic variants** in glucose or insulin metabolism
- **Tissue-specific** insulin resistance (liver vs. muscle)
- **Circadian** or **stress-related** fluctuations
- **Early disease states** not yet manifest in the ratio

If these factors systematically differ between healthy and diseased individuals, **the pattern of residuals will differ too.**

### 4.2 The U-Shape Hypothesis

From Paper 1 (galaxy dynamics), we learned that when two opposing mechanisms operate:
- Mechanism A dominates in low-X environments
- Mechanism B dominates in high-X environments
- Both are suppressed in intermediate environments

The result is a **U-shaped** residual pattern.

In cancer biology, this could manifest as:
- **Low metabolic activity:** Tumor suppression (or dormancy)
- **Intermediate activity:** Normal homeostasis
- **High metabolic activity:** Tumor promotion

If true, residuals should show elevated values at both extremes, with a minimum in the middle range.

---

## 5. Testable Predictions

### 5.1 Primary Predictions

**P1: Non-random residuals**
- Residuals from ratio models are not normally distributed
- They show structure (bimodality, skewness, or systematic trends)

**P2: State-dependent patterns**
- Cancer patients and healthy controls show different residual distributions
- The difference is not just in mean, but in shape

**P3: Quadratic relationships**
- Residuals show U-shaped (or inverted U-shaped) dependence on covariates
- The quadratic coefficient differs between groups

### 5.2 Falsification Criteria

This hypothesis is falsified if:
- Residuals are normally distributed with no group differences
- Residual patterns are identical between cancer and control groups
- No covariate shows a significant quadratic relationship with residuals

---

## 6. Methodological Approach

### 6.1 Analysis Pipeline

```
Step 1: Compute clinical ratios
        - HOMA-IR, Leptin/Adiponectin, Resistin/Adiponectin, etc.

Step 2: Fit ratio-outcome models
        - Logistic regression: Ratio → Cancer (yes/no)
        - Extract residuals (deviance residuals, Pearson residuals)

Step 3: Analyze residual structure
        - Normality tests (Shapiro-Wilk, Anderson-Darling)
        - Compare distributions between groups (K-S test, Mann-Whitney)
        - Test for quadratic relationships (polynomial regression)

Step 4: Visualize patterns
        - Residual vs. covariate plots
        - Density plots by group
        - U-shape detection

Step 5: Validate findings
        - Cross-validation
        - Bootstrap confidence intervals
        - Multiple testing correction
```

### 6.2 Key Statistics

| Test | Purpose | Threshold |
|------|---------|-----------|
| Shapiro-Wilk | Residual normality | p < 0.05 |
| Kolmogorov-Smirnov | Group distribution difference | p < 0.05 |
| Quadratic coefficient | U-shape presence | z > 2 (p < 0.05) |
| AIC comparison | Model selection | ΔAIC > 10 |

---

## 7. Limitations and Caveats

### 7.1 Known Limitations

1. **Small sample size:** N = 116 limits statistical power
2. **Single dataset:** Findings may not generalize
3. **Cross-sectional:** Cannot establish causality or temporal precedence
4. **Biomarker selection:** Limited to available variables

### 7.2 What This Study Cannot Prove

- That residual analysis is *better* than ratio analysis (only that it may be *complementary*)
- That the mechanism we propose is correct (only that patterns exist)
- That findings will replicate in other cancers or diseases

### 7.3 Honest Uncertainty

We may be wrong. The patterns we find may be artifacts of small samples, measurement error, or overfitting. We present this work as hypothesis generation, not hypothesis confirmation.

---

## 8. Connection to QO+R Framework

This work extends the Quotient Ontologique + Reliquat (QO+R) framework developed in Papers 1 and 3, but **does not assume** the string theory connection proposed there.

The philosophical core—that remainders carry information—is domain-independent. Whether it manifests in galaxy dynamics, clinical diagnostics, or other fields is an empirical question.

| Domain | Quotient (Q) | Reliquat (R) |
|--------|--------------|--------------|
| Galaxy dynamics | BTFR ratio | Environmental residual |
| Clinical diagnostics | Biomarker ratio | Individual residual |
| [Future domains?] | [Ratio] | [Residual] |

---

## 9. References

1. Patrício, M., et al. (2018). Using Resistin, glucose, age and BMI to predict the presence of breast cancer. BMC Cancer, 18(1), 29.

2. Matthews, D.R., et al. (1985). Homeostasis model assessment: insulin resistance and β-cell function from fasting plasma glucose and insulin concentrations in man. Diabetologia, 28(7), 412-419.

3. Slama, J.É. (2025). Environmental Modulation of the Baryonic Tully-Fisher Relation: A Two-Field Scalar Approach. [Paper 1]

---

*This document represents our current theoretical framework. It will be updated as the research progresses.*
