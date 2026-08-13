# Research Log: Paper 2 — Residual Diagnostics

**Project:** The Revelatory Division  
**Principal Investigator:** Jonathan Édouard Slama  
**Start Date:** December 2, 2025

---

## December 2, 2025 - Project Initialization

**Objective:** Establish project structure and document theoretical framework

**Method:** 
- Created directory structure for reproducible research
- Documented core hypothesis in HYPOTHESIS.md
- Identified primary dataset: Breast Cancer Coimbra (UCI)

**Results:**
- Project structure created
- Theoretical framework documented
- Dataset identified: 116 patients, 9 biomarkers

**Files Created:**
- `README.md`, `HYPOTHESIS.md`, `RESEARCH_LOG.md`

---

## December 2, 2025 - Data Download and Exploration

**Objective:** Download data and perform initial exploratory analysis

**Method:**
- Downloaded Breast Cancer Coimbra dataset from UCI (ID: 451)
- Computed distribution statistics for all variables
- Compared groups (Cancer vs. Healthy)
- Generated correlation matrix

**Results:**

Dataset characteristics:
- N = 116 (64 cancer, 52 healthy)
- 9 biomarkers: Age, BMI, Glucose, Insulin, HOMA, Leptin, Adiponectin, Resistin, MCP.1
- No missing values

Key group differences (p < 0.05):
| Variable | Healthy | Cancer | Change | p-value |
|----------|---------|--------|--------|---------|
| Glucose | 88.2 | 105.6 | +19.6% | 0.0000 |
| HOMA | 1.55 | 3.62 | +133% | 0.0029 |
| Insulin | 6.93 | 12.51 | +80.5% | 0.0266 |
| Resistin | 11.61 | 17.25 | +48.5% | 0.0019 |

Correlations with Cancer:
- Glucose: r = +0.384 (strongest)
- HOMA: r = +0.284
- Insulin: r = +0.277
- Resistin: r = +0.227

**Files Created:**
- `data/raw/breast_cancer_coimbra_full.csv`
- `data/processed/distribution_analysis.csv`
- `data/processed/group_comparison.csv`
- `figures/fig01-04.png`

---

## December 2, 2025 - Ratio Computation

**Objective:** Compute clinically relevant biomarker ratios (Quotients) and residuals (Reliquats)

**Method:**
- Computed 6 biomarker ratios
- Fitted linear models: Ratio ~ Age + BMI
- Extracted residuals from each model

**Results:**

Computed ratios:
| Ratio | Mean ± SD | Group Difference |
|-------|-----------|------------------|
| HOMA_IR | 2.70 ± 3.65 | p = 0.0029 ** |
| LAR (Leptin/Adiponectin) | 3.97 ± 4.45 | p = 0.887 |
| Resistin/Adiponectin | 2.34 ± 3.40 | p = 0.036 * |
| MCP1/Adiponectin | 78.9 ± 72.5 | p = 0.720 |
| Leptin/BMI | 0.92 ± 0.59 | p = 0.691 |
| Glucose/Insulin | 15.8 ± 8.52 | p = 0.301 |

Model R² values (Ratio ~ Age + BMI):
- LAR: R² = 0.211 (highest - BMI explains LAR variation)
- Leptin_BMI: R² = 0.167
- Glucose_Insulin: R² = 0.103
- Others: R² < 0.08 (Age/BMI explain little variance)

**Interpretation:**
Most ratio variance is NOT explained by Age and BMI, meaning residuals contain substantial information beyond these confounders.

**Files Created:**
- `data/processed/data_with_ratios_and_residuals.csv`
- `data/processed/computed_ratios.csv`
- `figures/fig05-06.png`

---

## December 2, 2025 - Core Residual Analysis (KEY RESULTS)

**Objective:** Test the Revelatory Division hypothesis

### TEST 1: Residual Distribution Differences

**Method:** Compare residual distributions between Cancer and Healthy groups using KS test, Mann-Whitney U, and Levene's test.

**Results:**

| Residual | Cohen's d | KS p-value | MW p-value | Levene p |
|----------|-----------|------------|------------|----------|
| HOMA_IR | **+0.677** | 0.0013 * | 0.0001 * | 0.0067 * |
| Resistin_Adipo | +0.388 | 0.0018 * | 0.0038 * | 0.091 |
| Glucose_Insulin | -0.217 | 0.0096 * | 0.123 | 0.0005 * |
| MCP1_Adipo | +0.221 | 0.312 | 0.373 | 0.308 |
| Leptin_BMI | +0.216 | 0.107 | 0.247 | 0.582 |
| LAR | +0.109 | 0.836 | 0.500 | 0.916 |

**Key Finding:** 
- 3/6 residuals show significantly different distributions
- 2/6 show significantly different locations (means)
- HOMA_IR residuals show **medium-to-large effect size** (d = 0.677)

→ **HYPOTHESIS SUPPORTED:** Residuals contain group-discriminating information

---

### TEST 2: U-Shape Detection

**Method:** Fit quadratic models (Residual ~ Covariate + Covariate²) and test significance of quadratic coefficient.

**Results:**

| Residual vs Covariate | Quad. Coef (a) | Z-score | p-value | Shape |
|-----------------------|----------------|---------|---------|-------|
| HOMA_IR vs Glucose | +0.351 | 4.68 | <0.0001 | **U-shape** |
| HOMA_IR vs Insulin | +0.306 | 7.78 | <0.0001 | **U-shape** |
| MCP1_Adipo vs Glucose | +13.06 | 7.32 | <0.0001 | **U-shape** |
| Glucose_Insulin vs Insulin | +2.04 | 13.99 | <0.0001 | **U-shape** |
| Leptin_BMI vs Age | +0.111 | 2.24 | 0.025 | U-shape |
| LAR vs Insulin | -0.289 | -2.40 | 0.016 | Inverted-U |
| Leptin_BMI vs Insulin | -0.035 | -2.16 | 0.031 | Inverted-U |

**Summary:** 7/24 combinations show significant non-linear patterns

**After FDR correction:** 4 patterns survive (all with p < 0.0001)

→ **HYPOTHESIS SUPPORTED:** Non-linear (U-shaped) patterns exist in residuals

---

### TEST 3: Classification Improvement

**Method:** Compare logistic regression AUC using:
1. Ratios only
2. Residuals only
3. Ratios + Residuals

**Results:**

| Model | AUC (5-fold CV) |
|-------|-----------------|
| Ratios only | 0.765 ± 0.090 |
| Residuals only | 0.743 ± 0.107 |
| Ratios + Residuals | 0.741 ± 0.102 |

Improvement from adding residuals: -0.024 (p = 0.358, not significant)

**Interpretation:**
- Residuals alone achieve 0.743 AUC (well above chance!)
- But they don't ADD information beyond ratios
- Residuals and ratios capture overlapping information

→ **HYPOTHESIS PARTIALLY REFUTED:** Residuals are informative but redundant with ratios

---

## December 2, 2025 - Robustness Testing

**Objective:** Validate findings with bootstrap, permutation, and sensitivity analyses

### Bootstrap Confidence Intervals (1000 resamples)

| Residual | 95% CI for Group Difference | Excludes Zero? |
|----------|----------------------------|----------------|
| HOMA_IR | [+1.18, +3.37] | **Yes** |
| Resistin_Adipo | [+0.22, +2.56] | **Yes** |
| Others | Include zero | No |

### Permutation Test (100 permutations)

- True AUC: 0.741
- Permuted AUC: 0.514 ± 0.074
- p-value: 0.0099

→ Classification is **significantly better than chance**

### Multiple Testing Correction

- Uncorrected significant U-shapes: 7/24
- After Bonferroni: 4/24
- After BH-FDR: 4/24

Surviving patterns:
1. HOMA_IR vs Glucose
2. HOMA_IR vs Insulin
3. MCP1_Adiponectin vs Glucose
4. Glucose_Insulin vs Insulin

### Sensitivity to Outliers

- 6/6 findings stable after removing outliers (>3 SD)
- Conclusions unchanged

**Overall Robustness:** MODERATE

---

## Summary of Findings

### What We Found

1. **Residuals contain diagnostic information**
   - HOMA_IR residuals differ strongly between groups (d = 0.677)
   - 3/6 residuals show different distributions
   - 2/6 differences survive bootstrap validation

2. **U-shaped patterns exist in residuals**
   - 4 patterns survive FDR correction (p < 0.0001)
   - Strongest: Glucose_Insulin vs Insulin (Z = 13.99)
   - All 4 show positive quadratic coefficient (concave up)

3. **Residuals are informative but not additive**
   - Residuals alone: AUC = 0.743
   - Adding to ratios: no improvement
   - Information is redundant, not complementary

### Interpretation

The "Revelatory Division" hypothesis receives **partial support**:

✅ **Confirmed:** Residuals are not random noise  
✅ **Confirmed:** Non-linear (U-shaped) patterns exist  
✅ **Confirmed:** Residuals can discriminate cancer from healthy  
❌ **Not confirmed:** Residuals add value beyond ratios  

The residuals and ratios appear to capture the **same underlying biological signal** through different mathematical lenses. This is actually an interesting finding—it suggests that the ratio values themselves implicitly contain the residual information.

### Limitations

1. Small sample size (N = 116)
2. Single dataset (no external validation)
3. Cross-sectional (no temporal dynamics)
4. Limited biomarkers (no genetic/proteomic data)

### Next Steps

1. Write manuscript documenting findings
2. Seek replication on larger datasets (PLCO with longitudinal PSA)
3. Investigate WHY residuals and ratios are redundant
4. Consider alternative residual definitions (non-linear models)

---

## Files Generated

### Data
- `data/raw/breast_cancer_coimbra_full.csv`
- `data/processed/data_with_ratios_and_residuals.csv`

### Results
- `results/test1_residual_distributions.csv`
- `results/test2_ushape_detection.csv`
- `results/test2_ushape_corrected.csv`
- `results/test3_classification.csv`
- `results/robustness_bootstrap.csv`
- `results/robustness_sensitivity.csv`
- `results/analysis_summary.txt`
- `results/robustness_report.txt`

### Figures
- `figures/fig01_distributions_by_group.png`
- `figures/fig02_correlation_heatmap.png`
- `figures/fig03_boxplots_by_group.png`
- `figures/fig04_pairplot_key_variables.png`
- `figures/fig05_ratio_distributions.png`
- `figures/fig06_ratio_correlations.png`
- `figures/fig07_residual_distributions.png`
- `figures/fig08_residuals_vs_age.png`
- `figures/fig09_residuals_vs_bmi.png`
- `figures/fig10_permutation_test.png`

---

*Research log complete. Ready for manuscript preparation.*
