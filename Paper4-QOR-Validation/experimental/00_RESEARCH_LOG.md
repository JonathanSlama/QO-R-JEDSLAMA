# Paper 4 - Research Log

## Chronological Record of Scientific Methodology

**Author**: Jonathan Edouard Slama  
**ORCID**: [0009-0002-1292-4350](https://orcid.org/0009-0002-1292-4350)  
**Affiliation**: Metafund Research Division, Strasbourg, France

---

### 2024-12-06 - Paper 4 Genesis

#### Context
Following validation of the QO+R framework on 708,086 galaxies (Papers 1-3), the central question became: **How to definitively distinguish between "poorly understood astrophysics" and "fundamental new physics"?**

#### Key Reasoning

**Argument against the "simple astrophysics" hypothesis:**

If the U-shape observed in BTFR residuals were merely poorly understood feedback/baryonic physics, one would expect an ad hoc phenomenological description.

However, QO+R is NOT ad hoc because:
- It derives naturally from Type IIB → Calabi-Yau → KKLT
- Two scalar fields emerge (dilaton Φ, Kähler modulus T)
- Theoretical prediction for coupling: λ_QR ~ O(1)
- Observed value: 1.01 ± 0.08 (TNG300 validation)
- The QR ≈ const constraint corresponds to S-T duality

**Decisive rhetorical question:** "Why would 'simple astrophysics' have exactly the mathematical structure of string theory?"

#### Status of Q ↔ dilaton, R ↔ Kähler Identification

**What is NOT circular:**
1. λ_QR ≈ 1 emerged from data BEFORE any string theory consideration
2. KKLT independently predicts O(1) for this type of coupling
3. The Q-R anti-correlation (r = -0.89) matches S-T duality without being sought

**What IS a working hypothesis:**
- The identification Q ↔ dilaton, R ↔ Kähler is *proposed*, not derived ab initio

#### Paper 4 Objective

Provide **specific falsifiable predictions** that ONLY the string theory connection would make, enabling definitive discrimination.

---

### Predictions Identified

#### 1. Redshift Evolution λ_QR(z) [PRIORITY]
```
λ_QR(z) = λ_QR(0) × (1 + εz + O(z²))
```
- ε calculable from Calabi-Yau geometry
- Specific predicted value, not a free fit parameter
- Testable with WALLABY (z < 0.26), future surveys for higher z

#### 2. Higher-Order Corrections
- KKLT potential predicts Q³R, QR³ terms, etc.
- Ratios fixed by theory
- Verifiable on large statistical samples

#### 3. Fine-Structure Constant Coupling
- If Q = dilaton → α_EM varies slightly with galactic environment
- Testable via quasar spectroscopy

#### 4. Field Masses
- m_Q and m_R have KKLT-predicted values
- Implies characteristic length scale for QO+R effects

---

### Next Steps

1. [ ] Calculate theoretical prediction for ε from KKLT
2. [x] Identify available high-redshift data sources → Real observations only
3. [x] Design appropriate statistical tests → λ_QR fit per dataset
4. [x] Implement analysis scripts
5. [ ] Execute analyses and interpret results

---

### 2024-12-06 (afternoon) - CRITICAL METHODOLOGICAL DECISION

#### Decision: Real Observational Data Only

**Choice**: Paper 4 uses EXCLUSIVELY real astronomical observations.

**Rationale**:
- Maximum scientific rigor
- Avoid confusion between simulations and observations
- Simulations (TNG) are pure ΛCDM without string theory physics
- Conclusions must rest on the actual universe

**Consequence**: Redshift depth limited to z < 0.26 (WALLABY)

#### Datasets Used

| Dataset | N galaxies | Redshift | Type |
|---------|-----------|----------|------|
| SPARC | 175 | z ≈ 0 | Rotation curves |
| ALFALFA | ~31,500 | z < 0.06 | HI 21cm |
| WALLABY | ~5,000+ | z < 0.26 | HI 21cm |

#### Scripts Created

**1. `tests/test_real_data.py`** (MAIN SCRIPT)
- Loads SPARC, ALFALFA, WALLABY
- Computes λ_QR on each dataset
- Tests the killer prediction (sign inversion)
- Preliminary λ_QR(z) test on WALLABY

**2. `data/DATA_README.md`**
- Instructions for obtaining each dataset
- URLs, references, formats

#### λ_QR Fit Methodology

For each dataset:
1. Define Q = gas_fraction (normalized)
2. Define R = 1 - gas_fraction (normalized)
3. Fit: BTFR_residual = α + β_Q·(Q·env²) + β_R·(R·env²)
4. Extract λ_QR = -β_R/β_Q
5. Bootstrap 200× for robust error estimation

#### Logic Validation (verified on synthetic data)
```
β_Q injected: 0.0200, recovered: 0.0207
β_R injected: -0.0200, recovered: -0.0207
λ_QR expected: 1.000, computed: 1.000
✓ Fit methodology validated
```

#### Immediate Next Actions
1. Execute `test_real_data.py` on SPARC/ALFALFA/WALLABY
2. Document results in 04_RESULTS/
3. Analyze λ_QR(z) using WALLABY redshift bins

---

### 2024-12-06 - Data Column Mapping

#### SPARC Processed File Structure
File: `Data/processed/sparc_with_environment.csv`

| Column | Description | Unit |
|--------|-------------|------|
| Galaxy | Galaxy name | - |
| M_star | Stellar mass | 10^10 M☉ |
| MHI | HI gas mass | 10^10 M☉ |
| M_bar | Baryonic mass | 10^10 M☉ |
| Vflat | Flat rotation velocity | km/s |
| log_Mbar | log10(M_bar) | - |
| log_Vflat | log10(Vflat) | - |
| density_proxy | Environment density | - |
| env_class | Environment type | field/group/cluster |

#### Derived Quantities (computed in analysis)
- `gas_fraction` = MHI / M_bar
- `log_env` = log10(density_proxy)
- `btfr_residual` = log_Vflat - BTFR_prediction

---

*This log will be updated at each significant research milestone.*
