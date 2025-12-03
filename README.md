# QO+R Framework: From String Theory to Galactic Observations

## A Novel Two-Field Modified Gravity Framework with Empirical Validation

**Author:** Jonathan Édouard Slama  
**Affiliation:** Metafund Research Division, Strasbourg, France  
**Contact:** jonathan@metafund.in  
**ORCID:** [0009-0002-1292-4350](https://orcid.org/0009-0002-1292-4350)  
**Version:** 3.0  
**Date:** December 2025

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17806442.svg)](https://doi.org/10.5281/zenodo.17806442)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Overview

This repository contains the complete **QO+R (Quotient Ontologique + Reliquat)** research framework, including:

- **Theoretical derivation** from Type IIB string theory
- **Empirical validation** on 708,086 galaxies across 5 datasets
- **Medical extension** demonstrating universal residual methodology
- **Full reproducibility** with scripts, data, and documentation

---

## 🧭 The Scientific Journey

This research began with a **failure** and ended with a **discovery**.

### Act 1: The Failed Prediction

We started with a simple hypothesis: a single scalar field *Q* coupled to galaxy gas content should produce a **monotonic** environmental dependence in Baryonic Tully-Fisher Relation (BTFR) residuals. Dense environments → higher residuals. Or lower. But monotonic.

**The data said no.** 

SPARC galaxies showed something unexpected: residuals were elevated in *both* voids AND clusters, with a minimum in between. A **U-shape**.

### Act 2: The Discovery

Rather than abandoning the project, we asked: *What could produce a U-shape?*

The answer required **two fields**, not one:
- **Q** (Quotient Ontologique): Couples to gas, dominates in voids
- **R** (Reliquat): Couples to stars, dominates in clusters

Their antagonistic behavior naturally produces the U-shape. The model worked—but a mystery remained: the coupling constant λ_QR came out almost exactly **1**. Why?

### Act 3: The Replication

A pattern in 175 galaxies could be a fluke. We tested on:
- **ALFALFA** (21,834 galaxies): U-shape **replicated** (p = 0.0065)
- **WALLABY** (2,047 galaxies): U-shape **consistent**
- **IllustrisTNG** (685,030 simulated galaxies): U-shape **confirmed**

### Act 4: The Killer Prediction

If Q and R are truly antagonistic, then **R-dominated systems** (gas-poor, massive galaxies) should show an **inverted** U-shape: coefficient *a < 0* instead of *a > 0*.

This prediction was made **before** analyzing TNG300 R-dominated galaxies.

**Result:** a = −0.019 ± 0.003 (significance: **6.3σ**)

The sign flip occurred exactly as predicted. Probability by chance: ~10⁻¹².

### Act 5: The String Theory Connection

Why λ_QR ≈ 1? We found a possible answer in string theory:
- **Q** ↔ Dilaton (controls string coupling)
- **R** ↔ Kähler modulus (controls compact dimensions)
- **λ_QR ~ O(1)** emerges naturally from Calabi-Yau geometry (quintic P⁴[5])

This connection remains a **hypothesis**, not a proof. But the mathematics is consistent, and the predictions work.

### The Bottom Line

| Stage | What Happened | Result |
|-------|---------------|--------|
| Hypothesis | Single field Q | ❌ Failed |
| Discovery | U-shape in residuals | ✅ Detected (4.4σ) |
| Replication | Independent datasets | ✅ Confirmed |
| Killer Test | Sign inversion prediction | ✅ 6.3σ |
| Theory | String theory derivation | ⚠️ Hypothesis |

---

## 🔬 The Three Papers

| Paper | Title | Pages | Figures | Galaxies | PDF |
|-------|-------|-------|---------|----------|-----|
| **Paper 1** | Environmental Modulation of the BTFR: Discovery of a U-Shaped Pattern | ~30 | 13 | 24,056 | [paper1_qor_btfr_v3.pdf](Paper1-BTFR-UShape/manuscript/paper1_qor_btfr_v3.pdf) |
| **Paper 2** | Residual Diagnostic Methodology: From Galaxies to Biomarkers | ~25 | 11 | N/A (NHANES) | [paper2_residuals_v3.pdf](Paper2-Residual-Diagnostics/nhanes_extension/manuscript/paper2_residuals_v3.pdf) |
| **Paper 3** | From String Theory to Galactic Observations: Complete Derivation and Validation | ~30 | 8 | 708,086 | [paper3_string_theory_v2.pdf](Paper3-ToE/manuscript/paper3_string_theory_v2.pdf) |

---

## 🏗️ Repository Structure

```
QO-R-JEDSLAMA/
│
├── Paper1-BTFR-UShape/           # Discovery paper
│   ├── manuscript/               # LaTeX + PDF
│   ├── figures/                  # 13 publication figures
│   ├── data/                     # SPARC processed data
│   ├── tests/                    # 13 validation scripts
│   ├── README.md
│   └── README_REPRODUCIBILITY.md
│
├── Paper2-Residual-Diagnostics/  # Medical extension
│   ├── nhanes_extension/         # NHANES analysis
│   │   ├── manuscript/           # LaTeX + PDF
│   │   ├── figures/              # 11 figures
│   │   ├── data/                 # NHANES data
│   │   ├── scripts/              # Analysis pipeline
│   │   └── results/              # Output tables
│   ├── data/                     # Breast cancer dataset
│   └── README.md
│
├── Paper3-ToE/                   # String theory paper
│   ├── manuscript/               # LaTeX + PDF
│   ├── figures/                  # 8 figures
│   ├── tests/                    # TNG validation scripts
│   ├── README.md
│   └── README_REPRODUCIBILITY.md
│
├── DATA_SOURCES.md               # How to obtain all datasets
├── PUBLICATION_AUDIT.md          # Quality checklist
├── REPLICABILITY_AUDIT.md        # Reproducibility checklist
├── LICENSE                       # MIT License
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🎯 Key Results

### Paper 1: Discovery
| Result | Value | Significance |
|--------|-------|--------------|
| U-shape coefficient (SPARC) | a = +0.035 ± 0.008 | 4.4σ |
| Monte Carlo survival | 100% (1000 runs) | Robust |
| Bootstrap confidence | 96.5% | Stable |

### Paper 3: String Theory Validation
| Prediction | Theory | Observed | Status |
|------------|--------|----------|--------|
| λ_QR ≈ 1 | 0.93 ± 0.15 | 1.01 (TNG300) | ✅ Confirmed |
| Sign inversion (R-dom) | a < 0 | -0.019 ± 0.003 | ✅ 6.3σ |
| QR ≈ const | Expected | r = -0.89 | ✅ Confirmed |

### The "Killer Prediction"
> **Predicted before analysis:** R-dominated (gas-poor) systems show inverted U-shape (a < 0)  
> **Result:** a = -0.019 ± 0.003 (significance: **6.3σ**)  
> **Probability by chance:** ~10⁻¹²

---

## 📊 Data Sources

| Dataset | Type | N | Source | Access |
|---------|------|---|--------|--------|
| SPARC | Observation | 175 | Spitzer + HI | [Public](http://astroweb.cwru.edu/SPARC/) |
| ALFALFA | Observation | 21,834 | Arecibo 21-cm | [Public](http://egg.astro.cornell.edu/alfalfa/data/) |
| WALLABY | Observation | 2,047 | ASKAP | [Registration](https://wallaby-survey.org/data/) |
| TNG50/100/300 | Simulation | 685,030 | IllustrisTNG | [Registration](https://www.tng-project.org/data/) |
| NHANES | Medical | 9,254 | CDC | [Public](https://wwwn.cdc.gov/nchs/nhanes/) |

See [DATA_SOURCES.md](DATA_SOURCES.md) for complete download instructions.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/JonathanSlama/QO-R-JEDSLAMA.git
cd QO-R-JEDSLAMA
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Paper 3 figures (uses pre-computed TNG results)
```bash
cd Paper3-ToE/tests
python generate_figures_from_results.py
```

### 4. Compile manuscripts
```bash
cd Paper1-BTFR-UShape/manuscript && pdflatex paper1_qor_btfr_v3.tex
cd ../../Paper2-Residual-Diagnostics/nhanes_extension/manuscript && pdflatex paper2_residuals_v3.tex
cd ../../../Paper3-ToE/manuscript && pdflatex paper3_string_theory_v2.tex
```

---

## 📐 Theoretical Framework

### The QO+R Lagrangian

```
ℒ_QO+R = ½(∂Q)² + ½(∂R)² - V(Q,R) - λ_QR Q²R² · T^μν
```

Where:
- **Q** (Quotient Ontologique) ↔ Dilaton φ: couples to gas (HI)
- **R** (Reliquat) ↔ Kähler modulus ψ: couples to stars
- **λ_QR ≈ 1**: emerges from Calabi-Yau geometry (quintic P⁴[5])

### Derivation Chain

```
10D Type IIB Supergravity
         ↓
Calabi-Yau Compactification (Quintic)
         ↓
4D Effective Theory (φ, ψ)
         ↓
KKLT Moduli Stabilization
         ↓
QO+R with λ_QR ~ O(1)
         ↓
U-shape in BTFR residuals
```

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@software{slama2025qor,
  author       = {Slama, Jonathan Édouard},
  title        = {{QO+R Framework: From String Theory to Galactic 
                   Observations}},
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17806442},
  url          = {https://github.com/JonathanSlama/QO-R-JEDSLAMA}
}
```

---

## ⚖️ License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **SPARC Team** (Lelli, McGaugh, Schombert) for the galaxy database
- **IllustrisTNG Collaboration** for simulation data access
- **ALFALFA Team** (Haynes et al.) for the HI survey
- **WALLABY Team** (Koribalski et al.) for ASKAP data
- **NHANES/CDC** for public health data

---

## ⚠️ Important Caveats

This work presents:
1. ✅ A mathematically consistent framework
2. ✅ Empirical patterns confirmed across multiple datasets
3. ✅ A possible connection to string theory

This work does **NOT** prove:
1. ❌ That string theory is correct
2. ❌ That Q and R are literally the dilaton and Kähler modulus
3. ❌ That no alternative explanations exist

The connection to string theory remains a **hypothesis**, not a conclusion.

---

*Last updated: December 2025*
