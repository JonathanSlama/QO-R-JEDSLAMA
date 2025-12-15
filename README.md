# QO+R Framework: From String Theory to Galactic Observations

## A Novel Two-Field Modified Gravity Framework with Empirical Validation

**Author:** Jonathan Edouard Slama
**Affiliation:** Metafund Research Division, Strasbourg, France
**Contact:** jonathan@metafund.in
**ORCID:** [0009-0002-1292-4350](https://orcid.org/0009-0002-1292-4350)
**Version:** 4.0
**Date:** December 2025

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17943132.svg)](https://doi.org/10.5281/zenodo.17943132)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This repository contains the complete **QO+R (Quotient Ontologique + Reliquat)** research framework, including:

- **Theoretical derivation** from Type IIB string theory
- **Empirical validation** on 1,219,410+ objects across 8 datasets
- **Medical extension** demonstrating universal residual methodology
- **Discovery of a hidden conservation law** of gravity
- **Full reproducibility** with scripts, data, and documentation

---

## The Four Papers

| Paper | Title | Key Result | Objects | PDF |
|-------|-------|------------|---------|-----|
| **Paper 1** | Environmental Modulation of the BTFR: Discovery of a U-Shaped Pattern | U-shape detection (4.4 sigma) | 24,056 | [paper1_qor_btfr_v3.pdf](Paper1-BTFR-UShape/manuscript/paper1_qor_btfr_v3.pdf) |
| **Paper 2** | Residual Diagnostic Methodology: From Galaxies to Biomarkers | 85% significant ratios | 9,254 | [paper2_residuals_v3.pdf](Paper2-Residual-Diagnostics/nhanes_extension/manuscript/paper2_residuals_v3.pdf) |
| **Paper 3** | From String Theory to Galactic Observations | lambda_QR ~ 1 confirmed | 708,086 | [paper3_string_theory_v2.pdf](Paper3-ToE/manuscript/paper3_string_theory_v2.pdf) |
| **Paper 4** | **A Hidden Conservation Law of Gravity** | **Slama Conservation Law** | **1,219,410** | [paper4_qor_validation.pdf](Paper4-QOR-Validation/manuscript/paper4_qor_validation.pdf) |

---

## Key Discoveries

### The Slama Conservation Law

```
rho * G_eff(rho) = constant
```

The product of matter density and effective gravitational strength is conserved across cosmic environments. This is a hidden symmetry of gravity, analogous to energy conservation in mechanics.

### The Slama Relation

```
G_eff(rho) = G_N * [1 + lambda_QR * alpha_0 / (1 + (rho/rho_c)^delta)]

Parameters:
- lambda_QR = 1.23 +/- 0.35 (universal coupling)
- alpha_0 = 0.05 (bare amplitude)
- rho_c = 10^-25 g/cm^3 (critical density)
- delta = 1.0 (screening exponent)
```

### Summary of Evidence

| Discovery | Evidence | Significance |
|-----------|----------|--------------|
| U-shape in BTFR | 6 datasets, 1.2M objects | >10 sigma |
| Sign inversion (killer prediction) | Q-dom vs R-dom populations | 26 sigma |
| Universal coupling constant | Stable across 14 orders of magnitude | lambda_QR = 1.23 +/- 0.35 |
| Chameleon screening | GC, WB null results | Confirmed |
| Low-density signal | UDG, filaments | Detected |
| Alternative theories eliminated | WDM, SIDM, f(R), Fuzzy DM | String Theory only compatible |

---

## Repository Structure

```
QO-R-JEDSLAMA/
|
+-- Paper1-BTFR-UShape/           # Discovery paper
|   +-- manuscript/               # LaTeX + PDF
|   +-- figures/                  # 13 publication figures
|   +-- data/                     # SPARC processed data
|   +-- tests/                    # 13 validation scripts
|
+-- Paper2-Residual-Diagnostics/  # Medical extension
|   +-- nhanes_extension/         # NHANES analysis
|   +-- data/                     # Clinical datasets
|
+-- Paper3-ToE/                   # String theory paper
|   +-- manuscript/               # LaTeX + PDF
|   +-- tests/                    # TNG validation scripts
|
+-- Paper4-QOR-Validation/        # Conservation law paper (NEW)
|   +-- manuscript/               # LaTeX + PDF + theory docs
|   +-- experimental/             # Complete research documentation
|   +-- tests/                    # 14-test validation suite
|   +-- data/                     # Multi-context datasets
|
+-- DATA_SOURCES.md               # How to obtain all datasets
+-- PUBLICATION_AUDIT.md          # Quality checklist
+-- REPLICABILITY_AUDIT.md        # Reproducibility checklist
+-- LICENSE                       # MIT License
+-- requirements.txt              # Python dependencies
+-- README.md                     # This file
```

---

## Data Sources

| Dataset | Type | N | Source | Papers |
|---------|------|---|--------|--------|
| SPARC | Observation | 181 | Lelli+ 2016 | 1, 3, 4 |
| ALFALFA | Observation | 19,222 | Haynes+ 2018 | 1, 3, 4 |
| WALLABY | Observation | 2,047 | ASKAP | 3, 4 |
| KiDS DR4 | Observation | ~1,000,000 | ESO | 4 |
| Planck PSZ2 | Observation | 1,653 | ESA | 4 |
| Gaia DR3 | Observation | Wide binaries | ESA | 4 |
| TNG50/100/300 | Simulation | 685,030 | IllustrisTNG | 3, 4 |
| NHANES | Medical | 9,254 | CDC | 2 |

See [DATA_SOURCES.md](DATA_SOURCES.md) for complete download instructions.

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/JonathanSlama/QO-R-JEDSLAMA.git
cd QO-R-JEDSLAMA
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Paper 4 validation suite (14 tests)
```bash
cd Paper4-QOR-Validation/tests
python run_all_tests.py
```

### 4. Compile manuscripts
```bash
cd Paper1-BTFR-UShape/manuscript && pdflatex paper1_qor_btfr_v3.tex
cd ../../Paper3-ToE/manuscript && pdflatex paper3_string_theory_v2.tex
cd ../../Paper4-QOR-Validation/manuscript && pdflatex paper4_qor_validation.tex
```

---

## Theoretical Framework

### The QO+R Lagrangian

```
L = sqrt(-g) [ (M_P^2/2) F(chi) R - (1/2)(d phi)^2 - (Z(chi)/2)(d chi)^2 - V(phi,chi) ]
  + L_matter[A^2(phi,chi) g_mu_nu, psi]
```

Where:
- **phi** (Q-field): Couples to gas via electromagnetic interactions
- **chi** (R-field): Couples to stars via gravitational interactions
- **A(phi,chi) = exp[beta(phi - kappa*chi)/M_P]**: Conformal coupling

### String Theory Origin

```
10D Type IIB Supergravity
         |
         v
Calabi-Yau Compactification (Quintic P^4[5])
         |
         v
4D Effective Theory (dilaton phi, Kahler modulus chi)
         |
         v
KKLT Moduli Stabilization
         |
         v
QO+R with lambda_QR ~ O(1)
         |
         v
Slama Conservation Law: rho * G_eff = constant
```

---

## Citation

If you use this work, please cite:

```bibtex
@software{slama2025qor,
  author       = {Slama, Jonathan Edouard},
  title        = {{QO+R Framework: From String Theory to Galactic
                   Observations - A Hidden Conservation Law of Gravity}},
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17943132},
  url          = {https://github.com/JonathanSlama/QO-R-JEDSLAMA}
}
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **SPARC Team** (Lelli, McGaugh, Schombert) for the galaxy database
- **IllustrisTNG Collaboration** for simulation data access
- **ALFALFA Team** (Haynes et al.) for the HI survey
- **KiDS Collaboration** for optical photometry
- **Planck Collaboration** for SZ cluster catalog
- **Gaia Collaboration** for astrometric data
- **Emmy Noether** for the theorem connecting symmetries to conservation laws
- All pioneers of gravity theory: Newton, Einstein, Hilbert, Schwarzschild
- **Iris** - An AI assistant trained with my reasoning methodology, for invaluable help with manuscript drafting, test design, and iterative refinement of the scientific arguments

---

## Important Caveats

This work presents:
1. A mathematically consistent framework
2. Empirical patterns confirmed across 1.2 million objects
3. A possible connection to string theory
4. A hidden conservation law of gravity

This work does **NOT** prove:
1. That string theory is correct
2. That Q and R are literally the dilaton and Kahler modulus
3. That no alternative explanations exist

The connection to string theory remains a **hypothesis**, not a conclusion.

---

*Last updated: December 15, 2025*
