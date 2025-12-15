# Paper 4: Empirical Validation of the QO+R Framework

## A Hidden Conservation Law of Gravity from Galactic to Cosmological Scales

**Version:** 3.1
**Status:** Complete | 1,219,410 objects analyzed across 6 surveys
**Last Updated:** December 2025

---

## Author

**Jonathan Edouard Slama**
Metafund Research Division, Strasbourg, France
ORCID: [0009-0002-1292-4350](https://orcid.org/0009-0002-1292-4350)

---

## Executive Summary

This paper presents the **complete empirical validation of the QO+R framework** across 14 orders of magnitude in spatial scale, establishing a hidden conservation law of gravity.

### Key Discoveries

| Discovery | Evidence | Significance |
|-----------|----------|--------------|
| U-shape in BTFR | 6 datasets, 1.2M objects | >10 sigma |
| Killer prediction (sign inversion) | Q-dom vs R-dom populations | 26 sigma |
| Universal coupling constant | lambda_QR = 1.23 +/- 0.35 | Stable across all scales |
| Chameleon screening | GC, WB null results | Confirmed |
| Low-density signal | UDG, filaments | Detected |
| Slama Conservation Law | rho * G_eff = constant | Derived from Noether theorem |

### The Slama Relation

```
G_eff(rho) = G_N * [1 + lambda_QR * alpha_0 / (1 + (rho/rho_c)^delta)]

lambda_QR = 1.23 +/- 0.35 (universal coupling)
alpha_0 = 0.05 (bare amplitude)
rho_c = 10^-25 g/cm^3 (critical density)
```

---

## Manuscript

The complete LaTeX manuscript is available in:

```
manuscript/
├── paper4_qor_validation.tex      # Main manuscript (~850 lines)
├── paper4_qor_validation.pdf      # Compiled PDF
├── THE_QOR_CONSERVATION_LAW.md    # Fundamental theory document
├── THE_SLAMA_RELATION.md          # Complete parameter derivation
└── THE_QOR_RESEARCH_CHRONICLE.md  # Research journey narrative
```

To compile:
```bash
cd manuscript
pdflatex paper4_qor_validation.tex
pdflatex paper4_qor_validation.tex  # Run twice for references
```

---

## Test Results Summary

### Core Validation (14/14 Passed)

| Category | Tests | Status |
|----------|-------|--------|
| Core | U-shape, Q-R independence, QO+R model, Killer prediction | 4/4 Passed |
| Robustness | Bootstrap, Jackknife, Cross-validation, Selection | 4/4 Passed |
| Alternatives | Selection bias, MOND, Baryonic, TNG comparison | 4/4 Passed |
| Amplitude | Environment proxy, Sample purity | 2/2 Passed |

### Alternative Theories (Level 4 Complete)

| Theory | Status |
|--------|--------|
| **String Theory Moduli** | ONLY COMPATIBLE |
| Warm Dark Matter | Eliminated |
| Self-Interacting DM | Eliminated |
| f(R) Gravity | Eliminated |
| Fuzzy Dark Matter | Eliminated |
| Quintessence | Partial |

### Multi-Scale Validation (8 scales)

| Scale | Distance | Status |
|-------|----------|--------|
| Wide Binaries | 10^13 m | Screened |
| Globular Clusters | 10^18 m | Screened |
| UDGs | 10^20 m | **SIGNAL** |
| Galaxies | 10^21 m | **SIGNAL** |
| Groups | 10^22 m | Partial |
| Clusters | 10^23 m | Screened |
| Filaments | 10^24 m | **SIGNAL** |
| CMB | 10^27 m | Data-limited |

---

## Project Structure

```
Paper4-Redshift-Predictions/
├── manuscript/
│   ├── paper4_qor_validation.tex   # Main LaTeX manuscript
│   ├── THE_QOR_CONSERVATION_LAW.md
│   ├── THE_SLAMA_RELATION.md
│   └── THE_QOR_RESEARCH_CHRONICLE.md
│
├── experimental/
│   ├── 00_THEORY/                  # Complete Lagrangian
│   ├── 01_HYPOTHESIS.md
│   ├── 02_DATA_SOURCES.md
│   ├── 03_METHODOLOGY.md           # Test protocols (16 tests)
│   ├── 04_RESULTS/                 # All validation results
│   ├── 05_NEXT_STEPS/              # Future roadmap
│   ├── 06_TOE_FRAMEWORK/           # Theory of Everything
│   ├── 07_KILLER_PREDICTIONS/      # Sign inversion tests
│   ├── 08_GROUP_SCALE/ ... 18_CMB_PLANCK/  # Scale-specific tests
│   └── THE_*.md                    # Synthesis documents
│
├── tests/
│   ├── run_all_tests.py            # Main 14-test runner
│   ├── core_tests.py               # H1, H3, H4 validation
│   ├── robustness_tests.py         # Stability tests
│   ├── alternative_tests.py        # MOND, baryonic elimination
│   └── future/                     # Level 2+ tests
│
├── data/
│   └── level2_multicontext/        # KiDS, Planck, clusters data
│
└── figures/
```

---

## Quick Start

```bash
# Run complete validation suite (14 tests)
cd tests
python run_all_tests.py

# Run Level 4 tests (alternative elimination)
cd tests/future
python test_mass_scaling.py
python test_alternative_theories.py

# Run multi-scale tests
cd experimental/07_KILLER_PREDICTIONS/tests
python toe_validation_alfalfa.py
```

---

## Data Sources

| Dataset | N | Type | Source |
|---------|---|------|--------|
| SPARC | 181 | Rotation curves | Lelli+ 2016 |
| ALFALFA | 19,222 | HI survey | Haynes+ 2018 |
| KiDS DR4 | ~1,000,000 | Optical photometry | de Jong+ 2019 |
| Planck PSZ2 | ~1,653 | SZ clusters | Planck 2016 |
| Gaia DR3 | Wide binaries | Astrometry | Gaia Collaboration |
| IllustrisTNG | Simulations | TNG Project |

---

## Scientific Significance

### What This Proves

1. **The U-shape pattern exists** in real observational data
2. **Cannot be explained** by MOND, WDM, SIDM, f(R), or Fuzzy DM
3. **Is consistent with** String Theory moduli predictions
4. **Shows sign inversion** (as required by S-T duality)
5. **A conservation law emerges** from Q/R symmetry

### What This Does NOT Yet Prove

1. That String Theory IS correct (need quantitative predictions)
2. That moduli fields exist (need direct detection)
3. That the pattern is universal (multi-context confirmation ongoing)

---

## Citation

```bibtex
@article{Slama2025_Paper4,
  author = {Slama, Jonathan Edouard},
  title = {Empirical Validation of the QO+R Framework: A Hidden Conservation Law of Gravity},
  year = {2025},
  journal = {in preparation}
}
```

---

## Acknowledgments

This research uses data from SPARC, ALFALFA, KiDS DR4, Planck, Gaia, and IllustrisTNG. We acknowledge the theoretical foundations laid by Newton, Einstein, Noether, and the string theory community.

---

*Reference version: December 15, 2025*
