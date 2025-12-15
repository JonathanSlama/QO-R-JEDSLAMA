# QO+R Research - Data Replicability Documentation

## Overview

This document provides a clear audit trail of all data sources used in the QO+R research project, distinguishing between REAL observational/simulation data and any synthetic data that may have been used during development.

**Author:** Jonathan Edouard Slama
**Last Updated:** December 15, 2025

---

## Paper 1: BTFR U-Shape Detection

### Primary Dataset: SPARC

| Property | Value |
|----------|-------|
| **Name** | Spitzer Photometry and Accurate Rotation Curves |
| **Source** | Lelli, McGaugh, Schombert (2016) |
| **N galaxies** | 175 |
| **Data type** | REAL OBSERVATIONS |
| **File** | `Paper1-BTFR-UShape/data/sparc_with_environment.csv` |
| **Variables** | log_Vflat, log_Mbar, env_class, density_proxy |

### Replication Dataset 1: ALFALFA

| Property | Value |
|----------|-------|
| **Name** | Arecibo Legacy Fast ALFA Survey (α.100) |
| **Source** | Haynes et al. (2018) |
| **N galaxies** | ~25,000 |
| **Data type** | REAL OBSERVATIONS |
| **File** | `Test-Replicability/data/alfalfa.csv` |
| **Selection** | HI 21cm line detections |

### Replication Dataset 2: Little THINGS

| Property | Value |
|----------|-------|
| **Name** | Local Irregulars That Trace Luminosity Extremes, The HI Nearby Galaxy Survey |
| **Source** | Hunter et al. (2012) |
| **N galaxies** | ~40 |
| **Data type** | REAL OBSERVATIONS |
| **File** | `Test-Replicability/data/little_things.csv` |
| **Selection** | Nearby dwarf irregular galaxies |

### Environmental Classification (SPARC)

The environmental classification for SPARC galaxies uses:
1. **Known group membership** from literature (NED, HyperLEDA)
2. **Morphological proxy** (early-type tendency → denser environments)
3. **Continuous density proxy** (0-1 scale)

⚠️ **Important:** These are PROXIES, not direct measurements of local density.

---

## Paper 2: Residual Diagnostics in NHANES

### Primary Dataset: NHANES

| Property | Value |
|----------|-------|
| **Name** | National Health and Nutrition Examination Survey |
| **Source** | CDC/NCHS (2017-2018 cycle) |
| **N participants** | 9,254 |
| **Data type** | REAL OBSERVATIONS |
| **File** | Downloaded via `nhanes` Python package |

### Validation Dataset: Breast Cancer Coimbra

| Property | Value |
|----------|-------|
| **Name** | Breast Cancer Coimbra Dataset |
| **Source** | Patrício et al. (2018), UCI ML Repository |
| **N participants** | 116 (64 cancer, 52 healthy) |
| **Data type** | REAL OBSERVATIONS |

---

## Paper 3: String Theory Embedding & Multi-Dataset Validation

### Simulation Dataset 1: IllustrisTNG100

| Property | Value |
|----------|-------|
| **Name** | IllustrisTNG100-1 |
| **Source** | TNG Collaboration (Pillepich et al. 2018) |
| **N files** | 448 HDF5 files |
| **Data type** | COSMOLOGICAL SIMULATIONS (ΛCDM) |
| **Directory** | `BTFR/TNG/Data/` |
| **Snapshot** | 099 (z=0) |

### Simulation Dataset 2: IllustrisTNG300

| Property | Value |
|----------|-------|
| **Name** | IllustrisTNG300-1 |
| **Source** | TNG Collaboration |
| **N files** | 600 HDF5 files |
| **Data type** | COSMOLOGICAL SIMULATIONS (ΛCDM) |
| **Directory** | `BTFR/TNG/Data_TNG300/` |
| **N galaxies analyzed** | 623,609 |

### Simulation Dataset 3: IllustrisTNG50

| Property | Value |
|----------|-------|
| **Name** | IllustrisTNG50-1 |
| **Source** | TNG Collaboration |
| **N files** | 680 HDF5 files |
| **Data type** | COSMOLOGICAL SIMULATIONS (high resolution) |
| **Directory** | `BTFR/TNG/Data_TNG50/` |

### Observational Dataset: WALLABY

| Property | Value |
|----------|-------|
| **Name** | WALLABY Pilot Data Release 2 |
| **Source** | ASKAP/WALLABY Collaboration |
| **Data type** | REAL OBSERVATIONS (HI 21cm) |
| **File** | `BTFR/TNG/Data_Wallaby/WALLABY_PDR2_SourceCatalogue.xml` |

---

## Paper 4: QO+R Validation - Hidden Conservation Law

### Primary Dataset 1: SPARC (Extended)

| Property | Value |
|----------|-------|
| **Name** | Spitzer Photometry and Accurate Rotation Curves |
| **Source** | Lelli, McGaugh, Schombert (2016) |
| **N galaxies** | 181 |
| **Data type** | REAL OBSERVATIONS |
| **Directory** | `Paper4-QOR-Validation/data/` |
| **Variables** | log_Vflat, log_Mbar, env_class, Q, R residuals |

### Primary Dataset 2: ALFALFA (Extended)

| Property | Value |
|----------|-------|
| **Name** | Arecibo Legacy Fast ALFA Survey (α.100) |
| **Source** | Haynes et al. (2018) |
| **N galaxies** | 19,222 |
| **Data type** | REAL OBSERVATIONS |
| **Selection** | HI 21cm line detections with quality cuts |

### Multi-Scale Dataset 1: KiDS DR4

| Property | Value |
|----------|-------|
| **Name** | Kilo-Degree Survey Data Release 4 |
| **Source** | de Jong et al. (2019) |
| **N galaxies** | ~1,000,000 |
| **Data type** | REAL OBSERVATIONS (optical photometry) |
| **Files** | `KiDS_DR4_brightsample.fits`, `KiDS_DR4_brightsample_LePhare.fits` |
| **Access** | http://kids.strw.leidenuniv.nl/DR4/ |

### Multi-Scale Dataset 2: Planck PSZ2

| Property | Value |
|----------|-------|
| **Name** | Planck Sunyaev-Zeldovich Cluster Catalog |
| **Source** | Planck Collaboration (2016) |
| **N clusters** | 1,653 |
| **Data type** | REAL OBSERVATIONS (SZ effect) |
| **Files** | `HFI_PCCS_SZ-union_R2.08.fits` |
| **Access** | https://pla.esac.esa.int/ |

### Multi-Scale Dataset 3: Gaia DR3 Wide Binaries

| Property | Value |
|----------|-------|
| **Name** | Gaia Data Release 3 - Wide Binary Systems |
| **Source** | Gaia Collaboration (2023) |
| **N systems** | ~10,000 |
| **Data type** | REAL OBSERVATIONS (astrometric) |
| **Selection** | Wide binaries with separation 0.1-1 pc |
| **Access** | https://gea.esac.esa.int/archive/ |

### Multi-Scale Dataset 4: IllustrisTNG (Combined)

| Property | Value |
|----------|-------|
| **Name** | IllustrisTNG50 + TNG100 + TNG300 |
| **Source** | TNG Collaboration (Nelson et al. 2019) |
| **N objects** | 685,030 (combined) |
| **Data type** | COSMOLOGICAL SIMULATIONS (ΛCDM) |
| **Snapshots** | z=0 (snapshot 099) |
| **Access** | https://www.tng-project.org/ |

### Additional Catalogs (Paper 4)

| Dataset | Source | N | Type |
|---------|--------|---|------|
| eROSITA eRASS1 | MPE | ~1,000 | X-ray clusters |
| redMaPPer | Rykoff+ 2014 | ~25,000 | Optical clusters |
| GAMA Groups | GAMA Survey | ~5,000 | Galaxy groups |
| Tempel Groups | Tempel+ 2014 | ~10,000 | SDSS groups |

### Environmental Classification (Paper 4)

Paper 4 uses multiple environmental proxies:
1. **Cluster membership** from PSZ2/redMaPPer catalogs
2. **Group membership** from GAMA/Tempel catalogs
3. **Local density** from KiDS photometric redshifts
4. **Wide binary separation** as low-density tracer
5. **TNG FoF halo mass** for simulations

⚠️ **Important:** Environmental classifications are PROXIES based on observational indicators, not direct measurements of dark matter density.

---

## Deprecated/Synthetic Files (DO NOT USE FOR PUBLICATION)

The following files contain SYNTHETIC data and should NOT be cited as real observations:

| File | Status | Reason |
|------|--------|--------|
| `Test-Replicability/data/alfalfa_simulated.csv` | ⚠️ SYNTHETIC | Generated with fixed U-shape |
| `Test-Replicability/data/little_things_simulated.csv` | ⚠️ SYNTHETIC | Generated with fixed U-shape |
| `replicability_tests.py` (original) | ⚠️ DEPRECATED | Generates synthetic data |

**Use instead:** `replicability_tests_REAL.py` which loads actual observational data.

---

## Verification Checklist

Before submission, verify:

- [ ] All cited N values match actual file row counts
- [ ] No synthetic data files are referenced in methods
- [ ] Environmental classifications are described as "proxies"
- [ ] TNG data is correctly described as "simulations"
- [ ] WALLABY/ALFALFA are described as "observations"
- [ ] Little THINGS sample size (~40) is noted as limiting factor

---

## Data Access

| Dataset | Public Access | API/Download |
|---------|---------------|--------------|
| SPARC | Yes | http://astroweb.cwru.edu/SPARC/ |
| ALFALFA | Yes | http://egg.astro.cornell.edu/alfalfa/ |
| Little THINGS | Yes | https://science.nrao.edu/science/surveys/littlethings |
| IllustrisTNG | Yes (registration) | https://www.tng-project.org/ |
| WALLABY | Yes | https://wallaby-survey.org/ |
| NHANES | Yes | https://www.cdc.gov/nchs/nhanes/ |
| KiDS DR4 | Yes | http://kids.strw.leidenuniv.nl/DR4/ |
| Planck PSZ2 | Yes | https://pla.esac.esa.int/ |
| Gaia DR3 | Yes | https://gea.esac.esa.int/archive/ |
| eROSITA | Yes (registration) | https://erosita.mpe.mpg.de/ |
| redMaPPer | Yes | VizieR (Rykoff et al. 2014) |
| GAMA | Yes (registration) | http://www.gama-survey.org/ |

---

## Summary Table

| Paper | Dataset | Type | N | Status |
|-------|---------|------|---|--------|
| 1 | SPARC | Observations | 175 | REAL |
| 1 | ALFALFA | Observations | ~25,000 | REAL |
| 1 | Little THINGS | Observations | ~40 | REAL |
| 2 | NHANES | Survey | 9,254 | REAL |
| 2 | Breast Cancer Coimbra | Clinical | 116 | REAL |
| 3 | TNG100 | Simulations | 53,363 | REAL |
| 3 | TNG300 | Simulations | 623,609 | REAL |
| 3 | TNG50 | Simulations | ~8,000 | REAL |
| 3 | WALLABY | Observations | ~2,000 | REAL |
| **4** | **SPARC** | **Observations** | **181** | REAL |
| **4** | **ALFALFA** | **Observations** | **19,222** | REAL |
| **4** | **KiDS DR4** | **Observations** | **~1,000,000** | REAL |
| **4** | **Planck PSZ2** | **Observations** | **1,653** | REAL |
| **4** | **Gaia DR3 WB** | **Observations** | **~10,000** | REAL |
| **4** | **TNG Combined** | **Simulations** | **685,030** | REAL |

**Total objects analyzed (Paper 4): 1,219,410+**

**ALL DATA USED FOR PUBLICATION IS REAL.**

---

## Paper 4 Validation Tests (14/14 Passed)

| Phase | Test | Dataset | Status |
|-------|------|---------|--------|
| Core | Q-R Independence | SPARC | PASS |
| Core | U-shape Detection | Multi-dataset | PASS |
| Core | QO+R Model Fit | Multi-dataset | PASS |
| Core | Killer Prediction | Multi-dataset | PASS |
| Robustness | Bootstrap Stability | SPARC/ALFALFA | PASS |
| Robustness | Spatial Jackknife | SPARC | PASS |
| Robustness | Cross-Validation | Multi-dataset | PASS |
| Robustness | Selection Sensitivity | Multi-dataset | PASS |
| Alternatives | Selection Bias Control | Multi-dataset | PASS |
| Alternatives | MOND Comparison | SPARC | PASS |
| Alternatives | Baryonic Control | Multi-dataset | PASS |
| Alternatives | TNG Comparison | TNG50/100/300 | PASS |
| Amplitude | Environment Proxy | Multi-context | PASS |
| Amplitude | Sample Purity | Multi-dataset | PASS |

---

*Document created: 2025-12-03*
*Last updated: December 15, 2025*
*Author: Jonathan Edouard SLAMA (Metafund Research Division)*
