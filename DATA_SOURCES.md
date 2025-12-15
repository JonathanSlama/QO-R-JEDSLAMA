# Data Sources and Download Instructions

This document explains how to obtain all datasets used in the QO+R research project.

**Author:** Jonathan Edouard Slama
**Contact:** jonathan@metafund.in
**Last Updated:** December 2025

---

## Overview

| Dataset | Size | Papers | Access |
|---------|------|--------|--------|
| SPARC | 100 KB | 1, 3, 4 | Public |
| ALFALFA | 5 MB | 1, 3, 4 | Public |
| WALLABY | 50 MB | 3, 4 | Public |
| KiDS DR4 | ~500 MB | 4 | Public |
| Planck PSZ2 | ~100 MB | 4 | Public |
| Gaia DR3 (WB) | ~1 GB | 4 | Public |
| IllustrisTNG | ~10 GB | 3, 4 | Registration |
| NHANES | ~50 MB | 2 | Public |

**Total (all datasets):** ~12 GB
**Minimum (SPARC + ALFALFA):** ~5 MB

---

## 1. SPARC Database (Papers 1, 3, 4)

**Description:** Spitzer Photometry and Accurate Rotation Curves - 175 galaxies with high-quality rotation curves.

**Source:** http://astroweb.cwru.edu/SPARC/

**Download:**
```bash
wget http://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt
```

**Reference:** Lelli, McGaugh, Schombert (2016), AJ, 152, 157

---

## 2. ALFALFA Survey (Papers 1, 3, 4)

**Description:** Arecibo Legacy Fast ALFA survey - HI 21cm observations.

**Source:** http://egg.astro.cornell.edu/alfalfa/data/

**Files needed:**
- `a100.code12.table2.190808.csv`

**Download:**
```bash
wget http://egg.astro.cornell.edu/alfalfa/data/a100.code12.table2.190808.csv
```

**Reference:** Haynes et al. (2018), ApJ, 861, 49

---

## 3. KiDS DR4 (Paper 4)

**Description:** Kilo-Degree Survey Data Release 4 - Optical photometry for ~1 million galaxies.

**Source:** http://kids.strw.leidenuniv.nl/DR4/

**Files needed:**
- `KiDS_DR4_brightsample.fits`
- `KiDS_DR4_brightsample_LePhare.fits`

**Download:** Via ESO archive or direct links from KiDS website.

**Reference:** de Jong et al. (2019), A&A, 625, A2

---

## 4. Planck PSZ2 Catalog (Paper 4)

**Description:** Planck Sunyaev-Zeldovich cluster catalog - 1,653 clusters.

**Source:** https://pla.esac.esa.int/

**Files needed:**
- `HFI_PCCS_SZ-union_R2.08.fits`
- MCXC cross-match catalog

**Download:**
```bash
# Via Planck Legacy Archive
wget https://irsa.ipac.caltech.edu/data/Planck/release_2/catalogs/
```

**Reference:** Planck Collaboration (2016), A&A, 594, A27

---

## 5. Gaia DR3 Wide Binaries (Paper 4)

**Description:** Wide binary star systems from Gaia DR3.

**Source:** https://gea.esac.esa.int/archive/

**Query:** Use ADQL to select wide binaries with appropriate separation cuts.

**Reference:** Gaia Collaboration (2023), A&A, 674, A1

---

## 6. IllustrisTNG Simulations (Papers 3, 4)

**Description:** Cosmological magnetohydrodynamical simulations.

**Source:** https://www.tng-project.org/

**Registration required:** Create an account at https://www.tng-project.org/users/register/

### TNG100-1
**Size:** ~2 GB for group catalogs

### TNG300-1
**Size:** ~5 GB for group catalogs

### TNG50-1
**Size:** ~3 GB for group catalogs

**API Key:**
```bash
export TNG_API_KEY="your-api-key-here"
```

**Reference:** Nelson et al. (2019), ComAC, 6, 2

---

## 7. WALLABY Survey (Papers 3, 4)

**Description:** ASKAP HI All-Sky Survey - Pilot Data Release 2.

**Source:** https://wallaby-survey.org/

**Files needed:**
- `WALLABY_PDR2_SourceCatalogue.xml`

**Reference:** Westmeier et al. (2022), PASA, 39, e058

---

## 8. NHANES (Paper 2)

**Description:** National Health and Nutrition Examination Survey.

**Source:** https://wwwn.cdc.gov/nchs/nhanes/

**Download via Python:**
```python
import nhanes
# Follow package documentation
```

**Reference:** CDC/NCHS

---

## 9. Additional Datasets (Paper 4)

### eROSITA eRASS1
**Description:** X-ray selected clusters
**Source:** https://erosita.mpe.mpg.de/

### redMaPPer
**Description:** Optical cluster catalog
**Source:** VizieR (Rykoff et al. 2014, 2016)

### GAMA Groups
**Description:** Galaxy group catalog
**Source:** http://www.gama-survey.org/

### Tempel Groups
**Description:** SDSS-based group catalog
**Source:** VizieR (Tempel et al. 2014)

---

## Quick Setup Script

```bash
#!/bin/bash
# setup_data.sh

# Create directories
mkdir -p Paper1-BTFR-UShape/data
mkdir -p Paper4-QOR-Validation/data/level2_multicontext

# Download SPARC (small, ~100 KB)
cd Paper1-BTFR-UShape/data
wget -q http://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt
echo "SPARC downloaded"

# For TNG/KiDS/Planck data, run individual download scripts
echo "For large datasets, see individual paper directories"
```

---

## Storage Requirements

| Dataset | Approximate Size |
|---------|-----------------|
| SPARC | 100 KB |
| ALFALFA catalog | 5 MB |
| KiDS DR4 bright sample | 500 MB |
| Planck PSZ2 | 100 MB |
| TNG100 groupcat | 2 GB |
| TNG300 groupcat | 5 GB |
| TNG50 groupcat | 3 GB |
| WALLABY DR2 | 50 MB |

---

*Last updated: December 15, 2025*
