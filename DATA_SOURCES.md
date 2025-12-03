# Data Sources and Download Instructions

This document explains how to obtain the large datasets used in the QO+R project. These files are excluded from Git due to their size.

**Author:** Jonathan Edouard SLAMA  
**Contact:** jonathan@metafund.in

---

## 1. SPARC Database (Primary Dataset)

**Description:** Spitzer Photometry and Accurate Rotation Curves - 175 galaxies with high-quality rotation curves.

**Source:** http://astroweb.cwru.edu/SPARC/

**Files needed:**
- `SPARC_Lelli2016c.mrt` (main catalog)

**Download:**
```bash
cd Shared/data/
wget http://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt
```

**Reference:** Lelli, McGaugh, Schombert (2016), AJ, 152, 157

---

## 2. IllustrisTNG Simulations

**Description:** Cosmological magnetohydrodynamical simulations of galaxy formation.

**Source:** https://www.tng-project.org/

**Registration required:** Create an account at https://www.tng-project.org/users/register/

### TNG100-1 (Primary)
**Size:** ~2 GB for group catalogs

```bash
cd Paper3-ToE/tests/tng-simulations/
python download_tng_groupcat.py
```

### TNG300-1 (Large volume)
**Size:** ~5 GB for group catalogs

```bash
python download_tng300.py
```

### TNG50-1 (High resolution)
**Size:** ~3 GB for group catalogs

```bash
python download_tng50.py
```

**API Key:** Set your API key in environment variable:
```bash
export TNG_API_KEY="your-api-key-here"
```

**Reference:** Nelson et al. (2019), ComAC, 6, 2

---

## 3. ALFALFA Survey

**Description:** Arecibo Legacy Fast ALFA survey - HI 21cm observations.

**Source:** http://egg.astro.cornell.edu/alfalfa/data/

**Files needed:**
- `a100.code12.table2.190808.csv`

**Download:**
```bash
cd Paper1-BTFR-UShape/tests/replicability/data/
wget http://egg.astro.cornell.edu/alfalfa/data/a100.code12.table2.190808.csv
```

**Reference:** Haynes et al. (2018), ApJ, 861, 49

---

## 4. WALLABY Survey (DR2)

**Description:** ASKAP HI All-Sky Survey - Pilot Data Release 2.

**Source:** https://wallaby-survey.org/

**Files needed:**
- `WALLABY_PDR2_SourceCatalogue.xml`

**Download script:**
```bash
python download_wallaby.py
```

**Reference:** Westmeier et al. (2022), PASA, 39, e058

---

## 5. Little THINGS

**Description:** Local Irregulars That Trace Luminosity Extremes - THINGS survey extension.

**Source:** https://science.nrao.edu/science/surveys/littlethings

**Reference:** Hunter et al. (2012), AJ, 144, 134

---

## Quick Setup Script

Create all data directories and download small datasets:

```bash
#!/bin/bash
# setup_data.sh

# Create directories
mkdir -p Shared/data
mkdir -p Paper1-BTFR-UShape/tests/replicability/data
mkdir -p Paper3-ToE/tests/tng-simulations/Data

# Download SPARC (small, ~100 KB)
cd Shared/data
wget -q http://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt
echo "SPARC downloaded"

# For TNG data, run the Python scripts with your API key
echo "For TNG data, set TNG_API_KEY and run download scripts"
```

---

## Data Checksums

To verify data integrity after download:

| File | MD5 Checksum |
|------|--------------|
| SPARC_Lelli2016c.mrt | (compute after download) |
| TNG100 groupcat | (compute after download) |

Generate checksums:
```bash
md5sum Shared/data/* > checksums.md5
```

Verify:
```bash
md5sum -c checksums.md5
```

---

## Storage Requirements

| Dataset | Approximate Size |
|---------|-----------------|
| SPARC | 100 KB |
| ALFALFA catalog | 5 MB |
| TNG100 groupcat | 2 GB |
| TNG300 groupcat | 5 GB |
| TNG50 groupcat | 3 GB |
| WALLABY DR2 | 50 MB |

**Total (all datasets):** ~10 GB

**Minimum (SPARC + ALFALFA only):** ~5 MB
