# UDG DATA SOURCES AND ACQUISITION GUIDE
## For QO+R Killer Prediction #1: UDG Inverted U-Shape Test

**Author:** Jonathan Édouard Slama  
**Institution:** Metafund Research Division, Strasbourg, France  
**Date:** December 2025  
**Document Type:** Data Documentation

---

## 1. Overview

Ultra-Diffuse Galaxies (UDGs) are critical for testing the QO+R framework because:
- They are **gas-poor** (R-dominated systems)
- QO+R predicts **inverted U-shape** (a < 0) for R-dominated environments
- They span diverse environments (field, groups, clusters)

This document describes all data sources and how to acquire them.

---

## 2. Downloaded Catalogs

### 2.1 SMUDGes (Zaritsky et al. 2019-2023)

**VizieR ID:** `J/ApJS/257/60`  
**Reference:** Zaritsky et al. 2023, ApJS, 257, 60  
**URL:** https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJS/257/60

| Property | Value |
|----------|-------|
| N objects | 226 |
| Coverage | SDSS Stripe 82 |
| Key columns | RAJ2000, DEJ2000, Re (arcsec), mu0g, mu0r, g-r color |
| Velocities | NO |
| Environment | NO (must cross-match) |

**File:** `data/udg_catalogs/SMUDGes_t0.fits`

### 2.2 MATLAS Dwarfs (Poulain et al. 2021)

**VizieR ID:** `J/MNRAS/506/5494`  
**Reference:** Poulain et al. 2021, MNRAS, 506, 5494  
**URL:** https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/MNRAS/506/5494

| Property | Value |
|----------|-------|
| N objects | 2210 dwarfs + 524 nuclei |
| Coverage | MATLAS survey fields |
| Key columns | RA, DEC, Re, mu_0, g-i color, **Distance**, Host galaxy |
| Velocities | 13.5% have spectroscopic z |
| Environment | YES (Host ETG association) |

**Files:** 
- `data/udg_catalogs/MATLAS_dwarfs_t0.fits` (2210 dwarfs)
- `data/udg_catalogs/MATLAS_dwarfs_t1.fits` (524 nuclei)

### 2.3 MATLAS UDGs (Marleau et al. 2021)

**VizieR ID:** `J/A+A/654/A105`  
**Reference:** Marleau et al. 2021, A&A, 654, A105  
**URL:** https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/654/A105

| Property | Value |
|----------|-------|
| N objects | 59 UDGs (×2 tables) |
| Coverage | MATLAS survey |
| Key columns | RA, DEC, Re, mu_e, **log(M_star)**, Distance |
| Velocities | Partial |
| Environment | YES (Host association) |

**Files:**
- `data/udg_catalogs/MATLAS_UDG_t0.fits` (59 UDGs)
- `data/udg_catalogs/MATLAS_UDG_t1.fits` (59 UDGs, different parameters)

---

## 3. Catalogs Requiring Manual Download

### 3.1 Gannon et al. 2024 - SPECTROSCOPIC UDG CATALOG (CRITICAL)

**Reference:** Gannon et al. 2024, MNRAS, 531, 1856  
**arXiv:** https://arxiv.org/abs/2405.09104  
**DOI:** https://doi.org/10.1093/mnras/stae1287

**THIS IS THE MOST IMPORTANT CATALOG** because it contains:
- Recessional velocities
- Stellar velocity dispersions
- **Environment classification** (1=cluster, 2=group, 3=field)
- Stellar masses
- Ages and metallicities
- GC counts

| Property | Value |
|----------|-------|
| N objects | ~60 UDGs |
| Key columns | RA, DEC, V_recession, sigma_star, Environment, M_star, Age, [M/H] |
| Velocities | **YES** |
| Environment | **YES** (explicit) |

**How to obtain:**

**Option A: MNRAS Supplementary Material**
1. Go to https://academic.oup.com/mnras/article/531/1/1856/7676195
2. Scroll to "Supplementary data" section at bottom
3. Download Table 1 (full UDG catalogue)
4. Save as `data/udg_catalogs/Gannon2024_spectroscopic.csv`

**Option B: arXiv Source**
1. Go to https://arxiv.org/abs/2405.09104
2. Click "Other formats" → "Download source"
3. Extract the .tar.gz file
4. Look for table files (usually .tex or .dat)

**Option C: Contact Authors**
- Lead author: Jonah Gannon (Swinburne University)
- Email: jonah.gannon@swinburne.edu.au
- The paper states: "Our catalogue is available online, and we aim to maintain it beyond the publication of this work."

**Expected Columns:**
- Name, RA, DEC
- Environment (1=cluster, 2=group, 3=field)
- Distance (Mpc)
- M_V (absolute magnitude)
- <mu>_e (surface brightness)
- log(M_star) (stellar mass)
- R_e (effective radius, kpc)
- V_rec (recessional velocity, km/s)
- sigma_star (stellar velocity dispersion, km/s)
- N_GC (globular cluster count)
- Age (Gyr), [M/H] (metallicity)

### 3.2 van Dokkum UDGs (DF2, DF4, etc.)

**References:** 
- van Dokkum et al. 2018, Nature, 555, 629 (DF2)
- van Dokkum et al. 2019, ApJL, 874, L5 (DF4)

**Key data:** Individual UDGs with detailed kinematics (very low velocity dispersion).

---

## 4. Environment Cross-Match Catalogs

To compute environmental density for UDGs without explicit environment data:

### 4.1 Galaxy Groups (Already Downloaded)

**Location:** `data/level2_multicontext/groups/`

| Catalog | N groups | Reference |
|---------|----------|-----------|
| GAMA groups | ~24000 | Robotham et al. |
| Tempel groups | ~70000 | Tempel et al. 2014 |
| Saulder 2MRS | ~5000 | Saulder et al. |

### 4.2 Galaxy Clusters (Already Downloaded)

**Location:** `data/level2_multicontext/clusters/`

| Catalog | N clusters | Reference |
|---------|------------|-----------|
| Planck SZ | ~1600 | Planck Collaboration |
| redMaPPer | ~26000 | Rykoff et al. |
| eRASS1 | ~12000 | Bulbul et al. 2024 |

### 4.3 Cross-Match Strategy

```python
# Pseudo-code for environment assignment
for udg in UDG_catalog:
    # 1. Check cluster membership (R < R_200)
    cluster_match = crossmatch(udg.ra, udg.dec, clusters, radius=R200)
    
    # 2. Check group membership
    group_match = crossmatch(udg.ra, udg.dec, groups, radius=R_vir)
    
    # 3. Compute local density
    n_neighbors = count_neighbors(udg, galaxy_catalog, radius=1Mpc)
    
    # 4. Assign environment
    if cluster_match:
        env = 'cluster'
        density = high
    elif group_match:
        env = 'group'
        density = medium
    else:
        env = 'field'
        density = low
```

---

## 5. Data Quality Assessment

### 5.1 What We Have

| Requirement | SMUDGes | MATLAS | Gannon |
|-------------|---------|--------|--------|
| Position | ✓ | ✓ | ✓ |
| Size (R_eff) | ✓ | ✓ | ✓ |
| Surface brightness | ✓ | ✓ | ✓ |
| Velocity | ✗ | Partial | **✓** |
| Environment | ✗ | Host-based | **✓ Explicit** |
| Stellar mass | ✗ | ✓ | ✓ |

### 5.2 Recommended Strategy

1. **Primary sample:** Gannon+2024 (once downloaded)
   - Has velocities AND environment
   - Can directly compute BTFR residuals
   - Test U-shape vs environment

2. **Extended sample:** MATLAS UDGs
   - Use host ETG distance
   - Derive environment from host
   - Larger sample but less precise

3. **Photometric sample:** SMUDGes + MATLAS dwarfs
   - Cross-match with environment catalogs
   - Statistical environmental density
   - Cannot compute individual BTFR

---

## 6. Test Implementation

### 6.1 Killer Prediction #1: UDG Inverted U-Shape

**Prediction:** For UDGs (R-dominated), the BTFR residual vs environment should show:
$$a < 0 \quad \text{(inverted U-shape)}$$

**Test script:** `tests/future/killer_prediction_1_udg.py`

**Falsification criterion:** If a > 0 at 3σ significance, QO+R is falsified.

### 6.2 Required Data Flow

```
Gannon2024 catalog
    │
    ├── Extract: RA, DEC, V_recession, sigma_star, Environment
    │
    ├── Compute: M_baryonic = M_star (UDGs are gas-poor)
    │
    ├── Compute: V_circ from sigma_star (or use V_recession for groups)
    │
    ├── Compute: BTFR prediction: V_pred = (G × M_bar × A0)^0.25
    │
    ├── Compute: Residual = log(V_obs) - log(V_pred)
    │
    ├── Fit: Residual = a × env² + b × env + c
    │
    └── Test: Is a < 0 at 3σ?
```

---

## 7. File Locations Summary

```
Paper4-Redshift-Predictions/
├── data/
│   └── udg_catalogs/
│       ├── SMUDGes_t0.fits              # 226 UDGs, photometry only
│       ├── MATLAS_dwarfs_t0.fits        # 2210 dwarfs
│       ├── MATLAS_dwarfs_t1.fits        # 524 nuclei
│       ├── MATLAS_UDG_t0.fits           # 59 UDGs with masses
│       ├── MATLAS_UDG_t1.fits           # 59 UDGs alternate params
│       ├── Gannon2024_TEMPLATE.fits     # Template (need real data)
│       └── [Gannon2024_spectroscopic.csv]  # TO DOWNLOAD MANUALLY
├── scripts/
│   ├── download_udg_catalogs.py         # Download script
│   ├── inspect_all_udg.py               # Inspection script
│   └── inspect_smudges.py               # SMUDGes details
└── tests/future/
    └── killer_prediction_1_udg.py       # Test script
```

---

## 8. References

1. Zaritsky D. et al., 2023, ApJS, 257, 60 (SMUDGes)
2. Poulain M. et al., 2021, MNRAS, 506, 5494 (MATLAS dwarfs)
3. Marleau F. et al., 2021, A&A, 654, A105 (MATLAS UDGs)
4. Gannon J. et al., 2024, MNRAS, 531, 1856 (Spectroscopic UDGs)
5. van Dokkum P. et al., 2018, Nature, 555, 629 (DF2)

---

**Author:** Jonathan Édouard Slama  
**Metafund Research Division, Strasbourg, France**
