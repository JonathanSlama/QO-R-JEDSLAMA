# Paper 4 - Data Sources

## Real Observations Only

**Author**: Jonathan Edouard Slama  
**ORCID**: [0009-0002-1292-4350](https://orcid.org/0009-0002-1292-4350)

**This paper does NOT use simulations. All data are real astronomical observations.**

---

## 1. SPARC

### Information
- **Full name**: Spitzer Photometry & Accurate Rotation Curves
- **Reference**: Lelli, McGaugh & Schombert (2016), AJ 152, 157
- **Galaxies**: 175
- **Redshift**: z ≈ 0 (local universe)
- **Type**: High-quality rotation curves

### Access
- URL: http://astroweb.cwru.edu/SPARC/
- Direct download, no registration required

### Fields Used
- `M_star`: Stellar mass (10^10 M☉)
- `MHI`: HI mass (10^10 M☉)
- `M_bar`: Baryonic mass = M_star + MHI
- `Vflat`: Flat rotation velocity (km/s)
- `log_Mbar`, `log_Vflat`: Log10 values
- `density_proxy`: Environment density estimate
- `env_class`: Environment classification (field, group, cluster)

### Derived Quantities
- `gas_fraction` = MHI / M_bar (computed in analysis)
- `log_env` = log10(density_proxy) (computed in analysis)

### Quality Assessment
Gold standard for BTFR studies - highest precision rotation curve data available

---

## 2. ALFALFA

### Information
- **Full name**: Arecibo Legacy Fast ALFA Survey
- **Reference**: Haynes et al. (2018), ApJ 861, 49
- **Sources**: ~31,500 extragalactic
- **Redshift**: z < 0.06
- **Type**: HI 21cm survey

### Access
- URL: https://egg.astro.cornell.edu/alfalfa/data/
- α.100 catalog publicly available

### Fields Used
- `MHI`: HI mass
- `W50`: Line width at 50% (Vrot proxy)
- `cz`: Recession velocity (km/s)
- `RA`, `Dec`: Position

### Quality Assessment
Large statistics - excellent for robustness testing

---

## 3. WALLABY

### Information
- **Full name**: Widefield ASKAP L-band Legacy All-sky Blind surveY
- **Reference**: Westmeier et al. (2022), PASA 39, e058
- **Galaxies**: ~5,000+ (PDR2)
- **Redshift**: z < 0.26 (DEEPEST AVAILABLE)
- **Type**: HI 21cm survey (ASKAP)

### Access
- URL: https://wallaby-survey.org/data/
- Registration required for data access

### Fields Used
- `hi_mass`: HI mass
- `w50`, `w20`: Line widths
- `z`: Redshift
- `ra`, `dec`: Position

### Quality Assessment
Only dataset enabling preliminary redshift evolution tests (up to z ~ 0.26)

---

## Environment Calculation

### Method
For each galaxy, environment is characterized by local neighbor density.

### Complementary Sources
- **2MASS Redshift Survey**: 3D positions of local galaxies
- **SDSS DR17**: Spectroscopy and photometry
- **NASA-Sloan Atlas**: Unified catalog

### Algorithm
```
1. For each target galaxy:
2.   Count neighbors within 2-5 Mpc sphere
3.   Or compute distance to 5th nearest neighbor
4.   log_env = log10(N_neighbors + 1) or log10(1/d_5nn)
```

---

## Comparative Summary

| Dataset | N | z_max | Primary Strength |
|---------|---|-------|------------------|
| SPARC | 175 | ~0 | Precision (rotation curves) |
| ALFALFA | 31,500 | 0.06 | Statistics |
| WALLABY | 5,000+ | 0.26 | Redshift depth |

---

## Data NOT Used

The following simulations are available but **not used** in this paper:
- TNG100, TNG300 (IllustrisTNG)
- EAGLE
- Illustris

**Rationale**: This work tests QO+R on the real universe, not on ΛCDM models.

---

## Reproducibility

All datasets are publicly accessible. Preparation scripts are provided in `tests/`.

To reproduce:
```bash
# 1. Download the three catalogs (see URLs above)
# 2. Place in data/observations/
# 3. Run preprocessing
python tests/test_real_data.py
```

---

*Document updated: 2024-12-06*
