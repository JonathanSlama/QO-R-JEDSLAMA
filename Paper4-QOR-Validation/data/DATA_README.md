# Data Acquisition Guide

## Paper 4 - Required Observational Data

This document explains how to obtain the datasets used in this research.
**All data used are real astronomical observations**, not simulations.

---

## 1. SPARC (Spitzer Photometry & Accurate Rotation Curves)

### Description
- **Type**: Real observations
- **Content**: 175 galaxies with high-quality rotation curves
- **Redshift**: z ≈ 0 (local universe)
- **Reference**: Lelli, McGaugh & Schombert (2016), AJ 152, 157

### Download
1. Navigate to: http://astroweb.cwru.edu/SPARC/
2. Download `SPARC_Lelli2016c.mrt` (main table)
3. Place in `data/observations/SPARC/`

### Fields Used (in processed file)
- `M_star`: Stellar mass (10^10 M☉)
- `MHI`: HI mass (10^10 M☉)
- `M_bar`: Baryonic mass = M_star + MHI
- `Vflat`: Flat rotation velocity (km/s)
- `log_Mbar`, `log_Vflat`: Log10 values
- `density_proxy`: Environment density estimate
- `env_class`: Environment classification (field, group, cluster)

### Derived in Analysis
- `gas_fraction` = MHI / M_bar
- `log_env` = log10(density_proxy)

---

## 2. ALFALFA (Arecibo Legacy Fast ALFA Survey)

### Description
- **Type**: Real observations (21cm radio)
- **Content**: ~31,500 extragalactic HI sources
- **Redshift**: z < 0.06 (primarily z ≈ 0)
- **Reference**: Haynes et al. (2018), ApJ 861, 49

### Download
1. Navigate to: https://egg.astro.cornell.edu/alfalfa/data/
2. Download the α.100 catalog
3. Place in `data/observations/ALFALFA/`

### Fields Used
- HI mass (M_HI)
- Line width W50 (velocity proxy)
- Position and redshift
- Integrated flux

---

## 3. WALLABY (Widefield ASKAP L-band Legacy All-sky Blind surveY)

### Description
- **Type**: Real observations (21cm radio, ASKAP)
- **Content**: HI galaxies in the southern hemisphere
- **Redshift**: z < 0.26 (DEEPEST IN REDSHIFT)
- **Reference**: Westmeier et al. (2022), PASA 39, e058

### Download
1. Navigate to: https://wallaby-survey.org/data/
2. Register for data access
3. Download PDR2 SourceCatalogue
4. Place in `data/observations/WALLABY/`

### Fields Used
- HI mass
- Line widths W50, W20
- Redshift (up to z ~ 0.26)
- Position and environment proxies

### Importance for Paper 4
WALLABY is our primary dataset for testing λ_QR(z) evolution as it reaches z ~ 0.26, enabling preliminary tests of redshift dependence on REAL observational data.

---

## 4. Environment Data

### Complementary Sources
To characterize galactic environment (local density), we use:

- **2MASS Redshift Survey (2MRS)**: 3D positions of nearby galaxies
- **SDSS DR17**: Spectroscopy and photometry
- **NASA-Sloan Atlas**: Local galaxy catalog

### Environment Calculation
Environment is computed as neighbor density within a 2-5 Mpc sphere, or via the 5th nearest neighbor distance (standard method).

---

## Data Structure

```
data/
├── observations/
│   ├── SPARC/
│   │   └── SPARC_Lelli2016c.mrt
│   ├── ALFALFA/
│   │   └── a100.csv
│   └── WALLABY/
│       └── WALLABY_PDR2_SourceCatalogue.xml
│
├── processed/
│   ├── sparc_with_environment.csv
│   ├── alfalfa_with_environment.csv
│   └── wallaby_with_environment.csv
│
└── DATA_README.md (this file)
```

---

## Reproducibility

To reproduce the analysis:

1. Download the three catalogs listed above
2. Run `scripts/01_prepare_data.py` for preprocessing
3. Run `scripts/02_add_environment.py` to add density estimates
4. The `*_with_environment.csv` files are ready for analysis

---

## Note on Simulations

This Paper 4 uses **exclusively observational data**.

Simulations (TNG, EAGLE, etc.) are NOT used for primary results.
All conclusions rest on real observations of the actual universe.

---

## References

1. Lelli F., McGaugh S.S., Schombert J.M., 2016, AJ, 152, 157 (SPARC)
2. Haynes M.P. et al., 2018, ApJ, 861, 49 (ALFALFA α.100)
3. Westmeier T. et al., 2022, PASA, 39, e058 (WALLABY PDR2)

---

**Author**: Jonathan Édouard Slama
**ORCID**: [0009-0002-1292-4350](https://orcid.org/0009-0002-1292-4350)

*Document created: 2024-12-06*
