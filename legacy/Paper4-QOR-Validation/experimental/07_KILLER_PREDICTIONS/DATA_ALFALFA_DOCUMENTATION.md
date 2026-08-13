# ALFALFA Dataset - Technical Documentation

**Document Type:** Data Provenance & Methodology  
**Author:** Jonathan Édouard Slama  
**Institution:** Metafund Research Division, Strasbourg, France  
**Date:** December 2025

---

## 1. Executive Summary

This document describes the ALFALFA dataset used for QO+R TOE validation, including data sources, acquisition methods, cross-matching procedures, and quality controls.

**Key Facts:**
- ~19,000 galaxies with both HI and stellar masses
- 100% real observational data
- Two independent measurement sources (radio + optical)
- Publicly available and reproducible

---

## 2. Primary Data Source: ALFALFA α.100

### 2.1 What is ALFALFA?

**ALFALFA** = Arecibo Legacy Fast ALFA Survey

- **Telescope:** Arecibo 305m radio telescope (Puerto Rico)
- **Instrument:** ALFA (Arecibo L-band Feed Array) - 7-beam receiver
- **Wavelength:** 21 cm (HI hyperfine transition, 1420.405 MHz)
- **Survey period:** 2005-2011
- **Sky coverage:** ~7,000 deg² (high galactic latitude)
- **Redshift range:** 0 < z < 0.06 (local universe)

### 2.2 What ALFALFA Measures

| Parameter | Description | How Measured |
|-----------|-------------|--------------|
| **M_HI** | Neutral hydrogen mass | Integrated 21cm flux × distance² |
| **W50** | Line width at 50% peak | Spectral profile fitting |
| **V_helio** | Heliocentric velocity | Doppler shift of 21cm line |
| **Distance** | Luminosity distance | From V_helio + Hubble flow model |

### 2.3 Publication & Access

**Reference:** Haynes, M. P., et al. (2018), ApJ, 861, 49  
**Title:** "The Arecibo Legacy Fast ALFA Survey: The ALFALFA Extragalactic HI Source Catalog"  
**ADS:** https://ui.adsabs.harvard.edu/abs/2018ApJ...861...49H  
**Data:** https://egg.astro.cornell.edu/alfalfa/data/

### 2.4 Catalog Statistics

- Total sources: 31,502
- Extragalactic detections: ~31,000
- Code 1 (high quality): ~25,000
- With optical counterparts: ~30,000

---

## 3. Secondary Data Source: Durbala et al. (2020)

### 3.1 Why We Need Stellar Masses

ALFALFA provides **HI masses only**. To compute:
- Baryonic mass: M_bar = M_HI + M_star
- Gas fraction: f_gas = M_HI / M_bar
- Q and R parameters for QO+R model

We need **independent stellar mass measurements**.

### 3.2 Durbala 2020 Dataset

**Reference:** Durbala, A., et al. (2020), AJ, 160, 271  
**Title:** "The ALFALFA-SDSS Galaxy Catalog"  
**ADS:** https://ui.adsabs.harvard.edu/abs/2020AJ....160..271D

This catalog provides:
| Parameter | Description | Source |
|-----------|-------------|--------|
| **logMstarTaylor** | log₁₀(M_star/M_☉) | SDSS photometry + Taylor+2011 method |
| **logMstarMPA** | log₁₀(M_star/M_☉) | MPA-JHU spectral fitting |
| **g-r, u-g colors** | Optical colors | SDSS photometry |
| **SFR** | Star formation rate | Hα emission |

### 3.3 Why Durbala 2020?

1. **Official ALFALFA-SDSS cross-match** - done by ALFALFA team
2. **AGC catalog numbers** - unique identifiers for matching
3. **Multiple M_star estimates** - can check consistency
4. **Quality flags** - reliable photometry indicators

---

## 4. Cross-Matching Procedure

### 4.1 Matching Key

Both catalogs use **AGC numbers** (Arecibo General Catalog):
- Unique identifier for each ALFALFA source
- Assigned by Cornell ALFALFA team
- No positional matching needed (exact ID match)

### 4.2 Matching Code

```python
# From prepare_alfalfa.py
merged = pd.merge(
    alfalfa_a100,      # HI data
    durbala_table2,    # Stellar masses
    on='AGCNr',        # AGC catalog number
    how='inner'        # Keep only matched sources
)
```

### 4.3 Quality Cuts Applied

| Cut | Criterion | Rationale |
|-----|-----------|-----------|
| Distance | 5 < D < 200 Mpc | Avoid local peculiar velocities, stay in Hubble flow |
| SNR | > 6.5 | ALFALFA reliability threshold |
| HI code | = 1 | High-confidence detections only |
| M_star | Valid (not NaN) | Requires SDSS counterpart |

### 4.4 Final Sample

After all cuts: **~19,000 galaxies** with:
- HI mass (ALFALFA)
- Stellar mass (SDSS via Durbala)
- Distance
- Line width (velocity proxy)

---

## 5. Derived Quantities for QO+R

### 5.1 Formulas

```
M_bar = M_HI + M_star                    # Baryonic mass
f_gas = M_HI / M_bar                     # Gas fraction
Q = log₁₀(M_HI / M_bar) = log₁₀(f_gas)   # Gas fraction proxy
R = log₁₀(M_star / M_bar)                # Stellar fraction proxy
V_proxy = W50 / 2                        # Rotation velocity proxy
```

### 5.2 BTFR Residual

```
log(V)_predicted = 0.25 × log(M_bar) + const   # MOND BTFR
residual = log(V)_observed - log(V)_predicted
```

### 5.3 Q-R Anticorrelation Check

By construction: Q + R ≈ log(f_gas) + log(1-f_gas) ≈ constant

This means Q and R are **mathematically anticorrelated**.
Expected correlation: r < -0.5 (confirmed at r = -0.81)

---

## 6. Why ALFALFA for QO+R Testing?

### 6.1 Advantages

| Feature | Benefit for QO+R |
|---------|------------------|
| **Large N (~19,000)** | High statistical power |
| **HI-selected** | Gas-rich galaxies (high Q) |
| **Independent M_star** | Avoids artificial Q-R correlations |
| **Local universe** | Minimal cosmological corrections |
| **Uniform selection** | Flux-limited, well-understood biases |

### 6.2 Comparison with SPARC

| Aspect | SPARC | ALFALFA |
|--------|-------|---------|
| N galaxies | 181 | ~19,000 |
| V measurement | Resolved rotation curves | Integrated line width |
| M_HI source | Individual observations | ALFALFA survey |
| M_star source | Spitzer 3.6μm | SDSS optical |
| Selection | Rotation curve quality | HI flux limit |

### 6.3 Complementarity

- **SPARC:** Small, high-quality rotation curves → precise V
- **ALFALFA:** Large, uniform survey → statistical power

Both are needed for robust QO+R validation.

---

## 7. Known Limitations

### 7.1 Line Width vs Rotation Velocity

W50 (line width) is NOT the same as V_flat (rotation velocity):
- W50 includes turbulence, inclination effects
- Without inclination correction: V_proxy = W50/2 is approximate
- Adds scatter to BTFR but should not bias QO+R signal

### 7.2 Selection Effects

ALFALFA is HI-flux limited:
- Preferentially detects gas-rich galaxies
- Under-represents early-types (gas-poor)
- May bias toward high-Q systems

### 7.3 Stellar Mass Uncertainties

SDSS-based stellar masses have ~0.1-0.2 dex systematic uncertainties:
- IMF assumptions
- Dust corrections
- Star formation history

---

## 8. Data Availability & Reproducibility

### 8.1 Public Data Sources

| Dataset | URL |
|---------|-----|
| ALFALFA α.100 | https://egg.astro.cornell.edu/alfalfa/data/ |
| Durbala 2020 | VizieR J/AJ/160/271 |
| SDSS DR7 | https://www.sdss.org/dr7/ |

### 8.2 Local Files

```
BTFR/Test-Replicability/data/
├── Alfalfa/
│   ├── a100.code12.table2.190808.csv    # Raw ALFALFA
│   └── durbala2020-table2.21-Sep-2020.fits.gz  # Durbala cross-match
└── alfalfa_corrected.csv                 # Final merged catalog
```

### 8.3 Processing Script

`Paper4-Redshift-Predictions/tests/prepare_alfalfa.py`

---

## 9. References

1. **Haynes, M. P., et al. (2018)**  
   "The Arecibo Legacy Fast ALFA Survey: The ALFALFA Extragalactic HI Source Catalog"  
   ApJ, 861, 49. doi:10.3847/1538-4357/aac956

2. **Durbala, A., et al. (2020)**  
   "The ALFALFA-SDSS Galaxy Catalog"  
   AJ, 160, 271. doi:10.3847/1538-3881/abc019

3. **Taylor, E. N., et al. (2011)**  
   "Galaxy And Mass Assembly (GAMA): stellar mass estimates"  
   MNRAS, 418, 1587. doi:10.1111/j.1365-2966.2011.19536.x

---

## 10. Conclusion

The ALFALFA dataset used for QO+R validation is:

✅ **100% real observational data** (radio + optical)  
✅ **Publicly available** and reproducible  
✅ **Well-documented** with known systematics  
✅ **Large sample** (~19,000 galaxies)  
✅ **Independent measurements** of M_HI and M_star  

This makes it suitable for rigorous testing of the QO+R TOE predictions.

---

**Document Version:** 1.0  
**Last Updated:** December 14, 2025
