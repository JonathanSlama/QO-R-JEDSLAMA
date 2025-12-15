# Level 2: Multi-Context Data Download Guide

**Updated:** December 7, 2025
**Status:** Ready for testing

---

## Data Summary

| Context | Dataset | Files | Status |
|---------|---------|-------|--------|
| **Optical Galaxies** | KiDS DR4 Bright | 2 files (331 MB total) | Downloaded |
| **Galaxy Clusters** | Planck PSZ2 | 1 file (in .tgz archive) | Need to extract |

---

## 1. KiDS DR4 Bright Sample - DONE

### Download Page
```
https://kids.strw.leidenuniv.nl/DR4/brightsample.php
```

### Files Downloaded
| File | Size | Content |
|------|------|---------|
| `KiDS_DR4_brightsample.fits` | 85 MB | Positions, photo-z, magnitudes |
| `KiDS_DR4_brightsample_LePhare.fits` | 246 MB | Stellar masses (MASS_MED), SFR |

### Key Columns
**brightsample.fits:**
- `ID` - Source identifier (ESO ID)
- `RAJ2000`, `DECJ2000` - J2000 coordinates
- `MAG_AUTO_calib` - r-band calibrated magnitude
- `zphot_ANNz2` - Photometric redshift (ANNz2)
- `masked` - Quality flag (0 = good)

**LePhare.fits:**
- `MASS_MED` - Median stellar mass: log₁₀(M★/M☉)
- `MASS_BEST` - Best-fit stellar mass
- `MAG_ABS_x` - Absolute magnitudes per band
- `REDSHIFT` - Redshift used for SED fitting

### Location
```
data/level2_multicontext/kids/
├── KiDS_DR4_brightsample.fits         [Downloaded]
└── KiDS_DR4_brightsample_LePhare.fits [Downloaded]
```

---

## 2. Planck SZ Galaxy Clusters - NEEDS EXTRACTION

### Download Page
```
https://irsa.ipac.caltech.edu
→ Holdings → Planck → Sunyaev-Zeldovich [SZ] Catalog
```

### Downloaded Archive
- `COM_PCCS_SZ-Catalogs_vPR2.tgz` (782 MB compressed)

### EXTRACTION REQUIRED
The .tgz file is a compressed archive. You need to extract it:

**Windows:**
1. Right-click on `COM_PCCS_SZ-Catalogs_vPR2.tgz`
2. Select "Extract All..." or use 7-Zip/WinRAR
3. Extract to the same folder

**After extraction, look for:**
- `HFI_PCCS_SZ-union_R2.08.fits` - This is the main catalog we need

### Key Columns (PSZ2 Union Catalog)
- `NAME` - Cluster name
- `GLON`, `GLAT` - Galactic coordinates
- `REDSHIFT` - Spectroscopic/photometric redshift
- `MSZ` - SZ mass (×10¹⁴ M☉)
- `Y5R500` - Integrated Compton Y parameter
- `SNR` - Signal-to-noise ratio

### Location After Extraction
```
data/level2_multicontext/clusters/
├── COM_PCCS_SZ-Catalogs_vPR2.tgz      (archive)
└── [extracted folder]/
    └── HFI_PCCS_SZ-union_R2.08.fits   (This is what we need)
```

The test script will automatically search for the file in subdirectories.

---

## Running the Tests

```bash
cd Paper4-Redshift-Predictions/tests/future

# Test 1: KiDS optical galaxies (~1 million galaxies)
python test_lensing_kids.py

# Test 2: Planck clusters (~1,600 clusters)
python test_cluster_planck.py
```

---

## What We're Testing

### Scientific Goal
Find the SAME Q²R² U-shape pattern in multiple independent contexts:

| Context | Q (gas proxy) | R (stellar proxy) | Scaling Relation |
|---------|---------------|-------------------|------------------|
| **Rotation curves** | f_gas (HI 21cm) | log M★ | BTFR |
| **KiDS optical** | g-r color → f_gas | MASS_MED | Mass-Luminosity |
| **Planck clusters** | Y_SZ / M | BCG mass | M-Y relation |

### Success Criteria
- **U-shape detected** (a > 0, p < 0.1)
- **Sign inversion** between Q-dominated and R-dominated samples
- **Consistent with** rotation curve results

### Implications
If the SAME pattern appears in 3 independent contexts:
- **Probability of coincidence:** < 10⁻⁶
- **Interpretation:** Strong evidence for universal Q²R² coupling (String Theory moduli)

---

## Troubleshooting

### "File not found" errors
1. Check that files are in the correct folders
2. For Planck: extract the .tgz archive first
3. The scripts search recursively, so exact subfolder doesn't matter

### KiDS: "MASS_MED = -99"
This is normal - these are non-galaxy sources (stars). The script filters them out automatically.

### Planck: Small sample warning
The PSZ2 catalog has ~1,600 clusters. After quality cuts, ~500-1,000 remain.
This is smaller than ALFALFA but sufficient for U-shape detection.

---

## Next Steps After Testing

1. **If U-shape detected in both:**
   - Document as Level 2 validation complete
   - Proceed to CMB lensing correlation (Level 2c)
   - Update roadmap to ~60% progress

2. **If U-shape in one context only:**
   - Investigate differences
   - May indicate context-dependent coupling
   - Still valuable scientific result

3. **If no U-shape detected:**
   - Check methodology
   - May need better environment proxy
   - Does not invalidate rotation curve results

---

*Guide updated: December 7, 2025*
