# Level 2 Multi-Context Tests - Erratum & Lessons Learned

**Date:** December 7, 2025
**Author:** Jonathan Édouard Slama
**Status:** Correction in progress

---

## Summary of Identified Errors

### KiDS Test (v1) - Methodological Failure

#### Error 1: Incorrect Definition of Mass-Luminosity Relation

```
Observed:  M-L slope: 0.017 (POSITIVE)
Expected:  M-L slope: ~ -2.5 (NEGATIVE)
```

**Cause:** The v1 script regressed **apparent** magnitude (`MAG_AUTO_calib`) against stellar mass.

**Problem:** Apparent magnitude depends on distance! Distant galaxies (high z) are:
- Fainter (larger apparent magnitude)
- AND more massive (Malmquist selection bias)

This creates an **artificial positive correlation**, not the true M-L relation.

**Solution (v2):** Use **absolute** magnitude calculated via:
```python
M_r = mag_app - 5 * log10(d_L / 10 pc)
```
where `d_L` is the luminosity distance derived from photo-z.

#### Error 2: g-r Color Proxy Not Extracted

The v1 script used a default value `g_r_color = 0.6` for all galaxies instead of extracting true colors from the LePhare catalog (`MAG_ABS_g - MAG_ABS_r`).

**Consequence:** No discrimination between Q-dom vs R-dom possible.

---

### Planck Test (v1) - Column Reading Error

#### Error 3: MSZ EXISTS but wasn't being used!

```
v1 displayed:  Available columns: [...] 'Y5R500'... (truncated!)
Reality:       30 columns including MSZ, MSZ_ERR_UP, MSZ_ERR_LOW
```

**Cause:** The v1 script only displayed the first 15 columns, hiding MSZ which is at position 20!

**Complete PSZ2 Union columns:**
```
INDEX, NAME, GLON, GLAT, RA, DEC, POS_ERR, SNR, PIPELINE, PIPE_DET,
PCCS2, PSZ, IR_FLAG, Q_NEURAL, Y5R500, Y5R500_ERR, VALIDATION,
REDSHIFT_ID, REDSHIFT, MSZ, MSZ_ERR_UP, MSZ_ERR_LOW, MCXC, REDMAPPER,
ACT, SPT, WISE_FLAG, AMI_EVIDENCE, COSMO, COMMENT
```

**Solution:** Use `data['MSZ']` directly (in 10^14 Msun)

### Planck Test (v2) - Fundamental Problem Identified

#### Observation

```
Slope M-Y: -0.373 (expected +1.67)
Correlation: r = -0.196 (expected ~ +0.8)
```

#### Diagnosis (diagnose_planck_data.py)

**Initial hypothesis (REFUTED):** MSZ derived from Y5R500

If this were the case, we would have r > 0.9. But r = -0.196 → quasi-independent!

**True problem:** The M-Y relation simply does NOT exist in this data.

Examples of huge scatter:
```
PSZ2 G000.77-35.69:  M=6.33, Y=1.61  (massive, low Y)
PSZ2 G002.82+39.23:  M=5.74, Y=9.70  (massive, high Y)
```
Same mass, Y differs by factor of 6!

#### Probable Causes

1. **Selection bias in z**: At high z, only certain types are detected
2. **Heterogeneous MSZ sources**: Mix of SZ, X-ray, optical masses
3. **Intrinsic scatter dominates**

#### Conclusion

**M-Y test impossible with PSZ2 alone**

The base relation doesn't exist in the data → impossible to analyze residuals.

#### Alternative Solutions

1. Cross-match with MCXC (hydrostatic X-ray masses) → **IMPLEMENTED!**
2. Use weak lensing masses (DES, KiDS WL)
3. Test Y alone vs environment (without mass)
4. Wait for eROSITA (better catalog coming)

---

### Implemented Solution: PSZ2 × MCXC Cross-match

#### Why MCXC?

MCXC = Meta-Catalogue of X-ray Clusters (Piffaretti+2011)
- **1743 X-ray clusters**
- **M_500 derived from L_X** (X-ray luminosity)
- **Totally independent from SZ signal!**

#### Cross-match Result

- **551 PSZ2 clusters** have an MCXC match
- **But r = -0.13** (still negative!)

#### New Problem: Angular Distance Bias

The observed SZ signal (Y_obs in arcmin²) depends on distance:

```
Y_obs [arcmin²] = Y_true [Mpc²] / D_A²
```

Combined with Malmquist bias (massive clusters at high z), this creates an **artificial anti-correlation**!

#### Solution: Distance Correction

```python
Y_scaled = Y_obs × D_A² / E(z)^(2/3)
```

Where E(z) = H(z)/H_0 is the cosmological evolution factor.

#### Scripts

1. `test_cluster_planck_mcxc.py` - Without correction → r = -0.13
2. `test_cluster_planck_mcxc_v2.py` - With correction → r > +0.5 (expected)

---

## What v1 Results DO NOT Mean

**NO:** The U-shape doesn't exist in other contexts
**NO:** String Theory is refuted
**NO:** Our 14/14 tests on rotation curves are invalid

## What v1 Results DO Mean

**YES:** The scripts had methodological bugs
**YES:** Variable extraction needs to be reviewed
**YES:** Rigorous validation requires well-understood data

---

## Correction Plan

### KiDS - Version 2

| Variable | v1 (WRONG) | v2 (CORRECTED) |
|----------|-----------|----------------|
| Luminosity | `MAG_AUTO_calib` (apparent) | `M_r` (absolute, calculated) |
| g-r Color | Fixed value 0.6 | `MAG_ABS_g - MAG_ABS_r` (LePhare) |
| Relation | `linregress(log_M*, mag_app)` | `linregress(M_r_abs, log_M*)` |

### Planck - Version 2

**Immediate action:** Explore ALL .fits files in Planck catalog to find:
1. A direct MSZ column
2. Or necessary parameters to calculate M_500 correctly

**Files to examine:**
```
COM_PCCS_SZ-Catalogs_vPR2/
├── HFI_PCCS_SZ-MMF1_R2.08.fits.gz      ← MMF1 method
├── HFI_PCCS_SZ-MMF3_R2.08.fits.gz      ← MMF3 method
├── HFI_PCCS_SZ-PwS_R2.08.fits.gz       ← PwS method
├── HFI_PCCS_SZ-union_R2.08.fits.gz     ← Union (used in v1)
└── HFI_PCCS_SZ-selfunc-*.fits          ← Selection functions
```

---

## Lessons Learned

1. **Always verify physical consistency**: A positive M-L slope or negative M-Y is an immediate red flag.

2. **Document available columns BEFORE coding**: List exactly what each file contains.

3. **Apparent magnitude ≠ intrinsic luminosity**: Classic error in observational astrophysics.

4. **SZ catalogs don't always provide masses**: Y_SZ is an observable, M_500 is derived.

---

## Next Steps

1. [x] Explore all Planck files to find MSZ → **FOUND!**
2. [x] Create test_lensing_kids_v2.py with corrections → **SUCCESS!**
3. [x] Create test_cluster_planck_v2.py with MSZ
4. [x] Diagnosis: M-Y relation nonexistent in PSZ2
5. [x] Solution: cross-match with MCXC (X-ray masses) → **551 matches**
6. [x] MCXC test v1: r = -0.13 (distance bias identified)
7. [ ] MCXC test v2: with D_A² correction (in progress...)
8. [ ] Document final results

---

**Author:** Jonathan Édouard Slama
**Date:** December 7, 2025
