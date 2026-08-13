# KiDS DR4 Bright Sample - Variable Review

**Date:** December 7, 2025
**Objective:** Correctly define variables for Q²R² test

---

## Available Catalogs

### 1. KiDS_DR4_brightsample.fits (85 MB)

| Column | Description | Usage |
|--------|-------------|-------|
| `ID` | Unique identifier | Join |
| `RAJ2000` | Right ascension (deg) | Position |
| `DECJ2000` | Declination (deg) | Position |
| `MAG_AUTO_calib` | **Apparent** r magnitude calibrated | NOT for direct M-L |
| `MAGERR_AUTO` | Magnitude error | Quality |
| `zphot_ANNz2` | Photo-z (ANNz2) | Distance |
| `MASK` | Mask flag | Quality |
| `masked` | Boolean masked (0/1) | Quality |

### 2. KiDS_DR4_brightsample_LePhare.fits (246 MB)

| Column | Description | Usage |
|--------|-------------|-------|
| `MASS_MED` | log10(M_star/Msun) median | **R (stellar proxy)** |
| `MASS_BEST` | log10(M_star/Msun) best-fit | Alternative |
| `MAG_ABS_g` | Absolute magnitude g | Color |
| `MAG_ABS_r` | Absolute magnitude r | **Luminosity** |
| `MAG_ABS_i` | Absolute magnitude i | - |
| `K_COR_*` | K-corrections per band | - |
| `REDSHIFT` | LePhare redshift | Verification |
| `CONTEXT` | Template used | Quality |

**Note:** `MASS_MED = -99` indicates a failed fit (non-galaxy template).

---

## Variables for Q²R² Test

### Variable Q (gas/cold baryon proxy)

**Choice:** Rest-frame g-r color

```python
g_r_color = MAG_ABS_g - MAG_ABS_r
```

**Mapping to f_gas:**
```python
# Blue (g-r ~ 0.3) → f_gas ~ 0.5 (gas-rich, star-forming)
# Red (g-r ~ 0.8) → f_gas ~ 0.1 (gas-poor, quiescent)
f_gas = 0.55 / (1 + exp(8 * (g_r - 0.55))) + 0.05
```

**Physical justification:** Blue galaxies have more gas (HI + H2) because they actively form stars. This color-gas correlation is well established (Catinella+2010, Saintonge+2011).

### Variable R (stellar/dark matter proxy)

**Choice:** Stellar mass

```python
log_Mstar = MASS_MED  # log10(M*/Msun)
```

**R-dominated selection criteria:**
- `g_r_color > 0.7` (red)
- `log_Mstar > 10.5` (massive)
- `f_gas < 0.15`

### Scaling Relation: Mass-Luminosity

**Correct definition:**
```python
# Absolute M_r from LePhare (already K-corrected)
M_r = MAG_ABS_r

# Expected relation: more massive galaxies = more luminous
# log(M*) = α * M_r + β
# with α ~ -0.4 (negative slope because more negative M_r = more luminous)
```

**Residuals:**
```python
log_Mstar_predicted = slope * M_r + intercept
mass_residual = log_Mstar - log_Mstar_predicted
```

Positive residual = **over-massive** galaxy for its luminosity (high M/L)
Negative residual = **under-massive** galaxy for its luminosity (low M/L)

### Environment Variable

**Method:** Local density via grid

```python
# Count neighbors in RA-DEC grid
# Normalize as z-score
env_density = (count - mean) / std
```

**Limitation:** Doesn't account for depth (redshift). For better estimation, would need 3D counting in redshift slices.

---

## Q²R² Prediction for KiDS

### If String Theory is correct:

1. **U-shape in M/L residuals vs environment**
   - Extreme residuals (very positive or very negative) in extreme environments
   - Minimum at center (average environment)

2. **Sign inversion between Q-dom and R-dom**
   - **Blue** galaxies (Q-dom): M/L residuals **decrease** toward high density
   - **Red massive** galaxies (R-dom): M/L residuals **increase** toward high density

### If String Theory is incorrect:

- No significant U-shape
- Or U-shape without sign inversion
- Or simple monotonic correlation

---

## Recommended Quality Cuts

```python
mask = (
    (masked == 0) &                          # Not masked
    (zphot_ANNz2 > 0.02) & (zphot_ANNz2 < 0.4) &  # Reliable photo-z
    (MAG_AUTO_calib > 14) & (MAG_AUTO_calib < 20) &  # Not saturated, detected
    (MAG_ABS_r > -24) & (MAG_ABS_r < -16) &  # Reasonable luminosity
    (MASS_MED > 7) & (MASS_MED < 12) &       # Reasonable mass
    (MASS_MED > 0) &                          # Exclude failed fits (-99)
    (g_r_color > -0.5) & (g_r_color < 1.5)   # Physical color
)
```

---

## Expected v2 Test Result

| Metric | Expected value | Error sign |
|--------|----------------|------------|
| M-L slope | ~ -0.4 | Positive → error |
| Correlation r | ~ -0.7 | Positive → error |
| N after cuts | > 100,000 | < 10,000 → problem |
| f_gas range | 0.05 - 0.50 | Constant → color error |

---

**Author:** Jonathan Édouard Slama
**Date:** December 7, 2025
