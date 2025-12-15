# Solution for Planck: Cross-match with MCXC

**Date:** December 7, 2025
**Problem:** The M-Y relation does not exist in PSZ2 (huge scatter)
**Solution:** Use MCXC for independent X-ray masses

---

## Why MCXC?

### The problem with PSZ2 alone
- MSZ and Y5R500 have almost no correlation (r = -0.20)
- Same mass yields Y differing by a factor of 6
- Impossible to analyze M-Y residuals

### Why MCXC solves the problem

| Catalogue | Observable | Mass derived from |
|-----------|------------|-------------------|
| PSZ2 | Y_SZ (SZ signal) | Y_SZ (circular!) |
| MCXC | L_X (X-ray luminosity) | L_X (independent!) |

**MCXC masses are derived from L_X via the L_X-M relation**, which is **completely independent** of the SZ signal.

---

## MCXC: Meta-Catalogue of X-ray Clusters

### Characteristics
- **N clusters:** 1743 (MCXC original) or 2221 (MCXC-II 2024)
- **Source:** ROSAT All-Sky Survey (NORAS, REFLEX, MACS, etc.)
- **Key columns:**
  - `RA`, `DEC` - Position
  - `z` - Redshift
  - `Lx_500` - X-ray luminosity [0.1-2.4 keV]
  - `Mass_500` - Total mass (10^14 Msun)
  - `Radius_500` - Radius R500 (Mpc)

### L_X - M relation used
```
M_500 = f(L_X, z) via the REXCESS relation (Pratt+2009)
```

---

## Action Plan

### Step 1: Download MCXC

**Option A: VizieR (original MCXC)**
```
URL: http://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/534/A109
Format: FITS or VOTable
```

**Option B: MCXC-II (2024, more complete)**
```
URL: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/688/A187
N = 2221 clusters
```

**Option C: HEASARC**
```
URL: https://heasarc.gsfc.nasa.gov/W3Browse/rosat/mcxc.html
Direct download in FITS
```

### Step 2: Cross-match PSZ2 x MCXC

Match criteria:
- Angular separation < 5 arcmin
- |Delta_z| < 0.05 (if redshifts available)

Expected result: ~500-800 clusters in common

### Step 3: Test Lx-Y (independent!)

Expected relation:
```
Y_SZ proportional to L_X^alpha with alpha ~ 1.0-1.2
```

This relation is **physically motivated** because:
- L_X proportional to n_e^2 x T^0.5 x V
- Y_SZ proportional to n_e x T x V
- Therefore Y_SZ proportional to L_X^(~1) for a given geometry

### Step 4: Test Q2R2 on the cross-match

Variables:
- **Q proxy:** f_gas = Y_SZ / M_X^(5/3) (SZ signal normalized by X-ray mass)
- **R proxy:** M_X (X-ray mass, independent of Y)
- **Residuals:** L_X - Y_SZ relation residuals

---

## Manual Download

### Via VizieR (recommended)

1. Go to https://vizier.cds.unistra.fr/
2. Search for "MCXC" or "J/A+A/534/A109"
3. Select the `mcxc` table
4. Click "Query selected tables"
5. Leave default parameters (all objects)
6. Choose "FITS" format at the bottom
7. Download

### Via TOPCAT

1. Open TOPCAT
2. VO -> VizieR Catalogue Service
3. Search for "MCXC"
4. Select J/A+A/534/A109
5. Download the entire catalogue

### Via Python (astroquery)

```python
from astroquery.vizier import Vizier

# Configure to retrieve everything
Vizier.ROW_LIMIT = -1

# Download MCXC
catalogs = Vizier.get_catalogs('J/A+A/534/A109')
mcxc = catalogs[0]

# Save as FITS
mcxc.write('MCXC_catalog.fits', format='fits', overwrite=True)
```

---

## File to create

Save to: `data/level2_multicontext/clusters/MCXC_catalog.fits`

---

## Cross-match script

The script `test_cluster_planck_mcxc.py` will:
1. Load PSZ2 (Y_SZ) and MCXC (M_X, L_X)
2. Cross-match by position
3. Calculate L_X - Y_SZ relation
4. Test U-shape in residuals vs environment

---

## Advantages of this approach

1. **Truly independent masses**: M_X derived from L_X, not from Y
2. **Two independent observables**: L_X (X-ray) and Y_SZ (SZ)
3. **Measurable physical scatter**: L_X-Y scatter reflects real physics
4. **Well-defined cross-match**: PSZ2 already includes an MCXC column!

---

## Note: PSZ2 already has the MCXC column!

In the PSZ2 union catalogue, there is an `MCXC` column that indicates the corresponding MCXC cluster name if a match exists!

This greatly simplifies the cross-match.

---

*Documentation created: December 7, 2025*
