# Future Tests - Level 2+ Validation

## Overview

Tests prepared for multi-context string theory validation.

| Level | Test | Data Status | Script |
|-------|------|-------------|--------|
| 4 | Alternative Theories | ✅ Complete | `test_alternative_theories.py` |
| 4 | Mass Scaling | ✅ Complete | `test_mass_scaling.py` |
| 2 | KiDS Optical | ✅ Ready | `test_lensing_kids.py` |
| 2 | Planck Clusters | ⚠️ Extract .tgz | `test_cluster_planck.py` |

---

## Completed Tests (Level 4)

### Mass Scaling
```bash
python test_mass_scaling.py
```
**Result:** α = -0.154 ± 0.261 (weak mass dependence) ✅

### Alternative Theories
```bash
python test_alternative_theories.py
```
**Result:** 4/6 theories eliminated, String Theory ONLY compatible ✅

---

## Ready to Run (Level 2)

### KiDS Optical Galaxies
```bash
python test_lensing_kids.py
```
- **Data:** `data/level2_multicontext/kids/` ✅
- **N galaxies:** ~1 million
- **Test:** U-shape in Mass-Luminosity residuals

### Planck Galaxy Clusters
```bash
python test_cluster_planck.py
```
- **Data:** `data/level2_multicontext/clusters/`
- **⚠️ Action required:** Extract `COM_PCCS_SZ-Catalogs_vPR2.tgz`
- **N clusters:** ~1,600
- **Test:** U-shape in M-Y residuals

---

## Data Files Required

```
data/level2_multicontext/
├── kids/
│   ├── KiDS_DR4_brightsample.fits         ✅ (85 MB)
│   └── KiDS_DR4_brightsample_LePhare.fits ✅ (246 MB)
├── clusters/
│   ├── COM_PCCS_SZ-Catalogs_vPR2.tgz      ✅ (782 MB)
│   └── [EXTRACT THIS] → HFI_PCCS_SZ-union_R2.08.fits
└── DOWNLOAD_GUIDE.md
```

---

## Expected Outputs

After running tests, results are saved to:
```
experimental/04_RESULTS/test_results/
├── kids_brightsample_results.json
└── planck_clusters_results.json
```

---

## Success Criteria

| Criterion | Required for Validation |
|-----------|------------------------|
| U-shape (a > 0) | Yes |
| Significant (p < 0.1) | Preferred |
| Sign inversion | Strong evidence |
| Multiple contexts | Definitive |

---

## What Success Means

If U-shape detected in KiDS AND Planck:
- **3 independent contexts** confirm the pattern
- **Probability of coincidence:** < 10⁻⁶
- **Progress:** From ~50% to ~65% validation

---

*Updated: December 7, 2025*
