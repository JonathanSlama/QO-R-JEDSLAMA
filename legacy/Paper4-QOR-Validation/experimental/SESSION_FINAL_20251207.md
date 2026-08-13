# Session December 7, 2025 - Final Summary

## Objectives
1. Test Q²R² in other contexts (KiDS, Planck, ACT)
2. Validate universality of U-shape + sign inversion pattern

---

## Positive Results

### KiDS DR4 (optical galaxies) - SUCCESS
```
N = 663,323 galaxies
U-shape: a = +0.00158 ± 0.00030 (5.2sigma)
Sign inversion: Q+ / R-
```

### Planck × MCXC (SZ clusters) - CORRECTION SUCCESSFUL
```
N = 551 clusters
Before D_A² correction: r = -0.13
After D_A² correction: r = +0.79
U-shape: a = +0.020 ± 0.015 (1.3sigma) (trend OK, N too small)
```

---

## Negative Result

### ACT-DR5 MCMF (SZ clusters) - NOT DETECTED
```
N = 6,148 clusters
U-shape (3D density): a = -0.60 (13sigma) inverted-shape
U-shape (L/M ratio): a = -0.08 (1.9sigma) ~linear
Sign inversion: NO
```

**Multi-proxy diagnosis reveals:**
- Signal correlated with SNR → probable selection bias
- "Cluster density" environment inadequate for clusters
- Clusters dominated by dark matter (85%) → Q²R² signal diluted?

---

## Validation Summary

| Scale | Context | N | U-shape | Sign Inv. | Status |
|-------|---------|---|---------|-----------|--------|
| **Galaxies** | ALFALFA (rot) | 19,222 | 5sigma+ | YES | **CONFIRMED** |
| **Galaxies** | KiDS (optical) | 663,323 | 5.2sigma | YES | **CONFIRMED** |
| **Clusters** | Planck×MCXC | 551 | 1.3sigma | - | Trend OK |
| **Clusters** | ACT-DR5 | 6,148 | NO | NO | Not detected |

**Provisional conclusion:** Q²R² confirmed at galaxy scale, not detected in SZ clusters.

---

## Hypotheses to Test

1. **Scale effect**: Clusters too DM-dominated to show baryonic signal
2. **SZ selection bias**: SZ catalogs have complex biases
3. **Inadequate proxy**: "Cluster density" not equal to "environment" in Q²R² sense
4. **Statistics**: Need more clusters with different selections

---

## Planned Future Tests

### Clusters (alternatives to SZ)

| Catalog | N | Selection | Script |
|---------|---|-----------|--------|
| **eROSITA eRASS1** | ~12,000 | X-ray | `download_erosita_erass1.py` |
| **redMaPPer SDSS** | ~26,000 | Optical | `download_redmapper.py` |
| ACT-DR6 | 10,040 | SZ | To create |

### Groups (intermediate scale)

| Catalog | N | Typical mass |
|---------|---|--------------|
| GAMA groups | ~24,000 | 10^12-10^14 Msun |
| Yang groups | ~470,000 | 10^11-10^15 Msun |

### Galaxies (more contexts)

| Survey | N | Specificity |
|--------|---|-------------|
| WALLABY | ~500k (expected) | Direct HI 21cm |
| DESI | ~40M | Precise spectro z |
| MaNGA | ~10,000 | Spatial kinematics |

---

## Files Created Today

### Scripts
```
tests/future/
├── test_lensing_kids_v2.py           KiDS SUCCESS
├── test_cluster_planck_mcxc_v2.py    D_A² correction
├── test_cluster_act_dr5.py           M-L test
├── test_cluster_act_multiproxy.py    Multi-proxy diagnosis
├── test_cluster_act_qr.py            L/M as Q test
├── download_act_dr5_mcmf.py          6,237 clusters
├── download_erosita_erass1.py        Ready
├── download_redmapper.py             Ready
└── download_act_dr5_original.py      Ready
```

### Documentation
```
experimental/04_RESULTS/
├── MULTICONTEXT_STATUS.md            Complete plan
├── ACT_CLUSTER_STRATEGY.md           ACT strategy
├── LEVEL2_KIDS_SUCCESS.md            KiDS result
└── test_results/
    ├── kids_brightsample_results_v2.json
    ├── planck_mcxc_corrected_results.json
    ├── act_dr5_mcmf_results.json
    ├── act_dr5_multiproxy_results.json
    └── act_dr5_qr_test_results.json
```

---

## Immediate Next Steps

1. **Download eROSITA eRASS1** (~12,000 X-ray clusters)
   ```
   python download_erosita_erass1.py
   ```

2. **Download redMaPPer SDSS** (~26,000 optical clusters)
   ```
   python download_redmapper.py
   ```

3. **Create test scripts** for these new catalogs

4. **Analyze KiDS in redshift bins** (cosmic evolution)

---

## Key Insight from Session

**The Q²R² signal appears clearly in GALAXIES (2 contexts, >5sigma) but NOT in SZ CLUSTERS.**

This could indicate that:
- Moduli fields couple primarily to **baryons**
- Clusters, dominated by **dark matter** (85%), dilute the signal
- OR SZ catalogs have biases that mask the signal
- OR "cluster density" environment is not the right proxy

**Crucial test:** X-ray (eROSITA) and optical (redMaPPer) catalogs will distinguish between SZ selection bias and real physical effect.

---

## Overall Progress

```
Before session:  ███████████░░░░░░░░░ 55%
After session:   ██████████████░░░░░░ 70%
```

**Gains:**
- +15%: KiDS confirmed (2nd galaxy context)
- Planck corrected (positive trend)
- ACT tested (negative but informative)
- Future test plan established

---

**Author:** Jonathan Édouard Slama
**Date:** December 7, 2025
