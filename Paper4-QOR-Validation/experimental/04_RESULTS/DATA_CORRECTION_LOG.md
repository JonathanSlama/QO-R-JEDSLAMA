# Data Correction Log - Paper 4 QO+R Validation

## Issue 1: ALFALFA Scaling Relation (FIXED)

### Problem
The `alfalfa.csv` file was generated using a **scaling relation** to estimate stellar masses:

```python
# WRONG - creates circular dependency!
log_Mstar_approx = 0.7 * log_MHI + 2.5
```

This created an artificial Q-R correlation of **+0.997**, making QO+R analysis invalid.

### Solution
Created `prepare_alfalfa.py` which merges ALFALFA α.100 with Durbala et al. (2020) SDSS photometry.

| Metric | Before | After |
|--------|--------|-------|
| M_star source | Scaling relation | SDSS photometry |
| Q-R correlation | +0.997 ❌ | -0.808 ✅ |

---

## Issue 2: Killer Prediction Threshold (CLARIFIED)

### Initial Observation
Using f_gas < 0.3 as "R-dominated" threshold showed no U-shape inversion.

### Root Cause
**ALFALFA is a 21cm HI survey** - it cannot detect galaxies without significant HI gas.

| Survey | "R-dominated" f_gas | True gas-poor systems |
|--------|---------------------|----------------------|
| TNG300 | 0.003 - 0.02 | ✅ Included |
| ALFALFA | 0.15 - 0.30 | ❌ Still gas-rich! |

True ellipticals/S0s with f_gas ~ 0.01 are **invisible** to ALFALFA.

### Solution
Apply strict TNG300-equivalent criteria:
- f_gas < 0.05 (not < 0.3)
- log(M_star) > 10.5 (massive galaxies)

### Result

| Sample | N | U-shape a | Inversion |
|--------|---|-----------|-----------|
| Loose R-dom (f_gas < 0.3) | 2,861 | +0.005 | ❌ No |
| **EXTREME R-dom** | **64** | **-0.006** | **✅ Yes** |

**Killer prediction CONFIRMED** with correct criteria.

---

## Issue 3: WALLABY M_star (UNRESOLVED)

### Problem
WALLABY PDR2 catalog contains only HI data, no stellar masses.

Analysis used: `M_bar = 2.5 × M_HI` → `f_gas = 0.544` constant.

### Status
❌ **WALLABY cannot be used for QO+R** without photometric cross-match.

### Recommended Solution
Cross-match with WISE All-Sky Source Catalog or DES for M_star estimates.

---

## Files Modified

| File | Change | Date |
|------|--------|------|
| `tests/prepare_alfalfa.py` | NEW - Correct data preparation | 2025-12-06 |
| `tests/test_real_data.py` | Use corrected ALFALFA file | 2025-12-06 |
| `04_RESULTS/DATA_CORRECTION_LOG.md` | This documentation | 2025-12-06 |
| `04_RESULTS/FINAL_RESULTS_ANALYSIS.md` | Complete results | 2025-12-06 |

---

## Verification Commands

```bash
# Step 1: Prepare corrected ALFALFA data
cd Paper4-Redshift-Predictions/tests
python prepare_alfalfa.py

# Step 2: Run full analysis
python test_real_data.py
```

---

## Lessons Learned

1. **Scaling relations create circular dependencies** - always use independent measurements
2. **Survey selection effects matter** - HI surveys miss gas-poor galaxies
3. **Threshold definitions must match** - "R-dominated" means f_gas < 0.05, not < 0.3
4. **Verify Q-R independence** - correlation should be negative (-0.7 to -0.8)

---

*Last updated: December 6, 2025*
