# Test GRP-001: Galaxy Groups (Tempel+2017)

**Document Type:** Pre-Registered Test Protocol  
**Author:** Jonathan Édouard Slama  
**Institution:** Metafund Research Division, Strasbourg, France  
**Date:** December 14, 2025  
**Status:** PRE-REGISTERED (criteria locked before execution)

---

## 1. Objective

Test QO+R TOE prediction at galaxy group scale (10²² m).

**Hypothesis:** The environmental U-shape signature observed at galactic scale should also appear at group scale, with sign depending on group baryonic composition.

---

## 2. Data Source

### Primary: Tempel et al. (2017)
- **Reference:** Tempel, E., et al. (2017), A&A, 602, A100
- **Title:** "Friends-of-friends galaxy group finder with membership refinement"
- **VizieR:** J/A+A/602/A100
- **Content:** ~70,000 galaxy groups from SDSS DR12
- **URL:** https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/602/A100

### Columns needed:
- `GroupID`: Group identifier
- `Ngal`: Number of member galaxies (richness)
- `sigma_v`: Velocity dispersion (km/s)
- `Mstar_group`: Total stellar mass of group
- `RA`, `Dec`: Position
- `z`: Redshift

### Secondary: ALFALFA cross-match (for gas content)
- Cross-match Tempel groups with ALFALFA to get HI content
- Compute group gas fraction

---

## 3. Variable Definitions

### 3.1 Q and R for Groups

```
Q_group = log₁₀(M_HI_group / M_bar_group)     # Group gas fraction
R_group = log₁₀(M_star_group / M_bar_group)   # Group stellar fraction
M_bar_group = M_HI_group + M_star_group       # Total baryonic mass
```

**Challenge:** ALFALFA may not detect all group members. Use:
- Sum of ALFALFA detections in group volume
- Upper limits for non-detections (statistical correction)

### 3.2 Environment Proxy

```
env = log₁₀(n_neighbors)   # Number of groups within 5 Mpc
```

Or use group richness as internal environment proxy:
```
env_internal = log₁₀(N_gal)
```

### 3.3 Observable: Velocity Dispersion Residual

Predicted from stellar mass (virial theorem):
```
log(σ_v)_pred = 0.33 × log(M_star) + const    # Virial scaling
```

Residual:
```
Δlog(σ_v) = log(σ_v)_obs - log(σ_v)_pred
```

---

## 4. Pre-Registered Criteria (LOCKED)

### 4.1 Detection Criterion
- **Criterion:** U-shape coefficient `a` significant at 3σ
- **Test:** Δlog(σ_v) = a × env² + b × env + c
- **PASS if:** |a / σ_a| > 3.0

### 4.2 Sign Prediction
Based on group composition:
- **Gas-rich groups** (Q > R): Expect a < 0 (inverted U)
- **Gas-poor groups** (Q < R): Expect a > 0 (normal U)

### 4.3 Consistency with Galactic Scale
- **Criterion:** Same sign pattern as galactic scale
- **PASS if:** Sign(a) correlates with Q/R balance as at galactic scale

### 4.4 Falsification Conditions
QO+R is **falsified at group scale** if:
1. No significant U-shape detected (|a/σ_a| < 2 in ALL subsamples), OR
2. Sign is OPPOSITE to prediction (gas-rich shows positive, gas-poor shows negative)

---

## 5. Analysis Plan

### Step 1: Data Acquisition
1. Download Tempel+2017 from VizieR
2. Download ALFALFA α.100
3. Cross-match by position (groups contain ALFALFA sources)

### Step 2: Sample Definition
- Minimum group size: N_gal ≥ 5
- Redshift range: 0.01 < z < 0.1
- Valid σ_v measurement: σ_v > 50 km/s

### Step 3: Compute Quantities
1. Compute M_HI_group (sum of ALFALFA members)
2. Compute Q_group, R_group
3. Compute environment (local group density)
4. Compute σ_v residual

### Step 4: Statistical Tests
1. Fit quadratic model to full sample
2. Split by Q/R ratio, fit separately
3. Test sign inversion between subsamples

### Step 5: Compare to Galactic Scale
1. Compare U-shape coefficients
2. Check sign consistency
3. Compute combined significance

---

## 6. Expected Outcomes

### If QO+R is correct:
- Significant U-shape in σ_v residuals (>3σ)
- Sign depends on group gas content
- Consistent with galactic-scale results

### If QO+R is wrong at group scale:
- No significant U-shape, OR
- Wrong sign pattern

### Implications:
- **PASS:** Extends TOE to second scale → strengthens claim
- **FAIL:** Limits QO+R to galactic scale only → weakens TOE claim

---

## 7. Timeline

| Task | Duration | Status |
|------|----------|--------|
| Download Tempel catalog | 10 min | ⬜ |
| Cross-match with ALFALFA | 30 min | ⬜ |
| Compute group quantities | 30 min | ⬜ |
| Run statistical tests | 20 min | ⬜ |
| Document results | 20 min | ⬜ |
| **Total** | **~2 hours** | |

---

## 8. Code Location

```
experimental/08_GROUP_SCALE/
├── README.md                      # This document
├── download_tempel.py             # Data acquisition
├── crossmatch_alfalfa.py          # Cross-matching
├── grp001_ushape_test.py          # Main analysis
└── results/
    └── grp001_results.json        # Output
```

---

**Document Status:** PRE-REGISTERED  
**Criteria locked at:** December 14, 2025, 22:30 UTC  
**DO NOT MODIFY CRITERIA AFTER EXECUTION**
