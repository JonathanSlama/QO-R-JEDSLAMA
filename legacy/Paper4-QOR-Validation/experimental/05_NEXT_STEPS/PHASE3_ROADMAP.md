# PHASE 3 IMPLEMENTATION ROADMAP
## Level 3: Direct Signatures

---

### Timeline

**Week 1-2: CMB Lensing Setup**
- Download Planck PR3 lensing maps
- Cross-match with KiDS galaxy positions
- Implement correlation analysis

**Week 3-4: CMB Lensing Analysis**
- Run cross-correlation
- Test significance
- Split by Q-dominated / R-dominated

**Week 5-6: Void Analysis**
- Download SDSS void catalog
- Extract void density profiles
- Test for Q2R2 modification

**Week 7-8: Integration**
- Combine results
- Write Phase 3 section for paper
- Prepare figures

---

### Data Requirements

| Dataset | Size | URL |
|---------|------|-----|
| Planck CMB lensing | ~500 MB | pla.esac.esa.int |
| SDSS void catalog | ~50 MB | VizieR |
| DES Y3 shear | ~10 GB | des.ncsa.illinois.edu |

---

### Success Criteria

**Positive Result:**
- Cross-correlation detected at >3 sigma
- Sign inversion in correlation (Q vs R)
- Consistent with Level 2 results

**Null Result:**
- Not necessarily negative
- May indicate scale limitation
- Still constrains theory

---

### Scripts to Create

1. `download_planck_lensing.py`
2. `test_cmb_lensing_correlation.py`
3. `download_sdss_voids.py`
4. `test_void_profiles.py`
5. `analyze_phase3_results.py`

---

### Technical Notes

**CMB Lensing Cross-Correlation Method:**

1. Load Planck kappa (convergence) map in HEALPix format
2. Extract kappa values at KiDS/ALFALFA galaxy positions
3. Bin galaxies by BTF/M-L residual
4. Compute mean kappa in each residual bin
5. Test for correlation between residuals and kappa

**Expected Signal:**
- If Q2R2 modifies gravitational potential
- Galaxies with positive residuals should trace different kappa
- Sign inversion between Q-dominated and R-dominated

**Challenges:**
- Planck resolution (~5 arcmin) vs galaxy positions
- Need sufficient statistics in each bin
- Systematic effects from foregrounds

---

### Current Progress

| Item | Status |
|------|--------|
| Data directory | Created |
| Download instructions | Documented |
| Analysis scripts | Template ready |
| healpy installation | Required |

---

*Phase 3 represents the most direct test of Q2R2 as new physics. A positive detection would be transformative.*
