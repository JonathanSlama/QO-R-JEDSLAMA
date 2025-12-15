# What Can We Do NOW vs What Requires Waiting

## Summary Table

| Level | Test | Can Start NOW? | Data Status | Timeline |
|-------|------|----------------|-------------|----------|
| **1** | Environment proxy improvement | ✅ YES | Have ALFALFA positions | 1-2 weeks |
| **1** | Mass scaling | ✅ YES | Have ALFALFA + SDSS M_star | 1 week |
| **1** | λ_QR(z) evolution | ❌ NO | Need SKA/LADUMA | 2028+ |
| **2** | Gravitational lensing | 🟡 PARTIAL | Public data available (DES, KiDS) | 2-4 weeks |
| **2** | Galaxy clusters | 🟡 PARTIAL | Public data available (Planck, eROSITA) | 2-4 weeks |
| **2** | CMB lensing correlation | 🟡 PARTIAL | Planck public | 2-4 weeks |
| **3** | Axion correlation | ❌ NO | Need axion detection | 2030+ |
| **3** | B-modes CMB | ❌ NO | Need CMB-S4 | 2030+ |
| **4** | WDM elimination | ✅ YES | Theoretical + existing data | 1 week |
| **4** | SIDM elimination | ✅ YES | Theoretical + existing data | 1 week |
| **4** | f(R) elimination | ✅ YES | Theoretical + existing data | 1 week |

---

## What We Can Do RIGHT NOW

### Week 1-2: Mass Scaling + Alternative Elimination

**Files ready:**
- `tests/future/test_mass_scaling.py` ✅
- `tests/future/test_alternative_theories.py` ✅

**Action:**
```bash
cd Paper4-Redshift-Predictions/tests/future
python test_mass_scaling.py
python test_alternative_theories.py
```

**Expected outcome:**
- Scaling exponent α for mass dependence
- Formal elimination of WDM, SIDM, f(R), Quintessence, Fuzzy DM

### Week 3-4: Environment Proxy Optimization

**Goal:** Increase U-shape amplitude from ~0.003 to ~0.01+

**Method:**
1. Download SDSS group catalog (Yang et al.)
2. Cross-match with ALFALFA
3. Use group membership as environment proxy
4. Compare with N-th neighbor and Supergalactic methods

**Data needed:** SDSS DR7 group catalog (public)

### Week 5-8: Multi-Context Pilot Tests

**Gravitational Lensing:**
- Download DES Y3 or KiDS-1000 shear catalog
- Define Q from color (g-r as gas proxy)
- Test for U-shape in lensing residuals

**Galaxy Clusters:**
- Use Planck SZ catalog (public)
- Define Q from X-ray gas fraction
- Test for U-shape in M-T residuals

---

## What Requires Waiting

### λ_QR(z) Redshift Evolution (Level 1)

**Why we can't do it now:**
- WALLABY lacks independent stellar masses
- ALFALFA limited to z < 0.06
- Need z ~ 0.5-1 for meaningful test

**What would enable it:**
- WALLABY × WISE cross-match (stellar masses) → 2-3 months work
- MIGHTEE-HI survey data release → 2026
- SKA pathfinders → 2028+

### Direct String Theory Signatures (Level 3)

**Axion detection:**
- ADMX, ABRACADABRA experiments ongoing
- If detected, we can test m_axion = f(M_moduli)
- Timeline: Could be 2025, could be never

**CMB B-modes:**
- LiteBIRD launch: 2028
- CMB-S4: 2030+
- Would constrain compactification energy scale

**Extra dimensions:**
- Tabletop gravity experiments ongoing
- So far null results at sub-mm scales
- Would be definitive if detected

---

## Recommended Immediate Actions

### Priority 1: Run Existing Tests
```bash
# Mass scaling
python tests/future/test_mass_scaling.py

# Alternative theories
python tests/future/test_alternative_theories.py
```

### Priority 2: Download Public Data
1. **SDSS Group Catalog** (for environment)
   - URL: https://gax.sjtu.edu.cn/data/Group.html
   - File: `group_DR7_M1.dat`

2. **DES Y3** (for lensing pilot)
   - URL: https://des.ncsa.illinois.edu/releases/y3a2
   - Registration required

3. **Planck SZ Catalog** (for clusters)
   - URL: https://pla.esac.esa.int/
   - File: `HFI_PCCS_SZ-union_R2.08.fits`

### Priority 3: Write Paper 4 Appendix
- Document complete alternative elimination
- Include mass scaling results
- Prepare for Paper 5 (multi-context)

---

## Timeline Summary

```
NOW (December 2025)
├── Mass scaling test
├── Alternative theories elimination
└── Environment proxy comparison

Q1 2026
├── SDSS group environment
├── Lensing pilot (DES)
└── Cluster pilot (Planck SZ)

Q2-Q3 2026
├── Full lensing analysis
├── Full cluster analysis
├── WALLABY × WISE cross-match
└── Paper 5 submission

2027-2028
├── MIGHTEE-HI analysis
├── λ_QR(z=0.3) measurement
└── Paper 6 submission

2030+
├── SKA full survey
├── CMB-S4 correlation
├── Potential axion correlation
└── Definitive validation
```

---

## Bottom Line

**Can do NOW (no waiting):**
- 4 tests ready to run immediately
- 3 multi-context pilots with public data
- Complete Level 4 (alternative elimination)

**Must wait:**
- Level 1 quantitative (λ_QR(z)) - need better redshift data
- Level 3 direct signatures - need particle physics experiments

**Estimated progress if we do everything possible NOW:**
- Current: ~35-40%
- After immediate actions: ~50-55%
- After 2026 multi-context: ~65-70%
- Definitive proof: requires 2030+ data

---

*Document created: December 7, 2025*
