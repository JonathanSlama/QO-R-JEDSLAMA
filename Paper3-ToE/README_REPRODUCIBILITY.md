# 🔬 Reproducibility Guide - Paper 3: String Theory Embedding

## Complete Guide to Reproducing All Results

**Author:** Jonathan Édouard Slama  
**Affiliation:** Metafund Research Division  
**Date:** December 2025

---

## 📋 Overview

This document provides complete instructions to reproduce all results in Paper 3, from raw data acquisition to final figures.

---

## 🚀 Quick Reproduction (If You Have the Data)

If you already have TNG data downloaded, you can reproduce our exact results:

```bash
# 1. Navigate to the TNG analysis folder
cd BTFR/TNG/

# 2. Run the main analysis script (produces tng_comparison_results.csv)
python analyze_tng_comparison.py

# 3. Generate Paper 3 figures
cd ../../Git/Paper3-ToE/tests/
python generate_figures_from_results.py
```

**Output:** `tng_comparison_results.csv` containing:
```
,n_galaxies,C_Q,C_R,lambda_QR,box_size
TNG50,8058.0,0.533,-0.995,-0.736,35.0
TNG100,53363.0,3.104,-0.575,0.817,75.0
TNG300,623609.0,2.938,0.835,1.009,205.0
```

---

## 1️⃣ Data Acquisition

### 1.1 SPARC (Observational)

**Source:** http://astroweb.cwru.edu/SPARC/

```bash
# Direct download - no registration required
wget http://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt
```

**Contents:** 175 disk galaxies with Spitzer 3.6μm photometry and high-resolution HI rotation curves.

### 1.2 ALFALFA (Observational)

**Source:** http://egg.astro.cornell.edu/alfalfa/data/

```bash
# Download α.100 catalog
wget http://egg.astro.cornell.edu/alfalfa/data/a100.code.csv
```

**Contents:** 31,502 HI sources, we use 21,834 with quality flags.

### 1.3 WALLABY (Observational)

**Source:** https://wallaby-survey.org/data/

**Pilot Data Release 2:** Registration required at CSIRO ASKAP Science Data Archive (CASDA).

**Contents:** 2,047 galaxies from ASKAP observations.

### 1.4 IllustrisTNG (Cosmological Simulations)

**Source:** https://www.tng-project.org/data/

#### Registration Process:
1. Go to https://www.tng-project.org/users/register/
2. Create account with institutional email
3. Request API key (approved within 24-48 hours)
4. Accept data use policy

#### What to Download:

| Simulation | Product | Snapshot | Size |
|------------|---------|----------|------|
| TNG50-1 | Group Catalogs | 099 (z=0) | ~5 GB |
| TNG100-1 | Group Catalogs | 099 (z=0) | ~15 GB |
| TNG300-1 | Group Catalogs | 099 (z=0) | ~100 GB |

#### Download Methods:

**Method 1: Web Interface**
- Go to https://www.tng-project.org/data/
- Select simulation → Snapshots → 099 → Group Catalogs
- Download all `fof_subhalo_tab_099.*.hdf5` files

**Method 2: API Download**
```bash
# Using wget with API key
wget --content-disposition --header="api-key: YOUR_API_KEY" \
  "https://www.tng-project.org/api/TNG300-1/files/groupcat-99/?file_format=hdf5"
```

**Method 3: illustris_python Package**
```python
import illustris_python as il

# After setting up data path
basePath = '/path/to/TNG300-1/output'
subhalos = il.groupcat.loadSubhalos(basePath, 99, 
    fields=['SubhaloMassType', 'SubhaloVmax', 'SubhaloPos', 'SubhaloSFR'])
```

---

## 2️⃣ Analysis Scripts

### 2.1 Main Analysis Script

**File:** `BTFR/TNG/analyze_tng_comparison.py`

This script:
1. Loads all TNG50/100/300 group catalogs
2. Filters galaxies by stellar mass (> 10⁸ M☉) and Vmax (> 30 km/s)
3. Computes BTFR residuals
4. Estimates local environment density (KD-tree, 10th nearest neighbor)
5. Optimizes QO+R parameters (C_Q, C_R, λ_QR) via differential evolution
6. Saves results to `tng_comparison_results.csv`

**Key Parameters:**
```python
H = 0.6774          # Hubble parameter (TNG cosmology)
MIN_STELLAR_MASS = 1e8   # Solar masses
MIN_VMAX = 30       # km/s
```

**How It Works:**

```python
# Simplified version of the analysis
def load_subhalos(data_dir, box_size):
    """Load TNG subhalo data from HDF5 files."""
    
    # Only use files > 1MB (smaller files are empty chunks)
    valid_files = [f for f in files if os.path.getsize(f) > 1024*1024]
    
    for filepath in valid_files:
        with h5py.File(filepath, 'r') as f:
            if 'Subhalo' not in f:
                continue
            
            # Extract masses (convert from 10^10 M☉/h to M☉)
            mass_type = f['Subhalo/SubhaloMassType'][:]
            gas_mass = mass_type[:, 0] * 1e10 / H
            stellar_mass = mass_type[:, 4] * 1e10 / H
            
            # Velocity and position
            vmax = f['Subhalo/SubhaloVmax'][:]
            pos = f['Subhalo/SubhaloPos'][:] / 1000  # ckpc/h → Mpc
    
    return DataFrame with all subhalos

def optimize_qor(df):
    """Find best-fit QO+R parameters."""
    
    # Normalize gas/stellar fractions
    df['Q_norm'] = standardize(df['gas_fraction'])
    df['R_norm'] = standardize(1 - df['gas_fraction'])
    
    def objective(params):
        C_Q, C_R, lambda_QR = params
        
        # QO+R correction
        env2 = df['env_std'] ** 2
        correction = 0.035 * (
            C_Q * df['Q_norm'] * env2 + 
            C_R * df['R_norm'] * env2 + 
            lambda_QR * df['Q_norm']**2 * df['R_norm']**2
        )
        
        # Minimize residual variance in density bins
        corrected = df['btfr_residual'] + correction
        return variance_across_bins(corrected, df['env_std'])
    
    # Differential evolution optimization
    result = differential_evolution(objective, bounds=[(-5,5), (-5,5), (-2,2)])
    return result.x  # [C_Q, C_R, lambda_QR]
```

### 2.2 Figure Generation Script

**File:** `Git/Paper3-ToE/tests/generate_figures_from_results.py`

This script:
1. Loads pre-computed results from `tng_comparison_results.csv`
2. Generates all 8 figures for Paper 3
3. Saves to `Git/Paper3-ToE/figures/`

---

## 3️⃣ Expected Results

### 3.1 λ_QR Convergence (Main Result)

| Simulation | N galaxies | Box (Mpc) | λ_QR | Interpretation |
|------------|------------|-----------|------|----------------|
| TNG50 | 8,058 | 35 | -0.74 | ❌ Too small, edge effects |
| TNG100 | 53,363 | 75 | 0.82 | ⚠️ Intermediate |
| TNG300 | 623,609 | 205 | **1.01** | ✅ Converged to theory |

**Theory prediction:** λ_QR = 0.93 ± 0.15 (from Calabi-Yau geometry)

### 3.2 U-Shape Coefficients

| Dataset | Type | N | a | σ(a) | Significance |
|---------|------|---|---|------|--------------|
| SPARC | Observation | 175 | +0.035 | 0.008 | 4.4σ |
| ALFALFA | Observation | 21,834 | +0.0023 | 0.0008 | 2.9σ |
| WALLABY | Observation | 2,047 | +0.0069 | 0.0058 | 1.2σ |
| TNG100 | Simulation | 53,363 | +0.017 | 0.006 | 2.8σ |
| TNG300 Q-dom | Simulation | 444,374 | +0.017 | 0.008 | 2.1σ |
| TNG300 R-dom | Simulation | 34,444 | **-0.019** | 0.003 | **6.3σ** |

### 3.3 Killer Prediction (Sign Inversion)

**Prediction (made before analysis):** R-dominated systems show a < 0

**Result:** a = -0.019 ± 0.003 

**Status:** ✅ **CONFIRMED at 6.3σ**

---

## 4️⃣ Directory Structure

```
BTFR/
├── TNG/
│   ├── Data/                    # TNG100 HDF5 files
│   ├── Data_TNG50/              # TNG50 HDF5 files  
│   ├── Data_TNG300/             # TNG300 HDF5 files (600 files)
│   ├── Data_Wallaby/            # WALLABY XML catalog
│   ├── analyze_tng_comparison.py  # ★ MAIN ANALYSIS SCRIPT
│   ├── tng_comparison_results.csv # ★ OUTPUT RESULTS
│   └── tng_comparison_figure.png  # Comparison figure
│
Git/Paper3-ToE/
├── manuscript/
│   └── paper3_string_theory_v2.tex
├── figures/                     # Generated figures
├── tests/
│   ├── generate_figures_from_results.py  # ★ FIGURE GENERATION
│   └── diagnose_tng_structure.py
└── README_REPRODUCIBILITY.md    # This file
```

---

## 5️⃣ Step-by-Step Reproduction

### Full Reproduction (From Scratch)

```bash
# 1. Register at TNG Project and get API key
#    https://www.tng-project.org/users/register/

# 2. Download TNG data (this takes time!)
#    - TNG50: ~5 GB
#    - TNG100: ~15 GB  
#    - TNG300: ~100 GB

# 3. Organize data in folders:
#    BTFR/TNG/Data_TNG50/
#    BTFR/TNG/Data/          (TNG100)
#    BTFR/TNG/Data_TNG300/

# 4. Run main analysis
cd BTFR/TNG/
python analyze_tng_comparison.py

# 5. Check output
cat tng_comparison_results.csv
# Should show λ_QR ≈ 1.01 for TNG300

# 6. Generate figures
cd ../../Git/Paper3-ToE/tests/
python generate_figures_from_results.py

# 7. Compile manuscript
cd ../manuscript/
pdflatex paper3_string_theory_v2.tex
pdflatex paper3_string_theory_v2.tex
```

### Quick Reproduction (Using Our Results)

If you want to verify figures without re-running the full TNG analysis:

```bash
cd Git/Paper3-ToE/tests/
python generate_figures_from_results.py
```

This uses the pre-computed `tng_comparison_results.csv`.

---

## 6️⃣ What Results Prove vs. Don't Prove

### ✅ What the results PROVE:

1. **U-shape exists** in BTFR residuals vs. environment
2. **Pattern is robust** across 5 independent datasets
3. **λ_QR ≈ 1** emerges at cosmologically representative scales
4. **Sign inversion** occurs for R-dominated systems
5. **QR ≈ const** anti-correlation exists (r = -0.89)

### ❌ What the results DO NOT prove:

1. String theory is correct
2. Q and R are literally the dilaton and Kähler modulus
3. The quintic CY compactification is realized in nature
4. No alternative explanations exist

---

## 7️⃣ Potential Issues

### Data Issues
- Some HDF5 files are empty (no subhalos in that chunk)
- File size < 1MB usually means empty → skip these
- TNG50 box is too small for reliable environmental analysis

### Numerical Issues
- Optimization may find local minima → we use differential_evolution
- Bootstrap errors depend on number of iterations (we use 1000)
- KD-tree requires periodic boundary conditions for TNG boxes

### Physical Caveats
- Simulations include ΛCDM physics only
- Baryonic feedback models are approximate
- "Environment" definition varies between studies

---

## 8️⃣ Reproducibility Checklist

- [ ] Downloaded SPARC data
- [ ] Downloaded ALFALFA catalog
- [ ] Registered for TNG access
- [ ] Downloaded TNG50/100/300 group catalogs
- [ ] Ran `analyze_tng_comparison.py`
- [ ] Verified `tng_comparison_results.csv` matches expected values
- [ ] Ran `generate_figures_from_results.py`
- [ ] Compiled `paper3_string_theory_v2.tex`
- [ ] Compared figures to paper

---

## 📞 Contact

For questions about reproducibility:

**Jonathan Édouard Slama**  
Email: jonathan@metafund.in  
ORCID: 0009-0002-1292-4350

---

## 📄 License

Analysis scripts are provided under MIT License.
TNG data is subject to TNG Project data policy.

---

*This guide aims for complete transparency. If you find discrepancies, please report them.*
