#!/usr/bin/env python3
"""
===============================================================================
PHASE 3 PREPARATION: CMB LENSING CROSS-CORRELATION
===============================================================================

Level 3 test: Direct signature in CMB lensing maps

THEORY:
If Q2R2 modifies the gravitational potential, it should also affect:
1. CMB photon deflection (lensing)
2. Integrated Sachs-Wolfe effect

Author: Jonathan E. Slama
Date: December 2025
===============================================================================
"""

import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "level3_direct")
OUTPUT_DIR = os.path.join(BASE_DIR, "experimental", "05_NEXT_STEPS")


def outline_phase3_tests():
    """
    Outline all Phase 3 (Level 3) tests for direct signatures.
    """
    
    print("\n" + "="*70)
    print(" PHASE 3: DIRECT SIGNATURE TESTS")
    print("="*70)
    
    print("""
  Level 3 tests look for DIRECT signatures of Q2R2 in:
  
  +---------------------------------------------------------------------------+
  | TEST                        DATA                 STATUS                   |
  +---------------------------------------------------------------------------+
  | 3.1 CMB Lensing            Planck PR3           Ready to download         |
  |     x Galaxy residuals                                                    |
  |                                                                           |
  | 3.2 ISW Effect             Planck + SDSS        Ready to download         |
  |     x Large-scale structure                                               |
  |                                                                           |
  | 3.3 Void Profiles          SDSS voids           Ready                     |
  |     Q2R2 should modify void density profiles                              |
  |                                                                           |
  | 3.4 Baryon Acoustic        DESI/eBOSS           Public data               |
  |     Oscillations                                                          |
  |                                                                           |
  | 3.5 Weak Lensing           DES Y3, KiDS-1000    Public data               |
  |     Shear correlations                                                    |
  +---------------------------------------------------------------------------+
  
  PRIORITY ORDER:
  
  1. CMB Lensing (3.1)
     - Most direct test
     - Data publicly available
     - Clear prediction from Q2R2
  
  2. Void Profiles (3.3)
     - Q2R2 predicts modified void emptiness
     - SDSS void catalogs available
     - Independent of galaxy selection
  
  3. Weak Lensing (3.5)
     - Extension of KiDS analysis
     - Uses same data
     - Tests lensing directly
  
  WHAT WOULD CONSTITUTE DETECTION:
  
  For CMB Lensing:
  - Correlation between BTF residuals and kappa
  - Different correlation for Q-dominated vs R-dominated
  - Significance > 3 sigma
  
  For Voids:
  - Modified density profile shape
  - Q2R2 signature in void walls
  - Significance > 3 sigma
    """)


def download_info():
    """Print download information for Phase 3 data."""
    
    print("\n" + "="*70)
    print(" DATA DOWNLOAD INSTRUCTIONS")
    print("="*70)
    
    print("""
  CMB LENSING (Planck PR3)
  ========================
  
  URL: https://pla.esac.esa.int/
  
  Navigate to: Lensing > PR3 > Convergence maps
  
  Download:
  - COM_Lensing_4096_R3.00_MV.fits.gz (Minimum Variance map, ~500 MB)
  - COM_Lensing_4096_R3.00_noise.fits.gz (Noise estimate)
  
  Alternative URL (IRSA):
  https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/lensing/
  
  
  VOID CATALOGS
  =============
  
  SDSS DR7 Voids (Pan et al. 2012):
  https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/MNRAS/421/926
  
  BOSS DR12 Voids (Mao et al. 2017):
  https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJ/835/161
  
  
  WEAK LENSING
  ============
  
  KiDS-1000 Shear Catalog:
  https://kids.strw.leidenuniv.nl/DR4/
  
  DES Y3 Shape Catalog:
  https://des.ncsa.illinois.edu/releases/y3a2
  
  
  REQUIRED PYTHON PACKAGES
  ========================
  
  For CMB lensing analysis:
    pip install healpy
  
  Note: healpy can be tricky on Windows. Consider using WSL or Linux.
    """)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"\n  Data directory created: {DATA_DIR}")


def create_phase3_roadmap():
    """Create Phase 3 implementation roadmap."""
    
    roadmap = """# PHASE 3 IMPLEMENTATION ROADMAP
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
"""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    roadmap_path = os.path.join(OUTPUT_DIR, "PHASE3_ROADMAP.md")
    
    with open(roadmap_path, 'w') as f:
        f.write(roadmap)
    
    print(f"\n  Roadmap saved: {roadmap_path}")
    
    return roadmap_path


def create_analysis_template():
    """Create template for CMB lensing analysis."""
    
    template = '''#!/usr/bin/env python3
"""
CMB Lensing x Galaxy Residuals Cross-Correlation
=================================================

This script performs the actual cross-correlation once
Planck lensing maps are available.

Requirements:
    pip install healpy

Usage:
    python test_cmb_lensing_correlation.py
"""

import os
import numpy as np

# Check for healpy
try:
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    HEALPY_AVAILABLE = False
    print("WARNING: healpy not installed. Install with: pip install healpy")

from astropy.io import fits
from astropy.table import Table
from scipy import stats
import json


# Configuration
LENSING_MAP = "COM_Lensing_4096_R3.00_MV.fits.gz"
NOISE_MAP = "COM_Lensing_4096_R3.00_noise.fits.gz"


def load_lensing_map(path):
    """Load Planck lensing convergence map."""
    if not HEALPY_AVAILABLE:
        print("ERROR: healpy required")
        return None, None
    
    kappa = hp.read_map(path, field=0)
    nside = hp.get_nside(kappa)
    print(f"  Loaded lensing map: NSIDE={nside}, Npix={len(kappa)}")
    return kappa, nside


def get_kappa_at_positions(kappa, nside, ra, dec):
    """Extract kappa values at galaxy positions."""
    if not HEALPY_AVAILABLE:
        return None
    
    # Convert to HEALPix coordinates
    theta = np.radians(90 - dec)  # colatitude
    phi = np.radians(ra)          # longitude
    
    # Get pixel indices
    pix = hp.ang2pix(nside, theta, phi)
    
    return kappa[pix]


def cross_correlate_residuals(residuals, kappa_values, n_bins=10):
    """
    Cross-correlate galaxy residuals with CMB lensing.
    
    If Q2R2 affects gravitational potential:
    - Positive residuals should correlate differently with kappa
    - This would be a DIRECT signature of new physics
    """
    
    # Bin by residual value
    residual_bins = np.percentile(residuals, np.linspace(0, 100, n_bins+1))
    
    results = []
    for i in range(n_bins):
        mask = (residuals >= residual_bins[i]) & (residuals < residual_bins[i+1])
        if mask.sum() > 10:
            mean_kappa = kappa_values[mask].mean()
            std_kappa = kappa_values[mask].std() / np.sqrt(mask.sum())
            results.append({
                'residual_min': float(residual_bins[i]),
                'residual_max': float(residual_bins[i+1]),
                'mean_residual': float(residuals[mask].mean()),
                'mean_kappa': float(mean_kappa),
                'std_kappa': float(std_kappa),
                'n': int(mask.sum())
            })
    
    return results


def main():
    print("="*70)
    print(" CMB LENSING x GALAXY RESIDUALS CROSS-CORRELATION")
    print("="*70)
    
    if not HEALPY_AVAILABLE:
        print("\\n  ERROR: healpy not available")
        print("  Install with: pip install healpy")
        print("  Note: On Windows, consider using WSL")
        return
    
    # Check for lensing map
    if not os.path.exists(LENSING_MAP):
        print(f"\\n  ERROR: Lensing map not found: {LENSING_MAP}")
        print("  Please download from Planck Legacy Archive:")
        print("  https://pla.esac.esa.int/")
        return
    
    # Load lensing map
    print("\\n  Loading Planck lensing map...")
    kappa, nside = load_lensing_map(LENSING_MAP)
    
    # Load galaxy data
    print("\\n  Loading galaxy data...")
    # TODO: Load KiDS or ALFALFA data with residuals
    
    # Cross-correlate
    print("\\n  Computing cross-correlation...")
    # TODO: Implement cross-correlation
    
    print("\\n  Analysis complete!")


if __name__ == "__main__":
    main()
'''
    
    script_path = os.path.join(os.path.dirname(__file__), "test_cmb_lensing_correlation.py")
    with open(script_path, 'w') as f:
        f.write(template)
    
    print(f"\n  Analysis template saved: {script_path}")


def main():
    print("="*70)
    print(" PHASE 3 PREPARATION")
    print(" Level 3: Direct Signatures")
    print("="*70)
    
    # Outline all Phase 3 tests
    outline_phase3_tests()
    
    # Download information
    download_info()
    
    # Create roadmap
    create_phase3_roadmap()
    
    # Create analysis template
    create_analysis_template()
    
    print("\n" + "="*70)
    print(" PHASE 3 PREPARATION COMPLETE")
    print("="*70)
    print("""
  NEXT STEPS:
  
  1. Install healpy (required for CMB lensing):
     pip install healpy
     
  2. Download Planck CMB lensing maps from:
     https://pla.esac.esa.int/
     
  3. Run test_cmb_lensing_correlation.py
  
  4. Analyze results
  
  If positive, this would be the FIRST direct signature
  of string theory in cosmological observations!
    """)


if __name__ == "__main__":
    main()
