#!/usr/bin/env python3
"""
===============================================================================
DOWNLOAD redMaPPer SDSS CLUSTER CATALOG
===============================================================================

Download redMaPPer SDSS DR8 cluster catalog from VizieR.
~26,000 optically-selected clusters with richness λ.

Key advantage: PURELY OPTICAL selection (no SZ or X-ray bias!)

References:
- Rykoff et al. (2014), ApJ 785, 104 - Algorithm
- Rykoff et al. (2016), ApJS 224, 1 - DES SV catalog

Author: Jonathan E. Slama
Date: December 2025
===============================================================================
"""

import os
import sys

try:
    from astroquery.vizier import Vizier
except ImportError:
    print("Installing astroquery...")
    os.system(f"{sys.executable} -m pip install astroquery --break-system-packages")
    from astroquery.vizier import Vizier

from astropy.table import Table
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "data", "level2_multicontext", "clusters")

# redMaPPer catalogs on VizieR
REDMAPPER_DES_ID = "J/ApJS/224/1"    # DES SV + SDSS DR8 (Rykoff+2016)
REDMAPPER_SDSS_ID = "J/ApJ/785/104"  # SDSS DR8 original (Rykoff+2014)


def download_catalog(vizier_id, name):
    """Download a catalog from VizieR."""
    
    print(f"\n  Downloading {name}...")
    print(f"  VizieR ID: {vizier_id}")
    
    Vizier.ROW_LIMIT = -1
    
    try:
        catalogs = Vizier.get_catalogs(vizier_id)
        
        if len(catalogs) == 0:
            print(f"  WARNING: No tables found for {vizier_id}")
            return None
        
        print(f"  Found {len(catalogs)} table(s)")
        
        for i, cat in enumerate(catalogs):
            print(f"    Table {i}: {len(cat)} rows - {cat.colnames[:5]}...")
        
        return catalogs
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    print("="*70)
    print(" DOWNLOADING redMaPPer CLUSTER CATALOGS")
    print("="*70)
    print("""
  redMaPPer = red-sequence Matched-filter Probabilistic Percolation
  
  Key features:
  - PURELY OPTICAL selection (no SZ/X-ray bias)
  - Richness λ = number of red-sequence galaxies
  - Photometric redshifts with σ_z/(1+z) ~ 0.01-0.02
  - Tightly correlated with mass
    """)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # =========================================================================
    # 1. DES SV + SDSS DR8 (Rykoff+2016)
    # =========================================================================
    print("\n" + "="*70)
    print(" 1. redMaPPer DES SV + SDSS DR8 (Rykoff+2016)")
    print("="*70)
    
    des_cats = download_catalog(REDMAPPER_DES_ID, "redMaPPer DES+SDSS")
    
    if des_cats:
        for i, cat in enumerate(des_cats):
            print(f"\n  Table {i}: {len(cat)} rows")
            print(f"  Columns: {cat.colnames}")
            
            # Save each table
            if len(cat) > 100:  # Only save substantial tables
                output_path = os.path.join(OUTPUT_DIR, f"redMaPPer_Rykoff2016_table{i}.fits")
                cat.write(output_path, format='fits', overwrite=True)
                print(f"  Saved: {output_path}")
                
                # Stats
                if 'lambda' in [c.lower() for c in cat.colnames]:
                    lam_col = [c for c in cat.colnames if 'lambda' in c.lower()][0]
                    lam = np.array(cat[lam_col])
                    lam_valid = lam[np.isfinite(lam) & (lam > 0)]
                    print(f"  Richness λ range: {lam_valid.min():.1f} - {lam_valid.max():.1f}")
                
                if 'z' in cat.colnames or 'zLambda' in cat.colnames:
                    z_col = 'z' if 'z' in cat.colnames else 'zLambda'
                    z = np.array(cat[z_col])
                    z_valid = z[np.isfinite(z) & (z > 0)]
                    if len(z_valid) > 0:
                        print(f"  Redshift range: {z_valid.min():.3f} - {z_valid.max():.3f}")
    
    # =========================================================================
    # 2. SDSS DR8 original (Rykoff+2014)
    # =========================================================================
    print("\n" + "="*70)
    print(" 2. redMaPPer SDSS DR8 (Rykoff+2014)")
    print("="*70)
    
    sdss_cats = download_catalog(REDMAPPER_SDSS_ID, "redMaPPer SDSS DR8")
    
    if sdss_cats:
        for i, cat in enumerate(sdss_cats):
            print(f"\n  Table {i}: {len(cat)} rows")
            
            if len(cat) > 100:
                output_path = os.path.join(OUTPUT_DIR, f"redMaPPer_Rykoff2014_table{i}.fits")
                cat.write(output_path, format='fits', overwrite=True)
                print(f"  Saved: {output_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    print("""
  redMaPPer advantages for Q²R² testing:
  
  1. PURE OPTICAL selection (no ICM bias)
  2. Richness λ is independent mass proxy
  3. ~26,000 clusters in SDSS DR8
  4. Well-calibrated mass-richness relation
  5. No distance-dependent selection effects
  
  Test strategy:
  - Observable: Richness λ
  - Mass: From stacked weak lensing or velocity dispersion
  - Environment: Cluster density or large-scale structure
  
  Next: Run test_cluster_redmapper.py
    """)
    
    print("="*70)


if __name__ == "__main__":
    main()
