#!/usr/bin/env python3
"""
===============================================================================
DOWNLOAD ACT-DR5 MCMF CLUSTER CATALOG
===============================================================================

Download the ACT-DR5 MCMF cluster catalog (Klein+2024) from VizieR.
6,237 clusters - the largest tSZE-selected cluster catalog!

This will allow us to test Q²R² with ~11x more clusters than Planck×MCXC.

Author: Jonathan E. Slama
Date: December 2025
===============================================================================
"""

import os
import sys

# Check if astroquery is available
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

# ACT-DR5 MCMF catalog ID on VizieR
ACT_VIZIER_ID = "J/A+A/690/A322"


def download_act_dr5_mcmf():
    """Download ACT-DR5 MCMF catalog from VizieR."""
    
    print("="*70)
    print(" DOWNLOADING ACT-DR5 MCMF CLUSTER CATALOG FROM VIZIER")
    print("="*70)
    print("\n  Reference: Klein et al. (2024), A&A, 690, A322")
    print("  6,237 tSZE-selected clusters with optical confirmation")
    
    # Configure Vizier to get all rows
    Vizier.ROW_LIMIT = -1
    
    print(f"\n  Querying VizieR for {ACT_VIZIER_ID}...")
    
    try:
        catalogs = Vizier.get_catalogs(ACT_VIZIER_ID)
        
        if len(catalogs) == 0:
            print("  ERROR: No catalogs found!")
            print("\n  Alternative: Download manually from VizieR")
            print(f"  URL: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source={ACT_VIZIER_ID}")
            return None
        
        print(f"  Found {len(catalogs)} table(s)")
        
        # Find the main cluster catalog
        for i, cat in enumerate(catalogs):
            print(f"\n  Table {i}: {len(cat)} rows")
            print(f"    Columns: {cat.colnames[:10]}...")  # First 10 columns
        
        # Usually the main catalog is the largest one
        main_cat = max(catalogs, key=len)
        print(f"\n  Using main catalog with {len(main_cat)} entries")
        print(f"  All columns: {main_cat.colnames}")
        
        # Save to FITS
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "ACT_DR5_MCMF_catalog.fits")
        
        # Convert to Table if needed
        if hasattr(main_cat, 'to_table'):
            main_cat = main_cat.to_table()
        
        main_cat.write(output_path, format='fits', overwrite=True)
        print(f"\n  Saved to: {output_path}")
        
        # Show some stats
        print("\n" + "="*70)
        print(" ACT-DR5 MCMF CATALOG SUMMARY")
        print("="*70)
        
        # Find key columns
        print(f"\n  Total clusters: {len(main_cat)}")
        
        # Redshift
        z_cols = [c for c in main_cat.colnames if 'z' in c.lower() or 'redshift' in c.lower()]
        print(f"  Redshift columns: {z_cols}")
        
        if z_cols:
            z_col = z_cols[0]
            z = np.array(main_cat[z_col])
            z_valid = z[np.isfinite(z) & (z > 0)]
            if len(z_valid) > 0:
                print(f"  Redshift range: {z_valid.min():.3f} - {z_valid.max():.3f}")
                print(f"  Median z: {np.median(z_valid):.3f}")
                print(f"  N(z > 1): {(z_valid > 1).sum()}")
        
        # Mass columns
        mass_cols = [c for c in main_cat.colnames if 'mass' in c.lower() or 'M500' in c]
        print(f"  Mass columns: {mass_cols}")
        
        # Y_SZ columns
        y_cols = [c for c in main_cat.colnames if 'y' in c.lower() or 'Y500' in c.lower()]
        print(f"  Y_SZ columns: {y_cols}")
        
        # SNR
        snr_cols = [c for c in main_cat.colnames if 'snr' in c.lower() or 'S/N' in c.lower()]
        print(f"  SNR columns: {snr_cols}")
        
        return output_path
        
    except Exception as e:
        print(f"  ERROR downloading: {e}")
        print("\n  Alternative: Download manually from VizieR")
        print(f"  URL: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source={ACT_VIZIER_ID}")
        return None


def main():
    print("\n" + "="*70)
    print(" ACT-DR5 MCMF: THE LARGEST tSZE CLUSTER CATALOG")
    print("="*70)
    print("""
  Why ACT-DR5 MCMF?
  
  Current sample (Planck × MCXC): 551 clusters
  ACT-DR5 MCMF: 6,237 clusters (×11 more!)
  
  Expected improvement in U-shape detection:
  - Current: a = 0.020 ± 0.015 → 1.3σ
  - With ACT: a = 0.020 ± 0.0045 → 4.4σ (if signal persists)
  
  This could be the THIRD independent context confirming String Theory!
    """)
    
    act_path = download_act_dr5_mcmf()
    
    if act_path:
        print("\n" + "="*70)
        print(" NEXT STEPS")
        print("="*70)
        print("""
  1. Run test_cluster_act_dr5.py to test for U-shape
  2. Apply distance correction (Y_scaled = Y_obs × D_A² / E(z)^(2/3))
  3. Look for Q²R² pattern in residuals
  
  Expected significance: 4-5σ (if U-shape is real)
        """)
    
    print("="*70)


if __name__ == "__main__":
    main()
