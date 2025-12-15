#!/usr/bin/env python3
"""
===============================================================================
DOWNLOAD eROSITA eRASS1 CLUSTER CATALOG
===============================================================================

Download eRASS1 X-ray cluster catalog from VizieR (Bulbul et al. 2024)
~12,000 clusters with X-ray masses INDEPENDENT of SZ signal.

Reference: 
- Bulbul et al. (2024), A&A 685, A106
- Kluge et al. (2024), A&A 688, A210 (optical IDs)

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

# eRASS1 catalogs on VizieR
ERASS1_MAIN_ID = "J/A+A/682/A34"      # Main source catalog (Merloni+2024)
ERASS1_CLUSTER_ID = "J/A+A/685/A106"  # Cluster catalog (Bulbul+2024)
ERASS1_OPTICAL_ID = "J/A+A/688/A210"  # Optical IDs (Kluge+2024)


def download_catalog(vizier_id, name, description):
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
        
        # Show all tables
        for i, cat in enumerate(catalogs):
            print(f"    Table {i}: {len(cat)} rows - {cat.colnames[:5]}...")
        
        return catalogs
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    print("="*70)
    print(" DOWNLOADING eROSITA eRASS1 CLUSTER CATALOGS")
    print("="*70)
    print("""
  eROSITA First All-Sky Survey (eRASS1) - Western Galactic Hemisphere
  Released: January 2024
  
  Key advantage: X-ray masses are INDEPENDENT of SZ signal!
  No Malmquist bias like Planck.
    """)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # =========================================================================
    # 1. Main cluster catalog (Bulbul+2024)
    # =========================================================================
    print("\n" + "="*70)
    print(" 1. eRASS1 CLUSTER CATALOG (Bulbul+2024)")
    print("="*70)
    
    cluster_cats = download_catalog(
        ERASS1_CLUSTER_ID,
        "eRASS1 Clusters",
        "Galaxy clusters and groups"
    )
    
    if cluster_cats:
        # Find the main cluster table
        main_cat = max(cluster_cats, key=len)
        print(f"\n  Main table: {len(main_cat)} entries")
        print(f"  Columns: {main_cat.colnames}")
        
        # Save
        output_path = os.path.join(OUTPUT_DIR, "eRASS1_clusters_Bulbul2024.fits")
        main_cat.write(output_path, format='fits', overwrite=True)
        print(f"\n  Saved: {output_path}")
        
        # Stats
        if 'z' in main_cat.colnames or 'Redshift' in main_cat.colnames:
            z_col = 'z' if 'z' in main_cat.colnames else 'Redshift'
            z = np.array(main_cat[z_col])
            z_valid = z[np.isfinite(z) & (z > 0)]
            print(f"\n  Redshift range: {z_valid.min():.3f} - {z_valid.max():.3f}")
            print(f"  Median z: {np.median(z_valid):.3f}")
    
    # =========================================================================
    # 2. Optical identifications (Kluge+2024)
    # =========================================================================
    print("\n" + "="*70)
    print(" 2. eRASS1 OPTICAL IDs (Kluge+2024)")
    print("="*70)
    
    optical_cats = download_catalog(
        ERASS1_OPTICAL_ID,
        "eRASS1 Optical",
        "Optical identification and properties"
    )
    
    if optical_cats:
        main_cat = max(optical_cats, key=len)
        print(f"\n  Main table: {len(main_cat)} entries")
        
        output_path = os.path.join(OUTPUT_DIR, "eRASS1_optical_Kluge2024.fits")
        main_cat.write(output_path, format='fits', overwrite=True)
        print(f"  Saved: {output_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    print("""
  eRASS1 advantages for Q²R² testing:
  
  1. X-ray selection (no SZ bias)
  2. Independent mass estimates from L_X
  3. Optical richness from eROMaPPer
  4. ~12,000 clusters (western hemisphere)
  
  Next: Run test_cluster_erosita.py
    """)
    
    print("="*70)


if __name__ == "__main__":
    main()
