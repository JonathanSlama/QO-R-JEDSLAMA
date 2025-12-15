#!/usr/bin/env python3
"""
===============================================================================
DOWNLOAD GALAXY GROUP CATALOGS
===============================================================================

Galaxy groups (10¹²-10¹⁴ M☉) are the CRITICAL intermediate scale between:
- Galaxies (10¹⁰-10¹² M☉): Full Q²R² signature (U-shape + inversion)
- Clusters (10¹⁴-10¹⁵ M☉): Partial signature (U-shape only)

Catalogs to download:
1. GAMA Galaxy Group Catalogue (Robotham+2011, updated versions)
2. Yang SDSS DR7 Group Catalogue (~470,000 groups)

PREDICTION:
If baryonic coupling hypothesis is correct, groups should show:
- U-shape: YES (like clusters)
- Sign inversion: WEAK/PARTIAL (transition regime)

Author: Jonathan E. Slama
Date: December 2025
===============================================================================
"""

import os
import numpy as np
from astroquery.vizier import Vizier
from astropy.table import Table
import warnings
warnings.filterwarnings('ignore')

# Increase row limit
Vizier.ROW_LIMIT = -1

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "data", "level2_multicontext", "groups")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_yang_sdss_groups():
    """
    Download Yang et al. SDSS DR7 Galaxy Group Catalogue.
    
    Reference: Yang et al. 2007 (ApJ 671, 153)
    Updated: Yang et al. 2012 (ApJ 752, 41)
    
    This is the largest group catalog available (~470,000 groups)
    with halo masses from abundance matching.
    """
    
    print("\n" + "="*70)
    print(" 1. YANG SDSS DR7 GROUP CATALOGUE")
    print("="*70)
    
    print("""
  Yang et al. group finder uses:
  - Friends-of-friends algorithm
  - Halo mass from abundance matching
  - ~470,000 groups in SDSS DR7
  
  Key variables:
  - Mh: Halo mass (from abundance matching)
  - Ngal: Number of member galaxies
  - L_group: Total group luminosity
  - sigma_v: Velocity dispersion
    """)
    
    # Try VizieR catalogs
    vizier_ids = [
        "J/ApJ/752/41",   # Yang+2012 updated
        "J/ApJ/671/153",  # Yang+2007 original
    ]
    
    for viz_id in vizier_ids:
        print(f"  Trying VizieR ID: {viz_id}")
        try:
            catalog = Vizier.get_catalogs(viz_id)
            if catalog:
                print(f"  Found {len(catalog)} table(s)")
                for i, table in enumerate(catalog):
                    print(f"    Table {i}: {len(table)} rows - {table.colnames[:5]}...")
                
                # Save the main table
                main_table = catalog[0]
                output_path = os.path.join(OUTPUT_DIR, f"Yang_SDSS_groups_{viz_id.replace('/', '_')}.fits")
                main_table.write(output_path, overwrite=True)
                print(f"  Saved: {output_path}")
                return True
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n  NOTE: Yang catalog may need direct download from:")
    print("  https://gax.sjtu.edu.cn/data/Group.html")
    
    return False


def download_gama_groups():
    """
    Download GAMA Galaxy Group Catalogue.
    
    Reference: Robotham et al. 2011 (MNRAS 416, 2640)
    
    GAMA has excellent spectroscopic completeness and
    well-characterized group properties.
    """
    
    print("\n" + "="*70)
    print(" 2. GAMA GALAXY GROUP CATALOGUE")
    print("="*70)
    
    print("""
  GAMA (Galaxy And Mass Assembly) groups:
  - High spectroscopic completeness (>98%)
  - ~24,000 groups
  - Well-calibrated halo masses
  
  Key variables:
  - GroupMass: Halo mass estimate
  - Nfof: Number of members
  - IterCenRA/Dec: Group center
  - Zfof: Group redshift
    """)
    
    # GAMA data is typically accessed via their database
    # Try VizieR first
    vizier_ids = [
        "J/MNRAS/416/2640",  # Robotham+2011
        "J/MNRAS/474/3875",  # GAMA DR3 groups
    ]
    
    for viz_id in vizier_ids:
        print(f"  Trying VizieR ID: {viz_id}")
        try:
            catalog = Vizier.get_catalogs(viz_id)
            if catalog:
                print(f"  Found {len(catalog)} table(s)")
                for i, table in enumerate(catalog):
                    print(f"    Table {i}: {len(table)} rows - {table.colnames[:5]}...")
                
                # Save tables
                for i, table in enumerate(catalog):
                    output_path = os.path.join(OUTPUT_DIR, f"GAMA_groups_{viz_id.replace('/', '_')}_table{i}.fits")
                    table.write(output_path, overwrite=True)
                    print(f"  Saved: {output_path}")
                return True
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n  NOTE: GAMA data may need access via:")
    print("  http://www.gama-survey.org/dr3/")
    
    return False


def download_tempel_groups():
    """
    Download Tempel et al. SDSS groups.
    
    Reference: Tempel et al. 2014 (A&A 566, A1)
    
    Alternative group catalog with different algorithm.
    """
    
    print("\n" + "="*70)
    print(" 3. TEMPEL SDSS GROUPS (Alternative)")
    print("="*70)
    
    vizier_id = "J/A+A/566/A1"
    
    print(f"  Downloading from VizieR: {vizier_id}")
    
    try:
        catalog = Vizier.get_catalogs(vizier_id)
        if catalog:
            print(f"  Found {len(catalog)} table(s)")
            for i, table in enumerate(catalog):
                print(f"    Table {i}: {len(table)} rows - {table.colnames[:5]}...")
                
                output_path = os.path.join(OUTPUT_DIR, f"Tempel_SDSS_groups_table{i}.fits")
                table.write(output_path, overwrite=True)
                print(f"  Saved: {output_path}")
            
            return True
    except Exception as e:
        print(f"  Error: {e}")
    
    return False


def download_berlind_groups():
    """
    Download Berlind et al. SDSS DR7 groups.
    
    Reference: Berlind et al. 2006 (ApJS 167, 1)
    """
    
    print("\n" + "="*70)
    print(" 4. BERLIND SDSS GROUPS")
    print("="*70)
    
    vizier_id = "J/ApJS/167/1"
    
    print(f"  Downloading from VizieR: {vizier_id}")
    
    try:
        catalog = Vizier.get_catalogs(vizier_id)
        if catalog:
            print(f"  Found {len(catalog)} table(s)")
            for i, table in enumerate(catalog):
                print(f"    Table {i}: {len(table)} rows - {table.colnames[:5]}...")
                
                output_path = os.path.join(OUTPUT_DIR, f"Berlind_SDSS_groups_table{i}.fits")
                table.write(output_path, overwrite=True)
                print(f"  Saved: {output_path}")
            
            return True
    except Exception as e:
        print(f"  Error: {e}")
    
    return False


def main():
    print("="*70)
    print(" DOWNLOADING GALAXY GROUP CATALOGUES")
    print("="*70)
    
    print("""
  Galaxy groups are the CRITICAL intermediate scale!
  
  Mass scale hierarchy:
  - Galaxies:  10¹⁰ - 10¹² M☉  → Full Q²R² (U-shape + inversion)
  - GROUPS:    10¹² - 10¹⁴ M☉  → TRANSITION REGIME (testing!)
  - Clusters:  10¹⁴ - 10¹⁵ M☉  → Partial Q²R² (U-shape only)
  
  PREDICTION: Groups should show WEAK sign inversion
    """)
    
    results = {}
    
    # Try different catalogs
    results['yang'] = download_yang_sdss_groups()
    results['gama'] = download_gama_groups()
    results['tempel'] = download_tempel_groups()
    results['berlind'] = download_berlind_groups()
    
    # Summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    print("\n  Download results:")
    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ Not available via VizieR"
        print(f"    {name.upper()}: {status}")
    
    print(f"\n  Files saved to: {OUTPUT_DIR}")
    
    # List downloaded files
    import glob
    files = glob.glob(os.path.join(OUTPUT_DIR, "*.fits"))
    if files:
        print(f"\n  Downloaded files:")
        for f in files:
            print(f"    - {os.path.basename(f)}")
    
    print("\n  Next: Run test_groups_qr2.py")
    print("="*70)


if __name__ == "__main__":
    main()
