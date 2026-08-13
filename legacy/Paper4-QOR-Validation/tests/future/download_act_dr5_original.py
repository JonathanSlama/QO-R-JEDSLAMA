#!/usr/bin/env python3
"""
===============================================================================
DOWNLOAD ACT-DR5 ORIGINAL CATALOG (Hilton et al. 2021)
===============================================================================

Download from LAMBDA to get SNR and potentially Y_500 values.
This catalog has 4,195 clusters.

We'll cross-match with ACT-DR5 MCMF (6,237 clusters) to get:
- From MCMF: Richness (L), redshift photo
- From DR5: SNR, Mass

Author: Jonathan E. Slama
Date: December 2025
===============================================================================
"""

import os
import sys
import urllib.request
from astropy.io import fits
from astropy.table import Table
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "data", "level2_multicontext", "clusters")

# ACT-DR5 original catalog from LAMBDA
ACT_DR5_URL = "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/DR5_cluster-catalog_v1.1.fits"


def download_act_dr5_original():
    """Download ACT-DR5 original catalog from LAMBDA."""
    
    print("="*70)
    print(" DOWNLOADING ACT-DR5 ORIGINAL CATALOG FROM LAMBDA")
    print("="*70)
    print("\n  Reference: Hilton et al. (2021), ApJS, 253, 3")
    print("  4,195 SZ-selected clusters with SNR > 4")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "ACT_DR5_Hilton2021_catalog.fits")
    
    if os.path.exists(output_path):
        print(f"\n  File already exists: {output_path}")
        print("  Skipping download...")
    else:
        print(f"\n  Downloading from LAMBDA...")
        print(f"  URL: {ACT_DR5_URL}")
        
        try:
            urllib.request.urlretrieve(ACT_DR5_URL, output_path)
            print(f"  Saved to: {output_path}")
        except Exception as e:
            print(f"  ERROR: {e}")
            return None
    
    # Explore the catalog
    print("\n" + "="*70)
    print(" ACT-DR5 ORIGINAL CATALOG EXPLORATION")
    print("="*70)
    
    data = Table.read(output_path)
    print(f"\n  Total clusters: {len(data)}")
    print(f"\n  All columns ({len(data.colnames)}):")
    
    for i, col in enumerate(data.colnames):
        print(f"    {i+1:2d}. {col}")
    
    # Key columns for our test
    print("\n  Key columns for Q²R² test:")
    
    # SNR
    if 'SNR' in data.colnames:
        snr = np.array(data['SNR'])
        snr_valid = snr[np.isfinite(snr)]
        print(f"    SNR: range {snr_valid.min():.1f} - {snr_valid.max():.1f}")
    
    # Mass
    mass_cols = [c for c in data.colnames if 'M500' in c or 'mass' in c.lower()]
    print(f"    Mass columns: {mass_cols}")
    
    # y0 (central Compton-y)
    y_cols = [c for c in data.colnames if 'y0' in c.lower() or 'y500' in c.lower() or 'yc' in c.lower()]
    print(f"    Y columns: {y_cols}")
    
    # Redshift
    z_cols = [c for c in data.colnames if 'redshift' in c.lower() or c == 'z']
    print(f"    Redshift columns: {z_cols}")
    
    # Coordinates
    coord_cols = [c for c in data.colnames if 'ra' in c.lower() or 'dec' in c.lower()]
    print(f"    Coordinate columns: {coord_cols}")
    
    return output_path, data


def main():
    print("\n" + "="*70)
    print(" ACT-DR5 ORIGINAL: SNR AND MASS DATA")
    print("="*70)
    print("""
  Why download ACT-DR5 original (Hilton 2021)?
  
  - Contains SNR (signal-to-noise of SZ detection)
  - SNR is proportional to Y_500 / noise
  - Can be used as a proxy for SZ signal strength
  
  Combined with MCMF richness, we can test:
  - Option 1: Mass vs Richness (M-L)
  - Option 2: SNR vs Richness (proxy for Y-L)
    """)
    
    result = download_act_dr5_original()
    
    if result:
        path, data = result
        print("\n" + "="*70)
        print(" NEXT STEPS")
        print("="*70)
        print("""
  1. Cross-match ACT-DR5 original with ACT-DR5 MCMF
  2. Test M-L relation (already available in MCMF)
  3. Test SNR-L relation (from cross-match)
  4. Look for U-shape in both relations
        """)
    
    print("="*70)


if __name__ == "__main__":
    main()
