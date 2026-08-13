#!/usr/bin/env python3
"""
===============================================================================
DOWNLOAD MCXC CATALOG
===============================================================================

Download the MCXC (Meta-Catalogue of X-ray Clusters) from VizieR.
This provides X-ray derived masses that are INDEPENDENT of SZ signal.

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

# MCXC catalog ID on VizieR
MCXC_VIZIER_ID = "J/A+A/534/A109"
MCXC_TABLE = "mcxc"

# MCXC-II (2024 update) - more clusters
MCXC2_VIZIER_ID = "J/A+A/688/A187"


def download_mcxc():
    """Download MCXC catalog from VizieR."""
    
    print("="*70)
    print(" DOWNLOADING MCXC CATALOG FROM VIZIER")
    print("="*70)
    
    # Configure Vizier to get all rows
    Vizier.ROW_LIMIT = -1
    
    print(f"\n  Querying VizieR for {MCXC_VIZIER_ID}...")
    
    try:
        catalogs = Vizier.get_catalogs(MCXC_VIZIER_ID)
        
        if len(catalogs) == 0:
            print("  ERROR: No catalogs found!")
            return None
        
        print(f"  Found {len(catalogs)} table(s)")
        
        # Get the main catalog
        mcxc = catalogs[0]
        print(f"  Table has {len(mcxc)} rows")
        print(f"  Columns: {mcxc.colnames}")
        
        # Save to FITS
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "MCXC_catalog.fits")
        
        # Convert to Table if needed
        if hasattr(mcxc, 'to_table'):
            mcxc = mcxc.to_table()
        
        mcxc.write(output_path, format='fits', overwrite=True)
        print(f"\n  Saved to: {output_path}")
        
        # Show some stats
        print("\n" + "="*70)
        print(" MCXC CATALOG SUMMARY")
        print("="*70)
        
        # Find mass column
        mass_cols = [c for c in mcxc.colnames if 'mass' in c.lower() or 'M500' in c]
        print(f"\n  Mass columns: {mass_cols}")
        
        # Find luminosity column
        lum_cols = [c for c in mcxc.colnames if 'lx' in c.lower() or 'L500' in c.lower()]
        print(f"  Luminosity columns: {lum_cols}")
        
        # Basic stats
        if 'z' in mcxc.colnames:
            z = mcxc['z']
            z_valid = z[~np.isnan(z) & (z > 0)]
            print(f"\n  Redshift range: {z_valid.min():.3f} - {z_valid.max():.3f}")
            print(f"  Median z: {np.median(z_valid):.3f}")
        
        return output_path
        
    except Exception as e:
        print(f"  ERROR downloading: {e}")
        print("\n  Alternative: Download manually from VizieR")
        print(f"  URL: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source={MCXC_VIZIER_ID}")
        return None


def check_psz2_mcxc_column():
    """Check if PSZ2 already has MCXC cross-match column."""
    
    print("\n" + "="*70)
    print(" CHECKING PSZ2 FOR MCXC CROSS-MATCH COLUMN")
    print("="*70)
    
    # Find PSZ2 file
    psz2_patterns = [
        os.path.join(OUTPUT_DIR, "COM_PCCS_SZ-Catalogs_vPR2", "HFI_PCCS_SZ-union_R2.08.fits.gz"),
        os.path.join(OUTPUT_DIR, "HFI_PCCS_SZ-union_R2.08.fits.gz"),
    ]
    
    psz2_path = None
    for p in psz2_patterns:
        if os.path.exists(p):
            psz2_path = p
            break
    
    if not psz2_path:
        print("  PSZ2 file not found")
        return
    
    from astropy.io import fits
    
    with fits.open(psz2_path) as hdul:
        data = hdul[1].data
        cols = data.names
        
        # Check for MCXC column
        mcxc_cols = [c for c in cols if 'MCXC' in c.upper()]
        
        if mcxc_cols:
            print(f"\n  ✓ Found MCXC column(s) in PSZ2: {mcxc_cols}")
            
            # Count matches
            mcxc_col = data['MCXC']
            
            # Count non-empty entries
            if mcxc_col.dtype.kind in ['S', 'U']:  # String type
                n_match = np.sum([len(str(x).strip()) > 0 and str(x).strip() != '' for x in mcxc_col])
            else:
                n_match = np.sum(~np.isnan(mcxc_col) & (mcxc_col != 0))
            
            print(f"  PSZ2 clusters with MCXC match: {n_match} / {len(data)}")
        else:
            print("  No MCXC column found in PSZ2")
            print(f"  Available columns: {cols}")


def main():
    # Download MCXC
    mcxc_path = download_mcxc()
    
    # Check PSZ2 for existing cross-match
    check_psz2_mcxc_column()
    
    if mcxc_path:
        print("\n" + "="*70)
        print(" NEXT STEPS")
        print("="*70)
        print("\n  1. Run test_cluster_planck_mcxc.py to cross-match and test")
        print("  2. This will use L_X from MCXC and Y_SZ from PSZ2")
        print("  3. The L_X - Y_SZ relation should show proper correlation")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
