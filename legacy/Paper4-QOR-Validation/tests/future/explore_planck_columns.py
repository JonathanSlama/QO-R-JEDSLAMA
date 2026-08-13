#!/usr/bin/env python3
"""
Explore all Planck SZ catalog files to find mass columns (MSZ).
"""

import os
import glob
from astropy.io import fits

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "level2_multicontext", "clusters", "COM_PCCS_SZ-Catalogs_vPR2")

# Find all FITS files
patterns = ["*.fits", "*.fits.gz"]
all_files = []
for p in patterns:
    all_files.extend(glob.glob(os.path.join(DATA_DIR, p)))

print("="*80)
print(" PLANCK SZ CATALOG EXPLORATION")
print(" Searching for mass columns (MSZ, M500, MASS, etc.)")
print("="*80)

mass_keywords = ['MSZ', 'MASS', 'M500', 'M_500', 'M200', 'M_200', 'MTOT']

found_mass_columns = {}

for filepath in sorted(all_files):
    filename = os.path.basename(filepath)
    print(f"\n{'='*80}")
    print(f"FILE: {filename}")
    print("="*80)
    
    try:
        with fits.open(filepath) as hdul:
            print(f"  HDU count: {len(hdul)}")
            
            for i, hdu in enumerate(hdul):
                if hasattr(hdu, 'columns') and hdu.columns is not None:
                    cols = [c.name for c in hdu.columns]
                    print(f"\n  HDU {i} ({hdu.name}): {len(cols)} columns")
                    print(f"  Columns: {cols}")
                    
                    # Search for mass-related columns
                    mass_cols = [c for c in cols if any(mk in c.upper() for mk in mass_keywords)]
                    if mass_cols:
                        print(f"\n  *** MASS COLUMNS FOUND: {mass_cols} ***")
                        found_mass_columns[filename] = mass_cols
                    
                    # Also look for interesting columns
                    interesting = ['Y5R500', 'Y500', 'REDSHIFT', 'Z', 'SNR', 'RA', 'DEC', 'GLON', 'GLAT']
                    found_interesting = [c for c in cols if c in interesting]
                    if found_interesting:
                        print(f"  Key columns: {found_interesting}")
                        
                    # If we find data, show some stats
                    if hasattr(hdu, 'data') and hdu.data is not None:
                        print(f"  Rows: {len(hdu.data)}")
                        
    except Exception as e:
        print(f"  ERROR reading file: {e}")

print("\n" + "="*80)
print(" SUMMARY - FILES WITH MASS COLUMNS")
print("="*80)

if found_mass_columns:
    for fname, mcols in found_mass_columns.items():
        print(f"\n  {fname}:")
        print(f"    Mass columns: {mcols}")
else:
    print("\n  NO MASS COLUMNS FOUND in any file!")
    print("\n  Options:")
    print("  1. Calculate M_500 from Y_500 using Y-M scaling relation")
    print("  2. Download a different catalog (e.g., PSZ2 with external mass estimates)")
    print("  3. Cross-match with X-ray catalogs (e.g., MCXC) that have hydrostatic masses")

print("\n" + "="*80)
