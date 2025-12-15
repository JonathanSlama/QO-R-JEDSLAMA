#!/usr/bin/env python3
"""
Inspect all downloaded UDG catalogs and verify data completeness.
"""
from astropy.io import fits
from astropy.table import Table
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "udg_catalogs")

print("="*70)
print(" UDG CATALOG INSPECTION")
print(" Author: Jonathan Édouard Slama")
print("="*70)

catalogs = [
    ("SMUDGes_t0.fits", "SMUDGes (Zaritsky+2023)"),
    ("MATLAS_dwarfs_t0.fits", "MATLAS Dwarfs (Poulain+2021)"),
    ("MATLAS_dwarfs_t1.fits", "MATLAS Nuclei"),
    ("MATLAS_UDG_t0.fits", "MATLAS UDG (Marleau+2021)"),
    ("MATLAS_UDG_t1.fits", "MATLAS UDG Table 2"),
]

# Key columns we need for QO+R test
KEY_COLUMNS = {
    'position': ['RA', 'DE', 'RAJ2000', 'DEJ2000', 'RAdeg', 'DEdeg'],
    'size': ['Re', 'Reff', 'r_eff', 'R_eff', 'reff'],
    'surface_brightness': ['mu0', 'mu0g', 'mu0r', 'SBe', 'mueg', '<mueg>'],
    'velocity': ['Vhel', 'Vrad', 'HRV', 'cz', 'Vel', 'RV'],
    'distance': ['Dist', 'D', 'Dmpc', 'dist'],
    'mass': ['logM', 'Mstar', 'M_star', 'logMstar'],
    'environment': ['Env', 'Host', 'Group', 'Cluster'],
}

for filename, description in catalogs:
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"\n{filename} NOT FOUND")
        continue
    
    print(f"\n{'='*70}")
    print(f" {description}")
    print(f" File: {filename}")
    print("="*70)
    
    t = Table.read(filepath)
    print(f"N objects: {len(t)}")
    print(f"N columns: {len(t.colnames)}")
    
    # Check for key columns
    print("\nKey columns found:")
    for category, possible_names in KEY_COLUMNS.items():
        found = []
        for col in t.colnames:
            for name in possible_names:
                if name.lower() in col.lower():
                    found.append(col)
                    break
        if found:
            print(f"  {category}: {', '.join(found[:3])}")
        else:
            print(f"  {category}: NOT FOUND")
    
    # Show all columns
    print(f"\nAll columns: {', '.join(t.colnames[:15])}")
    if len(t.colnames) > 15:
        print(f"             ... and {len(t.colnames)-15} more")

print("\n" + "="*70)
print(" ASSESSMENT FOR QO+R KILLER PREDICTION #1")
print("="*70)
print("""
For the UDG test we need:
1. Position (RA, DEC) - to cross-match with environment
2. Size (R_eff > 1.5 kpc) - UDG definition
3. Surface brightness (mu_0 > 24 mag/arcsec^2) - UDG definition
4. Velocity - for BTFR (limited availability)
5. Environment density - to test U-shape

STRATEGY:
- Use MATLAS UDGs (59) with assumed host distances
- Cross-match with environment catalogs (groups, clusters)
- Compute environmental density proxy
- Test for inverted U-shape (a < 0 predicted)
""")
