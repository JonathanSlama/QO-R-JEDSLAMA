#!/usr/bin/env python3
"""
Inspect SMUDGes UDG Catalog
"""
from astropy.io import fits
from astropy.table import Table
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "udg_catalogs")
SMUDGES_PATH = os.path.join(DATA_DIR, "SMUDGes_t0.fits")

print("="*70)
print(" INSPECTING SMUDGes CATALOG")
print(" Real UDG data from Zaritsky et al.")
print("="*70)

t = Table.read(SMUDGES_PATH)

print(f"\nN objects: {len(t)}")
print(f"\nColumns ({len(t.colnames)}):")
for col in t.colnames:
    dtype = t[col].dtype
    print(f"  - {col:<20} ({dtype})")

print("\n" + "="*70)
print(" FIRST 5 ROWS")
print("="*70)
print(t[:5])

print("\n" + "="*70)
print(" STATISTICS")
print("="*70)

# Check for key columns
key_cols = ['RA', 'DE', 'RAJ2000', 'DEJ2000', 'Reff', 'r_eff', 'mu', 'SBe', 'Dist', 'Vhel']
for col in t.colnames:
    for key in key_cols:
        if key.lower() in col.lower():
            try:
                data = t[col]
                print(f"{col}: min={min(data):.2f}, max={max(data):.2f}, mean={sum(data)/len(data):.2f}")
            except:
                print(f"{col}: (non-numeric)")
            break

print("\n" + "="*70)
print(" ASSESSMENT FOR QO+R TEST")
print("="*70)
print("We need: RA, DEC, R_eff, surface brightness, velocity or distance")
print("Check columns above to see what's available.")
