#!/usr/bin/env python3
"""
Quick exploration of downloaded group catalogs.
"""

import os
import glob
import numpy as np
from astropy.table import Table

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "data", "level2_multicontext", "groups")

def main():
    print("="*70)
    print(" EXPLORING DOWNLOADED GROUP CATALOGS")
    print("="*70)
    
    files = glob.glob(os.path.join(DATA_DIR, "*.fits"))
    
    for f in files:
        print(f"\n{'='*70}")
        print(f" FILE: {os.path.basename(f)}")
        print("="*70)
        
        data = Table.read(f)
        
        print(f"\n  Total rows: {len(data)}")
        print(f"\n  All columns:")
        for i, col in enumerate(data.colnames):
            print(f"    {i+1:3d}. {col}")
        
        print(f"\n  Sample data (first 3 rows):")
        print(data[:3])
        
        # Check for group-related columns
        print(f"\n  Looking for group-related columns...")
        
        group_keywords = ['group', 'grp', 'nfof', 'ngal', 'mass', 'halo', 'mult', 'rich']
        found_cols = []
        for col in data.colnames:
            for kw in group_keywords:
                if kw in col.lower():
                    found_cols.append(col)
                    break
        
        if found_cols:
            print(f"    Found: {found_cols}")
        else:
            print(f"    None found - might be galaxy catalog, not group catalog")
        
        # Check z distribution
        z_cols = [c for c in data.colnames if c.lower() == 'z' or 'redshift' in c.lower()]
        if z_cols:
            z = np.array(data[z_cols[0]])
            valid = z[np.isfinite(z) & (z > 0) & (z < 2)]
            print(f"\n  Redshift ({z_cols[0]}): {len(valid)} valid, range {valid.min():.3f} - {valid.max():.3f}")


if __name__ == "__main__":
    main()
