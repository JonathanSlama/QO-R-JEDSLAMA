#!/usr/bin/env python3
"""
Download ALFALFA a100 catalog from VizieR.
"""

import os
from astroquery.vizier import Vizier
from astropy.table import Table

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "data", "alfalfa")

def download_alfalfa():
    """Download ALFALFA a100 catalog."""
    
    print("="*70)
    print(" DOWNLOADING ALFALFA a100 CATALOG")
    print("="*70)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # ALFALFA a100 catalog: J/ApJ/861/49
    print("\n  Querying VizieR for ALFALFA a100...")
    
    Vizier.ROW_LIMIT = -1  # No limit
    
    try:
        catalogs = Vizier.get_catalogs('J/ApJ/861/49')
        print(f"  Found {len(catalogs)} tables")
        
        for i, cat in enumerate(catalogs):
            print(f"    Table {i}: {len(cat)} rows, columns: {cat.colnames[:5]}...")
            
            # Save each table
            output_path = os.path.join(DATA_DIR, f"alfalfa_a100_t{i}.fits")
            cat.write(output_path, format='fits', overwrite=True)
            print(f"    Saved: {output_path}")
        
        # The main table is usually t0 or t1
        main_table = catalogs[0]
        main_path = os.path.join(DATA_DIR, "alfalfa_a100.fits")
        main_table.write(main_path, format='fits', overwrite=True)
        print(f"\n  Main catalog: {main_path}")
        print(f"  Total sources: {len(main_table)}")
        
        return main_path
        
    except Exception as e:
        print(f"  ERROR: {e}")
        print("\n  Trying alternative method...")
        
        # Try direct URL download
        import urllib.request
        
        url = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/861/49/table1.dat.gz"
        print(f"  Downloading from: {url}")
        
        # This may not work directly - VizieR access is preferred
        return None


def main():
    download_alfalfa()
    print("\n" + "="*70)
    print(" DOWNLOAD COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
