#!/usr/bin/env python3
"""
===============================================================================
DOWNLOAD ADDITIONAL GROUP CATALOGS - RETRY
===============================================================================

Retry failed downloads and try alternative sources.
"""

import os
import time
import numpy as np
from astroquery.vizier import Vizier
from astropy.table import Table
import warnings
warnings.filterwarnings('ignore')

Vizier.ROW_LIMIT = -1

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "data", "level2_multicontext", "groups")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_tempel_retry():
    """Retry Tempel groups with timeout handling."""
    
    print("\n" + "="*70)
    print(" TEMPEL SDSS GROUPS - RETRY")
    print("="*70)
    
    vizier_ids = [
        "J/A+A/566/A1",    # Tempel+2014 SDSS DR10
        "J/A+A/588/A14",   # Tempel+2016 updated
        "J/A+A/602/A100",  # Tempel+2017
    ]
    
    for viz_id in vizier_ids:
        print(f"\n  Trying: {viz_id}")
        try:
            Vizier.TIMEOUT = 120  # Increase timeout
            catalog = Vizier.get_catalogs(viz_id)
            
            if catalog:
                print(f"    Found {len(catalog)} table(s)")
                
                for i, table in enumerate(catalog):
                    print(f"      Table {i}: {len(table)} rows")
                    print(f"      Columns: {table.colnames[:8]}...")
                    
                    # Save
                    fname = f"Tempel_groups_{viz_id.replace('/', '_')}_t{i}.fits"
                    output_path = os.path.join(OUTPUT_DIR, fname)
                    table.write(output_path, overwrite=True)
                    print(f"      Saved: {fname}")
                
                return True
                
        except Exception as e:
            print(f"    Error: {e}")
            time.sleep(2)  # Wait before retry
    
    return False


def download_saulder_groups():
    """
    Download Saulder et al. 2016 groups (2MRS).
    
    Reference: Saulder et al. 2016 (A&A 596, A14)
    2MASS Redshift Survey groups - local universe.
    """
    
    print("\n" + "="*70)
    print(" SAULDER 2MRS GROUPS")
    print("="*70)
    
    viz_id = "J/A+A/596/A14"
    
    print(f"  Downloading: {viz_id}")
    
    try:
        catalog = Vizier.get_catalogs(viz_id)
        
        if catalog:
            print(f"    Found {len(catalog)} table(s)")
            
            for i, table in enumerate(catalog):
                print(f"      Table {i}: {len(table)} rows - {table.colnames[:5]}...")
                
                output_path = os.path.join(OUTPUT_DIR, f"Saulder_2MRS_groups_t{i}.fits")
                table.write(output_path, overwrite=True)
                print(f"      Saved: {os.path.basename(output_path)}")
            
            return True
            
    except Exception as e:
        print(f"    Error: {e}")
    
    return False


def download_crook_groups():
    """
    Download Crook et al. 2007 groups.
    
    Reference: Crook et al. 2007 (ApJ 655, 790)
    """
    
    print("\n" + "="*70)
    print(" CROOK GROUPS (2MASS)")
    print("="*70)
    
    viz_id = "J/ApJ/655/790"
    
    print(f"  Downloading: {viz_id}")
    
    try:
        catalog = Vizier.get_catalogs(viz_id)
        
        if catalog:
            print(f"    Found {len(catalog)} table(s)")
            
            for i, table in enumerate(catalog):
                print(f"      Table {i}: {len(table)} rows - {table.colnames[:5]}...")
                
                output_path = os.path.join(OUTPUT_DIR, f"Crook_2MASS_groups_t{i}.fits")
                table.write(output_path, overwrite=True)
                print(f"      Saved: {os.path.basename(output_path)}")
            
            return True
            
    except Exception as e:
        print(f"    Error: {e}")
    
    return False


def download_knobel_cosmos():
    """
    Download Knobel et al. COSMOS groups.
    
    Reference: Knobel et al. 2012 (ApJ 753, 121)
    """
    
    print("\n" + "="*70)
    print(" KNOBEL COSMOS GROUPS")
    print("="*70)
    
    viz_id = "J/ApJ/753/121"
    
    print(f"  Downloading: {viz_id}")
    
    try:
        catalog = Vizier.get_catalogs(viz_id)
        
        if catalog:
            print(f"    Found {len(catalog)} table(s)")
            
            for i, table in enumerate(catalog):
                print(f"      Table {i}: {len(table)} rows - {table.colnames[:5]}...")
                
                output_path = os.path.join(OUTPUT_DIR, f"Knobel_COSMOS_groups_t{i}.fits")
                table.write(output_path, overwrite=True)
                print(f"      Saved: {os.path.basename(output_path)}")
            
            return True
            
    except Exception as e:
        print(f"    Error: {e}")
    
    return False


def download_robotham_gama_groups():
    """
    Download actual GAMA group catalog (not members).
    
    Reference: Robotham et al. 2011 (MNRAS 416, 2640)
    """
    
    print("\n" + "="*70)
    print(" ROBOTHAM GAMA GROUPS (actual groups, not members)")
    print("="*70)
    
    viz_ids = [
        "J/MNRAS/416/2640",  # Original paper
        "VIII/91",           # GAMA catalog
    ]
    
    for viz_id in viz_ids:
        print(f"\n  Trying: {viz_id}")
        try:
            catalog = Vizier.get_catalogs(viz_id)
            
            if catalog:
                print(f"    Found {len(catalog)} table(s)")
                
                for i, table in enumerate(catalog):
                    print(f"      Table {i}: {len(table)} rows - {table.colnames[:5]}...")
                    
                    output_path = os.path.join(OUTPUT_DIR, f"GAMA_groups_{viz_id.replace('/', '_')}_t{i}.fits")
                    table.write(output_path, overwrite=True)
                    print(f"      Saved: {os.path.basename(output_path)}")
                
                return True
                
        except Exception as e:
            print(f"    Error: {e}")
    
    return False


def main():
    print("="*70)
    print(" DOWNLOADING ADDITIONAL GROUP CATALOGS")
    print("="*70)
    
    results = {}
    
    results['tempel'] = download_tempel_retry()
    results['saulder'] = download_saulder_groups()
    results['crook'] = download_crook_groups()
    results['knobel'] = download_knobel_cosmos()
    results['gama_groups'] = download_robotham_gama_groups()
    
    # Summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ Failed"
        print(f"  {name.upper()}: {status}")
    
    # List all files
    import glob
    files = glob.glob(os.path.join(OUTPUT_DIR, "*.fits"))
    print(f"\n  Total files in {OUTPUT_DIR}:")
    for f in files:
        print(f"    - {os.path.basename(f)}")
    
    print("\n  Next: Run test_groups_qr2.py")
    print("="*70)


if __name__ == "__main__":
    main()
