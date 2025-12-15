#!/usr/bin/env python3
"""
===============================================================================
DOWNLOAD UDG CATALOGS - COMPLETE VERSION
===============================================================================
Downloads all available UDG catalogs including spectroscopic data.

Key catalogs:
1. SMUDGes (Zaritsky+2023) - 226 UDGs with photometry
2. Gannon+2024 - Spectroscopic catalog with VELOCITIES and ENVIRONMENT
3. MATLAS (Poulain+2021) - 2210 dwarfs including UDGs

Author: Jonathan Édouard Slama
Affiliation: Metafund Research Division, Strasbourg, France
===============================================================================
"""

import os
import sys
import requests
import time

try:
    from astropy.table import Table
    from astropy.io import fits, ascii
except ImportError:
    os.system("pip install astropy")
    from astropy.table import Table
    from astropy.io import fits, ascii

try:
    from astroquery.vizier import Vizier
    VIZIER_OK = True
except:
    VIZIER_OK = False

# Output directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "udg_catalogs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_with_timeout(catalog_id, name, timeout=45):
    """Download from VizieR with timeout."""
    if not VIZIER_OK:
        return None
    
    print(f"   Querying {name} ({catalog_id})...")
    
    try:
        Vizier.TIMEOUT = timeout
        v = Vizier(columns=['*'], row_limit=-1)
        v.TIMEOUT = timeout
        
        result = v.query_constraints(catalog=catalog_id)
        
        if result and len(result) > 0:
            print(f"   Found {len(result)} table(s), {sum(len(t) for t in result)} total rows")
            return result
        else:
            print(f"   No results")
            return None

    except Exception as e:
        print(f"   Error: {str(e)[:60]}")
        return None


def download_gannon_catalog():
    """
    Download Gannon+2024 UDG spectroscopic catalog.
    Contains: velocities, velocity dispersions, environment, stellar mass, age, metallicity
    """
    print("\n" + "="*70)
    print(" GANNON+2024 SPECTROSCOPIC UDG CATALOG")
    print(" Key data: velocities, environment, stellar populations")
    print("="*70)
    
    # Try VizieR first (J/MNRAS/531/1856)
    result = download_with_timeout("J/MNRAS/531/1856", "Gannon+2024")
    
    if result:
        for i, t in enumerate(result):
            output_path = os.path.join(OUTPUT_DIR, f"Gannon2024_UDG_spectroscopic_t{i}.fits")
            t.write(output_path, format='fits', overwrite=True)
            print(f"   Saved: {os.path.basename(output_path)}")
        return True
    
    # Alternative: Try direct URL from MNRAS supplementary material
    print("   Trying direct download from MNRAS...")
    urls_to_try = [
        "https://academic.oup.com/mnras/article/531/1/1856/7676195#supplementary-data",
        "https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/531/1856/",
    ]
    
    # Create a placeholder with expected columns
    print("   Creating template with expected Gannon catalog structure...")
    gannon_template = {
        'Name': ['Example_UDG'],
        'RA': [180.0],
        'DEC': [0.0],
        'Environment': [2],  # 1=cluster, 2=group, 3=field
        'Distance_Mpc': [20.0],
        'M_V': [-15.0],
        'mu_e_V': [25.0],
        'log_Mstar': [7.5],
        'R_eff_kpc': [2.0],
        'b_a': [0.7],
        'V_recession_kms': [1500.0],
        'sigma_star_kms': [20.0],
        'N_GC': [10],
        'Age_Gyr': [8.0],
        'MH_dex': [-1.0],
    }
    template = Table(gannon_template)
    template_path = os.path.join(OUTPUT_DIR, "Gannon2024_TEMPLATE.fits")
    template.write(template_path, format='fits', overwrite=True)
    print(f"   Created template: {os.path.basename(template_path)}")
    print("   NOTE: Download actual data from https://arxiv.org/abs/2405.09104")
    
    return False


def download_matlas_dwarfs():
    """
    Download MATLAS dwarf catalog (Poulain+2021).
    Contains: 2210 dwarfs including UDGs, with distances and photometry.
    """
    print("\n" + "="*70)
    print(" MATLAS DWARF CATALOG (Poulain+2021)")
    print(" 2210 dwarfs including UDGs, with distances")
    print("="*70)
    
    result = download_with_timeout("J/MNRAS/506/5494", "MATLAS Dwarfs")
    
    if result:
        total = 0
        for i, t in enumerate(result):
            output_path = os.path.join(OUTPUT_DIR, f"MATLAS_dwarfs_t{i}.fits")
            t.write(output_path, format='fits', overwrite=True)
            print(f"   Saved: {os.path.basename(output_path)} ({len(t)} rows)")
            total += len(t)
        print(f"   Total: {total} objects")
        return True
    
    return False


def download_matlas_udg():
    """
    Download MATLAS UDG specific catalog (Marleau+2021).
    """
    print("\n" + "="*70)
    print(" MATLAS UDG CATALOG (Marleau+2021)")
    print("="*70)
    
    result = download_with_timeout("J/A+A/654/A105", "MATLAS UDG")
    
    if result:
        for i, t in enumerate(result):
            output_path = os.path.join(OUTPUT_DIR, f"MATLAS_UDG_t{i}.fits")
            t.write(output_path, format='fits', overwrite=True)
            print(f"   Saved: {os.path.basename(output_path)} ({len(t)} rows)")
        return True
    
    return False


def check_smudges():
    """Check if SMUDGes is already downloaded."""
    smudges_path = os.path.join(OUTPUT_DIR, "SMUDGes_t0.fits")
    if os.path.exists(smudges_path):
        t = Table.read(smudges_path)
        print(f"\nSMUDGes already downloaded: {len(t)} UDGs")
        return True
    return False


def main():
    print("="*70)
    print(" UDG CATALOG DOWNLOADER - COMPLETE")
    print(" Author: Jonathan Édouard Slama")
    print(" Metafund Research Division, Strasbourg")
    print("="*70)
    
    results = {}
    
    # Check existing
    results['SMUDGes'] = check_smudges()
    
    # Download Gannon spectroscopic catalog (PRIORITY - has velocities!)
    results['Gannon2024'] = download_gannon_catalog()
    
    # Download MATLAS
    results['MATLAS_dwarfs'] = download_matlas_dwarfs()
    results['MATLAS_UDG'] = download_matlas_udg()
    
    # Summary
    print("\n" + "="*70)
    print(" DOWNLOAD SUMMARY")
    print("="*70)
    
    for name, success in results.items():
        status = "Available" if success else "Missing/Template"
        print(f"   {name:<20}: {status}")
    
    print("\n Files in output directory:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath) / 1024
        print(f"   {f:<45} ({size:.1f} KB)")
    
    print("\n" + "="*70)
    print(" NEXT STEPS")
    print("="*70)
    print("1. If Gannon2024 not downloaded, get data from arXiv:2405.09104")
    print("2. Run the UDG killer prediction test:")
    print("   python tests/future/killer_prediction_1_udg.py")
    print("="*70)


if __name__ == "__main__":
    main()
