#!/usr/bin/env python3
"""
===============================================================================
DOWNLOAD TEMPEL+2017 GROUP CATALOG
===============================================================================

Downloads the galaxy group catalog from Tempel et al. (2017)
VizieR catalog: J/A+A/602/A100

Author: Jonathan Édouard Slama
Date: December 2025
===============================================================================
"""

import os
import numpy as np
import pandas as pd
import urllib.request
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def download_tempel_vizier():
    """
    Download Tempel+2017 from VizieR using astroquery.
    """
    print("\n" + "="*70)
    print(" DOWNLOADING TEMPEL+2017 FROM VIZIER")
    print("="*70)
    
    try:
        from astroquery.vizier import Vizier
        
        # Configure Vizier - no row limit
        Vizier.ROW_LIMIT = -1
        
        print("  Querying VizieR for J/A+A/602/A100...")
        print("  This may take a few minutes...")
        
        # Get all tables from the catalog
        catalog_list = Vizier.get_catalogs('J/A+A/602/A100')
        
        print(f"  Found {len(catalog_list)} tables:")
        
        for i, cat in enumerate(catalog_list):
            print(f"    [{i}] {len(cat)} rows - columns: {cat.colnames[:5]}...")
        
        return catalog_list
        
    except ImportError:
        print("  astroquery not installed. Install with:")
        print("  pip install astroquery --break-system-packages")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def download_tempel_direct():
    """
    Download Tempel+2017 directly from CDS FTP.
    """
    print("\n" + "="*70)
    print(" DOWNLOADING TEMPEL+2017 FROM CDS (direct)")
    print("="*70)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    base_url = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/602/A100/"
    
    files_to_download = [
        ("ReadMe", "tempel_readme.txt"),
        ("table1.dat", "tempel_groups.dat"),      # Group catalog
        ("table2.dat", "tempel_galaxies.dat"),    # Galaxy catalog
    ]
    
    downloaded = []
    
    for remote_name, local_name in files_to_download:
        url = base_url + remote_name
        local_path = os.path.join(DATA_DIR, local_name)
        
        print(f"  Downloading {remote_name}...")
        
        try:
            urllib.request.urlretrieve(url, local_path)
            size = os.path.getsize(local_path)
            print(f"    ✓ Saved: {local_path} ({size:,} bytes)")
            downloaded.append(local_path)
        except Exception as e:
            print(f"    ✗ Failed: {e}")
    
    return downloaded


def parse_tempel_groups(filepath):
    """
    Parse Tempel+2017 group catalog (table1.dat).
    
    Format from ReadMe:
    Bytes   Format  Units   Label    Explanations
    1-  6   I6      ---     GroupID  Group identification number
    8- 17   F10.6   deg     RAdeg    Right ascension (J2000)
    19- 28   F10.6   deg     DEdeg    Declination (J2000)
    30- 37   F8.5    ---     z        Redshift of the group
    39- 42   I4      ---     Ngal     Number of galaxies in the group
    44- 51   F8.3    km/s    sigv     Velocity dispersion
    53- 59   F7.3    Mpc     Rmax     Maximum extent of the group
    61- 66   F6.3    Mpc     Rvir     Virial radius
    68- 74   F7.3    10+12Msun Mvir   Virial mass
    76- 82   F7.4    10+12Msun Mlum   Luminosity-based mass
    """
    print(f"\n  Parsing {filepath}...")
    
    # Fixed-width format
    colspecs = [
        (0, 6),    # GroupID
        (7, 17),   # RAdeg
        (18, 28),  # DEdeg
        (29, 37),  # z
        (38, 42),  # Ngal
        (43, 51),  # sigv
        (52, 59),  # Rmax
        (60, 66),  # Rvir
        (67, 74),  # Mvir
        (75, 82),  # Mlum
    ]
    
    names = ['GroupID', 'RA', 'Dec', 'z', 'Ngal', 'sigma_v', 'Rmax', 'Rvir', 'Mvir', 'Mlum']
    
    try:
        df = pd.read_fwf(filepath, colspecs=colspecs, names=names, comment='#')
        
        print(f"    Loaded {len(df)} groups")
        print(f"    Columns: {df.columns.tolist()}")
        print(f"    Ngal range: {df['Ngal'].min()} - {df['Ngal'].max()}")
        print(f"    z range: {df['z'].min():.4f} - {df['z'].max():.4f}")
        print(f"    σ_v range: {df['sigma_v'].min():.1f} - {df['sigma_v'].max():.1f} km/s")
        
        return df
        
    except Exception as e:
        print(f"    ERROR parsing: {e}")
        return None


def parse_tempel_galaxies(filepath):
    """
    Parse Tempel+2017 galaxy catalog (table2.dat).
    
    Format from ReadMe:
    Bytes   Format  Units   Label     Explanations
    1-  6   I6      ---     GroupID   Group identification number
    8- 25   I18     ---     SDSS      SDSS DR10 object ID (ObjID)
    27- 36   F10.6   deg     RAdeg     Right ascension (J2000)
    38- 47   F10.6   deg     DEdeg     Declination (J2000)
    49- 56   F8.5    ---     z         Redshift
    58- 64   F7.3    mag     Mag_r     Absolute r-band magnitude
    66- 71   F6.3    ---     Dist      Distance from group center (normalized)
    73- 78   F6.3    10+10Msun Mstar   Stellar mass
    """
    print(f"\n  Parsing {filepath}...")
    
    colspecs = [
        (0, 6),    # GroupID
        (7, 25),   # SDSS ObjID
        (26, 36),  # RAdeg
        (37, 47),  # DEdeg
        (48, 56),  # z
        (57, 64),  # Mag_r
        (65, 71),  # Dist
        (72, 78),  # Mstar
    ]
    
    names = ['GroupID', 'SDSS_ObjID', 'RA', 'Dec', 'z', 'Mag_r', 'Dist_norm', 'Mstar_1e10']
    
    try:
        df = pd.read_fwf(filepath, colspecs=colspecs, names=names, comment='#')
        
        print(f"    Loaded {len(df)} galaxies")
        print(f"    Unique groups: {df['GroupID'].nunique()}")
        print(f"    M_star range: {df['Mstar_1e10'].min():.3f} - {df['Mstar_1e10'].max():.3f} × 10^10 M_sun")
        
        return df
        
    except Exception as e:
        print(f"    ERROR parsing: {e}")
        return None


def compute_group_properties(df_groups, df_galaxies):
    """
    Compute additional group properties from member galaxies.
    """
    print("\n  Computing group properties...")
    
    # Sum stellar masses per group
    group_mstar = df_galaxies.groupby('GroupID')['Mstar_1e10'].sum().reset_index()
    group_mstar.columns = ['GroupID', 'Mstar_total_1e10']
    
    # Merge with groups
    df = df_groups.merge(group_mstar, on='GroupID', how='left')
    
    # Convert to log solar masses
    df['log_Mstar'] = np.log10(df['Mstar_total_1e10'] * 1e10)
    
    # Virial mass already in 10^12 Msun
    df['log_Mvir'] = np.log10(df['Mvir'] * 1e12)
    
    print(f"    log(M_star) range: {df['log_Mstar'].min():.2f} - {df['log_Mstar'].max():.2f}")
    print(f"    log(M_vir) range: {df['log_Mvir'].min():.2f} - {df['log_Mvir'].max():.2f}")
    
    return df


def main():
    print("="*70)
    print(" TEMPEL+2017 DATA DOWNLOAD")
    print(" Galaxy Group Catalog for TOE Testing")
    print("="*70)
    
    # Try VizieR first
    catalogs = download_tempel_vizier()
    
    if catalogs is None:
        # Fall back to direct download
        downloaded = download_tempel_direct()
        
        if len(downloaded) >= 2:
            # Parse the files
            groups_file = os.path.join(DATA_DIR, "tempel_groups.dat")
            galaxies_file = os.path.join(DATA_DIR, "tempel_galaxies.dat")
            
            df_groups = parse_tempel_groups(groups_file)
            df_galaxies = parse_tempel_galaxies(galaxies_file)
            
            if df_groups is not None and df_galaxies is not None:
                # Compute properties
                df_combined = compute_group_properties(df_groups, df_galaxies)
                
                # Save as CSV for easier use
                output_path = os.path.join(DATA_DIR, "tempel2017_groups.csv")
                df_combined.to_csv(output_path, index=False)
                print(f"\n  ✓ Saved processed data: {output_path}")
                
                return df_combined
    else:
        # Process VizieR catalogs
        print("\n  Processing VizieR catalogs...")
        
        # Save tables
        os.makedirs(DATA_DIR, exist_ok=True)
        
        for i, cat in enumerate(catalogs):
            df = cat.to_pandas()
            output_path = os.path.join(DATA_DIR, f"tempel2017_table{i+1}.csv")
            df.to_csv(output_path, index=False)
            print(f"    Saved table {i+1}: {output_path} ({len(df)} rows)")
        
        return catalogs
    
    return None


if __name__ == "__main__":
    result = main()
    
    if result is not None:
        print("\n" + "="*70)
        print(" DOWNLOAD COMPLETE")
        print("="*70)
        print("  Next step: Run grp001_toe_test.py with real data")
    else:
        print("\n" + "="*70)
        print(" DOWNLOAD FAILED")
        print("="*70)
        print("  Please install astroquery:")
        print("  pip install astroquery --break-system-packages")
