#!/usr/bin/env python3
"""
===============================================================================
GALAXY GROUPS - Q²R² TEST AT INTERMEDIATE SCALE
===============================================================================

CRITICAL TEST for the baryonic coupling hypothesis!

Scale hierarchy:
- Galaxies (10¹⁰-10¹² M☉): f_baryon ~15-20%, FULL Q²R² signature
- Groups (10¹²-10¹⁴ M☉): f_baryon ~10-15%, TRANSITION expected
- Clusters (10¹⁴-10¹⁵ M☉): f_baryon ~12%, PARTIAL signature (no inversion)

PREDICTIONS:
1. U-shape: Should be present (like clusters)
2. Sign inversion: WEAK or PARTIAL (transition regime)
3. Amplitude: Intermediate between galaxies and clusters

Author: Jonathan E. Slama
Date: December 2025
===============================================================================
"""

import os
import sys
import numpy as np
import json
import glob
from scipy import stats
from scipy.optimize import curve_fit
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "data", "level2_multicontext", "groups")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experimental", "04_RESULTS", "test_results")


def test_quadratic(residuals, env, label=""):
    """Test for quadratic relationship."""
    
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c
    
    try:
        env_norm = (env - env.min()) / (env.max() - env.min() + 1e-10)
        popt, pcov = curve_fit(quadratic, env_norm, residuals)
        a, b, c = popt
        a_err = np.sqrt(pcov[0, 0])
        sig = abs(a) / a_err if a_err > 0 else 0
        
        return {
            'a': float(a), 
            'a_err': float(a_err), 
            'significance': float(sig),
            'ushape': a > 0 and sig > 2,
            'n': len(residuals)
        }
    except:
        return None


def compute_density(ra, dec, z, n_neighbors=10):
    """Compute local density in 3D comoving space."""
    D_c = np.array([COSMO.comoving_distance(zi).value for zi in z])
    x = D_c * np.cos(np.radians(dec)) * np.cos(np.radians(ra))
    y = D_c * np.cos(np.radians(dec)) * np.sin(np.radians(ra))
    z_coord = D_c * np.sin(np.radians(dec))
    
    coords = np.column_stack([x, y, z_coord])
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=n_neighbors+1)
    
    r_n = distances[:, -1]
    density = n_neighbors / (4/3 * np.pi * r_n**3 + 1e-10)
    return np.log10(density + 1e-10)


def load_available_catalogs():
    """Load all available group catalogs."""
    
    catalogs = {}
    
    if not os.path.exists(DATA_DIR):
        print(f"  Data directory not found: {DATA_DIR}")
        return catalogs
    
    files = glob.glob(os.path.join(DATA_DIR, "*.fits"))
    
    for f in files:
        name = os.path.basename(f).replace('.fits', '')
        print(f"  Loading: {name}")
        try:
            data = Table.read(f)
            catalogs[name] = data
            print(f"    {len(data)} rows, columns: {data.colnames[:5]}...")
        except Exception as e:
            print(f"    Error: {e}")
    
    return catalogs


def analyze_tempel_groups(data):
    """
    Analyze Tempel et al. SDSS groups.
    
    Expected columns may include:
    - Gr: Group ID
    - RAJ2000, DEJ2000: Position
    - z: Redshift
    - Ngal: Number of members
    - Mass or Lum: Group mass/luminosity
    """
    
    print("\n" + "="*70)
    print(" TEMPEL GROUPS ANALYSIS")
    print("="*70)
    
    print(f"\n  Columns: {data.colnames}")
    
    # Find relevant columns
    ra_col = None
    dec_col = None
    z_col = None
    ngal_col = None
    mass_col = None
    lum_col = None
    
    for c in data.colnames:
        cl = c.lower()
        if 'raj2000' in cl or cl == 'ra':
            ra_col = c
        elif 'dej2000' in cl or cl == 'dec':
            dec_col = c
        elif cl == 'z' or 'redshift' in cl:
            z_col = c
        elif 'ngal' in cl or cl == 'n' or 'nfof' in cl:
            ngal_col = c
        elif 'mass' in cl or cl == 'm' or 'mh' in cl:
            mass_col = c
        elif 'lum' in cl or cl == 'l':
            lum_col = c
    
    print(f"\n  Identified columns:")
    print(f"    RA: {ra_col}, Dec: {dec_col}, z: {z_col}")
    print(f"    Ngal: {ngal_col}, Mass: {mass_col}, Lum: {lum_col}")
    
    if not all([ra_col, dec_col, z_col]):
        print("  ERROR: Missing essential columns")
        return None
    
    # Extract data
    ra = np.array(data[ra_col])
    dec = np.array(data[dec_col])
    z = np.array(data[z_col])
    
    ngal = np.array(data[ngal_col]) if ngal_col else None
    mass = np.array(data[mass_col]) if mass_col else None
    lum = np.array(data[lum_col]) if lum_col else None
    
    # Quality cuts
    mask = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z)
    mask &= (z > 0.01) & (z < 0.2)
    
    if ngal is not None:
        mask &= np.isfinite(ngal) & (ngal >= 2)  # At least 2 members
    
    ra, dec, z = ra[mask], dec[mask], z[mask]
    if ngal is not None:
        ngal = ngal[mask]
    if mass is not None:
        mass = mass[mask]
    if lum is not None:
        lum = lum[mask]
    
    print(f"\n  After cuts: {len(z)} groups")
    print(f"  z range: {z.min():.3f} - {z.max():.3f}")
    if ngal is not None:
        print(f"  Ngal range: {ngal.min():.0f} - {ngal.max():.0f}")
    
    # Compute environment
    print("\n  Computing environment...")
    density = compute_density(ra, dec, z)
    
    # Use Ngal as proxy for group "richness" (like cluster richness)
    if ngal is not None:
        log_ngal = np.log10(ngal)
        
        # Mean richness
        ngal_mean = log_ngal.mean()
        residuals = log_ngal - ngal_mean
        
        print("\n  TEST: U-shape in Ngal residuals vs density")
        result = test_quadratic(residuals, density)
        
        if result:
            shape = "U-shape ✓" if result['ushape'] else ("∩-shape" if result['a'] < 0 and result['significance'] > 2 else "~linear")
            print(f"    a = {result['a']:.4f} ± {result['a_err']:.4f} ({result['significance']:.1f}σ)")
            print(f"    Shape: {shape}")
        
        # Sign inversion test
        print("\n  SIGN INVERSION TEST (split by Ngal)")
        median_ngal = np.median(ngal)
        rich = ngal > median_ngal
        poor = ngal <= median_ngal
        
        result_rich = test_quadratic(residuals[rich], density[rich])
        result_poor = test_quadratic(residuals[poor], density[poor])
        
        if result_rich and result_poor:
            print(f"    Rich (Ngal > {median_ngal:.0f}): a = {result_rich['a']:.4f} ({result_rich['significance']:.1f}σ)")
            print(f"    Poor (Ngal ≤ {median_ngal:.0f}): a = {result_poor['a']:.4f} ({result_poor['significance']:.1f}σ)")
            
            sign_inv = (result_rich['a'] > 0) != (result_poor['a'] > 0)
            print(f"    Sign inversion: {'YES ✓' if sign_inv else 'NO'}")
            
            # Assess transition
            if sign_inv:
                print("\n    → TRANSITION DETECTED! Groups show sign inversion like galaxies!")
            elif result_rich['a'] > 0 and result_poor['a'] > 0:
                # Check if difference is significant
                diff = abs(result_rich['a'] - result_poor['a'])
                print(f"\n    → Both positive, but difference = {diff:.4f}")
                print("    → Possibly WEAK transition (both U-shape but different amplitudes)")
        
        return {
            'n_groups': len(z),
            'ushape_result': result,
            'sign_inversion': {
                'rich': result_rich,
                'poor': result_poor
            }
        }
    
    return None


def analyze_berlind_groups(data):
    """Analyze Berlind et al. groups."""
    
    print("\n" + "="*70)
    print(" BERLIND GROUPS ANALYSIS")
    print("="*70)
    
    print(f"\n  Columns: {data.colnames}")
    
    # Similar analysis...
    return analyze_generic_groups(data, "Berlind")


def analyze_generic_groups(data, name):
    """Generic group analysis."""
    
    print(f"\n  Analyzing {name} groups...")
    print(f"  Columns: {data.colnames[:10]}...")
    
    # Try to identify columns
    ra_col, dec_col, z_col, ngal_col = None, None, None, None
    
    for c in data.colnames:
        cl = c.lower()
        if 'ra' in cl and 'de' not in cl:
            ra_col = c
        elif 'de' in cl or 'dec' in cl:
            dec_col = c
        elif cl == 'z' or 'redshift' in cl or cl == 'zfof':
            z_col = c
        elif 'ngal' in cl or cl == 'n' or 'nfof' in cl or 'mult' in cl:
            ngal_col = c
    
    print(f"  Found: RA={ra_col}, Dec={dec_col}, z={z_col}, Ngal={ngal_col}")
    
    if not all([ra_col, dec_col]):
        print("  Missing position columns, skipping...")
        return None
    
    # Continue with analysis if we have enough columns
    ra = np.array(data[ra_col])
    dec = np.array(data[dec_col])
    
    if z_col:
        z = np.array(data[z_col])
        mask = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z)
        mask &= (z > 0.01) & (z < 0.5)
        
        ra, dec, z = ra[mask], dec[mask], z[mask]
        print(f"  After cuts: {len(z)} groups")
        
        if len(z) > 100:
            density = compute_density(ra, dec, z)
            
            if ngal_col:
                ngal = np.array(data[ngal_col])[mask]
                ngal = ngal[np.isfinite(ngal) & (ngal >= 2)]
                
                if len(ngal) > 100:
                    log_ngal = np.log10(ngal[:len(density)])
                    residuals = log_ngal - log_ngal.mean()
                    
                    result = test_quadratic(residuals, density[:len(log_ngal)])
                    if result:
                        shape = "U-shape" if result['ushape'] else "other"
                        print(f"  Result: a = {result['a']:.4f} ({result['significance']:.1f}σ) - {shape}")
                        return result
    
    return None


def main():
    print("="*70)
    print(" GALAXY GROUPS - Q²R² TEST AT INTERMEDIATE SCALE")
    print("="*70)
    
    print("""
  CRITICAL TEST for baryonic coupling hypothesis!
  
  PREDICTIONS:
  - Galaxies: Full signature (U-shape + inversion)
  - Groups: TRANSITION (U-shape + WEAK/PARTIAL inversion?)
  - Clusters: Partial (U-shape only)
    """)
    
    # Load catalogs
    print("\n  Loading available catalogs...")
    catalogs = load_available_catalogs()
    
    if not catalogs:
        print("\n  No catalogs found!")
        print("  Run download_galaxy_groups.py first.")
        return
    
    results = {}
    
    # Analyze each catalog
    for name, data in catalogs.items():
        print(f"\n{'='*70}")
        print(f" Analyzing: {name}")
        print(f"{'='*70}")
        
        if 'Tempel' in name:
            result = analyze_tempel_groups(data)
        elif 'Berlind' in name:
            result = analyze_generic_groups(data, "Berlind")
        else:
            result = analyze_generic_groups(data, name)
        
        if result:
            results[name] = result
    
    # Summary
    print("\n" + "="*70)
    print(" SUMMARY: GROUPS vs GALAXIES vs CLUSTERS")
    print("="*70)
    
    print("""
  COMPARISON TABLE:
  
  | Scale     | N        | U-shape | Sign Inversion | Status      |
  |-----------|----------|---------|----------------|-------------|
  | Galaxies  | 680k+    | 5σ+     | YES            | CONFIRMED   |
  | Groups    | Testing  | ?       | ?              | ANALYZING   |
  | Clusters  | 25k+     | 3-33σ   | NO             | PARTIAL     |
  
  If groups show WEAK inversion → Baryonic coupling CONFIRMED
  If groups show NO inversion → Scale transition at ~10¹² M☉
    """)
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "galaxy_groups_results.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"  Results saved: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
