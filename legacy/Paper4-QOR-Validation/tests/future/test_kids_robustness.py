#!/usr/bin/env python3
"""
===============================================================================
ROBUSTNESS TESTS FOR KiDS RESULTS
===============================================================================

Tests to ensure the Q2R2 detection in KiDS is not an artifact.

Author: Jonathan E. Slama
Date: December 2025
===============================================================================
"""

import os
import sys
import numpy as np
import json
from scipy import stats
from scipy.optimize import curve_fit
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "data", "level2_multicontext", "kids")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experimental", "04_RESULTS", "test_results")


def load_kids_data():
    """Load and prepare KiDS data."""
    
    print("  Loading KiDS data...")
    
    bright = Table.read(os.path.join(DATA_DIR, "KiDS_DR4_brightsample.fits"))
    lephare = Table.read(os.path.join(DATA_DIR, "KiDS_DR4_brightsample_LePhare.fits"))
    
    print(f"  Bright sample columns: {bright.colnames[:10]}...")
    print(f"  LePhare columns: {lephare.colnames[:10]}...")
    
    # Find RA/Dec columns - could be RAJ2000/DEJ2000 or RA/DEC or RAJ2000/DECJ2000
    ra_col = None
    dec_col = None
    for col in bright.colnames:
        if col.upper() in ['RA', 'RAJ2000', 'RA_J2000']:
            ra_col = col
        if col.upper() in ['DEC', 'DEJ2000', 'DECJ2000', 'DE', 'DEC_J2000']:
            dec_col = col
    
    if ra_col is None or dec_col is None:
        print(f"  ERROR: Could not find RA/Dec columns")
        print(f"  Available: {bright.colnames}")
        return None
    
    print(f"  Using RA column: {ra_col}")
    print(f"  Using Dec column: {dec_col}")
    
    ra = np.array(bright[ra_col])
    dec = np.array(bright[dec_col])
    
    # Find redshift column
    z_col = None
    for col in bright.colnames:
        if 'zphot' in col.lower() or col.lower() == 'z':
            z_col = col
            break
    
    if z_col is None:
        print(f"  Warning: No photo-z found, checking LePhare...")
        for col in lephare.colnames:
            if 'z' in col.lower():
                z_col = col
                z = np.array(lephare[z_col])
                break
    else:
        z = np.array(bright[z_col])
    
    print(f"  Using z column: {z_col}")
    
    # Get magnitudes and mass from LePhare
    mag_g = np.array(lephare['MAG_ABS_g'])
    mag_r = np.array(lephare['MAG_ABS_r'])
    mass = np.array(lephare['MASS_MED'])
    
    # Quality cuts
    mask = (
        np.isfinite(ra) & np.isfinite(dec) &
        np.isfinite(z) & (z > 0.01) & (z < 0.5) &
        np.isfinite(mag_g) & np.isfinite(mag_r) &
        np.isfinite(mass) & (mass > 7) & (mass < 13)
    )
    
    data = {
        'ra': ra[mask],
        'dec': dec[mask],
        'z': z[mask],
        'mag_g': mag_g[mask],
        'mag_r': mag_r[mask],
        'mass': mass[mask],
        'color': mag_g[mask] - mag_r[mask]
    }
    
    print(f"  Loaded {len(data['z'])} galaxies after quality cuts")
    
    return data


def compute_density(ra, dec, z, n_neighbors=10):
    """Compute local density."""
    print("  Computing 3D positions...")
    D_c = np.array([COSMO.comoving_distance(zi).value for zi in z])
    x = D_c * np.cos(np.radians(dec)) * np.cos(np.radians(ra))
    y = D_c * np.cos(np.radians(dec)) * np.sin(np.radians(ra))
    z_coord = D_c * np.sin(np.radians(dec))
    
    print("  Building KD-tree...")
    coords = np.column_stack([x, y, z_coord])
    tree = cKDTree(coords)
    
    print("  Querying neighbors...")
    distances, _ = tree.query(coords, k=n_neighbors+1)
    
    r_n = distances[:, -1]
    density = n_neighbors / (4/3 * np.pi * r_n**3 + 1e-10)
    return np.log10(density + 1e-10)


def test_quadratic(residuals, env):
    """Test for quadratic relationship."""
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c
    
    try:
        env_norm = (env - env.min()) / (env.max() - env.min() + 1e-10)
        popt, pcov = curve_fit(quadratic, env_norm, residuals)
        a, b, c = popt
        a_err = np.sqrt(pcov[0, 0])
        return {'a': float(a), 'a_err': float(a_err), 'sig': abs(a)/a_err if a_err > 0 else 0}
    except:
        return None


def test_redshift_bins(data, density):
    """TEST 1: Redshift bins"""
    
    print("\n" + "="*70)
    print(" TEST 1: REDSHIFT BINS (Cosmic Evolution)")
    print("="*70)
    
    slope, intercept, _, _, _ = stats.linregress(data['mass'], data['mag_r'])
    residuals = data['mag_r'] - (slope * data['mass'] + intercept)
    
    z_bins = [(0.01, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
    
    print(f"\n  {'z range':<15} {'N':>8} {'a':>12} {'+/-err':>10} {'sigma':>8} {'U-shape':>10}")
    print("-"*70)
    
    results = []
    
    for z_min, z_max in z_bins:
        mask = (data['z'] >= z_min) & (data['z'] < z_max)
        n = mask.sum()
        
        if n > 1000:
            result = test_quadratic(residuals[mask], density[mask])
            if result:
                shape = "YES" if result['a'] > 0 and result['sig'] > 2 else "no"
                print(f"  [{z_min:.2f}, {z_max:.2f}] {n:>8,} {result['a']:>12.5f} {result['a_err']:>10.5f} {result['sig']:>8.1f} {shape:>10}")
                results.append({
                    'z_min': z_min, 'z_max': z_max, 'n': n,
                    'a': result['a'], 'a_err': result['a_err'], 'sig': result['sig']
                })
    
    positive = sum(1 for r in results if r['a'] > 0)
    significant = sum(1 for r in results if r['a'] > 0 and r['sig'] > 2)
    
    print(f"\n  Summary: {positive}/{len(results)} bins positive, {significant}/{len(results)} significant (>2 sigma)")
    
    return results


def test_bootstrap(data, density, n_bootstrap=100):
    """TEST 2: Bootstrap"""
    
    print("\n" + "="*70)
    print(f" TEST 2: BOOTSTRAP ({n_bootstrap} iterations)")
    print("="*70)
    
    slope, intercept, _, _, _ = stats.linregress(data['mass'], data['mag_r'])
    residuals = data['mag_r'] - (slope * data['mass'] + intercept)
    
    n = len(residuals)
    a_values = []
    
    print(f"\n  Running {n_bootstrap} bootstrap iterations...")
    
    for i in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        result = test_quadratic(residuals[idx], density[idx])
        if result:
            a_values.append(result['a'])
        
        if (i + 1) % 20 == 0:
            print(f"    Iteration {i+1}/{n_bootstrap}")
    
    a_values = np.array(a_values)
    
    a_mean = a_values.mean()
    a_std = a_values.std()
    a_positive = (a_values > 0).sum() / len(a_values) * 100
    
    print(f"\n  Results:")
    print(f"    Mean a: {a_mean:.5f} +/- {a_std:.5f}")
    print(f"    Positive: {a_positive:.1f}%")
    print(f"    95% CI: [{np.percentile(a_values, 2.5):.5f}, {np.percentile(a_values, 97.5):.5f}]")
    
    robust = a_positive > 95
    print(f"\n  ROBUST: {'YES' if robust else 'NO'} (>95% positive required)")
    
    return {
        'mean': float(a_mean),
        'std': float(a_std),
        'positive_fraction': float(a_positive),
        'ci_95': [float(np.percentile(a_values, 2.5)), float(np.percentile(a_values, 97.5))],
        'robust': robust
    }


def test_jackknife_spatial(data, density, n_regions=10):
    """TEST 3: Jackknife spatial"""
    
    print("\n" + "="*70)
    print(f" TEST 3: JACKKNIFE SPATIAL ({n_regions} regions)")
    print("="*70)
    
    slope, intercept, _, _, _ = stats.linregress(data['mass'], data['mag_r'])
    residuals = data['mag_r'] - (slope * data['mass'] + intercept)
    
    ra_bins = np.percentile(data['ra'], np.linspace(0, 100, n_regions + 1))
    
    print(f"\n  {'Region':<10} {'RA range':<25} {'N removed':>10} {'a':>12} {'sigma':>8}")
    print("-"*70)
    
    a_values = []
    
    for i in range(n_regions):
        mask_remove = (data['ra'] >= ra_bins[i]) & (data['ra'] < ra_bins[i+1])
        mask_keep = ~mask_remove
        
        n_removed = mask_remove.sum()
        
        result = test_quadratic(residuals[mask_keep], density[mask_keep])
        if result:
            a_values.append(result['a'])
            print(f"  {i+1:<10} [{ra_bins[i]:.1f}, {ra_bins[i+1]:.1f}] {n_removed:>10,} {result['a']:>12.5f} {result['sig']:>8.1f}")
    
    a_values = np.array(a_values)
    
    sign_flips = np.sum(np.diff(np.sign(a_values)) != 0)
    all_positive = (a_values > 0).all()
    
    print(f"\n  Summary:")
    print(f"    Range of a: [{a_values.min():.5f}, {a_values.max():.5f}]")
    print(f"    Sign flips: {sign_flips}")
    print(f"    All positive: {'YES' if all_positive else 'NO'}")
    
    homogeneous = sign_flips == 0
    print(f"\n  SPATIALLY HOMOGENEOUS: {'YES' if homogeneous else 'NO'}")
    
    return {
        'a_values': a_values.tolist(),
        'sign_flips': int(sign_flips),
        'all_positive': bool(all_positive),
        'homogeneous': homogeneous
    }


def test_mass_bins(data, density):
    """TEST 4: Mass bins"""
    
    print("\n" + "="*70)
    print(" TEST 4: MASS BINS (Scale Dependence)")
    print("="*70)
    
    slope, intercept, _, _, _ = stats.linregress(data['mass'], data['mag_r'])
    residuals = data['mag_r'] - (slope * data['mass'] + intercept)
    
    mass_bins = [(7, 8.5), (8.5, 9.5), (9.5, 10.5), (10.5, 11.5), (11.5, 13)]
    
    print(f"\n  {'log M* range':<15} {'N':>8} {'a':>12} {'+/-err':>10} {'sigma':>8} {'U-shape':>10}")
    print("-"*70)
    
    results = []
    
    for m_min, m_max in mass_bins:
        mask = (data['mass'] >= m_min) & (data['mass'] < m_max)
        n = mask.sum()
        
        if n > 1000:
            result = test_quadratic(residuals[mask], density[mask])
            if result:
                shape = "YES" if result['a'] > 0 and result['sig'] > 2 else "no"
                print(f"  [{m_min:.1f}, {m_max:.1f}] {n:>8,} {result['a']:>12.5f} {result['a_err']:>10.5f} {result['sig']:>8.1f} {shape:>10}")
                results.append({
                    'm_min': m_min, 'm_max': m_max, 'n': n,
                    'a': result['a'], 'a_err': result['a_err'], 'sig': result['sig']
                })
    
    if len(results) >= 3:
        masses = [(r['m_min'] + r['m_max'])/2 for r in results]
        amplitudes = [r['a'] for r in results]
        r_scaling, p_scaling = stats.pearsonr(masses, amplitudes)
        print(f"\n  Mass scaling: r = {r_scaling:.3f}, p = {p_scaling:.3f}")
    
    return results


def test_sign_inversion_robustness(data, density):
    """TEST 5: Sign inversion robustness"""
    
    print("\n" + "="*70)
    print(" TEST 5: SIGN INVERSION ROBUSTNESS")
    print("="*70)
    
    slope, intercept, _, _, _ = stats.linregress(data['mass'], data['mag_r'])
    residuals = data['mag_r'] - (slope * data['mass'] + intercept)
    
    color_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
    
    print(f"\n  {'Color cut':<12} {'N_blue':>8} {'a_blue':>10} {'N_red':>8} {'a_red':>10} {'Inversion':>12}")
    print("-"*70)
    
    results = []
    inversions = 0
    
    for threshold in color_thresholds:
        blue = data['color'] < threshold
        red = data['color'] >= threshold
        
        result_blue = test_quadratic(residuals[blue], density[blue])
        result_red = test_quadratic(residuals[red], density[red])
        
        if result_blue and result_red:
            inversion = (result_blue['a'] > 0) != (result_red['a'] > 0)
            if inversion:
                inversions += 1
            
            inv_str = "YES" if inversion else "no"
            print(f"  g-r < {threshold:.1f} {blue.sum():>8,} {result_blue['a']:>10.5f} {red.sum():>8,} {result_red['a']:>10.5f} {inv_str:>12}")
            
            results.append({
                'threshold': threshold,
                'a_blue': result_blue['a'],
                'a_red': result_red['a'],
                'inversion': inversion
            })
    
    robust = inversions >= len(color_thresholds) - 1
    print(f"\n  Inversions detected: {inversions}/{len(color_thresholds)}")
    print(f"  SIGN INVERSION ROBUST: {'YES' if robust else 'NO'}")
    
    return {
        'results': results,
        'inversions': inversions,
        'robust': robust
    }


def main():
    print("="*70)
    print(" KiDS ROBUSTNESS TESTS")
    print(" Validating Q2R2 detection stability")
    print("="*70)
    
    data = load_kids_data()
    
    if data is None:
        print("  ERROR: Could not load data")
        return
    
    print("\n  Computing environment (this may take a few minutes for 600k+ galaxies)...")
    density = compute_density(data['ra'], data['dec'], data['z'])
    print("  Done.")
    
    results = {}
    
    results['redshift_bins'] = test_redshift_bins(data, density)
    results['bootstrap'] = test_bootstrap(data, density, n_bootstrap=100)
    results['jackknife_spatial'] = test_jackknife_spatial(data, density, n_regions=10)
    results['mass_bins'] = test_mass_bins(data, density)
    results['sign_inversion'] = test_sign_inversion_robustness(data, density)
    
    # Summary
    print("\n" + "="*70)
    print(" ROBUSTNESS SUMMARY")
    print("="*70)
    
    z_pass = sum(1 for r in results['redshift_bins'] if r['a'] > 0) >= len(results['redshift_bins']) - 1
    m_pass = sum(1 for r in results['mass_bins'] if r['a'] > 0) >= len(results['mass_bins']) - 1
    
    print(f"""
  TEST                          RESULT                      STATUS
  ----------------------------------------------------------------------
  Redshift bins                 Present in {sum(1 for r in results['redshift_bins'] if r['a'] > 0)}/{len(results['redshift_bins'])} bins            {'PASS' if z_pass else 'FAIL'}
  Bootstrap (100 iter)          {results['bootstrap']['positive_fraction']:.1f}% positive               {'PASS' if results['bootstrap']['robust'] else 'FAIL'}
  Jackknife spatial             {results['jackknife_spatial']['sign_flips']} sign flips                 {'PASS' if results['jackknife_spatial']['homogeneous'] else 'FAIL'}
  Mass bins                     Present in {sum(1 for r in results['mass_bins'] if r['a'] > 0)}/{len(results['mass_bins'])} bins            {'PASS' if m_pass else 'CHECK'}
  Sign inversion                {results['sign_inversion']['inversions']}/{len(results['sign_inversion']['results'])} thresholds           {'PASS' if results['sign_inversion']['robust'] else 'FAIL'}
    """)
    
    tests_passed = sum([
        z_pass,
        results['bootstrap']['robust'],
        results['jackknife_spatial']['homogeneous'],
        m_pass,
        results['sign_inversion']['robust']
    ])
    
    print(f"  OVERALL: {tests_passed}/5 tests passed")
    print(f"  KiDS RESULT IS {'ROBUST' if tests_passed >= 4 else 'NEEDS INVESTIGATION'}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "kids_robustness_results.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
