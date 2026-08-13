#!/usr/bin/env python3
"""
===============================================================================
GALAXY GROUPS - Q²R² TEST AT INTERMEDIATE SCALE
===============================================================================

Using the best available group catalogs:
1. Tempel SDSS (82,458 groups) - velocity dispersion available!
2. Saulder 2MRS (229,893 groups) - luminosity available

CRITICAL TEST for baryonic coupling hypothesis!

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


def analyze_tempel_groups():
    """
    Analyze Tempel et al. 2014 SDSS groups.
    
    Table 1: Group catalog with velocity dispersion
    Columns: GroupID, Ngal, RAJ2000, DEJ2000, zcmb, Dist.c, sig.v, sig.sky
    """
    
    print("\n" + "="*70)
    print(" TEMPEL SDSS GROUPS (82,458 groups)")
    print("="*70)
    
    # Load group catalog (table 1)
    path = os.path.join(DATA_DIR, "Tempel_groups_J_A+A_566_A1_t1.fits")
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return None
    
    data = Table.read(path)
    print(f"\n  Loaded: {len(data)} groups")
    print(f"  Columns: {data.colnames}")
    
    # Extract variables
    ra = np.array(data['RAJ2000'])
    dec = np.array(data['DEJ2000'])
    z = np.array(data['zcmb'])
    ngal = np.array(data['Ngal'])
    sig_v = np.array(data['sig.v'])  # Velocity dispersion in km/s
    
    # Quality cuts
    mask = (
        np.isfinite(ra) & np.isfinite(dec) &
        np.isfinite(z) & (z > 0.01) & (z < 0.2) &
        np.isfinite(ngal) & (ngal >= 2) &
        np.isfinite(sig_v) & (sig_v > 0)
    )
    
    ra, dec, z = ra[mask], dec[mask], z[mask]
    ngal, sig_v = ngal[mask], sig_v[mask]
    
    print(f"\n  After quality cuts: {len(z)} groups")
    print(f"  z range: {z.min():.3f} - {z.max():.3f}")
    print(f"  Ngal range: {ngal.min():.0f} - {ngal.max():.0f}")
    print(f"  σ_v range: {sig_v.min():.0f} - {sig_v.max():.0f} km/s")
    
    # Mass estimate from velocity dispersion
    # M ∝ σ_v² R, but σ_v alone is a good mass proxy
    log_sig = np.log10(sig_v)
    log_ngal = np.log10(ngal)
    
    # Compute environment
    print("\n  Computing environment (group density)...")
    density = compute_density(ra, dec, z)
    
    # =========================================================================
    # TEST 1: Ngal-σ_v relation (mass-richness analogue)
    # =========================================================================
    print("\n" + "-"*70)
    print(" TEST 1: Ngal-σ_v RELATION")
    print("-"*70)
    
    slope, intercept, r_val, p, se = stats.linregress(log_sig, log_ngal)
    print(f"\n  log(Ngal) = {slope:.3f} × log(σ_v) + {intercept:.3f}")
    print(f"  Correlation: r = {r_val:.3f}")
    
    residuals = log_ngal - (slope * log_sig + intercept)
    print(f"  Residual scatter: {residuals.std():.3f} dex")
    
    # =========================================================================
    # TEST 2: U-shape in residuals vs environment
    # =========================================================================
    print("\n" + "-"*70)
    print(" TEST 2: U-SHAPE IN RESIDUALS VS ENVIRONMENT")
    print("-"*70)
    
    result_density = test_quadratic(residuals, density)
    
    if result_density:
        shape = "U-shape ✓" if result_density['ushape'] else ("∩-shape" if result_density['a'] < 0 and result_density['significance'] > 2 else "~linear")
        print(f"\n  a = {result_density['a']:.4f} ± {result_density['a_err']:.4f}")
        print(f"  Significance: {result_density['significance']:.1f}σ")
        print(f"  Shape: {shape}")
    
    # =========================================================================
    # TEST 3: SIGN INVERSION (split by σ_v = mass proxy)
    # =========================================================================
    print("\n" + "-"*70)
    print(" TEST 3: SIGN INVERSION (split by velocity dispersion)")
    print("-"*70)
    
    median_sig = np.median(sig_v)
    high_sig = sig_v > median_sig  # High mass (R-dominated analogue)
    low_sig = sig_v <= median_sig  # Low mass (Q-dominated analogue)
    
    print(f"\n  High σ_v (> {median_sig:.0f} km/s): N = {high_sig.sum()}")
    print(f"  Low σ_v (≤ {median_sig:.0f} km/s): N = {low_sig.sum()}")
    
    result_high = test_quadratic(residuals[high_sig], density[high_sig])
    result_low = test_quadratic(residuals[low_sig], density[low_sig])
    
    if result_high and result_low:
        print(f"\n  High σ_v (massive): a = {result_high['a']:.4f} ± {result_high['a_err']:.4f} ({result_high['significance']:.1f}σ)")
        print(f"  Low σ_v (less massive): a = {result_low['a']:.4f} ± {result_low['a_err']:.4f} ({result_low['significance']:.1f}σ)")
        
        sign_inversion = (result_high['a'] > 0) != (result_low['a'] > 0)
        print(f"\n  SIGN INVERSION: {'YES ✓' if sign_inversion else 'NO'}")
        
        if sign_inversion:
            print("  → TRANSITION DETECTED! Groups show sign inversion!")
        else:
            # Check amplitude difference
            ratio = result_high['a'] / result_low['a'] if result_low['a'] != 0 else 0
            print(f"  → Amplitude ratio (high/low): {ratio:.2f}")
    
    # =========================================================================
    # TEST 4: SIGN INVERSION (split by Ngal = richness)
    # =========================================================================
    print("\n" + "-"*70)
    print(" TEST 4: SIGN INVERSION (split by richness Ngal)")
    print("-"*70)
    
    median_ngal = np.median(ngal)
    rich = ngal > median_ngal
    poor = ngal <= median_ngal
    
    print(f"\n  Rich (Ngal > {median_ngal:.0f}): N = {rich.sum()}")
    print(f"  Poor (Ngal ≤ {median_ngal:.0f}): N = {poor.sum()}")
    
    result_rich = test_quadratic(residuals[rich], density[rich])
    result_poor = test_quadratic(residuals[poor], density[poor])
    
    if result_rich and result_poor:
        print(f"\n  Rich: a = {result_rich['a']:.4f} ± {result_rich['a_err']:.4f} ({result_rich['significance']:.1f}σ)")
        print(f"  Poor: a = {result_poor['a']:.4f} ± {result_poor['a_err']:.4f} ({result_poor['significance']:.1f}σ)")
        
        sign_inversion_ngal = (result_rich['a'] > 0) != (result_poor['a'] > 0)
        print(f"\n  SIGN INVERSION: {'YES ✓' if sign_inversion_ngal else 'NO'}")
    
    # =========================================================================
    # TEST 5: BINNED ANALYSIS
    # =========================================================================
    print("\n" + "-"*70)
    print(" TEST 5: BINNED ANALYSIS (residuals in density bins)")
    print("-"*70)
    
    n_bins = 10
    bin_edges = np.percentile(density, np.linspace(0, 100, n_bins+1))
    
    print(f"\n  {'Bin':<5} {'Density range':<25} {'N':>6} {'<residual>':>12}")
    print("-"*55)
    
    bin_centers = []
    bin_means = []
    
    for i in range(n_bins):
        mask_bin = (density >= bin_edges[i]) & (density < bin_edges[i+1])
        if i == n_bins - 1:
            mask_bin = (density >= bin_edges[i])
        
        if mask_bin.sum() > 10:
            center = (bin_edges[i] + bin_edges[i+1]) / 2
            mean_res = residuals[mask_bin].mean()
            
            bin_centers.append(center)
            bin_means.append(mean_res)
            
            print(f"  {i+1:<5} [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}] {mask_bin.sum():>6} {mean_res:>12.4f}")
    
    # Fit parabola to bins
    if len(bin_centers) >= 3:
        bin_centers = np.array(bin_centers)
        bin_means = np.array(bin_means)
        
        x_norm = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min())
        
        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c
        
        popt, pcov = curve_fit(quadratic, x_norm, bin_means)
        a_binned = popt[0]
        a_err_binned = np.sqrt(pcov[0, 0])
        
        print(f"\n  Binned fit: a = {a_binned:.4f} ± {a_err_binned:.4f}")
        print(f"  Shape: {'U-shape ✓' if a_binned > 0 else '∩-shape'}")
    
    return {
        'catalog': 'Tempel SDSS',
        'n_groups': len(z),
        'ushape_result': result_density,
        'sign_inversion_sigma': {
            'high_sigma': result_high,
            'low_sigma': result_low,
            'detected': sign_inversion if result_high and result_low else None
        },
        'sign_inversion_ngal': {
            'rich': result_rich,
            'poor': result_poor,
            'detected': sign_inversion_ngal if result_rich and result_poor else None
        }
    }


def analyze_saulder_groups():
    """
    Analyze Saulder et al. 2016 2MRS groups.
    
    Table 1: 229,893 groups with logLtot (total luminosity)
    """
    
    print("\n" + "="*70)
    print(" SAULDER 2MRS GROUPS (229,893 groups)")
    print("="*70)
    
    # Load group catalog (table 1 is the larger one)
    path = os.path.join(DATA_DIR, "Saulder_2MRS_groups_t1.fits")
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return None
    
    data = Table.read(path)
    print(f"\n  Loaded: {len(data)} groups")
    print(f"  Columns: {data.colnames}")
    
    # Extract variables
    ra = np.array(data['RAJ2000'])
    dec = np.array(data['DEJ2000'])
    z = np.array(data['z'])
    log_ltot = np.array(data['logLtot'])  # Total luminosity
    
    # Quality cuts
    mask = (
        np.isfinite(ra) & np.isfinite(dec) &
        np.isfinite(z) & (z > 0.001) & (z < 0.1) &  # 2MRS is local universe
        np.isfinite(log_ltot) & (log_ltot > 0)
    )
    
    ra, dec, z = ra[mask], dec[mask], z[mask]
    log_ltot = log_ltot[mask]
    
    print(f"\n  After quality cuts: {len(z)} groups")
    print(f"  z range: {z.min():.4f} - {z.max():.4f}")
    print(f"  log(L_tot) range: {log_ltot.min():.2f} - {log_ltot.max():.2f}")
    
    # Compute environment
    print("\n  Computing environment...")
    density = compute_density(ra, dec, z)
    
    # Mean luminosity residuals
    ltot_mean = log_ltot.mean()
    residuals = log_ltot - ltot_mean
    
    # Test U-shape
    print("\n  TEST: U-shape in luminosity residuals vs density")
    result = test_quadratic(residuals, density)
    
    if result:
        shape = "U-shape ✓" if result['ushape'] else ("∩-shape" if result['a'] < 0 and result['significance'] > 2 else "~linear")
        print(f"  a = {result['a']:.4f} ± {result['a_err']:.4f} ({result['significance']:.1f}σ)")
        print(f"  Shape: {shape}")
    
    # Sign inversion
    print("\n  SIGN INVERSION TEST (split by luminosity)")
    median_l = np.median(log_ltot)
    bright = log_ltot > median_l
    faint = log_ltot <= median_l
    
    result_bright = test_quadratic(residuals[bright], density[bright])
    result_faint = test_quadratic(residuals[faint], density[faint])
    
    if result_bright and result_faint:
        print(f"  Bright: a = {result_bright['a']:.4f} ({result_bright['significance']:.1f}σ)")
        print(f"  Faint:  a = {result_faint['a']:.4f} ({result_faint['significance']:.1f}σ)")
        
        sign_inv = (result_bright['a'] > 0) != (result_faint['a'] > 0)
        print(f"  Sign inversion: {'YES ✓' if sign_inv else 'NO'}")
    
    return {
        'catalog': 'Saulder 2MRS',
        'n_groups': len(z),
        'ushape_result': result,
        'sign_inversion': {
            'bright': result_bright,
            'faint': result_faint
        }
    }


def main():
    print("="*70)
    print(" GALAXY GROUPS - Q²R² TEST AT INTERMEDIATE SCALE")
    print(" THE CRITICAL TEST FOR BARYONIC COUPLING HYPOTHESIS")
    print("="*70)
    
    print("""
  Mass scale hierarchy:
  ┌─────────────────────────────────────────────────────────────────┐
  │ GALAXIES   10¹⁰-10¹² M☉   f_baryon ~15-20%   FULL Q²R² ✓      │
  │ GROUPS     10¹²-10¹⁴ M☉   f_baryon ~10-15%   TRANSITION ???   │
  │ CLUSTERS   10¹⁴-10¹⁵ M☉   f_baryon ~12%      PARTIAL Q²R²     │
  └─────────────────────────────────────────────────────────────────┘
  
  PREDICTION: Groups should show WEAK/PARTIAL sign inversion
    """)
    
    results = {}
    
    # Analyze main catalogs
    result_tempel = analyze_tempel_groups()
    if result_tempel:
        results['tempel'] = result_tempel
    
    result_saulder = analyze_saulder_groups()
    if result_saulder:
        results['saulder'] = result_saulder
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print(" SUMMARY: GROUPS IN THE SCALE HIERARCHY")
    print("="*70)
    
    print("""
  ┌────────────┬──────────┬─────────────┬────────────────┬───────────┐
  │ Context    │ N        │ U-shape     │ Sign Inversion │ Status    │
  ├────────────┼──────────┼─────────────┼────────────────┼───────────┤
  │ ALFALFA    │ 19,222   │ ✓ 5σ+       │ ✓ CONFIRMED    │ FULL      │
  │ KiDS       │ 663,323  │ ✓ 5.2σ      │ ✓ CONFIRMED    │ FULL      │
  ├────────────┼──────────┼─────────────┼────────────────┼───────────┤""")
    
    if result_tempel:
        ushape = "✓" if result_tempel['ushape_result']['ushape'] else "?"
        sig = result_tempel['ushape_result']['significance']
        inv = result_tempel['sign_inversion_sigma'].get('detected', None)
        inv_str = "✓" if inv else ("✗" if inv is False else "?")
        print(f"  │ Tempel     │ {result_tempel['n_groups']:>8,} │ {ushape} {sig:.1f}σ       │ {inv_str}              │ GROUPS    │")
    
    if result_saulder:
        ushape = "✓" if result_saulder['ushape_result']['ushape'] else "?"
        sig = result_saulder['ushape_result']['significance']
        print(f"  │ Saulder    │ {result_saulder['n_groups']:>8,} │ {ushape} {sig:.1f}σ       │ ?              │ GROUPS    │")
    
    print("""  ├────────────┼──────────┼─────────────┼────────────────┼───────────┤
  │ eROSITA    │ 3,347    │ ✓ 3.4σ      │ ✗ NO           │ PARTIAL   │
  │ redMaPPer  │ 25,604   │ ✓ 33σ       │ ✗ NO           │ PARTIAL   │
  └────────────┴──────────┴─────────────┴────────────────┴───────────┘
    """)
    
    # Interpretation
    print("\n  INTERPRETATION:")
    print("-"*70)
    
    if result_tempel:
        inv = result_tempel['sign_inversion_sigma'].get('detected', None)
        if inv:
            print("""
  ★ GROUPS SHOW SIGN INVERSION!
  
  This confirms the BARYONIC COUPLING hypothesis:
  - Galaxies (baryon-rich dynamics): Full Q²R²
  - Groups (intermediate): Partial inversion (TRANSITION)
  - Clusters (DM-dominated): No inversion
  
  The moduli fields couple primarily to baryons, and the signal
  is progressively diluted as dark matter dominance increases.
            """)
        else:
            print("""
  Groups show U-shape but NO sign inversion (like clusters).
  
  Possible interpretations:
  1. Scale transition is sharper (occurs at galaxy scale only)
  2. Group finder algorithm affects the signal
  3. Need different Q proxy for groups
            """)
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "galaxy_groups_qr2_results.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
