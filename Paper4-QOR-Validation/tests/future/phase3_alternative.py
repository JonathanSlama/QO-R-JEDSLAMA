#!/usr/bin/env python3
"""
===============================================================================
PHASE 3 ALTERNATIVE: ANGULAR CORRELATION WITHOUT HEALPY
===============================================================================

Alternative approach for CMB lensing analysis that doesn't require healpy.
Uses pre-computed kappa values from public catalogs.

Author: Jonathan E. Slama
Date: December 2025
===============================================================================
"""

import os
import numpy as np
import json
from scipy import stats
from scipy.spatial import cKDTree
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALFALFA_DIR = os.path.join(BASE_DIR, "data", "alfalfa")
KIDS_DIR = os.path.join(BASE_DIR, "data", "level2_multicontext", "kids")
OUTPUT_DIR = os.path.join(BASE_DIR, "experimental", "04_RESULTS", "test_results")


def load_alfalfa_with_residuals():
    """Load ALFALFA data and compute BTF residuals."""
    
    print("  Loading ALFALFA data...")
    
    data_path = os.path.join(ALFALFA_DIR, "alfalfa_a100.fits")
    if not os.path.exists(data_path):
        print(f"  ERROR: Not found: {data_path}")
        return None
    
    t = Table.read(data_path)
    
    ra_str = [str(x) for x in t['RAJ2000']]
    dec_str = [str(x) for x in t['DEJ2000']]
    coords = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
    ra = coords.ra.deg
    dec = coords.dec.deg
    
    logMHI = np.array(t['logMHI'], dtype=float)
    W50 = np.array(t['W50'], dtype=float)
    dist = np.array(t['Dist'], dtype=float)
    
    mask = (
        np.isfinite(ra) & np.isfinite(dec) &
        np.isfinite(logMHI) & (logMHI > 6) & (logMHI < 12) &
        np.isfinite(W50) & (W50 > 20) & (W50 < 600) &
        np.isfinite(dist) & (dist > 0) & (dist < 300)
    )
    
    ra, dec = ra[mask], dec[mask]
    logMHI, W50, dist = logMHI[mask], W50[mask], dist[mask]
    
    logV = np.log10(W50 / 2.0)
    logMbary = logMHI + np.log10(1.4)
    
    slope, intercept, _, _, _ = stats.linregress(logMbary, logV)
    residuals = logV - (slope * logMbary + intercept)
    
    print(f"  Loaded {len(ra)} ALFALFA galaxies")
    
    return {'ra': ra, 'dec': dec, 'residuals': residuals, 'mass': logMbary, 'dist': dist}


def load_kids_with_residuals():
    """Load KiDS data and compute M-L residuals."""
    
    print("  Loading KiDS data...")
    
    bright_path = os.path.join(KIDS_DIR, "KiDS_DR4_brightsample.fits")
    lephare_path = os.path.join(KIDS_DIR, "KiDS_DR4_brightsample_LePhare.fits")
    
    if not os.path.exists(bright_path):
        print(f"  ERROR: Not found: {bright_path}")
        return None
    
    bright = Table.read(bright_path)
    lephare = Table.read(lephare_path)
    
    ra = np.array(bright['RAJ2000'])
    dec = np.array(bright['DECJ2000'])
    mag_r = np.array(lephare['MAG_ABS_r'])
    mass = np.array(lephare['MASS_MED'])
    
    mask = (
        np.isfinite(ra) & np.isfinite(dec) &
        np.isfinite(mag_r) & np.isfinite(mass) &
        (mass > 7) & (mass < 13)
    )
    
    ra, dec = ra[mask], dec[mask]
    mag_r, mass = mag_r[mask], mass[mask]
    
    slope, intercept, _, _, _ = stats.linregress(mass, mag_r)
    residuals = mag_r - (slope * mass + intercept)
    
    print(f"  Loaded {len(ra)} KiDS galaxies")
    
    return {'ra': ra, 'dec': dec, 'residuals': residuals, 'mass': mass}


def compute_density_proxy(ra, dec, n_neighbors=20, max_angle=1.0):
    """Compute angular density as gravitational potential proxy."""
    
    print("  Computing angular density (kappa proxy)...")
    
    # Convert to 3D unit vectors
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    
    coords = np.column_stack([x, y, z])
    tree = cKDTree(coords)
    
    # Find n-th nearest neighbor distance
    distances, _ = tree.query(coords, k=n_neighbors+1)
    r_n = distances[:, -1]
    
    # Angular density ~ 1/r^2 on sphere
    density = n_neighbors / (np.pi * r_n**2 + 1e-10)
    
    return np.log10(density + 1e-10)


def test_residual_density_correlation(galaxies, name):
    """Test correlation between residuals and local density."""
    
    print(f"\n" + "="*70)
    print(f" {name}: RESIDUAL-DENSITY CORRELATION")
    print("="*70)
    
    density = compute_density_proxy(galaxies['ra'], galaxies['dec'])
    residuals = galaxies['residuals']
    
    # Pearson correlation
    r, p = stats.pearsonr(density, residuals)
    
    print(f"\n  Pearson correlation: r = {r:.4f}, p = {p:.2e}")
    
    # Spearman for non-linear
    rho, p_spearman = stats.spearmanr(density, residuals)
    print(f"  Spearman correlation: rho = {rho:.4f}, p = {p_spearman:.2e}")
    
    return {
        'pearson_r': float(r),
        'pearson_p': float(p),
        'spearman_rho': float(rho),
        'spearman_p': float(p_spearman)
    }


def test_q_vs_r_density(galaxies, name):
    """Compare density environments for Q vs R dominated galaxies."""
    
    print(f"\n" + "="*70)
    print(f" {name}: Q vs R DENSITY COMPARISON")
    print("="*70)
    
    density = compute_density_proxy(galaxies['ra'], galaxies['dec'])
    residuals = galaxies['residuals']
    
    # Q-dominated: negative residuals (bottom 25%)
    # R-dominated: positive residuals (top 25%)
    q_mask = residuals < np.percentile(residuals, 25)
    r_mask = residuals > np.percentile(residuals, 75)
    
    density_q = density[q_mask]
    density_r = density[r_mask]
    
    mean_q = density_q.mean()
    mean_r = density_r.mean()
    std_q = density_q.std() / np.sqrt(len(density_q))
    std_r = density_r.std() / np.sqrt(len(density_r))
    
    t_stat, p_value = stats.ttest_ind(density_q, density_r)
    
    print(f"\n  Q-dominated (bottom 25% residuals):")
    print(f"    N = {len(density_q)}")
    print(f"    Mean log(density) = {mean_q:.4f} +/- {std_q:.4f}")
    
    print(f"\n  R-dominated (top 25% residuals):")
    print(f"    N = {len(density_r)}")
    print(f"    Mean log(density) = {mean_r:.4f} +/- {std_r:.4f}")
    
    print(f"\n  Difference: {mean_r - mean_q:.4f}")
    print(f"  T-statistic: {t_stat:.3f}")
    print(f"  P-value: {p_value:.2e}")
    
    if p_value < 0.05:
        if mean_r > mean_q:
            print(f"\n  RESULT: R-dominated in DENSER environments (p < 0.05)")
        else:
            print(f"\n  RESULT: Q-dominated in DENSER environments (p < 0.05)")
    else:
        print(f"\n  RESULT: No significant difference")
    
    return {
        'mean_density_q': float(mean_q),
        'mean_density_r': float(mean_r),
        'std_q': float(std_q),
        'std_r': float(std_r),
        'n_q': int(len(density_q)),
        'n_r': int(len(density_r)),
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }


def test_cross_correlation_alfalfa_kids(alfalfa, kids, max_sep_deg=1.0):
    """Cross-correlate ALFALFA and KiDS at same positions."""
    
    print(f"\n" + "="*70)
    print(" CROSS-CORRELATION: ALFALFA x KiDS")
    print("="*70)
    
    print(f"\n  Matching galaxies within {max_sep_deg} degree...")
    
    # Build KD-tree for KiDS
    kids_ra_rad = np.radians(kids['ra'])
    kids_dec_rad = np.radians(kids['dec'])
    kids_x = np.cos(kids_dec_rad) * np.cos(kids_ra_rad)
    kids_y = np.cos(kids_dec_rad) * np.sin(kids_ra_rad)
    kids_z = np.sin(kids_dec_rad)
    kids_coords = np.column_stack([kids_x, kids_y, kids_z])
    
    tree = cKDTree(kids_coords)
    
    # Query for ALFALFA positions
    alf_ra_rad = np.radians(alfalfa['ra'])
    alf_dec_rad = np.radians(alfalfa['dec'])
    alf_x = np.cos(alf_dec_rad) * np.cos(alf_ra_rad)
    alf_y = np.cos(alf_dec_rad) * np.sin(alf_ra_rad)
    alf_z = np.sin(alf_dec_rad)
    alf_coords = np.column_stack([alf_x, alf_y, alf_z])
    
    # Angular separation in radians
    max_sep_rad = np.radians(max_sep_deg)
    # Convert to chord length
    max_chord = 2 * np.sin(max_sep_rad / 2)
    
    distances, indices = tree.query(alf_coords, k=1)
    
    matched = distances < max_chord
    n_matched = matched.sum()
    
    print(f"  Matched pairs: {n_matched}")
    
    if n_matched < 100:
        print("  ERROR: Too few matches for correlation")
        return None
    
    alf_residuals = alfalfa['residuals'][matched]
    kids_residuals = kids['residuals'][indices[matched]]
    
    r, p = stats.pearsonr(alf_residuals, kids_residuals)
    
    print(f"\n  Correlation of residuals at same positions:")
    print(f"  Pearson r = {r:.4f}, p = {p:.2e}")
    
    # If ALFALFA (Q-dominated) and KiDS (R-dominated) show opposite residuals
    # at the same location, this supports Q2R2
    
    # Check sign agreement
    same_sign = np.sign(alf_residuals) == np.sign(kids_residuals)
    frac_same = same_sign.mean()
    
    print(f"\n  Fraction with same sign: {frac_same:.2%}")
    print(f"  Fraction with opposite sign: {1-frac_same:.2%}")
    
    if frac_same < 0.5:
        print(f"\n  RESULT: ALFALFA and KiDS show OPPOSITE residuals")
        print(f"         at same positions - supports Q2R2!")
    
    return {
        'n_matched': int(n_matched),
        'pearson_r': float(r),
        'pearson_p': float(p),
        'frac_same_sign': float(frac_same)
    }


def main():
    print("="*70)
    print(" PHASE 3 ALTERNATIVE: DENSITY CORRELATION ANALYSIS")
    print(" (Without healpy)")
    print("="*70)
    
    results = {}
    
    # Load data
    alfalfa = load_alfalfa_with_residuals()
    kids = load_kids_with_residuals()
    
    # Test 1: ALFALFA residual-density correlation
    if alfalfa is not None:
        results['alfalfa_correlation'] = test_residual_density_correlation(alfalfa, "ALFALFA")
        results['alfalfa_q_vs_r'] = test_q_vs_r_density(alfalfa, "ALFALFA")
    
    # Test 2: KiDS residual-density correlation
    if kids is not None:
        results['kids_correlation'] = test_residual_density_correlation(kids, "KiDS")
        results['kids_q_vs_r'] = test_q_vs_r_density(kids, "KiDS")
    
    # Test 3: Cross-correlation
    if alfalfa is not None and kids is not None:
        results['cross_correlation'] = test_cross_correlation_alfalfa_kids(alfalfa, kids)
    
    # Summary
    print("\n" + "="*70)
    print(" PHASE 3 SUMMARY")
    print("="*70)
    
    print("""
  This analysis tests whether Q2R2 residuals correlate with
  local gravitational environment (density as proxy for kappa).
  
  Key predictions:
  1. Residuals should correlate with density
  2. Q-dominated and R-dominated galaxies should inhabit
     different density environments
  3. ALFALFA and KiDS residuals at same positions should
     show OPPOSITE signs (Q vs R dominance)
    """)
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "phase3_alternative_results.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
