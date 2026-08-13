#!/usr/bin/env python3
"""
===============================================================================
ACT-DR5 MCMF - TEST Q²R² DANS LES CLUSTERS
===============================================================================

Test du pattern U-shape dans les résidus de la relation masse-richesse
pour 6,237 clusters ACT-DR5 MCMF.

Note: Ce catalogue n'a pas Y_SZ directement, mais on peut utiliser:
- Relation M-L (masse vs richesse optique)
- Ou cross-matcher avec le catalogue ACT-DR5 original pour Y_500

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
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
import warnings
warnings.filterwarnings('ignore')

# Cosmology
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "data", "level2_multicontext", "clusters")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experimental", "04_RESULTS", "test_results")


def find_act_file():
    """Find ACT-DR5 MCMF catalog."""
    patterns = [
        os.path.join(DATA_DIR, "ACT_DR5_MCMF*.fits"),
        os.path.join(DATA_DIR, "ACT*.fits"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def load_act_data():
    """Load ACT-DR5 MCMF catalog."""
    
    act_path = find_act_file()
    if not act_path:
        print("  ERROR: ACT catalog not found!")
        return None
    
    print(f"  Loading: {os.path.basename(act_path)}")
    data = Table.read(act_path)
    
    print(f"  Total clusters: {len(data)}")
    print(f"  Columns: {data.colnames}")
    
    return data


def explore_data(data):
    """Explore the data structure."""
    
    print("\n" + "="*70)
    print(" DATA EXPLORATION")
    print("="*70)
    
    # Redshift
    z1 = np.array(data['z1C'])
    z2 = np.array(data['z2C'])
    z_valid1 = z1[np.isfinite(z1) & (z1 > 0)]
    z_valid2 = z2[np.isfinite(z2) & (z2 > 0)]
    
    print(f"\n  z1C: {len(z_valid1)} valid, range {z_valid1.min():.3f}-{z_valid1.max():.3f}")
    print(f"  z2C: {len(z_valid2)} valid, range {z_valid2.min():.3f}-{z_valid2.max():.3f}")
    
    # Use z1C as primary redshift
    z = z1
    
    # Mass
    m1 = np.array(data['M500-1C'])
    m2 = np.array(data['M500-2C'])
    m_valid1 = m1[np.isfinite(m1) & (m1 > 0)]
    m_valid2 = m2[np.isfinite(m2) & (m2 > 0)]
    
    print(f"\n  M500-1C: {len(m_valid1)} valid, range {m_valid1.min():.2e}-{m_valid1.max():.2e}")
    print(f"  M500-2C: {len(m_valid2)} valid, range {m_valid2.min():.2e}-{m_valid2.max():.2e}")
    
    # Richness (optical)
    L1 = np.array(data['L1C'])
    L2 = np.array(data['L2C'])
    L_valid1 = L1[np.isfinite(L1) & (L1 > 0)]
    L_valid2 = L2[np.isfinite(L2) & (L2 > 0)]
    
    print(f"\n  L1C (richness): {len(L_valid1)} valid, range {L_valid1.min():.1f}-{L_valid1.max():.1f}")
    print(f"  L2C (richness): {len(L_valid2)} valid, range {L_valid2.min():.1f}-{L_valid2.max():.1f}")
    
    # SNR
    snr = np.array(data['SNR'])
    snr_valid = snr[np.isfinite(snr)]
    print(f"\n  SNR: {len(snr_valid)} valid, range {snr_valid.min():.1f}-{snr_valid.max():.1f}")
    
    # Contamination
    fcont = np.array(data['fcont1C'])
    fcont_valid = fcont[np.isfinite(fcont)]
    print(f"  fcont1C: range {fcont_valid.min():.3f}-{fcont_valid.max():.3f}")
    
    return z, m1, L1, snr, fcont


def test_mass_richness_relation(data):
    """
    Test 1: Mass-Richness relation (M-L)
    
    The mass-richness relation is a well-established scaling relation.
    We test for U-shape in the residuals vs environment.
    """
    
    print("\n" + "="*70)
    print(" TEST 1: MASS-RICHNESS RELATION (M-L)")
    print("="*70)
    
    # Extract data
    z = np.array(data['z1C'])
    M = np.array(data['M500-1C'])  # 10^14 M_sun
    L = np.array(data['L1C'])       # Optical richness
    snr = np.array(data['SNR'])
    fcont = np.array(data['fcont1C'])
    ra = np.array(data['RAJ2000'])
    dec = np.array(data['DEJ2000'])
    
    # Quality cuts
    mask = (
        np.isfinite(z) & (z > 0.05) & (z < 1.5) &
        np.isfinite(M) & (M > 0) &
        np.isfinite(L) & (L > 5) &  # Minimum richness
        np.isfinite(snr) & (snr > 4) &
        np.isfinite(fcont) & (fcont < 0.2)  # Low contamination
    )
    
    z = z[mask]
    M = M[mask]
    L = L[mask]
    snr = snr[mask]
    ra = ra[mask]
    dec = dec[mask]
    
    print(f"\n  After quality cuts: {len(z)} clusters")
    print(f"  z range: {z.min():.3f} - {z.max():.3f}")
    print(f"  M range: {M.min():.2e} - {M.max():.2e}")
    print(f"  L range: {L.min():.1f} - {L.max():.1f}")
    
    # Log transform
    log_M = np.log10(M)
    log_L = np.log10(L)
    
    # Fit M-L relation: log(M) = α × log(L) + β
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_L, log_M)
    
    print(f"\n  M-L relation: log(M) = {slope:.3f} × log(L) + {intercept:.3f}")
    print(f"  Correlation: r = {r_value:.3f}")
    print(f"  Expected slope: ~0.8-1.2 (mass ∝ richness)")
    
    if slope > 0.3:
        print("  ✓ Positive slope as expected!")
    else:
        print("  ✗ Unexpected slope!")
    
    # Compute residuals
    log_M_pred = slope * log_L + intercept
    residuals = log_M - log_M_pred
    
    print(f"\n  Residual scatter: {residuals.std():.3f} dex")
    
    return {
        'z': z,
        'M': M,
        'L': L,
        'log_M': log_M,
        'log_L': log_L,
        'residuals': residuals,
        'ra': ra,
        'dec': dec,
        'snr': snr,
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value
    }


def compute_environment(ra, dec, z, n_neighbors=10):
    """
    Compute local environment based on cluster density.
    
    For clusters, environment = density of nearby clusters.
    """
    
    print("\n  Computing environment (cluster density)...")
    
    from scipy.spatial import cKDTree
    
    # Convert to Cartesian (approximate for small angles)
    # Use comoving distance for radial coordinate
    D_c = np.array([COSMO.comoving_distance(zi).value for zi in z])
    
    x = D_c * np.cos(np.radians(dec)) * np.cos(np.radians(ra))
    y = D_c * np.cos(np.radians(dec)) * np.sin(np.radians(ra))
    z_coord = D_c * np.sin(np.radians(dec))
    
    coords = np.column_stack([x, y, z_coord])
    
    # Build KD-tree
    tree = cKDTree(coords)
    
    # Find distance to n-th nearest neighbor
    distances, _ = tree.query(coords, k=n_neighbors+1)
    
    # Local density ∝ 1 / volume to n-th neighbor
    r_n = distances[:, -1]  # Distance to n-th neighbor
    density = n_neighbors / (4/3 * np.pi * r_n**3)
    
    # Log density
    log_density = np.log10(density + 1e-10)
    
    # Normalize to [0, 1]
    env = (log_density - log_density.min()) / (log_density.max() - log_density.min())
    
    print(f"  Environment range: {env.min():.3f} - {env.max():.3f}")
    
    return env


def test_ushape(residuals, env, label="Full sample"):
    """Test for U-shape in residuals vs environment."""
    
    print(f"\n  {label}:")
    
    # Fit quadratic: residuals = a × env² + b × env + c
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c
    
    try:
        popt, pcov = curve_fit(quadratic, env, residuals)
        a, b, c = popt
        a_err = np.sqrt(pcov[0, 0])
        
        significance = abs(a) / a_err if a_err > 0 else 0
        
        print(f"    a = {a:.6f} ± {a_err:.6f} ({significance:.1f}σ)")
        print(f"    U-shape: {'YES' if a > 0 and significance > 2 else 'NO'}")
        
        return {
            'a': a,
            'a_err': a_err,
            'b': b,
            'c': c,
            'significance': significance,
            'ushape_detected': a > 0 and significance > 2
        }
    except Exception as e:
        print(f"    Fit failed: {e}")
        return None


def test_sign_inversion(results, data):
    """
    Test for sign inversion between Q-dominated and R-dominated clusters.
    
    For clusters:
    - Q-dominated: Low mass, high gas fraction (proxy: low M, high L/M)
    - R-dominated: High mass, low gas fraction (proxy: high M, low L/M)
    """
    
    print("\n" + "="*70)
    print(" TEST 2: SIGN INVERSION (Q-dominated vs R-dominated)")
    print("="*70)
    
    residuals = results['residuals']
    env = results['env']
    M = results['M']
    L = results['L']
    
    # Define Q and R proxies
    # Q proxy: richness per unit mass (high = gas-rich)
    # R proxy: mass (high = massive/evolved)
    
    L_over_M = L / M  # Richness per unit mass
    
    # Split by L/M ratio
    median_L_M = np.median(L_over_M)
    
    q_mask = L_over_M > median_L_M  # High L/M = Q-dominated (gas-rich)
    r_mask = L_over_M <= median_L_M  # Low L/M = R-dominated (massive)
    
    print(f"\n  Q-dominated (high L/M): N = {q_mask.sum()}")
    q_result = test_ushape(residuals[q_mask], env[q_mask], "Q-dominated")
    
    print(f"\n  R-dominated (low L/M): N = {r_mask.sum()}")
    r_result = test_ushape(residuals[r_mask], env[r_mask], "R-dominated")
    
    # Check sign inversion
    if q_result and r_result:
        sign_inversion = (q_result['a'] > 0 and r_result['a'] < 0)
        print(f"\n  Sign inversion: {'YES ✓' if sign_inversion else 'NO'}")
        print(f"    Q-dominated: a = {q_result['a']:.6f}")
        print(f"    R-dominated: a = {r_result['a']:.6f}")
        
        return {
            'q_dominated': q_result,
            'r_dominated': r_result,
            'sign_inversion': sign_inversion
        }
    
    return None


def main():
    print("="*70)
    print(" ACT-DR5 MCMF - Q²R² TEST IN GALAXY CLUSTERS")
    print(" N = 6,237 clusters (×11 more than Planck!)")
    print("="*70)
    
    # Load data
    data = load_act_data()
    if data is None:
        return
    
    # Explore
    explore_data(data)
    
    # Test 1: Mass-Richness relation
    results = test_mass_richness_relation(data)
    
    if results is None:
        print("\n  ERROR: Could not fit M-L relation")
        return
    
    # Compute environment
    env = compute_environment(results['ra'], results['dec'], results['z'])
    results['env'] = env
    
    # Test U-shape
    print("\n" + "="*70)
    print(" TEST: U-SHAPE IN RESIDUALS")
    print("="*70)
    
    full_result = test_ushape(results['residuals'], env, "Full sample")
    
    # Test sign inversion
    sign_result = test_sign_inversion(results, data)
    
    # Summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    print(f"\n  N clusters: {len(results['z'])}")
    print(f"  M-L correlation: r = {results['r_value']:.3f}")
    
    if full_result:
        print(f"\n  U-shape coefficient: a = {full_result['a']:.6f} ± {full_result['a_err']:.6f}")
        print(f"  Significance: {full_result['significance']:.1f}σ")
        print(f"  U-shape detected: {full_result['ushape_detected']}")
    
    if sign_result:
        print(f"\n  Sign inversion: {sign_result['sign_inversion']}")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "act_dr5_mcmf_results.json")
    
    output = {
        'n_clusters': len(results['z']),
        'ml_slope': results['slope'],
        'ml_intercept': results['intercept'],
        'ml_correlation': results['r_value'],
        'ushape': full_result,
        'sign_inversion': sign_result
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
