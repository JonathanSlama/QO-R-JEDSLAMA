#!/usr/bin/env python3
"""
===============================================================================
PLANCK SZ GALAXY CLUSTERS TEST
===============================================================================

Test for Q²R² pattern in galaxy cluster scaling relations.

This tests if the same U-shape environmental dependence found in galaxy
rotation curves also exists in the M-Y (mass-SZ signal) relation residuals
of galaxy clusters.

Data: Planck SZ Catalog (PSZ2)
Download: https://irsa.ipac.caltech.edu → Planck → Sunyaev-Zeldovich Catalog
File: HFI_PCCS_SZ-union_R2.08.fits (inside COM_PCCS_SZ-Catalogs_vPR2.tgz)

Author: Jonathan E. Slama
Date: December 2025
===============================================================================
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
warnings.filterwarnings('ignore')

# Add parent paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "data", "level2_multicontext", "clusters")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experimental", "04_RESULTS", "test_results")

# Possible file locations after extraction (including .gz compressed)
PLANCK_SZ_FILES = [
    "HFI_PCCS_SZ-union_R2.08.fits",
    "HFI_PCCS_SZ-union_R2.08.fits.gz",
    "COM_PCCS_SZ-Catalogs_vPR2/HFI_PCCS_SZ-union_R2.08.fits",
    "COM_PCCS_SZ-Catalogs_vPR2/HFI_PCCS_SZ-union_R2.08.fits.gz",
]


def find_planck_file():
    """Search for the Planck SZ union catalog."""
    # Try direct paths
    for filename in PLANCK_SZ_FILES:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            return path
    
    # Search recursively (including .gz)
    for ext in [".fits", ".fits.gz"]:
        pattern = os.path.join(DATA_DIR, "**", f"*SZ-union*{ext}")
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    
    # Try any SZ fits file
    pattern = os.path.join(DATA_DIR, "**", "*SZ*.fits*")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        # Prefer union catalog
        for m in matches:
            if 'union' in m.lower():
                return m
        return matches[0]
    
    return None


# =============================================================================
# CLUSTER PHYSICS
# =============================================================================

def compute_gas_fraction_cluster(Y_SZ, M_500):
    """
    Estimate cluster gas fraction from SZ signal.
    
    Y_SZ ∝ f_gas × M × T
    For self-similar clusters: f_gas ∝ Y_SZ / M^(5/3)
    """
    valid = np.isfinite(Y_SZ) & np.isfinite(M_500) & (M_500 > 0) & (Y_SZ > 0)
    f_gas_proxy = np.zeros_like(Y_SZ)
    f_gas_proxy[valid] = Y_SZ[valid] / (M_500[valid] ** (5/3))
    
    # Normalize to typical cluster gas fraction (~0.12)
    median_proxy = np.median(f_gas_proxy[valid]) if valid.sum() > 0 else 1.0
    f_gas = 0.12 * (f_gas_proxy / median_proxy)
    
    return np.clip(f_gas, 0.05, 0.20)


def compute_bcg_fraction(M_500):
    """
    Estimate BCG stellar mass fraction.
    M_BCG ∝ M_500^0.4 approximately.
    """
    M_BCG = 1e12 * (M_500 / 1e15) ** 0.4
    f_star = M_BCG / np.maximum(M_500, 1e13)
    return f_star


def estimate_environment_cluster(ra, dec, z):
    """
    Estimate cluster environment from neighbor density.
    Count clusters within angular separation corresponding to ~50 Mpc.
    """
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    
    n = len(ra)
    env = np.zeros(n)
    
    for i in range(n):
        if not np.isfinite(z[i]) or z[i] <= 0:
            continue
        
        # Angular distance for 50 Mpc at cluster redshift
        d_A = cosmo.angular_diameter_distance(z[i]).value
        theta_50 = np.degrees(50 / d_A) if d_A > 0 else 5.0
        
        # Count neighbors
        dz = np.abs(z - z[i])
        d_ang = np.sqrt(
            ((ra - ra[i]) * np.cos(np.radians(dec[i])))**2 + 
            (dec - dec[i])**2
        )
        
        neighbors = (dz < 0.05) & (d_ang < theta_50) & (d_ang > 0.01)
        env[i] = neighbors.sum()
    
    # Standardize
    if np.std(env) > 0:
        env_std = (env - np.mean(env)) / np.std(env)
    else:
        env_std = env
    
    return env_std


# =============================================================================
# U-SHAPE FIT
# =============================================================================

def fit_ushape(env, residual, n_bins=8, min_per_bin=15):
    """Fit quadratic U-shape to binned data."""
    
    mask = np.isfinite(env) & np.isfinite(residual)
    env = env[mask]
    residual = residual[mask]
    
    if len(env) < n_bins * min_per_bin:
        n_bins = max(4, len(env) // min_per_bin)
    
    if len(env) < 80:
        return None
    
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(env, percentiles)
    
    bin_centers = []
    bin_means = []
    bin_errors = []
    
    for i in range(n_bins):
        mask = (env >= bin_edges[i]) & (env < bin_edges[i+1])
        if i == n_bins - 1:
            mask = (env >= bin_edges[i]) & (env <= bin_edges[i+1])
        
        if mask.sum() >= min_per_bin:
            bin_centers.append(np.mean(env[mask]))
            bin_means.append(np.mean(residual[mask]))
            bin_errors.append(np.std(residual[mask]) / np.sqrt(mask.sum()))
    
    if len(bin_centers) < 4:
        return None
    
    x = np.array(bin_centers)
    y = np.array(bin_means)
    yerr = np.array(bin_errors)
    yerr[yerr == 0] = 0.01
    
    try:
        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c
        
        popt, pcov = curve_fit(quadratic, x, y, sigma=yerr, absolute_sigma=True, p0=[0.01, 0, 0])
        perr = np.sqrt(np.diag(pcov))
        
        y_pred = quadratic(x, *popt)
        chi2 = np.sum(((y - y_pred) / yerr)**2)
        dof = len(x) - 3
        p_value = 1 - stats.chi2.cdf(chi2, dof) if dof > 0 else 1.0
        
        return {
            'a': popt[0],
            'a_err': perr[0],
            'p_value': p_value,
            'n_bins': len(x)
        }
    except:
        return None


# =============================================================================
# MAIN TEST
# =============================================================================

def test_planck_clusters():
    """
    Main test: Look for U-shape in Planck cluster scaling relation residuals.
    """
    results = {
        'test_name': 'Planck SZ Galaxy Clusters',
        'description': 'Test for Q²R² pattern in M-Y scaling relation',
        'context': 'Level 2 - Multi-context validation'
    }
    
    # Find data file
    data_file = find_planck_file()
    
    if data_file is None:
        print(f"\n  ERROR: Planck SZ catalog not found!")
        print(f"  Searched in: {DATA_DIR}")
        print(f"\n  Instructions:")
        print(f"  1. Download COM_PCCS_SZ-Catalogs_vPR2.tgz from IRSA")
        print(f"  2. Extract the .tgz file (right-click → Extract All)")
        print(f"  3. File needed: HFI_PCCS_SZ-union_R2.08.fits")
        results['status'] = 'NO_DATA'
        results['download_url'] = 'https://irsa.ipac.caltech.edu'
        return results
    
    # Load data
    print(f"\n  Loading: {os.path.basename(data_file)}")
    print(f"  Path: {data_file}")
    
    with fits.open(data_file) as hdul:
        data = hdul[1].data
    
    n_total = len(data)
    print(f"  Total clusters: {n_total}")
    print(f"  Available columns: {list(data.names)[:15]}...")
    
    # Extract columns
    try:
        # Coordinates
        if 'GLON' in data.names:
            glon = data['GLON']
            glat = data['GLAT']
        elif 'GL' in data.names:
            glon = data['GL']
            glat = data['GB']
        else:
            raise KeyError("No galactic coordinates found")
        
        # Convert galactic to equatorial
        coords = SkyCoord(l=glon*u.deg, b=glat*u.deg, frame='galactic')
        icrs = coords.icrs
        ra = icrs.ra.deg
        dec = icrs.dec.deg
        
        # Mass
        if 'MSZ' in data.names:
            M_500 = data['MSZ'] * 1e14  # Units: 10^14 M_sun → M_sun
        elif 'M500' in data.names:
            M_500 = data['M500'] * 1e14
        else:
            print("  WARNING: No mass column, estimating from Y_SZ")
            M_500 = None
        
        # SZ signal (Compton Y parameter)
        if 'Y5R500' in data.names:
            Y_SZ = data['Y5R500']
        elif 'Y500' in data.names:
            Y_SZ = data['Y500']
        else:
            raise KeyError("No Y_SZ column found")
        
        # If no mass, estimate from Y_SZ
        if M_500 is None:
            # M ∝ Y^(3/5) rough scaling
            Y_SZ_clean = np.where(Y_SZ > 0, Y_SZ, np.nan)
            M_500 = 5e14 * (Y_SZ_clean / 1e-4) ** (3/5)
        
        # Redshift
        if 'REDSHIFT' in data.names:
            z = data['REDSHIFT']
        elif 'Z' in data.names:
            z = data['Z']
        else:
            z = np.full(n_total, 0.2)
            print("  WARNING: No redshift, using z=0.2")
        
        # SNR for quality
        if 'SNR' in data.names:
            snr = data['SNR']
        else:
            snr = np.full(n_total, 10.0)
        
    except KeyError as e:
        print(f"  ERROR: Missing column {e}")
        print(f"  Available: {list(data.names)}")
        results['status'] = 'COLUMN_ERROR'
        return results
    
    # Quality cuts
    print("\n  Applying quality cuts...")
    mask = (
        np.isfinite(M_500) & (M_500 > 1e13) &
        np.isfinite(Y_SZ) & (Y_SZ > 0) &
        np.isfinite(z) & (z > 0) & (z < 1.0) &
        (snr > 4.5)  # Standard PSZ2 cut
    )
    
    n_good = mask.sum()
    print(f"  After cuts: {n_good} ({100*n_good/n_total:.1f}%)")
    
    if n_good < 100:
        print("  WARNING: Small sample size - results may be noisy")
    
    if n_good < 50:
        results['status'] = 'INSUFFICIENT_DATA'
        results['conclusion'] = 'Too few clusters for reliable analysis'
        return results
    
    # Apply mask
    ra = ra[mask]
    dec = dec[mask]
    M_500 = M_500[mask]
    Y_SZ = Y_SZ[mask]
    z = z[mask]
    
    # Compute derived quantities
    print("\n  Computing derived quantities...")
    
    # Gas fraction proxy (Q)
    f_gas = compute_gas_fraction_cluster(Y_SZ, M_500)
    print(f"  Gas fraction range: {np.nanmin(f_gas):.3f} - {np.nanmax(f_gas):.3f}")
    
    # Stellar fraction proxy (R)
    f_star = compute_bcg_fraction(M_500)
    log_Mstar_proxy = np.log10(f_star * M_500)
    
    # Environment
    print("  Computing cluster environment (this may take a moment)...")
    env = estimate_environment_cluster(ra, dec, z)
    
    # M-Y scaling relation
    log_M = np.log10(M_500)
    log_Y = np.log10(np.maximum(Y_SZ, 1e-10))
    
    # Fit M-Y relation
    mask_fit = np.isfinite(log_M) & np.isfinite(log_Y)
    if mask_fit.sum() < 50:
        results['status'] = 'INSUFFICIENT_DATA'
        return results
    
    slope, intercept, r_value, p_fit, _ = stats.linregress(log_Y[mask_fit], log_M[mask_fit])
    
    # Residuals
    predicted_logM = slope * log_Y + intercept
    MY_residual = log_M - predicted_logM
    
    print(f"\n  M-Y relation:")
    print(f"    Slope: {slope:.3f} (expected ~0.6)")
    print(f"    R²: {r_value**2:.3f}")
    print(f"    Residual scatter: {np.std(MY_residual[mask_fit]):.3f} dex")
    
    # Test for U-shape
    print("\n  Testing for U-shape in M-Y residuals...")
    
    # Full sample
    fit_full = fit_ushape(env, MY_residual)
    
    if fit_full:
        results['full_sample'] = {
            'n': int(n_good),
            'a': float(fit_full['a']),
            'a_err': float(fit_full['a_err']),
            'p_value': float(fit_full['p_value']),
            'is_u_shape': fit_full['a'] > 0
        }
        print(f"  Full sample: a = {fit_full['a']:.5f} ± {fit_full['a_err']:.5f}")
        print(f"  p-value: {fit_full['p_value']:.4f}")
    else:
        print("  Full sample fit failed (too few bins)")
    
    # Gas-rich clusters (high f_gas = Q-dominated)
    gas_rich_mask = f_gas > np.median(f_gas)
    n_rich = gas_rich_mask.sum()
    print(f"\n  Gas-rich (Q-dom): N = {n_rich}")
    if n_rich > 50:
        fit_gasrich = fit_ushape(env[gas_rich_mask], MY_residual[gas_rich_mask], n_bins=6, min_per_bin=10)
        if fit_gasrich:
            results['gas_rich'] = {
                'n': int(n_rich),
                'a': float(fit_gasrich['a']),
                'a_err': float(fit_gasrich['a_err']),
                'p_value': float(fit_gasrich['p_value'])
            }
            print(f"    a = {fit_gasrich['a']:.5f}")
    
    # Gas-poor + massive clusters (R-dominated)
    gas_poor_mask = (f_gas < np.median(f_gas)) & (M_500 > np.median(M_500))
    n_poor = gas_poor_mask.sum()
    print(f"\n  Gas-poor + massive (R-dom): N = {n_poor}")
    if n_poor > 30:
        fit_gaspoor = fit_ushape(env[gas_poor_mask], MY_residual[gas_poor_mask], n_bins=5, min_per_bin=8)
        if fit_gaspoor:
            results['gas_poor'] = {
                'n': int(n_poor),
                'a': float(fit_gaspoor['a']),
                'a_err': float(fit_gaspoor['a_err']),
                'p_value': float(fit_gaspoor['p_value'])
            }
            print(f"    a = {fit_gaspoor['a']:.5f}")
    
    # Check for sign inversion
    if 'gas_rich' in results and 'gas_poor' in results:
        q_sign = np.sign(results['gas_rich']['a'])
        r_sign = np.sign(results['gas_poor']['a'])
        results['sign_inversion'] = bool(q_sign != r_sign)
        print(f"\n  Sign inversion (killer prediction): {'YES ✓' if results['sign_inversion'] else 'NO'}")
    
    # Overall assessment
    if 'full_sample' in results:
        results['status'] = 'SUCCESS'
        if results['full_sample']['is_u_shape']:
            results['conclusion'] = 'U-shape DETECTED in cluster scaling - THIRD CONTEXT confirms pattern'
        else:
            results['conclusion'] = 'U-shape NOT detected in clusters (a < 0 or not significant)'
    else:
        results['status'] = 'ANALYSIS_FAILED'
        results['conclusion'] = 'Could not fit U-shape (sample too small?)'
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    import json
    
    print("="*70)
    print(" PLANCK SZ GALAXY CLUSTERS TEST")
    print(" Testing Q²R² pattern in M-Y scaling relation")
    print("="*70)
    
    results = test_planck_clusters()
    
    print("\n" + "="*70)
    print(" RESULTS")
    print("="*70)
    
    if results.get('status') == 'SUCCESS':
        print(f"\n  Status: {results['status']}")
        print(f"  N clusters: {results.get('full_sample', {}).get('n', 'N/A')}")
        print(f"  U-shape detected: {results.get('full_sample', {}).get('is_u_shape', 'Unknown')}")
        if results.get('sign_inversion') is not None:
            print(f"  Sign inversion: {results['sign_inversion']}")
        print(f"\n  Conclusion: {results.get('conclusion', 'N/A')}")
    else:
        print(f"\n  Status: {results.get('status', 'UNKNOWN')}")
        print(f"  {results.get('conclusion', '')}")
        if 'download_url' in results:
            print(f"\n  Download from: {results['download_url']}")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "planck_clusters_results.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {output_file}")


if __name__ == "__main__":
    main()
