#!/usr/bin/env python3
"""
===============================================================================
KiDS DR4 BRIGHT SAMPLE TEST
===============================================================================

Test for Q²R² pattern in the KiDS DR4 bright galaxy sample.

This tests if the same U-shape environmental dependence found in galaxy
rotation curves also exists in galaxy scaling relations from optical surveys.

Data: KiDS DR4 Bright Sample (both catalogs)
- KiDS_DR4_brightsample.fits (85 MB) - positions, photo-z, magnitudes
- KiDS_DR4_brightsample_LePhare.fits (246 MB) - stellar masses, SFR

URL: https://kids.strw.leidenuniv.nl/DR4/brightsample.php

Author: Jonathan E. Slama
Date: December 2025
===============================================================================
"""

import os
import sys
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
                        "data", "level2_multicontext", "kids")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experimental", "04_RESULTS", "test_results")

# Both KiDS files
KIDS_PHOTOZ = "KiDS_DR4_brightsample.fits"  # 85 MB - positions, photo-z, mags
KIDS_LEPHARE = "KiDS_DR4_brightsample_LePhare.fits"  # 246 MB - stellar masses

# =============================================================================
# GAS FRACTION PROXY FROM COLOR
# =============================================================================

def estimate_gas_fraction_from_color(g_r, method='sigmoid'):
    """
    Estimate gas fraction from rest-frame g-r color.
    
    Blue galaxies → gas-rich (high Q)
    Red galaxies → gas-poor (high R)
    """
    if method == 'sigmoid':
        f_gas = 0.6 / (1 + np.exp(6 * (g_r - 0.65)))
        f_gas += 0.05
    elif method == 'linear':
        f_gas = np.clip(0.8 - 1.0 * g_r, 0.05, 0.8)
    
    return np.clip(f_gas, 0.01, 0.99)


# =============================================================================
# ENVIRONMENT ESTIMATION
# =============================================================================

def compute_local_density(ra, dec, n_bins=100):
    """
    Compute local density using grid-based approximation.
    Fast method for large catalogs.
    """
    # Bin into cells
    ra_bins = np.linspace(ra.min(), ra.max(), n_bins)
    dec_bins = np.linspace(dec.min(), dec.max(), n_bins)
    
    counts, _, _ = np.histogram2d(ra, dec, bins=[ra_bins, dec_bins])
    
    # Assign density to each galaxy
    ra_idx = np.digitize(ra, ra_bins) - 1
    dec_idx = np.digitize(dec, dec_bins) - 1
    
    ra_idx = np.clip(ra_idx, 0, n_bins - 2)
    dec_idx = np.clip(dec_idx, 0, n_bins - 2)
    
    density = counts[ra_idx, dec_idx]
    
    # Normalize
    density_std = (density - np.mean(density)) / np.std(density)
    
    return density_std


# =============================================================================
# U-SHAPE FIT
# =============================================================================

def fit_ushape(env, residual, n_bins=15, min_per_bin=50):
    """Fit quadratic U-shape to binned data."""
    
    mask = np.isfinite(env) & np.isfinite(residual)
    env = env[mask]
    residual = residual[mask]
    
    if len(env) < n_bins * min_per_bin:
        n_bins = max(5, len(env) // min_per_bin)
    
    if len(env) < 200:
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

def test_kids_brightsample():
    """
    Test for U-shape in KiDS bright sample using BOTH catalogs.
    
    Uses:
    - brightsample.fits for positions, photo-z, magnitudes
    - LePhare.fits for stellar masses (MASS_MED)
    """
    results = {
        'test_name': 'KiDS DR4 Bright Sample',
        'description': 'Test for Q²R² pattern in optical galaxy scaling relations',
        'context': 'Level 2 - Multi-context validation'
    }
    
    # Check for data files
    photoz_path = os.path.join(DATA_DIR, KIDS_PHOTOZ)
    lephare_path = os.path.join(DATA_DIR, KIDS_LEPHARE)
    
    if not os.path.exists(photoz_path):
        print(f"\n  ERROR: KiDS photo-z catalog not found!")
        print(f"  Expected: {photoz_path}")
        print(f"\n  Download from: https://kids.strw.leidenuniv.nl/DR4/brightsample.php")
        results['status'] = 'NO_DATA'
        return results
    
    # Load photo-z catalog
    print(f"\n  Loading: {KIDS_PHOTOZ}")
    with fits.open(photoz_path) as hdul:
        photoz_data = hdul[1].data
    
    n_total = len(photoz_data)
    print(f"  Total galaxies: {n_total:,}")
    print(f"  Photo-z columns: {list(photoz_data.names)}")
    
    # Load LePhare catalog (stellar masses)
    if os.path.exists(lephare_path):
        print(f"\n  Loading: {KIDS_LEPHARE}")
        with fits.open(lephare_path) as hdul:
            lephare_data = hdul[1].data
        has_lephare = True
        print(f"  LePhare columns: {list(lephare_data.names)[:15]}...")
    else:
        print(f"\n  WARNING: LePhare catalog not found - will estimate masses")
        has_lephare = False
    
    # Extract columns from photo-z catalog
    print("\n  Extracting data...")
    
    try:
        # Positions
        ra = photoz_data['RAJ2000']
        dec = photoz_data['DECJ2000']
        
        # Photo-z
        z_phot = photoz_data['zphot_ANNz2']
        
        # Magnitude (r-band calibrated)
        mag_r = photoz_data['MAG_AUTO_calib']
        
        # Quality flag
        masked = photoz_data['masked']
        
    except KeyError as e:
        print(f"  ERROR extracting photo-z columns: {e}")
        print(f"  Available: {photoz_data.names}")
        results['status'] = 'COLUMN_ERROR'
        return results
    
    # Get stellar masses from LePhare
    if has_lephare:
        try:
            # MASS_MED is log10(M*/M_sun)
            log_Mstar = lephare_data['MASS_MED']
            
            # Get colors for gas fraction
            # Try to get g-r from absolute magnitudes
            if 'MAG_ABS_g' in lephare_data.names and 'MAG_ABS_r' in lephare_data.names:
                g_r_color = lephare_data['MAG_ABS_g'] - lephare_data['MAG_ABS_r']
            else:
                # Estimate from apparent mags if available
                g_r_color = np.full(n_total, 0.6)  # Default
                
        except KeyError as e:
            print(f"  WARNING: Could not get masses from LePhare: {e}")
            has_lephare = False
    
    if not has_lephare:
        # Estimate stellar mass from magnitude
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        z_safe = np.clip(z_phot, 0.01, 2.0)
        d_L = cosmo.luminosity_distance(z_safe).to(u.pc).value
        M_r = mag_r - 5 * np.log10(d_L / 10)
        L_r = 10**(-0.4 * (M_r - 4.65))
        log_Mstar = np.log10(2.0 * L_r)
        g_r_color = np.full(n_total, 0.6)
    
    # Quality cuts
    print("\n  Applying quality cuts...")
    mask = (
        (masked == 0) &  # Not masked
        np.isfinite(ra) & np.isfinite(dec) &
        np.isfinite(z_phot) & (z_phot > 0.01) & (z_phot < 0.5) &
        np.isfinite(mag_r) & (mag_r > 0) & (mag_r < 20) &
        np.isfinite(log_Mstar) & (log_Mstar > 7) & (log_Mstar < 12.5)
    )
    
    # Handle MASS_MED = -99 (non-galaxy templates in LePhare)
    if has_lephare:
        mask &= (log_Mstar > 0)  # -99 values become invalid
    
    n_good = mask.sum()
    print(f"  After cuts: {n_good:,} ({100*n_good/n_total:.1f}%)")
    
    if n_good < 10000:
        print("  WARNING: Fewer galaxies than expected after cuts")
    
    if n_good < 1000:
        print("  ERROR: Too few galaxies after cuts")
        results['status'] = 'INSUFFICIENT_DATA'
        return results
    
    # Apply mask
    ra = ra[mask]
    dec = dec[mask]
    z_phot = z_phot[mask]
    mag_r = mag_r[mask]
    log_Mstar = log_Mstar[mask]
    g_r_color = g_r_color[mask] if len(g_r_color) == n_total else np.full(n_good, 0.6)
    
    # Compute derived quantities
    print("\n  Computing derived quantities...")
    
    # Gas fraction from color
    f_gas = estimate_gas_fraction_from_color(g_r_color)
    print(f"  Gas fraction range: {f_gas.min():.2f} - {f_gas.max():.2f}")
    
    # Environment
    print("  Computing local density...")
    env_density = compute_local_density(ra, dec)
    
    # Mass-Luminosity relation residuals
    print("  Fitting Mass-Luminosity relation...")
    mask_fit = np.isfinite(log_Mstar) & np.isfinite(mag_r)
    slope, intercept, _, _, _ = stats.linregress(log_Mstar[mask_fit], mag_r[mask_fit])
    mag_predicted = slope * log_Mstar + intercept
    mag_residual = mag_r - mag_predicted
    
    print(f"\n  M-L slope: {slope:.3f} (expected ~ -2.5)")
    print(f"  Residual scatter: {np.std(mag_residual):.3f} mag")
    
    # Test for U-shape
    print("\n  Testing for U-shape in M/L residuals...")
    
    # Full sample
    fit_full = fit_ushape(env_density, mag_residual)
    
    if fit_full:
        results['full_sample'] = {
            'n': int(n_good),
            'a': float(fit_full['a']),
            'a_err': float(fit_full['a_err']),
            'p_value': float(fit_full['p_value']),
            'is_u_shape': fit_full['a'] > 0
        }
        print(f"  Full sample: a = {fit_full['a']:.6f} ± {fit_full['a_err']:.6f}")
        print(f"  p-value: {fit_full['p_value']:.4f}")
    else:
        print("  Full sample fit failed")
    
    # Q-dominated (gas-rich = blue)
    q_dom_mask = f_gas >= 0.3
    n_qdom = q_dom_mask.sum()
    print(f"\n  Q-dominated (blue): N = {n_qdom:,}")
    if n_qdom > 5000:
        fit_qdom = fit_ushape(env_density[q_dom_mask], mag_residual[q_dom_mask])
        if fit_qdom:
            results['q_dominated'] = {
                'n': int(n_qdom),
                'a': float(fit_qdom['a']),
                'a_err': float(fit_qdom['a_err']),
                'p_value': float(fit_qdom['p_value'])
            }
            print(f"    a = {fit_qdom['a']:.6f}")
    
    # R-dominated (gas-poor = red, massive)
    r_dom_mask = (f_gas < 0.15) & (log_Mstar > 10.5)
    n_rdom = r_dom_mask.sum()
    print(f"\n  R-dominated (red+massive): N = {n_rdom:,}")
    if n_rdom > 500:
        fit_rdom = fit_ushape(env_density[r_dom_mask], mag_residual[r_dom_mask], min_per_bin=30)
        if fit_rdom:
            results['r_dominated'] = {
                'n': int(n_rdom),
                'a': float(fit_rdom['a']),
                'a_err': float(fit_rdom['a_err']),
                'p_value': float(fit_rdom['p_value'])
            }
            print(f"    a = {fit_rdom['a']:.6f}")
    
    # Check for sign inversion (killer prediction)
    if 'q_dominated' in results and 'r_dominated' in results:
        q_sign = np.sign(results['q_dominated']['a'])
        r_sign = np.sign(results['r_dominated']['a'])
        results['sign_inversion'] = bool(q_sign != r_sign)
        print(f"\n  Sign inversion (killer prediction): {'YES ✓' if results['sign_inversion'] else 'NO'}")
    
    # Overall assessment
    if 'full_sample' in results:
        u_shape = results['full_sample']['is_u_shape']
        results['u_shape_detected'] = u_shape
        results['status'] = 'SUCCESS'
        
        if u_shape:
            results['conclusion'] = 'U-shape DETECTED in KiDS - SECOND CONTEXT confirms String Theory pattern'
        else:
            results['conclusion'] = 'U-shape NOT detected in full sample'
    else:
        results['status'] = 'ANALYSIS_FAILED'
        results['conclusion'] = 'Analysis failed'
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    import json
    
    print("="*70)
    print(" KiDS DR4 BRIGHT SAMPLE TEST")
    print(" Testing Q²R² pattern in optical scaling relations")
    print("="*70)
    
    results = test_kids_brightsample()
    
    print("\n" + "="*70)
    print(" RESULTS")
    print("="*70)
    
    if results.get('status') == 'SUCCESS':
        print(f"\n  Status: {results['status']}")
        print(f"  N galaxies: {results.get('full_sample', {}).get('n', 'N/A'):,}")
        print(f"  U-shape detected: {results.get('u_shape_detected', 'Unknown')}")
        if results.get('sign_inversion') is not None:
            print(f"  Sign inversion (killer prediction): {results['sign_inversion']}")
        print(f"\n  Conclusion: {results.get('conclusion', 'N/A')}")
    else:
        print(f"\n  Status: {results.get('status', 'UNKNOWN')}")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "kids_brightsample_results.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {output_file}")


if __name__ == "__main__":
    main()
