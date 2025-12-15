#!/usr/bin/env python3
"""
===============================================================================
KiDS DR4 BRIGHT SAMPLE TEST - CORRECTED VERSION
===============================================================================

Test for Q²R² pattern in the KiDS DR4 bright galaxy sample.

CORRECTIONS from v1:
1. Use ABSOLUTE magnitude (computed from apparent mag + photo-z)
2. Properly extract g-r color from LePhare absolute mags
3. Fix the Mass-Luminosity relation definition

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
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import warnings
warnings.filterwarnings('ignore')

# Cosmology
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "data", "level2_multicontext", "kids")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experimental", "04_RESULTS", "test_results")

KIDS_PHOTOZ = "KiDS_DR4_brightsample.fits"
KIDS_LEPHARE = "KiDS_DR4_brightsample_LePhare.fits"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_absolute_magnitude(mag_app, z):
    """Compute absolute magnitude from apparent magnitude and redshift."""
    z_safe = np.clip(z, 0.001, 5.0)
    d_L = COSMO.luminosity_distance(z_safe).to(u.pc).value
    M = mag_app - 5 * np.log10(d_L / 10)
    return M


def estimate_gas_fraction_from_color(g_r):
    """
    Estimate gas fraction from rest-frame g-r color.
    Blue → gas-rich (high Q), Red → gas-poor (high R)
    
    Based on Kannappan 2004, Catinella+ 2010 calibrations.
    """
    # Sigmoid mapping: blue (g-r ~ 0.3) → f_gas ~ 0.5
    #                  red (g-r ~ 0.8) → f_gas ~ 0.1
    f_gas = 0.55 / (1 + np.exp(8 * (g_r - 0.55)))
    f_gas += 0.05  # Minimum floor
    return np.clip(f_gas, 0.02, 0.70)


def compute_local_density(ra, dec, n_bins=100):
    """Grid-based local density estimation."""
    ra_bins = np.linspace(ra.min(), ra.max(), n_bins)
    dec_bins = np.linspace(dec.min(), dec.max(), n_bins)
    
    counts, _, _ = np.histogram2d(ra, dec, bins=[ra_bins, dec_bins])
    
    ra_idx = np.clip(np.digitize(ra, ra_bins) - 1, 0, n_bins - 2)
    dec_idx = np.clip(np.digitize(dec, dec_bins) - 1, 0, n_bins - 2)
    
    density = counts[ra_idx, dec_idx]
    density_std = (density - np.mean(density)) / np.std(density)
    
    return density_std


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
    
    bin_centers, bin_means, bin_errors = [], [], []
    
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (env >= bin_edges[i]) & (env <= bin_edges[i+1])
        else:
            mask = (env >= bin_edges[i]) & (env < bin_edges[i+1])
        
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
            'b': popt[1],
            'c': popt[2],
            'p_value': p_value,
            'n_bins': len(x),
            'bin_centers': x.tolist(),
            'bin_means': y.tolist()
        }
    except:
        return None


# =============================================================================
# MAIN TEST
# =============================================================================

def test_kids_brightsample():
    """Test for U-shape in KiDS bright sample - CORRECTED VERSION."""
    
    results = {
        'test_name': 'KiDS DR4 Bright Sample (Corrected)',
        'description': 'Test for Q²R² pattern in optical M/L relation',
        'context': 'Level 2 - Multi-context validation',
        'version': '2.0'
    }
    
    # Check files
    photoz_path = os.path.join(DATA_DIR, KIDS_PHOTOZ)
    lephare_path = os.path.join(DATA_DIR, KIDS_LEPHARE)
    
    if not os.path.exists(photoz_path):
        print(f"\n  ERROR: Photo-z catalog not found: {photoz_path}")
        results['status'] = 'NO_DATA'
        return results
    
    # Load photo-z catalog
    print(f"\n  Loading: {KIDS_PHOTOZ}")
    with fits.open(photoz_path) as hdul:
        photoz_data = hdul[1].data
    
    n_total = len(photoz_data)
    print(f"  Total galaxies: {n_total:,}")
    
    # Load LePhare catalog
    has_lephare = os.path.exists(lephare_path)
    if has_lephare:
        print(f"  Loading: {KIDS_LEPHARE}")
        with fits.open(lephare_path) as hdul:
            lephare_data = hdul[1].data
        print(f"  LePhare columns available: {len(lephare_data.names)}")
    
    # Extract basic data
    print("\n  Extracting data...")
    
    ra = photoz_data['RAJ2000']
    dec = photoz_data['DECJ2000']
    z_phot = photoz_data['zphot_ANNz2']
    mag_r_app = photoz_data['MAG_AUTO_calib']  # Apparent r-band
    masked = photoz_data['masked']
    
    # Get stellar masses and colors from LePhare
    if has_lephare:
        log_Mstar = lephare_data['MASS_MED']  # log10(M*/M_sun)
        
        # Get ABSOLUTE magnitudes for color (these are K-corrected!)
        if 'MAG_ABS_g' in lephare_data.names and 'MAG_ABS_r' in lephare_data.names:
            M_g = lephare_data['MAG_ABS_g']
            M_r = lephare_data['MAG_ABS_r']
            g_r_color = M_g - M_r
            print("  Using LePhare absolute magnitudes for g-r color")
        else:
            g_r_color = np.full(n_total, 0.6)
            print("  WARNING: No absolute mag columns, using default color")
    else:
        # Estimate from apparent mag
        M_r = compute_absolute_magnitude(mag_r_app, np.clip(z_phot, 0.01, 1.0))
        L_r = 10**(-0.4 * (M_r - 4.65))
        log_Mstar = np.log10(2.0 * L_r)
        g_r_color = np.full(n_total, 0.6)
    
    # Compute absolute r magnitude for M-L relation
    print("  Computing absolute magnitudes...")
    M_r_abs = compute_absolute_magnitude(mag_r_app, np.clip(z_phot, 0.01, 1.0))
    
    # Quality cuts
    print("\n  Applying quality cuts...")
    mask = (
        (masked == 0) &
        np.isfinite(ra) & np.isfinite(dec) &
        np.isfinite(z_phot) & (z_phot > 0.02) & (z_phot < 0.4) &
        np.isfinite(mag_r_app) & (mag_r_app > 14) & (mag_r_app < 20) &
        np.isfinite(M_r_abs) & (M_r_abs > -24) & (M_r_abs < -16) &
        np.isfinite(log_Mstar) & (log_Mstar > 7) & (log_Mstar < 12) &
        np.isfinite(g_r_color) & (g_r_color > -0.5) & (g_r_color < 1.5)
    )
    
    # Exclude LePhare failed fits (MASS_MED = -99)
    if has_lephare:
        mask &= (log_Mstar > 0)
    
    n_good = mask.sum()
    print(f"  After cuts: {n_good:,} ({100*n_good/n_total:.1f}%)")
    
    if n_good < 5000:
        print("  ERROR: Too few galaxies after cuts")
        results['status'] = 'INSUFFICIENT_DATA'
        return results
    
    # Apply mask
    ra = ra[mask]
    dec = dec[mask]
    z_phot = z_phot[mask]
    M_r_abs = M_r_abs[mask]
    log_Mstar = log_Mstar[mask]
    g_r_color = g_r_color[mask]
    
    # Compute gas fraction from color
    print("\n  Computing derived quantities...")
    f_gas = estimate_gas_fraction_from_color(g_r_color)
    print(f"  Gas fraction range: {f_gas.min():.2f} - {f_gas.max():.2f}")
    print(f"  g-r color range: {np.percentile(g_r_color, 5):.2f} - {np.percentile(g_r_color, 95):.2f}")
    
    # Environment
    print("  Computing local density...")
    env_density = compute_local_density(ra, dec)
    
    # Mass-Luminosity relation: M* vs M_r (ABSOLUTE magnitude)
    # More massive galaxies should be MORE luminous (more negative M_r)
    # Expected slope: positive (as log_Mstar increases, M_r becomes more negative... wait)
    # Actually: M_r = a * log_Mstar + b, with a ~ -2.5 (Faber-Jackson-like)
    # OR: log_Mstar = a * M_r + b with a ~ -0.4
    
    print("\n  Fitting Mass-Luminosity relation...")
    # Let's fit log_Mstar vs M_r (more intuitive)
    mask_fit = np.isfinite(log_Mstar) & np.isfinite(M_r_abs)
    
    # M_r is negative, more luminous = more negative
    # log_Mstar is positive, more massive = more positive
    # So correlation should be NEGATIVE: brighter (more negative M_r) = more massive
    
    slope, intercept, r_val, p_val, _ = stats.linregress(M_r_abs[mask_fit], log_Mstar[mask_fit])
    
    print(f"  Relation: log(M*) = {slope:.3f} * M_r + {intercept:.3f}")
    print(f"  Correlation: r = {r_val:.3f} (expected negative)")
    print(f"  Slope: {slope:.3f} (expected ~ -0.4)")
    
    # Compute residuals from the relation
    log_Mstar_predicted = slope * M_r_abs + intercept
    mass_residual = log_Mstar - log_Mstar_predicted  # Excess mass for given luminosity
    
    print(f"  Residual scatter: {np.std(mass_residual):.3f} dex")
    
    results['ml_relation'] = {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_correlation': float(r_val),
        'scatter': float(np.std(mass_residual))
    }
    
    # Check if relation makes sense
    if slope > 0:
        print("\n  ⚠️ WARNING: Positive slope indicates potential issue with data")
        print("  (Brighter galaxies should be more massive → negative slope)")
    
    # Test for U-shape in M/L residuals vs environment
    print("\n  Testing for U-shape in M/L residuals...")
    
    # Full sample
    fit_full = fit_ushape(env_density, mass_residual)
    
    if fit_full:
        results['full_sample'] = {
            'n': int(n_good),
            'a': float(fit_full['a']),
            'a_err': float(fit_full['a_err']),
            'p_value': float(fit_full['p_value']),
            'is_u_shape': fit_full['a'] > 0 and fit_full['a'] > 2 * fit_full['a_err']
        }
        u_detected = results['full_sample']['is_u_shape']
        print(f"  Full sample: a = {fit_full['a']:.6f} ± {fit_full['a_err']:.6f}")
        print(f"  U-shape detected: {'YES ✓' if u_detected else 'NO'}")
        print(f"  p-value: {fit_full['p_value']:.4f}")
    else:
        print("  Full sample fit failed")
        u_detected = False
    
    # Q-dominated (blue, gas-rich)
    q_dom_mask = (f_gas >= 0.25) | (g_r_color < 0.5)
    n_qdom = q_dom_mask.sum()
    print(f"\n  Q-dominated (blue/gas-rich): N = {n_qdom:,}")
    
    if n_qdom > 5000:
        fit_qdom = fit_ushape(env_density[q_dom_mask], mass_residual[q_dom_mask])
        if fit_qdom:
            results['q_dominated'] = {
                'n': int(n_qdom),
                'a': float(fit_qdom['a']),
                'a_err': float(fit_qdom['a_err']),
                'p_value': float(fit_qdom['p_value'])
            }
            print(f"    a = {fit_qdom['a']:.6f} ± {fit_qdom['a_err']:.6f}")
    
    # R-dominated (red, massive, gas-poor)
    r_dom_mask = (f_gas < 0.15) & (g_r_color > 0.7) & (log_Mstar > 10.5)
    n_rdom = r_dom_mask.sum()
    print(f"\n  R-dominated (red/massive): N = {n_rdom:,}")
    
    if n_rdom > 1000:
        fit_rdom = fit_ushape(env_density[r_dom_mask], mass_residual[r_dom_mask], min_per_bin=30)
        if fit_rdom:
            results['r_dominated'] = {
                'n': int(n_rdom),
                'a': float(fit_rdom['a']),
                'a_err': float(fit_rdom['a_err']),
                'p_value': float(fit_rdom['p_value'])
            }
            print(f"    a = {fit_rdom['a']:.6f} ± {fit_rdom['a_err']:.6f}")
    
    # Check sign inversion
    if 'q_dominated' in results and 'r_dominated' in results:
        q_sign = np.sign(results['q_dominated']['a'])
        r_sign = np.sign(results['r_dominated']['a'])
        results['sign_inversion'] = bool(q_sign != r_sign)
        print(f"\n  Sign inversion (killer prediction): {'YES ✓' if results['sign_inversion'] else 'NO'}")
    
    # Summary
    results['u_shape_detected'] = u_detected
    results['status'] = 'SUCCESS'
    
    if u_detected and results.get('sign_inversion', False):
        results['conclusion'] = 'U-shape WITH sign inversion detected - Strong support for String Theory'
    elif u_detected:
        results['conclusion'] = 'U-shape detected (no clear sign inversion) - Partial support'
    else:
        results['conclusion'] = 'U-shape NOT detected in optical M/L relation'
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print(" KiDS DR4 BRIGHT SAMPLE TEST (CORRECTED)")
    print(" Testing Q²R² pattern in optical M/L relation")
    print("="*70)
    
    results = test_kids_brightsample()
    
    print("\n" + "="*70)
    print(" RESULTS")
    print("="*70)
    
    if results.get('status') == 'SUCCESS':
        print(f"\n  Status: {results['status']}")
        print(f"  N galaxies: {results.get('full_sample', {}).get('n', 'N/A'):,}")
        
        ml = results.get('ml_relation', {})
        print(f"\n  M-L relation slope: {ml.get('slope', 'N/A'):.3f} (expected ~ -0.4)")
        print(f"  M-L correlation: {ml.get('r_correlation', 'N/A'):.3f}")
        
        print(f"\n  U-shape detected: {results.get('u_shape_detected', 'Unknown')}")
        if results.get('sign_inversion') is not None:
            print(f"  Sign inversion: {results['sign_inversion']}")
        
        print(f"\n  Conclusion: {results.get('conclusion', 'N/A')}")
    else:
        print(f"\n  Status: {results.get('status', 'UNKNOWN')}")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "kids_brightsample_results_v2.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {output_file}")


if __name__ == "__main__":
    main()
