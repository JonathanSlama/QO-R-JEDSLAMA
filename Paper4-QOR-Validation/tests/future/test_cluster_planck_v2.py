#!/usr/bin/env python3
"""
===============================================================================
PLANCK PSZ2 GALAXY CLUSTERS TEST - VERSION 2 (CORRECTED)
===============================================================================

Test for Q²R² pattern in Planck PSZ2 galaxy clusters.

CORRECTIONS from v1:
1. USE MSZ column directly (it EXISTS in the union catalog!)
2. Proper M-Y relation with correct variables
3. Better quality cuts

Data: HFI_PCCS_SZ-union_R2.08.fits.gz
Columns: INDEX, NAME, GLON, GLAT, RA, DEC, POS_ERR, SNR, PIPELINE, PIPE_DET,
         PCCS2, PSZ, IR_FLAG, Q_NEURAL, Y5R500, Y5R500_ERR, VALIDATION,
         REDSHIFT_ID, REDSHIFT, MSZ, MSZ_ERR_UP, MSZ_ERR_LOW, MCXC, REDMAPPER,
         ACT, SPT, WISE_FLAG, AMI_EVIDENCE, COSMO, COMMENT

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
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
import astropy.units as u
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


def find_planck_file():
    """Find the Planck PSZ2 union catalog."""
    patterns = [
        os.path.join(DATA_DIR, "**", "*SZ-union*.fits*"),
        os.path.join(DATA_DIR, "*union*.fits*"),
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    
    return None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def estimate_gas_fraction(Y_500, M_500, z):
    """
    Estimate gas fraction from Y_SZ and mass.
    
    Y_SZ ∝ f_gas * M^(5/3) * E(z)^(2/3)
    So: f_gas ∝ Y_SZ / (M^(5/3) * E(z)^(2/3))
    
    Returns normalized gas fraction (typical ~0.12 for clusters).
    """
    E_z = COSMO.efunc(z)
    
    # Y / M^(5/3) ratio
    ratio = Y_500 / (M_500**(5/3) * E_z**(2/3))
    
    # Normalize to typical gas fraction
    ratio_median = np.nanmedian(ratio)
    if ratio_median > 0:
        f_gas = 0.12 * (ratio / ratio_median)
    else:
        f_gas = np.full_like(ratio, 0.12)
    
    return np.clip(f_gas, 0.05, 0.20)


def compute_cluster_environment(ra, dec, z, max_sep_mpc=50, dz_max=0.05):
    """
    Compute cluster environment from neighbor count.
    Counts neighbors within physical distance and redshift slice.
    """
    n_clusters = len(ra)
    env = np.zeros(n_clusters)
    
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    
    print("    Computing environment for", n_clusters, "clusters...")
    
    for i in range(n_clusters):
        if not np.isfinite(z[i]) or z[i] <= 0:
            continue
            
        # Angular diameter distance
        d_A = COSMO.angular_diameter_distance(z[i]).to(u.Mpc).value
        
        # Max angular separation for given physical distance
        max_sep_deg = np.degrees(max_sep_mpc / d_A) if d_A > 0 else 10
        
        # Find neighbors
        sep = coords[i].separation(coords).deg
        dz = np.abs(z - z[i])
        
        neighbors = (sep > 0.1) & (sep < max_sep_deg) & (dz < dz_max)
        env[i] = neighbors.sum()
    
    # Standardize
    if np.std(env) > 0:
        env = (env - np.mean(env)) / np.std(env)
    
    return env


def fit_ushape(env, residual, n_bins=8, min_per_bin=15):
    """Fit quadratic U-shape to binned data."""
    mask = np.isfinite(env) & np.isfinite(residual)
    env = env[mask]
    residual = residual[mask]
    
    n_data = len(env)
    if n_data < 80:
        n_bins = max(4, n_data // 20)
        min_per_bin = max(5, n_data // 20)
    
    if n_data < 50:
        print(f"    Too few data points ({n_data}) for U-shape fit")
        return None
    
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(env, percentiles)
    
    bin_centers, bin_means, bin_errors = [], [], []
    
    for i in range(n_bins):
        if i == n_bins - 1:
            bmask = (env >= bin_edges[i]) & (env <= bin_edges[i+1])
        else:
            bmask = (env >= bin_edges[i]) & (env < bin_edges[i+1])
        
        if bmask.sum() >= min_per_bin:
            bin_centers.append(np.mean(env[bmask]))
            bin_means.append(np.mean(residual[bmask]))
            bin_errors.append(np.std(residual[bmask]) / np.sqrt(bmask.sum()))
    
    if len(bin_centers) < 4:
        print(f"    Too few bins ({len(bin_centers)}) for quadratic fit")
        return None
    
    x = np.array(bin_centers)
    y = np.array(bin_means)
    yerr = np.array(bin_errors)
    yerr[yerr == 0] = 0.01
    
    try:
        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c
        
        popt, pcov = curve_fit(quadratic, x, y, sigma=yerr, absolute_sigma=True, 
                               p0=[0.01, 0, 0], maxfev=5000)
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
            'n_data': n_data,
            'bin_centers': x.tolist(),
            'bin_means': y.tolist()
        }
    except Exception as e:
        print(f"    Fit error: {e}")
        return None


# =============================================================================
# MAIN TEST
# =============================================================================

def test_planck_clusters():
    """Test for U-shape in Planck PSZ2 clusters using MSZ column."""
    
    results = {
        'test_name': 'Planck PSZ2 Galaxy Clusters v2',
        'description': 'Test for Q²R² pattern in cluster M-Y relation',
        'context': 'Level 2 - Multi-context validation',
        'version': '2.0 - Using MSZ column directly'
    }
    
    # Find data file
    planck_file = find_planck_file()
    
    if not planck_file:
        print(f"\n  ERROR: Planck PSZ2 catalog not found in {DATA_DIR}")
        results['status'] = 'NO_DATA'
        return results
    
    print(f"\n  Loading: {os.path.basename(planck_file)}")
    
    # Load catalog
    with fits.open(planck_file) as hdul:
        data = hdul[1].data
    
    n_total = len(data)
    print(f"  Total clusters: {n_total}")
    
    columns = list(data.names)
    print(f"  Columns: {columns}")
    
    # Extract data
    print("\n  Extracting data...")
    
    ra = data['RA']
    dec = data['DEC']
    z = data['REDSHIFT']
    Y_500 = data['Y5R500']  # arcmin^2
    snr = data['SNR']
    
    # THE KEY: MSZ exists!
    M_500 = data['MSZ']  # in 10^14 M_sun
    M_500_err_up = data['MSZ_ERR_UP']
    M_500_err_low = data['MSZ_ERR_LOW']
    
    print(f"  MSZ range: {np.nanmin(M_500):.2f} - {np.nanmax(M_500):.2f} × 10^14 M_sun")
    print(f"  Y5R500 range: {np.nanmin(Y_500):.4f} - {np.nanmax(Y_500):.4f} arcmin^2")
    print(f"  Redshift range: {np.nanmin(z):.3f} - {np.nanmax(z):.3f}")
    
    # Quality cuts
    print("\n  Applying quality cuts...")
    
    # Count valid entries
    n_valid_z = np.sum(np.isfinite(z) & (z > 0))
    n_valid_M = np.sum(np.isfinite(M_500) & (M_500 > 0))
    n_valid_Y = np.sum(np.isfinite(Y_500) & (Y_500 > 0))
    
    print(f"    Valid redshifts: {n_valid_z}")
    print(f"    Valid masses: {n_valid_M}")
    print(f"    Valid Y_500: {n_valid_Y}")
    
    # IMPORTANT: MSZ=0 and REDSHIFT=-1 are flags for missing data!
    mask = (
        np.isfinite(ra) & np.isfinite(dec) &
        np.isfinite(z) & (z > 0.01) & (z < 1.0) &  # Exclude z=-1 flags
        np.isfinite(Y_500) & (Y_500 > 0) &
        np.isfinite(M_500) & (M_500 > 0.1) &  # Exclude MSZ=0 flags, min 10^13 M_sun
        (snr > 4.5)
    )
    
    n_good = mask.sum()
    print(f"  After cuts: {n_good} ({100*n_good/n_total:.1f}%)")
    
    if n_good < 50:
        print("  ERROR: Too few clusters after cuts")
        results['status'] = 'INSUFFICIENT_DATA'
        results['n_after_cuts'] = int(n_good)
        return results
    
    # Apply mask
    ra = ra[mask]
    dec = dec[mask]
    z = z[mask]
    Y_500 = Y_500[mask]
    M_500 = M_500[mask]
    snr = snr[mask]
    
    # Compute derived quantities
    print("\n  Computing derived quantities...")
    
    # Gas fraction proxy
    f_gas = estimate_gas_fraction(Y_500, M_500, z)
    print(f"  Gas fraction range: {f_gas.min():.3f} - {f_gas.max():.3f}")
    
    # Environment
    print("  Computing cluster environment...")
    env = compute_cluster_environment(ra, dec, z)
    
    # M-Y relation: log(Y_500) vs log(M_500)
    # Expected: slope ~ 5/3 ≈ 1.67 (self-similar)
    print("\n  Fitting M-Y scaling relation...")
    
    log_Y = np.log10(Y_500)
    log_M = np.log10(M_500)
    
    mask_fit = np.isfinite(log_Y) & np.isfinite(log_M)
    slope, intercept, r_val, p_val, std_err = stats.linregress(log_M[mask_fit], log_Y[mask_fit])
    
    print(f"  Relation: log(Y) = {slope:.3f} × log(M) + {intercept:.3f}")
    print(f"  Slope: {slope:.3f} (expected ~ 1.67)")
    print(f"  Correlation: r = {r_val:.3f}")
    print(f"  p-value: {p_val:.2e}")
    
    results['my_relation'] = {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_correlation': float(r_val),
        'p_value': float(p_val),
        'expected_slope': 1.67
    }
    
    # Check if relation makes physical sense
    if slope < 0:
        print("\n  ⚠️ WARNING: Negative slope is unphysical!")
        print("     This suggests a data or methodology issue.")
    elif slope < 1.0:
        print("\n  ⚠️ WARNING: Slope lower than expected (mass-dependent f_gas?)")
    elif slope > 2.5:
        print("\n  ⚠️ WARNING: Slope higher than expected")
    else:
        print("\n  ✓ Slope is in reasonable range")
    
    # Compute residuals
    log_Y_predicted = slope * log_M + intercept
    Y_residual = log_Y - log_Y_predicted  # Excess SZ signal for given mass
    
    print(f"  Residual scatter: {np.std(Y_residual):.3f} dex")
    results['residual_scatter'] = float(np.std(Y_residual))
    
    # Test for U-shape in Y residuals vs environment
    print("\n  Testing for U-shape in M-Y residuals vs environment...")
    
    # Full sample
    fit_full = fit_ushape(env, Y_residual)
    
    if fit_full:
        u_detected = fit_full['a'] > 0 and fit_full['a'] > 1.5 * fit_full['a_err']
        results['full_sample'] = {
            'n': int(n_good),
            'a': float(fit_full['a']),
            'a_err': float(fit_full['a_err']),
            'p_value': float(fit_full['p_value']),
            'is_u_shape': u_detected
        }
        print(f"  Full sample: a = {fit_full['a']:.6f} ± {fit_full['a_err']:.6f}")
        print(f"  U-shape detected: {'YES ✓' if u_detected else 'NO'}")
    else:
        print("  Full sample fit failed")
        u_detected = False
    
    # Q-dominated (gas-rich clusters)
    q_dom_mask = f_gas >= np.percentile(f_gas, 60)
    n_qdom = q_dom_mask.sum()
    print(f"\n  Q-dominated (gas-rich): N = {n_qdom}")
    
    if n_qdom > 50:
        fit_qdom = fit_ushape(env[q_dom_mask], Y_residual[q_dom_mask], n_bins=6, min_per_bin=8)
        if fit_qdom:
            results['q_dominated'] = {
                'n': int(n_qdom),
                'a': float(fit_qdom['a']),
                'a_err': float(fit_qdom['a_err']),
                'p_value': float(fit_qdom['p_value'])
            }
            print(f"    a = {fit_qdom['a']:.6f} ± {fit_qdom['a_err']:.6f}")
    
    # R-dominated (massive, gas-poor)
    r_dom_mask = (f_gas < np.percentile(f_gas, 40)) & (M_500 > np.percentile(M_500, 50))
    n_rdom = r_dom_mask.sum()
    print(f"\n  R-dominated (massive, gas-poor): N = {n_rdom}")
    
    if n_rdom > 50:
        fit_rdom = fit_ushape(env[r_dom_mask], Y_residual[r_dom_mask], n_bins=6, min_per_bin=8)
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
    results['u_shape_detected'] = u_detected if fit_full else False
    results['status'] = 'SUCCESS'
    results['n_clusters'] = int(n_good)
    
    if results['u_shape_detected'] and results.get('sign_inversion', False):
        results['conclusion'] = 'U-shape WITH sign inversion - Strong support for String Theory'
    elif results['u_shape_detected']:
        results['conclusion'] = 'U-shape detected (no clear sign inversion) - Partial support'
    else:
        results['conclusion'] = 'U-shape NOT detected in cluster M-Y residuals'
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print(" PLANCK PSZ2 GALAXY CLUSTERS TEST v2")
    print(" Testing Q²R² pattern in M-Y scaling relation")
    print(" Using MSZ column directly")
    print("="*70)
    
    results = test_planck_clusters()
    
    print("\n" + "="*70)
    print(" RESULTS")
    print("="*70)
    
    if results.get('status') == 'SUCCESS':
        print(f"\n  Status: {results['status']}")
        print(f"  N clusters: {results.get('n_clusters', 'N/A')}")
        
        my = results.get('my_relation', {})
        print(f"\n  M-Y relation:")
        print(f"    Slope: {my.get('slope', 'N/A'):.3f} (expected ~ 1.67)")
        print(f"    Correlation: r = {my.get('r_correlation', 'N/A'):.3f}")
        
        print(f"\n  U-shape detected: {results.get('u_shape_detected', 'Unknown')}")
        if results.get('sign_inversion') is not None:
            print(f"  Sign inversion: {results['sign_inversion']}")
        
        print(f"\n  Conclusion: {results.get('conclusion', 'N/A')}")
    else:
        print(f"\n  Status: {results.get('status', 'UNKNOWN')}")
        if 'n_after_cuts' in results:
            print(f"  Clusters after cuts: {results['n_after_cuts']}")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "planck_clusters_results_v2.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {output_file}")


if __name__ == "__main__":
    main()
