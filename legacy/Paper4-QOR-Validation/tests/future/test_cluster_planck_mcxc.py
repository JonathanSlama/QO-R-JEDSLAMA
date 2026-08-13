#!/usr/bin/env python3
"""
===============================================================================
PLANCK PSZ2 × MCXC CROSS-MATCH TEST
===============================================================================

Test for Q²R² pattern using:
- Y_SZ from Planck PSZ2 (SZ signal)
- M_X and L_X from MCXC (X-ray, INDEPENDENT!)

This solves the problem that PSZ2 MSZ was not correlating with Y5R500.
MCXC masses are derived from X-ray luminosity, completely independent of SZ.

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
from astropy.coordinates import SkyCoord
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
                        "data", "level2_multicontext", "clusters")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experimental", "04_RESULTS", "test_results")


def find_psz2_file():
    """Find PSZ2 union catalog."""
    patterns = [
        os.path.join(DATA_DIR, "**", "*SZ-union*.fits*"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return None


def find_mcxc_file():
    """Find MCXC catalog."""
    patterns = [
        os.path.join(DATA_DIR, "MCXC*.fits"),
        os.path.join(DATA_DIR, "mcxc*.fits"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


# =============================================================================
# CROSS-MATCH USING PSZ2's MCXC COLUMN
# =============================================================================

def load_and_crossmatch():
    """
    Load PSZ2 and MCXC, cross-match using PSZ2's MCXC column.
    """
    
    # Find files
    psz2_path = find_psz2_file()
    mcxc_path = find_mcxc_file()
    
    if not psz2_path:
        print("  ERROR: PSZ2 file not found")
        return None, None
    
    print(f"  Loading PSZ2: {os.path.basename(psz2_path)}")
    with fits.open(psz2_path) as hdul:
        psz2 = Table(hdul[1].data)
    print(f"  PSZ2 clusters: {len(psz2)}")
    
    # Check if MCXC column exists in PSZ2
    if 'MCXC' not in psz2.colnames:
        print("  WARNING: No MCXC column in PSZ2")
        print("  Will need to do positional cross-match")
        
        if not mcxc_path:
            print("  ERROR: MCXC file not found. Run download_mcxc.py first!")
            return None, None
        
        print(f"\n  Loading MCXC: {os.path.basename(mcxc_path)}")
        mcxc = Table.read(mcxc_path)
        print(f"  MCXC clusters: {len(mcxc)}")
        
        # Positional cross-match
        return crossmatch_positional(psz2, mcxc)
    
    # Use PSZ2's MCXC column for matching
    print("  Using PSZ2's built-in MCXC cross-match")
    
    # Count how many have MCXC matches
    mcxc_col = psz2['MCXC']
    
    # Handle string column
    if mcxc_col.dtype.kind in ['S', 'U', 'O']:
        has_mcxc = np.array([len(str(x).strip()) > 0 and str(x).strip() not in ['', 'b\'\'', "b''"] 
                            for x in mcxc_col])
    else:
        has_mcxc = ~np.isnan(mcxc_col)
    
    n_match = has_mcxc.sum()
    print(f"  PSZ2 clusters with MCXC match: {n_match}")
    
    if not mcxc_path:
        print("\n  WARNING: MCXC catalog not found locally")
        print("  Cannot get X-ray masses. Run download_mcxc.py first!")
        return None, None
    
    print(f"\n  Loading MCXC: {os.path.basename(mcxc_path)}")
    mcxc = Table.read(mcxc_path)
    print(f"  MCXC clusters: {len(mcxc)}")
    print(f"  MCXC columns: {mcxc.colnames}")
    
    # Build MCXC lookup by name
    mcxc_lookup = {}
    name_col = None
    for col in ['MCXC', 'Name', 'name', 'NAME']:
        if col in mcxc.colnames:
            name_col = col
            break
    
    if name_col is None:
        print("  ERROR: Cannot find name column in MCXC")
        return crossmatch_positional(psz2, mcxc)
    
    for i, row in enumerate(mcxc):
        name = str(row[name_col]).strip()
        mcxc_lookup[name] = i
    
    # Match PSZ2 to MCXC
    matched_psz2 = []
    matched_mcxc = []
    
    for i, row in enumerate(psz2):
        mcxc_name = str(row['MCXC']).strip()
        # Clean up the name
        mcxc_name = mcxc_name.replace("b'", "").replace("'", "").strip()
        
        if mcxc_name and mcxc_name in mcxc_lookup:
            matched_psz2.append(i)
            matched_mcxc.append(mcxc_lookup[mcxc_name])
    
    print(f"  Successfully matched: {len(matched_psz2)} clusters")
    
    if len(matched_psz2) < 50:
        print("  Too few matches, trying positional cross-match...")
        return crossmatch_positional(psz2, mcxc)
    
    return psz2[matched_psz2], mcxc[matched_mcxc]


def crossmatch_positional(psz2, mcxc, max_sep_arcmin=5.0):
    """Cross-match by position if name matching fails."""
    
    print(f"\n  Performing positional cross-match (max sep = {max_sep_arcmin} arcmin)...")
    
    # Get coordinates
    if 'RA' in psz2.colnames and 'DEC' in psz2.colnames:
        psz2_coords = SkyCoord(ra=psz2['RA']*u.deg, dec=psz2['DEC']*u.deg)
    elif 'GLON' in psz2.colnames and 'GLAT' in psz2.colnames:
        from astropy.coordinates import Galactic
        gal = Galactic(l=psz2['GLON']*u.deg, b=psz2['GLAT']*u.deg)
        psz2_coords = gal.icrs
    else:
        print("  ERROR: Cannot find PSZ2 coordinates")
        return None, None
    
    # MCXC coordinates
    ra_col = None
    dec_col = None
    for ra_name in ['RAJ2000', 'RA', 'ra', '_RAJ2000']:
        if ra_name in mcxc.colnames:
            ra_col = ra_name
            break
    for dec_name in ['DEJ2000', 'DEC', 'dec', '_DEJ2000']:
        if dec_name in mcxc.colnames:
            dec_col = dec_name
            break
    
    if ra_col is None or dec_col is None:
        print(f"  ERROR: Cannot find MCXC coordinates")
        print(f"  Available: {mcxc.colnames}")
        return None, None
    
    mcxc_coords = SkyCoord(ra=mcxc[ra_col]*u.deg, dec=mcxc[dec_col]*u.deg)
    
    # Cross-match
    idx, sep, _ = psz2_coords.match_to_catalog_sky(mcxc_coords)
    
    # Filter by separation
    good_match = sep.arcmin < max_sep_arcmin
    
    matched_psz2_idx = np.where(good_match)[0]
    matched_mcxc_idx = idx[good_match]
    
    print(f"  Matches found: {len(matched_psz2_idx)}")
    print(f"  Median separation: {np.median(sep[good_match].arcmin):.2f} arcmin")
    
    if len(matched_psz2_idx) < 50:
        print("  WARNING: Very few matches!")
        return None, None
    
    return psz2[matched_psz2_idx], mcxc[matched_mcxc_idx]


# =============================================================================
# U-SHAPE FIT
# =============================================================================

def fit_ushape(env, residual, n_bins=8, min_per_bin=10):
    """Fit quadratic U-shape to binned data."""
    mask = np.isfinite(env) & np.isfinite(residual)
    env = env[mask]
    residual = residual[mask]
    
    n_data = len(env)
    if n_data < 60:
        n_bins = max(4, n_data // 15)
        min_per_bin = max(5, n_data // 15)
    
    if n_data < 40:
        print(f"    Too few data points ({n_data})")
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
        print(f"    Too few bins ({len(bin_centers)})")
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
        
        return {
            'a': popt[0],
            'a_err': perr[0],
            'b': popt[1],
            'c': popt[2],
            'n_bins': len(x),
            'n_data': n_data
        }
    except Exception as e:
        print(f"    Fit error: {e}")
        return None


def compute_environment(ra, dec, z, max_sep_mpc=50, dz_max=0.05):
    """Compute cluster environment from neighbor count."""
    n = len(ra)
    env = np.zeros(n)
    
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    
    for i in range(n):
        if not np.isfinite(z[i]) or z[i] <= 0:
            continue
        
        d_A = COSMO.angular_diameter_distance(z[i]).to(u.Mpc).value
        max_sep_deg = np.degrees(max_sep_mpc / d_A) if d_A > 0 else 10
        
        sep = coords[i].separation(coords).deg
        dz = np.abs(z - z[i])
        
        neighbors = (sep > 0.1) & (sep < max_sep_deg) & (dz < dz_max)
        env[i] = neighbors.sum()
    
    if np.std(env) > 0:
        env = (env - np.mean(env)) / np.std(env)
    
    return env


# =============================================================================
# MAIN TEST
# =============================================================================

def test_psz2_mcxc():
    """Test Q²R² pattern using PSZ2 × MCXC cross-match."""
    
    results = {
        'test_name': 'Planck PSZ2 × MCXC Cross-match',
        'description': 'Test for Q²R² pattern using independent X-ray masses',
        'context': 'Level 2 - Multi-context validation',
        'version': '3.0 - Using MCXC X-ray masses'
    }
    
    # Load and cross-match
    psz2, mcxc = load_and_crossmatch()
    
    if psz2 is None or mcxc is None:
        results['status'] = 'NO_DATA'
        results['error'] = 'Cross-match failed'
        return results
    
    n_match = len(psz2)
    print(f"\n  Working with {n_match} matched clusters")
    
    # Extract data from PSZ2
    Y_SZ = np.array(psz2['Y5R500'])
    z_psz2 = np.array(psz2['REDSHIFT'])
    ra = np.array(psz2['RA']) if 'RA' in psz2.colnames else None
    dec = np.array(psz2['DEC']) if 'DEC' in psz2.colnames else None
    
    # Extract data from MCXC
    # Find mass and luminosity columns
    M_X = None
    L_X = None
    
    for col in mcxc.colnames:
        if 'mass' in col.lower() or col in ['M500', 'Mass_500']:
            M_X = np.array(mcxc[col])
            print(f"  Using {col} for X-ray mass")
        if 'lx' in col.lower() or col in ['Lx_500', 'L500']:
            L_X = np.array(mcxc[col])
            print(f"  Using {col} for X-ray luminosity")
    
    if M_X is None:
        print("  ERROR: Cannot find mass column in MCXC")
        print(f"  Available: {mcxc.colnames}")
        results['status'] = 'COLUMN_ERROR'
        return results
    
    z_mcxc = None
    for col in ['z', 'Z', 'redshift', 'Redshift']:
        if col in mcxc.colnames:
            z_mcxc = np.array(mcxc[col])
            break
    
    # Use PSZ2 redshift if MCXC doesn't have it
    z = z_psz2 if z_mcxc is None else z_mcxc
    
    # Quality cuts
    print("\n  Applying quality cuts...")
    mask = (
        np.isfinite(Y_SZ) & (Y_SZ > 0) &
        np.isfinite(M_X) & (M_X > 0) &
        np.isfinite(z) & (z > 0.01) & (z < 1.0)
    )
    
    if ra is not None:
        mask &= np.isfinite(ra) & np.isfinite(dec)
    
    n_good = mask.sum()
    print(f"  After cuts: {n_good} ({100*n_good/n_match:.1f}%)")
    
    if n_good < 50:
        print("  ERROR: Too few clusters after cuts")
        results['status'] = 'INSUFFICIENT_DATA'
        return results
    
    # Apply mask
    Y_SZ = Y_SZ[mask]
    M_X = M_X[mask]
    z = z[mask]
    if ra is not None:
        ra = ra[mask]
        dec = dec[mask]
    if L_X is not None:
        L_X = L_X[mask]
    
    print(f"\n  Data ranges:")
    print(f"  Y_SZ: {Y_SZ.min():.3f} - {Y_SZ.max():.3f} arcmin²")
    print(f"  M_X: {M_X.min():.2e} - {M_X.max():.2e}")
    print(f"  z: {z.min():.3f} - {z.max():.3f}")
    
    # Test L_X - Y_SZ relation (should be positive correlation!)
    print("\n  Testing L_X - Y_SZ relation...")
    
    if L_X is not None:
        log_LX = np.log10(L_X[L_X > 0])
        log_Y = np.log10(Y_SZ[L_X > 0])
        
        slope, intercept, r, p, err = stats.linregress(log_LX, log_Y)
        
        print(f"  log(Y) = {slope:.3f} × log(L_X) + {intercept:.3f}")
        print(f"  Correlation: r = {r:.3f}")
        print(f"  p-value: {p:.2e}")
        
        results['lx_y_relation'] = {
            'slope': float(slope),
            'r_correlation': float(r),
            'p_value': float(p)
        }
        
        if r > 0.3:
            print("  ✓ Positive correlation as expected!")
        else:
            print("  ⚠️ Weak or negative correlation")
    
    # Test M_X - Y_SZ relation
    print("\n  Testing M_X - Y_SZ relation...")
    
    log_MX = np.log10(M_X)
    log_Y = np.log10(Y_SZ)
    
    slope, intercept, r, p, err = stats.linregress(log_MX, log_Y)
    
    print(f"  log(Y) = {slope:.3f} × log(M_X) + {intercept:.3f}")
    print(f"  Correlation: r = {r:.3f} (expected positive)")
    print(f"  p-value: {p:.2e}")
    
    results['mx_y_relation'] = {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_correlation': float(r),
        'p_value': float(p),
        'expected_slope': 1.67
    }
    
    if r > 0.3:
        print("  ✓ Positive correlation - we can analyze residuals!")
    else:
        print("  ⚠️ Weak correlation - residual analysis may be noisy")
    
    # Compute residuals
    log_Y_pred = slope * log_MX + intercept
    Y_residual = log_Y - log_Y_pred
    
    results['residual_scatter'] = float(np.std(Y_residual))
    print(f"  Residual scatter: {np.std(Y_residual):.3f} dex")
    
    # Compute environment
    if ra is not None:
        print("\n  Computing cluster environment...")
        env = compute_environment(ra, dec, z)
    else:
        print("  WARNING: No coordinates for environment, using redshift as proxy")
        env = (z - np.mean(z)) / np.std(z)
    
    # Gas fraction proxy
    E_z = COSMO.efunc(z)
    f_gas = Y_SZ / (M_X**(5/3) * E_z**(2/3))
    f_gas = f_gas / np.median(f_gas) * 0.12  # Normalize
    
    print(f"  Gas fraction proxy range: {f_gas.min():.3f} - {f_gas.max():.3f}")
    
    # Test for U-shape
    print("\n  Testing for U-shape in M-Y residuals vs environment...")
    
    fit_full = fit_ushape(env, Y_residual)
    
    if fit_full:
        u_detected = fit_full['a'] > 0 and fit_full['a'] > 1.5 * fit_full['a_err']
        results['full_sample'] = {
            'n': int(n_good),
            'a': float(fit_full['a']),
            'a_err': float(fit_full['a_err']),
            'is_u_shape': u_detected
        }
        print(f"  Full sample: a = {fit_full['a']:.6f} ± {fit_full['a_err']:.6f}")
        print(f"  U-shape detected: {'YES ✓' if u_detected else 'NO'}")
    else:
        u_detected = False
        print("  Full sample fit failed")
    
    # Q-dominated (gas-rich)
    q_dom_mask = f_gas >= np.percentile(f_gas, 60)
    n_qdom = q_dom_mask.sum()
    print(f"\n  Q-dominated (gas-rich): N = {n_qdom}")
    
    if n_qdom > 30:
        fit_qdom = fit_ushape(env[q_dom_mask], Y_residual[q_dom_mask], n_bins=6, min_per_bin=5)
        if fit_qdom:
            results['q_dominated'] = {
                'n': int(n_qdom),
                'a': float(fit_qdom['a']),
                'a_err': float(fit_qdom['a_err'])
            }
            print(f"    a = {fit_qdom['a']:.6f} ± {fit_qdom['a_err']:.6f}")
    
    # R-dominated (massive, gas-poor)
    r_dom_mask = (f_gas < np.percentile(f_gas, 40)) & (M_X > np.percentile(M_X, 50))
    n_rdom = r_dom_mask.sum()
    print(f"\n  R-dominated (massive, gas-poor): N = {n_rdom}")
    
    if n_rdom > 30:
        fit_rdom = fit_ushape(env[r_dom_mask], Y_residual[r_dom_mask], n_bins=6, min_per_bin=5)
        if fit_rdom:
            results['r_dominated'] = {
                'n': int(n_rdom),
                'a': float(fit_rdom['a']),
                'a_err': float(fit_rdom['a_err'])
            }
            print(f"    a = {fit_rdom['a']:.6f} ± {fit_rdom['a_err']:.6f}")
    
    # Sign inversion check
    if 'q_dominated' in results and 'r_dominated' in results:
        q_sign = np.sign(results['q_dominated']['a'])
        r_sign = np.sign(results['r_dominated']['a'])
        results['sign_inversion'] = bool(q_sign != r_sign)
        print(f"\n  Sign inversion: {'YES ✓' if results['sign_inversion'] else 'NO'}")
    
    # Summary
    results['u_shape_detected'] = u_detected
    results['status'] = 'SUCCESS'
    results['n_clusters'] = int(n_good)
    
    if results['u_shape_detected'] and results.get('sign_inversion', False):
        results['conclusion'] = 'U-shape WITH sign inversion - Strong support for String Theory'
    elif results['u_shape_detected']:
        results['conclusion'] = 'U-shape detected (no clear sign inversion) - Partial support'
    elif results['mx_y_relation']['r_correlation'] > 0.3:
        results['conclusion'] = 'Good M-Y correlation but no U-shape detected'
    else:
        results['conclusion'] = 'Weak M-Y correlation - inconclusive'
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print(" PLANCK PSZ2 × MCXC CROSS-MATCH TEST")
    print(" Testing Q²R² pattern with independent X-ray masses")
    print("="*70)
    
    results = test_psz2_mcxc()
    
    print("\n" + "="*70)
    print(" RESULTS")
    print("="*70)
    
    if results.get('status') == 'SUCCESS':
        print(f"\n  Status: {results['status']}")
        print(f"  N clusters: {results.get('n_clusters', 'N/A')}")
        
        mx_y = results.get('mx_y_relation', {})
        print(f"\n  M_X - Y_SZ relation:")
        print(f"    Correlation: r = {mx_y.get('r_correlation', 'N/A'):.3f}")
        print(f"    Slope: {mx_y.get('slope', 'N/A'):.3f}")
        
        print(f"\n  U-shape detected: {results.get('u_shape_detected', 'Unknown')}")
        if results.get('sign_inversion') is not None:
            print(f"  Sign inversion: {results['sign_inversion']}")
        
        print(f"\n  Conclusion: {results.get('conclusion', 'N/A')}")
    else:
        print(f"\n  Status: {results.get('status', 'UNKNOWN')}")
        if 'error' in results:
            print(f"  Error: {results['error']}")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "planck_mcxc_crossmatch_results.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {output_file}")


if __name__ == "__main__":
    main()
