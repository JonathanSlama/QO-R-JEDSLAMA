#!/usr/bin/env python3
"""
===============================================================================
PLANCK PSZ2 × MCXC - VERSION CORRIGÉE POUR LE REDSHIFT
===============================================================================

Le problème avec la v1 : on utilisait Y_obs (arcmin²) qui dépend de la distance !

Correction : Y_intrinsic = Y_obs × D_A²

La relation self-similaire est :
    Y_intrinsic × E(z)^(-2/3) ∝ M^(5/3)

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

# Cosmology (same as Planck)
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
# LOAD AND MATCH
# =============================================================================

def parse_sexagesimal_ra(ra_str):
    """Convert RA from 'HH MM SS.s' to degrees."""
    try:
        parts = str(ra_str).split()
        if len(parts) >= 3:
            h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
            return 15.0 * (h + m/60.0 + s/3600.0)  # 15 deg/hour
        else:
            return float(ra_str)
    except:
        return float(ra_str)


def parse_sexagesimal_dec(dec_str):
    """Convert DEC from '+/-DD MM SS.s' to degrees."""
    try:
        s = str(dec_str).strip()
        sign = -1 if s.startswith('-') else 1
        s = s.lstrip('+-')
        parts = s.split()
        if len(parts) >= 3:
            d, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
            return sign * (d + m/60.0 + sec/3600.0)
        else:
            return float(dec_str)
    except:
        return float(dec_str)


def load_and_match():
    """Load PSZ2 and MCXC, match using PSZ2's MCXC column."""
    
    psz2_path = find_psz2_file()
    mcxc_path = find_mcxc_file()
    
    if not psz2_path or not mcxc_path:
        print("  ERROR: Missing data files")
        return None
    
    print(f"  Loading PSZ2: {os.path.basename(psz2_path)}")
    with fits.open(psz2_path) as hdul:
        psz2 = Table(hdul[1].data)
    
    print(f"  Loading MCXC: {os.path.basename(mcxc_path)}")
    mcxc = Table.read(mcxc_path)
    
    # Build MCXC lookup
    mcxc_lookup = {}
    for i, row in enumerate(mcxc):
        name = str(row['MCXC']).strip()
        mcxc_lookup[name] = i
    
    # Match
    matched_data = []
    
    for i, row in enumerate(psz2):
        mcxc_name = str(row['MCXC']).strip().replace("b'", "").replace("'", "")
        
        if mcxc_name and mcxc_name in mcxc_lookup:
            j = mcxc_lookup[mcxc_name]
            mcxc_row = mcxc[j]
            
            # Parse coordinates (may be sexagesimal)
            ra = parse_sexagesimal_ra(mcxc_row['RAJ2000'])
            dec = parse_sexagesimal_dec(mcxc_row['DEJ2000'])
            
            matched_data.append({
                'name': mcxc_name,
                'ra': ra,
                'dec': dec,
                'z': float(mcxc_row['z']),
                'Y_obs': float(row['Y5R500']),  # arcmin²
                'M_X': float(mcxc_row['M500']),  # 10^14 M_sun
                'L_X': float(mcxc_row['L500']),  # 10^44 erg/s
                'R500': float(mcxc_row['R500']),  # Mpc
            })
    
    print(f"  Matched: {len(matched_data)} clusters")
    return matched_data


# =============================================================================
# COMPUTE INTRINSIC Y
# =============================================================================

def compute_intrinsic_Y(data):
    """
    Compute intrinsic (distance-independent) Y_SZ.
    
    Y_obs [arcmin²] = Y_true [Mpc²] / D_A² [Mpc²/arcmin²]
    
    So: Y_true = Y_obs × D_A² × (arcmin → rad)²
    
    For the M-Y relation:
    Y_true × E(z)^(-2/3) ∝ M^(5/3)
    """
    
    print("\n  Computing intrinsic Y_SZ (correcting for distance)...")
    
    for d in data:
        z = d['z']
        
        # Angular diameter distance in Mpc
        D_A = COSMO.angular_diameter_distance(z).to(u.Mpc).value
        
        # Convert Y_obs from arcmin² to Mpc²
        # 1 arcmin = π/(180×60) rad
        # D_A [Mpc] at angular scale θ [rad] → physical scale = θ × D_A
        # So 1 arcmin² → (π/(180×60))² × D_A² Mpc²
        
        arcmin_to_rad = np.pi / (180 * 60)
        
        # Y_true in Mpc² (arbitrary normalization)
        Y_true = d['Y_obs'] * (arcmin_to_rad ** 2) * (D_A ** 2)
        
        # E(z) correction for self-similar scaling
        E_z = COSMO.efunc(z)
        
        # Scaled Y for M-Y relation
        # Y_scaled = Y_true / E(z)^(2/3)
        Y_scaled = Y_true / (E_z ** (2/3))
        
        d['D_A'] = D_A
        d['E_z'] = E_z
        d['Y_true'] = Y_true
        d['Y_scaled'] = Y_scaled
    
    return data


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
    except:
        return None


def compute_environment(data, max_sep_mpc=50, dz_max=0.05):
    """Compute cluster environment from neighbor count."""
    
    ra = np.array([d['ra'] for d in data])
    dec = np.array([d['dec'] for d in data])
    z = np.array([d['z'] for d in data])
    
    n = len(data)
    env = np.zeros(n)
    
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    
    for i in range(n):
        if z[i] <= 0:
            continue
        
        D_A = data[i]['D_A']
        max_sep_deg = np.degrees(max_sep_mpc / D_A) if D_A > 0 else 10
        
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

def test_with_correction():
    """Test M-Y relation with proper distance correction."""
    
    results = {
        'test_name': 'Planck PSZ2 × MCXC (Distance Corrected)',
        'version': '4.0 - With D_A² and E(z) corrections'
    }
    
    # Load and match
    data = load_and_match()
    if not data:
        results['status'] = 'NO_DATA'
        return results
    
    # Compute intrinsic Y
    data = compute_intrinsic_Y(data)
    
    # Quality cuts
    data = [d for d in data if d['z'] > 0.01 and d['z'] < 1.0 
            and d['M_X'] > 0 and d['Y_scaled'] > 0]
    
    n = len(data)
    print(f"\n  After quality cuts: {n} clusters")
    
    if n < 50:
        results['status'] = 'INSUFFICIENT_DATA'
        return results
    
    # Extract arrays
    z = np.array([d['z'] for d in data])
    Y_obs = np.array([d['Y_obs'] for d in data])
    Y_scaled = np.array([d['Y_scaled'] for d in data])
    M_X = np.array([d['M_X'] for d in data])
    L_X = np.array([d['L_X'] for d in data])
    
    # =========================================================================
    # TEST 1: Uncorrected Y_obs vs M_X (should show anti-correlation!)
    # =========================================================================
    print("\n" + "="*60)
    print(" TEST 1: UNCORRECTED Y_obs vs M_X (expect anti-correlation)")
    print("="*60)
    
    log_Y_obs = np.log10(Y_obs)
    log_M = np.log10(M_X)
    
    slope1, intercept1, r1, p1, _ = stats.linregress(log_M, log_Y_obs)
    
    print(f"  log(Y_obs) = {slope1:.3f} × log(M) + {intercept1:.3f}")
    print(f"  Correlation: r = {r1:.3f}")
    print(f"  {'✓ Negative as expected (distance bias)' if r1 < 0 else '⚠️ Positive (unexpected)'}")
    
    results['uncorrected'] = {
        'slope': float(slope1),
        'r': float(r1),
        'interpretation': 'Distance bias creates artificial anti-correlation'
    }
    
    # =========================================================================
    # TEST 2: CORRECTED Y_scaled vs M_X (should show POSITIVE correlation!)
    # =========================================================================
    print("\n" + "="*60)
    print(" TEST 2: CORRECTED Y_scaled vs M_X (expect positive correlation!)")
    print("="*60)
    
    log_Y_scaled = np.log10(Y_scaled)
    
    slope2, intercept2, r2, p2, _ = stats.linregress(log_M, log_Y_scaled)
    
    print(f"  log(Y_scaled) = {slope2:.3f} × log(M) + {intercept2:.3f}")
    print(f"  Correlation: r = {r2:.3f}")
    print(f"  Expected slope: ~1.67 (self-similar)")
    
    if r2 > 0.3:
        print(f"  ✓ POSITIVE correlation - correction worked!")
    elif r2 > 0:
        print(f"  ⚠️ Weak positive correlation")
    else:
        print(f"  ❌ Still negative - something else is wrong")
    
    results['corrected'] = {
        'slope': float(slope2),
        'r': float(r2),
        'p_value': float(p2),
        'expected_slope': 1.67
    }
    
    # =========================================================================
    # TEST 3: Check redshift correlation
    # =========================================================================
    print("\n" + "="*60)
    print(" TEST 3: REDSHIFT CORRELATIONS (diagnostic)")
    print("="*60)
    
    r_z_M, _ = stats.pearsonr(z, M_X)
    r_z_Yobs, _ = stats.pearsonr(z, Y_obs)
    r_z_Yscaled, _ = stats.pearsonr(z, Y_scaled)
    
    print(f"  z vs M_X:      r = {r_z_M:.3f} (Malmquist bias if positive)")
    print(f"  z vs Y_obs:    r = {r_z_Yobs:.3f} (should be negative - distance dilution)")
    print(f"  z vs Y_scaled: r = {r_z_Yscaled:.3f} (should be ~0 if correction works)")
    
    results['z_correlations'] = {
        'z_vs_M': float(r_z_M),
        'z_vs_Y_obs': float(r_z_Yobs),
        'z_vs_Y_scaled': float(r_z_Yscaled)
    }
    
    # =========================================================================
    # TEST 4: U-shape in residuals (only if correlation is positive)
    # =========================================================================
    
    if r2 > 0.2:
        print("\n" + "="*60)
        print(" TEST 4: U-SHAPE IN RESIDUALS")
        print("="*60)
        
        # Compute residuals from M-Y relation
        log_Y_pred = slope2 * log_M + intercept2
        residuals = log_Y_scaled - log_Y_pred
        
        print(f"  Residual scatter: {np.std(residuals):.3f} dex")
        
        # Compute environment
        print("  Computing environment...")
        env = compute_environment(data)
        
        # Fit U-shape
        fit_full = fit_ushape(env, residuals)
        
        if fit_full:
            u_detected = fit_full['a'] > 0 and fit_full['a'] > 1.5 * fit_full['a_err']
            print(f"  Full sample: a = {fit_full['a']:.6f} ± {fit_full['a_err']:.6f}")
            print(f"  U-shape detected: {'YES ✓' if u_detected else 'NO'}")
            
            results['u_shape'] = {
                'a': float(fit_full['a']),
                'a_err': float(fit_full['a_err']),
                'detected': u_detected
            }
        else:
            print("  U-shape fit failed")
            results['u_shape'] = {'detected': False, 'error': 'fit_failed'}
        
        # Q-dominated vs R-dominated
        # Gas fraction proxy: Y/M^(5/3)
        f_gas = Y_scaled / (M_X ** (5/3))
        f_gas_norm = f_gas / np.median(f_gas)
        
        q_mask = f_gas_norm >= np.percentile(f_gas_norm, 60)
        r_mask = (f_gas_norm < np.percentile(f_gas_norm, 40)) & (M_X > np.percentile(M_X, 50))
        
        print(f"\n  Q-dominated (gas-rich): N = {q_mask.sum()}")
        if q_mask.sum() > 30:
            fit_q = fit_ushape(env[q_mask], residuals[q_mask], n_bins=6, min_per_bin=5)
            if fit_q:
                print(f"    a = {fit_q['a']:.6f} ± {fit_q['a_err']:.6f}")
                results['q_dominated'] = {'a': float(fit_q['a']), 'a_err': float(fit_q['a_err'])}
        
        print(f"  R-dominated (massive): N = {r_mask.sum()}")
        if r_mask.sum() > 30:
            fit_r = fit_ushape(env[r_mask], residuals[r_mask], n_bins=6, min_per_bin=5)
            if fit_r:
                print(f"    a = {fit_r['a']:.6f} ± {fit_r['a_err']:.6f}")
                results['r_dominated'] = {'a': float(fit_r['a']), 'a_err': float(fit_r['a_err'])}
        
        # Sign inversion
        if 'q_dominated' in results and 'r_dominated' in results:
            sign_inv = np.sign(results['q_dominated']['a']) != np.sign(results['r_dominated']['a'])
            results['sign_inversion'] = bool(sign_inv)
            print(f"\n  Sign inversion: {'YES ✓' if sign_inv else 'NO'}")
    
    else:
        print("\n  ⚠️ Skipping U-shape test (correlation too weak)")
        results['u_shape'] = {'detected': False, 'error': 'correlation_too_weak'}
    
    # Summary
    results['n_clusters'] = n
    results['status'] = 'SUCCESS'
    
    if results['corrected']['r'] > 0.3:
        results['conclusion'] = 'Distance correction successful - positive M-Y correlation'
    else:
        results['conclusion'] = 'Distance correction insufficient - need further investigation'
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print(" PLANCK PSZ2 × MCXC - DISTANCE CORRECTED TEST")
    print(" Correcting Y_SZ for angular diameter distance")
    print("="*70)
    
    results = test_with_correction()
    
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    if 'corrected' in results:
        print(f"\n  Before correction: r = {results.get('uncorrected', {}).get('r', 'N/A'):.3f}")
        print(f"  After correction:  r = {results['corrected']['r']:.3f}")
        
        if results['corrected']['r'] > results.get('uncorrected', {}).get('r', -1):
            print("\n  ✓ Distance correction IMPROVED the correlation!")
        
    print(f"\n  Conclusion: {results.get('conclusion', 'N/A')}")
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "planck_mcxc_corrected_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {output_file}")


if __name__ == "__main__":
    main()
