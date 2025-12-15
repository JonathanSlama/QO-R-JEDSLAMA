#!/usr/bin/env python3
"""
===============================================================================
PHASE 3B: PLANCK SZ CLUSTER ANALYSIS - CORRECTED
===============================================================================

Author: Jonathan E. Slama
Date: December 2025
===============================================================================
"""

import os
import numpy as np
import json
from scipy import stats
from scipy.optimize import curve_fit
from astropy.table import Table
from astropy.io import fits
from scipy.spatial import cKDTree
import gzip
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLANCK_DIR = os.path.join(BASE_DIR, "data", "level2_multicontext", "clusters", "COM_PCCS_SZ-Catalogs_vPR2")
CLUSTER_DIR = os.path.join(BASE_DIR, "data", "level2_multicontext", "clusters")
OUTPUT_DIR = os.path.join(BASE_DIR, "experimental", "04_RESULTS", "test_results")


def load_planck_sz():
    """Load Planck SZ cluster catalog."""
    
    print("  Loading Planck SZ catalog...")
    
    union_path = os.path.join(PLANCK_DIR, "HFI_PCCS_SZ-union_R2.08.fits.gz")
    
    if not os.path.exists(union_path):
        print(f"  ERROR: Not found: {union_path}")
        return None
    
    with gzip.open(union_path, 'rb') as f:
        hdul = fits.open(f)
        data = Table(hdul[1].data)
    
    print(f"  Total clusters: {len(data)}")
    print(f"  Key columns: RA, DEC, Y5R500, MSZ, REDSHIFT, SNR")
    
    return data


def analyze_planck_sz(data):
    """Analyze Planck SZ for Q2R2 signatures."""
    
    print("\n" + "="*70)
    print(" PLANCK SZ CLUSTER ANALYSIS")
    print("="*70)
    
    # Extract columns
    ra = np.array(data['RA'])
    dec = np.array(data['DEC'])
    y5r500 = np.array(data['Y5R500'])  # SZ signal
    msz = np.array(data['MSZ'])         # Mass
    redshift = np.array(data['REDSHIFT'])
    snr = np.array(data['SNR'])
    
    # Quality cuts
    mask = (
        np.isfinite(ra) & np.isfinite(dec) &
        np.isfinite(y5r500) & (y5r500 > 0) &
        np.isfinite(msz) & (msz > 0) &
        np.isfinite(redshift) & (redshift > 0) & (redshift < 1.5) &
        (snr > 4.5)  # Planck threshold
    )
    
    print(f"  After quality cuts: {mask.sum()} clusters")
    
    ra = ra[mask]
    dec = dec[mask]
    y5r500 = y5r500[mask]
    msz = msz[mask]
    redshift = redshift[mask]
    snr = snr[mask]
    
    results = {
        'n_total': len(data),
        'n_quality': int(mask.sum())
    }
    
    # Y-M scaling relation
    log_y = np.log10(y5r500)
    log_m = np.log10(msz)
    
    slope, intercept, r, p, se = stats.linregress(log_m, log_y)
    residuals = log_y - (slope * log_m + intercept)
    
    print(f"\n  Y-M Scaling Relation:")
    print(f"    log(Y5R500) = {slope:.3f} * log(MSZ) + {intercept:.3f}")
    print(f"    Pearson r = {r:.3f}")
    print(f"    Scatter = {residuals.std():.3f} dex")
    
    results['ym_relation'] = {
        'slope': float(slope),
        'intercept': float(intercept),
        'r': float(r),
        'scatter': float(residuals.std())
    }
    
    # Compute cluster environment (angular density)
    print("\n  Computing cluster environment...")
    
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y_coord = np.cos(dec_rad) * np.sin(ra_rad)
    z_coord = np.sin(dec_rad)
    coords = np.column_stack([x, y_coord, z_coord])
    
    tree = cKDTree(coords)
    n_neighbors = min(10, len(coords) - 1)
    distances, _ = tree.query(coords, k=n_neighbors+1)
    r_n = distances[:, -1]
    density = n_neighbors / (np.pi * r_n**2 + 1e-10)
    log_density = np.log10(density + 1e-10)
    
    # Residual-Environment correlation
    r_env, p_env = stats.pearsonr(residuals, log_density)
    rho_env, p_rho = stats.spearmanr(residuals, log_density)
    
    print(f"\n  Residual-Environment Correlation:")
    print(f"    Pearson r = {r_env:.4f}, p = {p_env:.2e}")
    print(f"    Spearman rho = {rho_env:.4f}, p = {p_rho:.2e}")
    
    results['env_correlation'] = {
        'pearson_r': float(r_env),
        'pearson_p': float(p_env),
        'spearman_rho': float(rho_env),
        'spearman_p': float(p_rho)
    }
    
    # Quadratic fit for U-shape
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c
    
    try:
        env_norm = (log_density - log_density.min()) / (log_density.max() - log_density.min() + 1e-10)
        
        popt, pcov = curve_fit(quadratic, env_norm, residuals)
        a, b, c = popt
        a_err = np.sqrt(pcov[0, 0])
        
        sig = abs(a) / a_err if a_err > 0 else 0
        
        print(f"\n  Quadratic Fit (U-shape test):")
        print(f"    a = {a:.5f} +/- {a_err:.5f}")
        print(f"    Significance: {sig:.1f} sigma")
        
        if a > 0 and sig > 2:
            shape = "U-shape"
            print(f"    Result: U-SHAPE DETECTED!")
        elif a < 0 and sig > 2:
            shape = "inverted"
            print(f"    Result: INVERTED U-shape!")
        else:
            shape = "flat"
            print(f"    Result: Flat (no significant curvature)")
        
        results['quadratic'] = {
            'a': float(a),
            'a_err': float(a_err),
            'significance': float(sig),
            'shape': shape
        }
        
    except Exception as e:
        print(f"  Quadratic fit failed: {e}")
    
    # Q vs R test: split by residual
    q25, q75 = np.percentile(residuals, [25, 75])
    q_mask = residuals < q25
    r_mask = residuals > q75
    
    density_q = log_density[q_mask]
    density_r = log_density[r_mask]
    
    t_stat, p_ttest = stats.ttest_ind(density_q, density_r)
    
    print(f"\n  Q vs R Environment Test:")
    print(f"    Q-dominated (low Y): mean density = {density_q.mean():.3f} +/- {density_q.std()/np.sqrt(len(density_q)):.3f}")
    print(f"    R-dominated (high Y): mean density = {density_r.mean():.3f} +/- {density_r.std()/np.sqrt(len(density_r)):.3f}")
    print(f"    Difference: {density_r.mean() - density_q.mean():.3f}")
    print(f"    T-statistic: {t_stat:.2f}, p = {p_ttest:.2e}")
    
    results['q_vs_r'] = {
        'mean_density_q': float(density_q.mean()),
        'mean_density_r': float(density_r.mean()),
        'difference': float(density_r.mean() - density_q.mean()),
        't_stat': float(t_stat),
        'p_value': float(p_ttest)
    }
    
    # Redshift evolution
    print(f"\n  Redshift Evolution:")
    z_bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.5)]
    
    z_results = []
    for z_min, z_max in z_bins:
        z_mask = (redshift >= z_min) & (redshift < z_max)
        n = z_mask.sum()
        if n > 20:
            r_z, p_z = stats.pearsonr(residuals[z_mask], log_density[z_mask])
            print(f"    z = [{z_min:.1f}, {z_max:.1f}]: n={n}, r={r_z:.3f}, p={p_z:.2e}")
            z_results.append({
                'z_min': z_min, 'z_max': z_max, 'n': int(n),
                'r': float(r_z), 'p': float(p_z)
            })
    
    results['redshift_evolution'] = z_results
    
    return results


def cross_match_erosita(planck_data):
    """Cross-match with eROSITA."""
    
    print("\n" + "="*70)
    print(" PLANCK x eROSITA CROSS-MATCH")
    print("="*70)
    
    erosita_path = os.path.join(CLUSTER_DIR, "eRASS1_clusters_Bulbul2024.fits")
    
    if not os.path.exists(erosita_path):
        print("  eROSITA not found")
        return None
    
    erosita = Table.read(erosita_path)
    print(f"  eROSITA clusters: {len(erosita)}")
    print(f"  eROSITA columns: {erosita.colnames[:10]}...")
    
    # Find coordinate columns
    ra_col = None
    for col in ['RA', 'RA_OPT', 'RA_X', 'ra']:
        if col in erosita.colnames:
            ra_col = col
            break
    
    dec_col = None
    for col in ['DEC', 'DEC_OPT', 'DEC_X', 'dec']:
        if col in erosita.colnames:
            dec_col = col
            break
    
    if ra_col is None or dec_col is None:
        print(f"  Could not find coordinates. Available: {erosita.colnames}")
        return None
    
    print(f"  Using: {ra_col}, {dec_col}")
    
    erosita_ra = np.array(erosita[ra_col])
    erosita_dec = np.array(erosita[dec_col])
    
    planck_ra = np.array(planck_data['RA'])
    planck_dec = np.array(planck_data['DEC'])
    
    # Build tree
    valid_e = np.isfinite(erosita_ra) & np.isfinite(erosita_dec)
    erosita_ra = erosita_ra[valid_e]
    erosita_dec = erosita_dec[valid_e]
    
    erosita_rad = np.radians(erosita_dec)
    erosita_x = np.cos(erosita_rad) * np.cos(np.radians(erosita_ra))
    erosita_y = np.cos(erosita_rad) * np.sin(np.radians(erosita_ra))
    erosita_z = np.sin(erosita_rad)
    erosita_coords = np.column_stack([erosita_x, erosita_y, erosita_z])
    
    tree = cKDTree(erosita_coords)
    
    valid_p = np.isfinite(planck_ra) & np.isfinite(planck_dec)
    planck_ra = planck_ra[valid_p]
    planck_dec = planck_dec[valid_p]
    
    planck_rad = np.radians(planck_dec)
    planck_x = np.cos(planck_rad) * np.cos(np.radians(planck_ra))
    planck_y = np.cos(planck_rad) * np.sin(np.radians(planck_ra))
    planck_z = np.sin(planck_rad)
    planck_coords = np.column_stack([planck_x, planck_y, planck_z])
    
    # Match within 5 arcmin
    max_sep_rad = np.radians(5/60)
    max_chord = 2 * np.sin(max_sep_rad / 2)
    
    distances, indices = tree.query(planck_coords, k=1)
    matched = distances < max_chord
    
    n_matched = matched.sum()
    print(f"  Matched: {n_matched} / {len(planck_ra)} Planck clusters")
    print(f"  Match fraction: {100*n_matched/len(planck_ra):.1f}%")
    
    return {
        'n_planck': int(len(planck_ra)),
        'n_erosita': int(len(erosita_ra)),
        'n_matched': int(n_matched),
        'match_fraction': float(n_matched / len(planck_ra))
    }


def main():
    print("="*70)
    print(" PHASE 3B: PLANCK SZ CLUSTER ANALYSIS")
    print("="*70)
    
    data = load_planck_sz()
    
    if data is None:
        return
    
    results = {}
    
    analysis = analyze_planck_sz(data)
    if analysis:
        results['planck_sz'] = analysis
    
    xmatch = cross_match_erosita(data)
    if xmatch:
        results['cross_match'] = xmatch
    
    # Summary
    print("\n" + "="*70)
    print(" PHASE 3B SUMMARY")
    print("="*70)
    
    if 'planck_sz' in results:
        r = results['planck_sz']
        print(f"""
  Planck SZ Analysis:
    Clusters: {r['n_quality']} (quality cut)
    Y-M scatter: {r['ym_relation']['scatter']:.3f} dex
    
  Environment Correlation:
    Pearson r = {r['env_correlation']['pearson_r']:.4f}
    p = {r['env_correlation']['pearson_p']:.2e}
    
  Q vs R Test:
    Density difference = {r['q_vs_r']['difference']:.3f}
    p = {r['q_vs_r']['p_value']:.2e}
        """)
        
        if 'quadratic' in r:
            q = r['quadratic']
            print(f"""
  U-shape Test:
    a = {q['a']:.5f} +/- {q['a_err']:.5f}
    Significance: {q['significance']:.1f} sigma
    Shape: {q['shape']}
            """)
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "phase3b_planck_sz_results.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
