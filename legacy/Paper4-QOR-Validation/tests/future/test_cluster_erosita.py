#!/usr/bin/env python3
"""
===============================================================================
eROSITA eRASS1 - TEST Q²R² DANS LES CLUSTERS X-RAY
===============================================================================

Test du pattern U-shape dans 12,247 clusters eROSITA.

AVANTAGES MAJEURS:
- Masses X-ray INDEPENDANTES du SZ
- Fgas500 (fraction de gaz) = proxy Q DIRECT !
- Pas de biais Malmquist type Planck
- Plus grand catalogue X-ray jamais publié

Variables disponibles:
- M500: Masse totale (10^14 M_sun)
- L500: Luminosité X-ray (10^44 erg/s)
- Fgas500: Fraction de gaz baryonique = Q proxy !
- KT: Température du gaz (keV)
- zBest: Redshift

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
                        "data", "level2_multicontext", "clusters")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experimental", "04_RESULTS", "test_results")


def load_erosita():
    """Load eROSITA eRASS1 cluster catalog."""
    
    import glob
    path = glob.glob(os.path.join(DATA_DIR, "eRASS1_clusters_Bulbul2024.fits"))[0]
    
    print(f"  Loading: {os.path.basename(path)}")
    data = Table.read(path)
    
    print(f"  Total clusters: {len(data)}")
    print(f"  Columns: {data.colnames}")
    
    return data


def explore_data(data):
    """Explore the eROSITA data."""
    
    print("\n" + "="*70)
    print(" DATA EXPLORATION")
    print("="*70)
    
    # Key variables
    vars_to_check = ['zBest', 'M500', 'L500', 'Fgas500', 'KT', 'R500']
    
    for var in vars_to_check:
        if var in data.colnames:
            vals = np.array(data[var])
            valid = vals[np.isfinite(vals) & (vals > 0)]
            print(f"  {var}: {len(valid)} valid, range {valid.min():.3g} - {valid.max():.3g}")
    
    return


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


def compute_cluster_density(ra, dec, z, n_neighbors=10):
    """Compute local cluster density in 3D."""
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


def main():
    print("="*70)
    print(" eROSITA eRASS1 - Q²R² TEST IN X-RAY CLUSTERS")
    print(" N = 12,247 clusters (largest X-ray sample ever!)")
    print("="*70)
    
    # Load data
    data = load_erosita()
    explore_data(data)
    
    # Extract variables
    z = np.array(data['zBest'])
    M = np.array(data['M500'])      # 10^14 M_sun
    L = np.array(data['L500'])      # 10^44 erg/s
    Fgas = np.array(data['Fgas500'])  # Gas fraction = Q proxy!
    KT = np.array(data['KT'])       # Temperature keV
    ra = np.array(data['RAJ2000'])
    dec = np.array(data['DEJ2000'])
    
    # Quality cuts
    mask = (
        np.isfinite(z) & (z > 0.01) & (z < 1.0) &
        np.isfinite(M) & (M > 0) &
        np.isfinite(L) & (L > 0) &
        np.isfinite(Fgas) & (Fgas > 0) & (Fgas < 0.5) &  # Physical range
        np.isfinite(KT) & (KT > 0)
    )
    
    z = z[mask]
    M = M[mask]
    L = L[mask]
    Fgas = Fgas[mask]
    KT = KT[mask]
    ra = ra[mask]
    dec = dec[mask]
    
    print(f"\n  After quality cuts: {len(z)} clusters")
    print(f"  z range: {z.min():.3f} - {z.max():.3f}")
    print(f"  M500 range: {M.min():.2e} - {M.max():.2e}")
    print(f"  Fgas range: {Fgas.min():.3f} - {Fgas.max():.3f}")
    
    # Log transform
    log_M = np.log10(M)
    log_L = np.log10(L)
    log_Fgas = np.log10(Fgas)
    
    # =========================================================================
    # TEST 1: L-M RELATION
    # =========================================================================
    print("\n" + "="*70)
    print(" TEST 1: L_X - M RELATION")
    print("="*70)
    
    slope, intercept, r_lm, p, se = stats.linregress(log_M, log_L)
    print(f"\n  log(L) = {slope:.3f} × log(M) + {intercept:.3f}")
    print(f"  Correlation: r = {r_lm:.3f}")
    print(f"  Expected slope: ~1.5-2.0 (self-similar: L ∝ M^(4/3))")
    
    residuals_lm = log_L - (slope * log_M + intercept)
    print(f"  Residual scatter: {residuals_lm.std():.3f} dex")
    
    # =========================================================================
    # TEST 2: U-SHAPE WITH DIFFERENT PROXIES
    # =========================================================================
    print("\n" + "="*70)
    print(" TEST 2: U-SHAPE IN L-M RESIDUALS")
    print("="*70)
    
    # Compute environment
    print("\n  Computing environment...")
    density = compute_cluster_density(ra, dec, z)
    
    # Test different proxies
    proxies = {
        '1. Fgas (gas fraction = Q!)': log_Fgas,
        '2. 3D cluster density': density,
        '3. Redshift z': z,
        '4. Temperature KT': np.log10(KT),
        '5. Mass M500': log_M,
    }
    
    print(f"\n  {'Proxy':<35} {'a':>10} {'±err':>8} {'σ':>6} {'Shape':>10}")
    print("-"*75)
    
    results = {}
    
    for name, env in proxies.items():
        result = test_quadratic(residuals_lm, env, name)
        if result:
            results[name] = result
            shape = "U-shape ✓" if result['ushape'] else ("∩-shape" if result['a'] < 0 and result['significance'] > 2 else "~linear")
            print(f"  {name:<35} {result['a']:>10.4f} {result['a_err']:>8.4f} {result['significance']:>6.1f} {shape:>10}")
    
    # =========================================================================
    # TEST 3: SIGN INVERSION WITH Fgas
    # =========================================================================
    print("\n" + "="*70)
    print(" TEST 3: SIGN INVERSION (split by Fgas = Q proxy)")
    print("="*70)
    
    median_Fgas = np.median(Fgas)
    high_Fgas = Fgas > median_Fgas  # Q-dominated (gas-rich)
    low_Fgas = Fgas <= median_Fgas   # R-dominated (gas-poor)
    
    print(f"\n  High Fgas (Q-dominated): N = {high_Fgas.sum()}")
    print(f"  Low Fgas (R-dominated): N = {low_Fgas.sum()}")
    
    # Use density as environment
    result_q = test_quadratic(residuals_lm[high_Fgas], density[high_Fgas])
    result_r = test_quadratic(residuals_lm[low_Fgas], density[low_Fgas])
    
    if result_q and result_r:
        print(f"\n  Q-dominated (high Fgas): a = {result_q['a']:.4f} ± {result_q['a_err']:.4f} ({result_q['significance']:.1f}σ)")
        print(f"  R-dominated (low Fgas):  a = {result_r['a']:.4f} ± {result_r['a_err']:.4f} ({result_r['significance']:.1f}σ)")
        
        sign_inversion = (result_q['a'] > 0 and result_r['a'] < 0) or (result_q['a'] < 0 and result_r['a'] > 0)
        print(f"\n  Sign inversion: {'YES ✓' if sign_inversion else 'NO'}")
    
    # =========================================================================
    # TEST 4: BINNED ANALYSIS IN Fgas
    # =========================================================================
    print("\n" + "="*70)
    print(" TEST 4: BINNED ANALYSIS (L-M residuals in Fgas bins)")
    print("="*70)
    
    n_bins = 10
    bin_edges = np.percentile(Fgas, np.linspace(0, 100, n_bins+1))
    
    print(f"\n  {'Bin':<5} {'Fgas':<20} {'N':>6} {'<residual>':>12} {'std':>8}")
    print("-"*55)
    
    bin_centers = []
    bin_means = []
    
    for i in range(n_bins):
        mask_bin = (Fgas >= bin_edges[i]) & (Fgas < bin_edges[i+1])
        if i == n_bins - 1:
            mask_bin = (Fgas >= bin_edges[i]) & (Fgas <= bin_edges[i+1])
        
        if mask_bin.sum() > 10:
            center = (bin_edges[i] + bin_edges[i+1]) / 2
            mean_res = residuals_lm[mask_bin].mean()
            std_res = residuals_lm[mask_bin].std()
            
            bin_centers.append(center)
            bin_means.append(mean_res)
            
            print(f"  {i+1:<5} [{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f}] {mask_bin.sum():>6} {mean_res:>12.4f} {std_res:>8.4f}")
    
    # Fit parabola to bin means
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
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    fgas_result = results.get('1. Fgas (gas fraction = Q!)', {})
    
    print(f"""
  N clusters: {len(z)}
  L-M correlation: r = {r_lm:.3f}
  
  KEY RESULTS:
  
  1. U-shape with Fgas (gas fraction):
     a = {fgas_result.get('a', 'N/A')} ({fgas_result.get('significance', 0):.1f}σ)
     {'✓ DETECTED!' if fgas_result.get('ushape', False) else '✗ Not detected'}
  
  2. eROSITA advantages:
     - X-ray selection (no SZ bias)
     - Direct Fgas measurement (= Q proxy!)
     - 12,000+ clusters
  
  COMPARISON WITH ACT-DR5:
  - ACT-DR5: No U-shape with density proxy
  - eROSITA: Testing with Fgas = direct Q measurement
    """)
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "erosita_erass1_results.json")
    
    output = {
        'n_clusters': len(z),
        'lm_correlation': float(r_lm),
        'lm_slope': float(slope),
        'proxy_results': results,
        'sign_inversion_test': {
            'q_dominated': result_q,
            'r_dominated': result_r
        } if result_q and result_r else None
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    
    print(f"  Results saved: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
