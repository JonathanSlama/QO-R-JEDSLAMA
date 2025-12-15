#!/usr/bin/env python3
"""
===============================================================================
ACT-DR5 MCMF - TEST AVEC DIFFÉRENTS PROXIES D'ENVIRONNEMENT
===============================================================================

Le test initial montre a < 0 (∩-shape au lieu de U-shape).
Testons différents proxies d'environnement pour comprendre le signal.

Proxies à tester:
1. Densité de clusters voisins (original)
2. Redshift (évolution cosmique)
3. SNR (proxy de détectabilité / biais de sélection)
4. Coordonnées galactiques (effets systématiques)

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


def load_data():
    """Load and prepare ACT-DR5 MCMF data."""
    
    import glob
    act_path = glob.glob(os.path.join(DATA_DIR, "ACT_DR5_MCMF*.fits"))[0]
    data = Table.read(act_path)
    
    # Extract and filter
    z = np.array(data['z1C'])
    M = np.array(data['M500-1C'])
    L = np.array(data['L1C'])
    snr = np.array(data['SNR'])
    fcont = np.array(data['fcont1C'])
    ra = np.array(data['RAJ2000'])
    dec = np.array(data['DEJ2000'])
    
    mask = (
        np.isfinite(z) & (z > 0.05) & (z < 1.5) &
        np.isfinite(M) & (M > 0) &
        np.isfinite(L) & (L > 5) &
        np.isfinite(snr) & (snr > 4) &
        np.isfinite(fcont) & (fcont < 0.2)
    )
    
    return {
        'z': z[mask],
        'M': M[mask],
        'L': L[mask],
        'snr': snr[mask],
        'ra': ra[mask],
        'dec': dec[mask],
        'log_M': np.log10(M[mask]),
        'log_L': np.log10(L[mask])
    }


def compute_residuals(log_M, log_L):
    """Compute M-L residuals."""
    slope, intercept, r, p, se = stats.linregress(log_L, log_M)
    residuals = log_M - (slope * log_L + intercept)
    return residuals, slope, intercept, r


def test_quadratic(residuals, env, label):
    """Test for quadratic relationship."""
    
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c
    
    try:
        # Normalize env to [0,1]
        env_norm = (env - env.min()) / (env.max() - env.min() + 1e-10)
        
        popt, pcov = curve_fit(quadratic, env_norm, residuals)
        a, b, c = popt
        a_err = np.sqrt(pcov[0, 0])
        sig = abs(a) / a_err if a_err > 0 else 0
        
        # Linear correlation
        r_lin = stats.pearsonr(env_norm, residuals)[0]
        
        return {
            'label': label,
            'a': a,
            'a_err': a_err,
            'significance': sig,
            'ushape': a > 0 and sig > 2,
            'nshape': a < 0 and sig > 2,
            'r_linear': r_lin,
            'n': len(residuals)
        }
    except:
        return None


def compute_cluster_density(ra, dec, z, n_neighbors=10):
    """Compute local cluster density."""
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


def compute_angular_density(ra, dec, n_neighbors=10):
    """Compute angular density (2D on sky)."""
    coords = np.column_stack([ra, dec])
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=n_neighbors+1)
    
    r_n = distances[:, -1]
    density = n_neighbors / (np.pi * r_n**2 + 1e-10)
    return np.log10(density + 1e-10)


def main():
    print("="*70)
    print(" ACT-DR5 MCMF - MULTI-PROXY ENVIRONMENT TEST")
    print("="*70)
    
    # Load data
    print("\n  Loading data...")
    d = load_data()
    print(f"  N clusters: {len(d['z'])}")
    
    # Compute residuals
    residuals, slope, intercept, r_ml = compute_residuals(d['log_M'], d['log_L'])
    print(f"\n  M-L relation: slope = {slope:.3f}, r = {r_ml:.3f}")
    
    # Define environment proxies
    print("\n  Computing environment proxies...")
    
    proxies = {
        '1. 3D cluster density': compute_cluster_density(d['ra'], d['dec'], d['z']),
        '2. 2D angular density': compute_angular_density(d['ra'], d['dec']),
        '3. Redshift (cosmic time)': d['z'],
        '4. SNR (detection quality)': d['snr'],
        '5. Galactic latitude |b|': np.abs(np.sin(np.radians(d['dec']))),
        '6. Mass M500': d['log_M'],
        '7. Richness L': d['log_L'],
        '8. L/M ratio (gas proxy)': d['L'] / d['M'],
    }
    
    # Test each proxy
    print("\n" + "="*70)
    print(" RESULTS: QUADRATIC FIT residuals = a×env² + b×env + c")
    print("="*70)
    
    results = []
    
    print(f"\n  {'Proxy':<30} {'a':>10} {'±err':>8} {'σ':>6} {'r_lin':>7} {'Shape':>10}")
    print("-"*75)
    
    for name, env in proxies.items():
        result = test_quadratic(residuals, env, name)
        if result:
            results.append(result)
            shape = "U-shape" if result['ushape'] else ("∩-shape" if result['nshape'] else "~linear")
            print(f"  {name:<30} {result['a']:>10.4f} {result['a_err']:>8.4f} {result['significance']:>6.1f} {result['r_linear']:>7.3f} {shape:>10}")
    
    # Analysis
    print("\n" + "="*70)
    print(" ANALYSIS")
    print("="*70)
    
    # Find strongest signals
    sorted_results = sorted(results, key=lambda x: x['significance'], reverse=True)
    
    print("\n  Strongest signals:")
    for i, r in enumerate(sorted_results[:3]):
        print(f"    {i+1}. {r['label']}: a = {r['a']:.4f} ({r['significance']:.1f}σ)")
    
    # Check for U-shapes
    ushapes = [r for r in results if r['ushape']]
    nshapes = [r for r in results if r['nshape']]
    
    print(f"\n  U-shapes (a > 0, σ > 2): {len(ushapes)}")
    for r in ushapes:
        print(f"    - {r['label']}: a = {r['a']:.4f} ({r['significance']:.1f}σ)")
    
    print(f"\n  ∩-shapes (a < 0, σ > 2): {len(nshapes)}")
    for r in nshapes:
        print(f"    - {r['label']}: a = {r['a']:.4f} ({r['significance']:.1f}σ)")
    
    # Interpretation
    print("\n" + "="*70)
    print(" INTERPRETATION")
    print("="*70)
    
    # Check if it's selection effects
    snr_result = next((r for r in results if 'SNR' in r['label']), None)
    density_result = next((r for r in results if '3D cluster' in r['label']), None)
    
    if snr_result and density_result:
        print(f"""
  The ∩-shape in 3D density (a = {density_result['a']:.3f}) might be due to:
  
  1. SELECTION EFFECTS if correlated with SNR:
     - SNR shows a = {snr_result['a']:.3f} ({snr_result['significance']:.1f}σ)
     - High-density regions may have confused detections
     - Low-density regions may have lower S/N
  
  2. PHYSICAL EFFECT if NOT correlated with SNR:
     - Could indicate real environmental dependence
     - But opposite to Q²R² prediction
  
  3. ENVIRONMENT DEFINITION:
     - Cluster-to-cluster density may not be the right proxy
     - Galaxy-level environment within clusters would be different
        """)
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "act_dr5_multiproxy_results.json")
    
    with open(output_path, 'w') as f:
        json.dump({
            'n_clusters': len(d['z']),
            'ml_correlation': r_ml,
            'proxy_results': results
        }, f, indent=2, default=float)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
