#!/usr/bin/env python3
"""
===============================================================================
redMaPPer SDSS - TEST Q²R² DANS LES CLUSTERS OPTIQUES
===============================================================================

Test du pattern U-shape dans ~25,000 clusters redMaPPer.

AVANTAGES MAJEURS:
- Sélection PUREMENT OPTIQUE (pas de biais SZ/X-ray)
- Richesse λ = proxy de masse bien calibré
- Photo-z excellents (σ_z ~ 0.01)
- Grande statistique

Variables disponibles:
- lambda: Richesse optique (nombre de galaxies red-sequence)
- zlambda: Photo-z du cluster
- Positions des 5 BCGs candidates

Strategy:
- Relation: Richesse vs Luminosité BCG (ou vs environnement)
- Ou cross-match avec eROSITA/Planck pour masses

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


def load_redmapper():
    """Load redMaPPer SDSS cluster catalog."""
    
    import glob
    
    # Try different versions
    patterns = [
        os.path.join(DATA_DIR, "redMaPPer_Rykoff2016_table4.fits"),  # 26,111 clusters
        os.path.join(DATA_DIR, "redMaPPer_Rykoff2014_table0.fits"),  # 25,325 clusters
    ]
    
    for pattern in patterns:
        if os.path.exists(pattern):
            print(f"  Loading: {os.path.basename(pattern)}")
            data = Table.read(pattern)
            print(f"  Total clusters: {len(data)}")
            print(f"  Columns: {data.colnames}")
            return data
    
    return None


def explore_data(data):
    """Explore the redMaPPer data."""
    
    print("\n" + "="*70)
    print(" DATA EXPLORATION")
    print("="*70)
    
    # Find lambda column
    lam_col = None
    for c in data.colnames:
        if 'lambda' in c.lower() and 'z' not in c.lower():
            lam_col = c
            break
    
    if lam_col:
        lam = np.array(data[lam_col])
        valid = lam[np.isfinite(lam) & (lam > 0)]
        print(f"  {lam_col}: {len(valid)} valid, range {valid.min():.1f} - {valid.max():.1f}")
    
    # Redshift
    z_col = None
    for c in data.colnames:
        if 'zlambda' in c.lower() or c == 'z':
            z_col = c
            break
    
    if z_col:
        z = np.array(data[z_col])
        valid = z[np.isfinite(z) & (z > 0)]
        print(f"  {z_col}: {len(valid)} valid, range {valid.min():.3f} - {valid.max():.3f}")
    
    # BCG magnitude
    mag_cols = [c for c in data.colnames if 'mag' in c.lower()]
    print(f"  Magnitude columns: {mag_cols[:5]}...")
    
    return lam_col, z_col


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
    print(" redMaPPer SDSS - Q²R² TEST IN OPTICAL CLUSTERS")
    print(" N ~ 25,000 clusters (pure optical selection!)")
    print("="*70)
    
    # Load data
    data = load_redmapper()
    if data is None:
        print("  ERROR: Could not load redMaPPer catalog")
        return
    
    lam_col, z_col = explore_data(data)
    
    # Extract variables
    lam = np.array(data[lam_col]) if lam_col else None
    z = np.array(data[z_col]) if z_col else None
    ra = np.array(data['RAJ2000'])
    dec = np.array(data['DEJ2000'])
    
    # Try to get BCG i-band magnitude
    imag_col = None
    for c in ['imagm', 'imag', 'iLum']:
        if c in data.colnames:
            imag_col = c
            break
    
    imag = np.array(data[imag_col]) if imag_col else None
    
    if lam is None or z is None:
        print("  ERROR: Missing required columns")
        return
    
    # Quality cuts
    mask = (
        np.isfinite(z) & (z > 0.08) & (z < 0.55) &
        np.isfinite(lam) & (lam > 20)  # Richness > 20 for good mass proxy
    )
    
    if imag is not None:
        mask &= np.isfinite(imag) & (imag > 0) & (imag < 25)
    
    z = z[mask]
    lam = lam[mask]
    ra = ra[mask]
    dec = dec[mask]
    if imag is not None:
        imag = imag[mask]
    
    print(f"\n  After quality cuts: {len(z)} clusters")
    print(f"  z range: {z.min():.3f} - {z.max():.3f}")
    print(f"  λ range: {lam.min():.1f} - {lam.max():.1f}")
    
    # Log transform
    log_lam = np.log10(lam)
    
    # =========================================================================
    # TEST 1: RICHNESS VS ENVIRONMENT (direct test)
    # =========================================================================
    print("\n" + "="*70)
    print(" TEST 1: RICHNESS VS ENVIRONMENT")
    print("="*70)
    
    print("\n  Computing environment (cluster density)...")
    density = compute_cluster_density(ra, dec, z)
    
    # Test if richness depends on environment
    r_lam_env = stats.pearsonr(log_lam, density)[0]
    print(f"\n  Correlation λ vs density: r = {r_lam_env:.3f}")
    
    # Richness residuals from mean
    lam_mean = log_lam.mean()
    residuals = log_lam - lam_mean
    
    # Test U-shape
    result_density = test_quadratic(residuals, density, "density")
    if result_density:
        shape = "U-shape ✓" if result_density['ushape'] else ("∩-shape" if result_density['a'] < 0 and result_density['significance'] > 2 else "~linear")
        print(f"  Quadratic fit: a = {result_density['a']:.4f} ± {result_density['a_err']:.4f} ({result_density['significance']:.1f}σ)")
        print(f"  Shape: {shape}")
    
    # =========================================================================
    # TEST 2: RICHNESS-LUMINOSITY RELATION (if we have BCG mag)
    # =========================================================================
    if imag is not None:
        print("\n" + "="*70)
        print(" TEST 2: RICHNESS-LUMINOSITY RELATION (λ vs BCG mag)")
        print("="*70)
        
        # Richness vs BCG magnitude
        slope, intercept, r_lam_mag, p, se = stats.linregress(imag, log_lam)
        print(f"\n  log(λ) = {slope:.3f} × i_mag + {intercept:.3f}")
        print(f"  Correlation: r = {r_lam_mag:.3f}")
        
        residuals_lm = log_lam - (slope * imag + intercept)
        print(f"  Residual scatter: {residuals_lm.std():.3f} dex")
        
        # Test U-shape in residuals vs environment
        result_lm = test_quadratic(residuals_lm, density)
        if result_lm:
            shape = "U-shape ✓" if result_lm['ushape'] else ("∩-shape" if result_lm['a'] < 0 and result_lm['significance'] > 2 else "~linear")
            print(f"\n  U-shape in λ-mag residuals vs density:")
            print(f"  a = {result_lm['a']:.4f} ± {result_lm['a_err']:.4f} ({result_lm['significance']:.1f}σ)")
            print(f"  Shape: {shape}")
    
    # =========================================================================
    # TEST 3: MULTIPLE PROXIES
    # =========================================================================
    print("\n" + "="*70)
    print(" TEST 3: U-SHAPE WITH DIFFERENT PROXIES")
    print("="*70)
    
    proxies = {
        '1. 3D cluster density': density,
        '2. Redshift z': z,
        '3. Richness λ': log_lam,
    }
    
    print(f"\n  {'Proxy':<30} {'a':>10} {'±err':>8} {'σ':>6} {'Shape':>10}")
    print("-"*70)
    
    results = {}
    
    for name, env in proxies.items():
        result = test_quadratic(residuals, env, name)
        if result:
            results[name] = result
            shape = "U-shape ✓" if result['ushape'] else ("∩-shape" if result['a'] < 0 and result['significance'] > 2 else "~linear")
            print(f"  {name:<30} {result['a']:>10.4f} {result['a_err']:>8.4f} {result['significance']:>6.1f} {shape:>10}")
    
    # =========================================================================
    # TEST 4: SIGN INVERSION BY RICHNESS
    # =========================================================================
    print("\n" + "="*70)
    print(" TEST 4: SIGN INVERSION (split by richness)")
    print("="*70)
    
    median_lam = np.median(lam)
    rich = lam > median_lam
    poor = lam <= median_lam
    
    print(f"\n  Rich clusters (λ > {median_lam:.0f}): N = {rich.sum()}")
    print(f"  Poor clusters (λ ≤ {median_lam:.0f}): N = {poor.sum()}")
    
    result_rich = test_quadratic(residuals[rich], density[rich])
    result_poor = test_quadratic(residuals[poor], density[poor])
    
    if result_rich and result_poor:
        print(f"\n  Rich:  a = {result_rich['a']:.4f} ± {result_rich['a_err']:.4f} ({result_rich['significance']:.1f}σ)")
        print(f"  Poor:  a = {result_poor['a']:.4f} ± {result_poor['a_err']:.4f} ({result_poor['significance']:.1f}σ)")
        
        sign_inversion = (result_rich['a'] > 0 and result_poor['a'] < 0) or (result_rich['a'] < 0 and result_poor['a'] > 0)
        print(f"\n  Sign inversion: {'YES ✓' if sign_inversion else 'NO'}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    print(f"""
  N clusters: {len(z)}
  
  KEY RESULTS:
  
  1. redMaPPer is PURELY OPTICAL (no ICM bias)
  2. Testing richness variations with environment
  
  COMPARISON:
  - ACT-DR5 (SZ): No U-shape
  - redMaPPer (optical): Testing...
  
  NOTE: For better mass-observable test, should cross-match
  with eROSITA (X-ray masses) or use weak lensing masses.
    """)
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "redmapper_sdss_results.json")
    
    output = {
        'n_clusters': len(z),
        'proxy_results': results,
        'sign_inversion_test': {
            'rich': result_rich,
            'poor': result_poor
        } if result_rich and result_poor else None
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    
    print(f"  Results saved: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
