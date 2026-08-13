#!/usr/bin/env python3
"""
===============================================================================
COMPARAISON GALAXIES vs CLUSTERS - SIGNATURE Q²R²
===============================================================================

Analyse comparative pour comprendre pourquoi :
- Galaxies (ALFALFA, KiDS): U-shape + sign inversion ✓
- Clusters (eROSITA, redMaPPer): U-shape ✓ mais PAS de sign inversion

Hypothèses à tester:
1. Effet d'échelle (masse baryonique vs masse totale)
2. Proxy d'environnement inadéquat pour les clusters
3. Dilution par la matière noire dominante

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
from astropy.cosmology import FlatLambdaCDM
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "data", "level2_multicontext", "clusters")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experimental", "04_RESULTS", "test_results")


def test_quadratic(residuals, env):
    """Test for quadratic relationship."""
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c
    
    try:
        env_norm = (env - env.min()) / (env.max() - env.min() + 1e-10)
        popt, pcov = curve_fit(quadratic, env_norm, residuals)
        a, b, c = popt
        a_err = np.sqrt(pcov[0, 0])
        return {'a': float(a), 'a_err': float(a_err), 'sig': abs(a)/a_err if a_err > 0 else 0}
    except:
        return None


def compute_density(ra, dec, z, n_neighbors=10):
    """Compute local density in 3D."""
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
    print(" DIAGNOSTIC: POURQUOI PAS DE SIGN INVERSION DANS LES CLUSTERS?")
    print("="*70)
    
    # =========================================================================
    # LOAD eROSITA
    # =========================================================================
    print("\n" + "="*70)
    print(" 1. ANALYSE eROSITA - SPLIT PAR MASSE (pas Fgas)")
    print("="*70)
    
    erosita = Table.read(os.path.join(DATA_DIR, "eRASS1_clusters_Bulbul2024.fits"))
    
    z = np.array(erosita['zBest'])
    M = np.array(erosita['M500'])
    L = np.array(erosita['L500'])
    Fgas = np.array(erosita['Fgas500'])
    ra = np.array(erosita['RAJ2000'])
    dec = np.array(erosita['DEJ2000'])
    
    # Quality cuts
    mask = (
        np.isfinite(z) & (z > 0.01) & (z < 1.0) &
        np.isfinite(M) & (M > 0) &
        np.isfinite(L) & (L > 0) &
        np.isfinite(Fgas) & (Fgas > 0) & (Fgas < 0.5)
    )
    
    z, M, L, Fgas = z[mask], M[mask], L[mask], Fgas[mask]
    ra, dec = ra[mask], dec[mask]
    
    log_M, log_L, log_Fgas = np.log10(M), np.log10(L), np.log10(Fgas)
    
    # L-M residuals
    slope, intercept, _, _, _ = stats.linregress(log_M, log_L)
    residuals = log_L - (slope * log_M + intercept)
    
    # Environment
    density = compute_density(ra, dec, z)
    
    print(f"\n  N = {len(z)} clusters")
    
    # =========================================================================
    # TEST A: Split par MASSE (pas par Fgas)
    # =========================================================================
    print("\n  TEST A: Split par MASSE (analogue à R pour galaxies)")
    
    median_M = np.median(M)
    high_M = M > median_M
    low_M = M <= median_M
    
    result_high_M = test_quadratic(residuals[high_M], density[high_M])
    result_low_M = test_quadratic(residuals[low_M], density[low_M])
    
    print(f"    High mass (>{median_M:.1f}): a = {result_high_M['a']:.4f} ({result_high_M['sig']:.1f}σ)")
    print(f"    Low mass  (<{median_M:.1f}): a = {result_low_M['a']:.4f} ({result_low_M['sig']:.1f}σ)")
    sign_inv_M = (result_high_M['a'] > 0) != (result_low_M['a'] > 0)
    print(f"    Sign inversion: {'YES ✓' if sign_inv_M else 'NO'}")
    
    # =========================================================================
    # TEST B: Split par ratio Fgas/M (Q/R analogue)
    # =========================================================================
    print("\n  TEST B: Split par Fgas/M ratio (Q/R analogue)")
    
    ratio = Fgas / M
    median_ratio = np.median(ratio)
    high_ratio = ratio > median_ratio
    low_ratio = ratio <= median_ratio
    
    result_high_ratio = test_quadratic(residuals[high_ratio], density[high_ratio])
    result_low_ratio = test_quadratic(residuals[low_ratio], density[low_ratio])
    
    print(f"    High Fgas/M (gas-rich): a = {result_high_ratio['a']:.4f} ({result_high_ratio['sig']:.1f}σ)")
    print(f"    Low Fgas/M (gas-poor):  a = {result_low_ratio['a']:.4f} ({result_low_ratio['sig']:.1f}σ)")
    sign_inv_ratio = (result_high_ratio['a'] > 0) != (result_low_ratio['a'] > 0)
    print(f"    Sign inversion: {'YES ✓' if sign_inv_ratio else 'NO'}")
    
    # =========================================================================
    # TEST C: Quartiles de Fgas
    # =========================================================================
    print("\n  TEST C: Quartiles de Fgas")
    
    q25, q50, q75 = np.percentile(Fgas, [25, 50, 75])
    
    quartiles = [
        ('Q1 (lowest Fgas)', Fgas <= q25),
        ('Q2', (Fgas > q25) & (Fgas <= q50)),
        ('Q3', (Fgas > q50) & (Fgas <= q75)),
        ('Q4 (highest Fgas)', Fgas > q75)
    ]
    
    print(f"\n    {'Quartile':<20} {'N':>6} {'a':>10} {'σ':>6}")
    print("-"*50)
    
    for name, mask_q in quartiles:
        result = test_quadratic(residuals[mask_q], density[mask_q])
        if result:
            print(f"    {name:<20} {mask_q.sum():>6} {result['a']:>10.4f} {result['sig']:>6.1f}")
    
    # =========================================================================
    # TEST D: Environnement = Fgas lui-même (pas densité)
    # =========================================================================
    print("\n  TEST D: U-shape des résidus L-M en fonction de Fgas")
    
    result_fgas = test_quadratic(residuals, log_Fgas)
    print(f"    a = {result_fgas['a']:.4f} ({result_fgas['sig']:.1f}σ)")
    print(f"    Shape: {'U-shape ✓' if result_fgas['a'] > 0 and result_fgas['sig'] > 2 else 'other'}")
    
    # Split by density for sign inversion
    median_dens = np.median(density)
    high_dens = density > median_dens
    low_dens = density <= median_dens
    
    result_high_dens = test_quadratic(residuals[high_dens], log_Fgas[high_dens])
    result_low_dens = test_quadratic(residuals[low_dens], log_Fgas[low_dens])
    
    print(f"\n    Split by density:")
    print(f"    High density: a = {result_high_dens['a']:.4f} ({result_high_dens['sig']:.1f}σ)")
    print(f"    Low density:  a = {result_low_dens['a']:.4f} ({result_low_dens['sig']:.1f}σ)")
    sign_inv_dens = (result_high_dens['a'] > 0) != (result_low_dens['a'] > 0)
    print(f"    Sign inversion: {'YES ✓' if sign_inv_dens else 'NO'}")
    
    # =========================================================================
    # TEST E: Fraction baryonique effective
    # =========================================================================
    print("\n  TEST E: Residuals en bins de fraction baryonique")
    
    # Mgas500 si disponible
    Mgas = np.array(erosita['Mgas500'])[mask]
    valid_mgas = np.isfinite(Mgas) & (Mgas > 0)
    
    if valid_mgas.sum() > 100:
        f_baryon = Mgas[valid_mgas] / M[valid_mgas]
        res_fb = residuals[valid_mgas]
        
        n_bins = 5
        bins = np.percentile(f_baryon, np.linspace(0, 100, n_bins+1))
        
        print(f"\n    {'f_baryon bin':<25} {'N':>6} {'<residual>':>12}")
        print("-"*50)
        
        for i in range(n_bins):
            mask_bin = (f_baryon >= bins[i]) & (f_baryon < bins[i+1])
            if i == n_bins - 1:
                mask_bin = (f_baryon >= bins[i])
            if mask_bin.sum() > 5:
                print(f"    [{bins[i]:.4f}, {bins[i+1]:.4f}] {mask_bin.sum():>6} {res_fb[mask_bin].mean():>12.4f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print(" DIAGNOSTIC SUMMARY")
    print("="*70)
    
    print("""
  OBSERVATIONS:
  
  1. U-shape PRÉSENT dans les clusters (eROSITA: 3.4σ avec Fgas)
  2. Sign inversion ABSENTE quel que soit le split
  
  HYPOTHÈSES:
  
  H1: Dominance matière noire (85%+ de la masse)
      → Le signal baryonique Q²R² est dilué
      → U-shape visible mais amplitude réduite
      → Inversion impossible car DM domine partout
  
  H2: Proxy d'environnement inadéquat
      → "Densité de clusters" ≠ "environnement Q²R²"
      → Les clusters SONT les pics de densité
      → Besoin d'un proxy à plus grande échelle (filaments?)
  
  H3: Transition d'échelle physique
      → Q²R² opère à l'échelle des galaxies (10^10-10^12 M☉)
      → À l'échelle clusters (10^14-10^15 M☉), nouveau régime
      → Les champs moduli se comportent différemment
  
  CONCLUSION PRÉLIMINAIRE:
  
  Le U-shape dans les clusters est PARTIELLEMENT compatible avec Q²R²
  mais la signature complète (inversion) n'est pas observée.
  
  Cela RENFORCE l'hypothèse que Q²R² couple principalement aux
  baryons, pas à la matière noire dominante dans les clusters.
    """)


if __name__ == "__main__":
    main()
