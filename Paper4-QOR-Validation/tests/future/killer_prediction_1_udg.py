#!/usr/bin/env python3
"""
===============================================================================
KILLER PREDICTION #1: ULTRA-DIFFUSE GALAXIES (UDGs)
===============================================================================

Tests the Q2R2 prediction that UDGs (extremely gas-poor, diffuse) should show
a STRONGLY INVERTED U-shape (a << 0) in scaling relation residuals.

PREDICTION: a_UDG < -1.0 at > 3σ

FALSIFICATION: If UDGs show a > 0 (U-shape), Q2R2 is falsified.

Author: Jonathan Édouard Slama
Date: December 2025
===============================================================================
"""

import os
import numpy as np
import json
from scipy import stats
from scipy.optimize import curve_fit
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "experimental", "04_RESULTS", "test_results")


def download_udg_catalog():
    """
    Download UDG catalog from VizieR or literature.
    
    Options:
    1. MATLAS survey UDGs (Marleau et al. 2021)
    2. Coma cluster UDGs (van Dokkum et al. 2015)
    3. SDSS UDG catalog (Greco et al. 2018)
    """
    
    print("  UDG Catalog Options:")
    print("  1. MATLAS survey: J/MNRAS/508/2847 (Marleau et al. 2021)")
    print("  2. Coma UDGs: Various catalogs")
    print("  3. Systematically-found UDGs from SDSS")
    print()
    
    # Try to download MATLAS catalog via VizieR
    try:
        from astroquery.vizier import Vizier
        
        print("  Querying VizieR for MATLAS UDGs...")
        
        v = Vizier(columns=["*"], row_limit=-1)
        
        # MATLAS UDG catalog
        result = v.query_constraints(catalog="J/MNRAS/508/2847")
        
        if result and len(result) > 0:
            udg_table = result[0]
            print(f"  Found {len(udg_table)} MATLAS UDGs")
            return udg_table
            
    except ImportError:
        print("  astroquery not installed, using mock data")
    except Exception as e:
        print(f"  VizieR query failed: {e}")
    
    # Create mock UDG data for pipeline testing
    print("  Creating mock UDG data for pipeline testing...")
    
    np.random.seed(42)
    n_udg = 200
    
    # UDG properties (typical values from literature)
    # UDGs are defined as: R_eff > 1.5 kpc, μ_0 > 24 mag/arcsec²
    
    mock_data = {
        'Name': [f'UDG-{i:04d}' for i in range(n_udg)],
        'RA': np.random.uniform(150, 220, n_udg),  # Typical survey footprint
        'Dec': np.random.uniform(-5, 25, n_udg),
        'R_eff': np.random.lognormal(np.log(3), 0.3, n_udg),  # kpc, peaked at 3 kpc
        'mu_0': np.random.normal(25.5, 0.8, n_udg),  # mag/arcsec², central SB
        'M_star': np.random.lognormal(np.log(1e8), 0.5, n_udg),  # M_sun, low mass
        'V_disp': np.random.lognormal(np.log(20), 0.3, n_udg),  # km/s, velocity dispersion
        'f_HI': np.random.beta(0.5, 5, n_udg) * 0.1,  # Very gas-poor (< 10%)
        'Distance': np.random.uniform(10, 100, n_udg),  # Mpc
    }
    
    udg_table = Table(mock_data)
    
    print(f"  Created {n_udg} mock UDGs")
    print("  WARNING: Mock data - results are for pipeline testing only!")
    
    return udg_table


def compute_udg_scaling_residuals(udg_table):
    """
    Compute scaling relation residuals for UDGs.
    
    We use the mass-velocity dispersion relation:
    M_* ~ σ^4 (Faber-Jackson like)
    """
    
    print("\n  Computing UDG scaling relation residuals...")
    
    # Extract data
    if 'M_star' in udg_table.colnames:
        log_mass = np.log10(np.array(udg_table['M_star']))
    else:
        print("  ERROR: No stellar mass column found")
        return None
    
    if 'V_disp' in udg_table.colnames:
        sigma = np.array(udg_table['V_disp'])
        log_sigma = np.log10(sigma)
    else:
        print("  ERROR: No velocity dispersion column found")
        return None
    
    # Quality cuts
    valid = np.isfinite(log_mass) & np.isfinite(log_sigma) & (sigma > 5) & (sigma < 100)
    log_mass = log_mass[valid]
    log_sigma = log_sigma[valid]
    
    print(f"  Valid UDGs after cuts: {len(log_mass)}")
    
    if len(log_mass) < 20:
        print("  WARNING: Very few UDGs - results may be unreliable")
    
    # Fit Faber-Jackson relation: log(M_*) = a * log(σ) + b
    slope, intercept, r, p, se = stats.linregress(log_sigma, log_mass)
    
    print(f"  Faber-Jackson fit: log(M_*) = {slope:.2f} * log(σ) + {intercept:.2f}")
    print(f"  Expected slope ~ 4.0, observed: {slope:.2f}")
    
    # Compute residuals
    residuals = log_mass - (slope * log_sigma + intercept)
    
    return {
        'residuals': residuals,
        'log_mass': log_mass,
        'log_sigma': log_sigma,
        'valid_mask': valid,
        'relation': {'slope': slope, 'intercept': intercept, 'r': r}
    }


def compute_environment(ra, dec, n_neighbors=10):
    """Compute local environment (angular density)."""
    
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    
    coords = np.column_stack([x, y, z])
    tree = cKDTree(coords)
    
    n_neighbors = min(n_neighbors, len(coords) - 1)
    distances, _ = tree.query(coords, k=n_neighbors+1)
    r_n = distances[:, -1]
    
    density = n_neighbors / (np.pi * r_n**2 + 1e-10)
    
    return np.log10(density + 1e-10)


def test_ushape_killer_prediction(udg_table, scaling_results):
    """
    Test the killer prediction: UDGs should show a << 0 (inverted U-shape).
    """
    
    print("\n" + "="*70)
    print(" KILLER PREDICTION TEST: UDG U-SHAPE")
    print("="*70)
    
    print("""
  Q2R2 PREDICTION:
  ================
  UDGs are extremely gas-poor (R-dominated) in low-density environments.
  They should show a STRONGLY INVERTED U-shape: a << 0
  
  QUANTITATIVE TARGET: a < -1.0 at > 3σ
  
  FALSIFICATION: If a > 0, Q2R2 is FALSIFIED.
    """)
    
    # Get coordinates for valid UDGs
    ra = np.array(udg_table['RA'])[scaling_results['valid_mask']]
    dec = np.array(udg_table['Dec'])[scaling_results['valid_mask']]
    
    # Compute environment
    log_density = compute_environment(ra, dec)
    residuals = scaling_results['residuals']
    
    # Normalize environment
    env_norm = (log_density - log_density.min()) / (log_density.max() - log_density.min() + 1e-10)
    
    # Quadratic fit
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c
    
    try:
        popt, pcov = curve_fit(quadratic, env_norm, residuals)
        a, b, c = popt
        a_err = np.sqrt(pcov[0, 0])
        
        significance = abs(a) / a_err if a_err > 0 else 0
        
        print(f"\n  RESULTS:")
        print(f"  --------")
        print(f"  Quadratic coefficient: a = {a:.4f} ± {a_err:.4f}")
        print(f"  Significance: {significance:.1f}σ")
        
        if a < 0:
            print(f"\n  Shape: INVERTED U-shape ✓")
            if a < -1.0 and significance > 3:
                print(f"  KILLER PREDICTION: CONFIRMED! (a < -1.0 at > 3σ)")
                status = "CONFIRMED"
            else:
                print(f"  KILLER PREDICTION: Consistent but not yet confirmed")
                status = "CONSISTENT"
        else:
            print(f"\n  Shape: U-shape or flat")
            if a > 0 and significance > 3:
                print(f"  KILLER PREDICTION: *** FALSIFIED *** (a > 0 at > 3σ)")
                print(f"  This would require major revision of Q2R2!")
                status = "FALSIFIED"
            else:
                print(f"  KILLER PREDICTION: Inconclusive (not significant)")
                status = "INCONCLUSIVE"
        
        # Linear correlation as backup test
        r_corr, p_corr = stats.pearsonr(env_norm, residuals)
        print(f"\n  Linear correlation: r = {r_corr:.4f}, p = {p_corr:.2e}")
        
        results = {
            'n_udg': len(residuals),
            'quadratic': {
                'a': float(a),
                'a_err': float(a_err),
                'b': float(b),
                'c': float(c),
                'significance': float(significance)
            },
            'linear': {
                'r': float(r_corr),
                'p': float(p_corr)
            },
            'prediction_status': status,
            'note': 'MOCK DATA - Results are for pipeline testing only'
        }
        
        return results
        
    except Exception as e:
        print(f"  ERROR in quadratic fit: {e}")
        return None


def main():
    print("="*70)
    print(" KILLER PREDICTION #1: ULTRA-DIFFUSE GALAXIES")
    print(" Testing Q2R2 prediction of inverted U-shape for UDGs")
    print("="*70)
    
    # Download/create UDG catalog
    udg_table = download_udg_catalog()
    
    if udg_table is None or len(udg_table) < 10:
        print("  ERROR: Insufficient UDG data")
        return
    
    print(f"\n  UDG sample: {len(udg_table)} galaxies")
    print(f"  Columns: {udg_table.colnames}")
    
    # Compute scaling relation residuals
    scaling_results = compute_udg_scaling_residuals(udg_table)
    
    if scaling_results is None:
        print("  ERROR: Could not compute scaling residuals")
        return
    
    # Test killer prediction
    results = test_ushape_killer_prediction(udg_table, scaling_results)
    
    # Summary
    print("\n" + "="*70)
    print(" KILLER PREDICTION #1 SUMMARY")
    print("="*70)
    
    if results:
        print(f"""
  Sample: {results['n_udg']} UDGs
  
  Quadratic coefficient: a = {results['quadratic']['a']:.4f} ± {results['quadratic']['a_err']:.4f}
  Significance: {results['quadratic']['significance']:.1f}σ
  
  PREDICTION STATUS: {results['prediction_status']}
  
  NOTE: {results['note']}
        """)
        
        # Save results
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "killer_prediction_1_udg.json")
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  Results saved: {output_path}")
    
    print("="*70)


if __name__ == "__main__":
    main()
