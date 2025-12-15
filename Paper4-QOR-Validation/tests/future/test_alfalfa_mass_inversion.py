#!/usr/bin/env python3
"""
===============================================================================
MASS-BASED SIGN INVERSION TEST FOR ALFALFA
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
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "alfalfa")
OUTPUT_DIR = os.path.join(BASE_DIR, "experimental", "04_RESULTS", "test_results")


def load_alfalfa_data():
    """Load ALFALFA data."""
    
    print("  Loading ALFALFA data...")
    
    data_path = os.path.join(DATA_DIR, "alfalfa_a100.fits")
    
    if not os.path.exists(data_path):
        data_path = os.path.join(DATA_DIR, "alfalfa_a100_t0.fits")
    
    if not os.path.exists(data_path):
        print(f"  ERROR: ALFALFA data not found")
        return None
    
    print(f"  Found: {data_path}")
    
    t = Table.read(data_path)
    print(f"  Total rows: {len(t)}")
    
    # Convert sexagesimal RA/Dec to degrees
    print("  Converting coordinates...")
    ra_str = [str(x) for x in t['RAJ2000']]
    dec_str = [str(x) for x in t['DEJ2000']]
    
    # Parse coordinates
    coords = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
    ra = coords.ra.deg
    dec = coords.dec.deg
    
    # Extract numeric columns
    vhel = np.array(t['Vhel'], dtype=float)
    dist = np.array(t['Dist'], dtype=float)
    logMHI = np.array(t['logMHI'], dtype=float)
    W50 = np.array(t['W50'], dtype=float)
    
    print(f"  RA range: {np.nanmin(ra):.1f} - {np.nanmax(ra):.1f} deg")
    print(f"  Dist range: {np.nanmin(dist):.1f} - {np.nanmax(dist):.1f} Mpc")
    print(f"  logMHI range: {np.nanmin(logMHI):.2f} - {np.nanmax(logMHI):.2f}")
    
    # Quality cuts
    mask = (
        np.isfinite(ra) & np.isfinite(dec) &
        np.isfinite(dist) & (dist > 0) & (dist < 300) &
        np.isfinite(logMHI) & (logMHI > 6) & (logMHI < 12) &
        np.isfinite(W50) & (W50 > 20) & (W50 < 600)
    )
    
    print(f"  After quality cuts: {mask.sum()} galaxies")
    
    # Compute BTF quantities
    V = W50[mask] / 2.0
    logV = np.log10(V)
    
    # Baryonic mass: M_bary ~ 1.4 * M_HI (accounting for He)
    logMbary = logMHI[mask] + np.log10(1.4)
    
    data = {
        'ra': ra[mask],
        'dec': dec[mask],
        'dist': dist[mask],
        'logMHI': logMHI[mask],
        'logMbary': logMbary,
        'W50': W50[mask],
        'logV': logV,
    }
    
    print(f"  Baryonic mass range: {data['logMbary'].min():.2f} - {data['logMbary'].max():.2f}")
    
    return data


def compute_density_3d(ra, dec, dist, n_neighbors=10):
    """Compute 3D local density."""
    
    print("  Computing 3D density...")
    
    x = dist * np.cos(np.radians(dec)) * np.cos(np.radians(ra))
    y = dist * np.cos(np.radians(dec)) * np.sin(np.radians(ra))
    z = dist * np.sin(np.radians(dec))
    
    coords = np.column_stack([x, y, z])
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=n_neighbors+1)
    
    r_n = distances[:, -1]
    density = n_neighbors / (4/3 * np.pi * r_n**3 + 1e-10)
    
    return np.log10(density + 1e-10)


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


def test_mass_bins(data, density):
    """Test Q2R2 in different mass bins."""
    
    print("\n" + "="*70)
    print(" TEST: MASS-BASED SIGN INVERSION (ALFALFA)")
    print("="*70)
    
    # Compute BTF residuals
    slope, intercept, _, _, _ = stats.linregress(data['logMbary'], data['logV'])
    residuals = data['logV'] - (slope * data['logMbary'] + intercept)
    
    print(f"\n  BTF: log V = {slope:.3f} * log M_bary + {intercept:.3f}")
    
    # Mass bins
    mass_bins = [
        (7.0, 8.0, "Dwarf"),
        (8.0, 8.5, "Low-mass"),
        (8.5, 9.5, "TRANSITION"),
        (9.5, 10.5, "Intermediate"),
        (10.5, 12.0, "Massive"),
    ]
    
    print(f"\n  {'Mass bin':<20} {'Type':<15} {'N':>8} {'a':>12} {'+/-err':>10} {'sigma':>8} {'Result':>12}")
    print("-"*90)
    
    results = []
    
    for m_min, m_max, label in mass_bins:
        mask = (data['logMbary'] >= m_min) & (data['logMbary'] < m_max)
        n = mask.sum()
        
        if n > 50:
            result = test_quadratic(residuals[mask], density[mask])
            if result:
                if result['a'] > 0 and result['sig'] > 2:
                    shape = "U-shape"
                elif result['a'] < 0 and result['sig'] > 2:
                    shape = "INVERTED"
                else:
                    shape = "flat"
                
                print(f"  [{m_min:.1f}, {m_max:.1f}] {label:<15} {n:>8,} {result['a']:>12.5f} {result['a_err']:>10.5f} {result['sig']:>8.1f} {shape:>12}")
                
                results.append({
                    'm_min': m_min, 'm_max': m_max, 'label': label, 'n': n,
                    'a': result['a'], 'a_err': result['a_err'], 'sig': result['sig'],
                    'shape': shape
                })
        else:
            print(f"  [{m_min:.1f}, {m_max:.1f}] {label:<15} {n:>8,} -- insufficient data --")
    
    return results


def test_gas_fraction_bins(data, density):
    """Test using gas fraction as Q proxy."""
    
    print("\n" + "="*70)
    print(" TEST: GAS FRACTION SIGN INVERSION (ALFALFA)")
    print("="*70)
    
    slope, intercept, _, _, _ = stats.linregress(data['logMbary'], data['logV'])
    residuals = data['logV'] - (slope * data['logMbary'] + intercept)
    
    log_fgas = data['logMHI'] - data['logMbary']
    
    fgas_percentiles = [0, 20, 40, 60, 80, 100]
    fgas_bins = np.percentile(log_fgas, fgas_percentiles)
    
    print(f"\n  {'f_gas bin':<25} {'N':>8} {'a':>12} {'+/-err':>10} {'sigma':>8} {'Result':>12}")
    print("-"*80)
    
    results = []
    
    for i in range(len(fgas_bins) - 1):
        mask = (log_fgas >= fgas_bins[i]) & (log_fgas < fgas_bins[i+1])
        n = mask.sum()
        
        if n > 50:
            result = test_quadratic(residuals[mask], density[mask])
            if result:
                if result['a'] > 0 and result['sig'] > 2:
                    shape = "U-shape"
                elif result['a'] < 0 and result['sig'] > 2:
                    shape = "INVERTED"
                else:
                    shape = "flat"
                
                label = f"[{fgas_bins[i]:.2f}, {fgas_bins[i+1]:.2f}]"
                print(f"  {label:<25} {n:>8,} {result['a']:>12.5f} {result['a_err']:>10.5f} {result['sig']:>8.1f} {shape:>12}")
                
                results.append({
                    'fgas_min': float(fgas_bins[i]), 
                    'fgas_max': float(fgas_bins[i+1]), 
                    'n': n,
                    'a': result['a'], 'a_err': result['a_err'], 'sig': result['sig'],
                    'shape': shape
                })
    
    return results


def main():
    print("="*70)
    print(" ALFALFA MASS-BASED SIGN INVERSION TEST")
    print(" Confirming KiDS discovery")
    print("="*70)
    
    data = load_alfalfa_data()
    
    if data is None:
        print("\n  ERROR: Could not load data")
        return
    
    density = compute_density_3d(data['ra'], data['dec'], data['dist'])
    
    results = {}
    results['mass_bins'] = test_mass_bins(data, density)
    results['gas_fraction'] = test_gas_fraction_bins(data, density)
    
    # Summary
    print("\n" + "="*70)
    print(" COMPARISON WITH KiDS")
    print("="*70)
    print("""
  KiDS Mass Bin Results (878,670 galaxies, STELLAR mass):
  
  Mass bin          a           sigma    Result
  -----------------------------------------------
  [7.0, 8.5]       +1.531       8.3     U-shape
  [8.5, 9.5]       -6.024      56.6     INVERTED
  [9.5, 10.5]      +0.280       5.4     U-shape
  [10.5, 11.5]     +0.748      23.0     U-shape
    """)
    
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    # Check transition bin
    transition_bin = None
    for r in results['mass_bins']:
        if r['m_min'] == 8.5 and r['m_max'] == 9.5:
            transition_bin = r
            break
    
    if transition_bin:
        print(f"""
  TRANSITION BIN [8.5, 9.5] (HI mass):
    a = {transition_bin['a']:.5f} ({transition_bin['sig']:.1f} sigma)
    Result: {transition_bin['shape']}
    
  Note: ALFALFA uses HI mass, KiDS uses stellar mass.
  The transition mass may differ between these tracers.
        """)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "alfalfa_mass_inversion_results.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"  Results saved: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
