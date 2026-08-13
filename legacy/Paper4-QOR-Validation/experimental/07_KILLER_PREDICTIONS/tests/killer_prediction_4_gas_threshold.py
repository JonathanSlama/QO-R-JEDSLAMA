#!/usr/bin/env python3
"""
===============================================================================
KILLER PREDICTION #4: GAS FRACTION THRESHOLD TEST
===============================================================================

Tests the QO+R prediction that there exists a critical gas fraction (f_crit)
where the system transitions from Q-dominated to R-dominated behavior.

THEORETICAL BASIS:
==================
- Q (gas) and R (stars) emit oscillations at different phases
- When f_gas > f_crit: Q dominates → U-shape (a > 0)
- When f_gas < f_crit: R dominates → inverted U-shape (a < 0)
- QO+R predicts: f_crit ~ 0.3 (from empirical ALFALFA/KiDS transition)

DATA:
=====
- ALFALFA survey (gas-rich galaxies with HI measurements)
- Cross-matched with stellar mass from SDSS
- Gas fraction: f_gas = M_HI / (M_HI + M_star)

PREDICTION:
===========
- U-shape coefficient 'a' should correlate with gas fraction
- Sign transition near f_crit ~ 0.3
- a(f_gas > 0.3) > 0, a(f_gas < 0.3) < 0

FALSIFICATION:
=============
If f_crit < 0.05 or f_crit > 0.9, QO+R is falsified (unphysical threshold).

Author: Jonathan Édouard Slama
Affiliation: Metafund Research Division, Strasbourg, France
Date: December 2025
===============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit, minimize_scalar
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "experimental", "07_KILLER_PREDICTIONS", "results")

# Try to find ALFALFA data
ALFALFA_PATHS = [
    os.path.join(DATA_DIR, "alfalfa", "alfalfa_crossmatch.fits"),
    os.path.join(DATA_DIR, "level1_ushape", "alfalfa_crossmatch.fits"),
    os.path.join(DATA_DIR, "a100_sdss_crossmatch.fits"),
]


def find_alfalfa_data():
    """Find ALFALFA data file."""
    for path in ALFALFA_PATHS:
        if os.path.exists(path):
            return path
    
    # Search recursively
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if 'alfalfa' in f.lower() or 'a100' in f.lower():
                if f.endswith('.fits') or f.endswith('.csv'):
                    return os.path.join(root, f)
    
    return None


def load_alfalfa_data(path):
    """Load ALFALFA data with gas fractions."""
    from astropy.io import fits
    from astropy.table import Table
    
    print(f"Loading: {path}")
    
    if path.endswith('.fits'):
        t = Table.read(path)
        df = t.to_pandas()
    else:
        df = pd.read_csv(path)
    
    print(f"Loaded {len(df)} objects")
    print(f"Columns: {df.columns.tolist()[:15]}...")
    
    return df


def compute_gas_fraction(df):
    """
    Compute gas fraction f_gas = M_HI / (M_HI + M_star).
    
    Handles different column naming conventions.
    """
    # Find HI mass column
    hi_cols = [c for c in df.columns if 'hi' in c.lower() and 'mass' in c.lower()]
    hi_cols += [c for c in df.columns if c in ['logMHI', 'logM_HI', 'MHI', 'HIflux']]
    
    # Find stellar mass column
    star_cols = [c for c in df.columns if 'star' in c.lower() and 'mass' in c.lower()]
    star_cols += [c for c in df.columns if c in ['logMstar', 'logM_star', 'Mstar', 'logmass']]
    
    print(f"HI mass candidates: {hi_cols}")
    print(f"Stellar mass candidates: {star_cols}")
    
    if not hi_cols or not star_cols:
        # Try to compute from available columns
        if 'W50' in df.columns and 'Dist' in df.columns:
            # Compute HI mass from velocity width and distance
            # M_HI = 2.36e5 * D^2 * S_int (for S_int in Jy km/s)
            print("Computing HI mass from W50 and distance...")
        return None, None
    
    hi_col = hi_cols[0]
    star_col = star_cols[0]
    
    # Get masses (convert from log if needed)
    if 'log' in hi_col.lower():
        M_HI = 10**df[hi_col]
    else:
        M_HI = df[hi_col]
    
    if 'log' in star_col.lower():
        M_star = 10**df[star_col]
    else:
        M_star = df[star_col]
    
    # Compute gas fraction
    f_gas = M_HI / (M_HI + M_star)
    
    return f_gas, df


def quadratic_model(x, a, b, c):
    """Quadratic model: y = a*x² + b*x + c"""
    return a * x**2 + b * x + c


def compute_ushape_coefficient(env, residual):
    """Compute U-shape coefficient 'a' for a subsample."""
    if len(env) < 10:
        return np.nan, np.nan
    
    try:
        popt, pcov = curve_fit(quadratic_model, env, residual, p0=[0, 0, 0])
        perr = np.sqrt(np.diag(pcov))
        return popt[0], perr[0]
    except:
        return np.nan, np.nan


def find_critical_threshold(f_gas, a_values, a_errors):
    """
    Find the critical gas fraction where 'a' changes sign.
    
    Uses interpolation to find f_crit where a(f_crit) = 0.
    """
    valid = np.isfinite(a_values) & np.isfinite(f_gas)
    f = f_gas[valid]
    a = a_values[valid]
    
    if len(f) < 5:
        return None, None
    
    # Sort by gas fraction
    idx = np.argsort(f)
    f = f[idx]
    a = a[idx]
    
    # Find sign changes
    sign_changes = np.where(np.diff(np.sign(a)) != 0)[0]
    
    if len(sign_changes) == 0:
        return None, "No sign change detected"
    
    # Linear interpolation to find crossing point
    i = sign_changes[0]
    f_crit = f[i] - a[i] * (f[i+1] - f[i]) / (a[i+1] - a[i])
    
    return f_crit, None


def run_binned_analysis(df, f_gas, env_col, residual_col, n_bins=5):
    """
    Bin galaxies by gas fraction and compute U-shape coefficient for each bin.
    """
    # Create gas fraction bins
    f_min, f_max = f_gas.min(), f_gas.max()
    bins = np.linspace(f_min, f_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    a_values = []
    a_errors = []
    n_per_bin = []
    
    for i in range(n_bins):
        mask = (f_gas >= bins[i]) & (f_gas < bins[i+1])
        if i == n_bins - 1:  # Include upper edge in last bin
            mask = (f_gas >= bins[i]) & (f_gas <= bins[i+1])
        
        env = df.loc[mask, env_col].values
        res = df.loc[mask, residual_col].values
        
        a, a_err = compute_ushape_coefficient(env, res)
        a_values.append(a)
        a_errors.append(a_err)
        n_per_bin.append(np.sum(mask))
    
    return bin_centers, np.array(a_values), np.array(a_errors), n_per_bin


def create_mock_test():
    """
    Create a mock test with simulated data demonstrating the expected behavior.
    
    This shows what we EXPECT to find when proper data is available.
    """
    print("\n" + "="*70)
    print(" MOCK TEST DEMONSTRATION")
    print(" (Simulated data showing expected QO+R behavior)")
    print("="*70)
    
    np.random.seed(42)
    
    # Simulate 1000 galaxies
    N = 1000
    
    # Gas fractions from 0.05 to 0.95
    f_gas = np.random.uniform(0.05, 0.95, N)
    
    # Environment (1-5 scale)
    env = np.random.uniform(1, 5, N)
    
    # QO+R prediction: a(f_gas) transitions from positive to negative
    # at f_crit ~ 0.3
    f_crit_true = 0.3
    
    # U-shape coefficient varies with gas fraction
    a_of_f = 2.0 * (f_gas - f_crit_true)  # Positive for f > 0.3, negative for f < 0.3
    
    # Generate residuals with U-shape modulated by gas fraction
    residual = np.zeros(N)
    for i in range(N):
        a_i = a_of_f[i]
        # Residual = a*env² + noise
        env_centered = env[i] - 3  # Center on middle environment
        residual[i] = a_i * env_centered**2 + np.random.normal(0, 0.2)
    
    # Bin by gas fraction and measure U-shape coefficient
    n_bins = 8
    bin_edges = np.linspace(0.05, 0.95, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    a_measured = []
    a_errors = []
    
    for i in range(n_bins):
        mask = (f_gas >= bin_edges[i]) & (f_gas < bin_edges[i+1])
        if np.sum(mask) > 20:
            env_bin = env[mask] - 3
            res_bin = residual[mask]
            
            try:
                popt, pcov = curve_fit(quadratic_model, env_bin, res_bin, p0=[0, 0, 0])
                a_measured.append(popt[0])
                a_errors.append(np.sqrt(pcov[0, 0]))
            except:
                a_measured.append(np.nan)
                a_errors.append(np.nan)
        else:
            a_measured.append(np.nan)
            a_errors.append(np.nan)
    
    a_measured = np.array(a_measured)
    a_errors = np.array(a_errors)
    
    # Find threshold
    valid = np.isfinite(a_measured)
    sign_changes = np.where(np.diff(np.sign(a_measured[valid])) != 0)[0]
    
    if len(sign_changes) > 0:
        i = sign_changes[0]
        f_crit_measured = bin_centers[valid][i] - a_measured[valid][i] * \
                         (bin_centers[valid][i+1] - bin_centers[valid][i]) / \
                         (a_measured[valid][i+1] - a_measured[valid][i])
    else:
        f_crit_measured = np.nan
    
    print(f"\nSimulated data: N = {N} galaxies")
    print(f"True f_crit = {f_crit_true:.2f}")
    print(f"\nMeasured U-shape coefficient by gas fraction bin:")
    print("-" * 50)
    print(f"{'f_gas bin':<15} {'a coefficient':<20} {'Status'}")
    print("-" * 50)
    
    for i, (f, a, err) in enumerate(zip(bin_centers, a_measured, a_errors)):
        if np.isfinite(a):
            status = "U-shape (Q-dom)" if a > 0 else "Inverted U (R-dom)"
            print(f"{f:.2f}            {a:+.3f} ± {err:.3f}       {status}")
    
    print("-" * 50)
    print(f"\nRecovered f_crit = {f_crit_measured:.2f} (true = {f_crit_true:.2f})")
    print(f"Error: {abs(f_crit_measured - f_crit_true):.3f}")
    
    return {
        'mock_test': True,
        'n_galaxies': N,
        'f_crit_true': float(f_crit_true),
        'f_crit_measured': float(f_crit_measured),
        'bin_centers': bin_centers.tolist(),
        'a_values': [float(x) if np.isfinite(x) else None for x in a_measured],
        'a_errors': [float(x) if np.isfinite(x) else None for x in a_errors],
        'status': 'DEMONSTRATION SUCCESSFUL'
    }


def main():
    print("="*70)
    print(" KILLER PREDICTION #4: GAS FRACTION THRESHOLD TEST")
    print(" Author: Jonathan Édouard Slama")
    print(" Metafund Research Division, Strasbourg, France")
    print("="*70)
    
    print("\n" + "-"*70)
    print(" THEORETICAL PREDICTION")
    print("-"*70)
    print("""
    QO+R Framework Prediction:
    - Q (gas) and R (stars) emit at different phases
    - Gas fraction determines which component dominates
    - Critical threshold: f_crit ~ 0.3
    
    Expected behavior:
    - f_gas > f_crit: Q-dominated → U-shape (a > 0)
    - f_gas < f_crit: R-dominated → inverted U-shape (a < 0)
    
    Falsification: f_crit < 0.05 or f_crit > 0.9 (unphysical)
    """)
    
    # Try to find real ALFALFA data
    print("\n" + "-"*70)
    print(" SEARCHING FOR ALFALFA DATA")
    print("-"*70)
    
    alfalfa_path = find_alfalfa_data()
    
    results = {}
    
    if alfalfa_path:
        print(f"Found: {alfalfa_path}")
        
        try:
            df = load_alfalfa_data(alfalfa_path)
            f_gas, df_clean = compute_gas_fraction(df)
            
            if f_gas is not None:
                print(f"\nGas fraction range: {f_gas.min():.2f} - {f_gas.max():.2f}")
                print(f"Mean gas fraction: {f_gas.mean():.2f}")
                
                # Would need environment and residual columns
                # This depends on the specific data structure
                
                results['data_found'] = True
                results['n_galaxies'] = len(df)
                results['f_gas_range'] = [float(f_gas.min()), float(f_gas.max())]
                results['status'] = 'DATA LOADED - ANALYSIS PENDING'
            else:
                print("Could not compute gas fractions from available columns")
                results['data_found'] = True
                results['status'] = 'INSUFFICIENT COLUMNS FOR GAS FRACTION'
                
        except Exception as e:
            print(f"Error loading data: {e}")
            results['data_found'] = False
            results['error'] = str(e)
    else:
        print("ALFALFA data not found in data directory")
        print("Searched paths:")
        for p in ALFALFA_PATHS:
            print(f"  - {p}")
        results['data_found'] = False
        results['status'] = 'DATA NOT FOUND'
    
    # Run mock demonstration
    mock_results = create_mock_test()
    results['mock_demonstration'] = mock_results
    
    # Final assessment
    print("\n" + "="*70)
    print(" ASSESSMENT")
    print("="*70)
    
    if results.get('data_found') and 'f_crit_measured' in results:
        f_crit = results['f_crit_measured']
        if 0.05 < f_crit < 0.9:
            verdict = "CONSISTENT WITH QO+R"
            status = f"f_crit = {f_crit:.2f} is in physical range [0.05, 0.9]"
        else:
            verdict = "POTENTIAL FALSIFICATION"
            status = f"f_crit = {f_crit:.2f} is outside physical range"
    else:
        verdict = "PENDING - AWAITING DATA"
        status = "Mock test demonstrates expected behavior"
    
    print(f"\n  Verdict: {verdict}")
    print(f"  {status}")
    
    print("""
    
    NOTE: Full test requires:
    1. ALFALFA HI masses
    2. Cross-matched stellar masses (SDSS)
    3. Environmental density metrics
    4. Sufficient sample across gas fraction range
    
    The mock test demonstrates that IF f_crit ~ 0.3 exists,
    we can recover it from binned U-shape analysis.
    """)
    
    results['verdict'] = verdict
    results['verdict_status'] = status
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "killer_prediction_4_gas_threshold.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
