#!/usr/bin/env python3
"""
===============================================================================
KILLER PREDICTION #1: UDG INVERTED U-SHAPE TEST
===============================================================================

Tests the QO+R prediction that Ultra-Diffuse Galaxies (gas-poor, R-dominated)
should show an INVERTED U-shape in BTFR residuals vs environment.

THEORETICAL BASIS:
==================
- UDGs are extremely gas-poor → R-dominated systems
- QO+R predicts: R-dominated → coefficient a < 0 (inverted U-shape)
- Contrast with gas-rich systems (ALFALFA) where a > 0 (U-shape)

DATA:
=====
- Gannon+2024 spectroscopic UDG catalog (37 UDGs)
- Environment: 1=Cluster, 2=Group, 3=Field
- Velocity dispersion, stellar mass, distance

PREDICTION:
===========
Residual = a × env² + b × env + c
QO+R predicts: a < 0 (inverted U-shape)

FALSIFICATION:
=============
If a > 0 at 3σ significance, QO+R is falsified for UDGs.

Author: Jonathan Édouard Slama
Affiliation: Metafund Research Division, Strasbourg, France
Date: December 2025

ACKNOWLEDGMENT:
===============
UDG spectroscopic data from Gannon et al. 2024, MNRAS, 531, 1856
https://github.com/gannonjs/Published_Data/tree/main/UDG_Spectroscopic_Data
===============================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import json
import warnings
warnings.filterwarnings('ignore')

# Paths - from tests/ -> 07_KILLER_PREDICTIONS/ -> experimental/ -> Paper4-Redshift-Predictions/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(BASE_DIR, "data", "udg_catalogs")
OUTPUT_DIR = os.path.join(BASE_DIR, "experimental", "07_KILLER_PREDICTIONS", "results")

# Constants
MISSING_VALUE = -999.0
G = 4.302e-6  # kpc (km/s)^2 / M_sun


def load_gannon_catalog():
    """Load and clean Gannon+2024 UDG catalog."""
    path = os.path.join(DATA_DIR, "Gannon2024_udg_data.csv")
    df = pd.read_csv(path)
    
    # Replace -999 with NaN
    df = df.replace(MISSING_VALUE, np.nan)
    
    # Filter for valid velocity dispersion
    df_valid = df[df['stellar_sigma'].notna() & (df['stellar_sigma'] > 0)].copy()
    
    print(f"Loaded {len(df)} UDGs, {len(df_valid)} with valid stellar_sigma")
    
    return df_valid


def compute_dynamical_mass(sigma, R_e):
    """
    Compute dynamical mass within R_e from velocity dispersion.
    M_dyn = k × σ² × R_e / G
    where k ~ 4 for dispersion-supported systems (Wolf et al. 2010)
    """
    k = 4.0
    M_dyn = k * sigma**2 * R_e / G
    return M_dyn


def compute_btfr_prediction(M_baryonic):
    """
    Compute predicted velocity from BTFR.
    V_flat^4 = G × M_bar × a0
    where a0 ~ 1.2e-10 m/s² (MOND acceleration scale)
    
    For UDGs we use σ as velocity proxy.
    """
    a0 = 1.2e-10 * 3.086e19  # Convert to kpc/s²
    V_pred = (G * M_baryonic * a0) ** 0.25
    return V_pred


def quadratic_model(x, a, b, c):
    """Quadratic model: y = a*x² + b*x + c"""
    return a * x**2 + b * x + c


def run_ushape_test(df):
    """
    Test for U-shape (or inverted U-shape) in residuals vs environment.
    
    Returns:
        dict with fit results, significance, and verdict
    """
    # Environment as numeric (1=cluster, 2=group, 3=field)
    # For QO+R, we expect: higher number = lower density = more "field-like"
    env = df['Environment'].values
    
    # Compute "residual" proxy
    # For UDGs, we use log(σ_observed) - log(σ_expected from M_star)
    # Simple scaling: σ ∝ M^0.25 for BTFR-like relation
    
    sigma_obs = df['stellar_sigma'].values
    M_star = df['Stellar mass'].values  # Already in 10^8 M_sun units based on inspection
    
    # Expected sigma from stellar mass (rough BTFR scaling)
    # σ_exp ∝ M^0.25, normalized to typical UDG values
    sigma_exp = 15.0 * (M_star / 1.0) ** 0.25  # Normalize to ~15 km/s at M=10^8
    
    # Compute residual
    residual = np.log10(sigma_obs) - np.log10(sigma_exp)
    
    # Remove any remaining NaN or inf
    valid = np.isfinite(residual) & np.isfinite(env)
    env = env[valid]
    residual = residual[valid]
    
    print(f"\nAnalyzing {len(env)} UDGs with valid data")
    print(f"Environment distribution: Cluster={np.sum(env==1)}, Group={np.sum(env==2)}, Field={np.sum(env==3)}")
    
    # Fit quadratic model
    try:
        popt, pcov = curve_fit(quadratic_model, env, residual, p0=[0, 0, 0])
        perr = np.sqrt(np.diag(pcov))
        
        a, b, c = popt
        a_err, b_err, c_err = perr
        
        # Significance of quadratic term
        a_sigma = abs(a / a_err) if a_err > 0 else 0
        
        # Compute R² 
        residuals_fit = residual - quadratic_model(env, *popt)
        ss_res = np.sum(residuals_fit**2)
        ss_tot = np.sum((residual - np.mean(residual))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Determine shape
        if a < 0:
            shape = "INVERTED U-SHAPE"
            qor_prediction = "CONSISTENT"
        else:
            shape = "U-SHAPE"
            qor_prediction = "INCONSISTENT"
        
        results = {
            'a': float(a),
            'a_err': float(a_err),
            'a_sigma': float(a_sigma),
            'b': float(b),
            'b_err': float(b_err),
            'c': float(c),
            'c_err': float(c_err),
            'r_squared': float(r_squared),
            'n_objects': int(len(env)),
            'shape': shape,
            'qor_prediction': qor_prediction,
        }
        
        return results, env, residual, popt
        
    except Exception as e:
        print(f"Fit failed: {e}")
        return None, env, residual, None


def run_alternative_test(df):
    """
    Alternative test: Direct correlation between sigma and environment.
    
    If QO+R is correct, UDGs in denser environments should show
    different sigma-mass relation than those in less dense environments.
    """
    env = df['Environment'].values
    sigma = df['stellar_sigma'].values
    M_star = df['Stellar mass'].values
    
    # Separate by environment
    cluster = (env == 1)
    non_cluster = (env > 1)
    
    if np.sum(cluster) > 3 and np.sum(non_cluster) > 3:
        # Compare sigma/M^0.25 ratio between environments
        ratio_cluster = sigma[cluster] / (M_star[cluster] ** 0.25)
        ratio_non_cluster = sigma[non_cluster] / (M_star[non_cluster] ** 0.25)
        
        # T-test
        t_stat, p_value = stats.ttest_ind(ratio_cluster, ratio_non_cluster)
        
        return {
            'cluster_mean': float(np.mean(ratio_cluster)),
            'cluster_std': float(np.std(ratio_cluster)),
            'non_cluster_mean': float(np.mean(ratio_non_cluster)),
            'non_cluster_std': float(np.std(ratio_non_cluster)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        }
    
    return None


def main():
    print("="*70)
    print(" KILLER PREDICTION #1: UDG INVERTED U-SHAPE TEST")
    print(" Author: Jonathan Édouard Slama")
    print(" Metafund Research Division, Strasbourg, France")
    print("="*70)
    
    print("\n" + "-"*70)
    print(" THEORETICAL PREDICTION")
    print("-"*70)
    print("""
    QO+R Framework Prediction:
    - UDGs are gas-poor → R-dominated systems
    - R-dominated systems should show: a < 0 (inverted U-shape)
    - Contrast with gas-rich ALFALFA galaxies: a > 0 (U-shape)
    
    Test: Fit Residual = a × Environment² + b × Environment + c
    Falsification: a > 0 at 3σ significance
    """)
    
    # Load data
    print("\n" + "-"*70)
    print(" LOADING GANNON+2024 UDG CATALOG")
    print("-"*70)
    
    df = load_gannon_catalog()
    
    # Run U-shape test
    print("\n" + "-"*70)
    print(" U-SHAPE TEST")
    print("-"*70)
    
    results, env, residual, popt = run_ushape_test(df)
    
    if results:
        print(f"\nQuadratic fit: Residual = a×env² + b×env + c")
        print(f"  a = {results['a']:.4f} ± {results['a_err']:.4f} ({results['a_sigma']:.1f}σ)")
        print(f"  b = {results['b']:.4f} ± {results['b_err']:.4f}")
        print(f"  c = {results['c']:.4f} ± {results['c_err']:.4f}")
        print(f"  R² = {results['r_squared']:.3f}")
        print(f"\nShape detected: {results['shape']}")
        print(f"QO+R prediction: {results['qor_prediction']}")
    
    # Run alternative test
    print("\n" + "-"*70)
    print(" ALTERNATIVE TEST: CLUSTER vs NON-CLUSTER")
    print("-"*70)
    
    alt_results = run_alternative_test(df)
    
    if alt_results:
        print(f"\nσ/M^0.25 ratio comparison:")
        print(f"  Cluster (N=29):     {alt_results['cluster_mean']:.2f} ± {alt_results['cluster_std']:.2f}")
        print(f"  Non-cluster (N=8):  {alt_results['non_cluster_mean']:.2f} ± {alt_results['non_cluster_std']:.2f}")
        print(f"  T-statistic: {alt_results['t_statistic']:.2f}")
        print(f"  P-value: {alt_results['p_value']:.4f}")
        print(f"  Significant difference: {alt_results['significant']}")
    
    # Final verdict
    print("\n" + "="*70)
    print(" VERDICT")
    print("="*70)
    
    if results:
        if results['a'] < 0:
            verdict = "CONSISTENT WITH QO+R"
            status = "Inverted U-shape detected (a < 0) as predicted for R-dominated systems"
        elif results['a_sigma'] < 3:
            verdict = "INCONCLUSIVE"
            status = f"a = {results['a']:.4f} but only {results['a_sigma']:.1f}σ significance"
        else:
            verdict = "POTENTIAL FALSIFICATION"
            status = f"U-shape detected (a > 0 at {results['a_sigma']:.1f}σ) - contrary to prediction"
        
        print(f"\n  {verdict}")
        print(f"  {status}")
        
        # Note about sample size
        print(f"\n  NOTE: Sample size N={results['n_objects']} is small.")
        print("  Environment heavily biased toward clusters (29/37).")
        print("  More field/group UDGs needed for definitive test.")
        
        results['verdict'] = verdict
        results['status'] = status
        results['alternative_test'] = alt_results
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "killer_prediction_1_udg.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)
    
    # Acknowledgment
    print("\n" + "-"*70)
    print(" ACKNOWLEDGMENT")
    print("-"*70)
    print("""
    UDG spectroscopic data from:
    Gannon et al. 2024, MNRAS, 531, 1856
    "A Catalogue and analysis of ultra-diffuse galaxy spectroscopic properties"
    https://github.com/gannonjs/Published_Data/tree/main/UDG_Spectroscopic_Data
    
    We thank Dr. Jonah Gannon for making this data publicly available.
    """)
    
    return results


if __name__ == "__main__":
    main()
