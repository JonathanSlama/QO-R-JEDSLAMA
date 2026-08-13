#!/usr/bin/env python3
"""
===============================================================================
QO+R TOE - RIGOROUS VALIDATION ON ALFALFA
===============================================================================

Same pre-registered criteria as SPARC test, applied to ALFALFA data.

DATA SOURCE:
============
- ALFALFA α.100 (Haynes et al. 2018) - HI masses from Arecibo radio telescope
- Durbala et al. (2020) - Stellar masses from SDSS photometry
- Cross-matched by AGC catalog number
- N ~ 19,000 galaxies

THIS IS 100% REAL OBSERVATIONAL DATA.

PRE-REGISTERED CRITERIA (identical to SPARC):
=============================================
1. λ_QR detection:     |λ/σ| > 3.0
2. Universality:       max/min ratio < 3.0
3. Q-R anticorrelation: r < -0.5
4. BTFR slope:         0.20 ≤ slope ≤ 0.30
5. String theory λ:    0.1 < |λ| < 100
6. Model comparison:   ΔAIC > 2.0

Author: Jonathan Édouard Slama
Affiliation: Metafund Research Division, Strasbourg, France
Date: December 2025
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

# =============================================================================
# PRE-REGISTERED CRITERIA (IDENTICAL TO SPARC - DO NOT MODIFY)
# =============================================================================

CRITERIA = {
    'lambda_detection_sigma': 3.0,
    'universality_ratio_max': 3.0,
    'qr_correlation_threshold': -0.5,
    'btfr_slope_min': 0.20,
    'btfr_slope_max': 0.30,
    'lambda_range_min': 0.1,
    'lambda_range_max': 100.0,
    'aic_difference_threshold': 2.0,
}

# =============================================================================
# PATHS
# =============================================================================

# Path to BTFR directory (relative to Git folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
GIT_DIR = os.path.dirname(PROJECT_DIR)
BTFR_DIR = os.path.join(os.path.dirname(GIT_DIR), "BTFR")
ALFALFA_PATH = os.path.join(BTFR_DIR, "Test-Replicability", "data", "alfalfa_corrected.csv")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("tests", "results")


def load_alfalfa_data():
    """Load ALFALFA cross-matched data."""
    df = pd.read_csv(ALFALFA_PATH)
    
    print(f"  Raw data: {len(df)} galaxies")
    print(f"  Columns: {df.columns.tolist()[:10]}...")
    
    # Check for required columns
    required_cols = ['logMHI', 'logMstar', 'Dist', 'W50']
    for col in required_cols:
        if col not in df.columns:
            # Try alternative names
            alt_names = {
                'logMHI': ['logM_HI', 'log_MHI', 'MHI'],
                'logMstar': ['logM_star', 'log_Mstar', 'Mstar', 'logMstarTaylor'],
                'Dist': ['Distance', 'dist', 'D'],
                'W50': ['W50_kms', 'linewidth']
            }
            for alt in alt_names.get(col, []):
                if alt in df.columns:
                    df[col] = df[alt]
                    break
    
    # Compute derived quantities
    # M_HI from log
    if 'logMHI' in df.columns:
        df['M_HI'] = 10**df['logMHI']
    
    # M_star from log
    if 'logMstar' in df.columns:
        df['M_star'] = 10**df['logMstar']
    elif 'logMstarTaylor' in df.columns:
        df['logMstar'] = df['logMstarTaylor']
        df['M_star'] = 10**df['logMstarTaylor']
    
    # Baryonic mass
    df['M_bar'] = df['M_HI'] + df['M_star']
    df['log_Mbar'] = np.log10(df['M_bar'])
    
    # Gas fraction and Q, R
    df['f_gas'] = df['M_HI'] / df['M_bar']
    df['Q'] = np.log10(df['M_HI'] / df['M_bar'])  # log(f_gas)
    df['R'] = np.log10(df['M_star'] / df['M_bar'])  # log(1 - f_gas)
    
    # Velocity from W50 (line width at 50% - proxy for rotation)
    # V_flat ≈ W50 / (2 * sin(i)), but without inclination we use W50/2
    if 'W50' in df.columns:
        df['V_proxy'] = df['W50'] / 2  # km/s
        df['log_V'] = np.log10(df['V_proxy'])
    
    # BTFR prediction
    btfr_slope = 0.25
    df['log_V_btfr'] = btfr_slope * df['log_Mbar']
    offset = np.nanmedian(df['log_V'] - df['log_V_btfr'])
    df['log_V_btfr'] += offset
    df['residual'] = df['log_V'] - df['log_V_btfr']
    
    # Clean data
    valid = (
        np.isfinite(df['Q']) & 
        np.isfinite(df['R']) & 
        np.isfinite(df['residual']) &
        np.isfinite(df['log_Mbar']) &
        (df['log_Mbar'] > 7) &  # Minimum mass cut
        (df['log_Mbar'] < 12)   # Maximum mass cut
    )
    
    df_clean = df[valid].copy()
    print(f"  After cleaning: {len(df_clean)} galaxies")
    
    return df_clean


def qor_model(X, beta_Q, beta_R, lambda_QR):
    """QO+R model"""
    Q, R = X
    return beta_Q * Q + beta_R * R + lambda_QR * Q * R


def linear_model(X, beta_Q, beta_R):
    """Linear model"""
    Q, R = X
    return beta_Q * Q + beta_R * R


def main():
    print("="*70)
    print(" QO+R TOE - RIGOROUS VALIDATION ON ALFALFA")
    print(" Author: Jonathan Édouard Slama")
    print(" Metafund Research Division, Strasbourg, France")
    print("="*70)
    
    print("\n" + "-"*70)
    print(" DATA SOURCE")
    print("-"*70)
    print("  ALFALFA α.100 (Haynes et al. 2018) - HI masses")
    print("  Durbala et al. (2020) - Stellar masses from SDSS")
    print("  THIS IS 100% REAL OBSERVATIONAL DATA")
    
    print("\n" + "-"*70)
    print(" PRE-REGISTERED CRITERIA (same as SPARC)")
    print("-"*70)
    print(f"  1. λ_QR detection:     |λ/σ| > {CRITERIA['lambda_detection_sigma']}")
    print(f"  2. Universality:       max/min ratio < {CRITERIA['universality_ratio_max']}")
    print(f"  3. Q-R anticorrelation: r < {CRITERIA['qr_correlation_threshold']}")
    print(f"  4. BTFR slope:         {CRITERIA['btfr_slope_min']} ≤ slope ≤ {CRITERIA['btfr_slope_max']}")
    print(f"  5. String theory λ:    {CRITERIA['lambda_range_min']} < |λ| < {CRITERIA['lambda_range_max']}")
    print(f"  6. Model comparison:   ΔAIC > {CRITERIA['aic_difference_threshold']}")
    
    print("\n" + "-"*70)
    print(" LOADING DATA")
    print("-"*70)
    
    try:
        df = load_alfalfa_data()
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        return None
    
    # Prepare arrays
    Q = df['Q'].values
    R = df['R'].values
    residual = df['residual'].values
    log_Mbar = df['log_Mbar'].values
    log_V = df['log_V'].values
    n = len(Q)
    
    results = {'criteria': CRITERIA, 'n_galaxies': n, 'dataset': 'ALFALFA', 'tests': []}
    
    # ==========================================================================
    # TEST 1: λ_QR DETECTION
    # ==========================================================================
    print("\n" + "-"*70)
    print(" TEST 1: λ_QR DETECTION")
    print("-"*70)
    
    try:
        popt_qor, pcov_qor = curve_fit(qor_model, (Q, R), residual, p0=[0, 0, 0.1], maxfev=10000)
        perr_qor = np.sqrt(np.diag(pcov_qor))
        lambda_QR = popt_qor[2]
        lambda_err = perr_qor[2]
        lambda_sigma = abs(lambda_QR / lambda_err) if lambda_err > 0 else 0
        
        print(f"  λ_QR = {lambda_QR:.6f} ± {lambda_err:.6f}")
        print(f"  Significance = {lambda_sigma:.2f}σ")
        print(f"  Criterion: > {CRITERIA['lambda_detection_sigma']}σ")
        
        test1_pass = lambda_sigma > CRITERIA['lambda_detection_sigma']
        print(f"  Result: {'PASS ✓' if test1_pass else 'FAIL ✗'}")
        
        results['tests'].append({
            'name': 'λ_QR detection',
            'lambda_QR': float(lambda_QR),
            'lambda_err': float(lambda_err),
            'sigma': float(lambda_sigma),
            'passed': bool(test1_pass)
        })
    except Exception as e:
        print(f"  FIT FAILED: {e}")
        test1_pass = False
        lambda_QR = np.nan
        results['tests'].append({'name': 'λ_QR detection', 'error': str(e), 'passed': False})
    
    # ==========================================================================
    # TEST 2: UNIVERSALITY
    # ==========================================================================
    print("\n" + "-"*70)
    print(" TEST 2: λ_QR UNIVERSALITY ACROSS MASS SCALES")
    print("-"*70)
    
    mass_bins = [
        (7.0, 9.0, "Low mass"),
        (9.0, 10.0, "Intermediate"),
        (10.0, 12.0, "High mass")
    ]
    lambda_values = []
    
    for m_min, m_max, label in mass_bins:
        mask = (log_Mbar >= m_min) & (log_Mbar < m_max)
        n_bin = np.sum(mask)
        
        if n_bin >= 100:
            try:
                popt, pcov = curve_fit(qor_model, (Q[mask], R[mask]), residual[mask], 
                                       p0=[0, 0, 0.1], maxfev=10000)
                perr = np.sqrt(np.diag(pcov))
                print(f"  {label} (N={n_bin}): λ_QR = {popt[2]:.6f} ± {perr[2]:.6f}")
                if popt[2] > 0:  # Only include positive values for ratio
                    lambda_values.append(popt[2])
            except:
                print(f"  {label} (N={n_bin}): Fit failed")
        else:
            print(f"  {label} (N={n_bin}): Insufficient data")
    
    if len(lambda_values) >= 2:
        ratio = max(lambda_values) / min(lambda_values)
        test2_pass = ratio < CRITERIA['universality_ratio_max']
        print(f"  Ratio max/min = {ratio:.2f}")
        print(f"  Criterion: < {CRITERIA['universality_ratio_max']}")
        print(f"  Result: {'PASS ✓' if test2_pass else 'FAIL ✗'}")
    else:
        test2_pass = False
        ratio = np.nan
        print(f"  Insufficient valid bins")
        print(f"  Result: FAIL ✗")
    
    results['tests'].append({
        'name': 'Universality',
        'lambda_values': [float(l) for l in lambda_values],
        'ratio': float(ratio) if np.isfinite(ratio) else None,
        'passed': bool(test2_pass)
    })
    
    # ==========================================================================
    # TEST 3: Q-R ANTICORRELATION
    # ==========================================================================
    print("\n" + "-"*70)
    print(" TEST 3: Q-R ANTICORRELATION")
    print("-"*70)
    
    correlation, p_value = stats.pearsonr(Q, R)
    print(f"  Correlation(Q, R) = {correlation:.3f}")
    print(f"  p-value = {p_value:.2e}")
    print(f"  Criterion: r < {CRITERIA['qr_correlation_threshold']}")
    
    test3_pass = correlation < CRITERIA['qr_correlation_threshold']
    print(f"  Result: {'PASS ✓' if test3_pass else 'FAIL ✗'}")
    
    results['tests'].append({
        'name': 'Q-R anticorrelation',
        'correlation': float(correlation),
        'p_value': float(p_value),
        'passed': bool(test3_pass)
    })
    
    # ==========================================================================
    # TEST 4: BTFR SLOPE
    # ==========================================================================
    print("\n" + "-"*70)
    print(" TEST 4: BTFR SLOPE (MOND LIMIT)")
    print("-"*70)
    
    slope, intercept, r, p, se = stats.linregress(log_Mbar, log_V)
    print(f"  Measured slope = {slope:.4f} ± {se:.4f}")
    print(f"  Criterion: {CRITERIA['btfr_slope_min']} ≤ slope ≤ {CRITERIA['btfr_slope_max']}")
    
    test4_pass = CRITERIA['btfr_slope_min'] <= slope <= CRITERIA['btfr_slope_max']
    print(f"  Result: {'PASS ✓' if test4_pass else 'FAIL ✗'}")
    
    results['tests'].append({
        'name': 'BTFR slope',
        'slope': float(slope),
        'slope_err': float(se),
        'passed': bool(test4_pass)
    })
    
    # ==========================================================================
    # TEST 5: STRING THEORY RANGE
    # ==========================================================================
    print("\n" + "-"*70)
    print(" TEST 5: STRING THEORY λ_QR RANGE")
    print("-"*70)
    
    if np.isfinite(lambda_QR):
        print(f"  |λ_QR| = {abs(lambda_QR):.6f}")
        print(f"  Criterion: {CRITERIA['lambda_range_min']} < |λ| < {CRITERIA['lambda_range_max']}")
        
        test5_pass = CRITERIA['lambda_range_min'] < abs(lambda_QR) < CRITERIA['lambda_range_max']
        print(f"  Result: {'PASS ✓' if test5_pass else 'FAIL ✗'}")
    else:
        test5_pass = False
        print(f"  λ_QR not available")
        print(f"  Result: FAIL ✗")
    
    results['tests'].append({
        'name': 'String theory range',
        'lambda_abs': float(abs(lambda_QR)) if np.isfinite(lambda_QR) else None,
        'passed': bool(test5_pass)
    })
    
    # ==========================================================================
    # TEST 6: MODEL COMPARISON
    # ==========================================================================
    print("\n" + "-"*70)
    print(" TEST 6: MODEL COMPARISON (AIC)")
    print("-"*70)
    
    try:
        # Linear model
        popt_lin, _ = curve_fit(linear_model, (Q, R), residual, p0=[0, 0], maxfev=10000)
        pred_lin = linear_model((Q, R), *popt_lin)
        rss_lin = np.sum((residual - pred_lin)**2)
        aic_lin = n * np.log(rss_lin / n) + 2 * 2
        
        # QO+R model
        pred_qor = qor_model((Q, R), *popt_qor)
        rss_qor = np.sum((residual - pred_qor)**2)
        aic_qor = n * np.log(rss_qor / n) + 2 * 3
        
        delta_aic = aic_lin - aic_qor
        
        print(f"  AIC (Linear) = {aic_lin:.1f}")
        print(f"  AIC (QO+R)   = {aic_qor:.1f}")
        print(f"  ΔAIC = {delta_aic:.1f}")
        print(f"  Criterion: ΔAIC > {CRITERIA['aic_difference_threshold']}")
        
        test6_pass = delta_aic > CRITERIA['aic_difference_threshold']
        print(f"  Result: {'PASS ✓' if test6_pass else 'FAIL ✗'}")
        
        results['tests'].append({
            'name': 'Model comparison',
            'aic_linear': float(aic_lin),
            'aic_qor': float(aic_qor),
            'delta_aic': float(delta_aic),
            'passed': bool(test6_pass)
        })
    except Exception as e:
        print(f"  ERROR: {e}")
        test6_pass = False
        results['tests'].append({'name': 'Model comparison', 'error': str(e), 'passed': False})
    
    # ==========================================================================
    # FINAL VERDICT
    # ==========================================================================
    print("\n" + "="*70)
    print(" FINAL VERDICT")
    print("="*70)
    
    all_tests = results['tests']
    n_passed = sum(1 for t in all_tests if t.get('passed', False))
    n_total = len(all_tests)
    
    print(f"\n  {'Test':<30} {'Result':<10}")
    print("  " + "-"*45)
    for t in all_tests:
        symbol = "✓ PASS" if t.get('passed', False) else "✗ FAIL"
        print(f"  {t['name']:<30} {symbol}")
    
    print("  " + "-"*45)
    print(f"  Total: {n_passed}/{n_total} tests passed")
    
    if n_passed == n_total:
        verdict = "ALL TESTS PASSED - QO+R TOE STRONGLY SUPPORTED"
    elif n_passed >= n_total - 1:
        verdict = "MOST TESTS PASSED - QO+R TOE SUPPORTED WITH CAVEATS"
    elif n_passed >= n_total // 2:
        verdict = "MIXED RESULTS - QO+R TOE PARTIALLY SUPPORTED"
    else:
        verdict = "MAJORITY FAILED - QO+R TOE NOT SUPPORTED BY THIS DATA"
    
    print(f"\n  VERDICT: {verdict}")
    
    results['summary'] = {
        'tests_passed': n_passed,
        'tests_total': n_total,
        'verdict': verdict
    }
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "toe_validation_alfalfa.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
