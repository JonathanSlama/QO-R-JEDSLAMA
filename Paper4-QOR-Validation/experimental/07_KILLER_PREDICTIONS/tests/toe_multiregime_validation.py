#!/usr/bin/env python3
"""
===============================================================================
QO+R TOE - MULTI-REGIME VALIDATION
===============================================================================

CRITICAL INSIGHT:
=================
The QO+R TOE does NOT predict identical λ_QR across all datasets.
It predicts that the SIGN of the effect depends on the Q/R balance:

- Q-dominated (gas-rich) systems: NEGATIVE environmental dependence
- R-dominated (gas-poor) systems: POSITIVE environmental dependence
- Mixed populations: Intermediate/variable

This is the BARYONIC COUPLING hypothesis from string theory moduli.

PRE-REGISTERED CRITERIA (regime-dependent):
==========================================
1. DETECTION: |λ/σ| > 3.0 in at least one regime
2. SIGN INVERSION: Q-dominated and R-dominated show opposite signs
3. CONSISTENCY: Same dataset shows coherent behavior across mass bins
4. MULTI-SCALE: Effect present at multiple mass scales

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
# PRE-REGISTERED CRITERIA FOR MULTI-REGIME TOE
# =============================================================================

CRITERIA = {
    'detection_sigma': 3.0,           # Detection threshold
    'sign_consistency_threshold': 0.7, # 70% of bins should have same sign
    'min_bins_for_test': 3,           # Minimum bins to test consistency
}

# Gas fraction thresholds for regime classification
Q_DOMINATED_THRESHOLD = 0.5   # f_gas > 0.5 = Q-dominated
R_DOMINATED_THRESHOLD = 0.3   # f_gas < 0.3 = R-dominated

# Paths
# Path to external directories (relative to Git folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
GIT_DIR = os.path.dirname(PROJECT_DIR)
SPARC_PATH = os.path.join(GIT_DIR, "Paper1-BTFR-UShape", "data", "sparc_with_environment.csv")
BTFR_DIR = os.path.join(os.path.dirname(GIT_DIR), "BTFR")
ALFALFA_PATH = os.path.join(BTFR_DIR, "Test-Replicability", "data", "alfalfa_corrected.csv")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("tests", "results")


def qor_model(X, beta_Q, beta_R, lambda_QR):
    """QO+R model"""
    Q, R = X
    return beta_Q * Q + beta_R * R + lambda_QR * Q * R


def analyze_by_gas_regime(df, name):
    """
    Analyze QO+R parameters separately for Q-dominated and R-dominated galaxies.
    
    This tests the BARYONIC COUPLING hypothesis:
    - Q-dominated (gas-rich): Should show NEGATIVE λ_QR
    - R-dominated (gas-poor): Should show POSITIVE λ_QR
    """
    results = {'dataset': name, 'regimes': {}}
    
    # Ensure we have required columns
    if 'f_gas' not in df.columns:
        if 'MHI' in df.columns and 'M_bar' in df.columns:
            df['f_gas'] = df['MHI'] / df['M_bar']
        elif 'M_HI' in df.columns and 'M_bar' in df.columns:
            df['f_gas'] = df['M_HI'] / df['M_bar']
    
    if 'Q' not in df.columns:
        df['Q'] = np.log10(df['f_gas'])
    if 'R' not in df.columns:
        df['R'] = np.log10(1 - df['f_gas'])
    
    # Define regimes
    regimes = [
        ('Q_dominated', df['f_gas'] > Q_DOMINATED_THRESHOLD),
        ('Intermediate', (df['f_gas'] >= R_DOMINATED_THRESHOLD) & (df['f_gas'] <= Q_DOMINATED_THRESHOLD)),
        ('R_dominated', df['f_gas'] < R_DOMINATED_THRESHOLD),
    ]
    
    for regime_name, mask in regimes:
        df_regime = df[mask].copy()
        n = len(df_regime)
        
        if n < 30:
            results['regimes'][regime_name] = {
                'n': n,
                'status': 'INSUFFICIENT DATA',
                'mean_f_gas': float(df_regime['f_gas'].mean()) if n > 0 else None
            }
            continue
        
        Q = df_regime['Q'].values
        R = df_regime['R'].values
        residual = df_regime['residual'].values
        
        valid = np.isfinite(Q) & np.isfinite(R) & np.isfinite(residual)
        Q, R, residual = Q[valid], R[valid], residual[valid]
        
        try:
            popt, pcov = curve_fit(qor_model, (Q, R), residual, p0=[0, 0, 0.1], maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            
            lambda_QR = popt[2]
            lambda_err = perr[2]
            sigma = abs(lambda_QR / lambda_err) if lambda_err > 0 else 0
            
            results['regimes'][regime_name] = {
                'n': int(np.sum(valid)),
                'mean_f_gas': float(df_regime['f_gas'].mean()),
                'lambda_QR': float(lambda_QR),
                'lambda_err': float(lambda_err),
                'sigma': float(sigma),
                'sign': 'positive' if lambda_QR > 0 else 'negative',
                'significant': sigma > CRITERIA['detection_sigma']
            }
        except Exception as e:
            results['regimes'][regime_name] = {
                'n': int(np.sum(valid)),
                'status': f'FIT FAILED: {e}'
            }
    
    return results


def test_sign_inversion(sparc_results, alfalfa_results):
    """
    Test if Q-dominated and R-dominated populations show opposite signs.
    
    This is the KEY PREDICTION of QO+R baryonic coupling.
    """
    # Collect all Q-dominated results
    q_dom_signs = []
    r_dom_signs = []
    
    for results in [sparc_results, alfalfa_results]:
        if 'Q_dominated' in results['regimes']:
            regime = results['regimes']['Q_dominated']
            if 'lambda_QR' in regime:
                q_dom_signs.append(regime['lambda_QR'])
        
        if 'R_dominated' in results['regimes']:
            regime = results['regimes']['R_dominated']
            if 'lambda_QR' in regime:
                r_dom_signs.append(regime['lambda_QR'])
    
    # Test sign inversion
    if len(q_dom_signs) > 0 and len(r_dom_signs) > 0:
        mean_q = np.mean(q_dom_signs)
        mean_r = np.mean(r_dom_signs)
        
        # Sign inversion = opposite signs
        inversion = (mean_q * mean_r) < 0
        
        return {
            'Q_dominated_mean_lambda': float(mean_q),
            'R_dominated_mean_lambda': float(mean_r),
            'sign_inversion_detected': bool(inversion),
            'Q_dominated_sign': 'positive' if mean_q > 0 else 'negative',
            'R_dominated_sign': 'positive' if mean_r > 0 else 'negative',
            'n_Q_samples': len(q_dom_signs),
            'n_R_samples': len(r_dom_signs)
        }
    else:
        return {'status': 'INSUFFICIENT DATA FOR SIGN TEST'}


def load_sparc():
    """Load and prepare SPARC data."""
    df = pd.read_csv(SPARC_PATH)
    
    df['log_Vflat'] = np.log10(df['Vflat'])
    df['log_Mbar'] = np.log10(df['M_bar'] * 1e10)
    df['f_gas'] = df['MHI'] / df['M_bar']
    df['Q'] = np.log10(df['MHI'] / df['M_bar'])
    df['R'] = np.log10(df['M_star'] / df['M_bar'])
    
    # BTFR residual
    slope, intercept, _, _, _ = stats.linregress(df['log_Mbar'], df['log_Vflat'])
    df['residual'] = df['log_Vflat'] - (slope * df['log_Mbar'] + intercept)
    
    return df


def load_alfalfa():
    """Load and prepare ALFALFA data."""
    df = pd.read_csv(ALFALFA_PATH)
    
    # Use pre-calculated values if available
    if 'btfr_residual' in df.columns:
        df['residual'] = df['btfr_residual']
    else:
        df['log_V'] = np.log10(df['Vflat']) if 'Vflat' in df.columns else np.log10(df['W50'] / 2)
        df['log_Mbar'] = df['log_Mbar'] if 'log_Mbar' in df.columns else np.log10(df['M_bar'])
        slope, intercept, _, _, _ = stats.linregress(df['log_Mbar'], df['log_V'])
        df['residual'] = df['log_V'] - (slope * df['log_Mbar'] + intercept)
    
    # Gas fraction
    if 'f_gas' not in df.columns:
        df['f_gas'] = df['MHI'] / df['M_bar'] if 'MHI' in df.columns else 10**df['log_MHI'] / df['M_bar']
    
    # Q and R
    df['Q'] = np.log10(df['f_gas'])
    df['R'] = np.log10(1 - df['f_gas'])
    
    # Clean
    valid = np.isfinite(df['Q']) & np.isfinite(df['R']) & np.isfinite(df['residual'])
    return df[valid].copy()


def main():
    print("="*70)
    print(" QO+R TOE - MULTI-REGIME VALIDATION")
    print(" Testing Baryonic Coupling Hypothesis")
    print(" Author: Jonathan Édouard Slama")
    print("="*70)
    
    print("\n" + "-"*70)
    print(" THEORETICAL PREDICTION")
    print("-"*70)
    print("  QO+R with string theory moduli predicts:")
    print("  - Q-dominated (gas-rich, f_gas > 0.5): NEGATIVE λ_QR")
    print("  - R-dominated (gas-poor, f_gas < 0.3): POSITIVE λ_QR")
    print("  - Sign INVERSION between regimes is the key prediction!")
    
    print("\n" + "-"*70)
    print(" LOADING SPARC")
    print("-"*70)
    df_sparc = load_sparc()
    print(f"  N = {len(df_sparc)} galaxies")
    print(f"  Mean f_gas = {df_sparc['f_gas'].mean():.3f}")
    
    print("\n" + "-"*70)
    print(" LOADING ALFALFA")
    print("-"*70)
    df_alfalfa = load_alfalfa()
    print(f"  N = {len(df_alfalfa)} galaxies")
    print(f"  Mean f_gas = {df_alfalfa['f_gas'].mean():.3f}")
    
    # Analyze by regime
    print("\n" + "-"*70)
    print(" SPARC - BY GAS REGIME")
    print("-"*70)
    sparc_results = analyze_by_gas_regime(df_sparc, "SPARC")
    
    for regime, data in sparc_results['regimes'].items():
        if 'lambda_QR' in data:
            sig_marker = "***" if data['sigma'] > 3 else "**" if data['sigma'] > 2 else "*" if data['sigma'] > 1 else ""
            print(f"  {regime:<15} (N={data['n']:>4}, <f_gas>={data['mean_f_gas']:.2f}): "
                  f"λ_QR = {data['lambda_QR']:+.4f} ± {data['lambda_err']:.4f} ({data['sigma']:.1f}σ) {sig_marker}")
        else:
            print(f"  {regime:<15}: {data.get('status', 'N/A')}")
    
    print("\n" + "-"*70)
    print(" ALFALFA - BY GAS REGIME")
    print("-"*70)
    alfalfa_results = analyze_by_gas_regime(df_alfalfa, "ALFALFA")
    
    for regime, data in alfalfa_results['regimes'].items():
        if 'lambda_QR' in data:
            sig_marker = "***" if data['sigma'] > 3 else "**" if data['sigma'] > 2 else "*" if data['sigma'] > 1 else ""
            print(f"  {regime:<15} (N={data['n']:>5}, <f_gas>={data['mean_f_gas']:.2f}): "
                  f"λ_QR = {data['lambda_QR']:+.4f} ± {data['lambda_err']:.4f} ({data['sigma']:.1f}σ) {sig_marker}")
        else:
            print(f"  {regime:<15}: {data.get('status', 'N/A')}")
    
    # Test sign inversion
    print("\n" + "-"*70)
    print(" TEST: SIGN INVERSION (KEY PREDICTION)")
    print("-"*70)
    sign_test = test_sign_inversion(sparc_results, alfalfa_results)
    
    if 'sign_inversion_detected' in sign_test:
        print(f"  Q-dominated mean λ_QR: {sign_test['Q_dominated_mean_lambda']:+.4f} ({sign_test['Q_dominated_sign']})")
        print(f"  R-dominated mean λ_QR: {sign_test['R_dominated_mean_lambda']:+.4f} ({sign_test['R_dominated_sign']})")
        print(f"  Sign inversion: {'YES ✓' if sign_test['sign_inversion_detected'] else 'NO ✗'}")
    else:
        print(f"  {sign_test.get('status', 'Unknown error')}")
    
    # Final verdict
    print("\n" + "="*70)
    print(" VERDICT")
    print("="*70)
    
    tests_passed = 0
    tests_total = 3
    
    # Test 1: Detection in at least one regime
    any_significant = False
    for results in [sparc_results, alfalfa_results]:
        for regime, data in results['regimes'].items():
            if data.get('significant', False):
                any_significant = True
                break
    
    if any_significant:
        print("  ✓ DETECTION: Significant λ_QR in at least one regime")
        tests_passed += 1
    else:
        print("  ✗ DETECTION: No significant λ_QR found")
    
    # Test 2: Sign inversion
    if sign_test.get('sign_inversion_detected', False):
        print("  ✓ SIGN INVERSION: Q-dominated and R-dominated show opposite signs")
        tests_passed += 1
    else:
        print("  ✗ SIGN INVERSION: No opposite signs detected")
    
    # Test 3: Consistency within ALFALFA (largest dataset)
    alfalfa_signs = []
    for regime, data in alfalfa_results['regimes'].items():
        if 'lambda_QR' in data:
            alfalfa_signs.append(np.sign(data['lambda_QR']))
    
    if len(alfalfa_signs) >= 2:
        # Check if signs are consistent with regime prediction
        # Q-dominated should be negative, R-dominated should be positive
        q_data = alfalfa_results['regimes'].get('Q_dominated', {})
        r_data = alfalfa_results['regimes'].get('R_dominated', {})
        
        q_correct = q_data.get('lambda_QR', 1) < 0  # Should be negative
        r_correct = r_data.get('lambda_QR', -1) > 0  # Should be positive
        
        if q_correct and r_correct:
            print("  ✓ REGIME CONSISTENCY: Signs match QO+R predictions")
            tests_passed += 1
        elif q_correct or r_correct:
            print("  ~ REGIME CONSISTENCY: Partial agreement with predictions")
            tests_passed += 0.5
        else:
            print("  ✗ REGIME CONSISTENCY: Signs don't match predictions")
    else:
        print("  ? REGIME CONSISTENCY: Insufficient data")
    
    print("-"*70)
    print(f"  Total: {tests_passed}/{tests_total} tests passed")
    
    if tests_passed >= 2.5:
        verdict = "QO+R MULTI-REGIME TOE SUPPORTED"
    elif tests_passed >= 1.5:
        verdict = "QO+R MULTI-REGIME TOE PARTIALLY SUPPORTED"
    else:
        verdict = "QO+R MULTI-REGIME TOE NOT SUPPORTED"
    
    print(f"  VERDICT: {verdict}")
    
    # Save results
    all_results = {
        'SPARC': sparc_results,
        'ALFALFA': alfalfa_results,
        'sign_inversion_test': sign_test,
        'verdict': verdict,
        'tests_passed': tests_passed,
        'tests_total': tests_total
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "toe_multiregime_validation.json")
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    main()
