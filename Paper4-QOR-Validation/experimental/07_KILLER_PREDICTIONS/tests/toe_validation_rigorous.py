#!/usr/bin/env python3
"""
===============================================================================
QO+R THEORY OF EVERYTHING - RIGOROUS VALIDATION
===============================================================================

IMPORTANT: All criteria are defined A PRIORI based on:
1. Theoretical predictions from the QO+R framework
2. Published literature values
3. Standard statistical thresholds

NO POST-HOC ADJUSTMENTS. If a test fails, it fails.

===============================================================================
PRE-REGISTERED CRITERIA (defined before seeing results)
===============================================================================

TEST 1: λ_QR DETECTION
  - Criterion: λ_QR significantly different from 0 at 3σ level
  - Rationale: Standard physics discovery threshold
  - PASS if: |λ_QR / σ_λ| > 3.0

TEST 2: λ_QR UNIVERSALITY
  - Criterion: λ_QR varies by less than factor of 3 across mass bins
  - Rationale: A "universal" constant shouldn't vary by orders of magnitude
  - PASS if: max(λ) / min(λ) < 3.0 AND both bins have same sign

TEST 3: Q-R ANTICORRELATION  
  - Criterion: Pearson r < -0.5
  - Rationale: Mass conservation (Q + R ≈ constant) implies strong anticorrelation
  - PASS if: r < -0.5

TEST 4: BTFR SLOPE (MOND LIMIT)
  - Criterion: Slope within 0.20 to 0.30 (MOND predicts exactly 0.25)
  - Rationale: McGaugh+2016, Lelli+2017 measure slopes 0.22-0.28 for SPARC
  - Literature: Lelli+2017 SPARC paper reports 0.265 ± 0.01
  - PASS if: 0.20 ≤ slope ≤ 0.30

TEST 5: STRING THEORY λ_QR RANGE
  - Criterion: 0.1 < |λ_QR| < 100
  - Rationale: KKLT predicts O(1) coupling, allowing 2 orders of magnitude margin
  - PASS if: 0.1 < |λ_QR| < 100

TEST 6: MODEL COMPARISON (AIC)
  - Criterion: QO+R model has lower AIC than linear model by > 2
  - Rationale: ΔAIC > 2 is standard threshold for model preference
  - PASS if: AIC_linear - AIC_QOR > 2

===============================================================================
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
# PRE-REGISTERED CRITERIA (DO NOT MODIFY AFTER SEEING RESULTS)
# =============================================================================

CRITERIA = {
    'lambda_detection_sigma': 3.0,      # 3σ detection threshold
    'universality_ratio_max': 3.0,      # Max ratio between mass bins
    'qr_correlation_threshold': -0.5,   # Anticorrelation threshold
    'btfr_slope_min': 0.20,             # MOND slope range
    'btfr_slope_max': 0.30,             # MOND slope range
    'lambda_range_min': 0.1,            # String theory range
    'lambda_range_max': 100.0,          # String theory range
    'aic_difference_threshold': 2.0,    # Model comparison threshold
}

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SPARC_PATH = os.path.join(BASE_DIR, "..", "Paper1-BTFR-UShape", "data", "sparc_with_environment.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "experimental", "07_KILLER_PREDICTIONS", "results")


def load_sparc_data():
    """Load SPARC dataset."""
    df = pd.read_csv(SPARC_PATH)
    
    df['log_Vflat'] = np.log10(df['Vflat'])
    df['log_Mbar'] = np.log10(df['M_bar'] * 1e10)
    df['log_Mstar'] = np.log10(df['M_star'] * 1e10)
    df['log_MHI'] = np.log10(df['MHI'] * 1e10)
    df['f_gas'] = df['MHI'] / df['M_bar']
    df['Q'] = np.log10(df['MHI'] / df['M_bar'])
    df['R'] = np.log10(df['M_star'] / df['M_bar'])
    
    # BTFR prediction
    btfr_slope = 0.25
    df['log_V_btfr'] = btfr_slope * df['log_Mbar']
    offset = np.median(df['log_Vflat'] - df['log_V_btfr'])
    df['log_V_btfr'] += offset
    df['residual'] = df['log_Vflat'] - df['log_V_btfr']
    
    return df


def qor_model(X, beta_Q, beta_R, lambda_QR):
    """QO+R model: Δlog(V) = β_Q × Q + β_R × R + λ_QR × Q × R"""
    Q, R = X
    return beta_Q * Q + beta_R * R + lambda_QR * Q * R


def linear_model(X, beta_Q, beta_R):
    """Linear model: Δlog(V) = β_Q × Q + β_R × R"""
    Q, R = X
    return beta_Q * Q + beta_R * R


def run_test(name, condition, pass_value, fail_value, criterion_desc):
    """Run a single test and return result."""
    passed = condition
    return {
        'name': name,
        'criterion': criterion_desc,
        'value': pass_value if passed else fail_value,
        'passed': bool(passed)
    }


def main():
    print("="*70)
    print(" QO+R TOE - RIGOROUS VALIDATION (PRE-REGISTERED CRITERIA)")
    print(" Author: Jonathan Édouard Slama")
    print(" Metafund Research Division, Strasbourg, France")
    print("="*70)
    
    # Print criteria FIRST
    print("\n" + "-"*70)
    print(" PRE-REGISTERED CRITERIA (defined before analysis)")
    print("-"*70)
    print(f"  1. λ_QR detection:     |λ/σ| > {CRITERIA['lambda_detection_sigma']}")
    print(f"  2. Universality:       max/min ratio < {CRITERIA['universality_ratio_max']}")
    print(f"  3. Q-R anticorrelation: r < {CRITERIA['qr_correlation_threshold']}")
    print(f"  4. BTFR slope:         {CRITERIA['btfr_slope_min']} ≤ slope ≤ {CRITERIA['btfr_slope_max']}")
    print(f"  5. String theory λ:    {CRITERIA['lambda_range_min']} < |λ| < {CRITERIA['lambda_range_max']}")
    print(f"  6. Model comparison:   ΔAIC > {CRITERIA['aic_difference_threshold']}")
    
    # Load data
    print("\n" + "-"*70)
    print(" LOADING DATA")
    print("-"*70)
    df = load_sparc_data()
    print(f"  N = {len(df)} galaxies")
    
    # Prepare data
    Q = df['Q'].values
    R = df['R'].values
    residual = df['residual'].values
    valid = np.isfinite(Q) & np.isfinite(R) & np.isfinite(residual)
    Q, R, residual = Q[valid], R[valid], residual[valid]
    n = len(Q)
    
    results = {'criteria': CRITERIA, 'n_galaxies': n, 'tests': []}
    
    # ==========================================================================
    # TEST 1: λ_QR DETECTION
    # ==========================================================================
    print("\n" + "-"*70)
    print(" TEST 1: λ_QR DETECTION")
    print("-"*70)
    
    popt_qor, pcov_qor = curve_fit(qor_model, (Q, R), residual, p0=[0, 0, 1])
    perr_qor = np.sqrt(np.diag(pcov_qor))
    lambda_QR = popt_qor[2]
    lambda_err = perr_qor[2]
    lambda_sigma = abs(lambda_QR / lambda_err)
    
    print(f"  λ_QR = {lambda_QR:.3f} ± {lambda_err:.3f}")
    print(f"  Significance = {lambda_sigma:.2f}σ")
    print(f"  Criterion: > {CRITERIA['lambda_detection_sigma']}σ")
    
    test1_pass = lambda_sigma > CRITERIA['lambda_detection_sigma']
    print(f"  Result: {'PASS ✓' if test1_pass else 'FAIL ✗'}")
    
    results['tests'].append({
        'name': 'λ_QR detection',
        'lambda_QR': float(lambda_QR),
        'lambda_err': float(lambda_err),
        'sigma': float(lambda_sigma),
        'criterion': f"> {CRITERIA['lambda_detection_sigma']}σ",
        'passed': test1_pass
    })
    
    # ==========================================================================
    # TEST 2: UNIVERSALITY
    # ==========================================================================
    print("\n" + "-"*70)
    print(" TEST 2: λ_QR UNIVERSALITY ACROSS MASS SCALES")
    print("-"*70)
    
    mass_bins = [(7.0, 9.0, "Low/Int"), (9.0, 12.0, "High")]
    lambda_values = []
    
    for m_min, m_max, label in mass_bins:
        mask = (df['log_Mbar'] >= m_min) & (df['log_Mbar'] < m_max)
        df_bin = df[mask]
        Q_bin = df_bin['Q'].values
        R_bin = df_bin['R'].values
        res_bin = df_bin['residual'].values
        v = np.isfinite(Q_bin) & np.isfinite(R_bin) & np.isfinite(res_bin)
        
        if np.sum(v) >= 20:
            try:
                popt, pcov = curve_fit(qor_model, (Q_bin[v], R_bin[v]), res_bin[v], p0=[0, 0, 1])
                perr = np.sqrt(np.diag(pcov))
                print(f"  {label} (N={np.sum(v)}): λ_QR = {popt[2]:.3f} ± {perr[2]:.3f}")
                lambda_values.append(popt[2])
            except:
                print(f"  {label}: Fit failed")
        else:
            print(f"  {label}: Insufficient data (N={np.sum(v)})")
    
    if len(lambda_values) >= 2 and all(l > 0 for l in lambda_values):
        ratio = max(lambda_values) / min(lambda_values)
        test2_pass = ratio < CRITERIA['universality_ratio_max']
        print(f"  Ratio max/min = {ratio:.2f}")
        print(f"  Criterion: < {CRITERIA['universality_ratio_max']}")
        print(f"  Result: {'PASS ✓' if test2_pass else 'FAIL ✗'}")
    elif len(lambda_values) >= 2:
        # Different signs
        test2_pass = False
        print(f"  λ values have different signs: {lambda_values}")
        print(f"  Result: FAIL ✗ (sign inconsistency)")
        ratio = np.nan
    else:
        test2_pass = False
        ratio = np.nan
        print(f"  Insufficient bins for comparison")
        print(f"  Result: FAIL ✗ (insufficient data)")
    
    results['tests'].append({
        'name': 'Universality',
        'lambda_values': [float(l) for l in lambda_values],
        'ratio': float(ratio) if np.isfinite(ratio) else None,
        'criterion': f"ratio < {CRITERIA['universality_ratio_max']}",
        'passed': test2_pass
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
        'criterion': f"r < {CRITERIA['qr_correlation_threshold']}",
        'passed': test3_pass
    })
    
    # ==========================================================================
    # TEST 4: BTFR SLOPE
    # ==========================================================================
    print("\n" + "-"*70)
    print(" TEST 4: BTFR SLOPE (MOND LIMIT)")
    print("-"*70)
    
    slope, intercept, r, p, se = stats.linregress(df['log_Mbar'], df['log_Vflat'])
    print(f"  Measured slope = {slope:.4f} ± {se:.4f}")
    print(f"  Criterion: {CRITERIA['btfr_slope_min']} ≤ slope ≤ {CRITERIA['btfr_slope_max']}")
    
    test4_pass = CRITERIA['btfr_slope_min'] <= slope <= CRITERIA['btfr_slope_max']
    print(f"  Result: {'PASS ✓' if test4_pass else 'FAIL ✗'}")
    
    results['tests'].append({
        'name': 'BTFR slope',
        'slope': float(slope),
        'slope_err': float(se),
        'criterion': f"{CRITERIA['btfr_slope_min']} ≤ slope ≤ {CRITERIA['btfr_slope_max']}",
        'passed': test4_pass
    })
    
    # ==========================================================================
    # TEST 5: STRING THEORY RANGE
    # ==========================================================================
    print("\n" + "-"*70)
    print(" TEST 5: STRING THEORY λ_QR RANGE")
    print("-"*70)
    
    print(f"  |λ_QR| = {abs(lambda_QR):.3f}")
    print(f"  Criterion: {CRITERIA['lambda_range_min']} < |λ| < {CRITERIA['lambda_range_max']}")
    
    test5_pass = CRITERIA['lambda_range_min'] < abs(lambda_QR) < CRITERIA['lambda_range_max']
    print(f"  Result: {'PASS ✓' if test5_pass else 'FAIL ✗'}")
    
    results['tests'].append({
        'name': 'String theory range',
        'lambda_abs': float(abs(lambda_QR)),
        'criterion': f"{CRITERIA['lambda_range_min']} < |λ| < {CRITERIA['lambda_range_max']}",
        'passed': test5_pass
    })
    
    # ==========================================================================
    # TEST 6: MODEL COMPARISON
    # ==========================================================================
    print("\n" + "-"*70)
    print(" TEST 6: MODEL COMPARISON (AIC)")
    print("-"*70)
    
    # Linear model
    popt_lin, pcov_lin = curve_fit(linear_model, (Q, R), residual, p0=[0, 0])
    pred_lin = linear_model((Q, R), *popt_lin)
    rss_lin = np.sum((residual - pred_lin)**2)
    aic_lin = n * np.log(rss_lin / n) + 2 * 2  # 2 parameters
    
    # QO+R model
    pred_qor = qor_model((Q, R), *popt_qor)
    rss_qor = np.sum((residual - pred_qor)**2)
    aic_qor = n * np.log(rss_qor / n) + 2 * 3  # 3 parameters
    
    delta_aic = aic_lin - aic_qor  # Positive if QO+R is better
    
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
        'criterion': f"ΔAIC > {CRITERIA['aic_difference_threshold']}",
        'passed': test6_pass
    })
    
    # ==========================================================================
    # FINAL VERDICT
    # ==========================================================================
    print("\n" + "="*70)
    print(" FINAL VERDICT")
    print("="*70)
    
    all_tests = results['tests']
    n_passed = sum(1 for t in all_tests if t['passed'])
    n_total = len(all_tests)
    
    print(f"\n  {'Test':<30} {'Result':<10}")
    print("  " + "-"*45)
    for t in all_tests:
        symbol = "✓ PASS" if t['passed'] else "✗ FAIL"
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
    
    # Convert for JSON
    def convert(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "toe_validation_rigorous.json")
    
    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
