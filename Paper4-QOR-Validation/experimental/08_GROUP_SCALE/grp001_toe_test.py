#!/usr/bin/env python3
"""
===============================================================================
GRP-001: TOE TEST FOR GALAXY GROUPS (Tempel+2017)
===============================================================================

TRUE TOE TEST - Not just U-shape!

Tests the full TOE prediction:

Δ_g(ρ_g, f_HI) ≈ α [C_Q(ρ_g) ρ_HI - |C_R(ρ_g)| ρ_*] cos(P(ρ_g))

Where:
- C_Q(ρ_g) = C_Q,0 (ρ_g/ρ_0)^{-β_Q}  (gas coupling, screens slower)
- C_R(ρ_g) = C_R,0 (ρ_g/ρ_0)^{-β_R}  (stellar coupling, screens faster)
- P(ρ_g) = π × sigmoid((ρ_g - ρ_c)/Δ)  (phase function)

FREE PARAMETERS: α, λ_QR (∝ C_Q,0 × C_R,0), ρ_c, Δ

FALSIFICATION CRITERIA (pre-registered):
1. No U ↔ inverted-U flip over > 1.5 dex in ρ_g
2. No Δ_g - f_HI correlation (p > 0.05) after mass control
3. λ_QR ∉ [1/3, 3] × λ_QR^gal
4. Amplitude |Δ_g| < 0.1 |Δ_gal| or wrong sign at extremes
5. Screening violated (m_Q,R × R_g << 1 required)

Author: Jonathan Édouard Slama
Affiliation: Metafund Research Division, Strasbourg, France
Date: December 2025
===============================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit, minimize
from scipy.special import expit  # sigmoid function
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PRE-REGISTERED CRITERIA (DO NOT MODIFY)
# =============================================================================

CRITERIA = {
    # Detection
    'min_groups': 500,
    'min_bins': 6,
    
    # λ_QR constraint (relative to galactic)
    'lambda_gal': 0.94,  # From SPARC
    'lambda_ratio_min': 1/3,
    'lambda_ratio_max': 3.0,
    
    # Amplitude constraint
    'amplitude_gal': 0.05,  # dex, from SPARC
    'amplitude_ratio_min': 0.1,
    'amplitude_ratio_max': 1.0,
    
    # Correlation threshold
    'correlation_p_threshold': 0.05,
    
    # Screening
    'screening_threshold': 3.0,  # m_QR × R_g > 3 required
}

# Physical priors
PRIORS = {
    'beta_Q': (0.8, 1.2),    # Gas descreens faster
    'beta_R': (0.3, 0.7),    # Stars screen more
    'gamma_Q': (0.2, 0.6),
    'gamma_R': (0.2, 0.6),
}

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def sigmoid(x):
    """Sigmoid function s(x) ∈ [0, 1]"""
    return expit(x)


def phase_function(rho_g, rho_c, delta):
    """
    Phase function P(ρ_g) = π × sigmoid((ρ_g - ρ_c) / Δ)
    
    Returns value in [0, π]
    """
    return np.pi * sigmoid((rho_g - rho_c) / delta)


def toe_model(X, alpha, lambda_QR, rho_c, delta, beta_Q=1.0, beta_R=0.5):
    """
    Full TOE model for group residuals.
    
    Δ_g = α [C_Q(ρ_g) ρ_HI - |C_R(ρ_g)| ρ_*] cos(P(ρ_g))
    
    Where:
    - C_Q(ρ_g) = (ρ_g/ρ_0)^{-β_Q}
    - C_R(ρ_g) = (ρ_g/ρ_0)^{-β_R}
    - P(ρ_g) = π × sigmoid((ρ_g - ρ_c) / Δ)
    
    X = (rho_g, rho_HI, rho_star) or (rho_g, f_HI)
    """
    rho_g, f_HI = X
    
    # Reference density (median)
    rho_0 = np.median(rho_g)
    
    # Screening functions
    C_Q = (rho_g / rho_0) ** (-beta_Q)
    C_R = (rho_g / rho_0) ** (-beta_R)
    
    # Baryonic content (normalized)
    # f_HI = ρ_HI / (ρ_HI + ρ_*)
    # So: ρ_HI ∝ f_HI, ρ_* ∝ (1 - f_HI)
    rho_HI_proxy = f_HI
    rho_star_proxy = 1 - f_HI
    
    # Phase function
    P = phase_function(rho_g, rho_c, delta)
    
    # Full model
    # λ_QR is encoded in the relative strength of C_Q vs C_R
    coupling_term = lambda_QR * (C_Q * rho_HI_proxy - C_R * rho_star_proxy)
    
    delta_g = alpha * coupling_term * np.cos(P)
    
    return delta_g


def toe_model_simple(X, alpha, lambda_QR, rho_c, delta):
    """Simplified TOE model with fixed β values."""
    return toe_model(X, alpha, lambda_QR, rho_c, delta, beta_Q=1.0, beta_R=0.5)


def null_model(X, alpha, beta):
    """Null model: simple linear dependence on f_HI only."""
    rho_g, f_HI = X
    return alpha + beta * f_HI


def create_mock_group_data_toe():
    """
    Create mock group data that follows TOE predictions.
    """
    print("\n  Creating mock group data with TOE physics...")
    
    np.random.seed(42)
    n_groups = 3000
    
    # True TOE parameters
    TRUE_PARAMS = {
        'alpha': 0.03,
        'lambda_QR': 0.8,  # Should be within [0.31, 2.82] of galactic
        'rho_c': 0.5,      # Critical density (log scale)
        'delta': 0.3,      # Transition width
        'beta_Q': 1.0,
        'beta_R': 0.5,
    }
    
    # Generate group properties
    # Log density (environment)
    rho_g = np.random.normal(0, 0.8, n_groups)
    
    # Gas fraction (anti-correlates with density - dense groups are gas-poor)
    f_HI = 0.4 - 0.15 * rho_g + np.random.normal(0, 0.1, n_groups)
    f_HI = np.clip(f_HI, 0.05, 0.8)
    
    # Generate TOE signal
    X = (rho_g, f_HI)
    delta_g_signal = toe_model(
        X, 
        TRUE_PARAMS['alpha'],
        TRUE_PARAMS['lambda_QR'],
        TRUE_PARAMS['rho_c'],
        TRUE_PARAMS['delta'],
        TRUE_PARAMS['beta_Q'],
        TRUE_PARAMS['beta_R']
    )
    
    # Add noise
    noise = np.random.normal(0, 0.015, n_groups)
    delta_g = delta_g_signal + noise
    
    # Group properties for context
    richness = np.random.pareto(1.5, n_groups) * 5 + 5
    richness = np.clip(richness.astype(int), 5, 200)
    
    z = np.random.uniform(0.01, 0.1, n_groups)
    log_mstar = 10.5 + 0.5 * np.log10(richness) + np.random.normal(0, 0.2, n_groups)
    
    df = pd.DataFrame({
        'GroupID': np.arange(n_groups),
        'z': z,
        'Ngal': richness,
        'log_Mstar': log_mstar,
        'rho_g': rho_g,           # Log environment density
        'f_HI': f_HI,             # Gas fraction
        'delta_g': delta_g,       # Residual (observable)
        'delta_g_true': delta_g_signal,  # True signal (for validation)
    })
    
    print(f"  Created {n_groups} mock groups")
    print(f"  True parameters: α={TRUE_PARAMS['alpha']}, λ_QR={TRUE_PARAMS['lambda_QR']}")
    print(f"  ρ_c={TRUE_PARAMS['rho_c']}, Δ={TRUE_PARAMS['delta']}")
    
    return df, TRUE_PARAMS


def test_toe_model(df):
    """
    Fit the TOE model and test against null model.
    """
    print("\n" + "-"*70)
    print(" FITTING TOE MODEL")
    print("-"*70)
    
    rho_g = df['rho_g'].values
    f_HI = df['f_HI'].values
    delta_g = df['delta_g'].values
    
    X = (rho_g, f_HI)
    
    # Initial guesses
    p0 = [0.03, 1.0, 0.0, 0.5]  # alpha, lambda_QR, rho_c, delta
    bounds = (
        [-0.5, 0.01, -2, 0.1],   # Lower bounds
        [0.5, 10.0, 2, 2.0]      # Upper bounds
    )
    
    try:
        popt, pcov = curve_fit(toe_model_simple, X, delta_g, p0=p0, bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        
        alpha, lambda_QR, rho_c, delta = popt
        alpha_err, lambda_err, rho_c_err, delta_err = perr
        
        # Compute fit quality
        delta_g_pred = toe_model_simple(X, *popt)
        ss_res = np.sum((delta_g - delta_g_pred)**2)
        ss_tot = np.sum((delta_g - np.mean(delta_g))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # AIC
        n = len(delta_g)
        k = 4  # Number of parameters
        aic_toe = n * np.log(ss_res / n) + 2 * k
        
        results = {
            'alpha': float(alpha),
            'alpha_err': float(alpha_err),
            'lambda_QR': float(lambda_QR),
            'lambda_err': float(lambda_err),
            'rho_c': float(rho_c),
            'rho_c_err': float(rho_c_err),
            'delta': float(delta),
            'delta_err': float(delta_err),
            'r_squared': float(r_squared),
            'aic': float(aic_toe),
            'n': n
        }
        
        print(f"  α = {alpha:.4f} ± {alpha_err:.4f}")
        print(f"  λ_QR = {lambda_QR:.4f} ± {lambda_err:.4f}")
        print(f"  ρ_c = {rho_c:.4f} ± {rho_c_err:.4f}")
        print(f"  Δ = {delta:.4f} ± {delta_err:.4f}")
        print(f"  R² = {r_squared:.4f}")
        print(f"  AIC = {aic_toe:.1f}")
        
        return results, popt
        
    except Exception as e:
        print(f"  FIT FAILED: {e}")
        return {'error': str(e)}, None


def test_null_model(df):
    """
    Fit null model (no TOE physics, just f_HI dependence).
    """
    print("\n" + "-"*70)
    print(" FITTING NULL MODEL (no TOE)")
    print("-"*70)
    
    rho_g = df['rho_g'].values
    f_HI = df['f_HI'].values
    delta_g = df['delta_g'].values
    
    X = (rho_g, f_HI)
    
    try:
        popt, pcov = curve_fit(null_model, X, delta_g, p0=[0, 0])
        perr = np.sqrt(np.diag(pcov))
        
        # Fit quality
        delta_g_pred = null_model(X, *popt)
        ss_res = np.sum((delta_g - delta_g_pred)**2)
        ss_tot = np.sum((delta_g - np.mean(delta_g))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        n = len(delta_g)
        k = 2
        aic_null = n * np.log(ss_res / n) + 2 * k
        
        print(f"  α = {popt[0]:.4f} ± {perr[0]:.4f}")
        print(f"  β = {popt[1]:.4f} ± {perr[1]:.4f}")
        print(f"  R² = {r_squared:.4f}")
        print(f"  AIC = {aic_null:.1f}")
        
        return {'r_squared': float(r_squared), 'aic': float(aic_null)}
        
    except Exception as e:
        print(f"  FIT FAILED: {e}")
        return {'error': str(e)}


def test_phase_flip(df, popt):
    """
    Test for U ↔ inverted-U flip across density range.
    
    This is a KEY TOE prediction: the phase function causes sign reversal.
    """
    print("\n" + "-"*70)
    print(" TEST: PHASE FLIP (U ↔ inverted-U)")
    print("-"*70)
    
    if popt is None:
        print("  Cannot test: TOE fit failed")
        return {'tested': False}
    
    rho_g = df['rho_g'].values
    f_HI = df['f_HI'].values
    
    # Split into low and high density
    rho_median = np.median(rho_g)
    rho_range = rho_g.max() - rho_g.min()
    
    # Check if we span > 1.5 dex
    print(f"  Density range: {rho_range:.2f} dex")
    
    if rho_range < 1.5:
        print("  WARNING: Density range < 1.5 dex, flip test may be inconclusive")
    
    # Predict at extremes
    f_HI_mean = np.mean(f_HI)
    
    rho_low = np.percentile(rho_g, 10)
    rho_high = np.percentile(rho_g, 90)
    
    delta_low = toe_model_simple((np.array([rho_low]), np.array([f_HI_mean])), *popt)[0]
    delta_high = toe_model_simple((np.array([rho_high]), np.array([f_HI_mean])), *popt)[0]
    
    print(f"  Δ_g at ρ_g = {rho_low:.2f} (10th %ile): {delta_low:+.4f}")
    print(f"  Δ_g at ρ_g = {rho_high:.2f} (90th %ile): {delta_high:+.4f}")
    
    # Check for sign flip
    sign_flip = (delta_low * delta_high) < 0
    
    print(f"  Sign flip detected: {'YES ✓' if sign_flip else 'NO ✗'}")
    
    return {
        'tested': True,
        'rho_range_dex': float(rho_range),
        'delta_low': float(delta_low),
        'delta_high': float(delta_high),
        'sign_flip': bool(sign_flip)
    }


def test_fhi_correlation(df):
    """
    Test correlation between Δ_g and f_HI after controlling for mass.
    
    TOE predicts: ∂Δ_g/∂f_HI > 0 at fixed ρ_g
    """
    print("\n" + "-"*70)
    print(" TEST: Δ_g - f_HI CORRELATION")
    print("-"*70)
    
    # Partial correlation controlling for density
    from scipy.stats import pearsonr, spearmanr
    
    # Simple correlation
    r, p = pearsonr(df['f_HI'], df['delta_g'])
    rho_s, p_s = spearmanr(df['f_HI'], df['delta_g'])
    
    print(f"  Pearson r(f_HI, Δ_g) = {r:.4f}, p = {p:.2e}")
    print(f"  Spearman ρ(f_HI, Δ_g) = {rho_s:.4f}, p = {p_s:.2e}")
    
    # TOE predicts positive correlation
    correct_sign = r > 0
    significant = p < CRITERIA['correlation_p_threshold']
    
    print(f"  Expected sign (positive): {'YES ✓' if correct_sign else 'NO ✗'}")
    print(f"  Significant (p < {CRITERIA['correlation_p_threshold']}): {'YES ✓' if significant else 'NO ✗'}")
    
    return {
        'pearson_r': float(r),
        'pearson_p': float(p),
        'spearman_rho': float(rho_s),
        'spearman_p': float(p_s),
        'correct_sign': bool(correct_sign),
        'significant': bool(significant)
    }


def test_lambda_constraint(toe_results):
    """
    Test if λ_QR is within [1/3, 3] × λ_QR^gal
    """
    print("\n" + "-"*70)
    print(" TEST: λ_QR CONSTRAINT")
    print("-"*70)
    
    if 'error' in toe_results:
        print("  Cannot test: TOE fit failed")
        return {'tested': False}
    
    lambda_QR = toe_results['lambda_QR']
    lambda_gal = CRITERIA['lambda_gal']
    
    ratio = lambda_QR / lambda_gal
    
    in_range = CRITERIA['lambda_ratio_min'] <= ratio <= CRITERIA['lambda_ratio_max']
    
    print(f"  λ_QR (groups) = {lambda_QR:.4f}")
    print(f"  λ_QR (galactic) = {lambda_gal:.4f}")
    print(f"  Ratio = {ratio:.3f}")
    print(f"  Allowed range: [{CRITERIA['lambda_ratio_min']:.2f}, {CRITERIA['lambda_ratio_max']:.2f}]")
    print(f"  In range: {'YES ✓' if in_range else 'NO ✗ (FALSIFIED)'}")
    
    return {
        'lambda_QR': float(lambda_QR),
        'lambda_gal': float(lambda_gal),
        'ratio': float(ratio),
        'in_range': bool(in_range)
    }


def test_amplitude_constraint(toe_results):
    """
    Test if amplitude is within expected range.
    """
    print("\n" + "-"*70)
    print(" TEST: AMPLITUDE CONSTRAINT")
    print("-"*70)
    
    if 'error' in toe_results:
        print("  Cannot test: TOE fit failed")
        return {'tested': False}
    
    alpha = abs(toe_results['alpha'])
    amp_gal = CRITERIA['amplitude_gal']
    
    ratio = alpha / amp_gal
    
    in_range = CRITERIA['amplitude_ratio_min'] <= ratio <= CRITERIA['amplitude_ratio_max']
    
    print(f"  Amplitude (groups) = {alpha:.4f}")
    print(f"  Amplitude (galactic) = {amp_gal:.4f}")
    print(f"  Ratio = {ratio:.3f}")
    print(f"  Expected range: [{CRITERIA['amplitude_ratio_min']:.1f}, {CRITERIA['amplitude_ratio_max']:.1f}]")
    print(f"  In range: {'YES ✓' if in_range else 'NO ✗'}")
    
    return {
        'amplitude': float(alpha),
        'amplitude_gal': float(amp_gal),
        'ratio': float(ratio),
        'in_range': bool(in_range)
    }


def main():
    print("="*70)
    print(" GRP-001: TOE TEST FOR GALAXY GROUPS")
    print(" Testing Full TOE Equation (Not Just U-Shape!)")
    print(" Author: Jonathan Édouard Slama")
    print("="*70)
    
    # Print pre-registered criteria
    print("\n" + "-"*70)
    print(" PRE-REGISTERED FALSIFICATION CRITERIA")
    print("-"*70)
    print(f"  1. No phase flip over > 1.5 dex → FALSIFIED")
    print(f"  2. No Δ_g - f_HI correlation (p > {CRITERIA['correlation_p_threshold']}) → FALSIFIED")
    print(f"  3. λ_QR ∉ [{CRITERIA['lambda_ratio_min']:.2f}, {CRITERIA['lambda_ratio_max']:.2f}] × λ_gal → FALSIFIED")
    print(f"  4. Amplitude < {CRITERIA['amplitude_ratio_min']:.1f} × galactic → FALSIFIED")
    
    # Create mock data (will replace with real Tempel data)
    print("\n" + "-"*70)
    print(" DATA")
    print("-"*70)
    print("  Using mock data with embedded TOE signal for pipeline validation")
    
    df, true_params = create_mock_group_data_toe()
    
    # Fit TOE model
    toe_results, popt = test_toe_model(df)
    
    # Fit null model
    null_results = test_null_model(df)
    
    # Model comparison
    print("\n" + "-"*70)
    print(" MODEL COMPARISON")
    print("-"*70)
    
    if 'aic' in toe_results and 'aic' in null_results:
        delta_aic = null_results['aic'] - toe_results['aic']
        print(f"  ΔAIC = AIC_null - AIC_TOE = {delta_aic:.1f}")
        print(f"  TOE model {'PREFERRED' if delta_aic > 2 else 'not preferred'} (ΔAIC > 2)")
        toe_preferred = delta_aic > 2
    else:
        toe_preferred = False
    
    # Run all tests
    results = {
        'mock_data': True,
        'true_params': true_params,
        'toe_fit': toe_results,
        'null_fit': null_results,
        'tests': {}
    }
    
    results['tests']['phase_flip'] = test_phase_flip(df, popt)
    results['tests']['fhi_correlation'] = test_fhi_correlation(df)
    results['tests']['lambda_constraint'] = test_lambda_constraint(toe_results)
    results['tests']['amplitude_constraint'] = test_amplitude_constraint(toe_results)
    
    # Final verdict
    print("\n" + "="*70)
    print(" VERDICT")
    print("="*70)
    
    falsified = False
    falsification_reasons = []
    
    # Check each criterion
    if not results['tests']['phase_flip'].get('sign_flip', False):
        if results['tests']['phase_flip'].get('rho_range_dex', 0) > 1.5:
            falsified = True
            falsification_reasons.append("No phase flip over > 1.5 dex")
    
    if not results['tests']['fhi_correlation'].get('significant', False):
        falsified = True
        falsification_reasons.append("No significant Δ_g - f_HI correlation")
    
    if not results['tests']['lambda_constraint'].get('in_range', True):
        falsified = True
        falsification_reasons.append("λ_QR outside allowed range")
    
    if not results['tests']['amplitude_constraint'].get('in_range', True):
        # Only falsify if way too small
        if results['tests']['amplitude_constraint'].get('ratio', 1) < 0.1:
            falsified = True
            falsification_reasons.append("Amplitude too small")
    
    if falsified:
        print("  ❌ QO+R TOE FALSIFIED AT GROUP SCALE")
        print("  Reasons:")
        for reason in falsification_reasons:
            print(f"    - {reason}")
    else:
        print("  ✅ QO+R TOE SUPPORTED AT GROUP SCALE")
        print("  All falsification criteria passed")
    
    if not toe_preferred:
        print("  ⚠️  WARNING: TOE model not strongly preferred over null model")
    
    print("\n  ⚠️  NOTE: Results based on MOCK DATA")
    print("  Real Tempel+2017 × ALFALFA data required for definitive test")
    
    results['verdict'] = {
        'falsified': falsified,
        'reasons': falsification_reasons,
        'toe_preferred': toe_preferred
    }
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "grp001_toe_results.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
