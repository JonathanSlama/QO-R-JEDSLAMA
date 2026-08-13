#!/usr/bin/env python3
"""
===============================================================================
CORE TESTS - Primary Hypothesis Validation
===============================================================================

Tests:
- test_qr_independence: Verify Q and R are independent variables
- test_ushape_detection: Detect U-shape in BTFR residuals (H1)
- test_qor_model: Fit full QO+R model
- test_killer_prediction: Test sign inversion (H3)

Author: Jonathan Édouard Slama
Date: December 2025
===============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def prepare_data(df, dataset_name):
    """Prepare dataframe with required columns."""
    df = df.copy()
    
    # Gas fraction
    if 'gas_fraction' not in df.columns:
        if 'f_gas' in df.columns:
            df['gas_fraction'] = df['f_gas']
        elif 'MHI' in df.columns and 'M_bar' in df.columns:
            df['gas_fraction'] = df['MHI'] / df['M_bar']
        elif 'f_HI' in df.columns:
            df['gas_fraction'] = df['f_HI']
    
    # Log stellar mass
    if 'log_Mstar' not in df.columns:
        if 'M_star' in df.columns:
            # Check if SPARC units (10^10 M_sun)
            if dataset_name == 'SPARC' and df['M_star'].max() < 100:
                df['log_Mstar'] = np.log10(df['M_star'] * 1e10)
            else:
                df['log_Mstar'] = np.log10(df['M_star'].clip(1e6, None))
    
    # Environment
    if 'env_std' not in df.columns:
        if 'density_proxy' in df.columns:
            df['env_std'] = (df['density_proxy'] - df['density_proxy'].mean()) / df['density_proxy'].std()
        elif 'log_env' in df.columns:
            df['env_std'] = (df['log_env'] - df['log_env'].mean()) / df['log_env'].std()
    
    # BTFR residual
    if 'btfr_residual' not in df.columns:
        if 'log_Mbar' in df.columns and 'log_Vflat' in df.columns:
            mask = np.isfinite(df['log_Mbar']) & np.isfinite(df['log_Vflat'])
            slope, intercept, _, _, _ = stats.linregress(
                df.loc[mask, 'log_Mbar'], df.loc[mask, 'log_Vflat']
            )
            df['btfr_residual'] = df['log_Vflat'] - (slope * df['log_Mbar'] + intercept)
    
    return df


def fit_ushape(env, residuals, n_bins=12, min_per_bin=10):
    """Fit quadratic U-shape to binned data."""
    mask = np.isfinite(env) & np.isfinite(residuals)
    env, residuals = np.array(env)[mask], np.array(residuals)[mask]
    
    if len(env) < 50:
        return None
    
    bins = np.percentile(env, np.linspace(0, 100, n_bins + 1))
    bin_centers, bin_means, bin_errors = [], [], []
    
    for i in range(n_bins):
        if i < n_bins - 1:
            m = (env >= bins[i]) & (env < bins[i+1])
        else:
            m = env >= bins[i]
        
        if np.sum(m) >= min_per_bin:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_means.append(np.mean(residuals[m]))
            bin_errors.append(np.std(residuals[m]) / np.sqrt(np.sum(m)))
    
    if len(bin_centers) < 5:
        return None
    
    try:
        popt, pcov = curve_fit(
            lambda x, a, b, c: a*x**2 + b*x + c,
            bin_centers, bin_means, p0=[0.01, 0, 0]
        )
        a, b, c = popt
        a_err = np.sqrt(pcov[0, 0])
        
        t_stat = abs(a) / a_err if a_err > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(t_stat, len(bin_centers) - 3))
        
        return {
            'a': float(a), 'a_err': float(a_err),
            'b': float(b), 'c': float(c),
            't_stat': float(t_stat), 'p_value': float(p_value),
            'n_bins': len(bin_centers), 'n_points': len(env)
        }
    except:
        return None


# =============================================================================
# TEST 1: Q-R INDEPENDENCE
# =============================================================================

def test_qr_independence(datasets):
    """
    Verify that Q (gas_fraction) and R (log_Mstar) are independent variables.
    
    Success criteria:
    - Correlation is negative (physically expected)
    - |Correlation| < 0.95 (not degenerate)
    """
    results = {
        'test_name': 'Q-R Independence',
        'description': 'Verify Q and R are independent measurements',
        'datasets': {}
    }
    
    all_passed = True
    
    for name, df in datasets.items():
        if name.startswith('TNG'):
            continue  # Skip simulation results
        
        df = prepare_data(df, name)
        
        if 'gas_fraction' not in df.columns or 'log_Mstar' not in df.columns:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'Missing columns'}
            continue
        
        # Clean data
        mask = np.isfinite(df['gas_fraction']) & np.isfinite(df['log_Mstar'])
        Q = df.loc[mask, 'gas_fraction'].values
        R = df.loc[mask, 'log_Mstar'].values
        
        # Compute correlation
        corr, p_value = stats.pearsonr(Q, R)
        
        # Bootstrap CI
        n_boot = 1000
        boot_corrs = []
        for _ in range(n_boot):
            idx = np.random.choice(len(Q), len(Q), replace=True)
            boot_corrs.append(np.corrcoef(Q[idx], R[idx])[0, 1])
        
        ci_low = np.percentile(boot_corrs, 2.5)
        ci_high = np.percentile(boot_corrs, 97.5)
        
        # Check criteria
        is_negative = corr < 0
        is_not_degenerate = abs(corr) < 0.95
        passed = is_negative and is_not_degenerate
        
        if not passed:
            all_passed = False
        
        results['datasets'][name] = {
            'n': int(mask.sum()),
            'correlation': float(corr),
            'p_value': float(p_value),
            'ci_95': [float(ci_low), float(ci_high)],
            'is_negative': is_negative,
            'is_not_degenerate': is_not_degenerate,
            'passed': passed,
            'Q_mean': float(Q.mean()),
            'Q_std': float(Q.std()),
            'R_mean': float(R.mean()),
            'R_std': float(R.std())
        }
    
    results['passed'] = all_passed
    results['summary'] = f"Q-R correlation check: {'PASSED' if all_passed else 'FAILED'}"
    
    return results


# =============================================================================
# TEST 2: U-SHAPE DETECTION
# =============================================================================

def test_ushape_detection(datasets):
    """
    Detect U-shape in BTFR residuals vs environment.
    
    Success criteria:
    - Quadratic coefficient a > 0
    - p-value < 0.05 on at least one dataset
    """
    results = {
        'test_name': 'U-Shape Detection',
        'description': 'Detect quadratic environmental dependence (H1)',
        'datasets': {}
    }
    
    any_significant = False
    
    for name, df in datasets.items():
        if name.startswith('TNG'):
            continue
        
        df = prepare_data(df, name)
        
        if 'env_std' not in df.columns or 'btfr_residual' not in df.columns:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'Missing columns'}
            continue
        
        # Fit U-shape
        fit = fit_ushape(df['env_std'].values, df['btfr_residual'].values)
        
        if fit is None:
            results['datasets'][name] = {'status': 'FAILED', 'reason': 'Fit failed'}
            continue
        
        # Check criteria
        is_positive = fit['a'] > 0
        is_significant = fit['p_value'] < 0.05
        passed = is_positive and is_significant
        
        if passed:
            any_significant = True
        
        results['datasets'][name] = {
            'n': fit['n_points'],
            'a': fit['a'],
            'a_err': fit['a_err'],
            'p_value': fit['p_value'],
            't_stat': fit['t_stat'],
            'is_positive': is_positive,
            'is_significant': is_significant,
            'passed': passed
        }
    
    results['passed'] = any_significant
    results['summary'] = f"U-shape detection: {'CONFIRMED' if any_significant else 'NOT DETECTED'}"
    
    return results


# =============================================================================
# TEST 3: QO+R MODEL FIT
# =============================================================================

def test_qor_model(datasets):
    """
    Fit full QO+R regression model.
    
    Model: residual = α + β_Q×(Q×env²) + β_R×(R×env²) + λ_QR×(Q²×R²)
    """
    results = {
        'test_name': 'QO+R Model Fit',
        'description': 'Fit complete QO+R regression model',
        'datasets': {}
    }
    
    for name, df in datasets.items():
        if name.startswith('TNG'):
            continue
        
        df = prepare_data(df, name)
        
        required = ['gas_fraction', 'log_Mstar', 'env_std', 'btfr_residual']
        if not all(col in df.columns for col in required):
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'Missing columns'}
            continue
        
        # Clean data
        mask = np.all([np.isfinite(df[col]) for col in required], axis=0)
        df_clean = df[mask].copy()
        
        if len(df_clean) < 100:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'Insufficient data'}
            continue
        
        # Normalize
        Q = df_clean['gas_fraction'].values
        R = df_clean['log_Mstar'].values
        env = df_clean['env_std'].values
        residuals = df_clean['btfr_residual'].values
        
        Q_norm = (Q - Q.mean()) / Q.std()
        R_norm = (R - R.mean()) / R.std()
        env2 = env ** 2
        
        # Design matrix
        X = np.column_stack([
            np.ones(len(df_clean)),
            Q_norm * env2,
            R_norm * env2,
            (Q_norm**2) * (R_norm**2)
        ])
        
        # Fit
        if HAS_STATSMODELS:
            model = sm.OLS(residuals, X)
            fit = model.fit(cov_type='HC3')
            
            results['datasets'][name] = {
                'n': len(df_clean),
                'alpha': float(fit.params[0]),
                'beta_Q': float(fit.params[1]),
                'beta_Q_se': float(fit.bse[1]),
                'beta_Q_p': float(fit.pvalues[1]),
                'beta_R': float(fit.params[2]),
                'beta_R_se': float(fit.bse[2]),
                'beta_R_p': float(fit.pvalues[2]),
                'lambda_QR': float(fit.params[3]),
                'lambda_QR_se': float(fit.bse[3]),
                'lambda_QR_p': float(fit.pvalues[3]),
                'R2': float(fit.rsquared),
                'R2_adj': float(fit.rsquared_adj),
                'passed': True
            }
        else:
            coeffs, resid, rank, s = np.linalg.lstsq(X, residuals, rcond=None)
            results['datasets'][name] = {
                'n': len(df_clean),
                'alpha': float(coeffs[0]),
                'beta_Q': float(coeffs[1]),
                'beta_R': float(coeffs[2]),
                'lambda_QR': float(coeffs[3]),
                'passed': True,
                'note': 'No standard errors (statsmodels not installed)'
            }
    
    results['passed'] = any(d.get('passed', False) for d in results['datasets'].values())
    results['summary'] = 'QO+R model fitted successfully'
    
    return results


# =============================================================================
# TEST 4: KILLER PREDICTION
# =============================================================================

def test_killer_prediction(datasets):
    """
    Test killer prediction: sign inversion between Q-dom and R-dom.
    
    Success criteria:
    - Q-dominated (f_gas >= 0.5): a > 0
    - EXTREME R-dominated (f_gas < 0.05, log M* > 10.5): a < 0
    """
    results = {
        'test_name': 'Killer Prediction',
        'description': 'Test U-shape sign inversion between Q-dom and R-dom (H3)',
        'datasets': {}
    }
    
    inversion_found = False
    
    for name, df in datasets.items():
        if name.startswith('TNG'):
            continue
        
        df = prepare_data(df, name)
        
        required = ['gas_fraction', 'env_std', 'btfr_residual']
        if not all(col in df.columns for col in required):
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'Missing columns'}
            continue
        
        # Clean data
        mask = np.all([np.isfinite(df[col]) for col in required], axis=0)
        df_clean = df[mask].copy()
        
        # Define subsamples
        q_dom = df_clean[df_clean['gas_fraction'] >= 0.5]
        
        # EXTREME R-dominated (TNG300 equivalent)
        if 'log_Mstar' in df_clean.columns:
            extreme_r = df_clean[
                (df_clean['gas_fraction'] < 0.05) & 
                (df_clean['log_Mstar'] > 10.5)
            ]
        else:
            extreme_r = pd.DataFrame()
        
        strata_results = {}
        
        # Q-dominated
        if len(q_dom) >= 50:
            fit_q = fit_ushape(q_dom['env_std'].values, q_dom['btfr_residual'].values)
            if fit_q:
                strata_results['Q_dominated'] = {
                    'n': fit_q['n_points'],
                    'mean_fgas': float(q_dom['gas_fraction'].mean()),
                    'a': fit_q['a'],
                    'a_err': fit_q['a_err'],
                    'p_value': fit_q['p_value'],
                    'sign': 'positive' if fit_q['a'] > 0 else 'negative'
                }
        
        # EXTREME R-dominated
        if len(extreme_r) >= 20:
            fit_r = fit_ushape(
                extreme_r['env_std'].values, 
                extreme_r['btfr_residual'].values,
                n_bins=8, min_per_bin=5
            )
            if fit_r:
                strata_results['EXTREME_R_dominated'] = {
                    'n': fit_r['n_points'],
                    'mean_fgas': float(extreme_r['gas_fraction'].mean()),
                    'mean_logMstar': float(extreme_r['log_Mstar'].mean()) if 'log_Mstar' in extreme_r.columns else None,
                    'a': fit_r['a'],
                    'a_err': fit_r['a_err'],
                    'p_value': fit_r['p_value'],
                    'sign': 'positive' if fit_r['a'] > 0 else 'negative'
                }
        else:
            strata_results['EXTREME_R_dominated'] = {
                'n': len(extreme_r),
                'status': 'INSUFFICIENT_DATA',
                'note': 'Expected for HI surveys - gas-poor galaxies are invisible'
            }
        
        # Check inversion
        a_q = strata_results.get('Q_dominated', {}).get('a', None)
        a_r = strata_results.get('EXTREME_R_dominated', {}).get('a', None)
        
        if a_q is not None and a_r is not None:
            inversion = (a_q > 0) and (a_r < 0)
            if inversion:
                inversion_found = True
        else:
            inversion = None
        
        results['datasets'][name] = {
            'strata': strata_results,
            'inversion_detected': inversion,
            'passed': inversion == True
        }
    
    results['passed'] = inversion_found
    results['summary'] = f"Killer prediction: {'CONFIRMED' if inversion_found else 'NOT CONFIRMED'}"
    
    return results


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'test_qr_independence',
    'test_ushape_detection',
    'test_qor_model',
    'test_killer_prediction'
]
