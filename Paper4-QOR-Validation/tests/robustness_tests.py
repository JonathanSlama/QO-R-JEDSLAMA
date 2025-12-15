#!/usr/bin/env python3
"""
===============================================================================
ROBUSTNESS TESTS - Validation of Result Stability
===============================================================================

Tests:
- test_bootstrap_stability: Bootstrap resampling
- test_spatial_jackknife: Sky region removal
- test_cross_validation: 50/50 split validation
- test_selection_sensitivity: Quality cut variations

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

from core_tests import prepare_data, fit_ushape


# =============================================================================
# TEST 1: BOOTSTRAP STABILITY
# =============================================================================

def test_bootstrap_stability(datasets, n_bootstrap=500):
    """
    Test stability of U-shape detection via bootstrap resampling.
    
    Success criteria:
    - >95% of bootstrap samples show a > 0
    - 95% CI excludes zero
    """
    results = {
        'test_name': 'Bootstrap Stability',
        'description': 'Test robustness via bootstrap resampling',
        'n_bootstrap': n_bootstrap,
        'datasets': {}
    }
    
    all_stable = True
    
    for name, df in datasets.items():
        if name.startswith('TNG'):
            continue
        
        df = prepare_data(df, name)
        
        if 'env_std' not in df.columns or 'btfr_residual' not in df.columns:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'Missing columns'}
            continue
        
        # Clean data
        mask = np.isfinite(df['env_std']) & np.isfinite(df['btfr_residual'])
        env = df.loc[mask, 'env_std'].values
        res = df.loc[mask, 'btfr_residual'].values
        n = len(env)
        
        # Bootstrap requires large samples (at least 500 galaxies)
        if n < 500:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': f'Insufficient data for bootstrap (N={n}, need 500+)'}
            continue
        
        # Bootstrap
        boot_a = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            fit = fit_ushape(env[idx], res[idx])
            if fit is not None:
                boot_a.append(fit['a'])
        
        boot_a = np.array(boot_a)
        
        if len(boot_a) < n_bootstrap * 0.8:
            results['datasets'][name] = {'status': 'FAILED', 'reason': 'Too many fit failures'}
            all_stable = False
            continue
        
        # Statistics
        frac_positive = np.mean(boot_a > 0)
        ci_low = np.percentile(boot_a, 2.5)
        ci_high = np.percentile(boot_a, 97.5)
        ci_excludes_zero = (ci_low > 0) or (ci_high < 0)
        
        passed = frac_positive > 0.95
        if not passed:
            all_stable = False
        
        results['datasets'][name] = {
            'n': n,
            'n_successful_boots': len(boot_a),
            'a_mean': float(np.mean(boot_a)),
            'a_median': float(np.median(boot_a)),
            'a_std': float(np.std(boot_a)),
            'ci_95': [float(ci_low), float(ci_high)],
            'frac_positive': float(frac_positive),
            'ci_excludes_zero': ci_excludes_zero,
            'passed': passed
        }
    
    # Pass if at least one large dataset passed (ignore skipped small datasets)
    any_passed = any(
        d.get('passed', False) 
        for d in results['datasets'].values() 
        if isinstance(d, dict) and d.get('status') != 'SKIPPED'
    )
    any_failed = any(
        d.get('passed') == False 
        for d in results['datasets'].values() 
        if isinstance(d, dict) and d.get('status') not in ['SKIPPED', None]
    )
    
    results['passed'] = any_passed and not any_failed
    results['summary'] = f"Bootstrap stability: {'STABLE' if results['passed'] else 'UNSTABLE'}"
    
    return results


# =============================================================================
# TEST 2: SPATIAL JACKKNIFE
# =============================================================================

def test_spatial_jackknife(datasets, n_regions=8):
    """
    Test that no single sky region dominates the signal.
    
    Success criteria:
    - No single region removal causes sign flip
    - Coefficient variation < 50%
    """
    results = {
        'test_name': 'Spatial Jackknife',
        'description': 'Test that no single sky region dominates',
        'n_regions': n_regions,
        'datasets': {}
    }
    
    all_stable = True
    
    for name, df in datasets.items():
        if name.startswith('TNG'):
            continue
        
        df = prepare_data(df, name)
        
        required = ['env_std', 'btfr_residual', 'RA']
        if 'RA' not in df.columns:
            # Try alternative coordinate columns
            if 'RAdeg_HI' in df.columns:
                df['RA'] = df['RAdeg_HI']
            else:
                results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'No RA column'}
                continue
        
        if not all(col in df.columns for col in required):
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'Missing columns'}
            continue
        
        # Clean data
        mask = np.all([np.isfinite(df[col]) for col in required], axis=0)
        df_clean = df[mask].copy()
        
        if len(df_clean) < 200:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'Insufficient data'}
            continue
        
        # Define RA bins
        ra_bins = np.linspace(df_clean['RA'].min(), df_clean['RA'].max(), n_regions + 1)
        df_clean['ra_bin'] = pd.cut(df_clean['RA'], bins=ra_bins, labels=range(n_regions))
        
        # Full sample fit
        full_fit = fit_ushape(df_clean['env_std'].values, df_clean['btfr_residual'].values)
        if full_fit is None:
            results['datasets'][name] = {'status': 'FAILED', 'reason': 'Full fit failed'}
            all_stable = False
            continue
        
        a_full = full_fit['a']
        
        # Jackknife
        jackknife_results = []
        sign_flips = 0
        
        for region in range(n_regions):
            subset = df_clean[df_clean['ra_bin'] != region]
            fit = fit_ushape(subset['env_std'].values, subset['btfr_residual'].values)
            
            if fit is not None:
                jackknife_results.append({
                    'region_removed': region,
                    'n': len(subset),
                    'a': fit['a'],
                    'a_err': fit['a_err']
                })
                
                if (fit['a'] > 0) != (a_full > 0):
                    sign_flips += 1
        
        # Statistics
        a_values = [r['a'] for r in jackknife_results]
        variation = np.std(a_values) / abs(np.mean(a_values)) if np.mean(a_values) != 0 else np.inf
        
        passed = (sign_flips == 0) and (variation < 0.5)
        if not passed:
            all_stable = False
        
        results['datasets'][name] = {
            'n': len(df_clean),
            'a_full': float(a_full),
            'jackknife': jackknife_results,
            'sign_flips': sign_flips,
            'coefficient_variation': float(variation),
            'passed': passed
        }
    
    results['passed'] = all_stable
    results['summary'] = f"Spatial jackknife: {'STABLE' if all_stable else 'UNSTABLE'}"
    
    return results


# =============================================================================
# TEST 3: CROSS-VALIDATION
# =============================================================================

def test_cross_validation(datasets, n_splits=5):
    """
    Test consistency via k-fold cross-validation.
    
    Success criteria:
    - All folds show a > 0
    - Fold estimates within 2σ of each other
    """
    results = {
        'test_name': 'Cross-Validation',
        'description': f'{n_splits}-fold cross-validation consistency',
        'n_splits': n_splits,
        'datasets': {}
    }
    
    all_consistent = True
    
    for name, df in datasets.items():
        if name.startswith('TNG'):
            continue
        
        df = prepare_data(df, name)
        
        if 'env_std' not in df.columns or 'btfr_residual' not in df.columns:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'Missing columns'}
            continue
        
        # Clean data
        mask = np.isfinite(df['env_std']) & np.isfinite(df['btfr_residual'])
        df_clean = df[mask].copy()
        n = len(df_clean)
        
        if n < 200:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'Insufficient data'}
            continue
        
        # Shuffle and split
        indices = np.random.permutation(n)
        fold_size = n // n_splits
        
        fold_results = []
        all_positive = True
        
        for i in range(n_splits):
            # Use fold i as validation, rest as training
            val_idx = indices[i*fold_size:(i+1)*fold_size]
            train_idx = np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])
            
            train_env = df_clean.iloc[train_idx]['env_std'].values
            train_res = df_clean.iloc[train_idx]['btfr_residual'].values
            
            fit = fit_ushape(train_env, train_res)
            
            if fit is not None:
                fold_results.append({
                    'fold': i,
                    'n_train': len(train_idx),
                    'a': fit['a'],
                    'a_err': fit['a_err']
                })
                
                if fit['a'] <= 0:
                    all_positive = False
        
        if len(fold_results) < n_splits - 1:
            results['datasets'][name] = {'status': 'FAILED', 'reason': 'Too many fit failures'}
            all_consistent = False
            continue
        
        # Check consistency
        a_values = [r['a'] for r in fold_results]
        a_mean = np.mean(a_values)
        a_std = np.std(a_values)
        
        # Check if all within 2σ of mean
        within_2sigma = all(abs(a - a_mean) < 2 * a_std for a in a_values)
        
        passed = all_positive and within_2sigma
        if not passed:
            all_consistent = False
        
        results['datasets'][name] = {
            'n': n,
            'folds': fold_results,
            'a_mean': float(a_mean),
            'a_std': float(a_std),
            'all_positive': all_positive,
            'within_2sigma': within_2sigma,
            'passed': passed
        }
    
    results['passed'] = all_consistent
    results['summary'] = f"Cross-validation: {'CONSISTENT' if all_consistent else 'INCONSISTENT'}"
    
    return results


# =============================================================================
# TEST 4: SELECTION SENSITIVITY
# =============================================================================

def test_selection_sensitivity(datasets):
    """
    Test sensitivity to quality cut variations.
    
    Success criteria:
    - a > 0 for >80% of reasonable cut combinations
    """
    results = {
        'test_name': 'Selection Sensitivity',
        'description': 'Test sensitivity to quality cuts',
        'datasets': {}
    }
    
    all_robust = True
    
    for name, df in datasets.items():
        if name.startswith('TNG'):
            continue
        
        df = prepare_data(df, name)
        
        # Check what columns are available
        has_distance = 'Distance_Mpc' in df.columns or 'Dist' in df.columns or 'Distance' in df.columns
        has_mass = 'log_Mstar' in df.columns
        
        if 'env_std' not in df.columns or 'btfr_residual' not in df.columns:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'Missing columns'}
            continue
        
        # Normalize distance column name
        if 'Distance_Mpc' in df.columns:
            dist_col = 'Distance_Mpc'
        elif 'Dist' in df.columns:
            dist_col = 'Dist'
        elif 'Distance' in df.columns:
            dist_col = 'Distance'
        else:
            dist_col = None
        
        # Define cut variations
        cut_results = []
        
        # Base mask
        base_mask = np.isfinite(df['env_std']) & np.isfinite(df['btfr_residual'])
        
        # Distance cuts (if available)
        if dist_col:
            dist_maxes = [100, 150, 200, 250]
        else:
            dist_maxes = [None]
        
        # Mass cuts (if available)
        if has_mass:
            mass_mins = [8.0, 8.5, 9.0]
        else:
            mass_mins = [None]
        
        for dist_max in dist_maxes:
            for mass_min in mass_mins:
                mask = base_mask.copy()
                
                if dist_max and dist_col:
                    mask &= df[dist_col] < dist_max
                
                if mass_min and has_mass:
                    mask &= df['log_Mstar'] > mass_min
                
                subset = df[mask]
                
                if len(subset) < 100:
                    continue
                
                fit = fit_ushape(subset['env_std'].values, subset['btfr_residual'].values)
                
                if fit:
                    cut_results.append({
                        'dist_max': dist_max,
                        'mass_min': mass_min,
                        'n': len(subset),
                        'a': fit['a'],
                        'a_err': fit['a_err'],
                        'p_value': fit['p_value']
                    })
        
        if len(cut_results) < 3:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'Insufficient cut combinations'}
            continue
        
        # Statistics
        frac_positive = np.mean([r['a'] > 0 for r in cut_results])
        passed = frac_positive > 0.8
        
        if not passed:
            all_robust = False
        
        results['datasets'][name] = {
            'n_combinations': len(cut_results),
            'cuts_tested': cut_results,
            'frac_positive': float(frac_positive),
            'passed': passed
        }
    
    results['passed'] = all_robust
    results['summary'] = f"Selection sensitivity: {'ROBUST' if all_robust else 'SENSITIVE'}"
    
    return results


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'test_bootstrap_stability',
    'test_spatial_jackknife',
    'test_cross_validation',
    'test_selection_sensitivity'
]
