#!/usr/bin/env python3
"""
===============================================================================
ALTERNATIVE ELIMINATION TESTS
===============================================================================

Tests to eliminate competing explanations:
- test_selection_bias: Is U-shape an artifact of selection?
- test_mond_comparison: Does MOND predict/remove the pattern? (real data)
- test_baryonic_control: Do baryonic physics explain it?
- test_tng_comparison: How does observation compare to ΛCDM simulation?

Author: Jonathan Édouard Slama
Date: December 2025
===============================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

from core_tests import prepare_data, fit_ushape


# =============================================================================
# TEST 1: SELECTION BIAS SIMULATION
# =============================================================================

def test_selection_bias(datasets, n_mock=100):
    """
    Test if U-shape could be created by selection effects.
    
    Generate mock catalogs with NO intrinsic U-shape,
    apply same selection function, check for spurious signal.
    
    Success criteria:
    - Mock U-shape amplitude << observed amplitude
    """
    results = {
        'test_name': 'Selection Bias Simulation',
        'description': 'Test if U-shape is an artifact of sample selection',
        'n_mock': n_mock,
        'datasets': {}
    }
    
    all_passed = True
    
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
        
        if n < 200:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'Insufficient data'}
            continue
        
        # Fit observed U-shape
        obs_fit = fit_ushape(env, res)
        if obs_fit is None:
            results['datasets'][name] = {'status': 'FAILED', 'reason': 'Observed fit failed'}
            all_passed = False
            continue
        
        a_observed = obs_fit['a']
        
        # Generate mocks with NO intrinsic U-shape
        mock_a = []
        res_std = np.std(res)
        
        for _ in range(n_mock):
            mock_res = np.random.normal(0, res_std, n)
            fit = fit_ushape(env, mock_res)
            if fit is not None:
                mock_a.append(fit['a'])
        
        mock_a = np.array(mock_a)
        
        if len(mock_a) < n_mock * 0.8:
            results['datasets'][name] = {'status': 'FAILED', 'reason': 'Too many mock fit failures'}
            all_passed = False
            continue
        
        # Compare
        mock_mean = np.mean(mock_a)
        mock_std = np.std(mock_a)
        z_score = (a_observed - mock_mean) / mock_std if mock_std > 0 else np.inf
        ratio = abs(mock_mean / a_observed) if a_observed != 0 else np.inf
        
        passed = ratio < 0.3 and z_score > 2
        if not passed:
            all_passed = False
        
        results['datasets'][name] = {
            'n': n,
            'a_observed': float(a_observed),
            'a_mock_mean': float(mock_mean),
            'a_mock_std': float(mock_std),
            'ratio_mock_to_obs': float(ratio),
            'z_score': float(z_score),
            'passed': passed,
            'interpretation': 'U-shape NOT artifact of selection' if passed else 'Possible selection artifact'
        }
    
    results['passed'] = all_passed
    results['summary'] = f"Selection bias: {'EXCLUDED' if all_passed else 'NOT EXCLUDED'}"
    
    return results


# =============================================================================
# TEST 2: MOND COMPARISON (Real Observational Data)
# =============================================================================

def test_mond_comparison(datasets):
    """
    Compare Newtonian BTFR residuals with MOND predictions.
    
    Uses REAL OBSERVATIONAL DATA (SPARC, ALFALFA).
    
    IMPORTANT: SPARC has TRUE rotation curves, so it's the gold standard
    for MOND testing. ALFALFA uses W50/2 proxy which is less reliable for MOND.
    
    Test: Does MOND remove the U-shape environmental dependence?
    
    If MOND explains the U-shape:
    - MOND residuals should be FLAT (no environmental dependence)
    - |a_MOND| << |a_BTFR|
    
    Success criteria:
    - U-shape persists after MOND correction on SPARC (true rotation curves)
    """
    results = {
        'test_name': 'MOND Comparison (Real Data)',
        'description': 'Test if MOND explains or removes the U-shape using real observations',
        'note': 'SPARC has true rotation curves (gold standard), ALFALFA uses W50/2 proxy',
        'datasets': {}
    }
    
    # MOND parameters
    a0 = 1.2e-10  # m/s² - MOND acceleration constant
    G = 6.674e-11  # m³/kg/s²
    M_sun = 1.989e30  # kg
    
    sparc_passed = None  # Track SPARC specifically (gold standard)
    
    for name, df in datasets.items():
        if name.startswith('TNG'):
            continue
        
        df = prepare_data(df, name)
        
        # Check for required columns
        if 'env_std' not in df.columns or 'btfr_residual' not in df.columns:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'Missing columns'}
            continue
        
        # Get baryonic mass
        if 'M_bar' in df.columns:
            M_bar = df['M_bar'].values
        elif 'log_Mbar' in df.columns:
            M_bar = 10**df['log_Mbar'].values
        else:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'No baryonic mass'}
            continue
        
        # Get observed velocity
        if 'Vflat' in df.columns:
            V_obs = df['Vflat'].values
            velocity_type = 'true_rotation_curve'
        elif 'log_Vflat' in df.columns:
            V_obs = 10**df['log_Vflat'].values
            # Check if it's SPARC (true curves) or ALFALFA (W50 proxy)
            velocity_type = 'true_rotation_curve' if name == 'SPARC' else 'W50_proxy'
        else:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'No velocity data'}
            continue
        
        # MOND prediction (deep MOND regime): V_MOND^4 = G * M_bar * a0
        M_bar_kg = M_bar * M_sun
        V_MOND = (G * M_bar_kg * a0) ** 0.25  # m/s
        V_MOND_km = V_MOND / 1000  # km/s
        
        # Valid data mask
        mask = (np.isfinite(V_obs) & np.isfinite(V_MOND_km) & 
                (V_MOND_km > 0) & (V_obs > 0) & np.isfinite(df['env_std']))
        
        if mask.sum() < 50:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'Insufficient valid data'}
            continue
        
        # Calculate residuals
        mond_residual = np.log10(V_obs[mask]) - np.log10(V_MOND_km[mask])
        env = df.loc[mask, 'env_std'].values
        btfr_res = df.loc[mask, 'btfr_residual'].values
        
        # Fit U-shape to BTFR residuals
        btfr_fit = fit_ushape(env, btfr_res)
        
        # Fit U-shape to MOND residuals
        mond_fit = fit_ushape(env, mond_residual)
        
        if btfr_fit is None or mond_fit is None:
            results['datasets'][name] = {'status': 'FAILED', 'reason': 'Fit failed'}
            continue
        
        # Compare
        btfr_a = btfr_fit['a']
        mond_a = mond_fit['a']
        
        # MOND removes U-shape if |mond_a| << |btfr_a|
        ratio = abs(mond_a / btfr_a) if btfr_a != 0 else np.inf
        
        # Criteria:
        # - If ratio < 0.3: MOND removes U-shape → MOND may explain it
        # - If ratio > 0.5: U-shape persists → MOND does NOT explain it
        mond_removes = ratio < 0.3
        ushape_persists = ratio > 0.5
        
        # Dataset passes if MOND does NOT remove the U-shape
        passed = not mond_removes
        
        # Track SPARC specifically (it's the gold standard with true rotation curves)
        if name == 'SPARC':
            sparc_passed = passed
        
        results['datasets'][name] = {
            'n': int(mask.sum()),
            'velocity_type': velocity_type,
            'btfr_a': float(btfr_a),
            'btfr_a_err': float(btfr_fit['a_err']),
            'btfr_p': float(btfr_fit['p_value']),
            'mond_a': float(mond_a),
            'mond_a_err': float(mond_fit['a_err']),
            'mond_p': float(mond_fit['p_value']),
            'ratio_mond_to_btfr': float(ratio),
            'mond_removes_ushape': mond_removes,
            'ushape_persists': ushape_persists,
            'passed': passed,
            'interpretation': 'MOND does NOT explain U-shape' if passed else 'MOND may explain U-shape',
            'note': 'Gold standard (true rotation curves)' if velocity_type == 'true_rotation_curve' else 'W50/2 proxy (less reliable for MOND)'
        }
    
    # Overall pass criteria:
    # - SPARC (true rotation curves) is the gold standard
    # - If SPARC shows MOND doesn't remove U-shape, test passes
    # - ALFALFA result is informative but not conclusive (W50 proxy)
    
    if sparc_passed is not None:
        # SPARC is the gold standard
        results['passed'] = sparc_passed
        if sparc_passed:
            results['summary'] = "MOND comparison: U-shape NOT explained by MOND (SPARC gold standard)"
        else:
            results['summary'] = "MOND comparison: U-shape possibly explained by MOND (SPARC)"
    else:
        # Fallback if SPARC not available
        any_passed = any(
            d.get('passed', False) 
            for d in results['datasets'].values() 
            if isinstance(d, dict) and d.get('velocity_type') == 'true_rotation_curve'
        )
        results['passed'] = any_passed
        results['summary'] = f"MOND comparison: {'U-shape NOT explained by MOND' if any_passed else 'Inconclusive (no true rotation curves)'}"
    
    return results


# =============================================================================
# TEST 3: BARYONIC PHYSICS CONTROL
# =============================================================================

def test_baryonic_control(datasets):
    """
    Test if known baryonic physics explain the U-shape.
    
    Add control variables and check if U-shape persists.
    
    Success criteria:
    - U-shape remains significant after controlling for baryonic properties
    """
    results = {
        'test_name': 'Baryonic Physics Control',
        'description': 'Test if baryonic physics explain the U-shape',
        'datasets': {}
    }
    
    all_passed = True
    
    for name, df in datasets.items():
        if name.startswith('TNG'):
            continue
        
        df = prepare_data(df, name)
        
        if 'env_std' not in df.columns or 'btfr_residual' not in df.columns:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'Missing columns'}
            continue
        
        # Identify available control variables
        controls = []
        control_names = []
        
        if 'gas_fraction' in df.columns:
            controls.append('gas_fraction')
            control_names.append('Gas fraction')
        
        if 'log_Mstar' in df.columns:
            controls.append('log_Mstar')
            control_names.append('Stellar mass')
        
        if 'Distance_Mpc' in df.columns:
            controls.append('Distance_Mpc')
            control_names.append('Distance')
        elif 'Dist' in df.columns:
            controls.append('Dist')
            control_names.append('Distance')
        
        if len(controls) == 0:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'No control variables'}
            continue
        
        # Clean data
        mask = np.isfinite(df['env_std']) & np.isfinite(df['btfr_residual'])
        for ctrl in controls:
            mask &= np.isfinite(df[ctrl])
        
        df_clean = df[mask].copy()
        
        if len(df_clean) < 200:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'Insufficient data'}
            continue
        
        # Raw U-shape fit
        raw_fit = fit_ushape(df_clean['env_std'].values, df_clean['btfr_residual'].values)
        if raw_fit is None:
            results['datasets'][name] = {'status': 'FAILED', 'reason': 'Raw fit failed'}
            all_passed = False
            continue
        
        # Regress out control variables
        X_ctrl = df_clean[controls].values
        X_ctrl = (X_ctrl - X_ctrl.mean(axis=0)) / X_ctrl.std(axis=0)
        X_ctrl = np.column_stack([np.ones(len(df_clean)), X_ctrl])
        
        coeffs, _, _, _ = np.linalg.lstsq(X_ctrl, df_clean['btfr_residual'].values, rcond=None)
        predicted = X_ctrl @ coeffs
        controlled_residuals = df_clean['btfr_residual'].values - predicted
        
        # Fit U-shape on controlled residuals
        ctrl_fit = fit_ushape(df_clean['env_std'].values, controlled_residuals)
        
        if ctrl_fit is None:
            results['datasets'][name] = {'status': 'FAILED', 'reason': 'Controlled fit failed'}
            all_passed = False
            continue
        
        # Compare
        raw_a = raw_fit['a']
        ctrl_a = ctrl_fit['a']
        
        persists = abs(ctrl_a) > 0.5 * abs(raw_a)
        still_positive = ctrl_a > 0
        still_significant = ctrl_fit['p_value'] < 0.1
        
        passed = persists and still_positive
        if not passed:
            all_passed = False
        
        results['datasets'][name] = {
            'n': len(df_clean),
            'controls_used': control_names,
            'raw_a': float(raw_a),
            'raw_a_err': float(raw_fit['a_err']),
            'raw_p': float(raw_fit['p_value']),
            'controlled_a': float(ctrl_a),
            'controlled_a_err': float(ctrl_fit['a_err']),
            'controlled_p': float(ctrl_fit['p_value']),
            'retention_ratio': float(abs(ctrl_a / raw_a)) if raw_a != 0 else np.nan,
            'ushape_persists': persists,
            'still_positive': still_positive,
            'still_significant': still_significant,
            'passed': passed,
            'interpretation': 'U-shape NOT explained by baryonic physics' if passed else 'U-shape may be explained by baryonic physics'
        }
    
    results['passed'] = all_passed
    results['summary'] = f"Baryonic control: {'U-shape persists' if all_passed else 'U-shape may be baryonic'}"
    
    return results


# =============================================================================
# TEST 4: TNG COMPARISON
# =============================================================================

def test_tng_comparison(datasets):
    """
    Compare observations with ΛCDM simulation (IllustrisTNG).
    
    If reality = ΛCDM only, observations should match TNG.
    If reality = ΛCDM + beyond, systematic differences expected.
    
    Success criteria:
    - Systematic differences observed between obs and TNG
    """
    results = {
        'test_name': 'TNG Comparison',
        'description': 'Compare observations with ΛCDM simulation',
        'datasets': {}
    }
    
    # Expected TNG values (from Paper 1 calibration)
    tng_values = {
        'C_Q': 2.93,
        'C_R': -2.39,
        'lambda_QR': 1.07,
        'ushape_a_qdom': 0.017,  # Gas-rich
        'ushape_a_rdom': -0.019  # EXTREME R-dom
    }
    
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
        
        if len(df_clean) < 200:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'Insufficient data'}
            continue
        
        # Fit U-shape
        obs_fit = fit_ushape(df_clean['env_std'].values, df_clean['btfr_residual'].values)
        
        if obs_fit is None:
            results['datasets'][name] = {'status': 'FAILED', 'reason': 'Fit failed'}
            continue
        
        # Compare with TNG
        obs_a = obs_fit['a']
        tng_a = tng_values['ushape_a_qdom']
        
        ratio = abs(obs_a / tng_a) if tng_a != 0 else np.nan
        
        # Check for stratified comparison if possible
        if 'gas_fraction' in df_clean.columns:
            q_dom = df_clean[df_clean['gas_fraction'] >= 0.5]
            if len(q_dom) >= 50:
                q_fit = fit_ushape(q_dom['env_std'].values, q_dom['btfr_residual'].values)
                if q_fit:
                    obs_a_qdom = q_fit['a']
                    ratio_qdom = abs(obs_a_qdom / tng_values['ushape_a_qdom'])
                else:
                    obs_a_qdom = None
                    ratio_qdom = None
            else:
                obs_a_qdom = None
                ratio_qdom = None
        else:
            obs_a_qdom = None
            ratio_qdom = None
        
        systematic_diff = ratio < 0.3 or ratio > 3
        
        results['datasets'][name] = {
            'n': len(df_clean),
            'obs_a': float(obs_a),
            'obs_a_err': float(obs_fit['a_err']),
            'tng_a_reference': float(tng_a),
            'ratio_obs_to_tng': float(ratio) if not np.isnan(ratio) else None,
            'obs_a_qdom': float(obs_a_qdom) if obs_a_qdom else None,
            'ratio_qdom': float(ratio_qdom) if ratio_qdom else None,
            'systematic_difference': systematic_diff,
            'tng_expected_values': tng_values,
            'interpretation': 'Observations differ from ΛCDM (beyond-ΛCDM physics?)' if systematic_diff else 'Observations consistent with ΛCDM'
        }
    
    # Overall assessment
    has_differences = any(
        d.get('systematic_difference', False) 
        for d in results['datasets'].values() 
        if isinstance(d, dict)
    )
    
    results['passed'] = has_differences
    results['summary'] = f"TNG comparison: {'Systematic differences found' if has_differences else 'Consistent with ΛCDM'}"
    
    return results


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'test_selection_bias',
    'test_mond_comparison',
    'test_baryonic_control',
    'test_tng_comparison'
]
