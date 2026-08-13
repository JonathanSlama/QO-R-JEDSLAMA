#!/usr/bin/env python3
"""
===============================================================================
AMPLITUDE INVESTIGATION TESTS
===============================================================================

Tests to understand why observed amplitudes are smaller than TNG predictions:
- test_environment_proxy: Is weak signal due to poor environment proxy?
- test_sample_purity: Do mixed populations dilute the signal?

Author: Jonathan Édouard Slama
Date: December 2025
===============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

from core_tests import prepare_data, fit_ushape


# =============================================================================
# TEST 1: ENVIRONMENT PROXY QUALITY
# =============================================================================

def test_environment_proxy(datasets):
    """
    Test if weak signal is due to poor environment proxy.
    
    Compare different environment estimators:
    1. Current: Supergalactic coordinates
    2. Nth nearest neighbor density
    3. Local counts within R Mpc
    
    Success criteria:
    - Better proxy → larger U-shape amplitude
    """
    results = {
        'test_name': 'Environment Proxy Quality',
        'description': 'Test if amplitude improves with better environment proxy',
        'datasets': {}
    }
    
    for name, df in datasets.items():
        if name.startswith('TNG'):
            continue
        
        df = prepare_data(df, name)
        
        if 'btfr_residual' not in df.columns:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'No residuals'}
            continue
        
        # Check for coordinates
        has_coords = False
        if 'RA' in df.columns and 'Dec' in df.columns:
            ra_col, dec_col = 'RA', 'Dec'
            has_coords = True
        elif 'RAdeg_HI' in df.columns and 'DECdeg_HI' in df.columns:
            ra_col, dec_col = 'RAdeg_HI', 'DECdeg_HI'
            has_coords = True
        
        has_distance = 'Distance_Mpc' in df.columns or 'Dist' in df.columns
        if 'Distance_Mpc' in df.columns:
            dist_col = 'Distance_Mpc'
        elif 'Dist' in df.columns:
            dist_col = 'Dist'
        else:
            dist_col = None
        
        proxy_results = []
        
        # Clean data
        mask = np.isfinite(df['btfr_residual'])
        if has_coords:
            mask &= np.isfinite(df[ra_col]) & np.isfinite(df[dec_col])
        if dist_col:
            mask &= np.isfinite(df[dist_col])
        
        df_clean = df[mask].copy()
        
        if len(df_clean) < 200:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'Insufficient data'}
            continue
        
        residuals = df_clean['btfr_residual'].values
        
        # =====================================================================
        # Proxy 1: Current (Supergalactic/density_proxy)
        # =====================================================================
        if 'density_proxy' in df_clean.columns or 'env_std' in df_clean.columns:
            if 'env_std' in df_clean.columns:
                env1 = df_clean['env_std'].values
            else:
                dp = df_clean['density_proxy'].values
                env1 = (dp - dp.mean()) / dp.std()
            
            fit1 = fit_ushape(env1, residuals)
            if fit1:
                proxy_results.append({
                    'proxy': 'Current (density_proxy)',
                    'a': fit1['a'],
                    'a_err': fit1['a_err'],
                    'p_value': fit1['p_value'],
                    'R2': None  # Would need full regression
                })
        
        # =====================================================================
        # Proxy 2: Nth Nearest Neighbor Density
        # =====================================================================
        if has_coords and dist_col:
            # Convert to 3D cartesian
            ra = np.radians(df_clean[ra_col].values)
            dec = np.radians(df_clean[dec_col].values)
            dist = df_clean[dist_col].values
            
            x = dist * np.cos(dec) * np.cos(ra)
            y = dist * np.cos(dec) * np.sin(ra)
            z = dist * np.sin(dec)
            
            coords = np.column_stack([x, y, z])
            
            # Build KD-tree
            tree = cKDTree(coords)
            
            for N in [5, 10, 20]:
                # Distance to Nth neighbor
                distances, _ = tree.query(coords, k=N+1)
                d_N = distances[:, -1]  # Distance to Nth neighbor
                
                # Density ∝ 1/d_N³
                density_NN = 1.0 / (d_N**3 + 1e-10)
                log_density = np.log10(density_NN + 1e-10)
                env_NN = (log_density - log_density.mean()) / log_density.std()
                
                fit_NN = fit_ushape(env_NN, residuals)
                if fit_NN:
                    proxy_results.append({
                        'proxy': f'{N}th Nearest Neighbor',
                        'a': fit_NN['a'],
                        'a_err': fit_NN['a_err'],
                        'p_value': fit_NN['p_value'],
                        'R2': None
                    })
        
        # =====================================================================
        # Proxy 3: Local Counts within R Mpc
        # =====================================================================
        if has_coords and dist_col:
            for R in [2, 5, 10]:  # Mpc
                counts = tree.query_ball_point(coords, R)
                n_neighbors = np.array([len(c) - 1 for c in counts])  # Exclude self
                
                log_counts = np.log10(n_neighbors + 1)
                env_counts = (log_counts - log_counts.mean()) / log_counts.std()
                
                fit_counts = fit_ushape(env_counts, residuals)
                if fit_counts:
                    proxy_results.append({
                        'proxy': f'Counts within {R} Mpc',
                        'a': fit_counts['a'],
                        'a_err': fit_counts['a_err'],
                        'p_value': fit_counts['p_value'],
                        'R2': None
                    })
        
        if len(proxy_results) == 0:
            results['datasets'][name] = {'status': 'FAILED', 'reason': 'No proxies computed'}
            continue
        
        # Find best proxy
        valid_proxies = [p for p in proxy_results if p['a'] is not None]
        if valid_proxies:
            best = max(valid_proxies, key=lambda x: abs(x['a']))
            worst = min(valid_proxies, key=lambda x: abs(x['a']))
            improvement = abs(best['a'] / worst['a']) if worst['a'] != 0 else np.inf
        else:
            best = worst = None
            improvement = None
        
        results['datasets'][name] = {
            'n': len(df_clean),
            'proxies_tested': proxy_results,
            'best_proxy': best['proxy'] if best else None,
            'best_a': best['a'] if best else None,
            'worst_proxy': worst['proxy'] if worst else None,
            'worst_a': worst['a'] if worst else None,
            'improvement_factor': float(improvement) if improvement and not np.isinf(improvement) else None,
            'conclusion': f"Best proxy is {best['proxy']}" if best else 'No valid proxy'
        }
    
    results['passed'] = True  # Informational test
    results['summary'] = 'Environment proxy comparison complete'
    
    return results


# =============================================================================
# TEST 2: SAMPLE PURITY EFFECT
# =============================================================================

def test_sample_purity(datasets):
    """
    Test if mixed galaxy populations dilute the signal.
    
    Stratify by galaxy type and compare U-shape amplitudes.
    
    Success criteria:
    - Purer samples show stronger signal
    """
    results = {
        'test_name': 'Sample Purity Effect',
        'description': 'Test if mixed populations dilute the U-shape signal',
        'datasets': {}
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
        
        strata_results = []
        
        # =====================================================================
        # Full sample
        # =====================================================================
        full_fit = fit_ushape(df_clean['env_std'].values, df_clean['btfr_residual'].values)
        if full_fit:
            strata_results.append({
                'stratum': 'Full sample',
                'n': len(df_clean),
                'a': full_fit['a'],
                'a_err': full_fit['a_err'],
                'p_value': full_fit['p_value']
            })
        
        # =====================================================================
        # By gas fraction (proxy for morphology)
        # =====================================================================
        if 'gas_fraction' in df_clean.columns:
            # Gas-rich (spirals)
            gas_rich = df_clean[df_clean['gas_fraction'] >= 0.5]
            if len(gas_rich) >= 50:
                fit = fit_ushape(gas_rich['env_std'].values, gas_rich['btfr_residual'].values)
                if fit:
                    strata_results.append({
                        'stratum': 'Gas-rich (f_gas >= 0.5)',
                        'n': len(gas_rich),
                        'mean_fgas': float(gas_rich['gas_fraction'].mean()),
                        'a': fit['a'],
                        'a_err': fit['a_err'],
                        'p_value': fit['p_value']
                    })
            
            # Gas-poor
            gas_poor = df_clean[df_clean['gas_fraction'] < 0.3]
            if len(gas_poor) >= 50:
                fit = fit_ushape(gas_poor['env_std'].values, gas_poor['btfr_residual'].values)
                if fit:
                    strata_results.append({
                        'stratum': 'Gas-poor (f_gas < 0.3)',
                        'n': len(gas_poor),
                        'mean_fgas': float(gas_poor['gas_fraction'].mean()),
                        'a': fit['a'],
                        'a_err': fit['a_err'],
                        'p_value': fit['p_value']
                    })
        
        # =====================================================================
        # By stellar mass
        # =====================================================================
        if 'log_Mstar' in df_clean.columns:
            mass_med = df_clean['log_Mstar'].median()
            
            # Low mass
            low_mass = df_clean[df_clean['log_Mstar'] < mass_med]
            if len(low_mass) >= 50:
                fit = fit_ushape(low_mass['env_std'].values, low_mass['btfr_residual'].values)
                if fit:
                    strata_results.append({
                        'stratum': f'Low mass (log M* < {mass_med:.1f})',
                        'n': len(low_mass),
                        'mean_logMstar': float(low_mass['log_Mstar'].mean()),
                        'a': fit['a'],
                        'a_err': fit['a_err'],
                        'p_value': fit['p_value']
                    })
            
            # High mass
            high_mass = df_clean[df_clean['log_Mstar'] >= mass_med]
            if len(high_mass) >= 50:
                fit = fit_ushape(high_mass['env_std'].values, high_mass['btfr_residual'].values)
                if fit:
                    strata_results.append({
                        'stratum': f'High mass (log M* >= {mass_med:.1f})',
                        'n': len(high_mass),
                        'mean_logMstar': float(high_mass['log_Mstar'].mean()),
                        'a': fit['a'],
                        'a_err': fit['a_err'],
                        'p_value': fit['p_value']
                    })
        
        # =====================================================================
        # By distance (Malmquist bias proxy)
        # =====================================================================
        dist_col = None
        if 'Distance_Mpc' in df_clean.columns:
            dist_col = 'Distance_Mpc'
        elif 'Dist' in df_clean.columns:
            dist_col = 'Dist'
        
        if dist_col:
            dist_med = df_clean[dist_col].median()
            
            # Nearby
            nearby = df_clean[df_clean[dist_col] < dist_med]
            if len(nearby) >= 50:
                fit = fit_ushape(nearby['env_std'].values, nearby['btfr_residual'].values)
                if fit:
                    strata_results.append({
                        'stratum': f'Nearby (D < {dist_med:.0f} Mpc)',
                        'n': len(nearby),
                        'mean_dist': float(nearby[dist_col].mean()),
                        'a': fit['a'],
                        'a_err': fit['a_err'],
                        'p_value': fit['p_value']
                    })
            
            # Distant
            distant = df_clean[df_clean[dist_col] >= dist_med]
            if len(distant) >= 50:
                fit = fit_ushape(distant['env_std'].values, distant['btfr_residual'].values)
                if fit:
                    strata_results.append({
                        'stratum': f'Distant (D >= {dist_med:.0f} Mpc)',
                        'n': len(distant),
                        'mean_dist': float(distant[dist_col].mean()),
                        'a': fit['a'],
                        'a_err': fit['a_err'],
                        'p_value': fit['p_value']
                    })
        
        if len(strata_results) < 2:
            results['datasets'][name] = {'status': 'FAILED', 'reason': 'Insufficient strata'}
            continue
        
        # Find strongest signal
        valid = [s for s in strata_results if s.get('a') is not None]
        if valid:
            strongest = max(valid, key=lambda x: abs(x['a']))
            weakest = min(valid, key=lambda x: abs(x['a']))
            ratio = abs(strongest['a'] / weakest['a']) if weakest['a'] != 0 else np.inf
        else:
            strongest = weakest = None
            ratio = None
        
        results['datasets'][name] = {
            'n': len(df_clean),
            'strata': strata_results,
            'strongest_stratum': strongest['stratum'] if strongest else None,
            'strongest_a': strongest['a'] if strongest else None,
            'weakest_stratum': weakest['stratum'] if weakest else None,
            'weakest_a': weakest['a'] if weakest else None,
            'amplitude_ratio': float(ratio) if ratio and not np.isinf(ratio) else None,
            'conclusion': f"Strongest signal in '{strongest['stratum']}'" if strongest else 'No clear pattern'
        }
    
    results['passed'] = True  # Informational test
    results['summary'] = 'Sample purity analysis complete'
    
    return results


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'test_environment_proxy',
    'test_sample_purity'
]
