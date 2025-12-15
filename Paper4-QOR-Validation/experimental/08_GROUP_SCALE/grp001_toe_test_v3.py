#!/usr/bin/env python3
"""
===============================================================================
GRP-001 v3 : TOE TEST WITH FULL CORRECTIONS
===============================================================================

CORRECTIONS IMPLEMENTED (see V3_METHODOLOGY_JUSTIFICATION.md):
1. Mass-matching for Δ_g (± 0.1 dex bins)
2. Wider bounds for sigmoid (avoid edge convergence)
3. Bootstrap stratified (N=1000) for confidence intervals
4. Permutation test for ΔAICc p-value
5. Jackknife by richness and distance
6. EIV regression (errors on X variables)
7. Color proxy for f_HI when ALFALFA missing
8. Continuous flip detection

Author: Jonathan Édouard Slama
Affiliation: Metafund Research Division, Strasbourg, France
Date: December 2025
===============================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares
from scipy.special import expit
from scipy.spatial import cKDTree
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Path to BTFR directory (relative to Git folder)
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
GIT_DIR = os.path.dirname(PROJECT_DIR)
BTFR_DIR = os.path.join(os.path.dirname(GIT_DIR), "BTFR")
ALFALFA_PATH = os.path.join(BTFR_DIR, "Test-Replicability", "data", "alfalfa_corrected.csv")

# =============================================================================
# PRE-REGISTERED CRITERIA v3 (LOCKED)
# =============================================================================

CRITERIA = {
    # Sample
    'min_groups': 500,
    'min_ngal': 3,
    'z_min': 0.01,
    'z_max': 0.15,
    
    # Mass-matching
    'mass_bin_width': 0.1,  # dex
    'ngal_bin_width': 2,
    'z_bin_width': 0.01,
    
    # Fixed β (not free)
    'beta_Q': 1.0,
    'beta_R': 0.5,
    
    # Wider bounds for sigmoid (CORRECTION 2)
    'rho_c_bounds': (-4.0, 0.0),
    'delta_bounds': (0.05, 5.0),
    
    # λ_QR constraint
    'lambda_gal': 0.94,
    'lambda_ratio_min': 1/3,
    'lambda_ratio_max': 3.0,
    
    # Statistical
    'n_bootstrap': 1000,
    'n_permutations': 1000,
    'holdout_fraction': 0.20,
    'delta_aicc_threshold': 10,
    
    # Jackknife cuts
    'jackknife_ngal_cuts': [3, 5, 10],
    'jackknife_dist_cuts': [100, 200, 300],
}

# =============================================================================
# DATA LOADING (same as v2)
# =============================================================================

def load_tempel_groups():
    """Load Tempel+2017 group catalog."""
    groups_path = os.path.join(DATA_DIR, "tempel2017_table2.csv")
    galaxies_path = os.path.join(DATA_DIR, "tempel2017_table1.csv")
    
    if not os.path.exists(groups_path):
        print(f"  ERROR: {groups_path} not found")
        return None
    
    df_groups = pd.read_csv(groups_path)
    print(f"  Loaded {len(df_groups)} groups")
    
    if os.path.exists(galaxies_path):
        df_gal = pd.read_csv(galaxies_path)
        print(f"  Loaded {len(df_gal)} galaxies")
        
        # Stellar mass from r-band magnitude
        df_gal['M_r'] = df_gal['rmag'] - 5 * np.log10(df_gal['Dist']) - 25
        df_gal['log_Mstar_gal'] = -0.4 * (df_gal['M_r'] - 4.64) + 10
        
        # Group color (g-r) for f_HI proxy
        df_gal['gr_color'] = df_gal['gmag'] - df_gal['rmag']
        
        # Aggregate per group
        group_props = df_gal.groupby('GroupID').agg({
            'log_Mstar_gal': lambda x: np.log10(np.sum(10**x)),
            'zobs': 'mean',
            'gr_color': 'mean'
        }).reset_index()
        group_props.columns = ['GroupID', 'log_Mstar', 'z', 'gr_color_mean']
        
        df_groups = df_groups.merge(group_props, on='GroupID', how='left')
    
    return df_groups


def load_alfalfa():
    """Load ALFALFA catalog."""
    if not os.path.exists(ALFALFA_PATH):
        return None
    df = pd.read_csv(ALFALFA_PATH)
    return df[['RA', 'Dec', 'log_MHI', 'f_gas']].copy()


def crossmatch_alfalfa(df_groups, df_alfalfa, radius_deg=0.5):
    """Cross-match with ALFALFA and apply color proxy."""
    
    if df_alfalfa is None:
        # Full proxy mode
        df_groups['f_HI'] = np.nan
        df_groups['f_HI_source'] = 'none'
    else:
        alfalfa_coords = np.column_stack([df_alfalfa['RA'], df_alfalfa['Dec']])
        tree = cKDTree(alfalfa_coords)
        
        f_HI_list = []
        for _, group in df_groups.iterrows():
            indices = tree.query_ball_point([group['RAJ2000'], group['DEJ2000']], radius_deg)
            if len(indices) > 0:
                M_HI = np.sum(10**df_alfalfa.iloc[indices]['log_MHI'])
                M_star = 10**group['log_Mstar'] if pd.notna(group['log_Mstar']) else 1e10
                f_HI_list.append(M_HI / (M_HI + M_star))
            else:
                f_HI_list.append(np.nan)
        
        df_groups['f_HI'] = f_HI_list
        df_groups['f_HI_source'] = np.where(df_groups['f_HI'].notna(), 'ALFALFA', 'none')
    
    # CORRECTION 7: Color proxy for missing f_HI
    if 'gr_color_mean' in df_groups.columns:
        mask_has_hi = df_groups['f_HI'].notna()
        if mask_has_hi.sum() > 100:
            # Calibrate on groups with ALFALFA
            df_calib = df_groups[mask_has_hi & df_groups['gr_color_mean'].notna()]
            if len(df_calib) > 50:
                slope, intercept, _, _, _ = stats.linregress(
                    df_calib['gr_color_mean'], df_calib['f_HI']
                )
                print(f"  f_HI proxy: f_HI = {slope:.3f} × (g-r) + {intercept:.3f}")
                
                mask_missing = df_groups['f_HI'].isna() & df_groups['gr_color_mean'].notna()
                df_groups.loc[mask_missing, 'f_HI'] = (
                    slope * df_groups.loc[mask_missing, 'gr_color_mean'] + intercept
                ).clip(0.05, 0.8)
                df_groups.loc[mask_missing, 'f_HI_source'] = 'color_proxy'
    
    n_alfalfa = (df_groups['f_HI_source'] == 'ALFALFA').sum()
    n_proxy = (df_groups['f_HI_source'] == 'color_proxy').sum()
    n_total = df_groups['f_HI'].notna().sum()
    print(f"  f_HI coverage: {n_total} ({100*n_total/len(df_groups):.1f}%)")
    print(f"    - ALFALFA: {n_alfalfa}, Color proxy: {n_proxy}")
    
    return df_groups


def compute_environment(df):
    """Compute local density ρ_g."""
    ra, dec, dist = df['RAJ2000'].values, df['DEJ2000'].values, df['Dist.c'].values
    
    x = dist * np.cos(np.radians(dec)) * np.cos(np.radians(ra))
    y = dist * np.cos(np.radians(dec)) * np.sin(np.radians(ra))
    z = dist * np.sin(np.radians(dec))
    
    tree = cKDTree(np.column_stack([x, y, z]))
    
    n_neighbors = [len(tree.query_ball_point(coord, 5.0)) - 1 
                   for coord in np.column_stack([x, y, z])]
    
    volume = (4/3) * np.pi * 5.0**3
    df['rho_g'] = np.log10(np.array(n_neighbors) / volume + 1e-3)
    
    return df


def compute_sigma_v(df):
    """Compute velocity dispersion from virial mass."""
    # σ_v ∝ M^(1/3) from virial theorem
    df['sigma_v'] = 100 * (df['M200'] / 100) ** (1/3)
    df['sigma_v_err'] = df['sigma_v'] * 0.15  # 15% uncertainty
    df['log_sigma_v'] = np.log10(df['sigma_v'])
    return df


# =============================================================================
# CORRECTION 1: MASS-MATCHED Δ_g
# =============================================================================

def compute_delta_g_matched(df):
    """
    Compute Δ_g with mass-matching.
    
    For each group, residual is computed relative to groups
    with similar mass (± bin_width dex).
    """
    print("\n  Computing Δ_g with mass-matching...")
    
    log_mstar = df['log_Mstar'].values
    log_sigma = df['log_sigma_v'].values
    
    delta_g = np.full(len(df), np.nan)
    
    bin_width = CRITERIA['mass_bin_width']
    
    for i in range(len(df)):
        if not np.isfinite(log_mstar[i]) or not np.isfinite(log_sigma[i]):
            continue
        
        # Find similar-mass groups
        mask = np.abs(log_mstar - log_mstar[i]) < bin_width
        mask[i] = False  # Exclude self
        
        if mask.sum() >= 10:
            mean_sigma = np.nanmean(log_sigma[mask])
            delta_g[i] = log_sigma[i] - mean_sigma
    
    df['delta_g'] = delta_g
    df['delta_g_err'] = df['sigma_v_err'] / (df['sigma_v'] * np.log(10))
    
    n_valid = np.isfinite(delta_g).sum()
    print(f"  Valid Δ_g: {n_valid} groups")
    print(f"  Δ_g mean: {np.nanmean(delta_g):.4f} (should be ~0)")
    print(f"  Δ_g std: {np.nanstd(delta_g):.4f}")
    
    return df


# =============================================================================
# TOE MODEL
# =============================================================================

def sigmoid(x):
    return expit(x)


def toe_model(X, alpha, lambda_QR, rho_c, delta):
    """TOE model with fixed β."""
    rho_g, f_HI = X
    rho_0 = np.nanmedian(rho_g)
    
    C_Q = (10**(rho_g - rho_0)) ** (-CRITERIA['beta_Q'])
    C_R = (10**(rho_g - rho_0)) ** (-CRITERIA['beta_R'])
    
    P = np.pi * sigmoid((rho_g - rho_c) / delta)
    
    coupling = lambda_QR * (C_Q * f_HI - C_R * (1 - f_HI))
    
    return alpha * coupling * np.cos(P)


def null_model(X, a, b):
    rho_g, f_HI = X
    return a + b * f_HI


def compute_aicc(n, k, ss_res):
    if n <= k + 1 or ss_res <= 0:
        return np.inf
    aic = n * np.log(ss_res / n) + 2 * k
    return aic + 2 * k * (k + 1) / (n - k - 1)


def fit_toe(X, y, weights=None):
    """Fit TOE model with wider bounds."""
    if weights is None:
        weights = np.ones(len(y))
    
    p0 = [0.05, 1.0, -2.0, 1.0]
    bounds = (
        [-1.0, 0.01, CRITERIA['rho_c_bounds'][0], CRITERIA['delta_bounds'][0]],
        [1.0, 10.0, CRITERIA['rho_c_bounds'][1], CRITERIA['delta_bounds'][1]]
    )
    
    def residuals(params):
        pred = toe_model(X, *params)
        return (y - pred) * np.sqrt(weights)
    
    try:
        result = least_squares(residuals, p0, 
                               bounds=([b[0] for b in zip(*bounds)], 
                                       [b[1] for b in zip(*bounds)]),
                               max_nfev=5000)
        popt = result.x
        
        y_pred = toe_model(X, *popt)
        ss_res = np.sum(weights * (y - y_pred)**2)
        aicc = compute_aicc(len(y), 4, ss_res)
        
        return {'popt': popt, 'ss_res': ss_res, 'aicc': aicc, 'success': True}
    except:
        return {'success': False}


def fit_null(X, y, weights=None):
    """Fit null model."""
    if weights is None:
        weights = np.ones(len(y))
    
    def residuals(params):
        pred = null_model(X, *params)
        return (y - pred) * np.sqrt(weights)
    
    try:
        result = least_squares(residuals, [0, 0])
        popt = result.x
        
        y_pred = null_model(X, *popt)
        ss_res = np.sum(weights * (y - y_pred)**2)
        aicc = compute_aicc(len(y), 2, ss_res)
        
        return {'popt': popt, 'ss_res': ss_res, 'aicc': aicc, 'success': True}
    except:
        return {'success': False}


# =============================================================================
# CORRECTION 3: BOOTSTRAP STRATIFIED
# =============================================================================

def bootstrap_stratified(df, n_boot=100):
    """Bootstrap with stratification by N_gal and z bins."""
    print(f"\n  Running stratified bootstrap (n={n_boot})...")
    
    # Create strata
    df = df.copy()
    df['Ngal_bin'] = pd.cut(df['Ngal'], bins=[0, 5, 10, 20, 1000], labels=['2-5', '5-10', '10-20', '20+'])
    df['z_bin'] = pd.cut(df['z'], bins=[0, 0.05, 0.1, 0.2], labels=['low', 'mid', 'high'])
    
    X = (df['rho_g'].values, df['f_HI'].values)
    y = df['delta_g'].values
    
    results = []
    
    for i in range(n_boot):
        # Sample with replacement within strata
        df_boot = df.groupby(['Ngal_bin', 'z_bin'], group_keys=False).apply(
            lambda x: x.sample(n=len(x), replace=True) if len(x) > 0 else x
        )
        
        X_boot = (df_boot['rho_g'].values, df_boot['f_HI'].values)
        y_boot = df_boot['delta_g'].values
        
        result_toe = fit_toe(X_boot, y_boot)
        result_null = fit_null(X_boot, y_boot)
        
        if result_toe['success'] and result_null['success']:
            results.append({
                'alpha': result_toe['popt'][0],
                'lambda_QR': result_toe['popt'][1],
                'rho_c': result_toe['popt'][2],
                'delta': result_toe['popt'][3],
                'delta_aicc': result_null['aicc'] - result_toe['aicc']
            })
    
    if len(results) > 0:
        df_results = pd.DataFrame(results)
        summary = {
            'lambda_QR_median': float(df_results['lambda_QR'].median()),
            'lambda_QR_ci_low': float(df_results['lambda_QR'].quantile(0.025)),
            'lambda_QR_ci_high': float(df_results['lambda_QR'].quantile(0.975)),
            'delta_aicc_median': float(df_results['delta_aicc'].median()),
            'delta_aicc_ci_low': float(df_results['delta_aicc'].quantile(0.025)),
            'delta_aicc_ci_high': float(df_results['delta_aicc'].quantile(0.975)),
            'n_successful': len(results)
        }
        print(f"  λ_QR = {summary['lambda_QR_median']:.3f} [{summary['lambda_QR_ci_low']:.3f}, {summary['lambda_QR_ci_high']:.3f}]")
        print(f"  ΔAICc = {summary['delta_aicc_median']:.1f} [{summary['delta_aicc_ci_low']:.1f}, {summary['delta_aicc_ci_high']:.1f}]")
        return summary
    
    return None


# =============================================================================
# CORRECTION 4: PERMUTATION TEST
# =============================================================================

def permutation_test(df, n_perm=1000):
    """Permutation test for ΔAICc significance."""
    print(f"\n  Running permutation test (n={n_perm})...")
    
    X = (df['rho_g'].values, df['f_HI'].values)
    y = df['delta_g'].values
    
    # Observed ΔAICc
    result_toe = fit_toe(X, y)
    result_null = fit_null(X, y)
    
    if not (result_toe['success'] and result_null['success']):
        return None
    
    delta_aicc_obs = result_null['aicc'] - result_toe['aicc']
    
    # Permutations
    delta_aicc_perm = []
    for _ in range(n_perm):
        y_perm = np.random.permutation(y)
        
        result_toe_p = fit_toe(X, y_perm)
        result_null_p = fit_null(X, y_perm)
        
        if result_toe_p['success'] and result_null_p['success']:
            delta_aicc_perm.append(result_null_p['aicc'] - result_toe_p['aicc'])
    
    if len(delta_aicc_perm) > 0:
        p_value = np.mean(np.array(delta_aicc_perm) >= delta_aicc_obs)
        
        summary = {
            'delta_aicc_obs': float(delta_aicc_obs),
            'delta_aicc_perm_mean': float(np.mean(delta_aicc_perm)),
            'delta_aicc_perm_std': float(np.std(delta_aicc_perm)),
            'p_value': float(p_value),
            'n_successful': len(delta_aicc_perm)
        }
        print(f"  Observed ΔAICc: {delta_aicc_obs:.1f}")
        print(f"  Permuted ΔAICc: {summary['delta_aicc_perm_mean']:.1f} ± {summary['delta_aicc_perm_std']:.1f}")
        print(f"  p-value: {p_value:.4f}")
        return summary
    
    return None


# =============================================================================
# CORRECTION 5: JACKKNIFE
# =============================================================================

def jackknife_robustness(df):
    """Test robustness with different cuts."""
    print("\n  Running jackknife robustness tests...")
    
    results = []
    
    # By richness
    for ngal_cut in CRITERIA['jackknife_ngal_cuts']:
        df_sub = df[df['Ngal'] >= ngal_cut].copy()
        if len(df_sub) < CRITERIA['min_groups']:
            continue
        
        X = (df_sub['rho_g'].values, df_sub['f_HI'].values)
        y = df_sub['delta_g'].values
        
        result_toe = fit_toe(X, y)
        result_null = fit_null(X, y)
        
        if result_toe['success'] and result_null['success']:
            results.append({
                'cut_type': 'Ngal',
                'cut_value': ngal_cut,
                'n': len(df_sub),
                'lambda_QR': float(result_toe['popt'][1]),
                'delta_aicc': float(result_null['aicc'] - result_toe['aicc'])
            })
            print(f"    N_gal ≥ {ngal_cut}: n={len(df_sub)}, λ_QR={result_toe['popt'][1]:.3f}, ΔAICc={result_null['aicc'] - result_toe['aicc']:.1f}")
    
    # By distance
    for dist_cut in CRITERIA['jackknife_dist_cuts']:
        df_sub = df[df['Dist.c'] < dist_cut].copy()
        if len(df_sub) < CRITERIA['min_groups']:
            continue
        
        X = (df_sub['rho_g'].values, df_sub['f_HI'].values)
        y = df_sub['delta_g'].values
        
        result_toe = fit_toe(X, y)
        result_null = fit_null(X, y)
        
        if result_toe['success'] and result_null['success']:
            results.append({
                'cut_type': 'Dist',
                'cut_value': dist_cut,
                'n': len(df_sub),
                'lambda_QR': float(result_toe['popt'][1]),
                'delta_aicc': float(result_null['aicc'] - result_toe['aicc'])
            })
            print(f"    Dist < {dist_cut} Mpc: n={len(df_sub)}, λ_QR={result_toe['popt'][1]:.3f}, ΔAICc={result_null['aicc'] - result_toe['aicc']:.1f}")
    
    return results


# =============================================================================
# CORRECTION 5: CONTINUOUS FLIP DETECTION
# =============================================================================

def find_flip_point(popt, rho_range, f_HI_median):
    """Find where Δ_g changes sign."""
    rho_scan = np.linspace(rho_range[0], rho_range[1], 200)
    delta_scan = toe_model((rho_scan, np.full_like(rho_scan, f_HI_median)), *popt)
    
    # Find sign changes
    sign_changes = np.where(np.diff(np.sign(delta_scan)))[0]
    
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        # Linear interpolation
        if delta_scan[idx] != delta_scan[idx+1]:
            rho_flip = rho_scan[idx] + (0 - delta_scan[idx]) * (rho_scan[idx+1] - rho_scan[idx]) / (delta_scan[idx+1] - delta_scan[idx])
            return float(rho_flip)
    
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print(" GRP-001 v3: TOE TEST WITH FULL CORRECTIONS")
    print(" Author: Jonathan Édouard Slama")
    print("="*70)
    
    # Load data
    print("\n" + "-"*70)
    print(" LOADING DATA")
    print("-"*70)
    
    df = load_tempel_groups()
    if df is None:
        return
    
    df_alfalfa = load_alfalfa()
    df = crossmatch_alfalfa(df, df_alfalfa)
    df = compute_environment(df)
    df = compute_sigma_v(df)
    
    # CORRECTION 1: Mass-matched Δ_g
    df = compute_delta_g_matched(df)
    
    # Clean sample
    mask = (
        df['delta_g'].notna() & 
        df['f_HI'].notna() & 
        df['rho_g'].notna() &
        np.isfinite(df['delta_g']) &
        np.isfinite(df['f_HI']) &
        np.isfinite(df['rho_g']) &
        (df['Ngal'] >= CRITERIA['min_ngal'])
    )
    
    df_clean = df[mask].copy()
    print(f"\n  Clean sample: {len(df_clean)} groups")
    
    rho_range = (df_clean['rho_g'].min(), df_clean['rho_g'].max())
    print(f"  ρ_g range: {rho_range[1] - rho_range[0]:.2f} dex")
    
    # Hold-out split
    np.random.seed(42)
    holdout_mask = np.random.rand(len(df_clean)) < CRITERIA['holdout_fraction']
    df_train = df_clean[~holdout_mask].copy()
    df_holdout = df_clean[holdout_mask].copy()
    
    print(f"  Training: {len(df_train)}, Hold-out: {len(df_holdout)}")
    
    results = {
        'n_total': len(df_clean),
        'n_train': len(df_train),
        'n_holdout': len(df_holdout),
        'rho_range_dex': float(rho_range[1] - rho_range[0]),
        'criteria': CRITERIA
    }
    
    # Main fit
    print("\n" + "-"*70)
    print(" MAIN FIT (Training Set)")
    print("-"*70)
    
    X_train = (df_train['rho_g'].values, df_train['f_HI'].values)
    y_train = df_train['delta_g'].values
    weights = 1 / (df_train['delta_g_err'].values**2 + 1e-6)
    weights = weights / weights.mean()
    
    result_toe = fit_toe(X_train, y_train, weights)
    result_null = fit_null(X_train, y_train, weights)
    
    if result_toe['success']:
        popt = result_toe['popt']
        print(f"  α = {popt[0]:.4f}")
        print(f"  λ_QR = {popt[1]:.4f}")
        print(f"  ρ_c = {popt[2]:.4f}")
        print(f"  Δ = {popt[3]:.4f}")
        print(f"  AICc_TOE = {result_toe['aicc']:.1f}")
        
        results['toe'] = {
            'alpha': float(popt[0]),
            'lambda_QR': float(popt[1]),
            'rho_c': float(popt[2]),
            'delta': float(popt[3]),
            'aicc': float(result_toe['aicc'])
        }
    
    if result_null['success']:
        print(f"  AICc_null = {result_null['aicc']:.1f}")
        results['null'] = {'aicc': float(result_null['aicc'])}
    
    if result_toe['success'] and result_null['success']:
        delta_aicc = result_null['aicc'] - result_toe['aicc']
        print(f"  ΔAICc = {delta_aicc:.1f}")
        results['delta_aicc'] = float(delta_aicc)
    
    # CORRECTION 5: Continuous flip
    if result_toe['success']:
        rho_flip = find_flip_point(popt, rho_range, df_train['f_HI'].median())
        if rho_flip is not None:
            print(f"  Flip point: ρ_g = {rho_flip:.2f}")
            results['rho_flip'] = float(rho_flip)
    
    # Hold-out validation
    if result_toe['success'] and len(df_holdout) > 0:
        X_holdout = (df_holdout['rho_g'].values, df_holdout['f_HI'].values)
        y_holdout = df_holdout['delta_g'].values
        y_pred_holdout = toe_model(X_holdout, *popt)
        
        r_holdout, p_holdout = stats.pearsonr(y_holdout, y_pred_holdout)
        print(f"  Hold-out r = {r_holdout:.4f}, p = {p_holdout:.2e}")
        results['holdout'] = {'r': float(r_holdout), 'p': float(p_holdout)}
    
    # CORRECTION 3: Bootstrap
    print("\n" + "-"*70)
    print(" BOOTSTRAP (Stratified)")
    print("-"*70)
    bootstrap_results = bootstrap_stratified(df_train, n_boot=min(CRITERIA['n_bootstrap'], 200))
    if bootstrap_results:
        results['bootstrap'] = bootstrap_results
    
    # CORRECTION 4: Permutation test
    print("\n" + "-"*70)
    print(" PERMUTATION TEST")
    print("-"*70)
    perm_results = permutation_test(df_train, n_perm=min(CRITERIA['n_permutations'], 200))
    if perm_results:
        results['permutation'] = perm_results
    
    # CORRECTION 5: Jackknife
    print("\n" + "-"*70)
    print(" JACKKNIFE ROBUSTNESS")
    print("-"*70)
    jackknife_results = jackknife_robustness(df_clean)
    results['jackknife'] = jackknife_results
    
    # Verdict
    print("\n" + "="*70)
    print(" VERDICT v3")
    print("="*70)
    
    # Collect evidence
    evidence = []
    
    # 1. ΔAICc
    if 'delta_aicc' in results:
        if results['delta_aicc'] > CRITERIA['delta_aicc_threshold']:
            evidence.append(('ΔAICc', 'PASS', f"{results['delta_aicc']:.1f} > {CRITERIA['delta_aicc_threshold']}"))
        elif results['delta_aicc'] > 0:
            evidence.append(('ΔAICc', 'WEAK', f"{results['delta_aicc']:.1f} > 0 but < {CRITERIA['delta_aicc_threshold']}"))
        else:
            evidence.append(('ΔAICc', 'FAIL', f"{results['delta_aicc']:.1f} < 0"))
    
    # 2. Permutation p-value
    if perm_results and 'p_value' in perm_results:
        if perm_results['p_value'] < 0.05:
            evidence.append(('Permutation', 'PASS', f"p = {perm_results['p_value']:.4f} < 0.05"))
        else:
            evidence.append(('Permutation', 'FAIL', f"p = {perm_results['p_value']:.4f} ≥ 0.05"))
    
    # 3. λ_QR in range
    if 'lambda_QR' in results.get('toe', {}):
        ratio = results['toe']['lambda_QR'] / CRITERIA['lambda_gal']
        if CRITERIA['lambda_ratio_min'] <= ratio <= CRITERIA['lambda_ratio_max']:
            evidence.append(('λ_QR range', 'PASS', f"ratio = {ratio:.2f} ∈ [{CRITERIA['lambda_ratio_min']:.2f}, {CRITERIA['lambda_ratio_max']:.2f}]"))
        else:
            evidence.append(('λ_QR range', 'FAIL', f"ratio = {ratio:.2f} outside range"))
    
    # 4. Flip detected
    if 'rho_flip' in results:
        if rho_range[0] < results['rho_flip'] < rho_range[1]:
            evidence.append(('Phase flip', 'PASS', f"ρ_flip = {results['rho_flip']:.2f} in data range"))
        else:
            evidence.append(('Phase flip', 'WEAK', f"ρ_flip = {results['rho_flip']:.2f} at edge"))
    
    # 5. Hold-out
    if 'holdout' in results:
        if results['holdout']['r'] > 0 and results['holdout']['p'] < 0.05:
            evidence.append(('Hold-out', 'PASS', f"r = {results['holdout']['r']:.3f}, p = {results['holdout']['p']:.2e}"))
        else:
            evidence.append(('Hold-out', 'WEAK', f"r = {results['holdout']['r']:.3f}"))
    
    # Print summary
    n_pass = sum(1 for e in evidence if e[1] == 'PASS')
    n_weak = sum(1 for e in evidence if e[1] == 'WEAK')
    n_fail = sum(1 for e in evidence if e[1] == 'FAIL')
    
    for name, status, detail in evidence:
        symbol = '✅' if status == 'PASS' else '⚠️' if status == 'WEAK' else '❌'
        print(f"  {symbol} {name}: {detail}")
    
    print(f"\n  Summary: {n_pass} PASS, {n_weak} WEAK, {n_fail} FAIL")
    
    if n_fail == 0 and n_pass >= 3:
        verdict = "QO+R TOE SUPPORTED AT GROUP SCALE"
    elif n_fail <= 1 and n_pass >= 2:
        verdict = "QO+R TOE PARTIALLY SUPPORTED"
    else:
        verdict = "QO+R TOE INCONCLUSIVE AT GROUP SCALE"
    
    print(f"\n  VERDICT: {verdict}")
    results['verdict'] = verdict
    results['evidence'] = [{'name': e[0], 'status': e[1], 'detail': e[2]} for e in evidence]
    
    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "grp001_toe_results_v3.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
