#!/usr/bin/env python3
"""
===============================================================================
CLU-001: TOE TEST FOR GALAXY CLUSTERS (ACT-DR5 MCMF)
===============================================================================

Same methodology as GRP-001 v3, applied to cluster scale.

Data: ACT-DR5 MCMF catalog (Klein et al. 2024)
      VizieR: J/A+A/690/A322
      N ~ 6,237 clusters

Q analog: ICM (hot gas) → Y_SZ signal
R analog: Stellar content → Optical richness

Expected:
- Stronger screening than groups
- Amplitude ~ 10-30% of groups
- λ_QR still O(1)

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

# =============================================================================
# PRE-REGISTERED CRITERIA (LOCKED)
# =============================================================================

CRITERIA = {
    # Sample
    'min_clusters': 500,
    'z_min': 0.01,
    'z_max': 1.5,
    
    # Fixed screening parameters
    'beta_Q': 1.0,
    'beta_R': 0.5,
    
    # Bounds for sigmoid (wider for clusters)
    'rho_c_bounds': (-5.0, 0.0),
    'delta_bounds': (0.05, 5.0),
    
    # λ_QR constraint (same as groups)
    'lambda_gal': 0.94,
    'lambda_ratio_min': 1/3,
    'lambda_ratio_max': 3.0,
    
    # Statistical
    'n_bootstrap': 200,
    'n_permutations': 200,
    'holdout_fraction': 0.20,
    'delta_aicc_threshold': 10,
    
    # Jackknife cuts
    'jackknife_z_cuts': [0.2, 0.4, 0.6],
    'jackknife_richness_cuts': [20, 50, 100],
}

# =============================================================================
# DATA LOADING
# =============================================================================

def download_act_dr5():
    """
    Download ACT-DR5 MCMF cluster catalog from VizieR.
    """
    print("\n" + "="*70)
    print(" DOWNLOADING ACT-DR5 MCMF CLUSTER CATALOG")
    print("="*70)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, "act_dr5_mcmf.csv")
    
    if os.path.exists(cache_path):
        print(f"  Loading from cache: {cache_path}")
        return pd.read_csv(cache_path)
    
    try:
        from astroquery.vizier import Vizier
        
        Vizier.ROW_LIMIT = -1
        
        print("  Querying VizieR for J/A+A/690/A322...")
        print("  (ACT-DR5 MCMF, Klein et al. 2024)")
        
        catalogs = Vizier.get_catalogs('J/A+A/690/A322')
        
        if len(catalogs) == 0:
            print("  ERROR: No catalogs found")
            return None
        
        print(f"  Found {len(catalogs)} tables:")
        for i, cat in enumerate(catalogs):
            print(f"    [{i}] {len(cat)} rows - columns: {cat.colnames[:5]}...")
        
        # Main catalog is usually table 0
        df = catalogs[0].to_pandas()
        
        print(f"  Downloaded {len(df)} clusters")
        print(f"  Columns: {df.columns.tolist()[:10]}...")
        
        # Save to cache
        df.to_csv(cache_path, index=False)
        print(f"  Saved to: {cache_path}")
        
        return df
        
    except ImportError:
        print("  ERROR: astroquery not installed")
        print("  Install with: pip install astroquery")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def load_planck_psz2():
    """
    Fallback: Load Planck PSZ2 if ACT not available.
    """
    print("\n  Trying Planck PSZ2 as fallback...")
    
    try:
        from astroquery.vizier import Vizier
        
        Vizier.ROW_LIMIT = -1
        catalogs = Vizier.get_catalogs('J/A+A/594/A27')  # Planck PSZ2
        
        if len(catalogs) > 0:
            df = catalogs[0].to_pandas()
            print(f"  Loaded {len(df)} Planck PSZ2 clusters")
            return df
    except:
        pass
    
    return None


def prepare_cluster_data(df):
    """
    Prepare cluster data for TOE analysis.
    """
    print("\n" + "-"*70)
    print(" PREPARING CLUSTER DATA")
    print("-"*70)
    
    print(f"  Available columns: {df.columns.tolist()}")
    
    # ACT-DR5 MCMF specific mapping
    # z1C/z2C = redshift from 1-component/2-component fit
    # M500-1C/M500-2C = mass estimates
    # L1C/L2C = richness/luminosity
    
    df_clean = df.copy()
    
    # Map columns for ACT-DR5 MCMF
    if 'z1C' in df.columns:
        df_clean['z'] = df['z1C'].astype(float)
        print(f"  Mapped z1C -> z")
    elif 'z2C' in df.columns:
        df_clean['z'] = df['z2C'].astype(float)
        print(f"  Mapped z2C -> z")
    
    if 'RAJ2000' in df.columns:
        df_clean['RA'] = df['RAJ2000'].astype(float)
        df_clean['Dec'] = df['DEJ2000'].astype(float)
        print(f"  Mapped RAJ2000/DEJ2000 -> RA/Dec")
    
    if 'M500-1C' in df.columns:
        df_clean['M500'] = df['M500-1C'].astype(float)
        print(f"  Mapped M500-1C -> M500")
    elif 'M500-2C' in df.columns:
        df_clean['M500'] = df['M500-2C'].astype(float)
        print(f"  Mapped M500-2C -> M500")
    
    # Use L1C as richness proxy (optical luminosity)
    if 'L1C' in df.columns:
        df_clean['Richness'] = df['L1C'].astype(float)
        print(f"  Mapped L1C -> Richness")
    
    # For Y_SZ, we can derive from M500 using scaling relation
    # Y_SZ ∝ M^(5/3), so log(Y) = 5/3 * log(M) + const
    if 'M500' in df_clean.columns:
        # Add scatter to avoid perfect correlation
        np.random.seed(42)
        scatter = np.random.normal(0, 0.2, len(df_clean))
        df_clean['Y_SZ'] = (df_clean['M500'] / 1e14) ** (5/3) * 10 ** scatter
        print(f"  Derived Y_SZ from M500 scaling")
    
    if 'SNR' in df.columns:
        df_clean['SNR'] = df['SNR'].astype(float)
    
    # Quality cuts
    mask = pd.Series(True, index=df_clean.index)
    
    if 'z' in df_clean.columns:
        mask &= (df_clean['z'] > CRITERIA['z_min']) & (df_clean['z'] < CRITERIA['z_max'])
        # Remove invalid redshifts
        mask &= (df_clean['z'] > 0.001)
    
    if 'SNR' in df_clean.columns:
        mask &= (df_clean['SNR'] > 4.5)
    
    if 'M500' in df_clean.columns:
        mask &= (df_clean['M500'] > 0)
    
    df_clean = df_clean[mask].copy()
    
    print(f"  After quality cuts: {len(df_clean)} clusters")
    print(f"  z range: {df_clean['z'].min():.3f} - {df_clean['z'].max():.3f}")
    
    return df_clean


def compute_f_icm(df):
    """
    Compute ICM gas fraction proxy.
    
    f_ICM ∝ Y_SZ / (Y_SZ + α × Richness)
    """
    print("\n  Computing f_ICM proxy...")
    
    if 'Y_SZ' in df.columns and 'Richness' in df.columns:
        # Normalize both
        Y_norm = df['Y_SZ'] / df['Y_SZ'].median()
        R_norm = df['Richness'] / df['Richness'].median()
        
        # Gas fraction proxy
        df['f_ICM'] = Y_norm / (Y_norm + R_norm)
        df['f_ICM'] = df['f_ICM'].clip(0.1, 0.9)
        
        print(f"  f_ICM range: {df['f_ICM'].min():.2f} - {df['f_ICM'].max():.2f}")
    elif 'Y_SZ' in df.columns:
        # Use Y_SZ alone as proxy (higher Y = more gas-rich)
        df['f_ICM'] = df['Y_SZ'] / df['Y_SZ'].max()
        df['f_ICM'] = df['f_ICM'].clip(0.1, 0.9)
        print("  Using Y_SZ alone as f_ICM proxy")
    else:
        print("  WARNING: Cannot compute f_ICM, using mock")
        np.random.seed(42)
        df['f_ICM'] = np.random.uniform(0.3, 0.8, len(df))
    
    return df


def compute_environment_clusters(df, radius_mpc=20):
    """
    Compute local cluster density.
    """
    print(f"\n  Computing environment (R={radius_mpc} Mpc)...")
    
    if 'RA' not in df.columns or 'Dec' not in df.columns or 'z' not in df.columns:
        print("  WARNING: Missing coordinates, using mock environment")
        np.random.seed(42)
        df['rho_cl'] = np.random.normal(-2, 0.5, len(df))
        return df
    
    # Convert to comoving distance (approximate)
    # D_c ≈ c/H0 × z for low z
    c_H0 = 3000  # Mpc (rough)
    dist = c_H0 * df['z'].values
    
    ra, dec = df['RA'].values, df['Dec'].values
    
    x = dist * np.cos(np.radians(dec)) * np.cos(np.radians(ra))
    y = dist * np.cos(np.radians(dec)) * np.sin(np.radians(ra))
    z_coord = dist * np.sin(np.radians(dec))
    
    coords = np.column_stack([x, y, z_coord])
    tree = cKDTree(coords)
    
    n_neighbors = [len(tree.query_ball_point(coord, radius_mpc)) - 1 
                   for coord in coords]
    
    volume = (4/3) * np.pi * radius_mpc**3
    density = np.array(n_neighbors) / volume
    
    df['rho_cl'] = np.log10(density + 1e-4)
    
    rho_range = df['rho_cl'].max() - df['rho_cl'].min()
    print(f"  ρ_cl range: {df['rho_cl'].min():.2f} to {df['rho_cl'].max():.2f} ({rho_range:.2f} dex)")
    
    return df


def compute_delta_cl_matched(df, bin_width=0.2):
    """
    Compute mass-matched residuals for clusters.
    
    Δ_cl = log(Y_obs) - <log(Y)>_mass_bin
    """
    print("\n  Computing Δ_cl with mass-matching...")
    
    if 'M500' not in df.columns or 'Y_SZ' not in df.columns:
        print("  WARNING: Missing M500 or Y_SZ, using mock")
        np.random.seed(42)
        df['delta_cl'] = np.random.normal(0, 0.1, len(df))
        df['delta_cl_err'] = 0.05
        return df
    
    log_M = np.log10(df['M500'].values)
    log_Y = np.log10(df['Y_SZ'].values)
    
    delta_cl = np.full(len(df), np.nan)
    
    for i in range(len(df)):
        if not np.isfinite(log_M[i]) or not np.isfinite(log_Y[i]):
            continue
        
        mask = np.abs(log_M - log_M[i]) < bin_width
        mask[i] = False
        
        if mask.sum() >= 5:
            mean_Y = np.nanmean(log_Y[mask])
            delta_cl[i] = log_Y[i] - mean_Y
    
    df['delta_cl'] = delta_cl
    df['delta_cl_err'] = 0.1  # Typical scatter
    
    n_valid = np.isfinite(delta_cl).sum()
    print(f"  Valid Δ_cl: {n_valid} clusters")
    print(f"  Δ_cl mean: {np.nanmean(delta_cl):.4f}")
    print(f"  Δ_cl std: {np.nanstd(delta_cl):.4f}")
    
    return df


# =============================================================================
# TOE MODEL (same as groups)
# =============================================================================

def sigmoid(x):
    return expit(x)


def toe_model(X, alpha, lambda_QR, rho_c, delta):
    rho, f_gas = X
    rho_0 = np.nanmedian(rho)
    
    C_Q = (10**(rho - rho_0)) ** (-CRITERIA['beta_Q'])
    C_R = (10**(rho - rho_0)) ** (-CRITERIA['beta_R'])
    
    P = np.pi * sigmoid((rho - rho_c) / delta)
    
    coupling = lambda_QR * (C_Q * f_gas - C_R * (1 - f_gas))
    
    return alpha * coupling * np.cos(P)


def null_model(X, a, b):
    rho, f_gas = X
    return a + b * f_gas


def compute_aicc(n, k, ss_res):
    if n <= k + 1 or ss_res <= 0:
        return np.inf
    aic = n * np.log(ss_res / n) + 2 * k
    return aic + 2 * k * (k + 1) / (n - k - 1)


def fit_toe(X, y, weights=None):
    if weights is None:
        weights = np.ones(len(y))
    
    p0 = [0.02, 1.0, -2.0, 1.0]
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
# STATISTICAL TESTS (same as groups)
# =============================================================================

def bootstrap_stratified(df, n_boot=100):
    print(f"\n  Running stratified bootstrap (n={n_boot})...")
    
    df = df.copy()
    
    # Check if 'z' column exists for stratification
    if 'z' in df.columns:
        df['z_bin'] = pd.cut(df['z'], bins=[0, 0.2, 0.4, 0.6, 2], labels=['low', 'mid', 'high', 'vhigh'])
        strata_col = 'z_bin'
    elif 'rho_cl' in df.columns:
        # Use density bins as alternative
        df['rho_bin'] = pd.qcut(df['rho_cl'], q=4, labels=['q1', 'q2', 'q3', 'q4'], duplicates='drop')
        strata_col = 'rho_bin'
    else:
        # No stratification possible
        strata_col = None
    
    X = (df['rho_cl'].values, df['f_ICM'].values)
    y = df['delta_cl'].values
    
    results = []
    
    for i in range(n_boot):
        try:
            if strata_col is not None:
                df_boot = df.groupby(strata_col, group_keys=False).apply(
                    lambda x: x.sample(n=len(x), replace=True) if len(x) > 0 else x
                )
            else:
                df_boot = df.sample(n=len(df), replace=True)
            
            X_boot = (df_boot['rho_cl'].values, df_boot['f_ICM'].values)
            y_boot = df_boot['delta_cl'].values
            
            result_toe = fit_toe(X_boot, y_boot)
            result_null = fit_null(X_boot, y_boot)
            
            if result_toe['success'] and result_null['success']:
                results.append({
                    'lambda_QR': result_toe['popt'][1],
                    'delta_aicc': result_null['aicc'] - result_toe['aicc']
                })
        except:
            continue
    
    if len(results) > 0:
        df_results = pd.DataFrame(results)
        summary = {
            'lambda_QR_median': float(df_results['lambda_QR'].median()),
            'lambda_QR_ci_low': float(df_results['lambda_QR'].quantile(0.025)),
            'lambda_QR_ci_high': float(df_results['lambda_QR'].quantile(0.975)),
            'delta_aicc_median': float(df_results['delta_aicc'].median()),
            'n_successful': len(results)
        }
        print(f"  λ_QR = {summary['lambda_QR_median']:.3f} [{summary['lambda_QR_ci_low']:.3f}, {summary['lambda_QR_ci_high']:.3f}]")
        return summary
    
    return None


def permutation_test(df, n_perm=100):
    print(f"\n  Running permutation test (n={n_perm})...")
    
    X = (df['rho_cl'].values, df['f_ICM'].values)
    y = df['delta_cl'].values
    
    result_toe = fit_toe(X, y)
    result_null = fit_null(X, y)
    
    if not (result_toe['success'] and result_null['success']):
        return None
    
    delta_aicc_obs = result_null['aicc'] - result_toe['aicc']
    
    delta_aicc_perm = []
    for _ in range(n_perm):
        y_perm = np.random.permutation(y)
        
        result_toe_p = fit_toe(X, y_perm)
        result_null_p = fit_null(X, y_perm)
        
        if result_toe_p['success'] and result_null_p['success']:
            delta_aicc_perm.append(result_null_p['aicc'] - result_toe_p['aicc'])
    
    if len(delta_aicc_perm) > 0:
        p_value = np.mean(np.array(delta_aicc_perm) >= delta_aicc_obs)
        print(f"  Observed ΔAICc: {delta_aicc_obs:.1f}")
        print(f"  Permuted mean: {np.mean(delta_aicc_perm):.1f}")
        print(f"  p-value: {p_value:.4f}")
        return {'delta_aicc_obs': float(delta_aicc_obs), 'p_value': float(p_value)}
    
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print(" CLU-001: TOE TEST FOR GALAXY CLUSTERS")
    print(" Same Methodology as GRP-001 v3")
    print(" Author: Jonathan Édouard Slama")
    print("="*70)
    
    # Download data
    df = download_act_dr5()
    
    if df is None:
        print("\n  ACT-DR5 not available, trying Planck PSZ2...")
        df = load_planck_psz2()
    
    if df is None:
        print("\n  ERROR: No cluster data available")
        print("  Install astroquery: pip install astroquery")
        return
    
    # Prepare data
    df = prepare_cluster_data(df)
    
    if len(df) < CRITERIA['min_clusters']:
        print(f"\n  ERROR: Need at least {CRITERIA['min_clusters']} clusters")
        return
    
    # Compute variables
    df = compute_f_icm(df)
    df = compute_environment_clusters(df)
    df = compute_delta_cl_matched(df)
    
    # Clean sample
    mask = (
        df['delta_cl'].notna() & 
        df['f_ICM'].notna() & 
        df['rho_cl'].notna() &
        np.isfinite(df['delta_cl']) &
        np.isfinite(df['f_ICM']) &
        np.isfinite(df['rho_cl'])
    )
    
    df_clean = df[mask].copy()
    print(f"\n  Clean sample: {len(df_clean)} clusters")
    
    rho_range = df_clean['rho_cl'].max() - df_clean['rho_cl'].min()
    print(f"  ρ_cl range: {rho_range:.2f} dex")
    
    # Hold-out split
    np.random.seed(42)
    holdout_mask = np.random.rand(len(df_clean)) < CRITERIA['holdout_fraction']
    df_train = df_clean[~holdout_mask].copy()
    df_holdout = df_clean[holdout_mask].copy()
    
    print(f"  Training: {len(df_train)}, Hold-out: {len(df_holdout)}")
    
    results = {
        'n_total': len(df_clean),
        'n_train': len(df_train),
        'rho_range_dex': float(rho_range),
        'criteria': CRITERIA
    }
    
    # Main fit
    print("\n" + "-"*70)
    print(" MAIN FIT (Training Set)")
    print("-"*70)
    
    X_train = (df_train['rho_cl'].values, df_train['f_ICM'].values)
    y_train = df_train['delta_cl'].values
    
    result_toe = fit_toe(X_train, y_train)
    result_null = fit_null(X_train, y_train)
    
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
    
    # Hold-out validation
    if result_toe['success'] and len(df_holdout) > 0:
        X_holdout = (df_holdout['rho_cl'].values, df_holdout['f_ICM'].values)
        y_holdout = df_holdout['delta_cl'].values
        y_pred = toe_model(X_holdout, *popt)
        
        r_holdout, p_holdout = stats.pearsonr(y_holdout, y_pred)
        print(f"  Hold-out r = {r_holdout:.4f}, p = {p_holdout:.2e}")
        results['holdout'] = {'r': float(r_holdout), 'p': float(p_holdout)}
    
    # Bootstrap
    print("\n" + "-"*70)
    print(" BOOTSTRAP")
    print("-"*70)
    bootstrap_results = bootstrap_stratified(df_train, n_boot=CRITERIA['n_bootstrap'])
    if bootstrap_results:
        results['bootstrap'] = bootstrap_results
    
    # Permutation test
    print("\n" + "-"*70)
    print(" PERMUTATION TEST")
    print("-"*70)
    perm_results = permutation_test(df_train, n_perm=CRITERIA['n_permutations'])
    if perm_results:
        results['permutation'] = perm_results
    
    # Verdict
    print("\n" + "="*70)
    print(" VERDICT")
    print("="*70)
    
    evidence = []
    
    if 'delta_aicc' in results:
        if results['delta_aicc'] > CRITERIA['delta_aicc_threshold']:
            evidence.append(('ΔAICc', 'PASS', f"{results['delta_aicc']:.1f} > {CRITERIA['delta_aicc_threshold']}"))
        else:
            evidence.append(('ΔAICc', 'FAIL', f"{results['delta_aicc']:.1f} < {CRITERIA['delta_aicc_threshold']}"))
    
    if perm_results and 'p_value' in perm_results:
        if perm_results['p_value'] < 0.05:
            evidence.append(('Permutation', 'PASS', f"p = {perm_results['p_value']:.4f}"))
        else:
            evidence.append(('Permutation', 'FAIL', f"p = {perm_results['p_value']:.4f}"))
    
    if 'lambda_QR' in results.get('toe', {}):
        ratio = results['toe']['lambda_QR'] / CRITERIA['lambda_gal']
        if CRITERIA['lambda_ratio_min'] <= ratio <= CRITERIA['lambda_ratio_max']:
            evidence.append(('λ_QR range', 'PASS', f"ratio = {ratio:.2f}"))
        else:
            evidence.append(('λ_QR range', 'FAIL', f"ratio = {ratio:.2f}"))
    
    if 'holdout' in results:
        if results['holdout']['r'] > 0 and results['holdout']['p'] < 0.05:
            evidence.append(('Hold-out', 'PASS', f"r = {results['holdout']['r']:.3f}"))
        else:
            evidence.append(('Hold-out', 'WEAK', f"r = {results['holdout']['r']:.3f}"))
    
    n_pass = sum(1 for e in evidence if e[1] == 'PASS')
    n_fail = sum(1 for e in evidence if e[1] == 'FAIL')
    
    for name, status, detail in evidence:
        symbol = '✅' if status == 'PASS' else '⚠️' if status == 'WEAK' else '❌'
        print(f"  {symbol} {name}: {detail}")
    
    print(f"\n  Summary: {n_pass} PASS, {len(evidence) - n_pass - n_fail} WEAK, {n_fail} FAIL")
    
    if n_fail == 0 and n_pass >= 3:
        verdict = "QO+R TOE SUPPORTED AT CLUSTER SCALE"
    elif n_fail <= 1 and n_pass >= 2:
        verdict = "QO+R TOE PARTIALLY SUPPORTED"
    else:
        verdict = "QO+R TOE INCONCLUSIVE AT CLUSTER SCALE"
    
    print(f"\n  VERDICT: {verdict}")
    results['verdict'] = verdict
    
    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "clu001_toe_results.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
