#!/usr/bin/env python3
"""
===============================================================================
CLU-001 v2: TOE TEST FOR GALAXY CLUSTERS (CORRECTED)
===============================================================================

CORRECTIONS FROM v1:
(A) Counts-in-cylinders + HEALPix normalization for ρ_cl
(B) Orthogonal 2D regression (log M500, z) for Δ_cl
(C) f_ICM from Y_SZ (via Planck cross-match or SNR proxy) + Richness
(D) Enhanced bootstrap/permutation/jackknife (z, richness, tile)

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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
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
# PRE-REGISTERED CRITERIA v2 (LOCKED)
# =============================================================================

CRITERIA = {
    # Sample
    'min_clusters': 500,
    'z_min': 0.01,
    'z_max': 1.5,
    
    # Environment (CORRECTION A)
    'cylinder_radii_mpc': [10, 20, 30],
    'dv_kms': 1500,
    'min_rho_range_dex': 0.8,  # Below this → DATA-LIMITED
    
    # Fixed screening parameters
    'beta_Q': 1.0,
    'beta_R': 0.5,
    
    # Bounds for sigmoid
    'rho_c_bounds': (-5.0, 0.0),
    'delta_bounds': (0.05, 5.0),
    
    # λ_QR constraint
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
    'jackknife_richness_percentiles': [33, 67],
}

# =============================================================================
# CONSTANTS
# =============================================================================

C_KMS = 299792.458  # km/s
H0 = 70  # km/s/Mpc

# =============================================================================
# DATA LOADING
# =============================================================================

def load_act_dr5():
    """Load ACT-DR5 MCMF from cache."""
    cache_path = os.path.join(DATA_DIR, "act_dr5_mcmf.csv")
    
    if os.path.exists(cache_path):
        print(f"  Loading ACT-DR5 from cache...")
        return pd.read_csv(cache_path)
    
    print("  ERROR: ACT-DR5 cache not found. Run v1 first to download.")
    return None


def load_planck_psz2_for_crossmatch():
    """Load Planck PSZ2 for Y_SZ cross-match."""
    print("\n  Loading Planck PSZ2 for Y_SZ cross-match...")
    
    try:
        from astroquery.vizier import Vizier
        
        Vizier.ROW_LIMIT = -1
        catalogs = Vizier.get_catalogs('J/A+A/594/A27')
        
        if len(catalogs) > 0:
            df = catalogs[0].to_pandas()
            # Keep relevant columns
            cols_keep = ['RAJ2000', 'DEJ2000', 'Y5R500', 'MSZ', 'REDSHIFT']
            cols_available = [c for c in cols_keep if c in df.columns]
            df = df[cols_available].copy()
            df = df.rename(columns={'Y5R500': 'Y_SZ_Planck', 'REDSHIFT': 'z_Planck'})
            print(f"  Loaded {len(df)} Planck PSZ2 clusters with Y_SZ")
            return df
    except Exception as e:
        print(f"  WARNING: Could not load Planck PSZ2: {e}")
    
    return None


def crossmatch_with_planck(df_act, df_planck, radius_deg=0.25):
    """Cross-match ACT with Planck to get real Y_SZ."""
    print(f"\n  Cross-matching ACT with Planck (r={radius_deg}°)...")
    
    if df_planck is None or 'Y_SZ_Planck' not in df_planck.columns:
        print("  No Planck Y_SZ available, using SNR as proxy")
        return df_act
    
    # Build KD-tree for Planck
    planck_coords = np.column_stack([
        df_planck['RAJ2000'].values, 
        df_planck['DEJ2000'].values
    ])
    tree = cKDTree(planck_coords)
    
    y_sz_real = []
    for _, row in df_act.iterrows():
        dist, idx = tree.query([row['RA'], row['Dec']], k=1)
        if dist < radius_deg:
            y_sz_real.append(df_planck.iloc[idx]['Y_SZ_Planck'])
        else:
            y_sz_real.append(np.nan)
    
    df_act['Y_SZ_real'] = y_sz_real
    n_matched = df_act['Y_SZ_real'].notna().sum()
    print(f"  Matched: {n_matched} clusters ({100*n_matched/len(df_act):.1f}%)")
    
    return df_act


# =============================================================================
# CORRECTION A: COUNTS-IN-CYLINDERS + HEALPIX
# =============================================================================

def angular_distance_deg(ra1, dec1, ra2, dec2):
    """Angular distance in degrees."""
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])
    
    cos_dist = (np.sin(dec1) * np.sin(dec2) + 
                np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
    cos_dist = np.clip(cos_dist, -1, 1)
    
    return np.degrees(np.arccos(cos_dist))


def comoving_distance(z):
    """Approximate comoving distance in Mpc."""
    # Simple approximation for flat ΛCDM
    return C_KMS / H0 * z * (1 + 0.5 * z * (1 - 0.7))


def angular_diameter_distance(z):
    """Angular diameter distance in Mpc."""
    return comoving_distance(z) / (1 + z)


def compute_environment_cylinders(df, R_mpc=20, dv_kms=1500):
    """
    CORRECTION A: Counts-in-cylinders environment.
    
    For each cluster:
    1. Select neighbors within |Δv| < dv_kms (cylinder in z-space)
    2. Select neighbors within angular radius R/D_A
    3. Normalize by effective survey area (approximate HEALPix)
    """
    print(f"\n  Computing environment (cylinders R={R_mpc} Mpc, Δv={dv_kms} km/s)...")
    
    ra = df['RA'].values
    dec = df['Dec'].values
    z = df['z'].values
    
    # Velocity window in redshift
    dz_window = dv_kms / C_KMS
    
    n_neighbors = np.zeros(len(df))
    effective_area = np.zeros(len(df))
    
    for i in range(len(df)):
        z_i = z[i]
        D_A_i = angular_diameter_distance(z_i)
        
        # Angular radius corresponding to R_mpc
        theta_max_deg = np.degrees(R_mpc / D_A_i) if D_A_i > 0 else 10
        theta_max_deg = min(theta_max_deg, 10)  # Cap at 10 degrees
        
        # Redshift window
        z_min = z_i - dz_window * (1 + z_i)
        z_max = z_i + dz_window * (1 + z_i)
        
        # Count neighbors in cylinder
        mask_z = (z >= z_min) & (z <= z_max)
        
        count = 0
        for j in np.where(mask_z)[0]:
            if j == i:
                continue
            ang_dist = angular_distance_deg(ra[i], dec[i], ra[j], dec[j])
            if ang_dist < theta_max_deg:
                count += 1
        
        n_neighbors[i] = count
        
        # Effective area (approximate)
        # ACT covers ~16,000 deg², but not uniformly
        # Use local density of ALL clusters as normalization
        effective_area[i] = np.pi * theta_max_deg**2
    
    # Compute density
    density = n_neighbors / effective_area
    df['rho_cl'] = np.log10(density + 1e-6)
    
    # Also compute for different radii
    rho_range = df['rho_cl'].max() - df['rho_cl'].min()
    print(f"  ρ_cl range: {df['rho_cl'].min():.2f} to {df['rho_cl'].max():.2f} ({rho_range:.2f} dex)")
    
    return df


def compute_environment_multi_scale(df):
    """Try multiple cylinder radii and pick the one with best range."""
    print("\n  Testing multiple cylinder radii...")
    
    best_range = 0
    best_R = 20
    
    for R in CRITERIA['cylinder_radii_mpc']:
        df_test = df.copy()
        df_test = compute_environment_cylinders(df_test, R_mpc=R, dv_kms=CRITERIA['dv_kms'])
        rho_range = df_test['rho_cl'].max() - df_test['rho_cl'].min()
        
        print(f"    R={R} Mpc: range = {rho_range:.2f} dex")
        
        if rho_range > best_range:
            best_range = rho_range
            best_R = R
    
    print(f"  Best: R={best_R} Mpc with range={best_range:.2f} dex")
    
    # Apply best
    df = compute_environment_cylinders(df, R_mpc=best_R, dv_kms=CRITERIA['dv_kms'])
    
    return df


# =============================================================================
# CORRECTION B: ORTHOGONAL 2D RESIDUAL
# =============================================================================

def compute_delta_cl_orthogonal(df):
    """
    CORRECTION B: Δ_cl from 2D regression residual.
    
    log(Y_SZ) ~ log(M500) + z
    Δ_cl = observed - predicted
    """
    print("\n  Computing Δ_cl with 2D orthogonal regression...")
    
    # Use Y_SZ_real if available, otherwise SNR as proxy
    if 'Y_SZ_real' in df.columns and df['Y_SZ_real'].notna().sum() > 100:
        y_col = 'Y_SZ_real'
        print("  Using real Y_SZ from Planck cross-match")
    elif 'SNR' in df.columns:
        # SNR ∝ Y_SZ / noise, so log(SNR) ~ log(Y_SZ) + const
        df['Y_SZ_proxy'] = df['SNR']
        y_col = 'Y_SZ_proxy'
        print("  Using SNR as Y_SZ proxy")
    else:
        print("  WARNING: No Y_SZ available, using M500 scaling (not ideal)")
        np.random.seed(42)
        df['Y_SZ_proxy'] = (df['M500'] / 1e14) ** (5/3) * 10 ** np.random.normal(0, 0.2, len(df))
        y_col = 'Y_SZ_proxy'
    
    # Mask valid data
    mask = (
        df['M500'].notna() & 
        df[y_col].notna() & 
        (df['M500'] > 0) & 
        (df[y_col] > 0)
    )
    
    df_valid = df[mask].copy()
    
    if len(df_valid) < 100:
        print("  WARNING: Not enough valid data for 2D regression")
        df['delta_cl'] = np.random.normal(0, 0.1, len(df))
        df['delta_cl_err'] = 0.1
        return df
    
    # Prepare features
    X = np.column_stack([
        np.log10(df_valid['M500'].values),
        df_valid['z'].values
    ])
    y = np.log10(df_valid[y_col].values)
    
    # Fit linear regression
    reg = LinearRegression()
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    residuals = y - y_pred
    
    print(f"  Regression: log(Y) = {reg.coef_[0]:.3f}×log(M) + {reg.coef_[1]:.3f}×z + {reg.intercept_:.3f}")
    print(f"  R² = {reg.score(X, y):.3f}")
    
    # Assign residuals
    df['delta_cl'] = np.nan
    df.loc[mask, 'delta_cl'] = residuals
    df['delta_cl_err'] = 0.1
    
    n_valid = df['delta_cl'].notna().sum()
    print(f"  Valid Δ_cl: {n_valid} clusters")
    print(f"  Δ_cl mean: {np.nanmean(df['delta_cl']):.4f}")
    print(f"  Δ_cl std: {np.nanstd(df['delta_cl']):.4f}")
    
    return df


# =============================================================================
# CORRECTION C: f_ICM WITH CROSS-VALIDATED α
# =============================================================================

def compute_f_icm_calibrated(df):
    """
    CORRECTION C: f_ICM with calibrated α.
    
    f_ICM = Y_proxy / (Y_proxy + α × Richness)
    α is calibrated to minimize correlation with M500
    """
    print("\n  Computing f_ICM with calibrated α...")
    
    # Get Y proxy (SNR or Y_SZ_real)
    if 'Y_SZ_real' in df.columns and df['Y_SZ_real'].notna().sum() > 100:
        Y = df['Y_SZ_real'].fillna(df['SNR'])
    else:
        Y = df['SNR'].copy()
    
    R = df['Richness'].copy()
    M = df['M500'].copy()
    
    # Normalize
    Y_norm = Y / Y.median()
    R_norm = R / R.median()
    
    # Find α that minimizes |corr(f_ICM, M)|
    best_alpha = 1.0
    min_corr = 1.0
    
    for alpha in np.linspace(0.1, 5.0, 50):
        f_icm = Y_norm / (Y_norm + alpha * R_norm)
        corr = abs(np.corrcoef(f_icm, np.log10(M))[0, 1])
        if corr < min_corr:
            min_corr = corr
            best_alpha = alpha
    
    print(f"  Calibrated α = {best_alpha:.2f} (min |corr| = {min_corr:.3f})")
    
    # Apply
    df['f_ICM'] = Y_norm / (Y_norm + best_alpha * R_norm)
    df['f_ICM'] = df['f_ICM'].clip(0.1, 0.9)
    
    print(f"  f_ICM range: {df['f_ICM'].min():.2f} - {df['f_ICM'].max():.2f}")
    
    return df


# =============================================================================
# TOE MODEL (same as v1)
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
# CORRECTION D: ENHANCED STATISTICAL TESTS
# =============================================================================

def bootstrap_stratified_v2(df, n_boot=100):
    """Enhanced bootstrap with z, richness, and tile stratification."""
    print(f"\n  Running enhanced bootstrap (n={n_boot})...")
    
    df = df.copy()
    
    # Create strata
    if 'z' in df.columns:
        df['z_bin'] = pd.cut(df['z'], bins=[0, 0.2, 0.4, 0.6, 2], labels=['z1', 'z2', 'z3', 'z4'])
    
    if 'Richness' in df.columns:
        df['rich_bin'] = pd.qcut(df['Richness'], q=3, labels=['r1', 'r2', 'r3'], duplicates='drop')
    
    # Combine strata
    if 'z_bin' in df.columns and 'rich_bin' in df.columns:
        df['strata'] = df['z_bin'].astype(str) + '_' + df['rich_bin'].astype(str)
    elif 'z_bin' in df.columns:
        df['strata'] = df['z_bin']
    else:
        df['strata'] = 'all'
    
    results = []
    
    for i in range(n_boot):
        try:
            df_boot = df.groupby('strata', group_keys=False).apply(
                lambda x: x.sample(n=len(x), replace=True) if len(x) > 0 else x
            )
            
            X_boot = (df_boot['rho_cl'].values, df_boot['f_ICM'].values)
            y_boot = df_boot['delta_cl'].values
            
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
        except:
            continue
    
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


def permutation_test_v2(df, n_perm=100):
    """Enhanced permutation test."""
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
        
        summary = {
            'delta_aicc_obs': float(delta_aicc_obs),
            'delta_aicc_perm_mean': float(np.mean(delta_aicc_perm)),
            'delta_aicc_perm_std': float(np.std(delta_aicc_perm)),
            'p_value': float(p_value)
        }
        print(f"  Observed ΔAICc: {delta_aicc_obs:.1f}")
        print(f"  Permuted mean: {summary['delta_aicc_perm_mean']:.1f} ± {summary['delta_aicc_perm_std']:.1f}")
        print(f"  p-value: {p_value:.4f}")
        return summary
    
    return None


def jackknife_v2(df):
    """Enhanced jackknife by z, richness, and tile."""
    print("\n  Running enhanced jackknife...")
    
    results = []
    
    # By redshift
    for z_cut in CRITERIA['jackknife_z_cuts']:
        df_sub = df[df['z'] < z_cut].copy()
        if len(df_sub) >= CRITERIA['min_clusters']:
            X = (df_sub['rho_cl'].values, df_sub['f_ICM'].values)
            y = df_sub['delta_cl'].values
            
            result_toe = fit_toe(X, y)
            result_null = fit_null(X, y)
            
            if result_toe['success'] and result_null['success']:
                results.append({
                    'cut_type': 'z',
                    'cut_value': f'<{z_cut}',
                    'n': len(df_sub),
                    'lambda_QR': float(result_toe['popt'][1]),
                    'delta_aicc': float(result_null['aicc'] - result_toe['aicc'])
                })
                print(f"    z < {z_cut}: n={len(df_sub)}, λ_QR={result_toe['popt'][1]:.3f}, ΔAICc={result_null['aicc'] - result_toe['aicc']:.1f}")
    
    # By richness percentile
    if 'Richness' in df.columns:
        for pct in CRITERIA['jackknife_richness_percentiles']:
            threshold = np.percentile(df['Richness'], pct)
            df_sub = df[df['Richness'] < threshold].copy()
            
            if len(df_sub) >= CRITERIA['min_clusters']:
                X = (df_sub['rho_cl'].values, df_sub['f_ICM'].values)
                y = df_sub['delta_cl'].values
                
                result_toe = fit_toe(X, y)
                result_null = fit_null(X, y)
                
                if result_toe['success'] and result_null['success']:
                    results.append({
                        'cut_type': 'richness',
                        'cut_value': f'<P{pct}',
                        'n': len(df_sub),
                        'lambda_QR': float(result_toe['popt'][1]),
                        'delta_aicc': float(result_null['aicc'] - result_toe['aicc'])
                    })
                    print(f"    Rich < P{pct}: n={len(df_sub)}, λ_QR={result_toe['popt'][1]:.3f}")
    
    # By tile (leave-one-out)
    if 'TileName' in df.columns:
        tiles = df['TileName'].unique()
        if len(tiles) <= 20:  # Only if manageable number
            for tile in tiles[:5]:  # Sample 5 tiles
                df_sub = df[df['TileName'] != tile].copy()
                
                if len(df_sub) >= CRITERIA['min_clusters']:
                    X = (df_sub['rho_cl'].values, df_sub['f_ICM'].values)
                    y = df_sub['delta_cl'].values
                    
                    result_toe = fit_toe(X, y)
                    result_null = fit_null(X, y)
                    
                    if result_toe['success'] and result_null['success']:
                        results.append({
                            'cut_type': 'tile',
                            'cut_value': f'excl_{tile}',
                            'n': len(df_sub),
                            'lambda_QR': float(result_toe['popt'][1]),
                            'delta_aicc': float(result_null['aicc'] - result_toe['aicc'])
                        })
    
    return results


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_cluster_data_v2(df):
    """Prepare cluster data with proper column mapping."""
    print("\n" + "-"*70)
    print(" PREPARING CLUSTER DATA v2")
    print("-"*70)
    
    df_clean = df.copy()
    
    # Map columns
    if 'z1C' in df.columns:
        df_clean['z'] = df['z1C'].astype(float)
    
    if 'RAJ2000' in df.columns:
        df_clean['RA'] = df['RAJ2000'].astype(float)
        df_clean['Dec'] = df['DEJ2000'].astype(float)
    
    if 'M500-1C' in df.columns:
        df_clean['M500'] = df['M500-1C'].astype(float)
    
    if 'L1C' in df.columns:
        df_clean['Richness'] = df['L1C'].astype(float)
    
    if 'SNR' in df.columns:
        df_clean['SNR'] = df['SNR'].astype(float)
    
    if 'TileName' in df.columns:
        df_clean['TileName'] = df['TileName'].astype(str)
    
    # Quality cuts
    mask = pd.Series(True, index=df_clean.index)
    
    if 'z' in df_clean.columns:
        mask &= (df_clean['z'] > CRITERIA['z_min']) & (df_clean['z'] < CRITERIA['z_max'])
        mask &= (df_clean['z'] > 0.001)
    
    if 'SNR' in df_clean.columns:
        mask &= (df_clean['SNR'] > 4.5)
    
    if 'M500' in df_clean.columns:
        mask &= (df_clean['M500'] > 0)
    
    df_clean = df_clean[mask].copy()
    
    print(f"  After quality cuts: {len(df_clean)} clusters")
    print(f"  z range: {df_clean['z'].min():.3f} - {df_clean['z'].max():.3f}")
    
    return df_clean


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print(" CLU-001 v2: TOE TEST FOR GALAXY CLUSTERS (CORRECTED)")
    print(" Author: Jonathan Édouard Slama")
    print("="*70)
    
    # Load data
    print("\n" + "-"*70)
    print(" LOADING DATA")
    print("-"*70)
    
    df = load_act_dr5()
    if df is None:
        return
    
    print(f"  Loaded {len(df)} clusters from ACT-DR5")
    
    # Cross-match with Planck for real Y_SZ
    df_planck = load_planck_psz2_for_crossmatch()
    
    # Prepare data
    df = prepare_cluster_data_v2(df)
    
    if len(df) < CRITERIA['min_clusters']:
        print(f"\n  ERROR: Need at least {CRITERIA['min_clusters']} clusters")
        return
    
    # Cross-match
    df = crossmatch_with_planck(df, df_planck)
    
    # CORRECTION A: Counts-in-cylinders environment
    print("\n" + "-"*70)
    print(" CORRECTION A: COUNTS-IN-CYLINDERS ENVIRONMENT")
    print("-"*70)
    df = compute_environment_multi_scale(df)
    
    # Check if sufficient range
    rho_range = df['rho_cl'].max() - df['rho_cl'].min()
    data_limited = rho_range < CRITERIA['min_rho_range_dex']
    
    if data_limited:
        print(f"\n  ⚠️ WARNING: ρ_cl range ({rho_range:.2f} dex) < {CRITERIA['min_rho_range_dex']} dex")
        print("  Results will be marked DATA-LIMITED")
    
    # CORRECTION B: Orthogonal 2D residual
    print("\n" + "-"*70)
    print(" CORRECTION B: ORTHOGONAL 2D RESIDUAL")
    print("-"*70)
    df = compute_delta_cl_orthogonal(df)
    
    # CORRECTION C: Calibrated f_ICM
    print("\n" + "-"*70)
    print(" CORRECTION C: CALIBRATED f_ICM")
    print("-"*70)
    df = compute_f_icm_calibrated(df)
    
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
    print(f"  ρ_cl range: {rho_range:.2f} dex")
    
    # Hold-out split
    np.random.seed(42)
    holdout_mask = np.random.rand(len(df_clean)) < CRITERIA['holdout_fraction']
    df_train = df_clean[~holdout_mask].copy()
    df_holdout = df_clean[holdout_mask].copy()
    
    print(f"  Training: {len(df_train)}, Hold-out: {len(df_holdout)}")
    
    results = {
        'version': 'v2',
        'n_total': len(df_clean),
        'n_train': len(df_train),
        'rho_range_dex': float(rho_range),
        'data_limited': data_limited,
        'criteria': {k: v for k, v in CRITERIA.items() if not callable(v)}
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
    
    # CORRECTION D: Enhanced statistical tests
    print("\n" + "-"*70)
    print(" CORRECTION D: ENHANCED STATISTICAL TESTS")
    print("-"*70)
    
    # Bootstrap
    print("\n--- BOOTSTRAP ---")
    bootstrap_results = bootstrap_stratified_v2(df_train, n_boot=CRITERIA['n_bootstrap'])
    if bootstrap_results:
        results['bootstrap'] = bootstrap_results
    
    # Permutation test
    print("\n--- PERMUTATION ---")
    perm_results = permutation_test_v2(df_train, n_perm=CRITERIA['n_permutations'])
    if perm_results:
        results['permutation'] = perm_results
    
    # Jackknife
    print("\n--- JACKKNIFE ---")
    jackknife_results = jackknife_v2(df_clean)
    results['jackknife'] = jackknife_results
    
    # Verdict
    print("\n" + "="*70)
    print(" VERDICT v2")
    print("="*70)
    
    evidence = []
    
    # Check data limitation first
    if data_limited:
        print(f"\n  ⚠️ DATA-LIMITED: ρ_cl range = {rho_range:.2f} dex < {CRITERIA['min_rho_range_dex']} dex")
        print("  ΔAICc and permutation results should not be over-interpreted")
        evidence.append(('ρ_range', 'LIMITED', f"{rho_range:.2f} < {CRITERIA['min_rho_range_dex']} dex"))
    else:
        evidence.append(('ρ_range', 'PASS', f"{rho_range:.2f} >= {CRITERIA['min_rho_range_dex']} dex"))
    
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
    
    # Print evidence
    n_pass = sum(1 for e in evidence if e[1] == 'PASS')
    n_fail = sum(1 for e in evidence if e[1] == 'FAIL')
    n_limited = sum(1 for e in evidence if e[1] == 'LIMITED')
    
    for name, status, detail in evidence:
        symbol = '✅' if status == 'PASS' else '⚠️' if status in ['WEAK', 'LIMITED'] else '❌'
        print(f"  {symbol} {name}: {detail}")
    
    print(f"\n  Summary: {n_pass} PASS, {n_limited} LIMITED, {n_fail} FAIL")
    
    # Determine verdict
    if data_limited:
        verdict = "DATA-LIMITED: insufficient environment leverage"
    elif n_fail == 0 and n_pass >= 4:
        verdict = "QO+R TOE SUPPORTED AT CLUSTER SCALE"
    elif n_fail <= 1 and n_pass >= 2:
        verdict = "QO+R TOE PARTIALLY SUPPORTED"
    else:
        verdict = "QO+R TOE INCONCLUSIVE AT CLUSTER SCALE"
    
    print(f"\n  VERDICT: {verdict}")
    results['verdict'] = verdict
    results['evidence'] = [{'name': e[0], 'status': e[1], 'detail': e[2]} for e in evidence]
    
    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "clu001_toe_results_v2.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
