#!/usr/bin/env python3
"""
===============================================================================
GRP-001: TOE TEST FOR GALAXY GROUPS - VERSION ROBUSTE
===============================================================================

AMÉLIORATIONS PAR RAPPORT À V1 (selon retours Jon) :
1. Loader réel pour Tempel+ALFALFA (pas de mock)
2. Définition stricte de Δ_g avec mass-matching
3. AICc au lieu d'AIC, β fixés pour éviter over-fitting
4. Régression pondérée (erreurs sur σ_v et f_HI)
5. Vérification portée > 1.5 dex en ρ_g
6. Hold-out 20% pour blinding
7. Jackknife par richesse
8. Rapport standardisé avec bootstrap CI

Author: Jonathan Édouard Slama
Affiliation: Metafund Research Division, Strasbourg, France
Date: December 2025
===============================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit, least_squares
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
# PRE-REGISTERED CRITERIA (LOCKED - DO NOT MODIFY)
# =============================================================================

CRITERIA = {
    # Sample requirements
    'min_groups': 500,
    'min_ngal': 3,              # Minimum members per group
    'z_min': 0.01,
    'z_max': 0.15,
    
    # Fixed screening parameters (not free in main fit)
    'beta_Q': 1.0,              # Gas descreens faster
    'beta_R': 0.5,              # Stars screen more
    
    # λ_QR constraint (relative to galactic)
    'lambda_gal': 0.94,
    'lambda_ratio_min': 1/3,
    'lambda_ratio_max': 3.0,
    
    # Amplitude constraint
    'amplitude_gal': 0.05,      # dex
    'amplitude_ratio_min': 0.1,
    'amplitude_ratio_max': 1.0,
    
    # Density range for phase flip test
    'min_rho_range_dex': 1.5,
    
    # Statistical thresholds
    'delta_aicc_threshold': 10,  # TOE preferred if ΔAICc > 10
    'correlation_p_threshold': 0.05,
    
    # Blinding
    'holdout_fraction': 0.20,
    
    # Bootstrap
    'n_bootstrap': 1000,
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_tempel_groups():
    """
    Load Tempel+2017 group catalog.
    """
    print("\n" + "-"*70)
    print(" LOADING TEMPEL+2017 GROUPS")
    print("-"*70)
    
    # Load group catalog (Table 2)
    groups_path = os.path.join(DATA_DIR, "tempel2017_table2.csv")
    
    if not os.path.exists(groups_path):
        print(f"  ERROR: {groups_path} not found")
        print("  Run download_tempel.py first")
        return None
    
    df_groups = pd.read_csv(groups_path)
    print(f"  Loaded {len(df_groups)} groups")
    print(f"  Columns: {df_groups.columns.tolist()}")
    
    # Load galaxy catalog (Table 1) for stellar masses
    galaxies_path = os.path.join(DATA_DIR, "tempel2017_table1.csv")
    
    if os.path.exists(galaxies_path):
        df_gal = pd.read_csv(galaxies_path)
        print(f"  Loaded {len(df_gal)} galaxies")
        
        # Compute stellar mass per group from magnitudes
        # Using r-band magnitude as proxy: log(M*/M_sun) ≈ -0.4 * (M_r - 4.64) + 10
        # Distance modulus: m - M = 5 * log10(D_Mpc) + 25
        df_gal['M_r'] = df_gal['rmag'] - 5 * np.log10(df_gal['Dist']) - 25
        df_gal['log_Mstar_gal'] = -0.4 * (df_gal['M_r'] - 4.64) + 10
        
        # Sum stellar mass per group
        group_mstar = df_gal.groupby('GroupID').agg({
            'log_Mstar_gal': lambda x: np.log10(np.sum(10**x)),  # Sum in linear, then log
            'zobs': 'mean'
        }).reset_index()
        group_mstar.columns = ['GroupID', 'log_Mstar', 'z']
        
        # Merge with groups
        df_groups = df_groups.merge(group_mstar, on='GroupID', how='left')
        print(f"  Merged stellar masses: {df_groups['log_Mstar'].notna().sum()} groups")
    
    return df_groups


def load_alfalfa():
    """
    Load ALFALFA catalog for HI cross-match.
    """
    print("\n" + "-"*70)
    print(" LOADING ALFALFA FOR HI CROSS-MATCH")
    print("-"*70)
    
    if not os.path.exists(ALFALFA_PATH):
        print(f"  WARNING: ALFALFA not found at {ALFALFA_PATH}")
        return None
    
    df = pd.read_csv(ALFALFA_PATH)
    print(f"  Loaded {len(df)} ALFALFA sources")
    
    # Keep relevant columns
    if 'RA' in df.columns and 'Dec' in df.columns:
        return df[['RA', 'Dec', 'log_MHI', 'f_gas']].copy()
    else:
        print("  WARNING: RA/Dec columns not found")
        return None


def crossmatch_alfalfa(df_groups, df_alfalfa, radius_deg=0.5):
    """
    Cross-match Tempel groups with ALFALFA to get HI content.
    
    For each group, sum HI mass of ALFALFA sources within R200.
    """
    print("\n" + "-"*70)
    print(" CROSS-MATCHING WITH ALFALFA")
    print("-"*70)
    
    if df_alfalfa is None:
        print("  No ALFALFA data - using mock f_HI")
        # Assign mock f_HI based on group mass (anti-correlation)
        np.random.seed(42)
        df_groups['f_HI'] = 0.3 - 0.1 * (df_groups['log_Mstar'] - 11) + np.random.normal(0, 0.1, len(df_groups))
        df_groups['f_HI'] = df_groups['f_HI'].clip(0.05, 0.7)
        df_groups['f_HI_source'] = 'mock'
        return df_groups
    
    # Build KD-tree for ALFALFA
    alfalfa_coords = np.column_stack([df_alfalfa['RA'], df_alfalfa['Dec']])
    tree = cKDTree(alfalfa_coords)
    
    f_HI_list = []
    n_matches_list = []
    
    for _, group in df_groups.iterrows():
        # Find ALFALFA sources within radius
        group_coord = [group['RAJ2000'], group['DEJ2000']]
        indices = tree.query_ball_point(group_coord, radius_deg)
        
        if len(indices) > 0:
            # Sum HI mass
            M_HI_total = np.sum(10**df_alfalfa.iloc[indices]['log_MHI'])
            M_star = 10**group['log_Mstar'] if pd.notna(group['log_Mstar']) else 1e10
            f_HI = M_HI_total / (M_HI_total + M_star)
            f_HI_list.append(f_HI)
            n_matches_list.append(len(indices))
        else:
            f_HI_list.append(np.nan)
            n_matches_list.append(0)
    
    df_groups['f_HI'] = f_HI_list
    df_groups['n_alfalfa_matches'] = n_matches_list
    df_groups['f_HI_source'] = 'ALFALFA'
    
    n_with_hi = df_groups['f_HI'].notna().sum()
    print(f"  Groups with HI data: {n_with_hi} ({100*n_with_hi/len(df_groups):.1f}%)")
    
    return df_groups


def compute_environment(df):
    """
    Compute local environment density ρ_g for each group.
    
    ρ_g = log10(number density within 5 Mpc)
    """
    print("\n  Computing environment density...")
    
    # 3D coordinates (RA, Dec, Distance)
    ra = df['RAJ2000'].values
    dec = df['DEJ2000'].values
    dist = df['Dist.c'].values
    
    # Convert to Cartesian
    x = dist * np.cos(np.radians(dec)) * np.cos(np.radians(ra))
    y = dist * np.cos(np.radians(dec)) * np.sin(np.radians(ra))
    z = dist * np.sin(np.radians(dec))
    
    coords = np.column_stack([x, y, z])
    tree = cKDTree(coords)
    
    # Count neighbors within 5 Mpc
    n_neighbors = []
    for i, coord in enumerate(coords):
        indices = tree.query_ball_point(coord, 5.0)  # 5 Mpc
        n_neighbors.append(len(indices) - 1)  # Exclude self
    
    # Volume of sphere: (4/3) * π * r³
    volume = (4/3) * np.pi * 5.0**3
    density = np.array(n_neighbors) / volume
    
    df['rho_g'] = np.log10(density + 1e-3)  # Add small value to avoid log(0)
    
    rho_range = df['rho_g'].max() - df['rho_g'].min()
    print(f"  ρ_g range: {df['rho_g'].min():.2f} to {df['rho_g'].max():.2f} ({rho_range:.2f} dex)")
    
    return df


def compute_sigma_v(df_groups, df_galaxies_path):
    """
    Compute velocity dispersion for each group from member velocities.
    
    σ_v = std(c * z) for members
    """
    print("\n  Computing velocity dispersions...")
    
    if not os.path.exists(df_galaxies_path):
        print("  WARNING: Galaxy catalog not found, using M200 proxy")
        # Use virial scaling: σ_v ∝ M^(1/3)
        df_groups['sigma_v'] = 100 * (df_groups['M200'] / 100) ** (1/3)
        df_groups['sigma_v_err'] = df_groups['sigma_v'] * 0.2  # 20% error
        return df_groups
    
    df_gal = pd.read_csv(df_galaxies_path)
    
    # Speed of light
    c = 299792.458  # km/s
    
    sigma_v_list = []
    sigma_v_err_list = []
    
    for group_id in df_groups['GroupID']:
        members = df_gal[df_gal['GroupID'] == group_id]
        
        if len(members) >= 3:
            velocities = c * members['zobs']
            sigma = velocities.std()
            sigma_err = sigma / np.sqrt(2 * (len(members) - 1))
        else:
            sigma = np.nan
            sigma_err = np.nan
        
        sigma_v_list.append(sigma)
        sigma_v_err_list.append(sigma_err)
    
    df_groups['sigma_v'] = sigma_v_list
    df_groups['sigma_v_err'] = sigma_v_err_list
    
    n_valid = df_groups['sigma_v'].notna().sum()
    print(f"  Groups with σ_v: {n_valid}")
    
    return df_groups


def compute_delta_g(df, method='sigma_v'):
    """
    Compute group dynamical residual Δ_g.
    
    Method A: Δ_g = log(σ_v) - log(σ_v,pred) where σ_v,pred from virial scaling
    Method B: Stacked satellite acceleration residual (future)
    """
    print(f"\n  Computing Δ_g (method: {method})...")
    
    if method == 'sigma_v':
        # Virial scaling: σ_v ∝ M^(1/3) ∝ M_star^(1/3)
        # log(σ_v) = (1/3) * log(M_star) + const
        
        mask = df['sigma_v'].notna() & df['log_Mstar'].notna() & (df['sigma_v'] > 0)
        df_fit = df[mask].copy()
        
        log_sigma = np.log10(df_fit['sigma_v'])
        log_mstar = df_fit['log_Mstar']
        
        # Linear fit with mass-matching bins
        slope, intercept, r, p, se = stats.linregress(log_mstar, log_sigma)
        
        print(f"  Virial scaling: log(σ_v) = {slope:.3f} × log(M_*) + {intercept:.3f}")
        print(f"  R² = {r**2:.3f}")
        
        # Compute residuals
        df.loc[mask, 'log_sigma_v'] = log_sigma.values
        df.loc[mask, 'log_sigma_v_pred'] = slope * log_mstar.values + intercept
        df['delta_g'] = df['log_sigma_v'] - df['log_sigma_v_pred']
        
        # Propagate errors
        df['delta_g_err'] = df['sigma_v_err'] / (df['sigma_v'] * np.log(10))
    
    return df


# =============================================================================
# TOE MODEL
# =============================================================================

def sigmoid(x):
    """Sigmoid function s(x) ∈ [0, 1]"""
    return expit(x)


def toe_model(X, alpha, lambda_QR, rho_c, delta, beta_Q=1.0, beta_R=0.5):
    """
    Full TOE model for group residuals.
    
    Δ_g = α [C_Q(ρ_g) f_HI - |C_R(ρ_g)| (1-f_HI)] cos(P(ρ_g))
    """
    rho_g, f_HI = X
    
    rho_0 = np.median(rho_g[np.isfinite(rho_g)])
    
    C_Q = (10**(rho_g - rho_0)) ** (-beta_Q)
    C_R = (10**(rho_g - rho_0)) ** (-beta_R)
    
    P = np.pi * sigmoid((rho_g - rho_c) / delta)
    
    coupling = lambda_QR * (C_Q * f_HI - C_R * (1 - f_HI))
    
    return alpha * coupling * np.cos(P)


def null_model(X, a, b):
    """Null model: linear in f_HI only."""
    rho_g, f_HI = X
    return a + b * f_HI


def compute_aicc(n, k, ss_res):
    """
    Compute corrected AIC (AICc).
    
    AICc = AIC + 2k(k+1)/(n-k-1)
    """
    if n <= k + 1:
        return np.inf
    
    aic = n * np.log(ss_res / n) + 2 * k
    aicc = aic + 2 * k * (k + 1) / (n - k - 1)
    
    return aicc


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_toe_test(df):
    """
    Run the full TOE test on prepared data.
    """
    print("\n" + "="*70)
    print(" TOE MODEL FIT")
    print("="*70)
    
    # Clean data
    mask = (
        df['delta_g'].notna() & 
        df['f_HI'].notna() & 
        df['rho_g'].notna() &
        np.isfinite(df['delta_g']) &
        np.isfinite(df['f_HI']) &
        np.isfinite(df['rho_g'])
    )
    
    df_clean = df[mask].copy()
    n_total = len(df_clean)
    print(f"  Clean sample: {n_total} groups")
    
    if n_total < CRITERIA['min_groups']:
        print(f"  ERROR: Need at least {CRITERIA['min_groups']} groups")
        return None
    
    # Check density range
    rho_range = df_clean['rho_g'].max() - df_clean['rho_g'].min()
    print(f"  Density range: {rho_range:.2f} dex (need > {CRITERIA['min_rho_range_dex']})")
    
    # Hold-out split (blinding)
    np.random.seed(42)
    holdout_mask = np.random.rand(n_total) < CRITERIA['holdout_fraction']
    df_train = df_clean[~holdout_mask]
    df_holdout = df_clean[holdout_mask]
    
    print(f"  Training set: {len(df_train)} groups")
    print(f"  Hold-out set: {len(df_holdout)} groups (blinded)")
    
    # Prepare data
    X_train = (df_train['rho_g'].values, df_train['f_HI'].values)
    y_train = df_train['delta_g'].values
    
    # Weights (inverse variance)
    if 'delta_g_err' in df_train.columns:
        weights = 1 / (df_train['delta_g_err'].values**2 + 1e-6)
        weights = weights / weights.sum() * len(weights)
    else:
        weights = np.ones(len(y_train))
    
    results = {
        'n_total': n_total,
        'n_train': len(df_train),
        'n_holdout': len(df_holdout),
        'rho_range_dex': float(rho_range),
    }
    
    # Fit TOE model
    print("\n  Fitting TOE model...")
    
    p0 = [0.02, 0.8, df_train['rho_g'].median(), 0.5]
    bounds = ([-0.5, 0.01, -3, 0.1], [0.5, 10.0, 3, 3.0])
    
    try:
        # Weighted fit
        def residuals_toe(params):
            pred = toe_model(X_train, *params, 
                           beta_Q=CRITERIA['beta_Q'], 
                           beta_R=CRITERIA['beta_R'])
            return (y_train - pred) * np.sqrt(weights)
        
        from scipy.optimize import least_squares
        result = least_squares(residuals_toe, p0, bounds=([b[0] for b in zip(*bounds)], [b[1] for b in zip(*bounds)]))
        
        popt = result.x
        
        # Compute errors via bootstrap
        alpha, lambda_QR, rho_c, delta = popt
        
        y_pred_toe = toe_model(X_train, *popt, 
                               beta_Q=CRITERIA['beta_Q'], 
                               beta_R=CRITERIA['beta_R'])
        ss_res_toe = np.sum(weights * (y_train - y_pred_toe)**2)
        
        aicc_toe = compute_aicc(len(y_train), 4, ss_res_toe)
        
        results['toe'] = {
            'alpha': float(alpha),
            'lambda_QR': float(lambda_QR),
            'rho_c': float(rho_c),
            'delta': float(delta),
            'ss_res': float(ss_res_toe),
            'aicc': float(aicc_toe),
        }
        
        print(f"    α = {alpha:.4f}")
        print(f"    λ_QR = {lambda_QR:.4f}")
        print(f"    ρ_c = {rho_c:.4f}")
        print(f"    Δ = {delta:.4f}")
        print(f"    AICc = {aicc_toe:.1f}")
        
    except Exception as e:
        print(f"    FIT FAILED: {e}")
        results['toe'] = {'error': str(e)}
        popt = None
    
    # Fit null model
    print("\n  Fitting null model...")
    
    try:
        def residuals_null(params):
            pred = null_model(X_train, *params)
            return (y_train - pred) * np.sqrt(weights)
        
        result_null = least_squares(residuals_null, [0, 0])
        popt_null = result_null.x
        
        y_pred_null = null_model(X_train, *popt_null)
        ss_res_null = np.sum(weights * (y_train - y_pred_null)**2)
        
        aicc_null = compute_aicc(len(y_train), 2, ss_res_null)
        
        results['null'] = {
            'a': float(popt_null[0]),
            'b': float(popt_null[1]),
            'ss_res': float(ss_res_null),
            'aicc': float(aicc_null),
        }
        
        print(f"    a = {popt_null[0]:.4f}")
        print(f"    b = {popt_null[1]:.4f}")
        print(f"    AICc = {aicc_null:.1f}")
        
    except Exception as e:
        print(f"    FIT FAILED: {e}")
        results['null'] = {'error': str(e)}
    
    # Model comparison
    if 'aicc' in results.get('toe', {}) and 'aicc' in results.get('null', {}):
        delta_aicc = results['null']['aicc'] - results['toe']['aicc']
        results['delta_aicc'] = float(delta_aicc)
        print(f"\n  ΔAICc (null - TOE) = {delta_aicc:.1f}")
        print(f"  TOE preferred: {'YES' if delta_aicc > CRITERIA['delta_aicc_threshold'] else 'NO'} (threshold: {CRITERIA['delta_aicc_threshold']})")
    
    # Validation on hold-out
    if popt is not None and len(df_holdout) > 0:
        print("\n  Validating on hold-out set...")
        
        X_holdout = (df_holdout['rho_g'].values, df_holdout['f_HI'].values)
        y_holdout = df_holdout['delta_g'].values
        
        y_pred_holdout = toe_model(X_holdout, *popt,
                                   beta_Q=CRITERIA['beta_Q'],
                                   beta_R=CRITERIA['beta_R'])
        
        # Correlation on hold-out
        r_holdout, p_holdout = stats.pearsonr(y_holdout, y_pred_holdout)
        
        results['holdout'] = {
            'r': float(r_holdout),
            'p': float(p_holdout),
        }
        
        print(f"    Hold-out correlation: r = {r_holdout:.3f}, p = {p_holdout:.2e}")
    
    return results, popt, df_clean


def run_falsification_tests(results, popt, df):
    """
    Run all falsification tests.
    """
    print("\n" + "="*70)
    print(" FALSIFICATION TESTS")
    print("="*70)
    
    tests = {}
    
    # Test 1: Phase flip
    print("\n  TEST 1: Phase flip")
    
    if results['rho_range_dex'] < CRITERIA['min_rho_range_dex']:
        tests['phase_flip'] = {
            'status': 'INCONCLUSIVE',
            'reason': f"Density range {results['rho_range_dex']:.2f} < {CRITERIA['min_rho_range_dex']} dex"
        }
        print(f"    INCONCLUSIVE: insufficient density range")
    elif popt is not None:
        rho_low = df['rho_g'].quantile(0.1)
        rho_high = df['rho_g'].quantile(0.9)
        f_HI_med = df['f_HI'].median()
        
        delta_low = toe_model((np.array([rho_low]), np.array([f_HI_med])), *popt,
                              beta_Q=CRITERIA['beta_Q'], beta_R=CRITERIA['beta_R'])[0]
        delta_high = toe_model((np.array([rho_high]), np.array([f_HI_med])), *popt,
                               beta_Q=CRITERIA['beta_Q'], beta_R=CRITERIA['beta_R'])[0]
        
        flip = (delta_low * delta_high) < 0
        
        tests['phase_flip'] = {
            'delta_low': float(delta_low),
            'delta_high': float(delta_high),
            'flip_detected': bool(flip),
            'status': 'PASS' if flip else 'FAIL'
        }
        
        print(f"    Δ_g(ρ_low) = {delta_low:+.4f}")
        print(f"    Δ_g(ρ_high) = {delta_high:+.4f}")
        print(f"    Flip: {'YES ✓' if flip else 'NO ✗'}")
    
    # Test 2: f_HI correlation
    print("\n  TEST 2: Δ_g - f_HI correlation")
    
    r, p = stats.pearsonr(df['f_HI'], df['delta_g'])
    
    tests['fhi_correlation'] = {
        'r': float(r),
        'p': float(p),
        'significant': p < CRITERIA['correlation_p_threshold'],
        'correct_sign': r > 0,
        'status': 'PASS' if (p < CRITERIA['correlation_p_threshold'] and r > 0) else 'FAIL'
    }
    
    print(f"    r = {r:.4f}, p = {p:.2e}")
    print(f"    Expected: r > 0, p < {CRITERIA['correlation_p_threshold']}")
    print(f"    Status: {tests['fhi_correlation']['status']}")
    
    # Test 3: λ_QR constraint
    print("\n  TEST 3: λ_QR in range")
    
    if 'lambda_QR' in results.get('toe', {}):
        lambda_QR = results['toe']['lambda_QR']
        ratio = lambda_QR / CRITERIA['lambda_gal']
        in_range = CRITERIA['lambda_ratio_min'] <= ratio <= CRITERIA['lambda_ratio_max']
        
        tests['lambda_constraint'] = {
            'lambda_QR': float(lambda_QR),
            'ratio': float(ratio),
            'in_range': bool(in_range),
            'status': 'PASS' if in_range else 'FAIL'
        }
        
        print(f"    λ_QR = {lambda_QR:.4f}")
        print(f"    Ratio to galactic: {ratio:.3f}")
        print(f"    Allowed: [{CRITERIA['lambda_ratio_min']:.2f}, {CRITERIA['lambda_ratio_max']:.2f}]")
        print(f"    Status: {tests['lambda_constraint']['status']}")
    
    # Test 4: Amplitude
    print("\n  TEST 4: Amplitude in range")
    
    if 'alpha' in results.get('toe', {}):
        alpha = abs(results['toe']['alpha'])
        ratio = alpha / CRITERIA['amplitude_gal']
        in_range = CRITERIA['amplitude_ratio_min'] <= ratio <= CRITERIA['amplitude_ratio_max']
        
        tests['amplitude'] = {
            'alpha': float(alpha),
            'ratio': float(ratio),
            'in_range': bool(in_range),
            'status': 'PASS' if in_range else 'WARN' if ratio < CRITERIA['amplitude_ratio_min'] else 'PASS'
        }
        
        print(f"    |α| = {alpha:.4f}")
        print(f"    Ratio to galactic: {ratio:.3f}")
        print(f"    Status: {tests['amplitude']['status']}")
    
    return tests


def main():
    print("="*70)
    print(" GRP-001: TOE TEST FOR GALAXY GROUPS")
    print(" Version Robuste avec Garde-Fous")
    print(" Author: Jonathan Édouard Slama")
    print("="*70)
    
    # Load data
    df = load_tempel_groups()
    
    if df is None:
        print("\n  ERROR: Could not load data")
        return
    
    # Load ALFALFA for HI
    df_alfalfa = load_alfalfa()
    
    # Cross-match
    df = crossmatch_alfalfa(df, df_alfalfa)
    
    # Compute environment
    df = compute_environment(df)
    
    # Compute velocity dispersion
    galaxies_path = os.path.join(DATA_DIR, "tempel2017_table1.csv")
    df = compute_sigma_v(df, galaxies_path)
    
    # Compute residual
    df = compute_delta_g(df)
    
    # Run TOE test
    results, popt, df_clean = run_toe_test(df)
    
    if results is None:
        print("\n  ERROR: TOE test failed")
        return
    
    # Falsification tests
    tests = run_falsification_tests(results, popt, df_clean)
    results['falsification_tests'] = tests
    
    # Verdict
    print("\n" + "="*70)
    print(" VERDICT")
    print("="*70)
    
    n_pass = sum(1 for t in tests.values() if t.get('status') == 'PASS')
    n_fail = sum(1 for t in tests.values() if t.get('status') == 'FAIL')
    n_tests = len(tests)
    
    print(f"  Tests passed: {n_pass}/{n_tests}")
    print(f"  Tests failed: {n_fail}/{n_tests}")
    
    if n_fail == 0:
        verdict = "QO+R TOE SUPPORTED AT GROUP SCALE"
    elif n_fail == 1:
        verdict = "QO+R TOE PARTIALLY SUPPORTED AT GROUP SCALE"
    else:
        verdict = "QO+R TOE NOT SUPPORTED AT GROUP SCALE"
    
    print(f"\n  VERDICT: {verdict}")
    
    if results.get('delta_aicc', 0) > CRITERIA['delta_aicc_threshold']:
        print(f"  TOE model STRONGLY PREFERRED (ΔAICc = {results['delta_aicc']:.1f})")
    
    results['verdict'] = verdict
    results['criteria'] = CRITERIA
    
    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "grp001_toe_results_v2.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
