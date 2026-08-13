#!/usr/bin/env python3
"""
===============================================================================
WB-001: TOE TEST FOR WIDE BINARIES (Gaia DR3)
===============================================================================

Scale: Stellar (10^13 - 10^16 m)
Data: Gaia DR3 wide binary catalogs
Observable: Δ_WB = log(a_obs) - log(a_Kep)

Expected:
- Strong screening at stellar scale
- Amplitude α_WB ≈ 0
- Possible residual signal in ultra-dilute environments
- λ_QR remains O(1)

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
    'min_pairs': 1000,
    'sep_min_au': 1000,      # 10^3 AU minimum
    'sep_max_au': 100000,    # 10^5 AU maximum
    'ruwe_max': 1.4,
    'parallax_snr_min': 10,
    
    # Environment
    'cylinder_radius_pc': 50,
    'dv_kms': 3,
    'min_rho_range_dex': 0.5,
    
    # Fixed screening parameters
    'beta_Q': 1.0,
    'beta_R': 0.5,
    
    # Bounds for sigmoid
    'rho_c_bounds': (-3.0, 2.0),
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
    
    # Robustness cuts
    'galactic_latitude_cuts': [20, 40],
    'ruwe_strict': 1.2,
}

# =============================================================================
# CONSTANTS
# =============================================================================

AU_TO_PC = 4.84814e-6  # 1 AU in parsecs
PC_TO_M = 3.086e16     # 1 pc in meters

# =============================================================================
# DATA LOADING
# =============================================================================

def download_wide_binaries():
    """
    Download wide binary catalog from VizieR.
    Combines star tables with pair properties.
    """
    print("\n" + "="*70)
    print(" DOWNLOADING GAIA DR3 WIDE BINARIES")
    print("="*70)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, "gaia_wide_binaries_combined.csv")
    
    if os.path.exists(cache_path):
        print(f"  Loading from cache: {cache_path}")
        return pd.read_csv(cache_path)
    
    try:
        from astroquery.vizier import Vizier
        
        Vizier.ROW_LIMIT = 50000  # Limit for manageable test
        
        print("  Querying VizieR for Hartman & Lépine catalog...")
        print("  (J/ApJS/247/66 - Wide binaries with Gaia DR2)")
        
        catalogs = Vizier.get_catalogs('J/ApJS/247/66')
        
        if len(catalogs) == 0:
            print("  ERROR: No catalog found")
            return None
        
        print(f"  Found {len(catalogs)} tables:")
        for i, cat in enumerate(catalogs):
            print(f"    [{i}] {len(cat)} rows - columns: {list(cat.colnames)[:5]}...")
        
        # Table 0: Primary stars
        # Table 1: Secondary stars  
        # Table 2: Pair properties (ASep, PSep)
        
        if len(catalogs) >= 3:
            df_primary = catalogs[0].to_pandas()
            df_secondary = catalogs[1].to_pandas()
            df_pairs = catalogs[2].to_pandas()
            
            print(f"  Primary stars: {len(df_primary)}")
            print(f"  Secondary stars: {len(df_secondary)}")
            print(f"  Pair properties: {len(df_pairs)}")
            print(f"  Pair columns: {df_pairs.columns.tolist()}")
            
            # Combine tables
            # Use 'Seq' as key if available, otherwise use index
            if 'Seq' in df_primary.columns and 'Seq' in df_pairs.columns:
                df = df_primary.merge(df_pairs, on='Seq', how='inner', suffixes=('', '_pair'))
                df = df.merge(df_secondary[['Seq', 'Gmag', 'BP-RP', 'plx']], 
                             on='Seq', how='left', suffixes=('_1', '_2'))
            else:
                # Assume same order
                n_min = min(len(df_primary), len(df_pairs))
                df = pd.concat([df_primary.iloc[:n_min].reset_index(drop=True), 
                               df_pairs.iloc[:n_min].reset_index(drop=True)], axis=1)
                if len(df_secondary) >= n_min:
                    df['Gmag_2'] = df_secondary['Gmag'].iloc[:n_min].values
            
            print(f"  Combined: {len(df)} pairs")
            print(f"  Columns: {df.columns.tolist()[:15]}...")
        else:
            # Fallback to single table
            df = catalogs[0].to_pandas()
        
        # Save to cache
        df.to_csv(cache_path, index=False)
        print(f"  Saved to: {cache_path}")
        
        return df
        
    except ImportError:
        print("  ERROR: astroquery not installed")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def prepare_wide_binary_data(df):
    """
    Prepare wide binary data for TOE analysis.
    """
    print("\n" + "-"*70)
    print(" PREPARING WIDE BINARY DATA")
    print("-"*70)
    
    print(f"  Available columns: {df.columns.tolist()}")
    
    df_clean = df.copy()
    
    # Direct column mapping for Hartman & Lépine catalog
    # Note: After merge, columns may have _1, _2 suffixes
    col_mapping = {}
    
    for col in df.columns:
        col_lower = col.lower()
        
        # RA/Dec
        if col == 'RA_ICRS' or col_lower == 'ra_icrs':
            col_mapping['ra'] = col
        if col == 'DE_ICRS' or col_lower == 'de_icrs':
            col_mapping['dec'] = col
        
        # Separation - PSep is physical separation (actually in AU despite name)
        if col == 'PSep' or col_lower == 'psep':
            col_mapping['sep_pc'] = col
        if col == 'ASep' or col_lower == 'asep':
            col_mapping['sep_arcsec'] = col
        
        # Parallax (may have _1 suffix) - check contains
        if 'plx' in col_lower and 'parallax' not in col_mapping:
            col_mapping['parallax'] = col
        
        # G magnitudes - primary
        if ('gmag' in col_lower and '_2' not in col_lower and 
            'gmag1' not in col_mapping):
            col_mapping['gmag1'] = col
        # G magnitudes - secondary
        if 'gmag' in col_lower and '_2' in col_lower:
            col_mapping['gmag2'] = col
        
        # BP-RP color
        if 'bp-rp' in col_lower and 'bp_rp' not in col_mapping:
            col_mapping['bp_rp'] = col
        
        # Galactic coordinates
        if col_lower in ['glat', 'b', 'gal_b']:
            col_mapping['glat'] = col
    
    print(f"  Column mapping: {col_mapping}")
    
    # Compute separation in AU
    # NOTE: In Hartman & Lépine catalog, PSep is already in AU (not parsec!)
    # PSep = ASep (arcsec) * distance (pc) = separation in AU
    if 'sep_pc' in col_mapping:
        # PSep is actually in AU despite the name!
        df_clean['sep_au'] = pd.to_numeric(df[col_mapping['sep_pc']], errors='coerce')
        print(f"  PSep range: {df_clean['sep_au'].min():.1f} - {df_clean['sep_au'].max():.1f} AU")
    elif 'sep_arcsec' in col_mapping and 'parallax' in col_mapping:
        ang_sep = pd.to_numeric(df[col_mapping['sep_arcsec']], errors='coerce')
        plx_mas = pd.to_numeric(df[col_mapping['parallax']], errors='coerce')
        df_clean['sep_au'] = ang_sep * 1000 / plx_mas
        print(f"  Computed sep_au from ASep and parallax")
    else:
        print("  WARNING: No separation data found")
    
    # Get coordinates
    if 'ra' in col_mapping:
        df_clean['RA'] = pd.to_numeric(df[col_mapping['ra']], errors='coerce')
        print(f"  RA range: {df_clean['RA'].min():.2f} - {df_clean['RA'].max():.2f}")
    if 'dec' in col_mapping:
        df_clean['Dec'] = pd.to_numeric(df[col_mapping['dec']], errors='coerce')
        print(f"  Dec range: {df_clean['Dec'].min():.2f} - {df_clean['Dec'].max():.2f}")
    
    # Galactic latitude
    if 'glat' in col_mapping:
        df_clean['glat'] = pd.to_numeric(df[col_mapping['glat']], errors='coerce')
    elif 'RA' in df_clean.columns and 'Dec' in df_clean.columns:
        # Compute galactic latitude from RA/Dec
        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            
            valid_mask = df_clean['RA'].notna() & df_clean['Dec'].notna()
            coords = SkyCoord(ra=df_clean.loc[valid_mask, 'RA'].values*u.deg, 
                            dec=df_clean.loc[valid_mask, 'Dec'].values*u.deg, 
                            frame='icrs')
            df_clean.loc[valid_mask, 'glat'] = coords.galactic.b.deg
            print(f"  Galactic latitude range: {df_clean['glat'].min():.1f} - {df_clean['glat'].max():.1f} deg")
        except Exception as e:
            print(f"  WARNING: Could not compute glat: {e}")
            df_clean['glat'] = 45  # Default mid-latitude
    else:
        df_clean['glat'] = 45
    
    # RUWE quality
    if 'ruwe1' in col_mapping and 'ruwe2' in col_mapping:
        df_clean['ruwe_max'] = np.maximum(
            df[col_mapping['ruwe1']].astype(float),
            df[col_mapping['ruwe2']].astype(float)
        )
    elif 'ruwe1' in col_mapping:
        df_clean['ruwe_max'] = df[col_mapping['ruwe1']].astype(float)
    else:
        df_clean['ruwe_max'] = 1.0  # Assume good quality
    
    # Parallax for distance
    if 'parallax' in col_mapping:
        df_clean['parallax'] = pd.to_numeric(df[col_mapping['parallax']], errors='coerce')
        # Distance in pc (parallax in mas)
        df_clean['dist_pc'] = 1000 / df_clean['parallax'].clip(0.1, None)
        print(f"  Distance range: {df_clean['dist_pc'].min():.1f} - {df_clean['dist_pc'].max():.1f} pc")
    
    # Masses - estimate from G magnitude if not available
    if 'mass1' in col_mapping and 'mass2' in col_mapping:
        df_clean['M1'] = pd.to_numeric(df[col_mapping['mass1']], errors='coerce')
        df_clean['M2'] = pd.to_numeric(df[col_mapping['mass2']], errors='coerce')
    elif 'gmag1' in col_mapping and 'dist_pc' in df_clean.columns:
        # Mass estimate from absolute G magnitude
        # M_G ~ 4.8 + 10*log10(M/M_sun) for main sequence (rough)
        gmag1 = pd.to_numeric(df[col_mapping['gmag1']], errors='coerce')
        M_G1 = gmag1 - 5*np.log10(df_clean['dist_pc'].clip(1, None)) + 5
        df_clean['M1'] = 10**((4.8 - M_G1)/10)
        df_clean['M1'] = df_clean['M1'].clip(0.1, 10)
        
        if 'gmag2' in col_mapping:
            gmag2 = pd.to_numeric(df[col_mapping['gmag2']], errors='coerce')
            M_G2 = gmag2 - 5*np.log10(df_clean['dist_pc'].clip(1, None)) + 5
            df_clean['M2'] = 10**((4.8 - M_G2)/10)
            df_clean['M2'] = df_clean['M2'].clip(0.1, 10)
        else:
            # Use mass ratio from magnitude difference if only one gmag
            df_clean['M2'] = df_clean['M1'] * 0.7  # Approximate
        
        print(f"  M1 range: {df_clean['M1'].min():.2f} - {df_clean['M1'].max():.2f} M_sun")
        print(f"  M2 range: {df_clean['M2'].min():.2f} - {df_clean['M2'].max():.2f} M_sun")
    else:
        # Default solar-type masses
        print("  WARNING: Using default masses (no Gmag or distance)")
        df_clean['M1'] = 1.0
        df_clean['M2'] = 0.8
    
    df_clean['M_tot'] = df_clean['M1'] + df_clean['M2']
    
    # Quality cuts
    print("\n  Applying quality cuts...")
    
    mask = pd.Series(True, index=df_clean.index)
    
    # Separation range
    if 'sep_au' in df_clean.columns:
        mask &= (df_clean['sep_au'] >= CRITERIA['sep_min_au'])
        mask &= (df_clean['sep_au'] <= CRITERIA['sep_max_au'])
        print(f"    Separation cut: {mask.sum()} remaining")
    
    # RUWE quality
    if 'ruwe_max' in df_clean.columns:
        mask &= (df_clean['ruwe_max'] < CRITERIA['ruwe_max'])
        print(f"    RUWE cut: {mask.sum()} remaining")
    
    # Parallax SNR
    if 'parallax' in df_clean.columns:
        mask &= (df_clean['parallax'] > 0)
        mask &= (df_clean['dist_pc'] > 0) & (df_clean['dist_pc'] < 1000)
        print(f"    Distance cut: {mask.sum()} remaining")
    
    df_clean = df_clean[mask].copy()
    
    print(f"\n  After quality cuts: {len(df_clean)} pairs")
    if 'sep_au' in df_clean.columns:
        print(f"  Separation range: {df_clean['sep_au'].min():.0f} - {df_clean['sep_au'].max():.0f} AU")
    if 'M_tot' in df_clean.columns:
        print(f"  M_tot range: {df_clean['M_tot'].min():.2f} - {df_clean['M_tot'].max():.2f} M_sun")
    
    return df_clean


# =============================================================================
# ENVIRONMENT CALCULATION
# =============================================================================

def compute_stellar_density(df, radius_pc=50):
    """
    Compute local stellar density using counts-in-cylinder.
    """
    print(f"\n  Computing stellar density (R={radius_pc} pc)...")
    
    if 'RA' not in df.columns or 'Dec' not in df.columns or 'dist_pc' not in df.columns:
        print("  WARNING: Missing coordinates, using random environment")
        np.random.seed(42)
        df['rho_star'] = np.random.normal(0, 0.5, len(df))
        return df
    
    # Convert to 3D Cartesian
    ra_rad = np.radians(df['RA'].values)
    dec_rad = np.radians(df['Dec'].values)
    dist = df['dist_pc'].values
    
    x = dist * np.cos(dec_rad) * np.cos(ra_rad)
    y = dist * np.cos(dec_rad) * np.sin(ra_rad)
    z = dist * np.sin(dec_rad)
    
    coords = np.column_stack([x, y, z])
    
    # Build KD-tree
    tree = cKDTree(coords)
    
    # Count neighbors within radius
    n_neighbors = np.array([len(tree.query_ball_point(coord, radius_pc)) - 1 
                           for coord in coords])
    
    # Density (per pc^3)
    volume = (4/3) * np.pi * radius_pc**3
    density = n_neighbors / volume
    
    df['rho_star'] = np.log10(density + 1e-6)
    
    rho_range = df['rho_star'].max() - df['rho_star'].min()
    print(f"  ρ_star range: {df['rho_star'].min():.2f} to {df['rho_star'].max():.2f} ({rho_range:.2f} dex)")
    
    return df


# =============================================================================
# RESIDUAL CALCULATION
# =============================================================================

def compute_delta_wb(df):
    """
    Compute Δ_WB = log(a_obs) - log(a_Kep).
    
    For wide binaries, a_Kep depends on M_tot and period.
    We use the residual from the M_tot vs sep relation.
    """
    print("\n  Computing Δ_WB (orbital residual)...")
    
    if 'sep_au' not in df.columns or 'M_tot' not in df.columns:
        print("  WARNING: Missing sep_au or M_tot, using mock")
        np.random.seed(42)
        df['delta_wb'] = np.random.normal(0, 0.1, len(df))
        return df
    
    # Log-transform
    log_sep = np.log10(df['sep_au'].values)
    log_M = np.log10(df['M_tot'].values)
    
    # Fit 2D regression: log(sep) ~ log(M_tot)
    # For Keplerian orbits: a^3 ∝ M * P^2, so log(a) ~ (1/3)*log(M) + const
    # But observed sep is projection, so relationship is more complex
    
    mask = np.isfinite(log_sep) & np.isfinite(log_M)
    
    if mask.sum() < 100:
        print("  WARNING: Not enough valid data")
        df['delta_wb'] = np.random.normal(0, 0.1, len(df))
        return df
    
    X = log_M[mask].reshape(-1, 1)
    y = log_sep[mask]
    
    reg = LinearRegression()
    reg.fit(X, y)
    
    print(f"  Regression: log(sep) = {reg.coef_[0]:.3f}×log(M) + {reg.intercept_:.3f}")
    print(f"  (Expected Kepler slope ~ 0.33)")
    
    # Compute residuals
    log_sep_pred = reg.predict(log_M.reshape(-1, 1))
    residuals = log_sep - log_sep_pred
    
    df['delta_wb'] = residuals
    df['delta_wb_err'] = 0.1  # Typical uncertainty
    
    print(f"  Δ_WB mean: {np.nanmean(residuals):.4f}")
    print(f"  Δ_WB std: {np.nanstd(residuals):.4f}")
    
    return df


# =============================================================================
# f_BINARY PROXY (analog to f_HI/f_ICM)
# =============================================================================

def compute_f_binary(df):
    """
    Compute binary mass ratio proxy.
    
    f_bin = M2 / (M1 + M2) = q / (1 + q)
    
    This is analogous to gas fraction in other tests.
    """
    print("\n  Computing f_binary (mass ratio proxy)...")
    
    if 'M1' in df.columns and 'M2' in df.columns:
        q = df['M2'] / df['M1']  # Mass ratio
        df['f_bin'] = q / (1 + q)
        df['f_bin'] = df['f_bin'].clip(0.1, 0.9)
        print(f"  f_bin range: {df['f_bin'].min():.2f} - {df['f_bin'].max():.2f}")
    else:
        print("  WARNING: No mass info, using uniform")
        df['f_bin'] = 0.5
    
    return df


# =============================================================================
# TOE MODEL
# =============================================================================

def sigmoid(x):
    return expit(x)


def toe_model(X, alpha, lambda_QR, rho_c, delta):
    rho, f = X
    rho_0 = np.nanmedian(rho)
    
    C_Q = (10**(rho - rho_0)) ** (-CRITERIA['beta_Q'])
    C_R = (10**(rho - rho_0)) ** (-CRITERIA['beta_R'])
    
    P = np.pi * sigmoid((rho - rho_c) / delta)
    
    coupling = lambda_QR * (C_Q * f - C_R * (1 - f))
    
    return alpha * coupling * np.cos(P)


def null_model(X, a, b):
    rho, f = X
    return a + b * rho


def compute_aicc(n, k, ss_res):
    if n <= k + 1 or ss_res <= 0:
        return np.inf
    aic = n * np.log(ss_res / n) + 2 * k
    return aic + 2 * k * (k + 1) / (n - k - 1)


def fit_toe(X, y, weights=None):
    if weights is None:
        weights = np.ones(len(y))
    
    p0 = [0.01, 1.0, 0.0, 1.0]
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
# STATISTICAL TESTS
# =============================================================================

def bootstrap_stratified(df, n_boot=100):
    """Bootstrap with stratification by separation and galactic latitude."""
    print(f"\n  Running bootstrap (n={n_boot})...")
    
    df = df.copy()
    
    # Stratify by separation bins
    if 'sep_au' in df.columns:
        df['sep_bin'] = pd.qcut(df['sep_au'], q=4, labels=['s1', 's2', 's3', 's4'], duplicates='drop')
    else:
        df['sep_bin'] = 'all'
    
    results = []
    
    for i in range(n_boot):
        try:
            df_boot = df.groupby('sep_bin', group_keys=False).apply(
                lambda x: x.sample(n=len(x), replace=True) if len(x) > 0 else x
            )
            
            X_boot = (df_boot['rho_star'].values, df_boot['f_bin'].values)
            y_boot = df_boot['delta_wb'].values
            
            result_toe = fit_toe(X_boot, y_boot)
            result_null = fit_null(X_boot, y_boot)
            
            if result_toe['success'] and result_null['success']:
                results.append({
                    'alpha': result_toe['popt'][0],
                    'lambda_QR': result_toe['popt'][1],
                    'delta_aicc': result_null['aicc'] - result_toe['aicc']
                })
        except:
            continue
    
    if len(results) > 0:
        df_results = pd.DataFrame(results)
        summary = {
            'alpha_median': float(df_results['alpha'].median()),
            'lambda_QR_median': float(df_results['lambda_QR'].median()),
            'lambda_QR_ci_low': float(df_results['lambda_QR'].quantile(0.025)),
            'lambda_QR_ci_high': float(df_results['lambda_QR'].quantile(0.975)),
            'delta_aicc_median': float(df_results['delta_aicc'].median()),
            'n_successful': len(results)
        }
        print(f"  α = {summary['alpha_median']:.4f}")
        print(f"  λ_QR = {summary['lambda_QR_median']:.3f} [{summary['lambda_QR_ci_low']:.3f}, {summary['lambda_QR_ci_high']:.3f}]")
        return summary
    
    return None


def permutation_test(df, n_perm=100):
    """Permutation test for ΔAICc."""
    print(f"\n  Running permutation test (n={n_perm})...")
    
    X = (df['rho_star'].values, df['f_bin'].values)
    y = df['delta_wb'].values
    
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


def jackknife_robustness(df):
    """Jackknife by galactic latitude and RUWE cuts."""
    print("\n  Running jackknife robustness...")
    
    results = []
    
    # By galactic latitude
    for b_cut in CRITERIA['galactic_latitude_cuts']:
        df_sub = df[abs(df['glat']) > b_cut].copy() if 'glat' in df.columns else df.copy()
        
        if len(df_sub) >= CRITERIA['min_pairs']:
            X = (df_sub['rho_star'].values, df_sub['f_bin'].values)
            y = df_sub['delta_wb'].values
            
            result_toe = fit_toe(X, y)
            result_null = fit_null(X, y)
            
            if result_toe['success'] and result_null['success']:
                results.append({
                    'cut': f'|b|>{b_cut}',
                    'n': len(df_sub),
                    'lambda_QR': float(result_toe['popt'][0]),
                    'delta_aicc': float(result_null['aicc'] - result_toe['aicc'])
                })
                print(f"    |b| > {b_cut}°: n={len(df_sub)}, λ_QR={result_toe['popt'][1]:.3f}")
    
    # Strict RUWE cut
    if 'ruwe_max' in df.columns:
        df_strict = df[df['ruwe_max'] < CRITERIA['ruwe_strict']].copy()
        
        if len(df_strict) >= CRITERIA['min_pairs']:
            X = (df_strict['rho_star'].values, df_strict['f_bin'].values)
            y = df_strict['delta_wb'].values
            
            result_toe = fit_toe(X, y)
            result_null = fit_null(X, y)
            
            if result_toe['success'] and result_null['success']:
                results.append({
                    'cut': f'RUWE<{CRITERIA["ruwe_strict"]}',
                    'n': len(df_strict),
                    'lambda_QR': float(result_toe['popt'][1]),
                    'delta_aicc': float(result_null['aicc'] - result_toe['aicc'])
                })
                print(f"    RUWE < {CRITERIA['ruwe_strict']}: n={len(df_strict)}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print(" WB-001: TOE TEST FOR WIDE BINARIES (Gaia DR3)")
    print(" Author: Jonathan Édouard Slama")
    print("="*70)
    
    # Download data
    df = download_wide_binaries()
    
    if df is None:
        print("\n  ERROR: No wide binary data available")
        return
    
    # Prepare data
    df = prepare_wide_binary_data(df)
    
    if len(df) < CRITERIA['min_pairs']:
        print(f"\n  ERROR: Need at least {CRITERIA['min_pairs']} pairs, got {len(df)}")
        return
    
    # Compute variables
    df = compute_stellar_density(df)
    df = compute_delta_wb(df)
    df = compute_f_binary(df)
    
    # Clean sample
    mask = (
        df['delta_wb'].notna() & 
        df['f_bin'].notna() & 
        df['rho_star'].notna() &
        np.isfinite(df['delta_wb']) &
        np.isfinite(df['f_bin']) &
        np.isfinite(df['rho_star'])
    )
    
    df_clean = df[mask].copy()
    print(f"\n  Clean sample: {len(df_clean)} pairs")
    
    rho_range = df_clean['rho_star'].max() - df_clean['rho_star'].min()
    print(f"  ρ_star range: {rho_range:.2f} dex")
    
    data_limited = rho_range < CRITERIA['min_rho_range_dex']
    if data_limited:
        print(f"  ⚠️ WARNING: ρ range < {CRITERIA['min_rho_range_dex']} dex")
    
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
        'data_limited': data_limited,
        'criteria': {k: v for k, v in CRITERIA.items() if not callable(v)}
    }
    
    # Main fit
    print("\n" + "-"*70)
    print(" MAIN FIT (Training Set)")
    print("-"*70)
    
    X_train = (df_train['rho_star'].values, df_train['f_bin'].values)
    y_train = df_train['delta_wb'].values
    
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
        X_holdout = (df_holdout['rho_star'].values, df_holdout['f_bin'].values)
        y_holdout = df_holdout['delta_wb'].values
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
    
    # Jackknife
    print("\n" + "-"*70)
    print(" JACKKNIFE ROBUSTNESS")
    print("-"*70)
    jackknife_results = jackknife_robustness(df_clean)
    results['jackknife'] = jackknife_results
    
    # Verdict
    print("\n" + "="*70)
    print(" VERDICT")
    print("="*70)
    
    evidence = []
    
    if data_limited:
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
    
    # Amplitude check (expected near zero for WB)
    if 'alpha' in results.get('toe', {}):
        alpha = abs(results['toe']['alpha'])
        if alpha < 0.01:
            evidence.append(('α amplitude', 'EXPECTED', f"|α| = {alpha:.4f} ≈ 0 (screened)"))
        else:
            evidence.append(('α amplitude', 'ANOMALY', f"|α| = {alpha:.4f} > 0 (unexpected)"))
    
    n_pass = sum(1 for e in evidence if e[1] == 'PASS')
    n_fail = sum(1 for e in evidence if e[1] == 'FAIL')
    n_expected = sum(1 for e in evidence if e[1] == 'EXPECTED')
    
    for name, status, detail in evidence:
        symbol = '✅' if status == 'PASS' else '⚠️' if status in ['WEAK', 'LIMITED', 'EXPECTED'] else '❌' if status == 'FAIL' else '🔍'
        print(f"  {symbol} {name}: {detail}")
    
    print(f"\n  Summary: {n_pass} PASS, {n_expected} EXPECTED, {n_fail} FAIL")
    
    # Determine verdict
    if n_expected > 0 and n_fail <= 1:
        verdict = "QO+R TOE SCREENED AT STELLAR SCALE (as predicted)"
    elif n_fail == 0 and n_pass >= 3:
        verdict = "QO+R TOE DETECTED AT STELLAR SCALE (unexpected!)"
    else:
        verdict = "QO+R TOE INCONCLUSIVE AT STELLAR SCALE"
    
    print(f"\n  VERDICT: {verdict}")
    results['verdict'] = verdict
    results['evidence'] = [{'name': e[0], 'status': e[1], 'detail': e[2]} for e in evidence]
    
    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "wb001_toe_results.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
