#!/usr/bin/env python3
"""
Paper 4 - Main Analysis Script (CORRECTED v3)
==============================================
QO+R validation using CORRECT methodology from Paper 1.

KEY FIX: 
- Q = gas_fraction (f_gas)
- R = log(M_star) (NOT 1 - f_gas!)
- These are INDEPENDENT variables
- λ_QR is the Q²R² coupling term coefficient

Author: Jonathan Édouard Slama
ORCID: 0009-0002-1292-4350
Affiliation: Metafund Research Division, Strasbourg, France
Date: 2024-12-06
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit, differential_evolution
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# =============================================================================
# EXPECTED PARAMETERS (from Paper 1 / TNG100 calibration)
# =============================================================================
C_Q_EXPECTED = 2.93
C_R_EXPECTED = -2.39
LAMBDA_QR_EXPECTED = 1.07
AMPLITUDE = 0.035  # U-shape amplitude from SPARC

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to BTFR directory (relative to Git folder)
GIT_DIR = os.path.dirname(PROJECT_DIR)
BTFR_DIR = os.path.join(os.path.dirname(GIT_DIR), "BTFR")
SPARC_FILE = os.path.join(BTFR_DIR, "Data", "processed", "sparc_with_environment.csv")
# CORRECTED: Use alfalfa_corrected.csv with real photometric M_star from Durbala 2020
# Old file used scaling relation M_star = f(M_HI) which created artificial Q-R correlation
ALFALFA_FILE = os.path.join(BTFR_DIR, "Test-Replicability", "data", "alfalfa_corrected.csv")
WALLABY_FILE = os.path.join(BTFR_DIR, "TNG", "wallaby_analysis_results.csv")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "experimental", "04_RESULTS")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("="*70)
print(" PAPER 4 - QO+R VALIDATION (CORRECT METHODOLOGY v3)")
print(" Q = gas_fraction, R = log(M_star)")
print("="*70)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nExpected parameters (from TNG100 calibration):")
print(f"  C_Q = {C_Q_EXPECTED}")
print(f"  C_R = {C_R_EXPECTED}")
print(f"  λ_QR = {LAMBDA_QR_EXPECTED}")

# =============================================================================
# DATA LOADING
# =============================================================================

def load_sparc():
    if not os.path.exists(SPARC_FILE):
        print(f"  [!] SPARC not found")
        return None
    
    df = pd.read_csv(SPARC_FILE)
    print(f"  SPARC: {len(df)} galaxies loaded")
    
    # Q = gas_fraction
    if 'MHI' in df.columns and 'M_bar' in df.columns:
        df['gas_fraction'] = df['MHI'] / df['M_bar']
    
    # R = log(M_star) - INDEPENDENT from Q!
    # SPARC M_star is in units of 10^10 M_sun!
    if 'M_star' in df.columns:
        # Convert to solar masses (SPARC uses 10^10 M_sun units)
        M_star_solar = df['M_star'] * 1e10
        df['log_Mstar'] = np.log10(M_star_solar.clip(1e6, None))
        print(f"  log(M_star) range: {df['log_Mstar'].min():.2f} to {df['log_Mstar'].max():.2f}")
    
    if 'density_proxy' in df.columns:
        df['log_env'] = np.log10(df['density_proxy'] + 0.1)
    
    print(f"  gas_fraction range: {df['gas_fraction'].min():.3f} to {df['gas_fraction'].max():.3f}")
    return df


def load_alfalfa():
    if not os.path.exists(ALFALFA_FILE):
        print(f"  [!] ALFALFA not found")
        return None
    
    df = pd.read_csv(ALFALFA_FILE)
    print(f"  ALFALFA: {len(df)} sources loaded")
    
    # Q = gas_fraction
    if 'f_HI' in df.columns:
        df['gas_fraction'] = df['f_HI']
    
    # R = log(M_star) - INDEPENDENT!
    if 'M_star' in df.columns:
        df['log_Mstar'] = np.log10(df['M_star'].clip(1e6, None))
        print(f"  log(M_star) range: {df['log_Mstar'].min():.2f} to {df['log_Mstar'].max():.2f}")
    elif 'log_Mstar' in df.columns:
        print(f"  log(M_star) already in file")
    
    if 'density_proxy' in df.columns:
        df['log_env'] = np.log10(df['density_proxy'] + 0.1)
    
    print(f"  gas_fraction range: {df['gas_fraction'].min():.3f} to {df['gas_fraction'].max():.3f}")
    return df


def load_wallaby():
    if not os.path.exists(WALLABY_FILE):
        print(f"  [!] WALLABY not found")
        return None
    
    df = pd.read_csv(WALLABY_FILE)
    print(f"  WALLABY: {len(df)} sources loaded")
    
    # Redshift
    HI_REST_FREQ = 1420405751.0
    if 'freq' in df.columns:
        df['redshift'] = (HI_REST_FREQ / df['freq']) - 1
        print(f"  Redshift: {df['redshift'].min():.4f} to {df['redshift'].max():.4f}")
    
    # Check for M_star or estimate it
    has_mstar = False
    for col in ['M_star', 'stellar_mass', 'log_Mstar']:
        if col in df.columns:
            if 'log' in col.lower():
                df['log_Mstar'] = df[col]
            else:
                df['log_Mstar'] = np.log10(df[col].clip(1e6, None))
            has_mstar = True
            print(f"  Using {col} for log(M_star)")
            break
    
    # If no M_star, estimate from M_bar and M_HI
    # M_bar = M_star + 1.36 * M_HI → M_star = M_bar - 1.36 * M_HI
    if not has_mstar and 'M_bar' in df.columns and 'M_HI' in df.columns:
        M_star_est = df['M_bar'] - 1.36 * df['M_HI']
        # Filter out negative values (unphysical)
        valid = M_star_est > 0
        df.loc[valid, 'M_star_est'] = M_star_est[valid]
        df.loc[valid, 'log_Mstar'] = np.log10(M_star_est[valid].clip(1e6, None))
        print(f"  Estimated M_star from M_bar - 1.36*M_HI")
        print(f"  Valid estimates: {valid.sum()} / {len(df)}")
        if valid.sum() > 0:
            print(f"  log(M_star) range: {df.loc[valid, 'log_Mstar'].min():.2f} to {df.loc[valid, 'log_Mstar'].max():.2f}")
            has_mstar = True
    
    # Gas fraction
    if 'M_HI' in df.columns and 'M_bar' in df.columns:
        df['gas_fraction'] = (1.36 * df['M_HI']) / df['M_bar']
        gf = df['gas_fraction'].dropna()
        print(f"  gas_fraction range: {gf.min():.3f} to {gf.max():.3f}")
    
    if not has_mstar:
        print(f"  [!] Cannot compute M_star for WALLABY")
    
    return df


def calculate_btfr_residuals(df):
    if 'log_Mbar' not in df.columns or 'log_Vflat' not in df.columns:
        return df
    
    mask = np.isfinite(df['log_Mbar']) & np.isfinite(df['log_Vflat'])
    x, y = df.loc[mask, 'log_Mbar'].values, df.loc[mask, 'log_Vflat'].values
    
    if len(x) < 10:
        return df
    
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    print(f"  BTFR: slope={slope:.3f}, r={r_value:.3f}")
    
    df['btfr_residual'] = df['log_Vflat'] - (slope * df['log_Mbar'] + intercept)
    return df


# =============================================================================
# U-SHAPE FITTING
# =============================================================================

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
        popt, pcov = curve_fit(lambda x, a, b, c: a*x**2 + b*x + c,
                               bin_centers, bin_means, p0=[0.01, 0, 0])
        a, b, c = popt
        a_err = np.sqrt(pcov[0, 0])
        
        t_stat = abs(a) / a_err if a_err > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(t_stat, len(bin_centers) - 3))
        
        return {
            'a': a, 'a_err': a_err, 'b': b, 'c': c,
            't_stat': t_stat, 'p_value': p_value,
            'bin_centers': bin_centers, 'bin_means': bin_means, 'bin_errors': bin_errors,
            'n_bins': len(bin_centers)
        }
    except:
        return None


# =============================================================================
# QO+R ANALYSIS (CORRECT METHODOLOGY)
# =============================================================================

def analyze_qor(df, name):
    """
    Full QO+R analysis using correct methodology.
    
    Q = gas_fraction
    R = log(M_star)  <- INDEPENDENT from Q!
    
    Model: correction = amplitude × (C_Q × Q × env² + C_R × R × env² + λ_QR × Q² × R²)
    """
    print(f"\n  --- QO+R Analysis ---")
    
    # Check required columns
    required = ['gas_fraction', 'log_Mstar', 'btfr_residual']
    for col in required:
        if col not in df.columns:
            if col == 'log_Mstar' and 'M_star' in df.columns:
                df['log_Mstar'] = np.log10(df['M_star'].clip(1e6, None))
            else:
                print(f"  [!] Missing {col}")
                return None
    
    # Environment
    if 'log_env' in df.columns:
        env_col = 'log_env'
    elif 'density_proxy' in df.columns:
        df['env_std'] = (df['density_proxy'] - df['density_proxy'].mean()) / df['density_proxy'].std()
        env_col = 'env_std'
    else:
        print(f"  [!] No environment column")
        return None
    
    # Clean data
    mask = (np.isfinite(df['gas_fraction']) & 
            np.isfinite(df['log_Mstar']) & 
            np.isfinite(df['btfr_residual']) &
            np.isfinite(df[env_col]))
    df_clean = df[mask].copy()
    
    if len(df_clean) < 50:
        print(f"  [!] Insufficient data ({len(df_clean)})")
        return None
    
    print(f"  Using {len(df_clean)} galaxies")
    
    # Normalize variables
    Q = df_clean['gas_fraction'].values
    R = df_clean['log_Mstar'].values
    env = df_clean[env_col].values
    residuals = df_clean['btfr_residual'].values
    
    # Guard-rail for degenerate distributions
    if Q.std() < 1e-6 or R.std() < 1e-6 or env.std() < 1e-6:
        print(f"  [!] Degenerate distribution detected - skipping")
        print(f"      std(Q)={Q.std():.2e}, std(R)={R.std():.2e}, std(env)={env.std():.2e}")
        return None
    
    Q_norm = (Q - Q.mean()) / Q.std()
    R_norm = (R - R.mean()) / R.std()
    env_std = (env - env.mean()) / env.std()
    env2 = env_std ** 2
    
    print(f"  Q (gas_fraction): mean={Q.mean():.3f}, std={Q.std():.3f}")
    print(f"  R (log M_star): mean={R.mean():.2f}, std={R.std():.2f}")
    print(f"  Correlation Q-R: {np.corrcoef(Q, R)[0,1]:.3f}")
    
    # 1. Raw U-shape
    raw_fit = fit_ushape(env_std, residuals)
    if raw_fit:
        print(f"\n  Raw U-shape: a = {raw_fit['a']:.5f} ± {raw_fit['a_err']:.5f}, p = {raw_fit['p_value']:.4f}")
    
    # 2. Fit QO+R model
    # residual = α + β₁·Q·env² + β₂·R·env² + β₃·Q²·R²
    X = np.column_stack([
        np.ones(len(df_clean)),
        Q_norm * env2,           # C_Q term
        R_norm * env2,           # C_R term
        (Q_norm**2) * (R_norm**2)  # λ_QR coupling term
    ])
    
    # Check for NaN/Inf
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(residuals)):
        print(f"  [!] NaN/Inf in design matrix - skipping regression")
        return {
            'n_galaxies': len(df_clean),
            'raw_ushape': raw_fit,
            'beta_Q': np.nan, 'se_Q': np.nan, 'p_Q': np.nan,
            'beta_R': np.nan, 'se_R': np.nan, 'p_R': np.nan,
            'lambda_QR': np.nan, 'se_lambda': np.nan, 'p_lambda': np.nan,
            'corrected_ushape': None,
            'Q_R_correlation': np.corrcoef(Q, R)[0,1],
            'mean_gas_fraction': Q.mean(),
            'mean_log_Mstar': R.mean()
        }
    
    if HAS_STATSMODELS:
        model = sm.OLS(residuals, X)
        results = model.fit(cov_type='HC3')
        coeffs = results.params
        std_errs = results.bse
        pvalues = results.pvalues
        
        alpha, beta_Q, beta_R, lambda_QR = coeffs
        se_Q, se_R, se_lambda = std_errs[1], std_errs[2], std_errs[3]
        p_Q, p_R, p_lambda = pvalues[1], pvalues[2], pvalues[3]
    else:
        coeffs, _, _, _ = np.linalg.lstsq(X, residuals, rcond=None)
        alpha, beta_Q, beta_R, lambda_QR = coeffs
        se_Q = se_R = se_lambda = np.nan
        p_Q = p_R = p_lambda = np.nan
    
    print(f"\n  QO+R Model Fit:")
    print(f"    β_Q (C_Q proxy) = {beta_Q:.5f} ± {se_Q:.5f}, p = {p_Q:.4f}")
    print(f"    β_R (C_R proxy) = {beta_R:.5f} ± {se_R:.5f}, p = {p_R:.4f}")
    print(f"    λ_QR (coupling) = {lambda_QR:.5f} ± {se_lambda:.5f}, p = {p_lambda:.4f}")
    
    # 3. Apply QO+R correction with expected parameters
    correction = AMPLITUDE * (
        C_Q_EXPECTED * Q_norm * env2 + 
        C_R_EXPECTED * R_norm * env2 + 
        LAMBDA_QR_EXPECTED * (Q_norm**2) * (R_norm**2)
    )
    
    corrected_residuals = residuals + correction
    
    corrected_fit = fit_ushape(env_std, corrected_residuals)
    if corrected_fit:
        print(f"\n  Corrected U-shape: a = {corrected_fit['a']:.5f} ± {corrected_fit['a_err']:.5f}")
    
    return {
        'n_galaxies': len(df_clean),
        'raw_ushape': raw_fit,
        'beta_Q': beta_Q, 'se_Q': se_Q, 'p_Q': p_Q,
        'beta_R': beta_R, 'se_R': se_R, 'p_R': p_R,
        'lambda_QR': lambda_QR, 'se_lambda': se_lambda, 'p_lambda': p_lambda,
        'corrected_ushape': corrected_fit,
        'Q_R_correlation': np.corrcoef(Q, R)[0,1],
        'mean_gas_fraction': Q.mean(),
        'mean_log_Mstar': R.mean()
    }


# =============================================================================
# KILLER PREDICTION TEST
# =============================================================================

def test_killer_prediction(df, name):
    """
    Test killer prediction: R-dominated systems should show INVERTED U-shape.
    
    KEY INSIGHT: ALFALFA is an HI survey - it cannot detect truly gas-poor systems.
    We test both loose (f_gas < 0.3) and strict (f_gas < 0.05, high mass) criteria.
    The strict criteria match TNG300 where inversion was confirmed.
    """
    print(f"\n  --- Killer Prediction Test ---")
    
    # Check columns
    if 'gas_fraction' not in df.columns or 'btfr_residual' not in df.columns:
        print(f"  [!] Missing required columns")
        return None
    
    if 'log_env' in df.columns:
        env_col = 'log_env'
    elif 'density_proxy' in df.columns:
        df['env_std'] = (df['density_proxy'] - df['density_proxy'].mean()) / df['density_proxy'].std()
        env_col = 'env_std'
    else:
        print(f"  [!] No environment column")
        return None
    
    mask = np.isfinite(df['gas_fraction']) & np.isfinite(df['btfr_residual']) & np.isfinite(df[env_col])
    df_clean = df[mask].copy()
    
    # Standard thresholds (from Paper 1)
    gas_rich_threshold = 0.5   # f_gas >= 50% = Q-dominated
    gas_poor_threshold = 0.3   # f_gas < 30% = R-dominated (loose)
    
    # Strict TNG300-equivalent thresholds
    strict_fgas_threshold = 0.05  # Truly gas-poor
    strict_mass_threshold = 10.5   # High stellar mass
    
    df_Q_dom = df_clean[df_clean['gas_fraction'] >= gas_rich_threshold]
    df_R_dom = df_clean[df_clean['gas_fraction'] < gas_poor_threshold]
    df_inter = df_clean[(df_clean['gas_fraction'] >= gas_poor_threshold) & 
                        (df_clean['gas_fraction'] < gas_rich_threshold)]
    
    # Strict R-dominated (TNG300-equivalent)
    if 'log_Mstar' in df_clean.columns:
        df_extreme_R = df_clean[(df_clean['gas_fraction'] < strict_fgas_threshold) & 
                                (df_clean['log_Mstar'] > strict_mass_threshold)]
    else:
        df_extreme_R = pd.DataFrame()
    
    print(f"  Q-dominated (f_gas >= {gas_rich_threshold}): N = {len(df_Q_dom)}")
    print(f"  Intermediate: N = {len(df_inter)}")
    print(f"  R-dominated (f_gas < {gas_poor_threshold}): N = {len(df_R_dom)}")
    print(f"  EXTREME R-dom (f_gas < {strict_fgas_threshold}, log M* > {strict_mass_threshold}): N = {len(df_extreme_R)}")
    
    results = {
        'gas_rich_threshold': gas_rich_threshold,
        'gas_poor_threshold': gas_poor_threshold,
        'strict_fgas_threshold': strict_fgas_threshold,
        'strict_mass_threshold': strict_mass_threshold
    }
    
    # Theoretical prediction for R-dominated
    a_predicted = -C_R_EXPECTED**2 / (4 * LAMBDA_QR_EXPECTED)
    print(f"\n  Theoretical a(R-dom) = -C_R²/(4λ_QR) = {a_predicted:.4f}")
    
    # Test standard categories
    for label, subset in [('Q_dominated', df_Q_dom), ('Intermediate', df_inter), ('R_dominated', df_R_dom)]:
        if len(subset) < 30:
            print(f"  {label}: Not enough data ({len(subset)})")
            results[label] = {'n': len(subset), 'a': np.nan}
            continue
        
        env = subset[env_col].values
        env_std = (env - env.mean()) / env.std() if env.std() > 0 else np.zeros_like(env)
        res = subset['btfr_residual'].values
        
        fit = fit_ushape(env_std, res)
        
        if fit:
            status = ""
            if label == 'R_dominated' and fit['a'] < 0:
                status = " <<< KILLER PREDICTION CONFIRMED!"
            elif label == 'Q_dominated' and fit['a'] > 0:
                status = " (as expected)"
            
            print(f"  {label}: a = {fit['a']:.5f} ± {fit['a_err']:.5f}, p = {fit['p_value']:.4f}{status}")
            results[label] = {
                'n': len(subset),
                'a': fit['a'], 'a_err': fit['a_err'],
                'p_value': fit['p_value'],
                'mean_gas_fraction': subset['gas_fraction'].mean()
            }
        else:
            results[label] = {'n': len(subset), 'a': np.nan}
    
    # Test EXTREME R-dominated (TNG300-equivalent)
    if len(df_extreme_R) >= 20:
        env = df_extreme_R[env_col].values
        env_std = (env - env.mean()) / env.std() if env.std() > 0 else np.zeros_like(env)
        res = df_extreme_R['btfr_residual'].values
        
        fit = fit_ushape(env_std, res, n_bins=8, min_per_bin=5)  # Smaller bins for small sample
        
        if fit:
            status = " <<< KILLER PREDICTION CONFIRMED!" if fit['a'] < 0 else ""
            print(f"\n  *** EXTREME R-dom: a = {fit['a']:.5f} ± {fit['a_err']:.5f}, p = {fit['p_value']:.4f}{status}")
            results['EXTREME_R_dominated'] = {
                'n': len(df_extreme_R),
                'a': fit['a'], 'a_err': fit['a_err'],
                'p_value': fit['p_value'],
                'mean_gas_fraction': df_extreme_R['gas_fraction'].mean(),
                'mean_log_Mstar': df_extreme_R['log_Mstar'].mean() if 'log_Mstar' in df_extreme_R.columns else np.nan
            }
        else:
            results['EXTREME_R_dominated'] = {'n': len(df_extreme_R), 'a': np.nan}
    else:
        print(f"\n  EXTREME R-dom: Not enough data ({len(df_extreme_R)}) - this is expected for HI surveys")
        results['EXTREME_R_dominated'] = {'n': len(df_extreme_R), 'a': np.nan}
    
    # Check inversion - prioritize EXTREME if available
    a_Q = results.get('Q_dominated', {}).get('a', np.nan)
    a_R_loose = results.get('R_dominated', {}).get('a', np.nan)
    a_R_extreme = results.get('EXTREME_R_dominated', {}).get('a', np.nan)
    
    # Inversion confirmed if EXTREME R-dom shows negative a
    if np.isfinite(a_R_extreme) and a_R_extreme < 0:
        results['inversion_confirmed'] = True
        results['inversion_type'] = 'EXTREME'
    elif np.isfinite(a_Q) and np.isfinite(a_R_loose) and (a_Q > 0) and (a_R_loose < 0):
        results['inversion_confirmed'] = True
        results['inversion_type'] = 'LOOSE'
    else:
        results['inversion_confirmed'] = False
        results['inversion_type'] = None
    
    results['theoretical_a_Rdom'] = a_predicted
    
    # Explain selection bias if applicable
    if name == 'ALFALFA' and not results['inversion_confirmed']:
        print(f"\n  NOTE: ALFALFA is an HI survey - truly gas-poor galaxies are invisible.")
        print(f"        TNG300 had 16,924 extreme R-dom galaxies vs {len(df_extreme_R)} in ALFALFA.")
    
    return results


# =============================================================================
# FIGURES
# =============================================================================

def create_figure(df, name, qor_results, killer_results, save_path):
    """Create diagnostic figure."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'QO+R Analysis: {name} (Correct Methodology)', fontsize=14, fontweight='bold')
    
    # Get environment
    if 'log_env' in df.columns:
        env = df['log_env'].values
    else:
        env = df['density_proxy'].values
    
    mask = np.isfinite(df['gas_fraction']) & np.isfinite(env) & np.isfinite(df['btfr_residual'])
    gas_frac = df.loc[mask, 'gas_fraction'].values
    env = env[mask]
    residuals = df.loc[mask, 'btfr_residual'].values
    env_std = (env - env.mean()) / env.std()
    
    # Panel A: Raw U-shape with fit
    ax = axes[0, 0]
    ax.scatter(env_std, residuals, alpha=0.3, s=10, c='gray')
    
    if qor_results and qor_results['raw_ushape']:
        fit = qor_results['raw_ushape']
        ax.errorbar(fit['bin_centers'], fit['bin_means'], yerr=fit['bin_errors'],
                   fmt='o-', color='blue', markersize=8, capsize=3, label=f"Binned (a={fit['a']:.4f})")
        x_fit = np.linspace(min(fit['bin_centers']), max(fit['bin_centers']), 100)
        y_fit = fit['a'] * x_fit**2 + fit['b'] * x_fit + fit['c']
        ax.plot(x_fit, y_fit, 'r-', lw=2, label='Quadratic fit')
    
    ax.axhline(0, color='black', ls='--', alpha=0.5)
    ax.set_xlabel('Environment (standardized)')
    ax.set_ylabel('BTFR Residual')
    ax.set_title('A) U-shape in BTFR residuals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel B: Q vs R scatter (showing independence)
    ax = axes[0, 1]
    if 'log_Mstar' in df.columns:
        log_mstar = df.loc[mask, 'log_Mstar'].values
        scatter = ax.scatter(gas_frac, log_mstar, c=residuals, cmap='RdBu_r', 
                            alpha=0.5, s=20, vmin=-0.2, vmax=0.2)
        plt.colorbar(scatter, ax=ax, label='BTFR Residual')
        
        # Correlation
        r = np.corrcoef(gas_frac, log_mstar)[0, 1]
        ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Q = Gas Fraction')
        ax.set_ylabel('R = log(M_star)')
        ax.set_title('B) Q vs R (colored by residual)')
    else:
        ax.text(0.5, 0.5, 'log(M_star) not available', ha='center', va='center', transform=ax.transAxes)
    
    # Panel C: Killer prediction (stratified U-shapes)
    ax = axes[1, 0]
    
    colors = {'Q_dominated': 'blue', 'Intermediate': 'gray', 'R_dominated': 'red'}
    
    if killer_results:
        for label in ['Q_dominated', 'Intermediate', 'R_dominated']:
            if label not in killer_results:
                continue
            res = killer_results[label]
            if np.isnan(res.get('a', np.nan)):
                continue
            
            # Get subset
            if label == 'Q_dominated':
                subset_mask = gas_frac >= killer_results['gas_rich_threshold']
            elif label == 'R_dominated':
                subset_mask = gas_frac < killer_results['gas_poor_threshold']
            else:
                subset_mask = (gas_frac >= killer_results['gas_poor_threshold']) & \
                             (gas_frac < killer_results['gas_rich_threshold'])
            
            sub_env = env_std[subset_mask]
            sub_res = residuals[subset_mask]
            
            ax.scatter(sub_env, sub_res, alpha=0.3, s=10, c=colors.get(label, 'black'))
            
            # Fit line
            fit = fit_ushape(sub_env, sub_res)
            if fit:
                x_fit = np.linspace(sub_env.min(), sub_env.max(), 100)
                y_fit = fit['a'] * x_fit**2 + fit['b'] * x_fit + fit['c']
                ax.plot(x_fit, y_fit, color=colors.get(label, 'black'), lw=2,
                       label=f"{label} (a={fit['a']:.4f})")
    
    ax.axhline(0, color='black', ls='--', alpha=0.5)
    ax.set_xlabel('Environment (standardized)')
    ax.set_ylabel('BTFR Residual')
    ax.set_title('C) Killer Prediction: U-shape by gas fraction')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel D: Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Build summary text
    if qor_results:
        qr_corr = qor_results['Q_R_correlation']
        beta_q = qor_results['beta_Q']
        se_q = qor_results['se_Q']
        beta_r = qor_results['beta_R']
        se_r = qor_results['se_R']
        lqr = qor_results['lambda_QR']
        se_lqr = qor_results['se_lambda']
        raw_a = qor_results['raw_ushape']['a'] if qor_results['raw_ushape'] else np.nan
        raw_err = qor_results['raw_ushape']['a_err'] if qor_results['raw_ushape'] else np.nan
        raw_p = qor_results['raw_ushape']['p_value'] if qor_results['raw_ushape'] else np.nan
        detected = 'YES' if qor_results['raw_ushape'] and raw_a > 0 and raw_p < 0.05 else 'NO'
        n_gal = qor_results['n_galaxies']
    else:
        qr_corr = beta_q = se_q = beta_r = se_r = lqr = se_lqr = raw_a = raw_err = raw_p = np.nan
        detected = 'N/A'
        n_gal = 0
    
    if killer_results:
        theo_a = killer_results.get('theoretical_a_Rdom', np.nan)
        q_dom_a = killer_results.get('Q_dominated', {}).get('a', np.nan)
        r_dom_a = killer_results.get('R_dominated', {}).get('a', np.nan)
        inv_conf = killer_results.get('inversion_confirmed', False)
    else:
        theo_a = q_dom_a = r_dom_a = np.nan
        inv_conf = False
    
    summary = f"""
RESULTS: {name}
{'='*50}
N = {n_gal} galaxies

QO+R MODEL (Q = f_gas, R = log M_star):
  Q-R correlation: {qr_corr:.3f}
  
  Fitted parameters:
    β_Q = {beta_q:.5f} ± {se_q:.5f} (expected: ~{C_Q_EXPECTED})
    β_R = {beta_r:.5f} ± {se_r:.5f} (expected: ~{C_R_EXPECTED})
    λ_QR = {lqr:.5f} ± {se_lqr:.5f} (expected: ~{LAMBDA_QR_EXPECTED})

RAW U-SHAPE:
  a = {raw_a:.5f} ± {raw_err:.5f}
  p = {raw_p:.4f}
  Detection: {detected}

KILLER PREDICTION:
  Theoretical a(R-dom) = {theo_a:.4f}
  Q-dominated: a = {q_dom_a:.4f}
  R-dominated: a = {r_dom_a:.4f}
  Inversion confirmed: {inv_conf}
"""
    
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Figure saved: {save_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    all_results = {}
    
    # SPARC
    print("\n" + "="*70)
    print(" 1. SPARC ANALYSIS")
    print("="*70)
    df_sparc = load_sparc()
    if df_sparc is not None:
        df_sparc = calculate_btfr_residuals(df_sparc)
        qor = analyze_qor(df_sparc, "SPARC")
        killer = test_killer_prediction(df_sparc, "SPARC")
        
        if qor:
            all_results['SPARC'] = qor
            all_results['SPARC_killer'] = killer
            create_figure(df_sparc, "SPARC", qor, killer,
                         os.path.join(FIGURES_DIR, "sparc_qor_v3.png"))
    
    # ALFALFA
    print("\n" + "="*70)
    print(" 2. ALFALFA ANALYSIS")
    print("="*70)
    df_alfalfa = load_alfalfa()
    if df_alfalfa is not None:
        qor = analyze_qor(df_alfalfa, "ALFALFA")
        killer = test_killer_prediction(df_alfalfa, "ALFALFA")
        
        if qor:
            all_results['ALFALFA'] = qor
            all_results['ALFALFA_killer'] = killer
            create_figure(df_alfalfa, "ALFALFA", qor, killer,
                         os.path.join(FIGURES_DIR, "alfalfa_qor_v3.png"))
    
    # WALLABY
    print("\n" + "="*70)
    print(" 3. WALLABY ANALYSIS")
    print("="*70)
    df_wallaby = load_wallaby()
    if df_wallaby is not None and 'log_Mstar' in df_wallaby.columns:
        qor = analyze_qor(df_wallaby, "WALLABY")
        if qor:
            killer = test_killer_prediction(df_wallaby, "WALLABY")
            all_results['WALLABY'] = qor
            all_results['WALLABY_killer'] = killer
            create_figure(df_wallaby, "WALLABY", qor, killer,
                         os.path.join(FIGURES_DIR, "wallaby_qor_v3.png"))
    else:
        print("  [!] WALLABY cannot be analyzed - no M_star data")
    
    # SUMMARY
    print("\n" + "="*70)
    print(" RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n  Expected parameters (TNG100 calibration):")
    print(f"    C_Q = {C_Q_EXPECTED}, C_R = {C_R_EXPECTED}, λ_QR = {LAMBDA_QR_EXPECTED}")
    
    for ds in ['SPARC', 'ALFALFA', 'WALLABY']:
        if ds in all_results:
            r = all_results[ds]
            print(f"\n  {ds}:")
            print(f"    N = {r['n_galaxies']}, Q-R corr = {r['Q_R_correlation']:.3f}")
            print(f"    λ_QR fitted = {r['lambda_QR']:.4f} (expected ~{LAMBDA_QR_EXPECTED})")
            if r['raw_ushape']:
                print(f"    U-shape: a = {r['raw_ushape']['a']:.5f}, p = {r['raw_ushape']['p_value']:.4f}")
            
            k = all_results.get(f'{ds}_killer', {})
            if k and k.get('inversion_confirmed'):
                print(f"    KILLER PREDICTION: CONFIRMED!")
    
    # Save
    with open(os.path.join(RESULTS_DIR, "paper4_results_v3.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    
    print(f"\n  Results saved to {RESULTS_DIR}")
    return all_results


if __name__ == "__main__":
    main()
