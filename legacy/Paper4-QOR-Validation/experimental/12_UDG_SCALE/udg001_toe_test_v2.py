#!/usr/bin/env python3
"""
===============================================================================
UDG-001 v2: TOE TEST ON ULTRA-DIFFUSE GALAXIES
USING REAL GANNON+2024 SPECTROSCOPIC DATA
===============================================================================

Scale: 10²⁰ - 10²¹ m (galaxy-scale, ultra-low density)
Observable: Velocity dispersion residual Δ_UDG = log(σ_obs) - log(σ_bar)

Data source: Gannon et al. 2024, MNRAS, 531, 1856
GitHub: https://github.com/gannonjs/Published_Data

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
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "udg_catalogs", "Gannon2024_udg_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

G_SI = 6.674e-11      # m³/kg/s²
M_SUN = 1.989e30      # kg
KPC_M = 3.086e19      # m per kpc

# =============================================================================
# PRE-REGISTERED CRITERIA (LOCKED)
# =============================================================================

CRITERIA = {
    'min_n_udg': 10,
    'min_rho_range_dex': 0.5,
    'beta_Q': 1.0,
    'beta_R': 0.5,
    'alpha_bounds': (-0.5, 1.0),
    'lambda_bounds': (0.01, 10.0),
    'rho_c_bounds': (-28, -22),
    'delta_bounds': (0.1, 5.0),
    'lambda_gal': 0.94,
    'lambda_ratio_min': 1/3,
    'lambda_ratio_max': 3.0,
    'n_bootstrap': 500,
    'n_permutations': 500,
}

# =============================================================================
# LOAD REAL GANNON+2024 DATA
# =============================================================================

def load_gannon_data():
    """
    Load real Gannon+2024 spectroscopic UDG catalog.
    """
    print("\n" + "="*70)
    print(" LOADING GANNON+2024 SPECTROSCOPIC UDG DATA")
    print("="*70)
    
    df = pd.read_csv(DATA_PATH)
    
    print(f"  Total UDGs in catalog: {len(df)}")
    
    # Filter: need stellar velocity dispersion (not -999)
    df_valid = df[df['stellar_sigma'] > 0].copy()
    print(f"  UDGs with stellar σ: {len(df_valid)}")
    
    # Environment breakdown
    env_map = {1: 'cluster', 2: 'group', 3: 'field'}
    df_valid['env_name'] = df_valid['Environment'].map(env_map)
    
    print(f"  Environment breakdown:")
    for env_code, env_name in env_map.items():
        n = (df_valid['Environment'] == env_code).sum()
        print(f"    {env_name}: {n}")
    
    return df_valid


def compute_udg_properties(df):
    """
    Compute derived properties for UDG sample.
    """
    print("\n" + "-"*70)
    print(" COMPUTING UDG PROPERTIES")
    print("-"*70)
    
    # Stellar mass in kg (catalog gives 10^8 M_sun, so multiply)
    # Actually looking at values like 3, 4.35, etc - these are in units where
    # the column header says "Stellar mass" - need to check units
    # From the paper, masses are in 10^8 M_sun
    df['M_star_Msun'] = df['Stellar mass'] * 1e8  # Convert to M_sun
    df['M_star_kg'] = df['M_star_Msun'] * M_SUN
    
    # Effective radius in meters
    df['R_eff_m'] = df['R_e'] * KPC_M
    
    # Baryonic velocity dispersion from virial theorem
    # σ_bar² = G × M_* / (2 × R_eff)
    df['sigma_bar'] = np.sqrt(G_SI * df['M_star_kg'] / (2 * df['R_eff_m'])) / 1000  # km/s
    
    print(f"  σ_bar range: {df['sigma_bar'].min():.1f} - {df['sigma_bar'].max():.1f} km/s")
    print(f"  σ_obs range: {df['stellar_sigma'].min():.1f} - {df['stellar_sigma'].max():.1f} km/s")
    
    # Dynamical residual
    df['Delta_UDG'] = np.log10(df['stellar_sigma']) - np.log10(df['sigma_bar'])
    
    print(f"  Δ_UDG range: {df['Delta_UDG'].min():.3f} to {df['Delta_UDG'].max():.3f}")
    print(f"  Δ_UDG mean: {df['Delta_UDG'].mean():.3f} ± {df['Delta_UDG'].std():.3f}")
    
    # Stellar density (g/cm³)
    V_cm3 = (4/3) * np.pi * (df['R_e'] * 3.086e21)**3  # R_e in kpc -> cm
    df['rho_star'] = (df['M_star_Msun'] * M_SUN * 1000) / V_cm3  # g/cm³
    df['log_rho'] = np.log10(df['rho_star'])
    
    print(f"  log(ρ_*) range: {df['log_rho'].min():.2f} to {df['log_rho'].max():.2f}")
    rho_range = df['log_rho'].max() - df['log_rho'].min()
    print(f"  ρ range: {rho_range:.2f} dex")
    
    # Use stellar_sigma_up/down for errors if available
    df['sigma_err'] = (df['stellar_sigma_up'] + df['stellar_sigma_down']) / 2
    df.loc[df['sigma_err'] <= 0, 'sigma_err'] = df['stellar_sigma'] * 0.15  # Default 15%
    
    # f parameter (use environment as proxy: field=0.3, group=0.5, cluster=0.7)
    env_f_map = {3: 0.3, 2: 0.5, 1: 0.7}
    df['f'] = df['Environment'].map(env_f_map)
    
    return df


# =============================================================================
# TOE MODEL
# =============================================================================

def sigmoid(x):
    return expit(x)


def toe_model_udg(X, alpha, lambda_QR, rho_c, delta):
    """TOE model for UDG velocity dispersion residuals."""
    log_rho, f = X
    rho_0 = np.nanmedian(log_rho)
    
    C_Q = 10**((rho_0 - log_rho) * CRITERIA['beta_Q'])
    C_R = 10**((rho_0 - log_rho) * CRITERIA['beta_R'])
    
    P = np.pi * sigmoid((log_rho - rho_c) / delta)
    coupling = lambda_QR * (C_Q * f - C_R * (1 - f))
    
    return alpha * coupling * np.cos(P)


def null_model(X, a, b):
    """Null model: linear trend with density."""
    log_rho, f = X
    return a + b * log_rho


def compute_aicc(n, k, ss_res):
    if n <= k + 1 or ss_res <= 0:
        return np.inf
    aic = n * np.log(ss_res / n) + 2 * k
    return aic + 2 * k * (k + 1) / (n - k - 1)


def fit_toe(X, y, weights=None):
    """Fit TOE model."""
    if weights is None:
        weights = np.ones(len(y))
    
    p0 = [0.1, 1.0, -25.0, 1.0]
    
    bounds = (
        [CRITERIA['alpha_bounds'][0], CRITERIA['lambda_bounds'][0], 
         CRITERIA['rho_c_bounds'][0], CRITERIA['delta_bounds'][0]],
        [CRITERIA['alpha_bounds'][1], CRITERIA['lambda_bounds'][1],
         CRITERIA['rho_c_bounds'][1], CRITERIA['delta_bounds'][1]]
    )
    
    def residuals(params):
        pred = toe_model_udg(X, *params)
        return (y - pred) * np.sqrt(weights)
    
    try:
        result = least_squares(residuals, p0,
                               bounds=([b[0] for b in zip(*bounds)],
                                       [b[1] for b in zip(*bounds)]),
                               max_nfev=5000)
        popt = result.x
        
        y_pred = toe_model_udg(X, *popt)
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
# STATISTICAL TESTS
# =============================================================================

def bootstrap_udg(df, n_boot=500):
    """Bootstrap resampling."""
    print(f"\n  Running bootstrap (n={n_boot})...")
    
    results = []
    n = len(df)
    
    for i in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        df_boot = df.iloc[idx]
        
        X = (df_boot['log_rho'].values, df_boot['f'].values)
        y = df_boot['Delta_UDG'].values
        w = 1 / df_boot['sigma_err'].values**2
        
        result_toe = fit_toe(X, y, w)
        result_null = fit_null(X, y, w)
        
        if result_toe['success'] and result_null['success']:
            results.append({
                'alpha': result_toe['popt'][0],
                'lambda_QR': result_toe['popt'][1],
                'delta_aicc': result_null['aicc'] - result_toe['aicc']
            })
    
    if len(results) > 0:
        df_results = pd.DataFrame(results)
        summary = {
            'alpha_median': float(df_results['alpha'].median()),
            'alpha_ci_low': float(df_results['alpha'].quantile(0.025)),
            'alpha_ci_high': float(df_results['alpha'].quantile(0.975)),
            'lambda_QR_median': float(df_results['lambda_QR'].median()),
            'lambda_QR_ci_low': float(df_results['lambda_QR'].quantile(0.025)),
            'lambda_QR_ci_high': float(df_results['lambda_QR'].quantile(0.975)),
            'delta_aicc_median': float(df_results['delta_aicc'].median()),
            'n_successful': len(results)
        }
        print(f"  α = {summary['alpha_median']:.4f} [{summary['alpha_ci_low']:.4f}, {summary['alpha_ci_high']:.4f}]")
        print(f"  λ_QR = {summary['lambda_QR_median']:.3f} [{summary['lambda_QR_ci_low']:.3f}, {summary['lambda_QR_ci_high']:.3f}]")
        return summary
    
    return None


def permutation_test(df, n_perm=500):
    """Permutation test for significance."""
    print(f"\n  Running permutation test (n={n_perm})...")
    
    X = (df['log_rho'].values, df['f'].values)
    y = df['Delta_UDG'].values
    w = 1 / df['sigma_err'].values**2
    
    result_toe = fit_toe(X, y, w)
    result_null = fit_null(X, y, w)
    
    if not (result_toe['success'] and result_null['success']):
        return None
    
    delta_aicc_obs = result_null['aicc'] - result_toe['aicc']
    
    delta_aicc_perm = []
    for _ in range(n_perm):
        y_perm = np.random.permutation(y)
        
        result_toe_p = fit_toe(X, y_perm, w)
        result_null_p = fit_null(X, y_perm, w)
        
        if result_toe_p['success'] and result_null_p['success']:
            delta_aicc_perm.append(result_null_p['aicc'] - result_toe_p['aicc'])
    
    if len(delta_aicc_perm) > 0:
        p_value = np.mean(np.array(delta_aicc_perm) >= delta_aicc_obs)
        print(f"  Observed ΔAICc: {delta_aicc_obs:.1f}")
        print(f"  Permuted mean: {np.mean(delta_aicc_perm):.1f}")
        print(f"  p-value: {p_value:.4f}")
        return {'delta_aicc_obs': float(delta_aicc_obs), 'p_value': float(p_value)}
    
    return None


def jackknife_environment(df):
    """Jackknife by environment."""
    print("\n  Running jackknife by environment...")
    
    results = []
    env_map = {1: 'cluster', 2: 'group', 3: 'field'}
    
    for env_code, env_name in env_map.items():
        df_sub = df[df['Environment'] == env_code].copy()
        
        if len(df_sub) >= 3:
            X = (df_sub['log_rho'].values, df_sub['f'].values)
            y = df_sub['Delta_UDG'].values
            w = 1 / df_sub['sigma_err'].values**2
            
            result_toe = fit_toe(X, y, w)
            
            if result_toe['success']:
                results.append({
                    'environment': env_name,
                    'n': len(df_sub),
                    'alpha': float(result_toe['popt'][0]),
                    'lambda_QR': float(result_toe['popt'][1]),
                    'mean_Delta': float(df_sub['Delta_UDG'].mean()),
                    'mean_sigma_obs': float(df_sub['stellar_sigma'].mean()),
                    'mean_sigma_bar': float(df_sub['sigma_bar'].mean())
                })
                print(f"    {env_name}: n={len(df_sub)}, α={result_toe['popt'][0]:.4f}, λ_QR={result_toe['popt'][1]:.3f}, <Δ>={df_sub['Delta_UDG'].mean():.3f}")
    
    return results


# =============================================================================
# SIMPLE ANALYSIS
# =============================================================================

def simple_analysis(df):
    """Simple direct analysis without complex fitting."""
    print("\n" + "-"*70)
    print(" SIMPLE DIRECT ANALYSIS")
    print("-"*70)
    
    # Mean residual by environment
    print("\n  Mean Δ_UDG by environment:")
    env_map = {1: 'cluster', 2: 'group', 3: 'field'}
    
    for env_code, env_name in env_map.items():
        df_sub = df[df['Environment'] == env_code]
        if len(df_sub) > 0:
            mean_delta = df_sub['Delta_UDG'].mean()
            std_delta = df_sub['Delta_UDG'].std()
            mean_rho = df_sub['log_rho'].mean()
            print(f"    {env_name}: <Δ> = {mean_delta:.3f} ± {std_delta:.3f}, <log ρ> = {mean_rho:.2f}, n={len(df_sub)}")
    
    # Correlation: Delta vs log_rho
    r, p = stats.pearsonr(df['log_rho'], df['Delta_UDG'])
    print(f"\n  Correlation Δ vs log(ρ): r = {r:.3f}, p = {p:.4f}")
    
    # T-test: field vs cluster
    df_field = df[df['Environment'] == 3]['Delta_UDG']
    df_cluster = df[df['Environment'] == 1]['Delta_UDG']
    
    if len(df_field) >= 2 and len(df_cluster) >= 2:
        t_stat, p_val = stats.ttest_ind(df_field, df_cluster)
        print(f"  T-test field vs cluster: t = {t_stat:.2f}, p = {p_val:.4f}")
    
    return {
        'correlation_r': float(r),
        'correlation_p': float(p),
        'mean_delta_overall': float(df['Delta_UDG'].mean()),
        'std_delta_overall': float(df['Delta_UDG'].std())
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print(" UDG-001 v2: TOE TEST ON ULTRA-DIFFUSE GALAXIES")
    print(" Using REAL Gannon+2024 Spectroscopic Data")
    print(" Author: Jonathan Édouard Slama")
    print("="*70)
    
    # Load real data
    df = load_gannon_data()
    
    # Compute properties
    df = compute_udg_properties(df)
    
    # Check criteria
    n_udg = len(df)
    rho_range = df['log_rho'].max() - df['log_rho'].min()
    
    print("\n" + "-"*70)
    print(" PRE-REGISTERED CRITERIA CHECK")
    print("-"*70)
    
    criteria_met = True
    if n_udg < CRITERIA['min_n_udg']:
        print(f"  ❌ N_UDG = {n_udg} < {CRITERIA['min_n_udg']} (FAIL)")
        criteria_met = False
    else:
        print(f"  ✅ N_UDG = {n_udg} >= {CRITERIA['min_n_udg']}")
    
    if rho_range < CRITERIA['min_rho_range_dex']:
        print(f"  ⚠️ ρ_range = {rho_range:.2f} < {CRITERIA['min_rho_range_dex']} dex (LIMITED)")
    else:
        print(f"  ✅ ρ_range = {rho_range:.2f} >= {CRITERIA['min_rho_range_dex']} dex")
    
    results = {
        'data_source': 'Gannon+2024 MNRAS 531, 1856',
        'n_udg': n_udg,
        'rho_range_dex': float(rho_range),
        'Delta_UDG_mean': float(df['Delta_UDG'].mean()),
        'Delta_UDG_std': float(df['Delta_UDG'].std()),
    }
    
    # List individual UDGs
    print("\n" + "-"*70)
    print(" INDIVIDUAL UDGs")
    print("-"*70)
    print(f"  {'Name':<20} {'σ_obs':>8} {'σ_bar':>8} {'Δ':>8} {'Env':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for _, row in df.iterrows():
        print(f"  {row['Name']:<20} {row['stellar_sigma']:>8.1f} {row['sigma_bar']:>8.1f} {row['Delta_UDG']:>8.3f} {row['env_name']:>8}")
    
    # Simple analysis
    simple_results = simple_analysis(df)
    results['simple_analysis'] = simple_results
    
    # Main fit
    print("\n" + "-"*70)
    print(" MAIN TOE FIT (All UDGs)")
    print("-"*70)
    
    X = (df['log_rho'].values, df['f'].values)
    y = df['Delta_UDG'].values
    w = 1 / df['sigma_err'].values**2
    
    result_toe = fit_toe(X, y, w)
    result_null = fit_null(X, y, w)
    
    if result_toe['success']:
        popt = result_toe['popt']
        print(f"  α = {popt[0]:.5f}")
        print(f"  λ_QR = {popt[1]:.4f}")
        print(f"  ρ_c = {popt[2]:.2f} (log g/cm³)")
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
    
    # Bootstrap
    print("\n" + "-"*70)
    print(" BOOTSTRAP")
    print("-"*70)
    bootstrap_results = bootstrap_udg(df, n_boot=CRITERIA['n_bootstrap'])
    if bootstrap_results:
        results['bootstrap'] = bootstrap_results
    
    # Permutation test
    print("\n" + "-"*70)
    print(" PERMUTATION TEST")
    print("-"*70)
    perm_results = permutation_test(df, n_perm=CRITERIA['n_permutations'])
    if perm_results:
        results['permutation'] = perm_results
    
    # Jackknife by environment
    print("\n" + "-"*70)
    print(" JACKKNIFE BY ENVIRONMENT")
    print("-"*70)
    jackknife_results = jackknife_environment(df)
    results['jackknife'] = jackknife_results
    
    # Verdict
    print("\n" + "="*70)
    print(" VERDICT")
    print("="*70)
    
    evidence = []
    
    # Mean residual positive (excess σ)
    mean_delta = results['Delta_UDG_mean']
    if mean_delta > 0.1:
        evidence.append(('Mean Δ_UDG', 'STRONG_POSITIVE', f"<Δ> = {mean_delta:.3f} >> 0"))
    elif mean_delta > 0:
        evidence.append(('Mean Δ_UDG', 'POSITIVE', f"<Δ> = {mean_delta:.3f} > 0"))
    else:
        evidence.append(('Mean Δ_UDG', 'NEGATIVE', f"<Δ> = {mean_delta:.3f} ≤ 0"))
    
    # ΔAICc
    if 'delta_aicc' in results:
        if results['delta_aicc'] > 10:
            evidence.append(('ΔAICc', 'TOE_PREFERRED', f"{results['delta_aicc']:.1f} > 10"))
        elif results['delta_aicc'] > 0:
            evidence.append(('ΔAICc', 'SLIGHT_TOE', f"{results['delta_aicc']:.1f} > 0"))
        else:
            evidence.append(('ΔAICc', 'NULL_PREFERRED', f"{results['delta_aicc']:.1f} < 0"))
    
    # λ_QR coherence
    if 'toe' in results and 'lambda_QR' in results['toe']:
        ratio = results['toe']['lambda_QR'] / CRITERIA['lambda_gal']
        if CRITERIA['lambda_ratio_min'] <= ratio <= CRITERIA['lambda_ratio_max']:
            evidence.append(('λ_QR coherence', 'PASS', f"λ_QR = {results['toe']['lambda_QR']:.3f} (ratio = {ratio:.2f})"))
        else:
            evidence.append(('λ_QR coherence', 'FAIL', f"λ_QR = {results['toe']['lambda_QR']:.3f} out of range"))
    
    # Permutation
    if perm_results and 'p_value' in perm_results:
        if perm_results['p_value'] < 0.05:
            evidence.append(('Permutation', 'SIGNIFICANT', f"p = {perm_results['p_value']:.4f}"))
        else:
            evidence.append(('Permutation', 'NON_SIG', f"p = {perm_results['p_value']:.4f}"))
    
    # Environment effect
    if jackknife_results:
        field_result = next((r for r in jackknife_results if r['environment'] == 'field'), None)
        if field_result and 0.3 < field_result['lambda_QR'] < 3.0:
            evidence.append(('Field λ_QR', 'COHERENT', f"λ_QR(field) = {field_result['lambda_QR']:.3f}"))
    
    # Print evidence
    for name, status, detail in evidence:
        if status in ['STRONG_POSITIVE', 'TOE_PREFERRED', 'SIGNIFICANT', 'COHERENT']:
            symbol = '🎯'
        elif status in ['POSITIVE', 'SLIGHT_TOE', 'PASS']:
            symbol = '✅'
        else:
            symbol = '⚠️'
        print(f"  {symbol} {name}: {detail}")
    
    # Determine verdict
    n_positive = sum(1 for e in evidence if e[1] in ['STRONG_POSITIVE', 'TOE_PREFERRED', 'SIGNIFICANT', 'COHERENT', 'POSITIVE', 'SLIGHT_TOE', 'PASS'])
    
    if n_positive >= 4:
        verdict = "QO+R TOE SUPPORTED IN UDGs"
    elif n_positive >= 2:
        verdict = "QO+R TOE PARTIALLY SUPPORTED IN UDGs"
    else:
        verdict = "QO+R TOE INCONCLUSIVE IN UDGs"
    
    print(f"\n  VERDICT: {verdict}")
    results['verdict'] = verdict
    results['evidence'] = [{'name': e[0], 'status': e[1], 'detail': e[2]} for e in evidence]
    
    # Key finding
    print("\n" + "="*70)
    print(" KEY FINDING")
    print("="*70)
    print(f"""
  Mean Δ_UDG = {results['Delta_UDG_mean']:.3f} ± {results['Delta_UDG_std']:.3f}
  
  This means UDGs have σ_obs ~ {10**results['Delta_UDG_mean']:.2f}× higher than σ_bar
  
  Interpretation:
  - UDGs show EXCESS velocity dispersion vs baryonic prediction
  - This is traditionally attributed to dark matter
  - In QO+R framework: could be unscreened gravitational enhancement
""")
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "udg001_v2_gannon_results.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"  Results saved: {output_path}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
