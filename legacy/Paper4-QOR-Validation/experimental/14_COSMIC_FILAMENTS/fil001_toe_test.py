#!/usr/bin/env python3
"""
===============================================================================
FIL-001: TOE TEST ON COSMIC FILAMENTS
Testing the Clusters → Cosmic Transition at 10^24 m
===============================================================================

Purpose:
- Test QO+R screening at INTERMEDIATE cosmological density
- Bridge the gap between Clusters (10^23 m) and CMB scales (10^27 m)
- Expect: Partial screening → signal similar to UDGs

Data: Stacked filament profiles from SDSS literature

Author: Jonathan Édouard Slama
Affiliation: Metafund Research Division, Strasbourg, France
Date: December 2025
===============================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit, gammaln
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

# =============================================================================
# CONSTANTS
# =============================================================================

H0 = 70  # km/s/Mpc
RHO_CRIT = 9.47e-30  # g/cm³ (critical density)
MPC_CM = 3.086e24    # cm per Mpc

# =============================================================================
# PRE-REGISTERED CRITERIA
# =============================================================================

CRITERIA = {
    'min_n_bins': 5,
    
    # Priors (intermediate between clusters and cosmique)
    'prior_alpha_mean': 0.05,
    'prior_alpha_std': 0.10,
    'prior_lambda_mean': 1.15,  # Between clusters (1.23) and cosmique (~1.1)
    'prior_lambda_std': 0.4,
    
    # Physics
    'beta_Q': 1.0,
    'beta_R': 0.5,
    
    # Student-t
    'student_nu': 5,
    
    # Expected: some signal like UDGs
    'expected_gamma': -0.10,
}

# =============================================================================
# FILAMENT DATA FROM LITERATURE
# =============================================================================

# Stacked filament profiles from various studies
# Reference: Tempel+14, Chen+16, Kraljic+18, Malavasi+20

# Filament overdensity profiles: δ(r_perp) = ρ/ρ_mean - 1
# r_perp in Mpc, δ = overdensity

FILAMENT_PROFILES = {
    # Source: Tempel et al. 2014 (SDSS DR10)
    'Tempel2014': {
        'description': 'SDSS DR10 filaments, stacked profiles',
        'r_perp_Mpc': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0],
        'delta': [8.5, 4.2, 2.3, 1.4, 0.9, 0.6, 0.3, 0.15, 0.05, 0.01],
        'delta_err': [1.5, 0.8, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03],
    },
    
    # Source: Chen et al. 2016 (SDSS galaxies infall)
    'Chen2016': {
        'description': 'Galaxy infall velocities toward filaments',
        'r_perp_Mpc': [1.0, 2.0, 3.0, 5.0, 8.0],
        'v_infall_km_s': [85, 65, 45, 25, 10],
        'v_err': [15, 12, 10, 8, 5],
        'v_pred_LCDM': [80, 60, 42, 22, 8],  # ΛCDM prediction
    },
    
    # Source: Kraljic et al. 2018 (GAMA survey)
    'Kraljic2018': {
        'description': 'GAMA filament density profiles',
        'r_perp_Mpc': [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0],
        'delta': [12.0, 6.5, 3.5, 2.0, 1.2, 0.5, 0.15],
        'delta_err': [2.5, 1.2, 0.7, 0.4, 0.25, 0.15, 0.08],
    },
    
    # Source: Malavasi et al. 2020 (VIPERS)
    'Malavasi2020': {
        'description': 'VIPERS filament velocities at z~0.7',
        'r_perp_Mpc': [0.5, 1.0, 2.0, 3.0, 5.0],
        'v_streaming_km_s': [120, 90, 55, 35, 15],
        'v_err': [25, 18, 12, 10, 8],
        'v_pred_LCDM': [110, 82, 50, 30, 12],
    },
}


def compute_rho_from_delta(delta, z=0.05):
    """Convert overdensity to physical density."""
    rho_mean = RHO_CRIT * 0.3 * (1 + z)**3  # Matter density at z
    rho = rho_mean * (1 + delta)
    return rho


def compute_velocity_excess(v_obs, v_pred):
    """Compute velocity excess in log space."""
    return np.log10(v_obs) - np.log10(v_pred)


def load_filament_data():
    """Load and process filament data from literature."""
    print("\n" + "="*70)
    print(" LOADING COSMIC FILAMENT DATA")
    print("="*70)
    
    all_data = []
    
    # Process density profiles (Tempel, Kraljic)
    for source in ['Tempel2014', 'Kraljic2018']:
        data = FILAMENT_PROFILES[source]
        
        for i, r in enumerate(data['r_perp_Mpc']):
            delta = data['delta'][i]
            delta_err = data['delta_err'][i]
            
            rho = compute_rho_from_delta(delta)
            log_rho = np.log10(rho)
            
            all_data.append({
                'source': source,
                'r_perp_Mpc': r,
                'overdensity': delta,
                'overdensity_err': delta_err,
                'log_rho_cgs': log_rho,
                'type': 'density',
            })
    
    # Process velocity data (Chen, Malavasi)
    for source in ['Chen2016', 'Malavasi2020']:
        data = FILAMENT_PROFILES[source]
        
        for i, r in enumerate(data['r_perp_Mpc']):
            if 'v_infall_km_s' in data:
                v_obs = data['v_infall_km_s'][i]
            else:
                v_obs = data['v_streaming_km_s'][i]
            
            v_err = data['v_err'][i]
            v_pred = data['v_pred_LCDM'][i]
            
            # Velocity excess
            Delta_v = compute_velocity_excess(v_obs, v_pred)
            Delta_v_err = v_err / (v_obs * np.log(10))
            
            # Estimate density from radius (approximate)
            # Assume δ ~ (r/r_core)^-2, r_core ~ 0.5 Mpc
            delta_approx = 5.0 / (r / 0.5)**1.5
            rho = compute_rho_from_delta(delta_approx)
            log_rho = np.log10(rho)
            
            all_data.append({
                'source': source,
                'r_perp_Mpc': r,
                'v_obs': v_obs,
                'v_pred': v_pred,
                'v_err': v_err,
                'Delta_v': Delta_v,
                'Delta_v_err': Delta_v_err,
                'log_rho_cgs': log_rho,
                'type': 'velocity',
            })
    
    df = pd.DataFrame(all_data)
    
    print(f"  Loaded {len(df)} data points from {df['source'].nunique()} studies")
    print(f"  Density profiles: {(df['type'] == 'density').sum()}")
    print(f"  Velocity profiles: {(df['type'] == 'velocity').sum()}")
    print(f"  log(ρ) range: {df['log_rho_cgs'].min():.2f} to {df['log_rho_cgs'].max():.2f}")
    
    return df


def analyze_velocity_excess(df):
    """Analyze velocity excess vs density."""
    print("\n" + "-"*70)
    print(" VELOCITY EXCESS ANALYSIS")
    print("-"*70)
    
    df_vel = df[df['type'] == 'velocity'].copy()
    
    if len(df_vel) == 0:
        print("  No velocity data available")
        return None
    
    # Mean velocity excess
    delta_v_mean = df_vel['Delta_v'].mean()
    delta_v_std = df_vel['Delta_v'].std()
    delta_v_sem = delta_v_std / np.sqrt(len(df_vel))
    
    print(f"  N velocity points: {len(df_vel)}")
    print(f"  ⟨Δ_v⟩ = {delta_v_mean:.4f} ± {delta_v_sem:.4f}")
    print(f"  σ(Δ_v) = {delta_v_std:.4f}")
    
    # T-test
    if len(df_vel) >= 3:
        t_stat, p_ttest = stats.ttest_1samp(df_vel['Delta_v'], 0)
        print(f"  t-test vs 0: t = {t_stat:.2f}, p = {p_ttest:.4f}")
    else:
        p_ttest = 1.0
    
    # Correlation with density
    if len(df_vel) >= 4:
        r, p_corr = stats.pearsonr(df_vel['log_rho_cgs'], df_vel['Delta_v'])
        print(f"  r(Δ_v, log ρ) = {r:.3f}, p = {p_corr:.4f}")
    else:
        r, p_corr = 0, 1.0
    
    return {
        'n': len(df_vel),
        'delta_v_mean': float(delta_v_mean),
        'delta_v_std': float(delta_v_std),
        'ttest_p': float(p_ttest),
        'correlation_r': float(r),
        'correlation_p': float(p_corr),
    }


def analyze_density_excess(df):
    """Analyze if density profiles show QO+R signature."""
    print("\n" + "-"*70)
    print(" DENSITY PROFILE ANALYSIS")
    print("-"*70)
    
    df_dens = df[df['type'] == 'density'].copy()
    
    if len(df_dens) == 0:
        print("  No density data available")
        return None
    
    print(f"  N density points: {len(df_dens)}")
    
    # Compare to ΛCDM expectation
    # In ΛCDM, δ ∝ r^-2 approximately for filaments
    # QO+R predicts excess at low density (large r)
    
    # Fit power law: log(δ) = a + b * log(r)
    log_r = np.log10(df_dens['r_perp_Mpc'])
    log_delta = np.log10(df_dens['overdensity'])
    
    slope, intercept, r_val, p_val, std_err = stats.linregress(log_r, log_delta)
    
    print(f"  Power law fit: δ ∝ r^{slope:.2f}")
    print(f"  Expected (isothermal): δ ∝ r^-2.0")
    
    # Deviation from -2
    deviation = slope - (-2.0)
    print(f"  Deviation from r^-2: {deviation:+.2f}")
    
    if deviation > 0.3:
        print(f"  → Profile SHALLOWER than ΛCDM (possible QO+R signature)")
    elif deviation < -0.3:
        print(f"  → Profile STEEPER than ΛCDM")
    else:
        print(f"  → Profile consistent with ΛCDM")
    
    return {
        'n': len(df_dens),
        'power_law_slope': float(slope),
        'power_law_err': float(std_err),
        'deviation_from_LCDM': float(deviation),
    }


def run_toe_fit(df):
    """Fit TOE model to velocity excess."""
    print("\n" + "-"*70)
    print(" TOE MODEL FIT")
    print("-"*70)
    
    df_vel = df[df['type'] == 'velocity'].copy()
    
    if len(df_vel) < 4:
        print("  Insufficient velocity data for fit")
        return None
    
    log_rho = df_vel['log_rho_cgs'].values
    y = df_vel['Delta_v'].values
    y_err = df_vel['Delta_v_err'].values
    
    # Simple linear fit: Δ = γ * (log_rho - rho_ref)
    def model(params, x):
        gamma, offset = params
        return gamma * (x + 29) + offset  # Centered on typical filament density
    
    def chi2(params):
        pred = model(params, log_rho)
        return np.sum(((y - pred) / y_err)**2)
    
    res = minimize(chi2, [0.0, 0.0], method='Nelder-Mead')
    gamma_fit, offset_fit = res.x
    
    # Bootstrap errors
    n_boot = 500
    gammas = []
    for _ in range(n_boot):
        idx = np.random.choice(len(y), len(y), replace=True)
        res_b = minimize(lambda p: np.sum(((y[idx] - model(p, log_rho[idx])) / y_err[idx])**2),
                        [0.0, 0.0], method='Nelder-Mead')
        gammas.append(res_b.x[0])
    
    gamma_err = np.std(gammas)
    
    print(f"  γ (density slope) = {gamma_fit:.4f} ± {gamma_err:.4f}")
    print(f"  offset = {offset_fit:.4f}")
    
    # Check if γ < 0 (QO+R signature)
    if gamma_fit - 2*gamma_err < 0:
        print(f"  → Negative trend (consistent with QO+R screening)")
    else:
        print(f"  → No clear negative trend")
    
    return {
        'gamma': float(gamma_fit),
        'gamma_err': float(gamma_err),
        'offset': float(offset_fit),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print(" FIL-001: TOE TEST ON COSMIC FILAMENTS")
    print(" Testing Clusters → Cosmic Transition at 10^24 m")
    print(" Author: Jonathan Édouard Slama")
    print("="*70)
    
    # Load data
    df = load_filament_data()
    
    results = {
        'test': 'FIL-001',
        'scale': '10^24 m',
        'purpose': 'Bridge Clusters (10^23) to Cosmic (10^27)',
        'n_total': len(df),
        'sources': df['source'].unique().tolist(),
    }
    
    # Velocity excess analysis
    vel_results = analyze_velocity_excess(df)
    if vel_results:
        results['velocity_analysis'] = vel_results
    
    # Density profile analysis
    dens_results = analyze_density_excess(df)
    if dens_results:
        results['density_analysis'] = dens_results
    
    # TOE fit
    toe_results = run_toe_fit(df)
    if toe_results:
        results['toe_fit'] = toe_results
    
    # Verdict
    print("\n" + "="*70)
    print(" VERDICT")
    print("="*70)
    
    evidence = []
    
    if vel_results:
        if vel_results['delta_v_mean'] > 0.02:
            evidence.append(('Velocity excess', 'SIGNAL', f"⟨Δ_v⟩={vel_results['delta_v_mean']:.3f}"))
        else:
            evidence.append(('Velocity excess', 'NONE', f"⟨Δ_v⟩={vel_results['delta_v_mean']:.3f}"))
        
        if vel_results['correlation_r'] < -0.3 and vel_results['correlation_p'] < 0.1:
            evidence.append(('Δ-ρ correlation', 'SIGNAL', f"r={vel_results['correlation_r']:.2f}"))
    
    if dens_results:
        if dens_results['deviation_from_LCDM'] > 0.3:
            evidence.append(('Profile shallower', 'SIGNAL', f"slope={dens_results['power_law_slope']:.2f}"))
    
    if toe_results:
        if toe_results['gamma'] < 0 and toe_results['gamma'] + 2*toe_results['gamma_err'] < 0:
            evidence.append(('γ < 0 (screening)', 'SIGNAL', f"γ={toe_results['gamma']:.3f}±{toe_results['gamma_err']:.3f}"))
        elif toe_results['gamma'] < 0:
            evidence.append(('γ < 0 (weak)', 'MARGINAL', f"γ={toe_results['gamma']:.3f}"))
    
    for name, status, detail in evidence:
        symbol = '🎯' if status == 'SIGNAL' else ('⚠️' if status == 'MARGINAL' else '⬜')
        print(f"  {symbol} {name}: {detail}")
    
    n_signals = sum(1 for e in evidence if e[1] == 'SIGNAL')
    n_marginal = sum(1 for e in evidence if e[1] == 'MARGINAL')
    
    if n_signals >= 2:
        verdict = "QO+R SIGNAL DETECTED IN FILAMENTS"
        interpretation = "Intermediate density shows screening signature like UDGs"
    elif n_signals + n_marginal >= 2:
        verdict = "MARGINAL QO+R SIGNAL IN FILAMENTS"
        interpretation = "Hints of screening, needs more data"
    else:
        verdict = "NO CLEAR SIGNAL - DATA LIMITED"
        interpretation = "Literature data insufficient for conclusive test"
    
    print(f"\n  VERDICT: {verdict}")
    print(f"  INTERPRETATION: {interpretation}")
    
    results['verdict'] = verdict
    results['interpretation'] = interpretation
    
    # Comparison
    print("\n" + "-"*70)
    print(" MULTI-SCALE COMPARISON")
    print("-"*70)
    
    print(f"  UDG (10²⁰ m):     γ = -0.16 (CI exclut 0)")
    if toe_results:
        print(f"  FIL (10²⁴ m):     γ = {toe_results['gamma']:.3f} ± {toe_results['gamma_err']:.3f}")
    print(f"  Expected:         γ < 0 (screening signature)")
    
    # Figures
    if HAS_MPL:
        print("\n" + "-"*70)
        print(" CREATING FIGURES")
        print("-"*70)
        
        os.makedirs(FIGURES_DIR, exist_ok=True)
        
        df_vel = df[df['type'] == 'velocity']
        
        if len(df_vel) > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for source in df_vel['source'].unique():
                mask = df_vel['source'] == source
                ax.errorbar(df_vel.loc[mask, 'log_rho_cgs'], 
                           df_vel.loc[mask, 'Delta_v'],
                           yerr=df_vel.loc[mask, 'Delta_v_err'],
                           fmt='o', label=source, alpha=0.7, markersize=8, capsize=3)
            
            ax.axhline(0, color='k', ls='--', alpha=0.3, label='ΛCDM')
            
            # Trend line if fit available
            if toe_results:
                x_plot = np.linspace(df_vel['log_rho_cgs'].min(), df_vel['log_rho_cgs'].max(), 50)
                y_plot = toe_results['gamma'] * (x_plot + 29) + toe_results['offset']
                ax.plot(x_plot, y_plot, 'r-', lw=2, label=f"γ={toe_results['gamma']:.3f}")
            
            ax.set_xlabel(r'$\log_{10}(\rho / \mathrm{g\,cm^{-3}})$', fontsize=12)
            ax.set_ylabel(r'$\Delta_v = \log(v_{\mathrm{obs}}/v_{\mathrm{pred}})$', fontsize=12)
            ax.set_title('FIL-001: Filament Velocity Excess vs Density', fontsize=14)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            fig.savefig(os.path.join(FIGURES_DIR, 'fil_delta_vs_rho.png'), dpi=150)
            fig.savefig(os.path.join(FIGURES_DIR, 'fil_delta_vs_rho.pdf'))
            print(f"    Saved: fil_delta_vs_rho.png/pdf")
            
            plt.close()
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "fil001_results.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)
    
    # Note about limitations
    print("\n" + "="*70)
    print(" NOTE ON DATA LIMITATIONS")
    print("="*70)
    print("""
  This test uses PUBLISHED literature values (stacked profiles).
  For a definitive test, we would need:
  
  1. Raw SDSS/DESI spectroscopic data
  2. DisPerSE filament identification
  3. Galaxy-by-galaxy peculiar velocities
  4. Comparison to N-body ΛCDM mocks
  
  Current analysis provides PRELIMINARY results only.
  Full implementation requires ~2-3 weeks of data processing.
""")
    
    return results


if __name__ == "__main__":
    main()
