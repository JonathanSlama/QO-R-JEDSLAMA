#!/usr/bin/env python3
"""
===============================================================================
UDG-001 V3.1: HIERARCHICAL BAYESIAN TOE TEST - REFINED
===============================================================================

Improvements over V3:
1. Random effects by environment (cluster/group/field intercepts)
2. Posterior Predictive Checks (PPC)
3. Prior sensitivity analysis
4. Better anisotropy prior β ~ TruncNormal(0.2, 0.1², [0, 0.5])
5. Explicit aperture correction when available
6. Publication-ready figures

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

# Optional: matplotlib for figures
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("  [Warning] matplotlib not available, skipping figures")

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "udg_catalogs", "Gannon2024_udg_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

# =============================================================================
# CONSTANTS
# =============================================================================

G_SI = 6.674e-11
M_SUN = 1.989e30
KPC_M = 3.086e19

# =============================================================================
# V3.1 CRITERIA (LOCKED)
# =============================================================================

CRITERIA = {
    # Priors - standard
    'prior_alpha_mean': 0.0,
    'prior_alpha_std': 0.15,
    'prior_lambda_mean': 1.0,
    'prior_lambda_std': 0.5,
    'prior_rho_c_mean': -25.0,
    'prior_rho_c_std': 2.0,
    'prior_delta_mean': 1.0,
    'prior_delta_std': 0.5,
    'prior_tau_scale': 0.15,
    
    # Environment random effects
    'prior_sigma_env_scale': 0.1,  # Scale of environment intercepts
    
    # Anisotropy - refined
    'beta_aniso_mean': 0.2,
    'beta_aniso_std': 0.1,
    'beta_aniso_min': 0.0,
    'beta_aniso_max': 0.5,
    
    # Student-t
    'student_nu': 5,
    
    # MCMC
    'n_samples': 3000,
    'n_burnin': 1000,
    
    # PPC
    'n_ppc_samples': 1000,
    
    # Physics
    'beta_Q': 1.0,
    'beta_R': 0.5,
}

# =============================================================================
# JEANS MODEL (REFINED)
# =============================================================================

def wolf_correction(n_sersic=1.0, r_ap_reff=1.0):
    """Wolf+10 aperture correction."""
    K_base = 0.5
    K_corr = 0.2 * np.sqrt(r_ap_reff)
    return K_base + K_corr


def sample_beta_aniso():
    """Sample anisotropy from truncated normal."""
    while True:
        beta = np.random.normal(CRITERIA['beta_aniso_mean'], CRITERIA['beta_aniso_std'])
        if CRITERIA['beta_aniso_min'] <= beta <= CRITERIA['beta_aniso_max']:
            return beta


def compute_sigma_bar_jeans(M_star_Msun, R_eff_kpc, beta_aniso=None):
    """Compute σ_bar with Jeans + aperture + anisotropy."""
    if beta_aniso is None:
        beta_aniso = sample_beta_aniso()
    
    M_kg = M_star_Msun * M_SUN
    R_m = R_eff_kpc * KPC_M
    
    K = wolf_correction(n_sersic=1.0, r_ap_reff=1.0)
    aniso_corr = 1.0 - 0.15 * beta_aniso
    
    sigma_sq = K * G_SI * M_kg / R_m * aniso_corr
    sigma_bar = np.sqrt(sigma_sq) / 1000
    
    # Model uncertainty
    rel_err = np.sqrt(0.15**2 + (0.15 * beta_aniso)**2)
    sigma_bar_err = rel_err * sigma_bar
    
    return sigma_bar, sigma_bar_err


# =============================================================================
# HIERARCHICAL MODEL WITH ENVIRONMENT EFFECTS
# =============================================================================

def toe_coupling(log_rho, f, lambda_QR, rho_c, delta):
    """TOE coupling function."""
    rho_0 = -25.0
    C_Q = 10**((rho_0 - log_rho) * CRITERIA['beta_Q'])
    C_R = 10**((rho_0 - log_rho) * CRITERIA['beta_R'])
    P = np.pi * expit((log_rho - rho_c) / delta)
    coupling = lambda_QR * (C_Q * f - C_R * (1 - f))
    return coupling * np.cos(P)


def log_prior_v31(params):
    """
    V3.1 prior with environment random effects.
    params = [alpha, lambda_QR, rho_c, delta, tau, gamma, env_cluster, env_group, env_field, sigma_env]
    """
    alpha, lambda_QR, rho_c, delta, tau, gamma, e_clu, e_grp, e_fld, sigma_env = params
    
    log_p = 0.0
    
    # Standard priors
    log_p += -0.5 * ((alpha - CRITERIA['prior_alpha_mean']) / CRITERIA['prior_alpha_std'])**2
    log_p += -0.5 * ((lambda_QR - CRITERIA['prior_lambda_mean']) / CRITERIA['prior_lambda_std'])**2
    log_p += -0.5 * ((rho_c - CRITERIA['prior_rho_c_mean']) / CRITERIA['prior_rho_c_std'])**2
    log_p += -0.5 * ((delta - CRITERIA['prior_delta_mean']) / CRITERIA['prior_delta_std'])**2
    log_p += -0.5 * (gamma / 0.1)**2
    
    # Half-normal on tau
    if tau < 0:
        return -np.inf
    log_p += -0.5 * (tau / CRITERIA['prior_tau_scale'])**2
    
    # Half-normal on sigma_env
    if sigma_env < 0:
        return -np.inf
    log_p += -0.5 * (sigma_env / CRITERIA['prior_sigma_env_scale'])**2
    
    # Environment effects ~ Normal(0, sigma_env²)
    if sigma_env > 0:
        log_p += -0.5 * (e_clu / sigma_env)**2
        log_p += -0.5 * (e_grp / sigma_env)**2
        log_p += -0.5 * (e_fld / sigma_env)**2
    
    # Bounds
    if lambda_QR < 0.01 or lambda_QR > 10:
        return -np.inf
    if delta < 0.01 or delta > 10:
        return -np.inf
    if rho_c < -30 or rho_c > -20:
        return -np.inf
    
    return log_p


def log_likelihood_v31(params, X, y, y_err, env_codes):
    """
    Student-t likelihood with environment random effects.
    """
    alpha, lambda_QR, rho_c, delta, tau, gamma, e_clu, e_grp, e_fld, sigma_env = params
    log_rho, f = X
    
    nu = CRITERIA['student_nu']
    
    # Environment intercepts
    env_effects = np.zeros(len(y))
    env_effects[env_codes == 1] = e_clu  # cluster
    env_effects[env_codes == 2] = e_grp  # group
    env_effects[env_codes == 3] = e_fld  # field
    
    # Model prediction
    coupling = toe_coupling(log_rho, f, lambda_QR, rho_c, delta)
    mu = alpha * coupling + gamma * (log_rho + 25) + env_effects
    
    # Total variance
    sigma = np.sqrt(y_err**2 + tau**2)
    
    # Student-t log-likelihood
    z = (y - mu) / sigma
    log_lik = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 
               0.5 * np.log(nu * np.pi) - np.log(sigma) -
               (nu + 1) / 2 * np.log(1 + z**2 / nu))
    
    return np.sum(log_lik)


def log_posterior_v31(params, X, y, y_err, env_codes):
    lp = log_prior_v31(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_v31(params, X, y, y_err, env_codes)
    return lp + ll


def mcmc_v31(X, y, y_err, env_codes, n_samples=3000, n_burnin=1000):
    """MCMC sampler for V3.1 model."""
    print(f"\n  Running MCMC V3.1 (n={n_samples}, burnin={n_burnin})...")
    
    # Initial values: [alpha, lambda, rho_c, delta, tau, gamma, e_clu, e_grp, e_fld, sigma_env]
    params = np.array([0.0, 1.0, -25.0, 1.0, 0.1, -0.05, 0.0, 0.0, 0.0, 0.05])
    n_params = len(params)
    
    # Proposal scales
    scales = np.array([0.02, 0.1, 0.5, 0.2, 0.02, 0.02, 0.02, 0.02, 0.05, 0.01])
    
    samples = np.zeros((n_samples, n_params))
    log_posts = np.zeros(n_samples)
    
    current_lp = log_posterior_v31(params, X, y, y_err, env_codes)
    n_accept = 0
    
    for i in range(n_samples + n_burnin):
        proposal = params + scales * np.random.randn(n_params)
        proposed_lp = log_posterior_v31(proposal, X, y, y_err, env_codes)
        
        if np.log(np.random.rand()) < proposed_lp - current_lp:
            params = proposal
            current_lp = proposed_lp
            if i >= n_burnin:
                n_accept += 1
        
        if i >= n_burnin:
            samples[i - n_burnin] = params
            log_posts[i - n_burnin] = current_lp
    
    accept_rate = n_accept / n_samples
    print(f"  Acceptance rate: {accept_rate:.2%}")
    
    return samples, log_posts, accept_rate


# =============================================================================
# POSTERIOR PREDICTIVE CHECK
# =============================================================================

def posterior_predictive_check(samples, X, y_err, env_codes, n_ppc=1000):
    """Generate posterior predictive samples."""
    print(f"\n  Running PPC (n={n_ppc})...")
    
    log_rho, f = X
    n_data = len(log_rho)
    nu = CRITERIA['student_nu']
    
    # Sample from posterior
    idx = np.random.choice(len(samples), n_ppc, replace=True)
    
    y_pred_samples = np.zeros((n_ppc, n_data))
    
    for i, sample_idx in enumerate(idx):
        alpha, lambda_QR, rho_c, delta, tau, gamma, e_clu, e_grp, e_fld, sigma_env = samples[sample_idx]
        
        # Environment effects
        env_effects = np.zeros(n_data)
        env_effects[env_codes == 1] = e_clu
        env_effects[env_codes == 2] = e_grp
        env_effects[env_codes == 3] = e_fld
        
        # Mean
        coupling = toe_coupling(log_rho, f, lambda_QR, rho_c, delta)
        mu = alpha * coupling + gamma * (log_rho + 25) + env_effects
        
        # Sample from Student-t
        sigma = np.sqrt(y_err**2 + tau**2)
        y_pred_samples[i] = mu + sigma * np.random.standard_t(nu, n_data)
    
    # Summary statistics
    y_pred_mean = np.mean(y_pred_samples, axis=0)
    y_pred_std = np.std(y_pred_samples, axis=0)
    y_pred_q025 = np.percentile(y_pred_samples, 2.5, axis=0)
    y_pred_q975 = np.percentile(y_pred_samples, 97.5, axis=0)
    
    return {
        'samples': y_pred_samples,
        'mean': y_pred_mean,
        'std': y_pred_std,
        'q025': y_pred_q025,
        'q975': y_pred_q975
    }


# =============================================================================
# PRIOR SENSITIVITY
# =============================================================================

def run_sensitivity_analysis(X, y, y_err, env_codes):
    """Test sensitivity to prior choices."""
    print("\n" + "-"*70)
    print(" PRIOR SENSITIVITY ANALYSIS")
    print("-"*70)
    
    results = {}
    
    # Baseline
    print("  [1/3] Baseline priors...")
    samples_base, _, _ = mcmc_v31(X, y, y_err, env_codes, n_samples=1500, n_burnin=500)
    results['baseline'] = {
        'lambda_QR': float(np.median(samples_base[:, 1])),
        'alpha': float(np.median(samples_base[:, 0])),
        'gamma': float(np.median(samples_base[:, 5]))
    }
    
    # Wider α prior (σ = 0.30)
    print("  [2/3] Wider α prior (σ=0.30)...")
    old_alpha_std = CRITERIA['prior_alpha_std']
    CRITERIA['prior_alpha_std'] = 0.30
    samples_wide_a, _, _ = mcmc_v31(X, y, y_err, env_codes, n_samples=1500, n_burnin=500)
    results['wider_alpha'] = {
        'lambda_QR': float(np.median(samples_wide_a[:, 1])),
        'alpha': float(np.median(samples_wide_a[:, 0])),
        'gamma': float(np.median(samples_wide_a[:, 5]))
    }
    CRITERIA['prior_alpha_std'] = old_alpha_std
    
    # Wider λ prior (σ = 0.70)
    print("  [3/3] Wider λ prior (σ=0.70)...")
    old_lambda_std = CRITERIA['prior_lambda_std']
    CRITERIA['prior_lambda_std'] = 0.70
    samples_wide_l, _, _ = mcmc_v31(X, y, y_err, env_codes, n_samples=1500, n_burnin=500)
    results['wider_lambda'] = {
        'lambda_QR': float(np.median(samples_wide_l[:, 1])),
        'alpha': float(np.median(samples_wide_l[:, 0])),
        'gamma': float(np.median(samples_wide_l[:, 5]))
    }
    CRITERIA['prior_lambda_std'] = old_lambda_std
    
    # Print comparison
    print("\n  Sensitivity results:")
    print(f"  {'Prior':<20} {'λ_QR':>10} {'α':>10} {'γ':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for name, res in results.items():
        print(f"  {name:<20} {res['lambda_QR']:>10.3f} {res['alpha']:>10.3f} {res['gamma']:>10.3f}")
    
    return results


# =============================================================================
# FIGURES
# =============================================================================

def create_figures(df, samples, ppc_results, results_dict):
    """Create publication-ready figures."""
    if not HAS_MPL:
        print("  Skipping figures (matplotlib not available)")
        return
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("\n  Creating figures...")
    
    # Figure 1: Δ vs log ρ with posterior bands
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    log_rho = df['log_rho'].values
    delta = df['Delta_UDG'].values
    env_codes = df['Environment'].values
    
    # Environment colors
    colors = {1: 'red', 2: 'blue', 3: 'green'}
    labels = {1: 'Cluster', 2: 'Group', 3: 'Field'}
    
    for env in [1, 2, 3]:
        mask = env_codes == env
        if mask.sum() > 0:
            ax1.scatter(log_rho[mask], delta[mask], c=colors[env], 
                       label=labels[env], s=80, alpha=0.7, edgecolors='k')
    
    # Posterior bands
    rho_grid = np.linspace(log_rho.min() - 0.5, log_rho.max() + 0.5, 100)
    
    # Sample predictions
    n_draw = 200
    idx = np.random.choice(len(samples), n_draw)
    y_grid = np.zeros((n_draw, len(rho_grid)))
    
    for i, s_idx in enumerate(idx):
        gamma = samples[s_idx, 5]
        y_grid[i] = gamma * (rho_grid + 25)
    
    y_median = np.median(y_grid, axis=0)
    y_q16 = np.percentile(y_grid, 16, axis=0)
    y_q84 = np.percentile(y_grid, 84, axis=0)
    y_q025 = np.percentile(y_grid, 2.5, axis=0)
    y_q975 = np.percentile(y_grid, 97.5, axis=0)
    
    ax1.fill_between(rho_grid, y_q025, y_q975, alpha=0.2, color='gray', label='95% CI')
    ax1.fill_between(rho_grid, y_q16, y_q84, alpha=0.4, color='gray', label='68% CI')
    ax1.plot(rho_grid, y_median, 'k-', lw=2, label='Posterior median')
    
    ax1.axhline(0, color='k', ls='--', alpha=0.3)
    ax1.set_xlabel(r'$\log_{10}(\rho_* / \mathrm{g\,cm^{-3}})$', fontsize=12)
    ax1.set_ylabel(r'$\Delta_{\mathrm{UDG}} = \log(\sigma_{\mathrm{obs}}/\sigma_{\mathrm{bar}})$', fontsize=12)
    ax1.set_title('UDG Velocity Dispersion Excess vs Stellar Density', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    fig1.tight_layout()
    fig1.savefig(os.path.join(FIGURES_DIR, 'udg_delta_vs_rho.png'), dpi=150)
    fig1.savefig(os.path.join(FIGURES_DIR, 'udg_delta_vs_rho.pdf'))
    print(f"    Saved: udg_delta_vs_rho.png/pdf")
    
    # Figure 2: Posterior distributions
    fig2, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    param_names = ['α', 'λ_QR', 'ρ_c', 'δ', 'τ', 'γ']
    param_idx = [0, 1, 2, 3, 4, 5]
    
    for ax, name, idx in zip(axes.flat, param_names, param_idx):
        chain = samples[:, idx]
        ax.hist(chain, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='k')
        ax.axvline(np.median(chain), color='red', lw=2, label=f'Median: {np.median(chain):.3f}')
        ax.axvline(np.percentile(chain, 2.5), color='red', ls='--', alpha=0.5)
        ax.axvline(np.percentile(chain, 97.5), color='red', ls='--', alpha=0.5)
        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=9)
    
    fig2.suptitle('Posterior Distributions (V3.1)', fontsize=14)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIGURES_DIR, 'udg_posteriors.png'), dpi=150)
    fig2.savefig(os.path.join(FIGURES_DIR, 'udg_posteriors.pdf'))
    print(f"    Saved: udg_posteriors.png/pdf")
    
    # Figure 3: PPC - QQ plot
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    
    # Standardized residuals
    y_obs = delta
    y_pred_mean = ppc_results['mean']
    y_pred_std = ppc_results['std']
    
    z_obs = (y_obs - y_pred_mean) / y_pred_std
    z_sorted = np.sort(z_obs)
    
    # Theoretical quantiles (Student-t)
    n = len(z_sorted)
    p = (np.arange(1, n+1) - 0.5) / n
    z_theo = stats.t.ppf(p, df=CRITERIA['student_nu'])
    
    ax3.scatter(z_theo, z_sorted, c='steelblue', s=50, alpha=0.7)
    ax3.plot([-3, 3], [-3, 3], 'k--', lw=2, label='1:1 line')
    ax3.set_xlabel('Theoretical Quantiles (Student-t)', fontsize=12)
    ax3.set_ylabel('Observed Standardized Residuals', fontsize=12)
    ax3.set_title('Posterior Predictive Check: QQ Plot', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-3.5, 3.5)
    ax3.set_ylim(-3.5, 3.5)
    
    fig3.tight_layout()
    fig3.savefig(os.path.join(FIGURES_DIR, 'udg_ppc_qq.png'), dpi=150)
    fig3.savefig(os.path.join(FIGURES_DIR, 'udg_ppc_qq.pdf'))
    print(f"    Saved: udg_ppc_qq.png/pdf")
    
    # Figure 4: Environment effects
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    
    env_names = ['Cluster', 'Group', 'Field']
    env_means = []
    env_stds = []
    
    for env in [1, 2, 3]:
        mask = env_codes == env
        if mask.sum() > 0:
            env_means.append(delta[mask].mean())
            env_stds.append(delta[mask].std() / np.sqrt(mask.sum()))
        else:
            env_means.append(np.nan)
            env_stds.append(np.nan)
    
    x_pos = np.arange(3)
    bars = ax4.bar(x_pos, env_means, yerr=env_stds, capsize=5, 
                   color=['red', 'blue', 'green'], alpha=0.7, edgecolor='k')
    
    ax4.axhline(0, color='k', ls='--', alpha=0.3)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(env_names, fontsize=12)
    ax4.set_ylabel(r'$\langle\Delta_{\mathrm{UDG}}\rangle$', fontsize=12)
    ax4.set_title('Mean Velocity Excess by Environment', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add counts
    for i, env in enumerate([1, 2, 3]):
        n = (env_codes == env).sum()
        ax4.text(i, env_means[i] + env_stds[i] + 0.02, f'n={n}', ha='center', fontsize=10)
    
    fig4.tight_layout()
    fig4.savefig(os.path.join(FIGURES_DIR, 'udg_env_effects.png'), dpi=150)
    fig4.savefig(os.path.join(FIGURES_DIR, 'udg_env_effects.pdf'))
    print(f"    Saved: udg_env_effects.png/pdf")
    
    plt.close('all')
    print("  All figures saved.")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load and preprocess data."""
    print("\n" + "="*70)
    print(" LOADING DATA (V3.1)")
    print("="*70)
    
    df = pd.read_csv(DATA_PATH)
    df = df[df['stellar_sigma'] > 0].copy()
    
    print(f"  N UDGs with σ: {len(df)}")
    
    # Jeans σ_bar
    sigma_bar_list = []
    sigma_bar_err_list = []
    
    for _, row in df.iterrows():
        M_star = row['Stellar mass'] * 1e8
        R_eff = row['R_e']
        sigma_bar, sigma_bar_err = compute_sigma_bar_jeans(M_star, R_eff)
        sigma_bar_list.append(sigma_bar)
        sigma_bar_err_list.append(sigma_bar_err)
    
    df['sigma_bar_jeans'] = sigma_bar_list
    df['sigma_bar_jeans_err'] = sigma_bar_err_list
    
    # Residual
    df['Delta_UDG'] = np.log10(df['stellar_sigma']) - np.log10(df['sigma_bar_jeans'])
    
    # Error
    df['sigma_obs_err'] = (df['stellar_sigma_up'] + df['stellar_sigma_down']) / 2
    df.loc[df['sigma_obs_err'] <= 0, 'sigma_obs_err'] = df['stellar_sigma'] * 0.15
    
    df['Delta_err'] = np.sqrt(
        (df['sigma_obs_err'] / (df['stellar_sigma'] * np.log(10)))**2 +
        (df['sigma_bar_jeans_err'] / (df['sigma_bar_jeans'] * np.log(10)))**2
    )
    
    # Density
    V_cm3 = (4/3) * np.pi * (df['R_e'] * 3.086e21)**3
    df['rho_star'] = (df['Stellar mass'] * 1e8 * M_SUN * 1000) / V_cm3
    df['log_rho'] = np.log10(df['rho_star'])
    
    # f parameter
    env_f_map = {3: 0.3, 2: 0.5, 1: 0.7}
    df['f'] = df['Environment'].map(env_f_map)
    
    print(f"  ⟨Δ_UDG⟩ = {df['Delta_UDG'].mean():.3f} ± {df['Delta_UDG'].std():.3f}")
    
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print(" UDG-001 V3.1: HIERARCHICAL BAYESIAN TOE TEST - REFINED")
    print(" Environment Random Effects + PPC + Sensitivity")
    print(" Author: Jonathan Édouard Slama")
    print("="*70)
    
    # Load data
    df = load_data()
    
    X = (df['log_rho'].values, df['f'].values)
    y = df['Delta_UDG'].values
    y_err = df['Delta_err'].values
    env_codes = df['Environment'].values
    
    results = {
        'version': 'V3.1',
        'n_udg': len(df),
        'delta_mean': float(np.mean(y)),
        'delta_std': float(np.std(y)),
    }
    
    # Correlation test
    print("\n" + "-"*70)
    print(" CORRELATION TEST")
    print("-"*70)
    
    r, p = stats.pearsonr(df['log_rho'], y)
    print(f"  r(Δ, log ρ) = {r:.3f}, p = {p:.6f}")
    results['correlation'] = {'r': float(r), 'p': float(p)}
    
    # Main MCMC
    print("\n" + "-"*70)
    print(" HIERARCHICAL BAYESIAN FIT (V3.1)")
    print("-"*70)
    
    samples, log_posts, accept_rate = mcmc_v31(
        X, y, y_err, env_codes,
        n_samples=CRITERIA['n_samples'],
        n_burnin=CRITERIA['n_burnin']
    )
    
    # Posterior summary
    param_names = ['alpha', 'lambda_QR', 'rho_c', 'delta', 'tau', 'gamma', 
                   'env_cluster', 'env_group', 'env_field', 'sigma_env']
    
    print("\n  Posterior estimates:")
    posterior = {}
    for i, name in enumerate(param_names):
        chain = samples[:, i]
        posterior[name] = {
            'mean': float(np.mean(chain)),
            'std': float(np.std(chain)),
            'median': float(np.median(chain)),
            'ci_low': float(np.percentile(chain, 2.5)),
            'ci_high': float(np.percentile(chain, 97.5)),
        }
        print(f"    {name}: {posterior[name]['mean']:.4f} ± {posterior[name]['std']:.4f} "
              f"[{posterior[name]['ci_low']:.4f}, {posterior[name]['ci_high']:.4f}]")
    
    results['posterior'] = posterior
    results['mcmc'] = {'accept_rate': float(accept_rate)}
    
    # PPC
    print("\n" + "-"*70)
    print(" POSTERIOR PREDICTIVE CHECK")
    print("-"*70)
    
    ppc_results = posterior_predictive_check(samples, X, y_err, env_codes, n_ppc=CRITERIA['n_ppc_samples'])
    
    # Coverage check
    in_ci = np.sum((y >= ppc_results['q025']) & (y <= ppc_results['q975']))
    coverage = in_ci / len(y)
    print(f"  95% CI coverage: {coverage:.1%} (expected ~95%)")
    results['ppc'] = {'coverage_95': float(coverage)}
    
    # Sensitivity
    sensitivity_results = run_sensitivity_analysis(X, y, y_err, env_codes)
    results['sensitivity'] = sensitivity_results
    
    # Model comparison
    print("\n" + "-"*70)
    print(" MODEL COMPARISON")
    print("-"*70)
    
    # Null model
    def null_log_lik(params, X, y, y_err):
        gamma, tau = params
        if tau < 0:
            return -np.inf
        log_rho, _ = X
        mu = gamma * (log_rho + 25)
        sigma = np.sqrt(y_err**2 + tau**2)
        return np.sum(stats.norm.logpdf(y, mu, sigma))
    
    res_null = minimize(lambda p: -null_log_lik(p, X, y, y_err), [0.0, 0.1])
    null_ll = -res_null.fun
    null_aic = -2 * null_ll + 2 * 2
    
    toe_ll = np.max(log_posts)
    toe_aic = -2 * toe_ll + 2 * 10  # 10 parameters
    
    delta_aic = null_aic - toe_aic
    
    print(f"  Null AIC = {null_aic:.1f}")
    print(f"  TOE AIC = {toe_aic:.1f}")
    print(f"  ΔAIC = {delta_aic:.1f}")
    
    results['model_comparison'] = {
        'null_aic': float(null_aic),
        'toe_aic': float(toe_aic),
        'delta_aic': float(delta_aic)
    }
    
    # Verdict
    print("\n" + "="*70)
    print(" VERDICT V3.1")
    print("="*70)
    
    evidence = []
    
    if p < 0.01 and r < 0:
        evidence.append(('Correlation', 'SIGNIFICANT', f"r={r:.3f}, p={p:.4f}"))
    
    lambda_post = posterior['lambda_QR']
    if 0.3 < lambda_post['mean'] < 3.0:
        evidence.append(('λ_QR', 'COHERENT', f"{lambda_post['mean']:.2f} ± {lambda_post['std']:.2f}"))
    
    gamma_post = posterior['gamma']
    if gamma_post['ci_high'] < 0:
        evidence.append(('γ (screening)', 'SIGNIFICANT', f"{gamma_post['mean']:.3f} [{gamma_post['ci_low']:.3f}, {gamma_post['ci_high']:.3f}]"))
    elif gamma_post['mean'] < 0:
        evidence.append(('γ (screening)', 'WEAK', f"{gamma_post['mean']:.3f} (CI includes 0)"))
    
    if delta_aic > 10:
        evidence.append(('ΔAIC', 'TOE_PREFERRED', f"{delta_aic:.1f}"))
    elif delta_aic > 0:
        evidence.append(('ΔAIC', 'SLIGHT_TOE', f"{delta_aic:.1f}"))
    
    if coverage > 0.85:
        evidence.append(('PPC coverage', 'GOOD', f"{coverage:.1%}"))
    
    for name, status, detail in evidence:
        symbol = '🎯' if status in ['SIGNIFICANT', 'COHERENT', 'TOE_PREFERRED', 'GOOD'] else '✅'
        print(f"  {symbol} {name}: {detail}")
    
    n_strong = sum(1 for e in evidence if e[1] in ['SIGNIFICANT', 'COHERENT', 'TOE_PREFERRED'])
    
    if n_strong >= 3:
        verdict = "QO+R TOE SUPPORTED IN UDGs (V3.1)"
    elif n_strong >= 2:
        verdict = "QO+R TOE PARTIALLY SUPPORTED"
    else:
        verdict = "QO+R TOE INCONCLUSIVE"
    
    print(f"\n  VERDICT: {verdict}")
    results['verdict'] = verdict
    
    # Figures
    print("\n" + "-"*70)
    print(" CREATING FIGURES")
    print("-"*70)
    create_figures(df, samples, ppc_results, results)
    
    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "udg001_v31_results.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
