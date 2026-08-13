#!/usr/bin/env python3
"""
===============================================================================
GC-001: TOE TEST ON GLOBULAR CLUSTERS
Testing the WB → Galactic Transition at 10^18 m
===============================================================================

Purpose:
- Test QO+R screening at very HIGH stellar density
- Bridge the gap between Wide Binaries (10^13-16 m) and UDGs (10^20 m)
- Expect: Strong screening → Δ ≈ 0, λ_QR ~ 1.7 (like WB)

Data: Baumgardt+19 dynamical masses + Harris 2010 catalog

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

G_SI = 6.674e-11      # m³/kg/s²
M_SUN = 1.989e30      # kg
PC_M = 3.086e16       # m per pc

# =============================================================================
# PRE-REGISTERED CRITERIA (LOCKED)
# =============================================================================

CRITERIA = {
    'min_n_gc': 50,
    'max_sigma_err_frac': 0.20,
    
    # Priors from multi-scale (high density regime)
    'prior_alpha_mean': 0.0,
    'prior_alpha_std': 0.10,      # Expect small
    'prior_lambda_mean': 1.7,     # Centered on WB value
    'prior_lambda_std': 0.4,
    'prior_rho_c_mean': -22.0,    # Higher than galactic (denser)
    'prior_rho_c_std': 2.0,
    
    # Physics
    'beta_Q': 1.0,
    'beta_R': 0.5,
    
    # Expected: screening → Δ ≈ 0
    'expected_delta_mean': 0.0,
    'expected_delta_std': 0.05,
    
    # Student-t
    'student_nu': 5,
    
    # MCMC
    'n_samples': 2000,
    'n_burnin': 500,
}

# =============================================================================
# BAUMGARDT+19 GLOBULAR CLUSTER DATA
# =============================================================================

# Key GCs from Baumgardt & Hilker 2018, Baumgardt+2019
# σ_0 = central velocity dispersion (km/s)
# M_dyn = dynamical mass (10^5 M_sun)
# r_h = half-light radius (pc)

BAUMGARDT_DATA = {
    # Name: (σ_0, σ_err, M_dyn, M_err, r_h, r_h_err, [Fe/H])
    'NGC 104':    (11.0, 0.3, 8.90, 0.40, 4.15, 0.10, -0.72),   # 47 Tuc
    'NGC 288':    (2.9, 0.3, 0.89, 0.10, 5.77, 0.20, -1.32),
    'NGC 362':    (6.4, 0.3, 2.80, 0.20, 1.94, 0.10, -1.26),
    'NGC 1261':   (4.8, 0.5, 1.70, 0.20, 2.52, 0.15, -1.27),
    'NGC 1851':   (10.4, 0.4, 3.20, 0.25, 1.09, 0.05, -1.18),
    'NGC 1904':   (5.3, 0.4, 1.40, 0.15, 1.72, 0.10, -1.60),
    'NGC 2298':   (3.1, 0.4, 0.45, 0.08, 1.42, 0.10, -1.92),
    'NGC 2808':   (13.4, 0.4, 9.60, 0.50, 2.23, 0.10, -1.14),
    'NGC 3201':   (5.0, 0.2, 1.50, 0.10, 4.32, 0.15, -1.59),
    'NGC 4147':   (2.6, 0.4, 0.39, 0.06, 1.41, 0.10, -1.80),
    'NGC 4590':   (2.6, 0.2, 1.10, 0.10, 4.40, 0.20, -2.23),
    'NGC 4833':   (4.3, 0.3, 2.20, 0.20, 4.32, 0.15, -1.85),
    'NGC 5024':   (4.2, 0.3, 4.10, 0.30, 5.64, 0.20, -2.10),
    'NGC 5053':   (1.4, 0.3, 0.72, 0.10, 6.85, 0.30, -2.27),
    'NGC 5139':   (16.8, 0.3, 36.0, 2.0, 6.44, 0.20, -1.53),   # Omega Cen
    'NGC 5272':   (5.5, 0.2, 4.50, 0.25, 3.38, 0.10, -1.50),   # M3
    'NGC 5286':   (8.1, 0.4, 3.80, 0.30, 2.33, 0.10, -1.69),
    'NGC 5466':   (1.5, 0.2, 0.93, 0.10, 6.76, 0.30, -1.98),
    'NGC 5694':   (5.8, 0.6, 2.10, 0.25, 1.62, 0.10, -1.98),
    'NGC 5824':   (10.7, 0.5, 5.00, 0.40, 2.02, 0.10, -1.91),
    'NGC 5897':   (2.8, 0.3, 1.50, 0.15, 5.17, 0.20, -1.90),
    'NGC 5904':   (5.5, 0.2, 3.90, 0.20, 3.36, 0.10, -1.29),   # M5
    'NGC 5927':   (5.1, 0.4, 2.20, 0.20, 2.13, 0.10, -0.49),
    'NGC 5986':   (6.4, 0.4, 3.00, 0.25, 2.52, 0.10, -1.59),
    'NGC 6093':   (12.5, 0.5, 3.00, 0.25, 0.96, 0.05, -1.75),  # M80
    'NGC 6101':   (2.9, 0.4, 1.00, 0.12, 3.61, 0.15, -1.98),
    'NGC 6121':   (4.0, 0.2, 0.87, 0.08, 3.65, 0.15, -1.16),   # M4
    'NGC 6139':   (9.5, 0.6, 3.00, 0.30, 1.43, 0.10, -1.65),
    'NGC 6144':   (2.2, 0.3, 0.75, 0.10, 3.37, 0.15, -1.76),
    'NGC 6171':   (4.1, 0.3, 0.79, 0.08, 2.32, 0.10, -1.02),   # M107
    'NGC 6205':   (7.1, 0.2, 4.60, 0.25, 2.54, 0.10, -1.53),   # M13
    'NGC 6218':   (4.0, 0.2, 1.00, 0.08, 2.48, 0.10, -1.37),   # M12
    'NGC 6254':   (6.6, 0.3, 1.50, 0.12, 2.11, 0.10, -1.56),   # M10
    'NGC 6256':   (7.2, 0.7, 1.40, 0.20, 1.10, 0.10, -1.02),
    'NGC 6266':   (14.3, 0.4, 7.80, 0.50, 1.51, 0.08, -1.18),  # M62
    'NGC 6273':   (9.6, 0.4, 6.50, 0.40, 2.58, 0.10, -1.74),   # M19
    'NGC 6284':   (7.1, 0.5, 1.70, 0.20, 1.24, 0.08, -1.26),
    'NGC 6287':   (5.7, 0.5, 1.10, 0.15, 1.38, 0.10, -2.10),
    'NGC 6293':   (8.0, 0.5, 1.80, 0.20, 1.09, 0.08, -1.99),
    'NGC 6304':   (5.0, 0.4, 1.20, 0.12, 1.69, 0.10, -0.45),
    'NGC 6316':   (8.4, 0.6, 2.80, 0.30, 1.57, 0.10, -0.45),
    'NGC 6333':   (6.7, 0.4, 2.50, 0.20, 1.81, 0.10, -1.77),   # M9
    'NGC 6341':   (5.9, 0.2, 3.30, 0.20, 2.03, 0.08, -2.31),   # M92
    'NGC 6352':   (3.5, 0.3, 0.58, 0.06, 2.51, 0.10, -0.64),
    'NGC 6362':   (2.8, 0.2, 1.20, 0.10, 3.92, 0.15, -0.99),
    'NGC 6366':   (1.3, 0.2, 0.29, 0.04, 3.38, 0.20, -0.59),
    'NGC 6388':   (18.9, 0.5, 12.0, 0.80, 1.50, 0.08, -0.55),
    'NGC 6397':   (4.5, 0.1, 0.77, 0.05, 2.33, 0.08, -2.02),
    'NGC 6402':   (7.8, 0.4, 5.50, 0.40, 3.02, 0.15, -1.28),   # M14
    'NGC 6440':   (13.0, 0.6, 4.50, 0.40, 1.13, 0.08, -0.36),
    'NGC 6441':   (18.0, 0.5, 11.0, 0.70, 1.57, 0.08, -0.46),
    'NGC 6496':   (3.8, 0.4, 0.95, 0.12, 2.96, 0.15, -0.46),
    'NGC 6517':   (9.7, 0.7, 2.20, 0.25, 0.97, 0.08, -1.23),
    'NGC 6522':   (7.0, 0.4, 1.70, 0.15, 1.21, 0.08, -1.34),
    'NGC 6528':   (4.9, 0.5, 0.70, 0.10, 0.83, 0.08, -0.11),
    'NGC 6535':   (1.6, 0.4, 0.14, 0.03, 1.27, 0.10, -1.79),
    'NGC 6541':   (8.2, 0.3, 3.50, 0.25, 1.71, 0.08, -1.81),
    'NGC 6544':   (4.0, 0.5, 0.80, 0.12, 1.31, 0.10, -1.40),
    'NGC 6553':   (5.4, 0.4, 2.00, 0.20, 1.94, 0.10, -0.18),
    'NGC 6558':   (4.2, 0.5, 0.52, 0.08, 0.97, 0.08, -1.32),
    'NGC 6569':   (7.2, 0.5, 2.00, 0.20, 1.31, 0.08, -0.76),
    'NGC 6584':   (4.0, 0.4, 1.40, 0.15, 2.33, 0.12, -1.50),
    'NGC 6624':   (8.0, 0.4, 1.50, 0.12, 0.93, 0.05, -0.44),
    'NGC 6626':   (9.3, 0.4, 2.80, 0.20, 1.27, 0.08, -1.46),   # M28
    'NGC 6637':   (6.5, 0.3, 1.60, 0.12, 1.34, 0.08, -0.64),   # M69
    'NGC 6638':   (6.0, 0.5, 1.00, 0.12, 1.05, 0.08, -0.95),
    'NGC 6642':   (5.2, 0.5, 0.55, 0.08, 0.77, 0.06, -1.26),
    'NGC 6652':   (5.5, 0.5, 0.65, 0.08, 0.69, 0.05, -0.81),
    'NGC 6656':   (7.8, 0.2, 4.30, 0.25, 2.64, 0.10, -1.70),   # M22
    'NGC 6681':   (5.2, 0.3, 1.10, 0.10, 1.17, 0.06, -1.62),   # M70
    'NGC 6712':   (4.3, 0.3, 1.20, 0.12, 2.02, 0.10, -1.02),
    'NGC 6715':   (14.2, 0.4, 15.0, 1.0, 2.93, 0.12, -1.49),   # M54
    'NGC 6717':   (3.0, 0.4, 0.20, 0.03, 0.96, 0.06, -1.26),
    'NGC 6723':   (4.9, 0.3, 1.80, 0.15, 2.40, 0.10, -1.10),
    'NGC 6752':   (4.9, 0.2, 2.10, 0.12, 2.34, 0.08, -1.54),
    'NGC 6779':   (4.5, 0.3, 1.30, 0.12, 1.88, 0.08, -1.98),   # M56
    'NGC 6809':   (4.0, 0.2, 1.80, 0.12, 3.51, 0.12, -1.94),   # M55
    'NGC 6838':   (2.3, 0.2, 0.30, 0.03, 2.19, 0.10, -0.78),   # M71
    'NGC 6864':   (10.3, 0.5, 4.30, 0.35, 1.46, 0.08, -1.29),  # M75
    'NGC 6934':   (5.4, 0.4, 1.50, 0.15, 1.80, 0.08, -1.47),
    'NGC 6981':   (3.4, 0.4, 0.89, 0.10, 2.14, 0.10, -1.42),   # M72
    'NGC 7006':   (4.4, 0.5, 1.40, 0.18, 2.22, 0.12, -1.52),
    'NGC 7078':   (13.5, 0.3, 5.60, 0.35, 1.42, 0.06, -2.37),  # M15
    'NGC 7089':   (8.2, 0.3, 6.20, 0.35, 2.16, 0.08, -1.65),   # M2
    'NGC 7099':   (5.0, 0.2, 1.30, 0.10, 1.64, 0.06, -2.27),   # M30
    'NGC 7492':   (1.5, 0.3, 0.40, 0.06, 3.02, 0.15, -1.78),
}


def load_gc_data():
    """Load and preprocess GC data."""
    print("\n" + "="*70)
    print(" LOADING GLOBULAR CLUSTER DATA")
    print("="*70)
    
    data = []
    for name, vals in BAUMGARDT_DATA.items():
        sigma_0, sigma_err, M_dyn, M_err, r_h, r_h_err, feh = vals
        
        # Filter by quality
        if sigma_err / sigma_0 > CRITERIA['max_sigma_err_frac']:
            continue
        
        data.append({
            'Name': name,
            'sigma_obs': sigma_0,
            'sigma_err': sigma_err,
            'M_dyn': M_dyn * 1e5,  # Solar masses
            'M_err': M_err * 1e5,
            'r_h': r_h,            # pc
            'r_h_err': r_h_err,
            'FeH': feh,
        })
    
    df = pd.DataFrame(data)
    print(f"  Loaded {len(df)} GCs (quality cut: err/σ < {CRITERIA['max_sigma_err_frac']:.0%})")
    
    # Compute σ_bar using virial theorem
    # σ² = η * G * M / r_h, where η ≈ 0.4 for King models
    eta = 0.4
    
    df['sigma_bar'] = np.sqrt(eta * G_SI * df['M_dyn'] * M_SUN / (df['r_h'] * PC_M)) / 1000
    
    # Propagate errors
    df['sigma_bar_err'] = df['sigma_bar'] * np.sqrt(
        (0.5 * df['M_err'] / df['M_dyn'])**2 + 
        (0.5 * df['r_h_err'] / df['r_h'])**2
    )
    
    # Compute residual
    df['Delta_GC'] = np.log10(df['sigma_obs']) - np.log10(df['sigma_bar'])
    
    # Error on Delta
    df['Delta_err'] = np.sqrt(
        (df['sigma_err'] / (df['sigma_obs'] * np.log(10)))**2 +
        (df['sigma_bar_err'] / (df['sigma_bar'] * np.log(10)))**2
    )
    
    # Compute density
    V_pc3 = (4/3) * np.pi * df['r_h']**3
    df['rho_Msun_pc3'] = df['M_dyn'] / V_pc3
    df['log_rho'] = np.log10(df['rho_Msun_pc3'])
    
    # Convert to g/cm³ for comparison
    df['rho_cgs'] = df['rho_Msun_pc3'] * M_SUN * 1000 / (PC_M * 100)**3
    df['log_rho_cgs'] = np.log10(df['rho_cgs'])
    
    print(f"  σ_obs range: {df['sigma_obs'].min():.1f} - {df['sigma_obs'].max():.1f} km/s")
    print(f"  σ_bar range: {df['sigma_bar'].min():.1f} - {df['sigma_bar'].max():.1f} km/s")
    print(f"  Δ_GC range: {df['Delta_GC'].min():.3f} to {df['Delta_GC'].max():.3f}")
    print(f"  ⟨Δ_GC⟩ = {df['Delta_GC'].mean():.4f} ± {df['Delta_GC'].std():.4f}")
    print(f"  log(ρ) range: {df['log_rho'].min():.2f} to {df['log_rho'].max():.2f} M☉/pc³")
    
    return df


# =============================================================================
# TOE MODEL
# =============================================================================

def toe_coupling(log_rho, lambda_QR, rho_c, delta):
    """TOE coupling function for GCs."""
    rho_0 = 3.0  # Reference density in log M_sun/pc³
    
    C_Q = 10**((rho_0 - log_rho) * CRITERIA['beta_Q'])
    C_R = 10**((rho_0 - log_rho) * CRITERIA['beta_R'])
    
    f = 0.7  # High density → high f
    
    P = np.pi * expit((log_rho - rho_c) / delta)
    coupling = lambda_QR * (C_Q * f - C_R * (1 - f))
    
    return coupling * np.cos(P)


def log_prior(params):
    """Log prior for GC model."""
    alpha, lambda_QR, rho_c, delta, tau, gamma = params
    
    log_p = 0.0
    
    # Priors centered on HIGH DENSITY regime
    log_p += -0.5 * ((alpha - CRITERIA['prior_alpha_mean']) / CRITERIA['prior_alpha_std'])**2
    log_p += -0.5 * ((lambda_QR - CRITERIA['prior_lambda_mean']) / CRITERIA['prior_lambda_std'])**2
    log_p += -0.5 * ((rho_c - CRITERIA['prior_rho_c_mean']) / CRITERIA['prior_rho_c_std'])**2
    log_p += -0.5 * ((delta - 1.0) / 0.5)**2
    log_p += -0.5 * (gamma / 0.1)**2
    
    if tau < 0:
        return -np.inf
    log_p += -0.5 * (tau / 0.10)**2
    
    # Bounds
    if lambda_QR < 0.1 or lambda_QR > 5:
        return -np.inf
    if delta < 0.1 or delta > 5:
        return -np.inf
    
    return log_p


def log_likelihood(params, log_rho, y, y_err):
    """Student-t likelihood."""
    alpha, lambda_QR, rho_c, delta, tau, gamma = params
    
    nu = CRITERIA['student_nu']
    
    # Model
    coupling = toe_coupling(log_rho, lambda_QR, rho_c, delta)
    mu = alpha * coupling + gamma * (log_rho - 3)  # Centered on typical GC density
    
    sigma = np.sqrt(y_err**2 + tau**2)
    
    z = (y - mu) / sigma
    log_lik = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 
               0.5 * np.log(nu * np.pi) - np.log(sigma) -
               (nu + 1) / 2 * np.log(1 + z**2 / nu))
    
    return np.sum(log_lik)


def log_posterior(params, log_rho, y, y_err):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, log_rho, y, y_err)


def run_mcmc(log_rho, y, y_err, n_samples=2000, n_burnin=500):
    """MCMC sampler."""
    print(f"\n  Running MCMC (n={n_samples}, burnin={n_burnin})...")
    
    # Initial: [alpha, lambda_QR, rho_c, delta, tau, gamma]
    params = np.array([0.0, 1.7, 3.0, 1.0, 0.05, 0.0])
    n_params = len(params)
    
    scales = np.array([0.01, 0.15, 0.3, 0.15, 0.01, 0.01])
    
    samples = np.zeros((n_samples, n_params))
    log_posts = np.zeros(n_samples)
    
    current_lp = log_posterior(params, log_rho, y, y_err)
    n_accept = 0
    
    for i in range(n_samples + n_burnin):
        proposal = params + scales * np.random.randn(n_params)
        proposed_lp = log_posterior(proposal, log_rho, y, y_err)
        
        if np.log(np.random.rand()) < proposed_lp - current_lp:
            params = proposal
            current_lp = proposed_lp
            if i >= n_burnin:
                n_accept += 1
        
        if i >= n_burnin:
            samples[i - n_burnin] = params
            log_posts[i - n_burnin] = current_lp
    
    print(f"  Acceptance rate: {n_accept / n_samples:.2%}")
    
    return samples, log_posts


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print(" GC-001: TOE TEST ON GLOBULAR CLUSTERS")
    print(" Testing WB → Galactic Transition at 10^18 m")
    print(" Author: Jonathan Édouard Slama")
    print("="*70)
    
    # Load data
    df = load_gc_data()
    
    results = {
        'test': 'GC-001',
        'scale': '10^18 m',
        'purpose': 'Bridge WB (10^13-16) to UDG (10^20)',
        'n_gc': len(df),
    }
    
    # Check criteria
    print("\n" + "-"*70)
    print(" CRITERIA CHECK")
    print("-"*70)
    
    if len(df) >= CRITERIA['min_n_gc']:
        print(f"  ✅ N = {len(df)} >= {CRITERIA['min_n_gc']}")
    else:
        print(f"  ⚠️ N = {len(df)} < {CRITERIA['min_n_gc']}")
    
    rho_range = df['log_rho'].max() - df['log_rho'].min()
    print(f"  ✅ ρ range = {rho_range:.2f} dex")
    
    # Basic statistics
    print("\n" + "-"*70)
    print(" BASIC STATISTICS")
    print("-"*70)
    
    delta_mean = df['Delta_GC'].mean()
    delta_std = df['Delta_GC'].std()
    delta_sem = delta_std / np.sqrt(len(df))
    
    print(f"  ⟨Δ_GC⟩ = {delta_mean:.4f} ± {delta_sem:.4f}")
    print(f"  σ(Δ_GC) = {delta_std:.4f}")
    
    # T-test against 0
    t_stat, p_ttest = stats.ttest_1samp(df['Delta_GC'], 0)
    print(f"  t-test vs 0: t = {t_stat:.2f}, p = {p_ttest:.4f}")
    
    if p_ttest > 0.05:
        print(f"  → ⟨Δ⟩ compatible with 0 (screening confirmed)")
    else:
        print(f"  → ⟨Δ⟩ significantly different from 0")
    
    results['delta_mean'] = float(delta_mean)
    results['delta_std'] = float(delta_std)
    results['ttest_p'] = float(p_ttest)
    
    # Correlation test
    print("\n" + "-"*70)
    print(" CORRELATION TEST")
    print("-"*70)
    
    r, p_corr = stats.pearsonr(df['log_rho'], df['Delta_GC'])
    print(f"  r(Δ, log ρ) = {r:.3f}, p = {p_corr:.4f}")
    
    if p_corr > 0.05:
        print(f"  → No significant correlation (screening uniform)")
    else:
        print(f"  → Significant correlation detected")
    
    results['correlation'] = {'r': float(r), 'p': float(p_corr)}
    
    # MCMC fit
    print("\n" + "-"*70)
    print(" BAYESIAN FIT")
    print("-"*70)
    
    samples, log_posts = run_mcmc(
        df['log_rho'].values,
        df['Delta_GC'].values,
        df['Delta_err'].values,
        n_samples=CRITERIA['n_samples'],
        n_burnin=CRITERIA['n_burnin']
    )
    
    param_names = ['alpha', 'lambda_QR', 'rho_c', 'delta', 'tau', 'gamma']
    
    print("\n  Posterior estimates:")
    posterior = {}
    for i, name in enumerate(param_names):
        chain = samples[:, i]
        posterior[name] = {
            'mean': float(np.mean(chain)),
            'std': float(np.std(chain)),
            'ci_low': float(np.percentile(chain, 2.5)),
            'ci_high': float(np.percentile(chain, 97.5)),
        }
        print(f"    {name}: {posterior[name]['mean']:.4f} ± {posterior[name]['std']:.4f}")
    
    results['posterior'] = posterior
    
    # Model comparison
    print("\n" + "-"*70)
    print(" MODEL COMPARISON")
    print("-"*70)
    
    # Null: just mean
    null_ll = np.sum(stats.norm.logpdf(df['Delta_GC'], 0, df['Delta_err']))
    null_aic = -2 * null_ll + 2 * 0
    
    toe_ll = np.max(log_posts)
    toe_aic = -2 * toe_ll + 2 * 6
    
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
    print(" VERDICT")
    print("="*70)
    
    evidence = []
    
    # Mean Δ
    if abs(delta_mean) < 0.05:
        evidence.append(('⟨Δ⟩ ≈ 0', 'SCREENING', f"{delta_mean:.4f}"))
    else:
        evidence.append(('⟨Δ⟩ ≠ 0', 'SIGNAL', f"{delta_mean:.4f}"))
    
    # Correlation
    if p_corr > 0.05:
        evidence.append(('No Δ-ρ correlation', 'SCREENING', f"p={p_corr:.3f}"))
    else:
        evidence.append(('Δ-ρ correlation', 'SIGNAL', f"r={r:.3f}"))
    
    # λ_QR
    lambda_post = posterior['lambda_QR']
    if 1.3 < lambda_post['mean'] < 2.1:
        evidence.append(('λ_QR ~ WB', 'COHERENT', f"{lambda_post['mean']:.2f}"))
    elif 0.5 < lambda_post['mean'] < 1.3:
        evidence.append(('λ_QR ~ Galactic', 'TRANSITION', f"{lambda_post['mean']:.2f}"))
    
    for name, status, detail in evidence:
        symbol = '🎯' if status in ['SCREENING', 'COHERENT'] else '⚠️'
        print(f"  {symbol} {name}: {detail}")
    
    # Final verdict
    if abs(delta_mean) < 0.05 and p_corr > 0.05:
        verdict = "SCREENING CONFIRMED - GCs follow Newtonian dynamics"
        interpretation = "High density environment → complete chameleon screening → no QO+R signal visible"
    elif abs(delta_mean) < 0.10:
        verdict = "PARTIAL SCREENING - Marginal signal"
        interpretation = "Some QO+R influence may be present"
    else:
        verdict = "SIGNAL DETECTED - Unexpected for GCs"
        interpretation = "Requires further investigation"
    
    print(f"\n  VERDICT: {verdict}")
    print(f"  INTERPRETATION: {interpretation}")
    
    results['verdict'] = verdict
    results['interpretation'] = interpretation
    
    # λ_QR comparison
    print("\n" + "-"*70)
    print(" MULTI-SCALE COMPARISON")
    print("-"*70)
    
    print(f"  λ_QR(Wide Binaries) = 1.71 ± 0.02")
    print(f"  λ_QR(GC)            = {lambda_post['mean']:.2f} ± {lambda_post['std']:.2f}")
    print(f"  λ_QR(UDG)           = 0.83 ± 0.41")
    print(f"  λ_QR(Galactic)      = 0.94 ± 0.23")
    
    if lambda_post['mean'] > 1.3:
        print(f"\n  → GC in HIGH DENSITY REGIME (like WB)")
    else:
        print(f"\n  → GC transitioning to GALACTIC REGIME")
    
    # Figures
    if HAS_MPL:
        print("\n" + "-"*70)
        print(" CREATING FIGURES")
        print("-"*70)
        
        os.makedirs(FIGURES_DIR, exist_ok=True)
        
        # Figure 1: Δ vs log ρ
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.errorbar(df['log_rho'], df['Delta_GC'], yerr=df['Delta_err'],
                   fmt='o', color='purple', alpha=0.6, markersize=6,
                   capsize=2, label='Globular Clusters')
        
        ax.axhline(0, color='k', ls='--', alpha=0.3, label='No excess')
        ax.axhline(delta_mean, color='red', ls='-', alpha=0.7, label=f'Mean: {delta_mean:.3f}')
        
        ax.set_xlabel(r'$\log_{10}(\rho / M_\odot\,\mathrm{pc}^{-3})$', fontsize=12)
        ax.set_ylabel(r'$\Delta_{\mathrm{GC}} = \log(\sigma_{\mathrm{obs}}/\sigma_{\mathrm{bar}})$', fontsize=12)
        ax.set_title('GC-001: Globular Cluster Velocity Dispersion Excess', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, 'gc_delta_vs_rho.png'), dpi=150)
        fig.savefig(os.path.join(FIGURES_DIR, 'gc_delta_vs_rho.pdf'))
        print(f"    Saved: gc_delta_vs_rho.png/pdf")
        
        plt.close()
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "gc001_results.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
