#!/usr/bin/env python3
"""
===============================================================================
GC-001 V2: TOE TEST ON GLOBULAR CLUSTERS
CORRECTED: Using PHOTOMETRIC mass M_* (not dynamical M_dyn)
===============================================================================

CORRECTION from V1:
- V1 used M_dyn (derived from σ_obs) → CIRCULAR!
- V2 uses M_* from photometry (M_V + M/L ratio)
- This gives an INDEPENDENT prediction for σ_bar

Expected: If GCs have no dark matter, σ_obs ≈ σ_bar(M_*)
If QO+R active: σ_obs > σ_bar(M_*) at low density (unlikely for GCs)

Author: Jonathan Édouard Slama
Date: December 2025
===============================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

# Constants
G_SI = 6.674e-11
M_SUN = 1.989e30
PC_M = 3.086e16

# =============================================================================
# GLOBULAR CLUSTER DATA - PHOTOMETRIC MASSES
# =============================================================================

# Harris 2010 catalog + Baumgardt σ_0
# M_V = absolute V magnitude
# (M/L)_V estimated from [Fe/H] using Kruijssen & Mieske 2009 relation
# M_* = 10^(-0.4*(M_V - 4.83)) * (M/L)_V

# Data: Name, σ_0 (km/s), σ_err, M_V, [Fe/H], r_h (pc)
GC_PHOTOMETRIC_DATA = {
    # Name: (σ_0, σ_err, M_V, FeH, r_h)
    'NGC 104':    (11.0, 0.3, -9.42, -0.72, 4.15),   # 47 Tuc
    'NGC 288':    (2.9, 0.3, -6.75, -1.32, 5.77),
    'NGC 362':    (6.4, 0.3, -8.43, -1.26, 1.94),
    'NGC 1261':   (4.8, 0.5, -7.80, -1.27, 2.52),
    'NGC 1851':   (10.4, 0.4, -8.33, -1.18, 1.09),
    'NGC 1904':   (5.3, 0.4, -7.86, -1.60, 1.72),
    'NGC 2298':   (3.1, 0.4, -6.31, -1.92, 1.42),
    'NGC 2808':   (13.4, 0.4, -9.39, -1.14, 2.23),
    'NGC 3201':   (5.0, 0.2, -7.45, -1.59, 4.32),
    'NGC 4147':   (2.6, 0.4, -6.17, -1.80, 1.41),
    'NGC 4590':   (2.6, 0.2, -7.37, -2.23, 4.40),
    'NGC 4833':   (4.3, 0.3, -8.16, -1.85, 4.32),
    'NGC 5024':   (4.2, 0.3, -8.70, -2.10, 5.64),
    'NGC 5053':   (1.4, 0.3, -6.72, -2.27, 6.85),
    'NGC 5139':   (16.8, 0.3, -10.26, -1.53, 6.44),  # Omega Cen
    'NGC 5272':   (5.5, 0.2, -8.88, -1.50, 3.38),    # M3
    'NGC 5286':   (8.1, 0.4, -8.61, -1.69, 2.33),
    'NGC 5466':   (1.5, 0.2, -6.98, -1.98, 6.76),
    'NGC 5694':   (5.8, 0.6, -7.81, -1.98, 1.62),
    'NGC 5824':   (10.7, 0.5, -8.84, -1.91, 2.02),
    'NGC 5897':   (2.8, 0.3, -7.23, -1.90, 5.17),
    'NGC 5904':   (5.5, 0.2, -8.81, -1.29, 3.36),    # M5
    'NGC 5927':   (5.1, 0.4, -7.80, -0.49, 2.13),
    'NGC 5986':   (6.4, 0.4, -8.44, -1.59, 2.52),
    'NGC 6093':   (12.5, 0.5, -8.23, -1.75, 0.96),   # M80
    'NGC 6101':   (2.9, 0.4, -6.91, -1.98, 3.61),
    'NGC 6121':   (4.0, 0.2, -7.19, -1.16, 3.65),    # M4
    'NGC 6139':   (9.5, 0.6, -8.36, -1.65, 1.43),
    'NGC 6144':   (2.2, 0.3, -6.75, -1.76, 3.37),
    'NGC 6171':   (4.1, 0.3, -7.12, -1.02, 2.32),    # M107
    'NGC 6205':   (7.1, 0.2, -8.55, -1.53, 2.54),    # M13
    'NGC 6218':   (4.0, 0.2, -7.31, -1.37, 2.48),    # M12
    'NGC 6254':   (6.6, 0.3, -7.48, -1.56, 2.11),    # M10
    'NGC 6266':   (14.3, 0.4, -9.18, -1.18, 1.51),   # M62
    'NGC 6273':   (9.6, 0.4, -9.17, -1.74, 2.58),    # M19
    'NGC 6284':   (7.1, 0.5, -7.96, -1.26, 1.24),
    'NGC 6287':   (5.7, 0.5, -7.36, -2.10, 1.38),
    'NGC 6293':   (8.0, 0.5, -7.77, -1.99, 1.09),
    'NGC 6304':   (5.0, 0.4, -7.30, -0.45, 1.69),
    'NGC 6316':   (8.4, 0.6, -8.35, -0.45, 1.57),
    'NGC 6333':   (6.7, 0.4, -7.93, -1.77, 1.81),    # M9
    'NGC 6341':   (5.9, 0.2, -8.20, -2.31, 2.03),    # M92
    'NGC 6352':   (3.5, 0.3, -6.47, -0.64, 2.51),
    'NGC 6362':   (2.8, 0.2, -6.95, -0.99, 3.92),
    'NGC 6366':   (1.3, 0.2, -5.74, -0.59, 3.38),
    'NGC 6388':   (18.9, 0.5, -9.41, -0.55, 1.50),
    'NGC 6397':   (4.5, 0.1, -6.64, -2.02, 2.33),
    'NGC 6402':   (7.8, 0.4, -9.12, -1.28, 3.02),    # M14
    'NGC 6440':   (13.0, 0.6, -8.75, -0.36, 1.13),
    'NGC 6441':   (18.0, 0.5, -9.63, -0.46, 1.57),
    'NGC 6496':   (3.8, 0.4, -7.20, -0.46, 2.96),
    'NGC 6517':   (9.7, 0.7, -8.28, -1.23, 0.97),
    'NGC 6522':   (7.0, 0.4, -7.66, -1.34, 1.21),
    'NGC 6528':   (4.9, 0.5, -6.57, -0.11, 0.83),
    'NGC 6535':   (1.6, 0.4, -4.75, -1.79, 1.27),
    'NGC 6541':   (8.2, 0.3, -8.37, -1.81, 1.71),
    'NGC 6544':   (4.0, 0.5, -6.66, -1.40, 1.31),
    'NGC 6553':   (5.4, 0.4, -7.77, -0.18, 1.94),
    'NGC 6558':   (4.2, 0.5, -6.44, -1.32, 0.97),
    'NGC 6569':   (7.2, 0.5, -7.68, -0.76, 1.31),
    'NGC 6584':   (4.0, 0.4, -7.68, -1.50, 2.33),
    'NGC 6624':   (8.0, 0.4, -7.49, -0.44, 0.93),
    'NGC 6626':   (9.3, 0.4, -8.16, -1.46, 1.27),    # M28
    'NGC 6637':   (6.5, 0.3, -7.64, -0.64, 1.34),    # M69
    'NGC 6638':   (6.0, 0.5, -7.13, -0.95, 1.05),
    'NGC 6642':   (5.2, 0.5, -6.77, -1.26, 0.77),
    'NGC 6652':   (5.5, 0.5, -6.66, -0.81, 0.69),
    'NGC 6656':   (7.8, 0.2, -8.50, -1.70, 2.64),    # M22
    'NGC 6681':   (5.2, 0.3, -7.11, -1.62, 1.17),    # M70
    'NGC 6712':   (4.3, 0.3, -7.50, -1.02, 2.02),
    'NGC 6715':   (14.2, 0.4, -10.01, -1.49, 2.93),  # M54
    'NGC 6717':   (3.0, 0.4, -5.66, -1.26, 0.96),
    'NGC 6723':   (4.9, 0.3, -7.83, -1.10, 2.40),
    'NGC 6752':   (4.9, 0.2, -7.73, -1.54, 2.34),
    'NGC 6779':   (4.5, 0.3, -7.38, -1.98, 1.88),    # M56
    'NGC 6809':   (4.0, 0.2, -7.57, -1.94, 3.51),    # M55
    'NGC 6838':   (2.3, 0.2, -5.61, -0.78, 2.19),    # M71
    'NGC 6864':   (10.3, 0.5, -8.55, -1.29, 1.46),   # M75
    'NGC 6934':   (5.4, 0.4, -7.45, -1.47, 1.80),
    'NGC 6981':   (3.4, 0.4, -7.04, -1.42, 2.14),    # M72
    'NGC 7006':   (4.4, 0.5, -7.68, -1.52, 2.22),
    'NGC 7078':   (13.5, 0.3, -9.19, -2.37, 1.42),   # M15
    'NGC 7089':   (8.2, 0.3, -9.03, -1.65, 2.16),    # M2
    'NGC 7099':   (5.0, 0.2, -7.45, -2.27, 1.64),    # M30
    'NGC 7492':   (1.5, 0.3, -5.77, -1.78, 3.02),
}


def compute_ML_ratio(feh):
    """
    Estimate V-band M/L ratio from metallicity.
    Using Kruijssen & Mieske 2009 relation for old stellar populations.
    
    (M/L)_V ≈ 1.5 + 0.5 * ([Fe/H] + 1.5)
    
    Typical range: 1.5 - 2.5 for GCs
    """
    ML = 1.8 + 0.3 * (feh + 1.5)
    return np.clip(ML, 1.2, 3.0)


def MV_to_Mstar(M_V, ML_ratio):
    """Convert absolute V magnitude to stellar mass."""
    # M_V(Sun) = 4.83
    L_V = 10**(-0.4 * (M_V - 4.83))  # In solar luminosities
    M_star = L_V * ML_ratio  # In solar masses
    return M_star


def load_gc_data_v2():
    """Load GC data using PHOTOMETRIC masses."""
    print("\n" + "="*70)
    print(" LOADING GC DATA V2 - PHOTOMETRIC MASSES")
    print("="*70)
    
    data = []
    for name, vals in GC_PHOTOMETRIC_DATA.items():
        sigma_0, sigma_err, M_V, feh, r_h = vals
        
        # Quality cut
        if sigma_err / sigma_0 > 0.20:
            continue
        
        # Compute M/L and M_*
        ML = compute_ML_ratio(feh)
        M_star = MV_to_Mstar(M_V, ML)
        
        data.append({
            'Name': name,
            'sigma_obs': sigma_0,
            'sigma_err': sigma_err,
            'M_V': M_V,
            'FeH': feh,
            'ML_ratio': ML,
            'M_star': M_star,
            'r_h': r_h,
        })
    
    df = pd.DataFrame(data)
    print(f"  Loaded {len(df)} GCs with photometric M_*")
    
    # Compute σ_bar from STELLAR mass (not dynamical!)
    # σ² = η * G * M_* / r_h, where η ≈ 0.4 for King models
    eta = 0.4
    
    df['sigma_bar'] = np.sqrt(eta * G_SI * df['M_star'] * M_SUN / (df['r_h'] * PC_M)) / 1000
    
    # Uncertainty on σ_bar (from M/L uncertainty ~20%)
    df['sigma_bar_err'] = df['sigma_bar'] * 0.15
    
    # Residual
    df['Delta_GC'] = np.log10(df['sigma_obs']) - np.log10(df['sigma_bar'])
    
    # Error on Delta
    df['Delta_err'] = np.sqrt(
        (df['sigma_err'] / (df['sigma_obs'] * np.log(10)))**2 +
        (df['sigma_bar_err'] / (df['sigma_bar'] * np.log(10)))**2
    )
    
    # Density (using M_star, not M_dyn)
    V_pc3 = (4/3) * np.pi * df['r_h']**3
    df['rho_Msun_pc3'] = df['M_star'] / V_pc3
    df['log_rho'] = np.log10(df['rho_Msun_pc3'])
    
    print(f"  M_* range: {df['M_star'].min():.2e} - {df['M_star'].max():.2e} M☉")
    print(f"  σ_obs range: {df['sigma_obs'].min():.1f} - {df['sigma_obs'].max():.1f} km/s")
    print(f"  σ_bar range: {df['sigma_bar'].min():.1f} - {df['sigma_bar'].max():.1f} km/s")
    print(f"  Δ_GC range: {df['Delta_GC'].min():.3f} to {df['Delta_GC'].max():.3f}")
    print(f"  ⟨Δ_GC⟩ = {df['Delta_GC'].mean():.4f} ± {df['Delta_GC'].std():.4f}")
    print(f"  log(ρ_*) range: {df['log_rho'].min():.2f} to {df['log_rho'].max():.2f}")
    
    return df


def main():
    print("="*70)
    print(" GC-001 V2: TOE TEST ON GLOBULAR CLUSTERS")
    print(" Using PHOTOMETRIC masses (not dynamical)")
    print(" Author: Jonathan Édouard Slama")
    print("="*70)
    
    df = load_gc_data_v2()
    
    results = {
        'test': 'GC-001',
        'version': 'V2',
        'correction': 'Using M_* from photometry, not M_dyn',
        'n_gc': len(df),
    }
    
    # Basic statistics
    print("\n" + "-"*70)
    print(" BASIC STATISTICS")
    print("-"*70)
    
    delta_mean = df['Delta_GC'].mean()
    delta_std = df['Delta_GC'].std()
    delta_sem = delta_std / np.sqrt(len(df))
    
    print(f"  ⟨Δ_GC⟩ = {delta_mean:.4f} ± {delta_sem:.4f}")
    print(f"  σ(Δ_GC) = {delta_std:.4f}")
    
    # T-test
    t_stat, p_ttest = stats.ttest_1samp(df['Delta_GC'], 0)
    print(f"  t-test vs 0: t = {t_stat:.2f}, p = {p_ttest:.4f}")
    
    results['delta_mean'] = float(delta_mean)
    results['delta_std'] = float(delta_std)
    results['ttest_p'] = float(p_ttest)
    
    # Interpretation
    print("\n  INTERPRETATION:")
    if delta_mean > 0.05:
        print(f"  → ⟨Δ⟩ > 0 : GCs have EXCESS σ vs stellar mass prediction")
        print(f"  → This could indicate: dark matter, MOND, or QO+R effect")
    elif delta_mean < -0.05:
        print(f"  → ⟨Δ⟩ < 0 : GCs have DEFICIT σ vs stellar mass prediction")
        print(f"  → This could indicate: mass loss, tidal stripping, or systematic error")
    else:
        print(f"  → ⟨Δ⟩ ≈ 0 : GCs follow Newtonian dynamics with stellar mass only")
        print(f"  → No dark matter or modified gravity needed (SCREENING CONFIRMED)")
    
    # Correlation
    print("\n" + "-"*70)
    print(" CORRELATION TEST")
    print("-"*70)
    
    r, p_corr = stats.pearsonr(df['log_rho'], df['Delta_GC'])
    print(f"  r(Δ, log ρ_*) = {r:.3f}, p = {p_corr:.4f}")
    
    results['correlation'] = {'r': float(r), 'p': float(p_corr)}
    
    if r < -0.2 and p_corr < 0.05:
        print(f"  → ANTI-correlation: Lower density → larger excess")
        print(f"  → Consistent with QO+R screening prediction!")
    elif r > 0.2 and p_corr < 0.05:
        print(f"  → POSITIVE correlation: unexpected pattern")
    else:
        print(f"  → No significant correlation")
    
    # Verdict
    print("\n" + "="*70)
    print(" VERDICT")
    print("="*70)
    
    if abs(delta_mean) < 0.10 and p_corr > 0.05:
        verdict = "SCREENING CONFIRMED"
        interpretation = "GCs follow Newtonian dynamics at high density (no QO+R signal)"
    elif delta_mean > 0.10:
        verdict = "EXCESS DETECTED"
        interpretation = "GCs show excess σ - possible DM or modified gravity"
    elif delta_mean < -0.10:
        verdict = "DEFICIT DETECTED"
        interpretation = "GCs show deficit σ - possible mass loss or systematics"
    else:
        verdict = "MARGINAL RESULTS"
        interpretation = "Small offset, requires more analysis"
    
    print(f"  ⟨Δ_GC⟩ = {delta_mean:.3f} ± {delta_sem:.3f}")
    print(f"  VERDICT: {verdict}")
    print(f"  INTERPRETATION: {interpretation}")
    
    results['verdict'] = verdict
    results['interpretation'] = interpretation
    
    # Multi-scale comparison
    print("\n" + "-"*70)
    print(" MULTI-SCALE COMPARISON")
    print("-"*70)
    print(f"  UDG (10²⁰ m):     ⟨Δ⟩ = +0.21 (excess, low density)")
    print(f"  GC (10¹⁸ m):      ⟨Δ⟩ = {delta_mean:+.3f} (high density)")
    print(f"  Expected (QO+R): ⟨Δ_GC⟩ ≈ 0 (screening)")
    
    # Figure
    if HAS_MPL:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.errorbar(df['log_rho'], df['Delta_GC'], yerr=df['Delta_err'],
                   fmt='o', color='purple', alpha=0.6, markersize=6, capsize=2)
        
        ax.axhline(0, color='k', ls='--', alpha=0.5, label='Newtonian (no DM)')
        ax.axhline(delta_mean, color='red', ls='-', lw=2, 
                  label=f'Mean: {delta_mean:.3f}')
        
        # Trend line
        z = np.polyfit(df['log_rho'], df['Delta_GC'], 1)
        x_plot = np.linspace(df['log_rho'].min(), df['log_rho'].max(), 50)
        ax.plot(x_plot, np.polyval(z, x_plot), 'g--', alpha=0.7, 
               label=f'Trend (r={r:.2f})')
        
        ax.set_xlabel(r'$\log_{10}(\rho_* / M_\odot\,\mathrm{pc}^{-3})$', fontsize=12)
        ax.set_ylabel(r'$\Delta_{\mathrm{GC}} = \log(\sigma_{\mathrm{obs}}/\sigma_{\mathrm{bar}})$', fontsize=12)
        ax.set_title('GC-001 V2: Globular Clusters (Photometric Masses)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, 'gc_delta_vs_rho_v2.png'), dpi=150)
        fig.savefig(os.path.join(FIGURES_DIR, 'gc_delta_vs_rho_v2.pdf'))
        print(f"\n  Saved: gc_delta_vs_rho_v2.png/pdf")
    
    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "gc001_v2_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Results saved: gc001_v2_results.json")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
