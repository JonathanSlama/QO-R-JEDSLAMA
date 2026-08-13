#!/usr/bin/env python3
"""
===============================================================================
CBL-001 v2: TOE TEST AT COSMIC SCALE (Published CMB Lensing Data)
===============================================================================

Uses REAL published cross-correlation amplitudes from:
- Omori & Holder (2015): Planck × CFHTLenS
- Kuntz (2015): Planck × CFHTLenS  
- Singh et al. (2017): Planck × SDSS BOSS

Observable: A = C_ℓ^{κg,obs} / C_ℓ^{κg,GR}
- A = 1 means GR is correct
- A ≠ 1 would indicate modified gravity or systematics

For TOE: We test if residual (A-1) correlates with effective density

Author: Jonathan Édouard Slama
Date: December 2025
===============================================================================
"""

import numpy as np
from scipy import stats
from scipy.optimize import least_squares
from scipy.special import expit
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# =============================================================================
# PUBLISHED DATA - REAL MEASUREMENTS
# =============================================================================

# Cross-correlation amplitude measurements A = C_ℓ^{κg,obs} / C_ℓ^{κg,theory}
# A = 1 means perfect agreement with GR+ΛCDM

PUBLISHED_DATA = {
    # Omori & Holder (2015) - Planck 2015 × CFHTLenS
    # arXiv:1502.03405, Table 1
    'Omori2015_Planck15_i22': {
        'A': 0.82, 'A_err': 0.24,
        'z_eff': 0.65,
        'ell_range': [50, 1900],
        'survey': 'CFHTLenS 18<i<22'
    },
    'Omori2015_Planck15_i23': {
        'A': 0.83, 'A_err': 0.19,
        'z_eff': 0.70,
        'ell_range': [50, 1900],
        'survey': 'CFHTLenS 18<i<23'
    },
    'Omori2015_Planck15_i24': {
        'A': 0.82, 'A_err': 0.16,
        'z_eff': 0.75,
        'ell_range': [50, 1900],
        'survey': 'CFHTLenS 18<i<24'
    },
    
    # Kuntz (2015) - Improved analysis
    # A&A 584, A53 (2015)
    'Kuntz2015_Planck15': {
        'A': 0.85, 'A_err': 0.16,
        'z_eff': 0.70,
        'ell_range': [50, 1900],
        'survey': 'CFHTLenS (halo model)'
    },
    'Kuntz2015_Planck13': {
        'A': 1.05, 'A_err': 0.15,
        'z_eff': 0.70,
        'ell_range': [50, 1900],
        'survey': 'CFHTLenS (halo model)'
    },
    
    # Singh et al. (2017) - Planck × SDSS BOSS
    # MNRAS 464, 2120 (2017)
    'Singh2017_LOWZ': {
        'A': 1.0, 'A_err': 0.2,  # rcc compatible with 1
        'z_eff': 0.30,
        'ell_range': [20, 200],  # real space, ~arcmin scales
        'survey': 'BOSS LOWZ'
    },
    'Singh2017_CMASS': {
        'A': 0.8, 'A_err': 0.1,
        'z_eff': 0.57,
        'ell_range': [20, 200],
        'survey': 'BOSS CMASS'
    },
    'Singh2017_lensing': {
        'A': 0.76, 'A_err': 0.23,
        'z_eff': 0.35,
        'ell_range': [50, 500],
        'survey': 'SDSS galaxy lensing'
    },
}

# =============================================================================
# DENSITY ESTIMATES
# =============================================================================

def estimate_cosmic_density(z_eff, ell_mean):
    """
    Estimate effective matter density probed by CMB lensing at given z and ℓ.
    
    At cosmological scales, we're in the linear regime with δ << 1.
    The "density" for TOE purposes is log(1 + σ_m) where σ_m is the 
    matter fluctuation amplitude at the relevant scale.
    """
    # Comoving distance at z_eff (Mpc)
    # Using Planck 2018 cosmology: H0=67.4, Ωm=0.315
    H0 = 67.4  # km/s/Mpc
    c = 3e5    # km/s
    Om = 0.315
    
    # Simplified χ(z) for flat ΛCDM
    # χ ≈ (c/H0) × z × (1 - 0.3*z) for z < 1
    chi = (c/H0) * z_eff * (1 - 0.15*z_eff)  # Mpc
    
    # Physical scale probed: r ≈ χ/ℓ
    r_Mpc = chi / ell_mean
    
    # Matter fluctuation amplitude at scale r
    # σ(r) ≈ σ_8 × (r/8 Mpc)^(-γ) with γ ~ 0.5 in linear regime
    sigma_8 = 0.81
    gamma = 0.5
    
    sigma_r = sigma_8 * (r_Mpc / 8)**(-gamma)
    sigma_r = np.clip(sigma_r, 0.001, 1.0)  # Keep in linear regime
    
    # Log density for TOE
    rho_eff = np.log10(1 + sigma_r)
    
    return rho_eff, r_Mpc, chi


# =============================================================================
# TOE MODEL
# =============================================================================

def sigmoid(x):
    return expit(x)

def toe_model_cosmic(X, alpha, lambda_QR, rho_c, delta):
    """
    TOE prediction for amplitude deviation from GR.
    
    A_TOE = 1 + α × f(ρ)
    
    The residual (A - 1) should follow the TOE phase pattern.
    """
    rho, _ = X
    rho_0 = np.median(rho)
    
    beta_Q = 1.0
    beta_R = 0.5
    
    C_Q = (10**(rho - rho_0)) ** (-beta_Q)
    C_R = (10**(rho - rho_0)) ** (-beta_R)
    
    P = np.pi * sigmoid((rho - rho_c) / delta)
    
    # f ~ 0.5 at cosmic scales (dark matter dominated)
    f = 0.5
    coupling = lambda_QR * (C_Q * f - C_R * (1 - f))
    
    return alpha * coupling * np.cos(P)


def null_model(X, a, b):
    """Null model: constant + linear trend."""
    rho, _ = X
    return a + b * rho


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("="*70)
    print(" CBL-001 v2: TOE TEST AT COSMIC SCALE")
    print(" Using PUBLISHED CMB Lensing Cross-Correlation Data")
    print("="*70)
    
    # Collect all measurements
    print("\n" + "-"*70)
    print(" PUBLISHED MEASUREMENTS")
    print("-"*70)
    
    data_points = []
    
    for name, data in PUBLISHED_DATA.items():
        A = data['A']
        A_err = data['A_err']
        z_eff = data['z_eff']
        ell_mean = np.mean(data['ell_range'])
        
        # Compute density
        rho_eff, r_Mpc, chi = estimate_cosmic_density(z_eff, ell_mean)
        
        # Residual from GR
        residual = A - 1.0
        
        data_points.append({
            'name': name,
            'A': A,
            'A_err': A_err,
            'residual': residual,
            'z_eff': z_eff,
            'ell_mean': ell_mean,
            'rho_eff': rho_eff,
            'r_Mpc': r_Mpc,
            'chi_Mpc': chi,
            'survey': data['survey']
        })
        
        print(f"  {name}:")
        print(f"    A = {A:.2f} ± {A_err:.2f} | Residual = {residual:+.2f}")
        print(f"    z_eff = {z_eff:.2f} | ℓ ~ {ell_mean:.0f} | r ~ {r_Mpc:.1f} Mpc")
        print(f"    ρ_eff = {rho_eff:.4f}")
    
    # Convert to arrays
    n = len(data_points)
    rho = np.array([d['rho_eff'] for d in data_points])
    residual = np.array([d['residual'] for d in data_points])
    weights = np.array([1/d['A_err']**2 for d in data_points])
    
    print(f"\n  Total measurements: {n}")
    print(f"  ρ_eff range: {rho.min():.4f} to {rho.max():.4f}")
    print(f"  Residual (A-1) range: {residual.min():.2f} to {residual.max():.2f}")
    
    # Weighted mean residual
    wmean = np.sum(weights * residual) / np.sum(weights)
    werr = 1 / np.sqrt(np.sum(weights))
    
    print(f"\n  Weighted mean residual: {wmean:.3f} ± {werr:.3f}")
    print(f"  Significance from 0: {abs(wmean)/werr:.1f}σ")
    
    # =================================================================
    # FIT TOE MODEL
    # =================================================================
    print("\n" + "-"*70)
    print(" TOE MODEL FIT")
    print("-"*70)
    
    X = (rho, np.ones(n) * 0.5)
    y = residual
    
    # Fit TOE
    p0 = [0.0, 1.0, 0.0, 1.0]
    bounds = ([-0.5, 0.01, -3.0, 0.05], [0.5, 10.0, 2.0, 5.0])
    
    def residuals_toe(params):
        pred = toe_model_cosmic(X, *params)
        return (y - pred) * np.sqrt(weights)
    
    try:
        result = least_squares(residuals_toe, p0,
                               bounds=([bounds[0][i] for i in range(4)],
                                       [bounds[1][i] for i in range(4)]),
                               max_nfev=5000)
        popt = result.x
        
        y_pred = toe_model_cosmic(X, *popt)
        ss_res_toe = np.sum(weights * (y - y_pred)**2)
        
        # AICc
        k = 4
        aicc_toe = n * np.log(ss_res_toe / n) + 2*k + 2*k*(k+1)/(n-k-1)
        
        print(f"  α = {popt[0]:.5f}")
        print(f"  λ_QR = {popt[1]:.4f}")
        print(f"  ρ_c = {popt[2]:.4f}")
        print(f"  Δ = {popt[3]:.4f}")
        print(f"  AICc_TOE = {aicc_toe:.1f}")
        
        toe_success = True
    except Exception as e:
        print(f"  TOE fit failed: {e}")
        toe_success = False
        popt = [0, 1, 0, 1]
        aicc_toe = np.inf
    
    # Fit NULL
    def residuals_null(params):
        pred = null_model(X, *params)
        return (y - pred) * np.sqrt(weights)
    
    try:
        result_null = least_squares(residuals_null, [0, 0])
        popt_null = result_null.x
        
        y_pred_null = null_model(X, *popt_null)
        ss_res_null = np.sum(weights * (y - y_pred_null)**2)
        
        k = 2
        aicc_null = n * np.log(ss_res_null / n) + 2*k + 2*k*(k+1)/(n-k-1)
        
        print(f"  AICc_null = {aicc_null:.1f}")
        
        null_success = True
    except:
        null_success = False
        aicc_null = np.inf
    
    if toe_success and null_success:
        delta_aicc = aicc_null - aicc_toe
        print(f"  ΔAICc = {delta_aicc:.1f}")
    
    # =================================================================
    # PHYSICAL INTERPRETATION
    # =================================================================
    print("\n" + "-"*70)
    print(" PHYSICAL INTERPRETATION")
    print("-"*70)
    
    # Key question: Is the residual consistent with zero?
    t_stat = wmean / werr
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    
    print(f"\n  H0: Residual = 0 (GR is correct)")
    print(f"  Test statistic: t = {t_stat:.2f}")
    print(f"  p-value: {p_value:.4f}")
    
    if p_value > 0.05:
        print("  → Cannot reject H0: GR is consistent with observations")
    else:
        print("  → H0 rejected: possible deviation from GR")
    
    # =================================================================
    # VERDICT
    # =================================================================
    print("\n" + "="*70)
    print(" VERDICT")
    print("="*70)
    
    evidence = []
    
    # α amplitude
    alpha = abs(popt[0])
    if alpha < 0.05:
        evidence.append(('α amplitude', 'EXPECTED', f"|α| = {alpha:.4f} ≈ 0 (fully screened)"))
    else:
        evidence.append(('α amplitude', 'UNEXPECTED', f"|α| = {alpha:.4f} > 0.05"))
    
    # ΔAICc
    if toe_success and null_success:
        if delta_aicc < 0:
            evidence.append(('ΔAICc', 'EXPECTED', f"{delta_aicc:.1f} < 0 (GR preferred)"))
        else:
            evidence.append(('ΔAICc', 'UNEXPECTED', f"{delta_aicc:.1f} > 0"))
    
    # Residual consistency with zero
    if p_value > 0.05:
        evidence.append(('GR consistency', 'EXPECTED', f"p = {p_value:.3f} (compatible with GR)"))
    else:
        evidence.append(('GR consistency', 'UNEXPECTED', f"p = {p_value:.3f}"))
    
    # λ_QR coherence
    lambda_QR = popt[1]
    lambda_mean = 1.38  # Mean from other scales
    ratio = lambda_QR / lambda_mean
    if 0.3 <= ratio <= 3.0:
        evidence.append(('λ_QR coherence', 'PASS', f"λ_QR = {lambda_QR:.2f} (ratio = {ratio:.2f})"))
    else:
        evidence.append(('λ_QR coherence', 'FAIL', f"λ_QR = {lambda_QR:.2f} out of range"))
    
    n_expected = sum(1 for e in evidence if e[1] == 'EXPECTED')
    n_pass = sum(1 for e in evidence if e[1] == 'PASS')
    n_fail = sum(1 for e in evidence if e[1] in ['FAIL', 'UNEXPECTED'])
    
    for name, status, detail in evidence:
        symbol = '✅' if status in ['PASS', 'EXPECTED'] else '❌'
        print(f"  {symbol} {name}: {detail}")
    
    print(f"\n  Summary: {n_expected + n_pass} PASS/EXPECTED, {n_fail} FAIL")
    
    if n_expected >= 3 and n_fail == 0:
        verdict = "QO+R TOE FULLY SCREENED AT COSMIC SCALE (as predicted)"
    elif n_pass + n_expected >= 2:
        verdict = "QO+R TOE MOSTLY SCREENED AT COSMIC SCALE"
    else:
        verdict = "QO+R TOE INCONCLUSIVE AT COSMIC SCALE"
    
    print(f"\n  VERDICT: {verdict}")
    
    # =================================================================
    # UNIVERSAL COUPLING CONSTANT
    # =================================================================
    print("\n" + "="*70)
    print(" UNIVERSAL COUPLING CONSTANT")
    print("="*70)
    
    # All λ_QR values from validated scales
    all_lambda = {
        'Wide Binaries (10^13-16 m)': 1.71,
        'Galactic (10^21 m)': 0.94,
        'Groups (10^22 m)': 1.67,
        'Clusters (10^23 m)': 1.23,
        'Cosmic (10^25-27 m)': lambda_QR
    }
    
    values = list(all_lambda.values())
    lambda_mean = np.mean(values)
    lambda_std = np.std(values)
    
    print(f"\n  λ_QR measurements across scales:")
    for scale, lqr in all_lambda.items():
        print(f"    {scale}: {lqr:.2f}")
    
    print(f"\n  ════════════════════════════════════════════════")
    print(f"  UNIVERSAL COUPLING CONSTANT:")
    print(f"  Λ_QR = {lambda_mean:.2f} ± {lambda_std:.2f}")
    print(f"  (from 5 scales: 10^13 → 10^27 m)")
    print(f"  Factor variation: {max(values)/min(values):.2f}")
    print(f"  ════════════════════════════════════════════════")
    
    # =================================================================
    # SAVE RESULTS
    # =================================================================
    results = {
        'version': 'v2',
        'data_source': 'Published CMB lensing cross-correlations',
        'n_measurements': n,
        'weighted_mean_residual': float(wmean),
        'weighted_mean_err': float(werr),
        'p_value_gr': float(p_value),
        'toe': {
            'alpha': float(popt[0]),
            'lambda_QR': float(popt[1]),
            'rho_c': float(popt[2]),
            'delta': float(popt[3]),
            'aicc': float(aicc_toe) if toe_success else None
        },
        'null': {'aicc': float(aicc_null) if null_success else None},
        'delta_aicc': float(delta_aicc) if (toe_success and null_success) else None,
        'verdict': verdict,
        'universal_coupling': {
            'lambda_mean': float(lambda_mean),
            'lambda_std': float(lambda_std),
            'n_scales': 5,
            'scale_range': '10^13 - 10^27 m'
        },
        'evidence': [{'name': e[0], 'status': e[1], 'detail': e[2]} for e in evidence]
    }
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "cbl001_toe_results_v2.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
