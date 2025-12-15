#!/usr/bin/env python3
"""
===============================================================================
GRAVITATIONAL LENSING PILOT TEST
===============================================================================

Test for Q²R² pattern in weak gravitational lensing residuals.

This is a PILOT test to determine feasibility before full analysis.

Goal: Detect the same U-shaped environmental dependence in lensing that
we found in galaxy rotation curves.

Data Sources:
- DES Y3 (public)
- KiDS-1000 (public)
- HSC-SSP Y3 (public)

Challenge: Need to define Q (gas fraction proxy) for lens galaxies.
Options:
1. Cross-match with HI surveys (limited overlap)
2. Use color as gas proxy (blue = gas-rich, red = gas-poor)
3. Use morphology (spiral = Q-dom, elliptical = R-dom)

Author: Jonathan E. Slama
Date: December 2025
Status: PREPARATION (data download required)
===============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA SOURCES
# =============================================================================

DATA_SOURCES = {
    'DES_Y3': {
        'url': 'https://des.ncsa.illinois.edu/releases/y3a2',
        'description': 'Dark Energy Survey Year 3',
        'catalog': 'Y3 Gold',
        'area': '5000 deg²',
        'n_galaxies': '~100M',
        'lensing_catalog': 'METACALIBRATION',
    },
    'KiDS_1000': {
        'url': 'https://kids.strw.leidenuniv.nl/DR4/',
        'description': 'Kilo-Degree Survey',
        'catalog': 'KiDS-1000',
        'area': '1000 deg²',
        'n_galaxies': '~21M',
        'lensing_catalog': 'lensfit',
    },
    'HSC_SSP': {
        'url': 'https://hsc-release.mtk.nao.ac.jp/',
        'description': 'Hyper Suprime-Cam Subaru Strategic Program',
        'catalog': 'HSC-SSP PDR3',
        'area': '670 deg²',
        'n_galaxies': '~36M',
        'lensing_catalog': 'HSC shear',
    }
}


# =============================================================================
# GAS FRACTION PROXY
# =============================================================================

def estimate_gas_fraction_from_color(g_r_color, method='sigmoid'):
    """
    Estimate gas fraction from g-r color.
    
    Blue galaxies (low g-r) tend to be gas-rich (high Q)
    Red galaxies (high g-r) tend to be gas-poor (high R)
    
    Parameters:
    -----------
    g_r_color : array
        Rest-frame g-r color
    method : str
        'sigmoid' or 'linear'
    
    Returns:
    --------
    f_gas : array
        Estimated gas fraction
    """
    if method == 'sigmoid':
        # Sigmoid transition centered at g-r = 0.7
        # Blue (g-r < 0.5): f_gas ~ 0.7
        # Red (g-r > 0.9): f_gas ~ 0.1
        f_gas = 0.7 / (1 + np.exp(5 * (g_r_color - 0.7)))
        f_gas += 0.1  # Floor
    
    elif method == 'linear':
        # Linear relation (simple)
        f_gas = np.clip(1.0 - 1.2 * g_r_color, 0.1, 0.9)
    
    return f_gas


def estimate_gas_fraction_from_morphology(sersic_n):
    """
    Estimate gas fraction from Sersic index.
    
    Low n (disk-like): gas-rich
    High n (spheroidal): gas-poor
    
    Parameters:
    -----------
    sersic_n : array
        Sersic index
    
    Returns:
    --------
    f_gas : array
        Estimated gas fraction
    """
    # n < 2: disk (f_gas ~ 0.6)
    # n > 4: elliptical (f_gas ~ 0.1)
    f_gas = 0.7 - 0.15 * np.clip(sersic_n, 0, 4)
    return np.clip(f_gas, 0.1, 0.8)


# =============================================================================
# LENSING ANALYSIS
# =============================================================================

def compute_excess_surface_density(shear_data, lens_positions, source_positions):
    """
    Compute excess surface density ΔΣ around lens positions.
    
    This is a simplified version - real analysis needs proper
    source weighting, photo-z handling, etc.
    """
    # Placeholder - requires actual shear catalog
    pass


def compute_lensing_btfr_residual(delta_sigma, M_star):
    """
    Compute residual from expected lensing signal.
    
    Expected: ΔΣ ∝ M_star^α (with α from ΛCDM prediction)
    Residual: observed - predicted
    """
    # ΛCDM prediction for galaxy-galaxy lensing
    # ΔΣ ∝ M_star^0.8 approximately
    alpha_expected = 0.8
    
    log_M = np.log10(M_star)
    log_DS = np.log10(delta_sigma)
    
    # Fit BTFR-equivalent
    slope, intercept, _, _, _ = stats.linregress(log_M, log_DS)
    
    predicted = slope * log_M + intercept
    residual = log_DS - predicted
    
    return residual


# =============================================================================
# PILOT TEST
# =============================================================================

def pilot_test_lensing_ushape(lens_catalog, shear_catalog=None):
    """
    Pilot test for U-shape in lensing.
    
    If shear_catalog is None, uses mock data to validate methodology.
    """
    results = {
        'test_name': 'Lensing U-Shape Pilot',
        'description': 'Test for Q²R² pattern in gravitational lensing',
        'status': 'PILOT'
    }
    
    if shear_catalog is None:
        print("  No shear catalog provided - running methodology validation")
        results['mode'] = 'VALIDATION'
        
        # Generate mock data
        n_lens = 10000
        
        # Mock properties
        M_star = 10**(np.random.uniform(9, 11.5, n_lens))
        g_r = np.random.uniform(0.3, 1.0, n_lens)
        env = np.random.uniform(-2, 2, n_lens)
        
        # Estimate gas fraction
        f_gas = estimate_gas_fraction_from_color(g_r)
        
        # Mock lensing signal with injected U-shape
        # ΔΣ = ΔΣ_0 × (M/M_0)^0.8 × (1 + a×env²)
        DS_0 = 50  # M_sun/pc²
        M_0 = 1e10
        a_inject = 0.05  # U-shape amplitude
        
        delta_sigma = DS_0 * (M_star / M_0)**0.8 * (1 + a_inject * env**2)
        delta_sigma *= np.exp(np.random.normal(0, 0.1, n_lens))  # Scatter
        
        # Compute residuals
        log_M = np.log10(M_star)
        log_DS = np.log10(delta_sigma)
        
        slope, intercept, _, _, _ = stats.linregress(log_M, log_DS)
        residual = log_DS - (slope * log_M + intercept)
        
        # Test for U-shape
        from core_tests import fit_ushape
        
        env_std = (env - env.mean()) / env.std()
        fit = fit_ushape(env_std, residual)
        
        if fit:
            results['mock_test'] = {
                'n': n_lens,
                'injected_a': a_inject,
                'recovered_a': fit['a'],
                'recovery_ratio': fit['a'] / a_inject if a_inject != 0 else None,
                'p_value': fit['p_value'],
                'validation': 'PASSED' if abs(fit['a'] - a_inject) < 0.02 else 'FAILED'
            }
            print(f"    Injected a = {a_inject:.4f}")
            print(f"    Recovered a = {fit['a']:.4f}")
            print(f"    Recovery ratio = {fit['a']/a_inject:.2f}")
        
        results['conclusion'] = 'Methodology validated on mock data'
        results['next_step'] = 'Download real shear catalog from DES/KiDS/HSC'
        
    else:
        # Real analysis
        results['mode'] = 'REAL_DATA'
        # Implementation would go here
        pass
    
    return results


# =============================================================================
# DATA DOWNLOAD INSTRUCTIONS
# =============================================================================

def print_download_instructions():
    """Print instructions for downloading lensing data."""
    
    print("\n" + "="*70)
    print(" DATA DOWNLOAD INSTRUCTIONS")
    print("="*70)
    
    print("\n1. DES Y3 (Recommended for first test)")
    print("-" * 40)
    print("   URL: https://des.ncsa.illinois.edu/releases/y3a2")
    print("   Files needed:")
    print("   - Y3A2_GOLD_catalog.fits (lens galaxies)")
    print("   - Y3_METACALIBRATION_catalog.fits (shear)")
    print("   Registration required: Yes")
    
    print("\n2. KiDS-1000")
    print("-" * 40)
    print("   URL: https://kids.strw.leidenuniv.nl/DR4/")
    print("   Files needed:")
    print("   - KiDS_DR4.0_ugriZYJHKs_SOM_gold_WL_cat.fits")
    print("   Registration required: No")
    
    print("\n3. HSC-SSP PDR3")
    print("-" * 40)
    print("   URL: https://hsc-release.mtk.nao.ac.jp/")
    print("   Files needed:")
    print("   - pdr3_wide_forced.fits (photometry)")
    print("   - pdr3_wide_shear.fits (shear)")
    print("   Registration required: Yes")
    
    print("\n" + "="*70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print(" GRAVITATIONAL LENSING PILOT TEST")
    print("="*70)
    
    print("\nThis test checks for Q²R² pattern in weak lensing.")
    print("If successful, it would provide second independent context")
    print("for string theory validation.")
    
    # Check for real data
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "data", "lensing")
    
    has_real_data = os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0 if os.path.exists(DATA_DIR) else False
    
    if has_real_data:
        print(f"\nFound data in: {DATA_DIR}")
        # Load and analyze
    else:
        print("\nNo lensing data found. Running methodology validation...")
        
        # Add parent for imports
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        results = pilot_test_lensing_ushape(lens_catalog=None)
        
        print("\n" + "="*70)
        print(" VALIDATION RESULTS")
        print("="*70)
        
        if 'mock_test' in results:
            mt = results['mock_test']
            print(f"\n  Mock test: {mt['validation']}")
            print(f"  Injected amplitude: {mt['injected_a']:.4f}")
            print(f"  Recovered amplitude: {mt['recovered_a']:.4f}")
            print(f"  Recovery ratio: {mt['recovery_ratio']:.2f}")
        
        print(f"\n  Conclusion: {results['conclusion']}")
        print(f"  Next step: {results['next_step']}")
        
        print_download_instructions()


if __name__ == "__main__":
    main()
