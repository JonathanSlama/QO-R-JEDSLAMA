#!/usr/bin/env python3
"""
===============================================================================
MASS SCALING TEST
===============================================================================

Test the mass dependence of the U-shape amplitude.

String theory predicts:
    a(M_star) ∝ (M_star)^α × f(Calabi-Yau geometry)

This test bins galaxies by stellar mass and measures the U-shape amplitude
in each bin to determine the scaling exponent α.

Author: Jonathan Édouard Slama
Date: December 2025
===============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import json

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_tests import prepare_data, fit_ushape

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to BTFR directory (relative to Git folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
GIT_DIR = os.path.dirname(PROJECT_DIR)
BTFR_DIR = os.path.join(os.path.dirname(GIT_DIR), "BTFR")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          "experimental", "04_RESULTS", "test_results")

# =============================================================================
# MASS SCALING ANALYSIS
# =============================================================================

def test_mass_scaling(datasets, n_bins=5, min_per_bin=500):
    """
    Test mass dependence of U-shape amplitude.
    
    Parameters:
    -----------
    datasets : dict
        Dictionary of DataFrames
    n_bins : int
        Number of mass bins
    min_per_bin : int
        Minimum galaxies per bin
    
    Returns:
    --------
    dict : Results including scaling exponent
    """
    results = {
        'test_name': 'Mass Scaling',
        'description': 'Test stellar mass dependence of U-shape amplitude',
        'theoretical_prediction': 'a ∝ M_star^α with α predicted by string theory',
        'datasets': {}
    }
    
    for name, df in datasets.items():
        if name.startswith('TNG'):
            continue
        
        df = prepare_data(df, name)
        
        # Check required columns
        if 'log_Mstar' not in df.columns:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'No stellar mass'}
            continue
        
        if 'env_std' not in df.columns or 'btfr_residual' not in df.columns:
            results['datasets'][name] = {'status': 'SKIPPED', 'reason': 'Missing columns'}
            continue
        
        # Clean data
        mask = (np.isfinite(df['log_Mstar']) & 
                np.isfinite(df['env_std']) & 
                np.isfinite(df['btfr_residual']))
        df_clean = df[mask].copy()
        
        if len(df_clean) < n_bins * min_per_bin:
            results['datasets'][name] = {
                'status': 'SKIPPED', 
                'reason': f'Insufficient data (N={len(df_clean)}, need {n_bins * min_per_bin})'
            }
            continue
        
        # Create mass bins (equal count)
        df_clean['mass_bin'] = pd.qcut(df_clean['log_Mstar'], n_bins, labels=False)
        
        # Analyze each bin
        bin_results = []
        
        for i in range(n_bins):
            bin_data = df_clean[df_clean['mass_bin'] == i]
            
            if len(bin_data) < min_per_bin:
                continue
            
            # Fit U-shape
            fit = fit_ushape(bin_data['env_std'].values, bin_data['btfr_residual'].values)
            
            if fit is None:
                continue
            
            # Record
            bin_results.append({
                'bin': i,
                'n': len(bin_data),
                'log_Mstar_min': float(bin_data['log_Mstar'].min()),
                'log_Mstar_max': float(bin_data['log_Mstar'].max()),
                'log_Mstar_mean': float(bin_data['log_Mstar'].mean()),
                'a': fit['a'],
                'a_err': fit['a_err'],
                'p_value': fit['p_value']
            })
        
        if len(bin_results) < 3:
            results['datasets'][name] = {'status': 'FAILED', 'reason': 'Too few valid bins'}
            continue
        
        # Fit scaling relation: log(a) = α × log(M_star) + const
        # Or equivalently: a = a_0 × M_star^α
        log_M = np.array([b['log_Mstar_mean'] for b in bin_results])
        a_values = np.array([b['a'] for b in bin_results])
        a_errors = np.array([b['a_err'] for b in bin_results])
        
        # Only use positive a values for log fit
        positive_mask = a_values > 0
        
        if positive_mask.sum() < 3:
            results['datasets'][name] = {
                'status': 'PARTIAL',
                'reason': 'Not enough positive a values for scaling fit',
                'bins': bin_results,
                'n_positive_bins': int(positive_mask.sum())
            }
            continue
        
        log_a = np.log10(a_values[positive_mask])
        log_M_pos = log_M[positive_mask]
        
        # Linear fit: log(a) = α × log(M) + β
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_M_pos, log_a)
            
            # Scaling exponent
            alpha = slope
            alpha_err = std_err
            
            # Reference amplitude at log(M_star) = 10
            a_0 = 10**(intercept + alpha * 10)
            
            results['datasets'][name] = {
                'n': len(df_clean),
                'n_bins': len(bin_results),
                'bins': bin_results,
                'scaling_exponent_alpha': float(alpha),
                'scaling_exponent_err': float(alpha_err),
                'scaling_p_value': float(p_value),
                'scaling_r_squared': float(r_value**2),
                'a_0_at_logM10': float(a_0),
                'interpretation': interpret_scaling(alpha, alpha_err),
                'status': 'SUCCESS'
            }
            
        except Exception as e:
            results['datasets'][name] = {
                'status': 'ERROR',
                'reason': str(e),
                'bins': bin_results
            }
    
    # Overall summary
    successful = [d for d in results['datasets'].values() 
                  if isinstance(d, dict) and d.get('status') == 'SUCCESS']
    
    if successful:
        avg_alpha = np.mean([d['scaling_exponent_alpha'] for d in successful])
        results['summary'] = f"Mass scaling: α = {avg_alpha:.3f}"
        results['passed'] = True
    else:
        results['summary'] = "Mass scaling: Could not determine"
        results['passed'] = False
    
    return results


def interpret_scaling(alpha, alpha_err):
    """Interpret the scaling exponent."""
    if abs(alpha) < 2 * alpha_err:
        return "No significant mass dependence (α ≈ 0)"
    elif alpha > 0:
        return f"Positive scaling: stronger U-shape at higher mass (α = {alpha:.3f})"
    else:
        return f"Negative scaling: stronger U-shape at lower mass (α = {alpha:.3f})"


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_mass_scaling(results, output_path=None):
    """Plot mass scaling results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, data in results['datasets'].items():
        if not isinstance(data, dict) or 'bins' not in data:
            continue
        
        bins = data['bins']
        
        # Left: U-shape amplitude vs mass
        ax1 = axes[0]
        log_M = [b['log_Mstar_mean'] for b in bins]
        a_vals = [b['a'] for b in bins]
        a_errs = [b['a_err'] for b in bins]
        
        ax1.errorbar(log_M, a_vals, yerr=a_errs, fmt='o-', label=name, capsize=3)
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('log(M★/M☉)')
        ax1.set_ylabel('U-shape amplitude a')
        ax1.set_title('U-shape Amplitude vs Stellar Mass')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right: Log-log scaling
        ax2 = axes[1]
        positive_mask = np.array(a_vals) > 0
        if positive_mask.sum() >= 2:
            log_M_pos = np.array(log_M)[positive_mask]
            log_a = np.log10(np.array(a_vals)[positive_mask])
            
            ax2.scatter(log_M_pos, log_a, s=50, label=name)
            
            # Fit line
            if data.get('status') == 'SUCCESS':
                alpha = data['scaling_exponent_alpha']
                # Reconstruct fit line
                x_fit = np.linspace(min(log_M_pos), max(log_M_pos), 100)
                y_fit = alpha * (x_fit - 10) + np.log10(data['a_0_at_logM10'])
                ax2.plot(x_fit, y_fit, '--', label=f'α = {alpha:.3f}')
        
        ax2.set_xlabel('log(M★/M☉)')
        ax2.set_ylabel('log(a)')
        ax2.set_title('Log-Log Scaling')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print(" MASS SCALING TEST")
    print("="*70)
    
    # Load data
    print("\nLoading datasets...")
    
    datasets = {}
    
    # ALFALFA (corrected)
    alfalfa_path = os.path.join(BTFR_DIR, "Test-Replicability", "data", "alfalfa_corrected.csv")
    if os.path.exists(alfalfa_path):
        datasets['ALFALFA'] = pd.read_csv(alfalfa_path)
        print(f"  ALFALFA: {len(datasets['ALFALFA'])} galaxies")

    # SPARC
    sparc_path = os.path.join(BTFR_DIR, "Data", "processed", "sparc_with_environment.csv")
    if os.path.exists(sparc_path):
        datasets['SPARC'] = pd.read_csv(sparc_path)
        print(f"  SPARC: {len(datasets['SPARC'])} galaxies")
    
    if not datasets:
        print("ERROR: No datasets found!")
        return
    
    # Run test
    print("\nRunning mass scaling analysis...")
    results = test_mass_scaling(datasets)
    
    # Print results
    print("\n" + "="*70)
    print(" RESULTS")
    print("="*70)
    
    for name, data in results['datasets'].items():
        print(f"\n{name}:")
        if isinstance(data, dict):
            if data.get('status') == 'SUCCESS':
                print(f"  Scaling exponent α = {data['scaling_exponent_alpha']:.4f} ± {data['scaling_exponent_err']:.4f}")
                print(f"  R² = {data['scaling_r_squared']:.3f}")
                print(f"  p-value = {data['scaling_p_value']:.4f}")
                print(f"  Interpretation: {data['interpretation']}")
            else:
                print(f"  Status: {data.get('status', 'UNKNOWN')}")
                print(f"  Reason: {data.get('reason', 'N/A')}")
    
    print(f"\n{results['summary']}")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "mass_scaling_results.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {output_file}")
    
    # Plot
    fig_path = os.path.join(os.path.dirname(OUTPUT_DIR), "..", "figures", "mass_scaling.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plot_mass_scaling(results, fig_path)


if __name__ == "__main__":
    main()
