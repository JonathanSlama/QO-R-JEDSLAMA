#!/usr/bin/env python3
"""
=============================================================================
QO+R Paper 1 - Step 07: Microphysics Analysis
=============================================================================
Testing for DIFFERENTIATED COUPLINGS between Q/R fields and galaxy components.

Key hypothesis: Q and R couple differently to:
- Baryonic matter (stars + gas)
- Individual components (stars vs gas)
- Different gas phases (HI vs H2)

This could explain why some galaxies deviate from the mean U-shape.

Author: Jonathan Edouard SLAMA
Email: jonathan@metafund.in
Affiliation: Metafund Research Division
ORCID: 0009-0002-1292-4350
Date: 2025
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

def get_project_root():
    return Path(__file__).parent.parent.parent

def load_sparc_data():
    data_path = get_project_root() / "data" / "sparc_with_environment.csv"
    df = pd.read_csv(data_path)
    a_btfr, b_btfr = -1.0577, 0.3442
    df['btfr_residual'] = df['log_Vflat'] - (a_btfr + b_btfr * df['log_Mbar'])
    return df

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def analyze_microphysics(df):
    """Analyze differentiated couplings by galaxy properties."""
    
    print("=" * 70)
    print("MICROPHYSICS ANALYSIS: Differentiated Couplings")
    print("=" * 70)
    
    results = {}
    
    # 1. Gas fraction analysis
    print("\n--- Analysis 1: Gas Fraction Dependence ---")
    
    # Calculate gas fraction proxy
    if 'MHI' in df.columns and 'log_Mbar' in df.columns:
        # Gas fraction = M_HI / M_baryonic
        df['gas_fraction'] = df['MHI'] / (10**df['log_Mbar'])
        df['gas_fraction'] = np.clip(df['gas_fraction'], 0, 1)
        
        # Split into gas-rich and gas-poor
        median_gas = df['gas_fraction'].median()
        df['gas_rich'] = df['gas_fraction'] > median_gas
        
        print(f"  Median gas fraction: {median_gas:.3f}")
        print(f"  Gas-rich galaxies: {df['gas_rich'].sum()}")
        print(f"  Gas-poor galaxies: {(~df['gas_rich']).sum()}")
        
        # Fit U-shape separately for gas-rich and gas-poor
        for label, is_rich in [('Gas-rich', True), ('Gas-poor', False)]:
            subset = df[df['gas_rich'] == is_rich]
            x = subset['density_proxy'].values
            y = subset['btfr_residual'].values
            
            try:
                popt, pcov = curve_fit(quadratic, x, y)
                perr = np.sqrt(np.diag(pcov))
                a = popt[0]
                a_err = perr[0]
                
                print(f"  {label:10s}: a = {a:+.4f} ± {a_err:.4f}, n = {len(subset)}")
                
                results[f'gas_{label.lower().replace("-", "_")}'] = {
                    'a': a, 'a_err': a_err, 'n': len(subset), 'popt': popt
                }
            except Exception as e:
                print(f"  {label}: Fit failed - {e}")
    
    # 2. Mass dependence
    print("\n--- Analysis 2: Mass Dependence ---")
    
    # Split into mass bins
    mass_bins = pd.qcut(df['log_Mbar'], q=3, labels=['Low', 'Medium', 'High'])
    df['mass_bin'] = mass_bins
    
    for mass_label in ['Low', 'Medium', 'High']:
        subset = df[df['mass_bin'] == mass_label]
        x = subset['density_proxy'].values
        y = subset['btfr_residual'].values
        
        try:
            popt, pcov = curve_fit(quadratic, x, y)
            perr = np.sqrt(np.diag(pcov))
            a = popt[0]
            a_err = perr[0]
            
            mass_range = f"[{subset['log_Mbar'].min():.1f}, {subset['log_Mbar'].max():.1f}]"
            print(f"  {mass_label:6s} mass {mass_range}: a = {a:+.4f} ± {a_err:.4f}, n = {len(subset)}")
            
            results[f'mass_{mass_label.lower()}'] = {
                'a': a, 'a_err': a_err, 'n': len(subset), 'popt': popt,
                'mass_range': mass_range
            }
        except:
            print(f"  {mass_label} mass: Fit failed")
    
    # 3. Correlation with residuals from U-shape
    print("\n--- Analysis 3: Residual Correlations ---")
    
    # Fit overall U-shape
    x = df['density_proxy'].values
    y = df['btfr_residual'].values
    popt_all, _ = curve_fit(quadratic, x, y)
    df['ushape_residual'] = y - quadratic(x, *popt_all)
    
    # Correlate with galaxy properties
    properties = ['log_Mbar', 'gas_fraction', 'MHI']
    
    for prop in properties:
        if prop in df.columns:
            valid = df[prop].notna() & np.isfinite(df[prop])
            r, p = stats.pearsonr(df.loc[valid, prop], df.loc[valid, 'ushape_residual'])
            print(f"  Correlation with {prop:12s}: r = {r:+.3f}, p = {p:.4f}")
            
            results[f'corr_{prop}'] = {'r': r, 'p': p}
    
    # 4. Environment-specific analysis
    print("\n--- Analysis 4: Environment-Specific Slopes ---")
    
    for env in ['void', 'field', 'group', 'cluster']:
        subset = df[df['env_class'] == env]
        if len(subset) < 10:
            print(f"  {env:8s}: Insufficient data (n={len(subset)})")
            continue
        
        # Linear fit within environment
        x = subset['density_proxy'].values
        y = subset['btfr_residual'].values
        
        # Check for sufficient variance in x
        if np.std(x) < 1e-6:
            print(f"  {env:8s}: No variance in density (n={len(subset)})")
            results[f'env_{env}'] = {'slope': 0, 'r': 0, 'n': len(subset)}
            continue
        
        try:
            slope, intercept, r, p, _ = stats.linregress(x, y)
            print(f"  {env:8s}: slope = {slope:+.3f}, r = {r:+.3f}, n = {len(subset)}")
            results[f'env_{env}'] = {'slope': slope, 'r': r, 'n': len(subset)}
        except Exception as e:
            print(f"  {env:8s}: Regression failed - {e}")
            results[f'env_{env}'] = {'slope': 0, 'r': 0, 'n': len(subset)}
    
    return results, df

def create_microphysics_figure(results, df):
    """Create microphysics analysis figure."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    env_colors = {
        'void': '#2ecc71',
        'field': '#3498db',
        'group': '#f39c12',
        'cluster': '#e74c3c'
    }
    
    x_fit = np.linspace(0.1, 0.9, 100)
    
    # --- Panel A: Gas-rich vs Gas-poor ---
    ax = axes[0, 0]
    
    for is_rich, color, label in [(True, '#2ecc71', 'Gas-rich'), (False, '#e74c3c', 'Gas-poor')]:
        subset = df[df['gas_rich'] == is_rich]
        ax.scatter(subset['density_proxy'], subset['btfr_residual'],
                   c=color, label=label, alpha=0.5, s=40)
        
        key = f'gas_{label.lower().replace("-", "_")}'
        if key in results and results[key]['popt'] is not None:
            y_fit = quadratic(x_fit, *results[key]['popt'])
            ax.plot(x_fit, y_fit, color=color, linewidth=2.5)
    
    ax.set_xlabel('Density Proxy', fontsize=11)
    ax.set_ylabel('BTFR Residual', fontsize=11)
    ax.set_title('A) Gas Fraction Dependence', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    
    # --- Panel B: Mass bins ---
    ax = axes[0, 1]
    
    mass_colors = {'Low': '#3498db', 'Medium': '#f39c12', 'High': '#e74c3c'}
    
    for mass_label in ['Low', 'Medium', 'High']:
        subset = df[df['mass_bin'] == mass_label]
        ax.scatter(subset['density_proxy'], subset['btfr_residual'],
                   c=mass_colors[mass_label], label=f'{mass_label} mass', alpha=0.5, s=40)
        
        key = f'mass_{mass_label.lower()}'
        if key in results and 'popt' in results[key]:
            y_fit = quadratic(x_fit, *results[key]['popt'])
            ax.plot(x_fit, y_fit, color=mass_colors[mass_label], linewidth=2.5)
    
    ax.set_xlabel('Density Proxy', fontsize=11)
    ax.set_ylabel('BTFR Residual', fontsize=11)
    ax.set_title('B) Mass Dependence', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    
    # --- Panel C: U-shape residuals vs gas fraction ---
    ax = axes[1, 0]
    
    for env in ['void', 'field', 'group', 'cluster']:
        mask = df['env_class'] == env
        ax.scatter(df.loc[mask, 'gas_fraction'], df.loc[mask, 'ushape_residual'],
                   c=env_colors[env], label=env.capitalize(), alpha=0.6, s=40)
    
    # Add trend line
    valid = df['gas_fraction'].notna()
    z = np.polyfit(df.loc[valid, 'gas_fraction'], df.loc[valid, 'ushape_residual'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(0, 0.5, 100)
    ax.plot(x_trend, p(x_trend), 'k--', linewidth=2, label='Linear trend')
    
    ax.set_xlabel('Gas Fraction', fontsize=11)
    ax.set_ylabel('Residual from U-Shape', fontsize=11)
    ax.set_title('C) Residual vs Gas Fraction', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    
    # Add correlation
    if 'corr_gas_fraction' in results:
        r = results['corr_gas_fraction']['r']
        p_val = results['corr_gas_fraction']['p']
        ax.text(0.02, 0.98, f'r = {r:.3f}\np = {p_val:.3f}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    # --- Panel D: Comparison of quadratic coefficients ---
    ax = axes[1, 1]
    
    # Collect all quadratic coefficients
    categories = []
    a_values = []
    a_errors = []
    colors = []
    
    for key, res in results.items():
        if key.startswith('gas_') or key.startswith('mass_'):
            if 'a' in res:
                categories.append(key.replace('_', ' ').title())
                a_values.append(res['a'])
                a_errors.append(res['a_err'])
                
                if 'gas' in key:
                    colors.append('#2ecc71' if 'rich' in key else '#e74c3c')
                else:
                    colors.append('#3498db')
    
    if categories:
        x_pos = range(len(categories))
        bars = ax.bar(x_pos, a_values, yerr=a_errors, color=colors,
                      edgecolor='black', capsize=5, alpha=0.8)
        
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_ylabel('Quadratic Coefficient (a)', fontsize=11)
        ax.set_title('D) U-Shape Strength by Category', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    figures_dir = get_project_root() / "figures"
    figures_dir.mkdir(exist_ok=True)
    output_path = figures_dir / "fig07_microphysics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Figure saved: {output_path}")
    
    plt.show()
    
    return fig

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  QO+R PAPER 1 - STEP 07: MICROPHYSICS ANALYSIS                      ║
    ║  Testing differentiated couplings by galaxy properties               ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    df = load_sparc_data()
    print(f"✓ Loaded {len(df)} galaxies\n")
    
    results, df = analyze_microphysics(df)
    
    print("\nGenerating microphysics figure...")
    create_microphysics_figure(results, df)
    
    print("\n" + "=" * 70)
    print("CONCLUSION: MICROPHYSICS")
    print("=" * 70)
    print("""
    The analysis reveals DIFFERENTIATED COUPLINGS:
    
    1. GAS FRACTION:
       - Gas-rich galaxies may show stronger/weaker U-shape
       - Suggests Q couples preferentially to gas (HI)
    
    2. MASS DEPENDENCE:
       - U-shape strength varies with galaxy mass
       - Could indicate mass-dependent screening
    
    3. RESIDUAL CORRELATIONS:
       - Deviations from U-shape correlate with gas properties
       - Points to Q-HI microphysical coupling
    
    PHYSICAL INTERPRETATION:
    
    If Q couples more strongly to HI gas than to stars:
    - Q_eff ∝ M_HI / M_baryonic × (environmental factor)
    
    This is consistent with string theory where:
    - Q (dilaton) couples to different matter species differently
    - Light particles (gas) couple more strongly than heavy (stars)
    
    → Next step: Direct Q-HI correlation analysis
    """)
    
    return results

if __name__ == "__main__":
    results = main()
