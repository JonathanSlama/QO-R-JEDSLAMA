#!/usr/bin/env python3
"""
=============================================================================
QO+R Paper 1 - Step 06: Replicability Tests (REAL DATA VERSION)
=============================================================================
Testing the U-shape on INDEPENDENT datasets with REAL OBSERVATIONAL DATA:

1. SPARC (reference): 175 galaxies with rotation curves
2. ALFALFA (replication): ~21,834 HI-selected galaxies from α.100 catalog
3. Little THINGS (replication): ~40 dwarf irregular galaxies

THIS VERSION USES ONLY REAL DATA - NO SYNTHETIC GENERATION.

Results from execution (2025-12-03):
- SPARC: a = 1.3606 ± 0.2368, p < 0.000001 → U-SHAPE CONFIRMED
- ALFALFA: a = 0.0701 ± 0.0282, p = 0.0065 → U-SHAPE REPLICATED  
- Little THINGS: a = 0.2891 ± 0.3242, p = 0.186 → Not significant (small N)

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
import warnings
warnings.filterwarnings('ignore')

def get_project_root():
    return Path(__file__).parent.parent.parent

def get_data_root():
    """Get path to Test-Replicability data folder"""
    # Navigate from Git/Paper1-BTFR-UShape/tests/06_replicability to BTFR/Test-Replicability/data
    return Path(__file__).parent.parent.parent.parent.parent / "BTFR" / "Test-Replicability" / "data"

def load_sparc_data():
    """Load SPARC data (reference dataset)"""
    data_path = get_project_root() / "data" / "sparc_with_environment.csv"
    
    df = pd.read_csv(data_path)
    
    # Compute BTFR residual if not present
    if 'btfr_residual' not in df.columns:
        a_btfr, b_btfr = -1.0577, 0.3442
        df['btfr_residual'] = df['log_Vflat'] - (a_btfr + b_btfr * df['log_Mbar'])
    
    print(f"[SPARC] Loaded {len(df)} galaxies (REAL DATA)")
    print(f"        Environments: {df['env_class'].value_counts().to_dict()}")
    
    return df

def load_alfalfa_data():
    """
    Load REAL ALFALFA data from α.100 catalog.
    
    ALFALFA = Arecibo Legacy Fast ALFA survey
    HI-selected galaxy catalog covering ~7000 deg² 
    Reference: Haynes et al. (2018)
    """
    data_path = get_data_root() / "alfalfa.csv"
    
    if not data_path.exists():
        print(f"[ERROR] ALFALFA data not found at {data_path}")
        print(f"        Expected location: BTFR/Test-Replicability/data/alfalfa.csv")
        return None
    
    df = pd.read_csv(data_path)
    
    print(f"[ALFALFA] Loaded {len(df):,} galaxies (REAL DATA)")
    print(f"          Source: α.100 catalog (Haynes et al. 2018)")
    print(f"          Environments: {df['env_class'].value_counts().to_dict()}")
    
    return df

def load_little_things_data():
    """
    Load REAL Little THINGS data.
    
    LITTLE THINGS = Local Irregulars That Trace Luminosity Extremes, 
    The HI Nearby Galaxy Survey
    Reference: Hunter et al. (2012)
    
    High-resolution HI observations of ~40 nearby dwarf irregular galaxies.
    """
    data_path = get_data_root() / "little_things.csv"
    
    if not data_path.exists():
        print(f"[ERROR] Little THINGS data not found at {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    
    print(f"[Little THINGS] Loaded {len(df)} galaxies (REAL DATA)")
    print(f"                Source: Hunter et al. (2012)")
    print(f"                Environments: {df['env_class'].value_counts().to_dict()}")
    
    return df

def quadratic(x, a, b, c):
    """Quadratic function for U-shape fitting"""
    return a * x**2 + b * x + c

def test_ushape(df, name):
    """Test for U-shape in a dataset using quadratic regression."""
    
    x = df['density_proxy'].values
    y = df['btfr_residual'].values
    
    # Remove NaN
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    
    if len(x) < 30:
        print(f"  [!] Not enough data points: {len(x)}")
        return None
    
    try:
        # Fit quadratic
        popt, pcov = curve_fit(quadratic, x, y, p0=[0.1, -0.1, 0])
        perr = np.sqrt(np.diag(pcov))
        
        a, b, c = popt
        a_err = perr[0]
        
        # Statistical significance (one-sided test for a > 0)
        z_score = a / a_err if a_err > 0 else 0
        p_value = 1 - stats.norm.cdf(z_score) if z_score > 0 else 1
        
        # R-squared
        y_pred = quadratic(x, *popt)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # Linear comparison for AIC
        popt_lin, _ = curve_fit(lambda x, m, c: m*x + c, x, y)
        y_pred_lin = popt_lin[0]*x + popt_lin[1]
        ss_res_lin = np.sum((y - y_pred_lin)**2)
        
        n = len(x)
        aic_quad = n * np.log(ss_res/n) + 2*3  # 3 params
        aic_lin = n * np.log(ss_res_lin/n) + 2*2  # 2 params
        delta_aic = aic_lin - aic_quad
        
        return {
            'name': name,
            'n': len(x),
            'a': a,
            'a_err': a_err,
            'b': b,
            'c': c,
            'z_score': z_score,
            'p_value': p_value,
            'r_squared': r_squared,
            'delta_aic': delta_aic,
            'ushape': a > 0 and p_value < 0.05,
            'popt': popt
        }
    except Exception as e:
        print(f"  [!] Fit failed for {name}: {e}")
        return None

def run_replicability_tests():
    """Run replicability tests on REAL datasets."""
    
    print("=" * 70)
    print("REPLICABILITY TESTS: REAL OBSERVATIONAL DATA")
    print("=" * 70)
    print("\nNOTE: All datasets use REAL observational data.")
    print("      No synthetic or simulated data is used.\n")
    
    results = {}
    datasets = {}
    
    # 1. SPARC (reference)
    print("-" * 50)
    print("Dataset 1: SPARC (Reference)")
    print("-" * 50)
    df_sparc = load_sparc_data()
    datasets['SPARC'] = df_sparc
    results['SPARC'] = test_ushape(df_sparc, 'SPARC')
    
    if results['SPARC']:
        r = results['SPARC']
        print(f"\n  N = {r['n']} galaxies")
        print(f"  a = {r['a']:.4f} ± {r['a_err']:.4f}")
        print(f"  Z-score = {r['z_score']:.2f}, p = {r['p_value']:.6f}")
        print(f"  R² = {r['r_squared']:.4f}")
        print(f"  ΔAIC (linear - quadratic) = {r['delta_aic']:.1f}")
        print(f"  U-shape: {'✓ DETECTED' if r['ushape'] else '✗ Not detected'}")
    
    # 2. ALFALFA (replication)
    print("\n" + "-" * 50)
    print("Dataset 2: ALFALFA (Replication)")
    print("-" * 50)
    df_alfalfa = load_alfalfa_data()
    if df_alfalfa is not None:
        datasets['ALFALFA'] = df_alfalfa
        results['ALFALFA'] = test_ushape(df_alfalfa, 'ALFALFA')
        
        if results['ALFALFA']:
            r = results['ALFALFA']
            print(f"\n  N = {r['n']:,} galaxies")
            print(f"  a = {r['a']:.4f} ± {r['a_err']:.4f}")
            print(f"  Z-score = {r['z_score']:.2f}, p = {r['p_value']:.6f}")
            print(f"  R² = {r['r_squared']:.4f}")
            print(f"  ΔAIC (linear - quadratic) = {r['delta_aic']:.1f}")
            print(f"  U-shape: {'✓ REPLICATED' if r['ushape'] else '✗ Not replicated'}")
    
    # 3. Little THINGS (replication)
    print("\n" + "-" * 50)
    print("Dataset 3: Little THINGS (Replication)")
    print("-" * 50)
    df_things = load_little_things_data()
    if df_things is not None:
        datasets['Little_THINGS'] = df_things
        results['Little_THINGS'] = test_ushape(df_things, 'Little_THINGS')
        
        if results['Little_THINGS']:
            r = results['Little_THINGS']
            print(f"\n  N = {r['n']} galaxies")
            print(f"  a = {r['a']:.4f} ± {r['a_err']:.4f}")
            print(f"  Z-score = {r['z_score']:.2f}, p = {r['p_value']:.6f}")
            print(f"  R² = {r['r_squared']:.4f}")
            print(f"  ΔAIC (linear - quadratic) = {r['delta_aic']:.1f}")
            print(f"  U-shape: {'✓ REPLICATED' if r['ushape'] else '✗ Not replicated (small N)'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("REPLICABILITY SUMMARY")
    print("=" * 70)
    
    n_ushape = sum(1 for r in results.values() if r and r['ushape'])
    n_total = sum(1 for r in results.values() if r)
    
    print(f"\n  Datasets with significant U-shape (p < 0.05): {n_ushape}/{n_total}")
    print()
    
    # Table
    print(f"  {'Dataset':<15} {'N':>8} {'a':>10} {'±err':>8} {'p-value':>12} {'Result':>15}")
    print(f"  {'-'*68}")
    
    for name, r in results.items():
        if r:
            status = "✓ U-shape" if r['ushape'] else "✗ No"
            print(f"  {name:<15} {r['n']:>8,} {r['a']:>+10.4f} {r['a_err']:>8.4f} {r['p_value']:>12.6f} {status:>15}")
    
    return results, datasets

def create_replicability_figure(results, datasets):
    """Create publication-quality replicability figure."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    env_colors = {
        'void': '#2ecc71',
        'field': '#3498db', 
        'group': '#f39c12',
        'cluster': '#e74c3c'
    }
    
    dataset_list = [
        ('SPARC', datasets.get('SPARC'), results.get('SPARC')),
        ('ALFALFA', datasets.get('ALFALFA'), results.get('ALFALFA')),
        ('Little THINGS', datasets.get('Little_THINGS'), results.get('Little_THINGS'))
    ]
    
    # Panels A, B, C: Individual datasets
    for idx, (name, df, res) in enumerate(dataset_list):
        if df is None or res is None:
            continue
            
        ax = axes[idx // 2, idx % 2]
        
        # Plot by environment
        for env in ['void', 'field', 'group', 'cluster']:
            if 'env_class' in df.columns:
                mask = df['env_class'] == env
            else:
                continue
            if mask.sum() > 0:
                ax.scatter(df.loc[mask, 'density_proxy'], 
                          df.loc[mask, 'btfr_residual'],
                          c=env_colors.get(env, 'gray'), 
                          label=f"{env.capitalize()} (N={mask.sum()})", 
                          alpha=0.4 if len(df) > 1000 else 0.7, 
                          s=15 if len(df) > 1000 else 50)
        
        # Fit line
        if res and res['popt'] is not None:
            x_fit = np.linspace(df['density_proxy'].min(), df['density_proxy'].max(), 100)
            y_fit = quadratic(x_fit, *res['popt'])
            ax.plot(x_fit, y_fit, 'k-', linewidth=2.5, label='Quadratic fit')
        
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('Density Proxy', fontsize=11)
        ax.set_ylabel('BTFR Residual', fontsize=11)
        
        status = "✓" if res and res['ushape'] else "✗"
        data_type = "REAL DATA"
        ax.set_title(f'{chr(65+idx)}) {name} (N={len(df):,}) - {data_type} {status}', 
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Stats box
        if res:
            text = f"a = {res['a']:.4f} ± {res['a_err']:.4f}\np = {res['p_value']:.4f}\nΔAIC = {res['delta_aic']:.1f}"
            ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Panel D: Coefficient comparison
    ax = axes[1, 1]
    
    valid_results = [(name, r) for name, r in results.items() if r is not None]
    names = [name for name, _ in valid_results]
    a_values = [r['a'] for _, r in valid_results]
    a_errors = [r['a_err'] for _, r in valid_results]
    n_values = [r['n'] for _, r in valid_results]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(names)]
    x_pos = range(len(names))
    
    bars = ax.bar(x_pos, a_values, yerr=a_errors, color=colors,
                  edgecolor='black', capsize=10, alpha=0.8, linewidth=2)
    
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{n}\n(N={nv:,})" for n, nv in zip(names, n_values)])
    ax.set_ylabel('Quadratic Coefficient (a)', fontsize=12)
    ax.set_title('D) Cross-Dataset Comparison (REAL DATA ONLY)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Significance markers
    for i, (a, err, r) in enumerate(zip(a_values, a_errors, [r for _, r in valid_results])):
        if r['ushape']:
            ax.text(i, a + err + 0.02, '★\np<0.05', ha='center', fontsize=10, 
                   color='gold', fontweight='bold')
    
    plt.tight_layout()
    
    # Save to figures directory
    figures_dir = get_project_root() / "figures"
    figures_dir.mkdir(exist_ok=True)
    output_path = figures_dir / "fig06_replicability.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Figure saved: {output_path}")
    
    plt.close()
    
    return fig

def save_results_csv(results):
    """Save results to CSV for reproducibility"""
    
    rows = []
    for name, r in results.items():
        if r:
            rows.append({
                'Dataset': name,
                'N_galaxies': r['n'],
                'Data_type': 'REAL',
                'a_coefficient': r['a'],
                'a_error': r['a_err'],
                'z_score': r['z_score'],
                'p_value': r['p_value'],
                'r_squared': r['r_squared'],
                'delta_AIC': r['delta_aic'],
                'U_shape_detected': r['ushape']
            })
    
    df_results = pd.DataFrame(rows)
    output_path = Path(__file__).parent / "replicability_REAL_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"✓ Results saved: {output_path}")
    
    return df_results

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  QO+R PAPER 1 - REPLICABILITY TESTS (REAL DATA VERSION)             ║
    ║                                                                      ║
    ║  This script uses ONLY real observational data:                      ║
    ║  • SPARC: Rotation curve sample (Lelli et al. 2016)                  ║
    ║  • ALFALFA: α.100 HI catalog (Haynes et al. 2018)                   ║
    ║  • Little THINGS: Dwarf irregulars (Hunter et al. 2012)              ║
    ║                                                                      ║
    ║  NO SYNTHETIC DATA IS GENERATED OR USED.                             ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run tests
    results, datasets = run_replicability_tests()
    
    # Create figure
    print("\nGenerating replicability figure...")
    create_replicability_figure(results, datasets)
    
    # Save CSV
    save_results_csv(results)
    
    # Final conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION: REPLICABILITY ON REAL DATA")
    print("=" * 70)
    
    n_confirmed = sum(1 for r in results.values() if r and r['ushape'])
    n_total = sum(1 for r in results.values() if r)
    
    print(f"""
    DATASETS ANALYZED: {n_total}
    U-SHAPE CONFIRMED: {n_confirmed}/{n_total}
    
    Summary:
    """)
    
    for name, r in results.items():
        if r:
            status = "✓ U-SHAPE CONFIRMED" if r['ushape'] else "✗ Not significant"
            print(f"    • {name}: a = {r['a']:.4f}, p = {r['p_value']:.6f} → {status}")
    
    if n_confirmed >= 2:
        print(f"""
    ═══════════════════════════════════════════════════════════════════════
    REPLICABILITY ESTABLISHED: The U-shape pattern is detected in {n_confirmed}
    independent observational datasets, confirming it is not an artifact
    of the SPARC sample.
    
    NOTE ON AMPLITUDE DIFFERENCE:
    The coefficient 'a' differs significantly between SPARC (a~1.36) and 
    ALFALFA (a~0.07). This ~20x difference may be due to:
    1. Different velocity measures (Vflat vs W50)
    2. Selection effects (optical+HI vs HI-only)
    3. Environmental coverage differences
    
    This amplitude difference should be discussed in the paper.
    ═══════════════════════════════════════════════════════════════════════
    """)
    else:
        print(f"""
    Note: Limited replicability ({n_confirmed}/{n_total} datasets).
    Further analysis with additional datasets recommended.
    """)
    
    return results

if __name__ == "__main__":
    results = main()
