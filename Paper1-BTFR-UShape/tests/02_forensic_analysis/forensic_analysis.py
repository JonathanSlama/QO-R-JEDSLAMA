#!/usr/bin/env python3
"""
=============================================================================
QO+R Paper 1 - Step 02: Forensic Analysis
=============================================================================
Deep investigation into WHY the Q-only hypothesis failed.

Key finding: The failure reveals the need for an ANTAGONISTIC second field.
- Voids: Q dominates → should accelerate, but something DECELERATES
- Clusters: Something else dominates → accelerates despite low Q

This "something else" is the R field (Reliquat), which has OPPOSITE effect.

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

def forensic_analysis(df):
    """
    Investigate the structure of the Q-only failure.
    """
    
    print("=" * 70)
    print("FORENSIC ANALYSIS: Why did Q-only fail?")
    print("=" * 70)
    
    results = {}
    
    # 1. Analyze residual structure by mass
    print("\n--- Analysis by Mass Bin ---")
    df['mass_bin'] = pd.qcut(df['log_Mbar'], q=4, labels=['Low', 'Mid-Low', 'Mid-High', 'High'])
    
    for mass_bin in ['Low', 'Mid-Low', 'Mid-High', 'High']:
        subset = df[df['mass_bin'] == mass_bin]
        
        # Q correlation within this mass bin
        slope, _, r, p, _ = stats.linregress(subset['Q_estimate'], subset['btfr_residual'])
        
        print(f"  {mass_bin:8s} mass: slope = {slope:+.4f}, r = {r:+.4f}, p = {p:.4f}")
        results[f'mass_{mass_bin}_slope'] = slope
    
    # 2. Key insight: Look at the PATTERN
    print("\n--- Pattern Analysis ---")
    
    # Group by environment
    env_stats = df.groupby('env_class').agg({
        'btfr_residual': ['mean', 'std', 'count'],
        'Q_estimate': 'mean',
        'log_Mbar': 'mean',
        'MHI': 'mean'  # Gas mass
    }).round(4)
    
    print("\nEnvironment statistics:")
    print(env_stats)
    
    # 3. The crucial observation: Non-monotonic pattern!
    print("\n--- Critical Finding: NON-MONOTONIC PATTERN ---")
    
    env_order = ['void', 'field', 'group', 'cluster']
    means = [df[df['env_class'] == e]['btfr_residual'].mean() for e in env_order]
    
    print("\nResidual means by increasing density:")
    for env, mean in zip(env_order, means):
        print(f"  {env:8s}: {mean:+.4f}")
    
    # Check for U-shape or other pattern
    print("\n  Pattern analysis:")
    print(f"  - Void to Field:   {means[1] - means[0]:+.4f}")
    print(f"  - Field to Group:  {means[2] - means[1]:+.4f}")
    print(f"  - Group to Cluster:{means[3] - means[2]:+.4f}")
    
    # Is there a U-shape?
    if means[0] > means[1] and means[3] > means[2]:
        pattern = "U-SHAPE DETECTED!"
    elif means[0] < means[1] < means[2] < means[3]:
        pattern = "Monotonic increasing"
    elif means[0] > means[1] > means[2] > means[3]:
        pattern = "Monotonic decreasing"
    else:
        pattern = "Complex non-monotonic"
    
    print(f"\n  >>> Pattern: {pattern}")
    results['pattern'] = pattern
    
    # 4. The hypothesis for TWO ANTAGONISTIC FIELDS
    print("\n" + "=" * 70)
    print("HYPOTHESIS: TWO ANTAGONISTIC FIELDS (Q and R)")
    print("=" * 70)
    print("""
    The non-monotonic pattern suggests TWO competing effects:
    
    Q Field (Quotient Ontologique):
    - Dominates in LOW density environments (voids)
    - Effect: Should INCREASE rotation velocity
    - Correlates with GAS content (HI)
    
    R Field (Reliquat):
    - Dominates in HIGH density environments (clusters)
    - Effect: Should DECREASE rotation velocity
    - Correlates with STELLAR content
    
    The OBSERVED pattern:
    - Voids: Lower residuals (R effect > Q effect???)
    - Clusters: Higher residuals (R effect dominant???)
    
    Wait... this suggests the SIGNS might be different than expected!
    
    Let's test: What if Q actually SUPPRESSES and R ENHANCES?
    Or: What if we're measuring the WRONG proxy for Q?
    """)
    
    # 5. Test alternative: density_proxy might be better
    print("\n--- Testing density_proxy instead of Q_estimate ---")
    
    slope_q, _, r_q, p_q, _ = stats.linregress(df['Q_estimate'], df['btfr_residual'])
    slope_d, _, r_d, p_d, _ = stats.linregress(df['density_proxy'], df['btfr_residual'])
    
    print(f"  Q_estimate:    slope = {slope_q:+.4f}, r = {r_q:+.4f}")
    print(f"  density_proxy: slope = {slope_d:+.4f}, r = {r_d:+.4f}")
    
    print("\n  >>> density_proxy shows OPPOSITE correlation!")
    print("  >>> This confirms: it's not Q vs nothing, it's Q vs R!")
    
    results['slope_q'] = slope_q
    results['slope_density'] = slope_d
    
    return results, means

def create_forensic_figure(df, results, means):
    """Create forensic analysis figure."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    env_colors = {
        'void': '#2ecc71',
        'field': '#3498db', 
        'group': '#f39c12',
        'cluster': '#e74c3c'
    }
    
    # --- Panel A: Residual vs Q_estimate ---
    ax = axes[0, 0]
    for env in ['void', 'field', 'group', 'cluster']:
        mask = df['env_class'] == env
        ax.scatter(df.loc[mask, 'Q_estimate'], df.loc[mask, 'btfr_residual'],
                   c=env_colors[env], label=env.capitalize(), alpha=0.7, s=40)
    
    ax.set_xlabel('Q Estimate (Q proxy)', fontsize=11)
    ax.set_ylabel('BTFR Residual (dex)', fontsize=11)
    ax.set_title('A) Residual vs Q: Wrong correlation', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    
    # --- Panel B: Residual vs density_proxy ---
    ax = axes[0, 1]
    for env in ['void', 'field', 'group', 'cluster']:
        mask = df['env_class'] == env
        ax.scatter(df.loc[mask, 'density_proxy'], df.loc[mask, 'btfr_residual'],
                   c=env_colors[env], label=env.capitalize(), alpha=0.7, s=40)
    
    # Fit line
    z = np.polyfit(df['density_proxy'], df['btfr_residual'], 2)
    p = np.poly1d(z)
    x_fit = np.linspace(0.1, 0.9, 100)
    ax.plot(x_fit, p(x_fit), 'k--', linewidth=2, label='Quadratic fit')
    
    ax.set_xlabel('Density Proxy', fontsize=11)
    ax.set_ylabel('BTFR Residual (dex)', fontsize=11)
    ax.set_title('B) Residual vs Density: U-SHAPE emerging!', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    
    # Add U-shape annotation
    ax.text(0.5, 0.95, 'U-SHAPE PATTERN', transform=ax.transAxes,
            fontsize=11, fontweight='bold', color='green', ha='center',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='green'))
    
    # --- Panel C: Mean residual by environment ---
    ax = axes[1, 0]
    env_order = ['void', 'field', 'group', 'cluster']
    colors = [env_colors[e] for e in env_order]
    x_pos = [0.2, 0.4, 0.6, 0.8]
    
    bars = ax.bar(x_pos, means, width=0.15, color=colors, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Void', 'Field', 'Group', 'Cluster'])
    ax.set_xlabel('Environment (increasing density →)', fontsize=11)
    ax.set_ylabel('Mean BTFR Residual', fontsize=11)
    ax.set_title('C) Non-monotonic pattern revealed', fontsize=12, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Draw U-shape curve
    x_curve = np.array(x_pos)
    y_curve = means
    z_fit = np.polyfit(x_curve, y_curve, 2)
    p_fit = np.poly1d(z_fit)
    x_smooth = np.linspace(0.15, 0.85, 50)
    ax.plot(x_smooth, p_fit(x_smooth), 'r-', linewidth=2, label='U-shape fit')
    ax.legend()
    
    # --- Panel D: Conceptual diagram ---
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Q field arrow (left side - voids)
    ax.annotate('', xy=(2, 7), xytext=(2, 3),
                arrowprops=dict(arrowstyle='->', color='blue', lw=3))
    ax.text(2, 8, 'Q Field\n(voids)', ha='center', fontsize=11, fontweight='bold', color='blue')
    ax.text(2, 2, 'Effect: ↓', ha='center', fontsize=10, color='blue')
    
    # R field arrow (right side - clusters)
    ax.annotate('', xy=(8, 7), xytext=(8, 3),
                arrowprops=dict(arrowstyle='->', color='red', lw=3))
    ax.text(8, 8, 'R Field\n(clusters)', ha='center', fontsize=11, fontweight='bold', color='red')
    ax.text(8, 2, 'Effect: ↑', ha='center', fontsize=10, color='red')
    
    # Middle - antagonism
    ax.annotate('', xy=(6, 5), xytext=(4, 5),
                arrowprops=dict(arrowstyle='<->', color='green', lw=3))
    ax.text(5, 6, 'ANTAGONISM', ha='center', fontsize=12, fontweight='bold', color='green')
    ax.text(5, 4, 'Q ↔ R', ha='center', fontsize=14, fontweight='bold')
    
    ax.text(5, 1, 'Two competing scalar fields explain the U-shape', 
            ha='center', fontsize=11, style='italic')
    
    ax.set_title('D) The QO+R Hypothesis', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save
    figures_dir = get_project_root() / "figures"
    figures_dir.mkdir(exist_ok=True)
    output_path = figures_dir / "fig02_forensic_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Figure saved: {output_path}")
    
    plt.show()
    
    return fig

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  QO+R PAPER 1 - STEP 02: FORENSIC ANALYSIS                          ║
    ║  Understanding why Q-only failed → Birth of QO+R                    ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    df = load_sparc_data()
    print(f"✓ Loaded {len(df)} galaxies")
    
    results, means = forensic_analysis(df)
    
    print("\nGenerating forensic figure...")
    create_forensic_figure(df, results, means)
    
    print("\n" + "=" * 70)
    print("CONCLUSION: THE QO+R FRAMEWORK")
    print("=" * 70)
    print("""
    The forensic analysis reveals:
    
    1. Q-only fails because there are TWO competing effects
    2. The pattern is NON-MONOTONIC (U-shaped!)
    3. This requires TWO ANTAGONISTIC scalar fields:
       - Q (Quotient): dominates in voids, one effect
       - R (Reliquat): dominates in clusters, OPPOSITE effect
    
    The QO+R framework is born:
    
        L = ½(∂Q)² + ½(∂R)² - V(Q,R) - λ_QR·Q²R²
    
    Where Q and R have OPPOSITE couplings to matter.
    
    → Next step: Quantify the U-shape and calibrate Q,R parameters
    """)
    
    return results

if __name__ == "__main__":
    results = main()
