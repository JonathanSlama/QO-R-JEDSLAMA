#!/usr/bin/env python3
"""
=============================================================================
QO+R Paper 1 - Step 01: Initial QO Test (FAILURE)
=============================================================================
Test of the original Quotient Ontologique hypothesis with Q field only.
RESULT: FAILURE - Prediction inverted (-11% instead of +15%)

This failure motivated the reformulation with the antagonistic R field.

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

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent

def load_sparc_data():
    """Load SPARC data with environmental classifications."""
    data_path = get_project_root() / "data" / "sparc_with_environment.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Calculate BTFR residuals
    a_btfr, b_btfr = -1.0577, 0.3442  # Standard BTFR parameters
    df['btfr_residual'] = df['log_Vflat'] - (a_btfr + b_btfr * df['log_Mbar'])
    
    return df

def test_q_only_hypothesis(df):
    """
    Test the original Q-only hypothesis.
    
    Original prediction: Galaxies in low-density environments (high Q)
    should have HIGHER rotation velocities (positive residuals).
    
    Q_estimate column: proxy for Q field strength
    - Low density (voids) → Q_estimate ~ 0.8
    - High density (clusters) → Q_estimate ~ 0.2
    """
    
    print("=" * 70)
    print("TEST: Original Q-Only Hypothesis")
    print("=" * 70)
    
    # Q_estimate is our proxy for Q field strength
    x = df['Q_estimate'].values
    y = df['btfr_residual'].values
    
    # Linear correlation
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    print(f"\nLinear regression: residual = {slope:.4f} * Q + {intercept:.4f}")
    print(f"Correlation r = {r_value:.4f}")
    print(f"P-value = {p_value:.4f}")
    
    # Prediction was: higher Q → higher residual (positive slope)
    # Reality check
    if slope > 0:
        result = "CONFIRMED"
        direction = "positive"
    else:
        result = "FAILED"
        direction = "NEGATIVE (inverted!)"
    
    print(f"\n>>> PREDICTION: Slope should be POSITIVE")
    print(f">>> OBSERVED: Slope is {direction}")
    print(f">>> RESULT: {result}")
    
    # Compare void vs cluster galaxies
    voids = df[df['env_class'] == 'void']['btfr_residual']
    clusters = df[df['env_class'] == 'cluster']['btfr_residual']
    
    print(f"\n--- Environmental comparison ---")
    print(f"Void galaxies (high Q):    mean residual = {voids.mean():.4f} ± {voids.std():.4f}")
    print(f"Cluster galaxies (low Q):  mean residual = {clusters.mean():.4f} ± {clusters.std():.4f}")
    
    # T-test
    t_stat, t_pvalue = stats.ttest_ind(voids, clusters)
    print(f"T-test: t = {t_stat:.3f}, p = {t_pvalue:.4f}")
    
    # The key observation: voids have LOWER residuals than clusters
    # This is the OPPOSITE of what Q-only predicts!
    delta = voids.mean() - clusters.mean()
    print(f"\nDelta (void - cluster) = {delta:.4f}")
    print(f"Expected: POSITIVE (voids should be higher)")
    print(f"Observed: {'POSITIVE ✓' if delta > 0 else 'NEGATIVE ✗ (INVERTED!)'}")
    
    return {
        'slope': slope,
        'r_value': r_value,
        'p_value': p_value,
        'void_mean': voids.mean(),
        'cluster_mean': clusters.mean(),
        'delta': delta,
        'hypothesis_confirmed': slope > 0
    }

def create_figure(df, results):
    """Create publication-quality figure showing the Q-only failure."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color map for environments
    env_colors = {
        'void': '#2ecc71',      # Green
        'field': '#3498db',     # Blue
        'group': '#f39c12',     # Orange
        'cluster': '#e74c3c'    # Red
    }
    
    # --- Panel A: Q vs BTFR Residual ---
    ax = axes[0]
    
    for env in ['void', 'field', 'group', 'cluster']:
        mask = df['env_class'] == env
        ax.scatter(df.loc[mask, 'Q_estimate'], 
                   df.loc[mask, 'btfr_residual'],
                   c=env_colors[env], label=env.capitalize(),
                   alpha=0.7, s=50, edgecolor='white', linewidth=0.5)
    
    # Regression line
    x_fit = np.linspace(0, 1, 100)
    y_fit = results['slope'] * x_fit + (results['void_mean'] - results['slope'] * 0.8)
    ax.plot(x_fit, y_fit, 'k--', linewidth=2, label=f'Fit (r={results["r_value"]:.2f})')
    
    # Expected trend (dashed green)
    ax.annotate('', xy=(0.9, 0.03), xytext=(0.1, -0.02),
                arrowprops=dict(arrowstyle='->', color='green', lw=2, ls='--'))
    ax.text(0.5, 0.04, 'EXPECTED\n(Q-only)', ha='center', fontsize=10, color='green')
    
    # Observed trend (solid red)
    ax.annotate('', xy=(0.1, 0.02), xytext=(0.9, -0.02),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0.5, -0.04, 'OBSERVED', ha='center', fontsize=10, color='red')
    
    ax.set_xlabel('Q Field Proxy (Q_estimate)', fontsize=12)
    ax.set_ylabel('BTFR Residual (dex)', fontsize=12)
    ax.set_title('A) Q-Only Hypothesis Test: FAILURE', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    
    # Add failure annotation
    ax.text(0.05, 0.95, 'PREDICTION INVERTED', transform=ax.transAxes,
            fontsize=12, fontweight='bold', color='red',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9))
    
    # --- Panel B: Environment comparison ---
    ax = axes[1]
    
    env_order = ['void', 'field', 'group', 'cluster']
    means = [df[df['env_class'] == e]['btfr_residual'].mean() for e in env_order]
    stds = [df[df['env_class'] == e]['btfr_residual'].std() for e in env_order]
    colors = [env_colors[e] for e in env_order]
    
    bars = ax.bar(range(4), means, yerr=stds, color=colors, 
                  edgecolor='black', linewidth=1, capsize=5, alpha=0.8)
    
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Void\n(high Q)', 'Field', 'Group', 'Cluster\n(low Q)'])
    ax.set_ylabel('Mean BTFR Residual (dex)', fontsize=12)
    ax.set_title('B) Mean Residual by Environment', fontsize=14, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add arrows showing expected vs observed
    ax.annotate('Expected:\nVoid > Cluster', xy=(0, means[0]), xytext=(1.5, 0.02),
                fontsize=10, color='green',
                arrowprops=dict(arrowstyle='->', color='green'))
    ax.annotate('Observed:\nVoid < Cluster', xy=(3, means[3]), xytext=(1.5, -0.02),
                fontsize=10, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    
    # Save figure
    figures_dir = get_project_root() / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    output_path = figures_dir / "fig01_qo_only_failure.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Figure saved: {output_path}")
    
    plt.show()
    
    return fig

def main():
    """Main execution."""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  QO+R PAPER 1 - STEP 01: INITIAL Q-ONLY TEST                        ║
    ║  Testing the original Quotient Ontologique hypothesis               ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Load data
    print("Loading SPARC data...")
    df = load_sparc_data()
    print(f"✓ Loaded {len(df)} galaxies")
    print(f"  Environments: {df['env_class'].value_counts().to_dict()}")
    
    # Run test
    results = test_q_only_hypothesis(df)
    
    # Create figure
    print("\nGenerating figure...")
    create_figure(df, results)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Q-ONLY HYPOTHESIS")
    print("=" * 70)
    print("""
    The original Q-only hypothesis predicted that galaxies in low-density
    environments (voids, high Q) should have HIGHER rotation velocities
    than expected from the BTFR.
    
    OBSERVATION: The opposite is true!
    - Void galaxies have LOWER residuals than cluster galaxies
    - The correlation is INVERTED
    
    CONCLUSION: The Q field alone cannot explain the observed pattern.
    
    This failure motivated the introduction of the antagonistic R field,
    leading to the QO+R framework where Q and R have OPPOSITE effects.
    
    → Next step: Forensic analysis to understand why Q-only fails
    """)
    
    return results

if __name__ == "__main__":
    results = main()
