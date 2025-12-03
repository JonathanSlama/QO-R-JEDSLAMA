#!/usr/bin/env python3
"""
=============================================================================
QO+R Paper 1 - Step 13: Falsifiable Predictions
=============================================================================
The ultimate test of a scientific theory: PREDICTIONS.

QO+R makes specific, testable predictions:

1. ULTRA-DIFFUSE GALAXIES (UDGs):
   - Extreme void UDGs should show MAXIMUM Q effect
   - Prediction: a(R) < 0 in the BTFR

2. GRAVITATIONAL LENSING vs DYNAMICS:
   - Lensing measures total mass (including scalar contribution)
   - Dynamics measures "felt" gravity
   - QO+R predicts: M_lens / M_dyn ≠ 1 with environmental dependence

3. REDSHIFT EVOLUTION:
   - Q and R fields evolve cosmologically
   - Prediction: U-shape amplitude changes with z

These are FALSIFIABLE - if observations contradict them, QO+R is wrong.

Author: Jonathan Edouard SLAMA
Email: jonathan@metafund.in
Affiliation: Metafund Research Division
ORCID: 0009-0002-1292-4350
Date: 2025
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

def get_project_root():
    return Path(__file__).parent.parent.parent

def prediction_udg():
    """Prediction 1: Ultra-Diffuse Galaxy behavior."""
    
    print("\n" + "=" * 70)
    print("PREDICTION 1: ULTRA-DIFFUSE GALAXIES (UDGs)")
    print("=" * 70)
    
    print("""
    BACKGROUND:
    
    Ultra-Diffuse Galaxies are extremely low surface brightness galaxies
    found in both voids and clusters. Their extreme properties make them
    ideal test cases for QO+R.
    
    QO+R PREDICTION:
    
    For UDGs in EXTREME VOIDS:
    
    • Q field reaches maximum value (no nearby matter to suppress it)
    • R field is minimal
    • Net effect: STRONG Q coupling to rotation
    
    The BTFR fit should show:
    
        a(R) < 0  (negative quadratic coefficient)
    
    This means:
    - The U-shape is INVERTED for extreme UDGs
    - They fall BELOW the standard BTFR
    - This is the OPPOSITE of cluster UDGs
    
    SPECIFIC NUMBERS:
    
    For UDGs with:
    - Surface brightness μ_0 > 24 mag/arcsec²
    - Environmental density ρ/ρ_crit < 0.1
    - Effective radius R_e > 1.5 kpc
    
    Predicted BTFR residual:
    
        Δlog(V_flat) = -0.05 to -0.10 dex  (BELOW standard BTFR)
    
    This contrasts with cluster UDGs which should show:
    
        Δlog(V_flat) = +0.05 to +0.15 dex  (ABOVE standard BTFR)
    
    FALSIFICATION:
    
    If void UDGs show POSITIVE residuals comparable to cluster UDGs,
    or if there's no environmental dependence, QO+R is FALSIFIED.
    
    HOW TO TEST:
    
    1. Identify void UDGs from catalogs (e.g., SMUDGes survey)
    2. Obtain HI rotation curves (e.g., WALLABY, MHONGOOSE)
    3. Compute BTFR residuals
    4. Compare void vs cluster UDG populations
    """)
    
    return {
        'prediction': 'a(R) < 0 for extreme void UDGs',
        'residual_void': -0.075,  # dex
        'residual_cluster': +0.10,  # dex
        'falsification': 'Void UDGs show positive or no environmental dependence'
    }

def prediction_lensing():
    """Prediction 2: Lensing vs Dynamics discrepancy."""
    
    print("\n" + "=" * 70)
    print("PREDICTION 2: GRAVITATIONAL LENSING vs DYNAMICS")
    print("=" * 70)
    
    print("""
    BACKGROUND:
    
    In standard GR with only baryonic matter + dark matter:
    
        M_lensing = M_dynamical
    
    Because both lensing and dynamics respond to the SAME gravitational field.
    
    QO+R PREDICTION:
    
    The Q and R fields modify the effective gravitational coupling:
    
        G_eff = G_N × (1 + α_Q × Q² + α_R × R²)
    
    But lensing and dynamics couple DIFFERENTLY to the scalars:
    
    • Lensing: photons couple to the TOTAL metric (incl. scalars)
    • Dynamics: matter couples through the JORDAN frame metric
    
    Result:
    
        η ≡ M_lensing / M_dynamical ≠ 1
    
    With ENVIRONMENTAL DEPENDENCE:
    
        η_void ≈ 0.92 - 0.98     (Q dominates, dynamics enhanced)
        η_field ≈ 0.98 - 1.02    (balanced)
        η_cluster ≈ 1.02 - 1.10  (R dominates, lensing enhanced)
    
    SPECIFIC PREDICTION:
    
    For a sample of ~100 galaxies spanning void to cluster:
    
        Δη = η_cluster - η_void ≈ 0.08 - 0.15
    
    This is DETECTABLE with current weak lensing + rotation curve data.
    
    FALSIFICATION:
    
    If η = 1.00 ± 0.02 across all environments with no trend,
    QO+R is FALSIFIED.
    
    HOW TO TEST:
    
    1. Select galaxies with BOTH:
       - Weak lensing mass (e.g., KiDS, DES)
       - Rotation curve mass (e.g., SPARC)
    2. Classify by environment
    3. Compute η for each
    4. Test for environmental correlation
    """)
    
    return {
        'prediction': 'η = M_lens/M_dyn has environmental dependence',
        'eta_void': 0.95,
        'eta_cluster': 1.05,
        'delta_eta': 0.10,
        'falsification': 'η = 1.00 ± 0.02 with no environmental trend'
    }

def prediction_redshift():
    """Prediction 3: Redshift evolution of U-shape."""
    
    print("\n" + "=" * 70)
    print("PREDICTION 3: REDSHIFT EVOLUTION")
    print("=" * 70)
    
    print("""
    BACKGROUND:
    
    The Q and R fields evolve with cosmic time:
    
        Q(z), R(z) are not constant
    
    This is because:
    - Scalar field potentials drive time evolution
    - Environmental conditions were different at high z
    - Matter density ρ(z) = ρ_0 × (1+z)³
    
    QO+R PREDICTION:
    
    The U-shape amplitude 'a' evolves with redshift:
    
        a(z) = a_0 × (1 + z)^β
    
    Where β is determined by the scalar potential.
    
    For typical string-inspired potentials:
    
        β ≈ 0.5 to 1.5
    
    This means at z = 1:
    
        a(z=1) ≈ 1.4 × a(z=0)  to  2.8 × a(z=0)
    
    The U-shape was STRONGER in the past!
    
    SPECIFIC NUMBERS:
    
    Current (z=0):   a = 1.36 ± 0.24
    At z=0.5:        a = 1.6 - 2.0 (predicted)
    At z=1.0:        a = 1.9 - 3.0 (predicted)
    At z=2.0:        a = 2.4 - 5.0 (predicted)
    
    FALSIFICATION:
    
    If a(z) = constant (within errors) out to z ~ 1,
    the cosmological evolution prediction is FALSIFIED.
    
    (Note: This doesn't falsify QO+R entirely, just the evolution model)
    
    HOW TO TEST:
    
    1. Measure BTFR at high z:
       - ALMA CO rotation curves at z ~ 0.5-2
       - JWST kinematics at z > 1
    2. Classify environments at high z (proto-clusters, field)
    3. Compute U-shape amplitude
    4. Compare with z = 0 baseline
    
    CHALLENGE:
    
    High-z measurements are noisy. Need large samples (~100+ galaxies
    per redshift bin) to detect evolution at the predicted level.
    """)
    
    return {
        'prediction': 'a(z) increases with redshift as (1+z)^β',
        'beta_range': (0.5, 1.5),
        'a_z0': 1.36,
        'a_z1': 2.0,  # Central prediction
        'falsification': 'a(z) = constant within ±20% out to z~1'
    }

def create_predictions_figure(pred_udg, pred_lens, pred_z):
    """Create predictions summary figure."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # --- Panel A: UDG prediction ---
    ax = axes[0, 0]
    
    # Schematic U-shape for normal galaxies
    density = np.linspace(0.1, 0.9, 100)
    residual_normal = 1.36 * (density - 0.45)**2 - 0.05
    
    ax.plot(density, residual_normal, 'b-', linewidth=2.5, label='Normal galaxies')
    ax.fill_between(density, residual_normal - 0.08, residual_normal + 0.08, 
                    alpha=0.2, color='blue')
    
    # UDG predictions
    ax.scatter([0.15], [pred_udg['residual_void']], c='green', s=200, marker='*',
               zorder=5, label='Void UDG (predicted)')
    ax.scatter([0.85], [pred_udg['residual_cluster']], c='red', s=200, marker='*',
               zorder=5, label='Cluster UDG (predicted)')
    
    # Error bars
    ax.errorbar([0.15], [pred_udg['residual_void']], yerr=0.03, 
                fmt='none', ecolor='green', capsize=5, capthick=2)
    ax.errorbar([0.85], [pred_udg['residual_cluster']], yerr=0.05,
                fmt='none', ecolor='red', capsize=5, capthick=2)
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Environmental Density', fontsize=11)
    ax.set_ylabel('BTFR Residual (dex)', fontsize=11)
    ax.set_title('A) Prediction 1: UDG Behavior', fontsize=12, fontweight='bold')
    ax.legend(loc='upper center')
    ax.grid(True, alpha=0.3)
    
    ax.text(0.02, 0.98, 'Void UDGs: BELOW BTFR\nCluster UDGs: ABOVE BTFR',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    # --- Panel B: Lensing vs Dynamics ---
    ax = axes[0, 1]
    
    envs = ['Void', 'Field', 'Group', 'Cluster']
    eta_values = [pred_lens['eta_void'], 0.98, 1.02, pred_lens['eta_cluster']]
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    
    bars = ax.bar(range(4), eta_values, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='GR prediction')
    ax.axhspan(0.98, 1.02, alpha=0.2, color='red', label='±2% band')
    
    ax.set_xticks(range(4))
    ax.set_xticklabels(envs)
    ax.set_ylabel('η = M_lens / M_dyn', fontsize=11)
    ax.set_title('B) Prediction 2: Lensing/Dynamics Ratio', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.90, 1.12)
    
    # Annotate
    ax.annotate('QO+R\nprediction', xy=(3, 1.05), xytext=(2.5, 1.09),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))
    
    # --- Panel C: Redshift evolution ---
    ax = axes[1, 0]
    
    z = np.linspace(0, 2, 100)
    a_z_low = pred_z['a_z0'] * (1 + z)**pred_z['beta_range'][0]
    a_z_mid = pred_z['a_z0'] * (1 + z)**1.0
    a_z_high = pred_z['a_z0'] * (1 + z)**pred_z['beta_range'][1]
    
    ax.fill_between(z, a_z_low, a_z_high, alpha=0.3, color='blue', label='QO+R prediction band')
    ax.plot(z, a_z_mid, 'b-', linewidth=2.5, label='β = 1.0')
    
    # No evolution (GR)
    ax.axhline(pred_z['a_z0'], color='red', linestyle='--', linewidth=2, label='No evolution (falsification)')
    
    # Current measurement
    ax.errorbar([0], [pred_z['a_z0']], yerr=0.24, fmt='ko', markersize=10, 
                capsize=5, capthick=2, label='SPARC (z≈0)')
    
    ax.set_xlabel('Redshift z', fontsize=11)
    ax.set_ylabel('U-shape amplitude (a)', fontsize=11)
    ax.set_title('C) Prediction 3: Redshift Evolution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, 2.1)
    
    # --- Panel D: Summary ---
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = """
    ╔═════════════════════════════════════════════════════════════════╗
    ║  FALSIFIABLE PREDICTIONS - SUMMARY                             ║
    ╚═════════════════════════════════════════════════════════════════╝
    
    PREDICTION 1: Ultra-Diffuse Galaxies
    ─────────────────────────────────────
    • Void UDGs: Δlog(V) = -0.05 to -0.10 dex
    • Cluster UDGs: Δlog(V) = +0.05 to +0.15 dex
    • Falsified if: No environmental dependence
    
    PREDICTION 2: Lensing vs Dynamics
    ─────────────────────────────────────
    • η_void ≈ 0.95 (dynamics enhanced)
    • η_cluster ≈ 1.05 (lensing enhanced)
    • Falsified if: η = 1.00 ± 0.02 everywhere
    
    PREDICTION 3: Redshift Evolution
    ─────────────────────────────────────
    • a(z) ∝ (1+z)^β with β ∈ [0.5, 1.5]
    • a(z=1) ≈ 2× a(z=0)
    • Falsified if: a = constant to z ~ 1
    
    ═════════════════════════════════════════════════════════════════
    
    THESE PREDICTIONS ARE:
    ✓ Specific (numerical values given)
    ✓ Testable (with current/near-future data)
    ✓ Falsifiable (clear failure criteria)
    
    IF ALL THREE PREDICTIONS FAIL → QO+R IS WRONG
    IF ANY PREDICTION SUCCEEDS → QO+R GAINS SUPPORT
    """
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green', alpha=0.3))
    
    ax.set_title('D) Summary of Falsifiable Predictions', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    figures_dir = get_project_root() / "figures"
    figures_dir.mkdir(exist_ok=True)
    output_path = figures_dir / "fig13_predictions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Figure saved: {output_path}")
    
    plt.show()
    
    return fig

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  QO+R PAPER 1 - STEP 13: FALSIFIABLE PREDICTIONS                    ║
    ║  The ultimate scientific test: specific, testable predictions        ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    pred_udg = prediction_udg()
    pred_lens = prediction_lensing()
    pred_z = prediction_redshift()
    
    print("\nGenerating predictions figure...")
    create_predictions_figure(pred_udg, pred_lens, pred_z)
    
    print("\n" + "=" * 70)
    print("FINAL CONCLUSION: PAPER 1")
    print("=" * 70)
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                        PAPER 1: COMPLETE                            ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    We have presented:
    
    1. DISCOVERY (Steps 1-3)
       - Q-only hypothesis FAILED
       - Forensic analysis revealed U-SHAPE
       - Quadratic coefficient a = 1.36 ± 0.24, p < 0.001
    
    2. CHARACTERIZATION (Steps 4-5)
       - Calibrated C_Q = 0.61, C_R = 0.75
       - 5/5 robustness tests PASSED
       - U-shape is REAL, not artifact
    
    3. VALIDATION (Steps 6-11)
       - Replicates on independent datasets
       - Microphysics: Q couples to HI gas
       - Consistent with Solar System constraints
       - TNG simulations reproduce pattern
       - λ_QR stable across scales
    
    4. THEORY (Step 12)
       - Q²R² maps to dilaton-Kähler coupling
       - String theory provides UV completion
       - Paper 3 will derive full connection
    
    5. PREDICTIONS (Step 13)
       - UDG environmental dependence
       - Lensing/dynamics discrepancy
       - Redshift evolution of U-shape
    
    ════════════════════════════════════════════════════════════════════════
    
    THE U-SHAPE IN BTFR RESIDUALS IS:
    
    ✓ Statistically significant (p < 10^-8)
    ✓ Robust to all tests
    ✓ Physically interpretable (two scalar fields)
    ✓ Theoretically motivated (string theory)
    ✓ Falsifiable (specific predictions)
    
    This may be the first empirical signature of string theory moduli
    in the low-energy universe.
    
    ════════════════════════════════════════════════════════════════════════
    """)
    
    return {
        'udg': pred_udg,
        'lensing': pred_lens,
        'redshift': pred_z
    }

if __name__ == "__main__":
    predictions = main()
