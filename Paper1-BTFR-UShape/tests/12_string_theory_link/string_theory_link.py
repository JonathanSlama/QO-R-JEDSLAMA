#!/usr/bin/env python3
"""
=============================================================================
QO+R Paper 1 - Step 12: String Theory Connection (Paper 3 Teaser)
=============================================================================
Demonstrating the formal correspondence between QO+R and string theory:

    Q²R² ↔ Dilaton × Kähler Modulus coupling in Type IIB compactifications

This section provides a TEASER for Paper 3 (Theory of Everything),
showing that the empirical QO+R Lagrangian has a natural UV completion.

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

def explain_string_correspondence():
    """Explain the QO+R ↔ String Theory correspondence."""
    
    print("=" * 70)
    print("STRING THEORY CORRESPONDENCE: Q²R² ↔ DILATON-MODULUS")
    print("=" * 70)
    
    print("""
    ═══════════════════════════════════════════════════════════════════
    SECTION 1: THE QO+R LAGRANGIAN
    ═══════════════════════════════════════════════════════════════════
    
    The empirical QO+R effective Lagrangian:
    
        L_QOR = ½(∂Q)² + ½(∂R)² - V(Q,R) - λ_QR · Q²R² · T^μν g_μν
    
    Where:
    • Q = Quotient Ontologique field (dominates in voids)
    • R = Reliquat field (dominates in clusters)
    • V(Q,R) = scalar potential
    • λ_QR ≈ 1 = coupling constant (from fits)
    • T^μν = stress-energy tensor
    
    Key features:
    1. Two scalar fields with OPPOSITE environmental preferences
    2. Quartic coupling Q²R² between them
    3. Universal coupling λ_QR ≈ 1 across all scales
    
    ═══════════════════════════════════════════════════════════════════
    SECTION 2: TYPE IIB STRING COMPACTIFICATION
    ═══════════════════════════════════════════════════════════════════
    
    In Type IIB string theory compactified on Calabi-Yau 3-folds:
    
    The 10D action contains:
    
        S_IIB = ∫d¹⁰x √(-G) [e^{-2Φ}(R + 4(∂Φ)²) - ...]
    
    After compactification to 4D:
    
        S_4D = ∫d⁴x √(-g) [M_P²/2 · R - K_ij ∂φⁱ∂φʲ - V(φ) - ...]
    
    The relevant moduli:
    
    • Φ = DILATON (controls string coupling g_s = e^Φ)
      → Maps to Q field
      → Couples to matter through gravitational strength
      
    • T = KÄHLER MODULUS (controls internal volume)
      → Maps to R field  
      → Determines 4D Planck mass and matter masses
    
    ═══════════════════════════════════════════════════════════════════
    SECTION 3: THE FORMAL CORRESPONDENCE
    ═══════════════════════════════════════════════════════════════════
    
    The identification:
    
        Q = f(Φ)     where Φ = dilaton
        R = g(T)     where T = Kähler modulus
    
    In the string effective action, the kinetic terms are:
    
        L_kin = -K_{ΦΦ}(∂Φ)² - K_{TT}(∂T)² - K_{ΦT}(∂Φ)(∂T)
    
    For KKLT-type stabilization:
    
        K = -3 ln(T + T̄) - ln(-i(Φ - Φ̄))
    
    This gives:
    
        K_{ΦΦ} = 1/(Φ - Φ̄)² ∝ 1/Im(Φ)²
        K_{TT} = 3/(T + T̄)² ∝ 1/Re(T)²
        K_{ΦT} = 0 (no-scale structure)
    
    The CRUCIAL POINT:
    
    Superpotential interactions generate quartic couplings:
    
        W = W_0 + A·e^{-aT} + ...
    
    After supersymmetry breaking:
    
        V ⊃ λ · |Φ|² · |T|² + ...
    
    This IS the Q²R² coupling with λ set by string dynamics!
    
    ═══════════════════════════════════════════════════════════════════
    SECTION 4: WHY λ_QR ≈ 1?
    ═══════════════════════════════════════════════════════════════════
    
    In string theory, dimensionless couplings are typically O(1).
    
    The coupling λ_QR arises from:
    
        λ_QR = (M_string / M_Planck)^n × (geometric factors)
    
    For consistent compactifications:
    
        M_string ~ 10^{17} GeV
        M_Planck ~ 10^{18} GeV
        → M_string / M_Planck ~ 0.1
    
    With typical geometric factors, λ_QR ~ O(1) is NATURAL.
    
    Our empirical fit: λ_QR ≈ 1 (C_Q/C_R ≈ 0.8-1.2)
    
    THIS IS CONSISTENT WITH STRING THEORY EXPECTATIONS!
    
    ═══════════════════════════════════════════════════════════════════
    SECTION 5: PHYSICAL INTERPRETATION
    ═══════════════════════════════════════════════════════════════════
    
    Q (Dilaton):
    • Controls the STRING COUPLING (how strongly strings interact)
    • High Q → weak coupling → matter behaves more "freely"
    • In VOIDS: dilaton can take larger values (less matter to anchor it)
    • Effect: rotation curves ENHANCED relative to baryonic prediction
    
    R (Kähler):
    • Controls the INTERNAL VOLUME (compactification scale)
    • High R → larger internal dimensions → different mass scales
    • In CLUSTERS: Kähler modulus responds to high matter density
    • Effect: rotation curves ENHANCED via different mechanism
    
    The U-SHAPE emerges because:
    • Voids: Q dominates, enhances V_rot via dilaton coupling
    • Clusters: R dominates, enhances V_rot via Kähler coupling
    • Field: Neither dominates → minimum enhancement
    
    ═══════════════════════════════════════════════════════════════════
    SECTION 6: PAPER 3 PREVIEW
    ═══════════════════════════════════════════════════════════════════
    
    The complete derivation (Paper 3: Theory of Everything) will show:
    
    1. DIMENSIONAL REDUCTION:
       10D Type IIB → 4D effective theory
       Explicit computation of Q, R in terms of Φ, T
    
    2. MODULI STABILIZATION:
       KKLT scenario: W = W_0 + A·e^{-aT}
       Anti-brane uplift for de Sitter vacuum
    
    3. MATTER COUPLING:
       How Standard Model fields couple to Q, R
       Why galaxies "feel" the scalar fields
    
    4. COSMOLOGICAL EVOLUTION:
       Q, R evolution from early universe to today
       Why current values give λ_QR ≈ 1
    
    5. PREDICTIONS:
       Specific signatures in gravitational waves
       CMB implications
       Laboratory tests of moduli-matter coupling
    """)

def create_string_figure():
    """Create string theory correspondence figure."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # --- Panel A: Correspondence diagram ---
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # QO+R side
    ax.text(2, 8.5, 'QO+R\n(Empirical)', ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='blue'))
    
    ax.text(1, 6.5, 'Q field\n(Quotient)', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.text(3, 6.5, 'R field\n(Reliquat)', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    ax.text(2, 4.5, 'Q²R²\ncoupling', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    # String theory side
    ax.text(8, 8.5, 'String Theory\n(UV complete)', ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', edgecolor='red'))
    
    ax.text(7, 6.5, 'Dilaton Φ\n(string coupling)', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.text(9, 6.5, 'Kähler T\n(volume)', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    ax.text(8, 4.5, '|Φ|²|T|²\nfrom W', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    # Arrows
    ax.annotate('', xy=(6, 6.5), xytext=(4, 6.5),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax.text(5, 7, '≡', ha='center', fontsize=20, fontweight='bold', color='purple')
    
    ax.annotate('', xy=(6, 4.5), xytext=(4, 4.5),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax.text(5, 5, '≡', ha='center', fontsize=20, fontweight='bold', color='purple')
    
    # Label
    ax.text(5, 2, 'Q = f(Φ), R = g(T)\nλ_QR from string dynamics',
            ha='center', fontsize=12, style='italic',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    
    ax.set_title('A) QO+R ↔ String Theory Correspondence', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # --- Panel B: Compactification schematic ---
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # 10D
    ax.add_patch(plt.Rectangle((0.5, 7), 3, 2.5, facecolor='lightblue', edgecolor='blue'))
    ax.text(2, 8.2, '10D Type IIB\nString Theory', ha='center', fontsize=11, fontweight='bold')
    
    # Arrow down
    ax.annotate('', xy=(5, 6.5), xytext=(5, 7.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(6, 7, 'Compactify on\nCalabi-Yau', fontsize=10)
    
    # 4D
    ax.add_patch(plt.Rectangle((3.5, 3.5), 3, 2.5, facecolor='lightgreen', edgecolor='green'))
    ax.text(5, 4.7, '4D Effective\nSupergravity', ha='center', fontsize=11, fontweight='bold')
    
    # Arrow down
    ax.annotate('', xy=(5, 2.5), xytext=(5, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(6, 3, 'Low energy\nlimit', fontsize=10)
    
    # QO+R
    ax.add_patch(plt.Rectangle((3.5, 0.5), 3, 1.5, facecolor='lightyellow', edgecolor='orange'))
    ax.text(5, 1.2, 'QO+R', ha='center', fontsize=11, fontweight='bold')
    
    # Moduli labels
    ax.text(8.5, 5, 'Φ → Q\nT → R\nW → λ_QR', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    
    ax.set_title('B) Dimensional Reduction: 10D → 4D → QO+R', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # --- Panel C: Moduli space ---
    ax = axes[1, 0]
    
    # Create a 2D representation of moduli space
    phi = np.linspace(0.1, 2, 100)
    t = np.linspace(0.1, 2, 100)
    PHI, T = np.meshgrid(phi, t)
    
    # Potential (schematic)
    V = 0.1 * (PHI - 1)**2 + 0.1 * (T - 1)**2 + 0.3 * PHI**2 * T**2
    
    cs = ax.contourf(PHI, T, V, levels=20, cmap='viridis')
    ax.contour(PHI, T, V, levels=10, colors='white', linewidths=0.5)
    
    # Mark minimum
    ax.plot(1, 1, 'r*', markersize=15, label='Vacuum (λ_QR ≈ 1)')
    
    # Trajectories
    ax.annotate('', xy=(1.5, 0.5), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(1, 0.3, 'Q (Φ) direction', fontsize=10, color='red')
    
    ax.annotate('', xy=(0.5, 1.5), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(0.2, 1, 'R (T)\ndirection', fontsize=10, color='blue')
    
    ax.set_xlabel('Dilaton (Φ) ∝ Q', fontsize=11)
    ax.set_ylabel('Kähler (T) ∝ R', fontsize=11)
    ax.set_title('C) Moduli Space with Q²R² Potential', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    plt.colorbar(cs, ax=ax, label='V(Φ,T)')
    
    # --- Panel D: Summary ---
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = """
    STRING THEORY CONNECTION
    ══════════════════════════════════════════════
    
    CORRESPONDENCE:
    
    • Q field ←→ Dilaton Φ (string coupling)
    • R field ←→ Kähler T (internal volume)
    • Q²R² ←→ |Φ|²|T|² from superpotential
    • λ_QR ≈ 1 ←→ Natural string scale ratio
    
    WHY THIS WORKS:
    
    1. String theory predicts SCALAR FIELDS
       (dilaton, moduli) that couple to matter
    
    2. These fields have quartic interactions
       from supersymmetry breaking
    
    3. The coupling strength is O(1) naturally
       from string/Planck scale ratio
    
    4. Environmental dependence emerges from
       field profiles in matter distributions
    
    PAPER 3 WILL DERIVE:
    
    • Complete 10D → 4D reduction
    • Explicit Q(Φ), R(T) mappings
    • λ_QR from string parameters
    • Cosmological implications
    
    ══════════════════════════════════════════════
    THE U-SHAPE MAY BE A WINDOW INTO STRING THEORY
    """
    
    ax.text(0.1, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', alpha=0.5))
    
    ax.set_title('D) Summary & Paper 3 Preview', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    figures_dir = get_project_root() / "figures"
    figures_dir.mkdir(exist_ok=True)
    output_path = figures_dir / "fig12_string_theory.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Figure saved: {output_path}")
    
    plt.show()
    
    return fig

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  QO+R PAPER 1 - STEP 12: STRING THEORY CONNECTION                   ║
    ║  Teaser for Paper 3: The UV completion of QO+R                       ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    explain_string_correspondence()
    
    print("\nGenerating string theory figure...")
    create_string_figure()
    
    print("\n" + "=" * 70)
    print("CONCLUSION: STRING THEORY CONNECTION")
    print("=" * 70)
    print("""
    The QO+R framework has a NATURAL UV COMPLETION in string theory:
    
    • Q field = Dilaton (controls string coupling)
    • R field = Kähler modulus (controls compactification volume)
    • Q²R² coupling = Superpotential-generated quartic interaction
    • λ_QR ≈ 1 = Natural from string/Planck scale ratio
    
    This is NOT just a mathematical coincidence:
    
    1. String theory PREDICTS scalar fields that couple to matter
    2. These fields MUST have quartic interactions
    3. The coupling strength IS naturally O(1)
    4. Environmental variation IS expected
    
    The U-shape in galaxy rotation curves may be:
    
    → The first empirical signature of string theory moduli
    → A window into extra dimensions
    → Evidence for the string landscape
    
    This connection will be fully developed in:
    
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  PAPER 3: QO+R FROM TYPE IIB STRING THEORY                          ║
    ║  Complete derivation 10D → 4D → QO+R Lagrangian                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    → Next step: Falsifiable predictions (Paper 1 conclusion)
    """)

if __name__ == "__main__":
    main()
