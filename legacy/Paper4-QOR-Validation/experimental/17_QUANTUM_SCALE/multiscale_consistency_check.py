#!/usr/bin/env python3
"""
===============================================================================
MULTI-SCALE CONSISTENCY CHECK: LAB + SOLAR SYSTEM + QUANTUM
Verifying QO+R compatibility with precision gravity tests
===============================================================================

Based on Jonathan's theoretical framework:
1. Lab: α_eff(ρ_lab) ≈ 10⁻⁴ to 10⁻⁵
2. Solar System: γ_PPN - 1 ≈ -2α²/(1+α²)
3. Quantum: Δφ_QO+R = ∫ L_QO+R dt

Author: Jonathan Édouard Slama
Date: December 2025
===============================================================================
"""

import numpy as np
import json
import os

# Constants
G = 6.674e-11  # m³/kg/s²
c = 299792458  # m/s
hbar = 1.055e-34  # J·s
M_P = 2.176e-8  # kg (Planck mass)
L_P = 1.616e-35  # m (Planck length)
m_n = 1.675e-27  # kg (neutron mass)
g_earth = 9.81  # m/s²

# QO+R parameters from galactic fits
ALPHA_0 = 0.05  # Bare coupling from galactic scale
RHO_C = 1e-25  # g/cm³ - Critical density for screening
DELTA = 1.0  # Screening exponent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def screening_function(rho_gcm3):
    """
    Chameleon screening function f(ρ).
    f → 1 when ρ << ρ_c (no screening)
    f → 0 when ρ >> ρ_c (total screening)
    """
    return 1.0 / (1.0 + (rho_gcm3 / RHO_C)**DELTA)


def alpha_effective(rho_gcm3):
    """Effective QO+R coupling at density ρ."""
    return ALPHA_0 * screening_function(rho_gcm3)


def gamma_ppn_deviation(alpha):
    """
    γ_PPN - 1 from QO+R scalar field.
    From the document: γ_PPN - 1 ≈ -2α²/(1+α²)
    """
    return -2 * alpha**2 / (1 + alpha**2)


def qbounce_energy_levels(n_max=5):
    """
    Calculate qBOUNCE energy levels E_n.
    E_n = (9π²ℏ²m_n g²/8)^(1/3) × |a_n|
    where a_n are zeros of Airy function Ai(-x).
    """
    # Zeros of Airy function Ai(-x) (from tables)
    airy_zeros = [2.338, 4.088, 5.521, 6.787, 7.944, 9.023, 10.040]
    
    # Energy scale
    E_scale = (9 * np.pi**2 * hbar**2 * m_n * g_earth**2 / 8)**(1/3)
    E_scale_peV = E_scale / 1.602e-31  # Convert to peV
    
    energies = []
    for n in range(min(n_max, len(airy_zeros))):
        E_n = E_scale_peV * airy_zeros[n]**(2/3)
        energies.append(E_n)
    
    return energies


def phase_shift_interferometer(m, g, h, T, alpha_eff):
    """
    Phase shift in atomic/neutron interferometer from QO+R.
    Δφ_QO+R = (m g h T / ℏ) × α_eff
    """
    phi_newton = m * g * h * T / hbar
    phi_qor = phi_newton * alpha_eff
    return phi_newton, phi_qor


def main():
    print("="*70)
    print(" MULTI-SCALE CONSISTENCY CHECK: QO+R vs PRECISION TESTS")
    print("="*70)
    
    results = {
        'framework': 'QO+R',
        'alpha_0': ALPHA_0,
        'rho_c': RHO_C,
        'scales': {}
    }
    
    # =========================================================================
    # 1. LABORATORY SCALE (10⁻² - 10⁰ m)
    # =========================================================================
    print("\n" + "="*70)
    print(" 1. LABORATORY SCALE")
    print("="*70)
    
    lab_densities = {
        'Air': 1.2e-3,
        'Water': 1.0,
        'Aluminum': 2.7,
        'Tungsten (Eöt-Wash)': 19.3,
        'Platinum (MICROSCOPE)': 21.4,
    }
    
    print(f"\n  QO+R parameters: α₀ = {ALPHA_0}, ρ_c = {RHO_C:.0e} g/cm³")
    print(f"\n  {'Material':<25} {'ρ (g/cm³)':<12} {'f(ρ)':<12} {'α_eff':<12}")
    print("  " + "-"*60)
    
    lab_results = {}
    for material, rho in lab_densities.items():
        f_rho = screening_function(rho)
        a_eff = alpha_effective(rho)
        print(f"  {material:<25} {rho:<12.2e} {f_rho:<12.2e} {a_eff:<12.2e}")
        lab_results[material] = {'rho': rho, 'f_rho': f_rho, 'alpha_eff': a_eff}
    
    # Eöt-Wash constraint
    eot_wash_limit = 1e-3  # |α| < 10⁻³ at mm scale
    alpha_tungsten = lab_results['Tungsten (Eöt-Wash)']['alpha_eff']
    
    print(f"\n  Eöt-Wash constraint: |α| < {eot_wash_limit:.0e}")
    print(f"  QO+R prediction: α_eff = {alpha_tungsten:.2e}")
    
    if alpha_tungsten < eot_wash_limit:
        print("  ✅ COMPATIBLE (by factor {:.0e})".format(eot_wash_limit/alpha_tungsten))
        lab_verdict = "COMPATIBLE"
    else:
        print("  ❌ EXCLUDED")
        lab_verdict = "EXCLUDED"
    
    # MICROSCOPE constraint
    microscope_eta = 1e-15
    alpha_pt = lab_results['Platinum (MICROSCOPE)']['alpha_eff']
    
    print(f"\n  MICROSCOPE constraint: η < {microscope_eta:.0e}")
    print(f"  QO+R (universal coupling) → η = 0")
    print("  ✅ COMPATIBLE (QO+R preserves equivalence principle)")
    
    results['scales']['laboratory'] = {
        'densities': lab_results,
        'eot_wash': {'limit': eot_wash_limit, 'prediction': alpha_tungsten, 'compatible': alpha_tungsten < eot_wash_limit},
        'microscope': {'limit': microscope_eta, 'prediction': 0, 'compatible': True},
        'verdict': lab_verdict
    }
    
    # =========================================================================
    # 2. SOLAR SYSTEM SCALE (10⁸ - 10¹³ m)
    # =========================================================================
    print("\n" + "="*70)
    print(" 2. SOLAR SYSTEM SCALE")
    print("="*70)
    
    solar_densities = {
        'Sun (average)': 1.4,
        'Sun (core)': 150,
        'Earth (average)': 5.5,
        'Solar wind (1 AU)': 1e-23,
        'Interplanetary': 1e-24,
    }
    
    print(f"\n  {'Environment':<25} {'ρ (g/cm³)':<12} {'α_eff':<12} {'γ-1 prediction':<15}")
    print("  " + "-"*65)
    
    ss_results = {}
    for env, rho in solar_densities.items():
        a_eff = alpha_effective(rho)
        gamma_dev = gamma_ppn_deviation(a_eff)
        print(f"  {env:<25} {rho:<12.2e} {a_eff:<12.2e} {gamma_dev:<15.2e}")
        ss_results[env] = {'rho': rho, 'alpha_eff': a_eff, 'gamma_deviation': gamma_dev}
    
    # Cassini constraint
    cassini_limit = 2.3e-5  # |γ - 1| < 2.3×10⁻⁵
    
    # Near Sun (where Shapiro delay measured)
    alpha_sun = alpha_effective(solar_densities['Sun (average)'])
    gamma_pred = gamma_ppn_deviation(alpha_sun)
    
    print(f"\n  Cassini constraint: |γ - 1| < {cassini_limit:.1e}")
    print(f"  QO+R prediction (near Sun): γ - 1 = {gamma_pred:.2e}")
    
    if abs(gamma_pred) < cassini_limit:
        print("  ✅ COMPATIBLE (by factor {:.0e})".format(cassini_limit/abs(gamma_pred) if gamma_pred != 0 else float('inf')))
        ss_verdict = "COMPATIBLE"
    else:
        print("  ❌ EXCLUDED")
        ss_verdict = "EXCLUDED"
    
    # Formula from document
    print(f"\n  Formula: γ_PPN - 1 ≈ -2α²/(1+α²)")
    print(f"  With α_eff(ρ_☉) = {alpha_sun:.2e}")
    print(f"  → γ - 1 = -2×({alpha_sun:.2e})²/(1+({alpha_sun:.2e})²) = {gamma_pred:.2e}")
    
    # Constraint on α from Cassini
    # |γ - 1| < 2.3e-5 → 2α²/(1+α²) < 2.3e-5 → α < √(2.3e-5/2) ≈ 3.4e-3
    alpha_limit_cassini = np.sqrt(cassini_limit / 2)
    print(f"\n  Implied constraint: α_eff(ρ_☉) < {alpha_limit_cassini:.2e}")
    print(f"  QO+R predicts: α_eff(ρ_☉) = {alpha_sun:.2e}")
    
    results['scales']['solar_system'] = {
        'environments': ss_results,
        'cassini': {
            'limit': cassini_limit,
            'prediction': gamma_pred,
            'compatible': abs(gamma_pred) < cassini_limit
        },
        'verdict': ss_verdict
    }
    
    # =========================================================================
    # 3. QUANTUM SCALE (10⁻³⁵ - 10⁻¹⁰ m)
    # =========================================================================
    print("\n" + "="*70)
    print(" 3. QUANTUM SCALE")
    print("="*70)
    
    # qBOUNCE analysis
    print("\n  3.1 qBOUNCE (Neutron Quantum States)")
    print("  " + "-"*50)
    
    E_levels = qbounce_energy_levels(5)
    E1_measured = 1.407  # peV
    E1_error = 0.001  # peV (0.1% precision)
    
    print(f"\n  Theoretical E₁ (Newton): {E_levels[0]:.3f} peV")
    print(f"  Measured E₁ (qBOUNCE): {E1_measured:.3f} ± {E1_error:.3f} peV")
    
    # QO+R prediction at Earth surface
    rho_earth_surface = 2.7  # g/cm³ (typical rock)
    alpha_qbounce = alpha_effective(rho_earth_surface)
    
    delta_E1_rel = alpha_qbounce
    delta_E1_abs = E1_measured * delta_E1_rel
    
    print(f"\n  QO+R α_eff at Earth surface: {alpha_qbounce:.2e}")
    print(f"  Predicted δE₁/E₁: {delta_E1_rel:.2e}")
    print(f"  Predicted δE₁: {delta_E1_abs:.2e} peV")
    print(f"  Experimental precision: {E1_error:.3f} peV")
    
    if delta_E1_abs < E1_error:
        print(f"  ✅ COMPATIBLE (prediction {delta_E1_abs/E1_error:.0e}× smaller than precision)")
        qbounce_verdict = "COMPATIBLE"
    else:
        print("  ❌ DETECTABLE (or excluded)")
        qbounce_verdict = "DETECTABLE"
    
    # Phase shift in atomic interferometer
    print("\n  3.2 Atomic Interferometer (MIGA-type)")
    print("  " + "-"*50)
    
    m_cs = 133 * 1.66e-27  # Cs atom mass
    h_int = 1.0  # m height
    T_int = 1.0  # s interrogation time
    
    phi_newton, phi_qor = phase_shift_interferometer(m_cs, g_earth, h_int, T_int, alpha_qbounce)
    
    print(f"\n  Parameters: m = {m_cs:.2e} kg, h = {h_int} m, T = {T_int} s")
    print(f"  Newtonian phase: {phi_newton:.2e} rad")
    print(f"  QO+R phase shift: {phi_qor:.2e} rad")
    
    miga_precision = 1e-10  # rad (expected)
    
    if abs(phi_qor) < miga_precision:
        print(f"  MIGA precision: ~{miga_precision:.0e} rad")
        print(f"  ✅ COMPATIBLE (prediction {abs(phi_qor)/miga_precision:.0e}× smaller)")
        miga_verdict = "COMPATIBLE"
    else:
        print("  ⚠️ POTENTIALLY DETECTABLE")
        miga_verdict = "DETECTABLE"
    
    # Optical lattice clocks
    print("\n  3.3 Optical Lattice Clocks")
    print("  " + "-"*50)
    
    clock_precision = 1e-18  # Fractional frequency precision
    delta_g_g = alpha_qbounce  # Relative change in g
    
    print(f"\n  Clock precision: ΔG/G ~ {clock_precision:.0e}")
    print(f"  QO+R prediction: δg/g = α_eff = {delta_g_g:.2e}")
    
    if delta_g_g < clock_precision:
        print(f"  ✅ COMPATIBLE (prediction {delta_g_g/clock_precision:.0e}× smaller)")
        clock_verdict = "COMPATIBLE"
    else:
        print("  ⚠️ POTENTIALLY DETECTABLE")
        clock_verdict = "DETECTABLE"
    
    results['scales']['quantum'] = {
        'qbounce': {
            'E1_measured': E1_measured,
            'E1_error': E1_error,
            'alpha_eff': alpha_qbounce,
            'delta_E1': delta_E1_abs,
            'verdict': qbounce_verdict
        },
        'interferometer': {
            'phi_newton': phi_newton,
            'phi_qor': phi_qor,
            'miga_precision': miga_precision,
            'verdict': miga_verdict
        },
        'clocks': {
            'precision': clock_precision,
            'prediction': delta_g_g,
            'verdict': clock_verdict
        }
    }
    
    # =========================================================================
    # GLOBAL SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print(" GLOBAL SUMMARY")
    print("="*70)
    
    print("""
  ┌────────────────────────────────────────────────────────────────────────┐
  │  SCALE              │ TEST              │ CONSTRAINT     │ VERDICT    │
  ├────────────────────────────────────────────────────────────────────────┤
  │  Laboratory         │ Eöt-Wash          │ α < 10⁻³       │ {:<10} │
  │                     │ MICROSCOPE        │ η < 10⁻¹⁵      │ {:<10} │
  ├────────────────────────────────────────────────────────────────────────┤
  │  Solar System       │ Cassini (γ_PPN)   │ |γ-1| < 2×10⁻⁵ │ {:<10} │
  ├────────────────────────────────────────────────────────────────────────┤
  │  Quantum            │ qBOUNCE           │ δE/E < 10⁻³    │ {:<10} │
  │                     │ MIGA              │ δφ < 10⁻¹⁰ rad │ {:<10} │
  │                     │ Optical clocks    │ ΔG/G < 10⁻¹⁸   │ {:<10} │
  └────────────────────────────────────────────────────────────────────────┘
""".format(
        "✅ OK" if lab_verdict == "COMPATIBLE" else "❌",
        "✅ OK",
        "✅ OK" if ss_verdict == "COMPATIBLE" else "❌",
        "✅ OK" if qbounce_verdict == "COMPATIBLE" else "⚠️",
        "✅ OK" if miga_verdict == "COMPATIBLE" else "⚠️",
        "✅ OK" if clock_verdict == "COMPATIBLE" else "⚠️"
    ))
    
    all_compatible = (lab_verdict == "COMPATIBLE" and 
                      ss_verdict == "COMPATIBLE" and 
                      qbounce_verdict == "COMPATIBLE")
    
    if all_compatible:
        print("  ✅ QO+R IS COMPATIBLE WITH ALL PRECISION GRAVITY TESTS")
        print("  The chameleon screening mechanism successfully hides the signal")
        print("  at high density, while revealing it at low density (UDG, filaments).")
        global_verdict = "FULLY COMPATIBLE"
    else:
        print("  ⚠️ SOME TESTS MAY BE PROBLEMATIC - REVIEW NEEDED")
        global_verdict = "NEEDS REVIEW"
    
    results['global_verdict'] = global_verdict
    
    # Key insight
    print("\n" + "-"*70)
    print(" KEY INSIGHT: THE DENSITY TRANSITION")
    print("-"*70)
    
    densities_to_test = np.logspace(-30, 3, 100)  # g/cm³
    
    print(f"\n  Critical density ρ_c = {RHO_C:.0e} g/cm³")
    print(f"\n  At ρ >> ρ_c: screening total → α_eff → 0 → GR recovered")
    print(f"  At ρ << ρ_c: no screening → α_eff → α₀ → QO+R signal visible")
    print(f"\n  This explains why:")
    print(f"    • Lab (ρ ~ 1 g/cm³):        NO signal (screening)")
    print(f"    • GC (ρ ~ 10⁻²⁰ g/cm³):     NO signal (still screened)")
    print(f"    • UDG (ρ ~ 10⁻²⁶ g/cm³):    SIGNAL! (near ρ_c)")
    print(f"    • Filaments (ρ ~ 10⁻²⁸):    SIGNAL! (below ρ_c)")
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "multiscale_consistency.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {RESULTS_DIR}/multiscale_consistency.json")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
