#!/usr/bin/env python3
"""
===============================================================================
ALTERNATIVE THEORIES TEST
===============================================================================

Systematically test and eliminate alternative explanations for the U-shape:

1. Warm Dark Matter (WDM) - Suppressed small-scale power
2. Self-Interacting Dark Matter (SIDM) - Core formation
3. f(R) Modified Gravity - Scale-dependent growth
4. Fuzzy Dark Matter (FDM) - Soliton cores
5. Quintessence - Dark energy variation

Each theory makes specific predictions that differ from string theory moduli.
This test derives and checks those predictions.

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
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_tests import prepare_data, fit_ushape

# =============================================================================
# THEORETICAL PREDICTIONS
# =============================================================================

THEORIES = {
    'String_Theory_Moduli': {
        'description': 'Calabi-Yau compactification with KKLT stabilization',
        'predictions': {
            'u_shape': True,
            'sign_inversion': True,  # Q-dom vs R-dom
            'mass_dependence': 'weak',  # α ≈ 0
            'environmental_dependence': 'quadratic',  # U-shape
            'redshift_evolution': 'logarithmic',  # λ(z) = λ_0 × (1 + α×ln(1+z))
        }
    },
    'WDM': {
        'description': 'Warm Dark Matter with free-streaming cutoff',
        'predictions': {
            'u_shape': False,  # No environmental U-shape predicted
            'sign_inversion': False,
            'mass_dependence': 'strong_negative',  # Suppressed at low mass
            'environmental_dependence': 'monotonic',  # Not quadratic
            'redshift_evolution': 'power_law',
        }
    },
    'SIDM': {
        'description': 'Self-Interacting Dark Matter with core formation',
        'predictions': {
            'u_shape': 'partial',  # Some curvature but different shape
            'sign_inversion': False,  # No sign flip predicted
            'mass_dependence': 'strong',  # Core size scales with mass
            'environmental_dependence': 'linear',  # Not quadratic
            'redshift_evolution': 'power_law',
        }
    },
    'fR_Gravity': {
        'description': 'f(R) modified gravity with Chameleon screening',
        'predictions': {
            'u_shape': 'inverted',  # Opposite curvature (screening in dense regions)
            'sign_inversion': False,
            'mass_dependence': 'strong',  # Screening threshold
            'environmental_dependence': 'step_function',  # Screened vs unscreened
            'redshift_evolution': 'rapid',
        }
    },
    'Fuzzy_DM': {
        'description': 'Ultra-light axion dark matter with soliton cores',
        'predictions': {
            'u_shape': False,
            'sign_inversion': False,
            'mass_dependence': 'strong_negative',  # M_soliton ∝ M_halo^(-1/3)
            'environmental_dependence': 'weak',
            'redshift_evolution': 'constant',
        }
    },
    'Quintessence': {
        'description': 'Dynamical dark energy with scalar field',
        'predictions': {
            'u_shape': False,  # No Q²R² coupling
            'sign_inversion': False,
            'mass_dependence': 'none',
            'environmental_dependence': 'none',  # Dark energy is uniform
            'redshift_evolution': 'power_law',
        }
    }
}


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_u_shape_presence(datasets):
    """Test whether U-shape is present."""
    results = {}
    
    for name, df in datasets.items():
        if name.startswith('TNG'):
            continue
        
        df = prepare_data(df, name)
        
        if 'env_std' not in df.columns or 'btfr_residual' not in df.columns:
            continue
        
        mask = np.isfinite(df['env_std']) & np.isfinite(df['btfr_residual'])
        fit = fit_ushape(df.loc[mask, 'env_std'].values, 
                        df.loc[mask, 'btfr_residual'].values)
        
        if fit:
            results[name] = {
                'a': fit['a'],
                'a_err': fit['a_err'],
                'p_value': fit['p_value'],
                'is_u_shape': fit['a'] > 0 and fit['p_value'] < 0.1,
                'is_inverted': fit['a'] < 0 and fit['p_value'] < 0.1
            }
    
    return results


def test_sign_inversion(datasets):
    """Test for sign inversion between Q-dom and R-dom."""
    results = {}
    
    for name, df in datasets.items():
        if name.startswith('TNG'):
            continue
        
        df = prepare_data(df, name)
        
        if 'gas_fraction' not in df.columns:
            continue
        if 'env_std' not in df.columns or 'btfr_residual' not in df.columns:
            continue
        
        # Q-dominated
        q_dom = df[df['gas_fraction'] >= 0.5]
        if len(q_dom) >= 100:
            q_fit = fit_ushape(q_dom['env_std'].values, q_dom['btfr_residual'].values)
        else:
            q_fit = None
        
        # R-dominated (EXTREME)
        if 'log_Mstar' in df.columns:
            r_dom = df[(df['gas_fraction'] < 0.05) & (df['log_Mstar'] > 10.5)]
        else:
            r_dom = df[df['gas_fraction'] < 0.1]
        
        if len(r_dom) >= 20:
            r_fit = fit_ushape(r_dom['env_std'].values, r_dom['btfr_residual'].values,
                              n_bins=8, min_per_bin=5)
        else:
            r_fit = None
        
        if q_fit and r_fit:
            inversion = (q_fit['a'] > 0) and (r_fit['a'] < 0)
        else:
            inversion = None
        
        results[name] = {
            'q_dom_a': q_fit['a'] if q_fit else None,
            'r_dom_a': r_fit['a'] if r_fit else None,
            'sign_inversion': inversion
        }
    
    return results


def test_mass_dependence(datasets):
    """Test strength of mass dependence."""
    results = {}
    
    for name, df in datasets.items():
        if name.startswith('TNG'):
            continue
        
        df = prepare_data(df, name)
        
        if 'log_Mstar' not in df.columns:
            continue
        if 'env_std' not in df.columns or 'btfr_residual' not in df.columns:
            continue
        
        # Split by mass
        mass_med = df['log_Mstar'].median()
        
        low_mass = df[df['log_Mstar'] < mass_med]
        high_mass = df[df['log_Mstar'] >= mass_med]
        
        if len(low_mass) >= 100:
            low_fit = fit_ushape(low_mass['env_std'].values, low_mass['btfr_residual'].values)
        else:
            low_fit = None
        
        if len(high_mass) >= 100:
            high_fit = fit_ushape(high_mass['env_std'].values, high_mass['btfr_residual'].values)
        else:
            high_fit = None
        
        if low_fit and high_fit and low_fit['a'] != 0:
            ratio = high_fit['a'] / low_fit['a']
            
            # Classify dependence strength
            if abs(np.log10(abs(ratio))) < 0.2:
                strength = 'weak'
            elif abs(np.log10(abs(ratio))) < 0.5:
                strength = 'moderate'
            else:
                strength = 'strong'
            
            if ratio > 1:
                direction = 'positive'
            else:
                direction = 'negative'
        else:
            ratio = None
            strength = 'unknown'
            direction = 'unknown'
        
        results[name] = {
            'low_mass_a': low_fit['a'] if low_fit else None,
            'high_mass_a': high_fit['a'] if high_fit else None,
            'ratio': ratio,
            'strength': strength,
            'direction': direction
        }
    
    return results


def test_environmental_shape(datasets):
    """Test whether environmental dependence is quadratic, linear, or step-like."""
    results = {}
    
    for name, df in datasets.items():
        if name.startswith('TNG'):
            continue
        
        df = prepare_data(df, name)
        
        if 'env_std' not in df.columns or 'btfr_residual' not in df.columns:
            continue
        
        mask = np.isfinite(df['env_std']) & np.isfinite(df['btfr_residual'])
        env = df.loc[mask, 'env_std'].values
        res = df.loc[mask, 'btfr_residual'].values
        
        # Fit different models
        models = {}
        
        # Linear: residual = a*env + b
        try:
            slope, intercept, r_lin, p_lin, _ = stats.linregress(env, res)
            models['linear'] = {
                'r_squared': r_lin**2,
                'p_value': p_lin,
                'aic': len(env) * np.log(np.var(res - (slope*env + intercept))) + 2*2
            }
        except:
            pass
        
        # Quadratic: residual = a*env² + b*env + c
        try:
            popt, _ = curve_fit(lambda x, a, b, c: a*x**2 + b*x + c, env, res, p0=[0.01, 0, 0])
            pred = popt[0]*env**2 + popt[1]*env + popt[2]
            ss_res = np.sum((res - pred)**2)
            ss_tot = np.sum((res - np.mean(res))**2)
            r2_quad = 1 - ss_res/ss_tot
            models['quadratic'] = {
                'r_squared': r2_quad,
                'a': popt[0],
                'aic': len(env) * np.log(ss_res/len(env)) + 2*3
            }
        except:
            pass
        
        # Determine best model
        if models:
            best = min(models.items(), key=lambda x: x[1].get('aic', np.inf))
            results[name] = {
                'models': models,
                'best_model': best[0],
                'is_quadratic': best[0] == 'quadratic'
            }
        else:
            results[name] = {'status': 'FAILED'}
    
    return results


# =============================================================================
# THEORY COMPARISON
# =============================================================================

def compare_with_theories(observations):
    """Compare observations with theory predictions."""
    
    comparison = {}
    
    for theory_name, theory in THEORIES.items():
        preds = theory['predictions']
        matches = 0
        total = 0
        details = {}
        
        # Check U-shape
        if 'u_shape' in observations:
            total += 1
            obs_ushape = any(d.get('is_u_shape', False) for d in observations['u_shape'].values())
            pred_ushape = preds['u_shape']
            
            if pred_ushape == True:
                match = obs_ushape
            elif pred_ushape == False:
                match = not obs_ushape
            elif pred_ushape == 'inverted':
                match = any(d.get('is_inverted', False) for d in observations['u_shape'].values())
            else:
                match = True  # 'partial' matches any
            
            if match:
                matches += 1
            details['u_shape'] = {'observed': obs_ushape, 'predicted': pred_ushape, 'match': match}
        
        # Check sign inversion
        if 'sign_inversion' in observations:
            total += 1
            obs_inversion = any(d.get('sign_inversion', False) for d in observations['sign_inversion'].values() 
                               if d.get('sign_inversion') is not None)
            pred_inversion = preds['sign_inversion']
            
            match = (obs_inversion == pred_inversion)
            if match:
                matches += 1
            details['sign_inversion'] = {'observed': obs_inversion, 'predicted': pred_inversion, 'match': match}
        
        # Check mass dependence
        if 'mass_dependence' in observations:
            total += 1
            strengths = [d.get('strength', 'unknown') for d in observations['mass_dependence'].values()]
            obs_strength = 'weak' if 'weak' in strengths else ('strong' if 'strong' in strengths else 'moderate')
            pred_strength = preds['mass_dependence']
            
            if 'weak' in pred_strength:
                match = obs_strength == 'weak'
            elif 'strong' in pred_strength:
                match = obs_strength == 'strong'
            else:
                match = True
            
            if match:
                matches += 1
            details['mass_dependence'] = {'observed': obs_strength, 'predicted': pred_strength, 'match': match}
        
        # Check environmental shape
        if 'env_shape' in observations:
            total += 1
            obs_quad = any(d.get('is_quadratic', False) for d in observations['env_shape'].values())
            pred_shape = preds['environmental_dependence']
            
            if pred_shape == 'quadratic':
                match = obs_quad
            elif pred_shape in ['linear', 'monotonic']:
                match = not obs_quad
            else:
                match = True
            
            if match:
                matches += 1
            details['env_shape'] = {'observed': 'quadratic' if obs_quad else 'other', 
                                   'predicted': pred_shape, 'match': match}
        
        # Calculate compatibility
        compatibility = matches / total if total > 0 else 0
        
        comparison[theory_name] = {
            'description': theory['description'],
            'matches': matches,
            'total_tests': total,
            'compatibility': compatibility,
            'details': details,
            'verdict': 'COMPATIBLE' if compatibility >= 0.75 else ('PARTIALLY_COMPATIBLE' if compatibility >= 0.5 else 'INCOMPATIBLE')
        }
    
    return comparison


# =============================================================================
# MAIN TEST
# =============================================================================

def test_alternative_theories(datasets):
    """
    Run complete alternative theories test.
    
    Returns comparison of observations with each theory's predictions.
    """
    results = {
        'test_name': 'Alternative Theories Elimination',
        'description': 'Compare observations with predictions from alternative dark matter/gravity theories',
    }
    
    # Run observational tests
    print("  Testing U-shape presence...")
    observations = {
        'u_shape': test_u_shape_presence(datasets)
    }
    
    print("  Testing sign inversion...")
    observations['sign_inversion'] = test_sign_inversion(datasets)
    
    print("  Testing mass dependence...")
    observations['mass_dependence'] = test_mass_dependence(datasets)
    
    print("  Testing environmental shape...")
    observations['env_shape'] = test_environmental_shape(datasets)
    
    results['observations'] = observations
    
    # Compare with theories
    print("  Comparing with theory predictions...")
    comparison = compare_with_theories(observations)
    results['theory_comparison'] = comparison
    
    # Determine which theories are eliminated
    eliminated = [name for name, comp in comparison.items() 
                  if comp['verdict'] == 'INCOMPATIBLE']
    compatible = [name for name, comp in comparison.items() 
                  if comp['verdict'] == 'COMPATIBLE']
    
    results['eliminated_theories'] = eliminated
    results['compatible_theories'] = compatible
    
    # Check if string theory is the only compatible one
    if 'String_Theory_Moduli' in compatible and len(compatible) == 1:
        results['conclusion'] = 'String Theory Moduli is the ONLY compatible theory'
        results['passed'] = True
    elif 'String_Theory_Moduli' in compatible:
        results['conclusion'] = f'String Theory compatible, along with: {[t for t in compatible if t != "String_Theory_Moduli"]}'
        results['passed'] = True
    else:
        results['conclusion'] = 'String Theory Moduli is NOT compatible with observations'
        results['passed'] = False
    
    results['summary'] = f"Eliminated: {len(eliminated)}/{len(THEORIES)} theories"
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print(" ALTERNATIVE THEORIES ELIMINATION TEST")
    print("="*70)
    
    # Path to BTFR directory (relative to Git folder)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    GIT_DIR = os.path.dirname(PROJECT_DIR)
    BTFR_DIR = os.path.join(os.path.dirname(GIT_DIR), "BTFR")
    
    # Load data
    print("\nLoading datasets...")
    datasets = {}
    
    alfalfa_path = os.path.join(BTFR_DIR, "Test-Replicability", "data", "alfalfa_corrected.csv")
    if os.path.exists(alfalfa_path):
        datasets['ALFALFA'] = pd.read_csv(alfalfa_path)
        print(f"  ALFALFA: {len(datasets['ALFALFA'])} galaxies")

    sparc_path = os.path.join(BTFR_DIR, "Data", "processed", "sparc_with_environment.csv")
    if os.path.exists(sparc_path):
        datasets['SPARC'] = pd.read_csv(sparc_path)
        print(f"  SPARC: {len(datasets['SPARC'])} galaxies")
    
    if not datasets:
        print("ERROR: No datasets found!")
        return
    
    # Run test
    print("\nRunning alternative theories test...")
    results = test_alternative_theories(datasets)
    
    # Print results
    print("\n" + "="*70)
    print(" THEORY COMPARISON")
    print("="*70)
    
    for theory, comp in results['theory_comparison'].items():
        status = 'PASS' if comp['verdict'] == 'COMPATIBLE' else ('PART' if comp['verdict'] == 'PARTIALLY_COMPATIBLE' else 'FAIL')
        print(f"\n{status} {theory}")
        print(f"   {comp['description']}")
        print(f"   Compatibility: {comp['compatibility']*100:.0f}% ({comp['matches']}/{comp['total_tests']} tests)")
        print(f"   Verdict: {comp['verdict']}")
    
    print("\n" + "="*70)
    print(f" ELIMINATED: {results['eliminated_theories']}")
    print(f" COMPATIBLE: {results['compatible_theories']}")
    print(f" CONCLUSION: {results['conclusion']}")
    print("="*70)
    
    # Save results
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "experimental", "04_RESULTS", "test_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "alternative_theories_results.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {output_file}")


if __name__ == "__main__":
    main()
