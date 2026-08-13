#!/usr/bin/env python3
"""
===============================================================================
PAPER 4 - COMPLETE TEST SUITE
===============================================================================

This script runs all validation tests for the QO+R framework.

Test Categories:
1. CORE: Primary hypothesis tests (H1, H3, H4)
2. ROBUSTNESS: Bootstrap, jackknife, cross-validation
3. ALTERNATIVES: Eliminate competing explanations
4. AMPLITUDE: Investigate coefficient magnitudes

Usage:
    python run_all_tests.py                    # Run all tests
    python run_all_tests.py --category core    # Run core tests only
    python run_all_tests.py --test ushape_detection  # Run specific test

Author: Jonathan Édouard Slama
Date: December 2025
===============================================================================
"""

import os
import sys
import json
import argparse
from datetime import datetime

# =============================================================================
# SETUP PATHS
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "experimental", "04_RESULTS", "test_results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures", "tests")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# =============================================================================
# IMPORTS (from local files)
# =============================================================================

from core_tests import (
    test_qr_independence,
    test_ushape_detection,
    test_qor_model,
    test_killer_prediction
)
from robustness_tests import (
    test_bootstrap_stability,
    test_spatial_jackknife,
    test_cross_validation,
    test_selection_sensitivity
)
from alternative_tests import (
    test_selection_bias,
    test_mond_comparison,
    test_baryonic_control,
    test_tng_comparison
)
from amplitude_tests import (
    test_environment_proxy,
    test_sample_purity
)

# =============================================================================
# TEST REGISTRY
# =============================================================================

TESTS = {
    'core': {
        'qr_independence': test_qr_independence,
        'ushape_detection': test_ushape_detection,
        'qor_model': test_qor_model,
        'killer_prediction': test_killer_prediction,
    },
    'robustness': {
        'bootstrap_stability': test_bootstrap_stability,
        'spatial_jackknife': test_spatial_jackknife,
        'cross_validation': test_cross_validation,
        'selection_sensitivity': test_selection_sensitivity,
    },
    'alternatives': {
        'selection_bias': test_selection_bias,
        'mond_comparison': test_mond_comparison,
        'baryonic_control': test_baryonic_control,
        'tng_comparison': test_tng_comparison,
    },
    'amplitude': {
        'environment_proxy': test_environment_proxy,
        'sample_purity': test_sample_purity,
    }
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_datasets():
    """Load all datasets for testing."""
    import pandas as pd
    
    # Path to BTFR directory (relative to Git folder)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
    GIT_DIR = os.path.dirname(PROJECT_DIR)
    BTFR_DIR = os.path.join(os.path.dirname(GIT_DIR), "BTFR")
    
    datasets = {}
    
    # SPARC
    sparc_path = os.path.join(BTFR_DIR, "Data", "processed", "sparc_with_environment.csv")
    if os.path.exists(sparc_path):
        datasets['SPARC'] = pd.read_csv(sparc_path)
        print(f"  SPARC: {len(datasets['SPARC'])} galaxies")
    else:
        print(f"  SPARC not found")

    # ALFALFA (corrected)
    alfalfa_path = os.path.join(BTFR_DIR, "Test-Replicability", "data", "alfalfa_corrected.csv")
    if os.path.exists(alfalfa_path):
        datasets['ALFALFA'] = pd.read_csv(alfalfa_path)
        print(f"  ALFALFA: {len(datasets['ALFALFA'])} galaxies")
    else:
        print(f"  ALFALFA not found (run prepare_alfalfa.py first)")

    # TNG results (for comparison)
    tng_path = os.path.join(BTFR_DIR, "TNG", "tng300_stratified_results.csv")
    if os.path.exists(tng_path):
        datasets['TNG300_results'] = pd.read_csv(tng_path)
        print(f"  TNG300 results: {len(datasets['TNG300_results'])} rows")
    else:
        print(f"  TNG300 results not found (some tests will skip)")
    
    return datasets

# =============================================================================
# TEST RUNNER
# =============================================================================

def run_test(category, name, test_func, datasets, verbose=True):
    """Run a single test and return results."""
    if verbose:
        print(f"\n  Running: {name}...", end=" ")
    
    try:
        results = test_func(datasets)
        results['status'] = 'PASSED' if results.get('passed', False) else 'FAILED'
        results['timestamp'] = datetime.now().isoformat()
        
        if verbose:
            status_symbol = 'PASS' if results['status'] == 'PASSED' else 'FAIL'
            print(f"{status_symbol} {results['status']}")
        
        return results
    
    except Exception as e:
        if verbose:
            print(f"ERROR: {str(e)}")
        return {
            'status': 'ERROR',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def run_category(category, datasets, verbose=True):
    """Run all tests in a category."""
    results = {}
    
    if category not in TESTS:
        print(f"Unknown category: {category}")
        return results
    
    print(f"\n{'='*60}")
    print(f" {category.upper()} TESTS")
    print(f"{'='*60}")
    
    for name, func in TESTS[category].items():
        results[name] = run_test(category, name, func, datasets, verbose)
    
    return results


def run_all(datasets, verbose=True):
    """Run all tests."""
    all_results = {}
    
    for category in ['core', 'robustness', 'alternatives', 'amplitude']:
        all_results[category] = run_category(category, datasets, verbose)
    
    return all_results


def save_results(results, filename):
    """Save results to JSON."""
    filepath = os.path.join(RESULTS_DIR, filename)
    
    def json_serializer(obj):
        if hasattr(obj, 'item'):  # numpy types
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        return str(obj)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=json_serializer)
    
    print(f"\n  Results saved: {filepath}")


def print_summary(results):
    """Print summary of all test results."""
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    total_passed = 0
    total_failed = 0
    total_error = 0
    
    for category, cat_results in results.items():
        passed = sum(1 for r in cat_results.values() if r.get('status') == 'PASSED')
        failed = sum(1 for r in cat_results.values() if r.get('status') == 'FAILED')
        errors = sum(1 for r in cat_results.values() if r.get('status') == 'ERROR')
        
        total_passed += passed
        total_failed += failed
        total_error += errors
        
        print(f"\n  {category.upper()}: {passed}/{len(cat_results)} passed")
        
        for name, res in cat_results.items():
            status = res.get('status', 'UNKNOWN')
            symbol = 'PASS' if status == 'PASSED' else ('FAIL' if status == 'FAILED' else 'WARN')
            summary = res.get('summary', '')[:40]
            print(f"    {symbol} {name}: {summary}")
    
    total = total_passed + total_failed + total_error
    
    print(f"\n{'='*70}")
    print(f" TOTAL: {total_passed}/{total} passed, {total_failed} failed, {total_error} errors")
    print(f"{'='*70}")
    
    # Overall verdict
    if total_failed == 0 and total_error == 0:
        print("\n  ALL TESTS PASSED!")
    elif total_passed > total_failed:
        print("\n  SOME TESTS FAILED - review results")
    else:
        print("\n  MAJORITY OF TESTS FAILED - investigate!")

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Paper 4 Test Suite')
    parser.add_argument('--category', '-c', 
                        choices=['core', 'robustness', 'alternatives', 'amplitude', 'all'],
                        default='all', help='Test category to run')
    parser.add_argument('--test', '-t', help='Specific test to run')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    parser.add_argument('--output', '-o', default='test_results.json', help='Output filename')
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    print("="*70)
    print(" PAPER 4 - QO+R VALIDATION TEST SUITE")
    print("="*70)
    print(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Output: {args.output}")
    
    # Load data
    print("\n Loading datasets...")
    datasets = load_datasets()
    
    if len(datasets) == 0:
        print("\n ERROR: No datasets loaded!")
        print(" Run prepare_alfalfa.py first if ALFALFA is missing.")
        return 1
    
    # Run tests
    if args.test:
        # Run specific test
        found = False
        for category, tests in TESTS.items():
            if args.test in tests:
                results = {category: {args.test: run_test(category, args.test, tests[args.test], datasets, verbose)}}
                found = True
                break
        
        if not found:
            print(f"\n Unknown test: {args.test}")
            print(" Available tests:")
            for cat, tests in TESTS.items():
                for t in tests:
                    print(f"   {cat}/{t}")
            return 1
    
    elif args.category == 'all':
        results = run_all(datasets, verbose)
    else:
        results = {args.category: run_category(args.category, datasets, verbose)}
    
    # Save and summarize
    save_results(results, args.output)
    print_summary(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
