"""
Paper 4 Test Suite
==================

Comprehensive validation tests for the QO+R framework.

Test Categories:
- core_tests: Primary hypothesis validation (H1, H3, H4)
- robustness_tests: Bootstrap, jackknife, cross-validation
- alternative_tests: Eliminate competing explanations
- amplitude_tests: Investigate coefficient magnitudes

Usage:
    python run_all_tests.py --category all
    python run_all_tests.py --category core
    python run_all_tests.py --test ushape_detection
"""

from .core_tests import (
    test_qr_independence,
    test_ushape_detection,
    test_qor_model,
    test_killer_prediction
)

from .robustness_tests import (
    test_bootstrap_stability,
    test_spatial_jackknife,
    test_cross_validation,
    test_selection_sensitivity
)

from .alternative_tests import (
    test_selection_bias,
    test_mond_comparison,
    test_baryonic_control,
    test_tng_comparison
)

from .amplitude_tests import (
    test_environment_proxy,
    test_sample_purity
)

__all__ = [
    # Core
    'test_qr_independence',
    'test_ushape_detection',
    'test_qor_model',
    'test_killer_prediction',
    # Robustness
    'test_bootstrap_stability',
    'test_spatial_jackknife',
    'test_cross_validation',
    'test_selection_sensitivity',
    # Alternatives
    'test_selection_bias',
    'test_mond_comparison',
    'test_baryonic_control',
    'test_tng_comparison',
    # Amplitude
    'test_environment_proxy',
    'test_sample_purity',
]
