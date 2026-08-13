# Paper 4 Test Suite

Comprehensive validation tests for the QO+R framework empirical analysis.

## Quick Start

```bash
# First time: prepare ALFALFA data
python prepare_alfalfa.py

# Run all tests
python run_all_tests.py

# Run specific category
python run_all_tests.py --category core
python run_all_tests.py --category robustness

# Run specific test
python run_all_tests.py --test ushape_detection
```

## Test Categories

### Core Tests (Required) ✅
| Test | Description | Hypothesis |
|------|-------------|------------|
| `qr_independence` | Verify Q and R are independent | H4 |
| `ushape_detection` | Detect U-shape in residuals | H1 |
| `qor_model` | Fit full QO+R regression | H1 |
| `killer_prediction` | Test sign inversion | H3 |

### Robustness Tests (Required) 🔶
| Test | Description |
|------|-------------|
| `bootstrap_stability` | 500× bootstrap resampling |
| `spatial_jackknife` | Sky region removal |
| `cross_validation` | 5-fold cross-validation |
| `selection_sensitivity` | Quality cut variations |

### Alternative Elimination Tests (Recommended) 🔷
| Test | Description |
|------|-------------|
| `selection_bias` | Mock catalog simulation |
| `mond_comparison` | Compare with MOND predictions |
| `baryonic_control` | Control for baryonic physics |
| `tng_comparison` | Compare with ΛCDM simulation |

### Amplitude Investigation Tests (Optional) ⚪
| Test | Description |
|------|-------------|
| `environment_proxy` | Compare proxy methods |
| `sample_purity` | Stratified amplitude analysis |

## Output

Results are saved to:
- `experimental/04_RESULTS/test_results/test_results.json`
- Individual test JSONs in subdirectories

## Requirements

```
numpy
pandas
scipy
statsmodels (optional, for robust standard errors)
astropy (for prepare_alfalfa.py only)
```

## Test Execution Order

1. **Phase 1: Data Preparation**
   ```bash
   python prepare_alfalfa.py  # Creates alfalfa_corrected.csv
   ```

2. **Phase 2: Core Validation**
   ```bash
   python run_all_tests.py --category core
   ```

3. **Phase 3: Robustness**
   ```bash
   python run_all_tests.py --category robustness
   ```

4. **Phase 4: Alternatives**
   ```bash
   python run_all_tests.py --category alternatives
   ```

5. **Phase 5: Amplitude Investigation**
   ```bash
   python run_all_tests.py --category amplitude
   ```

## Interpreting Results

### Pass Criteria

| Test Type | Criterion |
|-----------|-----------|
| Core | All must pass |
| Robustness | All must pass |
| Alternatives | Should pass (eliminate explanations) |
| Amplitude | Informational (no pass/fail) |

### Status Codes

- `PASSED` ✓ - Test criteria met
- `FAILED` ✗ - Test criteria not met
- `ERROR` ⚠ - Test execution failed
- `SKIPPED` - Missing data or prerequisites

## Files

| File | Description |
|------|-------------|
| `run_all_tests.py` | Main test runner |
| `core_tests.py` | Core hypothesis tests |
| `robustness_tests.py` | Stability tests |
| `alternative_tests.py` | Alternative elimination |
| `amplitude_tests.py` | Amplitude investigation |
| `prepare_alfalfa.py` | ALFALFA data preparation |
| `test_real_data.py` | Legacy main analysis (v3) |

---

*Last updated: December 6, 2025*
