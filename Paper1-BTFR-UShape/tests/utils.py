#!/usr/bin/env python3
"""
QO+R Analysis Utilities
=======================
Common functions for all QO+R validation tests.

Author: Jonathan Edouard SLAMA
Email: jonathan@metafund.in
Affiliation: Metafund Research Division
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def get_project_root():
    """Returns the project root directory."""
    return Path(__file__).parent.parent


def get_data_path():
    """Returns the path to the data directory."""
    return get_project_root() / "data"


def get_figures_path():
    """Returns the path to the figures directory."""
    return get_project_root() / "figures"


def load_sparc_data():
    """
    Load the SPARC dataset with environmental classifications.
    
    Returns:
        pd.DataFrame: SPARC data with columns:
            - Galaxy: Galaxy name
            - log_Mbar: log10 of baryonic mass
            - log_Vflat: log10 of flat rotation velocity
            - env_class: Environment classification (void/field/group/cluster)
            - density_proxy: Continuous density proxy [0, 1]
            - btfr_residual: Residual from BTFR fit
    """
    data_path = get_data_path() / "sparc_with_environment.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"SPARC data not found at {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Ensure required columns exist
    required = ['log_Mbar', 'log_Vflat', 'env_class', 'density_proxy']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Calculate BTFR residuals if not present
    if 'btfr_residual' not in df.columns:
        # Standard BTFR parameters
        a_btfr, b_btfr = -1.0577, 0.3442
        df['btfr_residual'] = df['log_Vflat'] - (a_btfr + b_btfr * df['log_Mbar'])
    
    # Clean NaN values
    df = df.dropna(subset=required + ['btfr_residual'])
    
    print(f"Loaded {len(df)} galaxies from SPARC")
    print(f"Environments: {df['env_class'].value_counts().to_dict()}")
    
    return df


# =============================================================================
# MODEL FITTING FUNCTIONS
# =============================================================================

def linear_model(x, a, b):
    """Linear model: y = a*x + b"""
    return a * x + b


def quadratic_model(x, a, b, c):
    """Quadratic model: y = a*x^2 + b*x + c"""
    return a * x**2 + b * x + c


def cubic_model(x, a, b, c, d):
    """Cubic model: y = a*x^3 + b*x^2 + c*x + d"""
    return a * x**3 + b * x**2 + c * x + d


def fit_models(x, y):
    """
    Fit linear, quadratic, and cubic models to data.
    
    Returns:
        dict: Results for each model including parameters, AIC, BIC
    """
    n = len(x)
    results = {}
    
    models = {
        'linear': (linear_model, 2),
        'quadratic': (quadratic_model, 3),
        'cubic': (cubic_model, 4)
    }
    
    for name, (func, n_params) in models.items():
        try:
            popt, pcov = curve_fit(func, x, y, maxfev=10000)
            y_pred = func(x, *popt)
            
            # Calculate statistics
            rss = np.sum((y - y_pred)**2)
            r_squared = 1 - rss / np.sum((y - np.mean(y))**2)
            
            # AIC and BIC
            aic = n * np.log(rss / n) + 2 * n_params
            bic = n * np.log(rss / n) + n_params * np.log(n)
            
            results[name] = {
                'params': popt,
                'pcov': pcov,
                'r_squared': r_squared,
                'aic': aic,
                'bic': bic,
                'rss': rss
            }
        except Exception as e:
            results[name] = None
    
    return results


def detect_ushape(df, column='btfr_residual', density_col='density_proxy'):
    """
    Detect U-shape pattern in residuals vs density.
    
    Returns:
        dict: U-shape detection results
    """
    x = df[density_col].values
    y = df[column].values
    
    fits = fit_models(x, y)
    
    if fits['quadratic'] is None:
        return {'detected': False, 'reason': 'Quadratic fit failed'}
    
    a_quad = fits['quadratic']['params'][0]
    
    # U-shape = positive quadratic coefficient
    is_ushape = a_quad > 0
    
    # Compare models
    if fits['linear'] is not None:
        delta_aic = fits['linear']['aic'] - fits['quadratic']['aic']
    else:
        delta_aic = 0
    
    return {
        'detected': is_ushape,
        'a_coefficient': a_quad,
        'delta_aic_vs_linear': delta_aic,
        'r_squared': fits['quadratic']['r_squared'],
        'significance': 'strong' if delta_aic > 10 else 'moderate' if delta_aic > 2 else 'weak'
    }


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def bootstrap_ushape(df, n_bootstrap=1000, seed=42):
    """
    Bootstrap test for U-shape robustness.
    
    Returns:
        dict: Bootstrap results including survival rate
    """
    np.random.seed(seed)
    
    a_values = []
    n = len(df)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        df_boot = df.iloc[idx]
        
        x = df_boot['density_proxy'].values
        y = df_boot['btfr_residual'].values
        
        try:
            popt, _ = curve_fit(quadratic_model, x, y)
            a_values.append(popt[0])
        except:
            continue
    
    a_values = np.array(a_values)
    ushape_rate = np.mean(a_values > 0) * 100
    
    return {
        'n_successful': len(a_values),
        'ushape_rate': ushape_rate,
        'a_mean': np.mean(a_values),
        'a_std': np.std(a_values),
        'a_values': a_values
    }


def permutation_test(df, n_permutations=1000, seed=42):
    """
    Permutation test for significance of U-shape.
    
    Returns:
        dict: Permutation test results including p-value
    """
    np.random.seed(seed)
    
    x = df['density_proxy'].values
    y = df['btfr_residual'].values
    
    # Original fit
    try:
        popt_orig, _ = curve_fit(quadratic_model, x, y)
        a_orig = popt_orig[0]
    except:
        return {'error': 'Original fit failed'}
    
    # Permutation distribution
    a_perm = []
    
    for i in range(n_permutations):
        x_shuffled = np.random.permutation(x)
        
        try:
            popt, _ = curve_fit(quadratic_model, x_shuffled, y)
            a_perm.append(popt[0])
        except:
            continue
    
    a_perm = np.array(a_perm)
    
    # P-value: proportion of permuted values >= original
    p_value = np.mean(np.abs(a_perm) >= np.abs(a_orig))
    
    return {
        'a_original': a_orig,
        'a_permuted_mean': np.mean(a_perm),
        'a_permuted_std': np.std(a_perm),
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def jackknife_by_environment(df):
    """
    Jackknife test: remove one environment at a time.
    
    Returns:
        dict: Results for each left-out environment
    """
    results = {}
    
    for env_out in df['env_class'].unique():
        df_jack = df[df['env_class'] != env_out]
        
        if len(df_jack) < 20:
            continue
        
        x = df_jack['density_proxy'].values
        y = df_jack['btfr_residual'].values
        
        try:
            popt, _ = curve_fit(quadratic_model, x, y)
            results[env_out] = {
                'a': popt[0],
                'ushape': popt[0] > 0,
                'n_remaining': len(df_jack)
            }
        except:
            results[env_out] = {'error': 'Fit failed'}
    
    return results


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def setup_plot_style():
    """Set up matplotlib style for publication-quality figures."""
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'figure.figsize': (10, 8),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


if __name__ == "__main__":
    # Quick test
    print("QO+R Analysis Utilities")
    print("=" * 50)
    
    try:
        df = load_sparc_data()
        result = detect_ushape(df)
        print(f"\nU-shape detection: {result}")
    except FileNotFoundError as e:
        print(f"\nData not found: {e}")
        print("Run from the project root with data/ directory.")
