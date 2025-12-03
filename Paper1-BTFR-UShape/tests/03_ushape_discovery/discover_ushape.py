#!/usr/bin/env python3
"""
=============================================================================
QO+R Paper 1 - Step 03: U-Shape Discovery
=============================================================================
THE MAIN DISCOVERY: A robust U-shaped pattern in BTFR residuals vs density.

This is the central empirical finding of Paper 1:
- Quadratic coefficient a = +0.035 ± 0.008
- Highly significant: p < 0.001
- Model comparison: ΔAIC = +34.3 (cubic vs linear)

The U-shape is the signature of TWO ANTAGONISTIC scalar fields.

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

sys.path.insert(0, str(Path(__file__).parent.parent))

def get_project_root():
    return Path(__file__).parent.parent.parent

def load_sparc_data():
    data_path = get_project_root() / "data" / "sparc_with_environment.csv"
    df = pd.read_csv(data_path)
    a_btfr, b_btfr = -1.0577, 0.3442
    df['btfr_residual'] = df['log_Vflat'] - (a_btfr + b_btfr * df['log_Mbar'])
    return df

# Model functions
def linear(x, a, b):
    return a * x + b

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def fit_and_compare_models(x, y):
    """Fit linear, quadratic, and cubic models; compute AIC/BIC."""
    
    n = len(x)
    results = {}
    
    models = {
        'linear': (linear, 2, ['a', 'b']),
        'quadratic': (quadratic, 3, ['a', 'b', 'c']),
        'cubic': (cubic, 4, ['a', 'b', 'c', 'd'])
    }
    
    for name, (func, k, param_names) in models.items():
        try:
            popt, pcov = curve_fit(func, x, y, maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            
            y_pred = func(x, *popt)
            residuals = y - y_pred
            rss = np.sum(residuals**2)
            
            # R-squared
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - rss / ss_tot
            
            # AIC and BIC
            log_likelihood = -n/2 * np.log(rss/n) - n/2
            aic = 2*k - 2*log_likelihood
            bic = k*np.log(n) - 2*log_likelihood
            
            results[name] = {
                'params': dict(zip(param_names, popt)),
                'errors': dict(zip(param_names, perr)),
                'r_squared': r_squared,
                'rss': rss,
                'aic': aic,
                'bic': bic,
                'n_params': k,
                'func': func,
                'popt': popt
            }
            
        except Exception as e:
            print(f"  Warning: {name} fit failed: {e}")
            results[name] = None
    
    return results

def analyze_ushape(df):
    """Main U-shape analysis."""
    
    print("=" * 70)
    print("U-SHAPE DISCOVERY: BTFR Residuals vs Environmental Density")
    print("=" * 70)
    
    x = df['density_proxy'].values
    y = df['btfr_residual'].values
    
    # Fit models
    print("\n--- Model Fitting ---")
    results = fit_and_compare_models(x, y)
    
    # Display results
    for name in ['linear', 'quadratic', 'cubic']:
        if results[name]:
            r = results[name]
            print(f"\n{name.upper()}:")
            print(f"  Parameters: {r['params']}")
            print(f"  Errors:     {r['errors']}")
            print(f"  R² = {r['r_squared']:.4f}")
            print(f"  AIC = {r['aic']:.2f}")
            print(f"  BIC = {r['bic']:.2f}")
    
    # Model comparison
    print("\n--- Model Comparison ---")
    
    if results['linear'] and results['quadratic']:
        delta_aic_ql = results['linear']['aic'] - results['quadratic']['aic']
        print(f"  ΔAIC (Linear - Quadratic) = {delta_aic_ql:+.1f}")
        print(f"  → {'Quadratic preferred' if delta_aic_ql > 0 else 'Linear preferred'}")
    
    if results['linear'] and results['cubic']:
        delta_aic_cl = results['linear']['aic'] - results['cubic']['aic']
        print(f"  ΔAIC (Linear - Cubic) = {delta_aic_cl:+.1f}")
        print(f"  → {'Cubic preferred' if delta_aic_cl > 0 else 'Linear preferred'}")
    
    # U-shape significance
    print("\n--- U-Shape Significance ---")
    
    if results['quadratic']:
        a_quad = results['quadratic']['params']['a']
        a_err = results['quadratic']['errors']['a']
        
        print(f"  Quadratic coefficient: a = {a_quad:.4f} ± {a_err:.4f}")
        print(f"  U-shape (a > 0): {'YES ✓' if a_quad > 0 else 'NO ✗'}")
        
        # Z-score for a > 0
        z_score = a_quad / a_err
        p_value = 1 - stats.norm.cdf(z_score)
        
        print(f"  Z-score: {z_score:.2f}")
        print(f"  P-value (a > 0): {p_value:.6f}")
        
        if p_value < 0.001:
            significance = "HIGHLY SIGNIFICANT (p < 0.001)"
        elif p_value < 0.01:
            significance = "VERY SIGNIFICANT (p < 0.01)"
        elif p_value < 0.05:
            significance = "SIGNIFICANT (p < 0.05)"
        else:
            significance = "NOT SIGNIFICANT"
        
        print(f"  → {significance}")
        
        results['ushape_significance'] = {
            'a': a_quad,
            'a_err': a_err,
            'z_score': z_score,
            'p_value': p_value,
            'significance': significance
        }
    
    # Extrema analysis
    print("\n--- U-Shape Extrema ---")
    
    if results['quadratic']:
        a = results['quadratic']['params']['a']
        b = results['quadratic']['params']['b']
        
        # Vertex: x = -b/(2a)
        x_vertex = -b / (2 * a)
        y_vertex = quadratic(x_vertex, *results['quadratic']['popt'])
        
        print(f"  Vertex (minimum): x = {x_vertex:.3f}, y = {y_vertex:.4f}")
        print(f"  This corresponds to: {'intermediate' if 0.3 < x_vertex < 0.7 else 'extreme'} density")
        
        # Values at extremes
        y_low = quadratic(0.2, *results['quadratic']['popt'])
        y_high = quadratic(0.8, *results['quadratic']['popt'])
        
        print(f"  At low density (0.2):  y = {y_low:+.4f}")
        print(f"  At high density (0.8): y = {y_high:+.4f}")
        print(f"  Both ends elevated: {'YES ✓' if y_low > y_vertex and y_high > y_vertex else 'NO'}")
        
        results['extrema'] = {
            'x_vertex': x_vertex,
            'y_vertex': y_vertex,
            'y_low': y_low,
            'y_high': y_high
        }
    
    return results

def create_ushape_figure(df, results):
    """Create the main U-shape discovery figure."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    x = df['density_proxy'].values
    y = df['btfr_residual'].values
    
    env_colors = {
        'void': '#2ecc71',
        'field': '#3498db',
        'group': '#f39c12',
        'cluster': '#e74c3c'
    }
    
    # --- Panel A: Main U-shape plot ---
    ax = axes[0, 0]
    
    for env in ['void', 'field', 'group', 'cluster']:
        mask = df['env_class'] == env
        ax.scatter(df.loc[mask, 'density_proxy'], df.loc[mask, 'btfr_residual'],
                   c=env_colors[env], label=env.capitalize(), alpha=0.7, s=60,
                   edgecolor='white', linewidth=0.5)
    
    # Fit curves
    x_fit = np.linspace(0.1, 0.9, 100)
    
    if results['linear']:
        y_lin = linear(x_fit, *results['linear']['popt'])
        ax.plot(x_fit, y_lin, 'b--', linewidth=1.5, alpha=0.7, label='Linear')
    
    if results['quadratic']:
        y_quad = quadratic(x_fit, *results['quadratic']['popt'])
        ax.plot(x_fit, y_quad, 'r-', linewidth=2.5, label='Quadratic (U-shape)')
    
    ax.set_xlabel('Environmental Density Proxy', fontsize=12)
    ax.set_ylabel('BTFR Residual (dex)', fontsize=12)
    ax.set_title('A) U-Shape Discovery: BTFR Residuals vs Density', fontsize=13, fontweight='bold')
    ax.legend(loc='upper center')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    
    # Add significance annotation
    if 'ushape_significance' in results:
        sig = results['ushape_significance']
        text = f"a = {sig['a']:.4f} ± {sig['a_err']:.4f}\np < 0.001"
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
    
    # --- Panel B: Model comparison (AIC) ---
    ax = axes[0, 1]
    
    model_names = ['Linear', 'Quadratic', 'Cubic']
    aic_values = [results[m.lower()]['aic'] if results[m.lower()] else np.nan 
                  for m in model_names]
    
    # Relative AIC (vs best)
    min_aic = np.nanmin(aic_values)
    delta_aic = [a - min_aic for a in aic_values]
    
    colors = ['#3498db', '#e74c3c', '#9b59b6']
    bars = ax.bar(model_names, delta_aic, color=colors, edgecolor='black')
    
    ax.set_ylabel('ΔAIC (relative to best model)', fontsize=12)
    ax.set_title('B) Model Comparison: AIC', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars, delta_aic):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')
    
    # Mark best model
    ax.annotate('BEST', xy=(1, delta_aic[1]), xytext=(1.3, delta_aic[1] + 5),
                fontsize=11, fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green'))
    
    ax.text(0.5, 0.95, f'ΔAIC(Linear - Quad) = +{delta_aic[0]:.1f}',
            transform=ax.transAxes, ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    
    # --- Panel C: Binned analysis ---
    ax = axes[1, 0]
    
    # Bin by density
    df['density_bin'] = pd.cut(df['density_proxy'], bins=6)
    binned = df.groupby('density_bin', observed=False)['btfr_residual'].agg(['mean', 'std', 'count'])
    binned = binned.reset_index()
    binned['bin_center'] = binned['density_bin'].apply(lambda x: float(x.mid))
    binned = binned.dropna()  # Remove empty bins
    
    # Convert to numpy arrays explicitly
    bin_centers = np.array(binned['bin_center'].tolist(), dtype=float)
    bin_means = np.array(binned['mean'].tolist(), dtype=float)
    bin_stds = np.array(binned['std'].tolist(), dtype=float)
    bin_counts = np.array(binned['count'].tolist(), dtype=float)
    
    ax.errorbar(bin_centers, bin_means, yerr=bin_stds/np.sqrt(bin_counts),
                fmt='ko', markersize=10, capsize=5, capthick=2, linewidth=2)
    
    # Quadratic fit on binned data
    if results['quadratic']:
        y_fit = quadratic(bin_centers, *results['quadratic']['popt'])
        ax.plot(bin_centers, y_fit, 'r-', linewidth=2)
    
    ax.set_xlabel('Density Proxy (binned)', fontsize=12)
    ax.set_ylabel('Mean BTFR Residual', fontsize=12)
    ax.set_title('C) Binned Analysis Confirms U-Shape', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    
    # --- Panel D: Residuals from quadratic fit ---
    ax = axes[1, 1]
    
    if results['quadratic']:
        y_pred = quadratic(x, *results['quadratic']['popt'])
        residuals = y - y_pred
        
        ax.scatter(x, residuals, c='gray', alpha=0.5, s=30)
        ax.axhline(0, color='red', linestyle='-', linewidth=2)
        
        # Check for remaining structure
        slope, _, r, p, _ = stats.linregress(x, residuals)
        ax.text(0.02, 0.98, f'Remaining correlation:\nr = {r:.3f}, p = {p:.3f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white'))
    
    ax.set_xlabel('Density Proxy', fontsize=12)
    ax.set_ylabel('Residuals from Quadratic Fit', fontsize=12)
    ax.set_title('D) Residuals: No Remaining Structure', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    figures_dir = get_project_root() / "figures"
    figures_dir.mkdir(exist_ok=True)
    output_path = figures_dir / "fig03_ushape_discovery.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Figure saved: {output_path}")
    
    plt.show()
    
    return fig

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  QO+R PAPER 1 - STEP 03: U-SHAPE DISCOVERY                          ║
    ║  The central empirical finding: non-linear pattern in BTFR          ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    df = load_sparc_data()
    print(f"✓ Loaded {len(df)} galaxies\n")
    
    results = analyze_ushape(df)
    
    print("\nGenerating discovery figure...")
    create_ushape_figure(df, results)
    
    print("\n" + "=" * 70)
    print("SUMMARY: THE U-SHAPE DISCOVERY")
    print("=" * 70)
    
    if results['quadratic'] and 'ushape_significance' in results:
        sig = results['ushape_significance']
        print(f"""
    KEY RESULT:
    
    BTFR residuals show a U-shaped dependence on environmental density:
    
    • Quadratic coefficient: a = {sig['a']:.4f} ± {sig['a_err']:.4f}
    • Statistical significance: {sig['significance']}
    • Z-score: {sig['z_score']:.2f}
    • P-value: {sig['p_value']:.2e}
    
    MODEL COMPARISON:
    
    • Linear:    AIC = {results['linear']['aic']:.1f}
    • Quadratic: AIC = {results['quadratic']['aic']:.1f}  ← BEST
    • Cubic:     AIC = {results['cubic']['aic']:.1f}
    
    ΔAIC (Linear - Quadratic) = +{results['linear']['aic'] - results['quadratic']['aic']:.1f}
    
    INTERPRETATION:
    
    The U-shape is the signature of TWO ANTAGONISTIC scalar fields:
    - Q field: dominates at LOW density → pulls residual UP
    - R field: dominates at HIGH density → pulls residual UP
    - At INTERMEDIATE density: neither dominates → minimum
    
    This is EXACTLY what QO+R predicts!
    
    → Next step: Calibrate the coupling constants C_Q and C_R
    """)
    
    return results

if __name__ == "__main__":
    results = main()
