#!/usr/bin/env python3
"""
04_residual_analysis.py
=======================
Core analysis: Testing the Revelatory Division hypothesis.

Author: Jonathan Édouard Slama
Email: jonathan@metafund.in
Date: December 2025

This script tests whether residuals from biomarker ratio models contain
diagnostic information that the ratios themselves miss.

Key analyses:
1. Residual distribution comparison (Cancer vs. Healthy)
2. U-shape detection in residuals vs. covariates
3. Residual-based classification improvement
4. Pattern discovery in the "discarded" information
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "figures"
RESULTS = PROJECT_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def load_processed_data():
    """Load the processed data with ratios and residuals."""
    data_path = DATA_PROCESSED / "data_with_ratios_and_residuals.csv"
    
    if not data_path.exists():
        print(f"ERROR: Processed data not found at {data_path}")
        print("Run previous scripts first.")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    print(f"✓ Data loaded: {len(df)} samples")
    return df


def test_residual_distributions(df):
    """
    TEST 1: Do residual distributions differ between groups?
    
    This tests whether the "discarded" information (residuals) 
    contains group-discriminating signal.
    """
    print("\n" + "=" * 60)
    print("TEST 1: RESIDUAL DISTRIBUTION DIFFERENCES")
    print("=" * 60)
    
    residual_cols = [col for col in df.columns if col.endswith('_resid')]
    
    results = []
    
    for resid_col in residual_cols:
        healthy = df.loc[df['Cancer'] == 0, resid_col].dropna()
        cancer = df.loc[df['Cancer'] == 1, resid_col].dropna()
        
        # Kolmogorov-Smirnov test for distribution difference
        ks_stat, ks_pval = stats.ks_2samp(healthy, cancer)
        
        # Mann-Whitney U test for location difference
        u_stat, u_pval = stats.mannwhitneyu(healthy, cancer, alternative='two-sided')
        
        # Levene's test for variance difference
        lev_stat, lev_pval = stats.levene(healthy, cancer)
        
        # Effect size
        pooled_std = np.sqrt((healthy.var() + cancer.var()) / 2)
        cohens_d = (cancer.mean() - healthy.mean()) / pooled_std if pooled_std > 0 else 0
        
        results.append({
            'Residual': resid_col.replace('_resid', ''),
            'Healthy_Mean': healthy.mean(),
            'Cancer_Mean': cancer.mean(),
            'Healthy_Std': healthy.std(),
            'Cancer_Std': cancer.std(),
            'KS_statistic': ks_stat,
            'KS_pvalue': ks_pval,
            'MW_pvalue': u_pval,
            'Levene_pvalue': lev_pval,
            'Cohens_d': cohens_d,
            'Distribution_differs': ks_pval < 0.05,
            'Location_differs': u_pval < 0.05,
            'Variance_differs': lev_pval < 0.05
        })
        
        # Print results
        print(f"\n{resid_col.replace('_resid', '')} Residuals:")
        print(f"  Healthy: mean = {healthy.mean():+.4f}, std = {healthy.std():.4f}")
        print(f"  Cancer:  mean = {cancer.mean():+.4f}, std = {cancer.std():.4f}")
        print(f"  Cohen's d: {cohens_d:+.3f}")
        print(f"  Distribution test (KS): p = {ks_pval:.4f} {'*' if ks_pval < 0.05 else ''}")
        print(f"  Location test (MW):     p = {u_pval:.4f} {'*' if u_pval < 0.05 else ''}")
        print(f"  Variance test (Levene): p = {lev_pval:.4f} {'*' if lev_pval < 0.05 else ''}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS / "test1_residual_distributions.csv", index=False)
    
    # Summary
    n_dist_diff = results_df['Distribution_differs'].sum()
    n_loc_diff = results_df['Location_differs'].sum()
    n_var_diff = results_df['Variance_differs'].sum()
    
    print("\n" + "-" * 40)
    print("SUMMARY:")
    print(f"  Residuals with different distributions: {n_dist_diff}/{len(residual_cols)}")
    print(f"  Residuals with different locations: {n_loc_diff}/{len(residual_cols)}")
    print(f"  Residuals with different variances: {n_var_diff}/{len(residual_cols)}")
    
    return results_df


def test_ushape_in_residuals(df):
    """
    TEST 2: Do residuals show U-shaped patterns vs. covariates?
    
    This is the key test from the QO+R framework:
    If two opposing mechanisms operate, residuals should show
    elevated values at both extremes of a covariate.
    """
    print("\n" + "=" * 60)
    print("TEST 2: U-SHAPE DETECTION IN RESIDUALS")
    print("=" * 60)
    
    residual_cols = [col for col in df.columns if col.endswith('_resid')]
    covariates = ['Age', 'BMI', 'Glucose', 'Insulin']
    
    results = []
    
    for resid_col in residual_cols:
        for covar in covariates:
            # Prepare data
            mask = df[resid_col].notna() & df[covar].notna()
            x = df.loc[mask, covar].values
            y = df.loc[mask, resid_col].values
            cancer = df.loc[mask, 'Cancer'].values
            
            # Normalize x for numerical stability
            x_norm = (x - x.mean()) / x.std()
            
            # Fit linear model: residual ~ covariate
            linear_coef = np.polyfit(x_norm, y, 1)
            y_pred_linear = np.polyval(linear_coef, x_norm)
            ss_res_linear = np.sum((y - y_pred_linear) ** 2)
            
            # Fit quadratic model: residual ~ covariate + covariate²
            quad_coef = np.polyfit(x_norm, y, 2)
            y_pred_quad = np.polyval(quad_coef, x_norm)
            ss_res_quad = np.sum((y - y_pred_quad) ** 2)
            
            # Quadratic coefficient (a in ax² + bx + c)
            a = quad_coef[0]
            
            # Standard error of quadratic coefficient (approximate)
            n = len(y)
            mse = ss_res_quad / (n - 3)
            x2_var = np.var(x_norm ** 2)
            se_a = np.sqrt(mse / (n * x2_var)) if x2_var > 0 else np.inf
            
            # Z-score for quadratic coefficient
            z_score = a / se_a if se_a > 0 and se_a != np.inf else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            # Model comparison (F-test)
            df1, df2 = 1, n - 3
            f_stat = ((ss_res_linear - ss_res_quad) / df1) / (ss_res_quad / df2) if ss_res_quad > 0 else 0
            f_pval = 1 - stats.f.cdf(f_stat, df1, df2) if f_stat > 0 else 1
            
            # Is it a U-shape (positive a) or inverted U-shape (negative a)?
            shape = "U-shape" if a > 0 else "Inverted-U"
            significant = p_value < 0.05
            
            results.append({
                'Residual': resid_col.replace('_resid', ''),
                'Covariate': covar,
                'Quadratic_coef': a,
                'SE': se_a,
                'Z_score': z_score,
                'P_value': p_value,
                'F_statistic': f_stat,
                'F_pvalue': f_pval,
                'Shape': shape if significant else 'Linear',
                'Significant': significant
            })
            
            if significant:
                print(f"\n{resid_col.replace('_resid', '')} vs {covar}:")
                print(f"  Quadratic coef (a): {a:+.4f} ± {se_a:.4f}")
                print(f"  Z = {z_score:.2f}, p = {p_value:.4f}")
                print(f"  Shape: {shape}")
                print(f"  F-test (quad vs linear): F = {f_stat:.2f}, p = {f_pval:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS / "test2_ushape_detection.csv", index=False)
    
    # Summary
    sig_ushapes = results_df[results_df['Significant']]
    
    print("\n" + "-" * 40)
    print("SUMMARY:")
    print(f"  Total combinations tested: {len(results_df)}")
    print(f"  Significant U/Inverted-U shapes: {len(sig_ushapes)}")
    
    if len(sig_ushapes) > 0:
        print("\n  Significant patterns:")
        for _, row in sig_ushapes.iterrows():
            print(f"    {row['Residual']} vs {row['Covariate']}: {row['Shape']} (p={row['P_value']:.4f})")
    
    return results_df


def test_classification_improvement(df):
    """
    TEST 3: Do residuals improve classification beyond ratios alone?
    
    If residuals contain diagnostic information, adding them to a
    classifier should improve predictive performance.
    """
    print("\n" + "=" * 60)
    print("TEST 3: CLASSIFICATION IMPROVEMENT")
    print("=" * 60)
    
    ratio_cols = ['HOMA_IR', 'LAR', 'Resistin_Adiponectin', 
                  'MCP1_Adiponectin', 'Leptin_BMI', 'Glucose_Insulin']
    residual_cols = [col for col in df.columns if col.endswith('_resid')]
    
    # Prepare target
    y = df['Cancer'].values
    
    # Remove rows with any NaN in features
    feature_cols = ratio_cols + residual_cols
    mask = df[feature_cols].notna().all(axis=1)
    y = y[mask]
    
    # Standardize features
    scaler = StandardScaler()
    
    # Model 1: Ratios only
    X_ratios = df.loc[mask, ratio_cols].values
    X_ratios_scaled = scaler.fit_transform(X_ratios)
    
    # Model 2: Ratios + Residuals
    X_full = df.loc[mask, feature_cols].values
    X_full_scaled = scaler.fit_transform(X_full)
    
    # Model 3: Residuals only
    X_resid = df.loc[mask, residual_cols].values
    X_resid_scaled = scaler.fit_transform(X_resid)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Logistic regression with regularization
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Evaluate each feature set
    results = {}
    
    print("\nCross-validated AUC-ROC scores:")
    
    for name, X in [('Ratios only', X_ratios_scaled), 
                    ('Residuals only', X_resid_scaled),
                    ('Ratios + Residuals', X_full_scaled)]:
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        results[name] = {
            'mean_auc': scores.mean(),
            'std_auc': scores.std(),
            'scores': scores
        }
        print(f"\n  {name}:")
        print(f"    AUC = {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"    Folds: {[f'{s:.3f}' for s in scores]}")
    
    # Statistical comparison
    print("\n" + "-" * 40)
    print("IMPROVEMENT ANALYSIS:")
    
    ratios_scores = results['Ratios only']['scores']
    full_scores = results['Ratios + Residuals']['scores']
    resid_scores = results['Residuals only']['scores']
    
    # Paired t-test: Does adding residuals help?
    t_stat, t_pval = stats.ttest_rel(full_scores, ratios_scores)
    improvement = full_scores.mean() - ratios_scores.mean()
    
    print(f"\n  Adding residuals to ratios:")
    print(f"    Improvement: {improvement:+.3f} AUC")
    print(f"    Paired t-test: t = {t_stat:.2f}, p = {t_pval:.4f}")
    print(f"    Significant: {'Yes' if t_pval < 0.05 else 'No'}")
    
    # Are residuals alone useful?
    print(f"\n  Residuals alone performance:")
    print(f"    AUC = {resid_scores.mean():.3f}")
    print(f"    vs random (0.5): {'Informative' if resid_scores.mean() > 0.5 else 'Not informative'}")
    
    # Save results
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Mean_AUC': [r['mean_auc'] for r in results.values()],
        'Std_AUC': [r['std_auc'] for r in results.values()]
    })
    results_df.to_csv(RESULTS / "test3_classification.csv", index=False)
    
    return results, improvement, t_pval


def visualize_residual_patterns(df):
    """Generate visualizations of residual patterns."""
    print("\n" + "=" * 60)
    print("GENERATING RESIDUAL VISUALIZATIONS")
    print("=" * 60)
    
    residual_cols = [col for col in df.columns if col.endswith('_resid')]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Residual distributions by group
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, resid_col in enumerate(residual_cols[:6]):
        ax = axes[i]
        
        for cancer, color, label in [(0, 'green', 'Healthy'), (1, 'red', 'Cancer')]:
            data = df.loc[df['Cancer'] == cancer, resid_col].dropna()
            ax.hist(data, bins=20, alpha=0.5, color=color, label=label, density=True)
        
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel(f'{resid_col.replace("_resid", "")} Residual')
        ax.set_ylabel('Density')
        ax.set_title(f'Residual Distribution')
        ax.legend()
    
    plt.suptitle('Residual Distributions by Cancer Status\n(The "Discarded" Information)', fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES / 'fig07_residual_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Figure 7: Residual distributions saved")
    
    # 2. Residuals vs. Age (looking for U-shape)
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, resid_col in enumerate(residual_cols[:6]):
        ax = axes[i]
        
        for cancer, color, marker, label in [(0, 'green', 'o', 'Healthy'), 
                                               (1, 'red', 's', 'Cancer')]:
            mask = df['Cancer'] == cancer
            ax.scatter(df.loc[mask, 'Age'], df.loc[mask, resid_col], 
                      c=color, marker=marker, alpha=0.5, label=label, s=30)
        
        # Fit and plot quadratic
        mask = df[resid_col].notna()
        x = df.loc[mask, 'Age'].values
        y = df.loc[mask, resid_col].values
        
        # Fit
        x_norm = (x - x.mean()) / x.std()
        coef = np.polyfit(x_norm, y, 2)
        
        # Plot curve
        x_plot = np.linspace(x.min(), x.max(), 100)
        x_plot_norm = (x_plot - x.mean()) / x.std()
        y_plot = np.polyval(coef, x_plot_norm)
        ax.plot(x_plot, y_plot, 'b-', linewidth=2, label=f'Quadratic fit (a={coef[0]:.3f})')
        
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Age')
        ax.set_ylabel('Residual')
        ax.set_title(resid_col.replace('_resid', ''))
        ax.legend(fontsize=8)
    
    plt.suptitle('Residuals vs. Age: Looking for U-Shapes', fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES / 'fig08_residuals_vs_age.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Figure 8: Residuals vs. Age saved")
    
    # 3. Residuals vs. BMI
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, resid_col in enumerate(residual_cols[:6]):
        ax = axes[i]
        
        for cancer, color, marker, label in [(0, 'green', 'o', 'Healthy'), 
                                               (1, 'red', 's', 'Cancer')]:
            mask = df['Cancer'] == cancer
            ax.scatter(df.loc[mask, 'BMI'], df.loc[mask, resid_col], 
                      c=color, marker=marker, alpha=0.5, label=label, s=30)
        
        # Fit quadratic
        mask = df[resid_col].notna()
        x = df.loc[mask, 'BMI'].values
        y = df.loc[mask, resid_col].values
        x_norm = (x - x.mean()) / x.std()
        coef = np.polyfit(x_norm, y, 2)
        
        x_plot = np.linspace(x.min(), x.max(), 100)
        x_plot_norm = (x_plot - x.mean()) / x.std()
        y_plot = np.polyval(coef, x_plot_norm)
        ax.plot(x_plot, y_plot, 'b-', linewidth=2, label=f'Quadratic (a={coef[0]:.3f})')
        
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('BMI')
        ax.set_ylabel('Residual')
        ax.set_title(resid_col.replace('_resid', ''))
        ax.legend(fontsize=8)
    
    plt.suptitle('Residuals vs. BMI: Looking for U-Shapes', fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES / 'fig09_residuals_vs_bmi.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Figure 9: Residuals vs. BMI saved")


def generate_summary_report(test1_results, test2_results, test3_results):
    """Generate a summary report of all findings."""
    
    improvement, t_pval = test3_results[1], test3_results[2]
    
    report = []
    report.append("=" * 70)
    report.append("PAPER 2: RESIDUAL ANALYSIS SUMMARY REPORT")
    report.append("The Revelatory Division - Testing the Hypothesis")
    report.append("=" * 70)
    report.append("")
    
    # Test 1 summary
    report.append("TEST 1: RESIDUAL DISTRIBUTION DIFFERENCES")
    report.append("-" * 40)
    n_dist = test1_results['Distribution_differs'].sum()
    n_loc = test1_results['Location_differs'].sum()
    report.append(f"  Residuals with different distributions: {n_dist}/6")
    report.append(f"  Residuals with different locations: {n_loc}/6")
    
    if n_dist > 0 or n_loc > 0:
        report.append("  → FINDING: Residuals contain group-discriminating information")
    else:
        report.append("  → FINDING: No significant group differences in residuals")
    report.append("")
    
    # Test 2 summary
    report.append("TEST 2: U-SHAPE DETECTION")
    report.append("-" * 40)
    sig_ushapes = test2_results[test2_results['Significant']]
    report.append(f"  Significant U-shapes detected: {len(sig_ushapes)}/{len(test2_results)}")
    
    if len(sig_ushapes) > 0:
        report.append("  Patterns found:")
        for _, row in sig_ushapes.iterrows():
            report.append(f"    - {row['Residual']} vs {row['Covariate']}: {row['Shape']}")
        report.append("  → FINDING: Non-linear patterns exist in residuals")
    else:
        report.append("  → FINDING: No significant U-shapes detected")
    report.append("")
    
    # Test 3 summary
    report.append("TEST 3: CLASSIFICATION IMPROVEMENT")
    report.append("-" * 40)
    report.append(f"  AUC improvement from adding residuals: {improvement:+.3f}")
    report.append(f"  Statistical significance: p = {t_pval:.4f}")
    
    if t_pval < 0.05 and improvement > 0:
        report.append("  → FINDING: Residuals significantly improve classification")
    elif improvement > 0:
        report.append("  → FINDING: Marginal improvement, not statistically significant")
    else:
        report.append("  → FINDING: No improvement from adding residuals")
    report.append("")
    
    # Overall conclusion
    report.append("=" * 70)
    report.append("OVERALL CONCLUSION")
    report.append("=" * 70)
    
    positive_findings = (
        (n_dist > 0 or n_loc > 0) + 
        (len(sig_ushapes) > 0) + 
        (improvement > 0 and t_pval < 0.05)
    )
    
    if positive_findings >= 2:
        report.append("The Revelatory Division hypothesis is SUPPORTED by the data.")
        report.append("Residuals from biomarker ratios contain structured diagnostic information.")
    elif positive_findings == 1:
        report.append("The hypothesis receives PARTIAL SUPPORT.")
        report.append("Some evidence suggests residuals contain information, but findings are mixed.")
    else:
        report.append("The hypothesis is NOT SUPPORTED by this dataset.")
        report.append("Residuals do not appear to contain significant diagnostic information.")
    
    report.append("")
    report.append("Note: This is exploratory research on a small dataset (N=116).")
    report.append("Findings require replication on larger, independent datasets.")
    report.append("")
    
    # Save report
    report_text = "\n".join(report)
    with open(RESULTS / "analysis_summary.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    
    return report_text


def main():
    """Main execution function."""
    print("=" * 60)
    print("PAPER 2: RESIDUAL ANALYSIS")
    print("Testing the Revelatory Division Hypothesis")
    print("=" * 60)
    
    # Load data
    df = load_processed_data()
    
    # Run tests
    test1_results = test_residual_distributions(df)
    test2_results = test_ushape_in_residuals(df)
    test3_results = test_classification_improvement(df)
    
    # Visualizations
    visualize_residual_patterns(df)
    
    # Summary report
    generate_summary_report(test1_results, test2_results, test3_results)
    
    print("\n✓ Analysis complete!")
    print(f"  Results saved to: {RESULTS}")
    print(f"  Figures saved to: {FIGURES}")
    print("\nNext step: Run 05_statistical_tests.py for robustness checks")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
