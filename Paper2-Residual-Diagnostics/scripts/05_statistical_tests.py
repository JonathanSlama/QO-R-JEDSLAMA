#!/usr/bin/env python3
"""
05_statistical_tests.py
=======================
Robustness testing and validation of findings.

Author: Jonathan Édouard Slama
Email: jonathan@metafund.in
Date: December 2025

This script performs:
1. Bootstrap confidence intervals
2. Permutation tests
3. Multiple testing correction
4. Cross-validation stability
5. Sensitivity analysis
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, permutation_test_score
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "figures"
RESULTS = PROJECT_ROOT / "results"


def load_data():
    """Load processed data."""
    df = pd.read_csv(DATA_PROCESSED / "data_with_ratios_and_residuals.csv")
    print(f"✓ Data loaded: {len(df)} samples")
    return df


def bootstrap_group_differences(df, n_bootstrap=1000):
    """
    Bootstrap confidence intervals for group differences in residuals.
    """
    print("\n" + "=" * 60)
    print("ROBUSTNESS TEST 1: BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 60)
    
    residual_cols = [col for col in df.columns if col.endswith('_resid')]
    
    np.random.seed(42)
    results = []
    
    for resid_col in residual_cols:
        healthy = df.loc[df['Cancer'] == 0, resid_col].dropna().values
        cancer = df.loc[df['Cancer'] == 1, resid_col].dropna().values
        
        # Observed difference
        obs_diff = cancer.mean() - healthy.mean()
        
        # Bootstrap
        boot_diffs = []
        for _ in range(n_bootstrap):
            h_sample = np.random.choice(healthy, size=len(healthy), replace=True)
            c_sample = np.random.choice(cancer, size=len(cancer), replace=True)
            boot_diffs.append(c_sample.mean() - h_sample.mean())
        
        boot_diffs = np.array(boot_diffs)
        
        # Confidence interval
        ci_low = np.percentile(boot_diffs, 2.5)
        ci_high = np.percentile(boot_diffs, 97.5)
        
        # Does CI exclude zero?
        significant = (ci_low > 0) or (ci_high < 0)
        
        results.append({
            'Residual': resid_col.replace('_resid', ''),
            'Observed_diff': obs_diff,
            'CI_2.5%': ci_low,
            'CI_97.5%': ci_high,
            'CI_excludes_zero': significant
        })
        
        sig = "*" if significant else ""
        print(f"\n{resid_col.replace('_resid', '')}: {sig}")
        print(f"  Observed difference: {obs_diff:+.4f}")
        print(f"  95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
        print(f"  Excludes zero: {'Yes' if significant else 'No'}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS / "robustness_bootstrap.csv", index=False)
    
    n_sig = results_df['CI_excludes_zero'].sum()
    print(f"\nSummary: {n_sig}/{len(residual_cols)} residuals show significant group differences")
    
    return results_df


def permutation_test_classification(df, n_permutations=100):
    """
    Permutation test for classification improvement.
    """
    print("\n" + "=" * 60)
    print("ROBUSTNESS TEST 2: PERMUTATION TEST")
    print("=" * 60)
    
    ratio_cols = ['HOMA_IR', 'LAR', 'Resistin_Adiponectin', 
                  'MCP1_Adiponectin', 'Leptin_BMI', 'Glucose_Insulin']
    residual_cols = [col for col in df.columns if col.endswith('_resid')]
    feature_cols = ratio_cols + residual_cols
    
    # Prepare data
    mask = df[feature_cols].notna().all(axis=1)
    X = df.loc[mask, feature_cols].values
    y = df.loc[mask, 'Cancer'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"Running permutation test with {n_permutations} permutations...")
    
    score, perm_scores, pvalue = permutation_test_score(
        model, X_scaled, y, 
        scoring='roc_auc', 
        cv=cv, 
        n_permutations=n_permutations,
        random_state=42,
        n_jobs=-1
    )
    
    print(f"\n  True AUC: {score:.3f}")
    print(f"  Permuted AUC (mean ± std): {perm_scores.mean():.3f} ± {perm_scores.std():.3f}")
    print(f"  Permutation p-value: {pvalue:.4f}")
    print(f"  Significant: {'Yes' if pvalue < 0.05 else 'No'}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.hist(perm_scores, bins=30, alpha=0.7, label='Permuted scores')
    plt.axvline(score, color='red', linewidth=2, label=f'True score ({score:.3f})')
    plt.xlabel('AUC-ROC')
    plt.ylabel('Frequency')
    plt.title(f'Permutation Test (p = {pvalue:.4f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / 'fig10_permutation_test.png', dpi=150)
    plt.close()
    print("✓ Figure 10: Permutation test saved")
    
    return score, perm_scores, pvalue


def multiple_testing_correction(df):
    """
    Apply multiple testing correction to U-shape results.
    """
    print("\n" + "=" * 60)
    print("ROBUSTNESS TEST 3: MULTIPLE TESTING CORRECTION")
    print("=" * 60)
    
    # Load U-shape results
    ushape_path = RESULTS / "test2_ushape_detection.csv"
    if not ushape_path.exists():
        print("U-shape results not found. Run 04_residual_analysis.py first.")
        return None
    
    ushape_df = pd.read_csv(ushape_path)
    
    # Get p-values
    pvalues = ushape_df['P_value'].values
    n_tests = len(pvalues)
    
    # Bonferroni correction
    bonf_alpha = 0.05 / n_tests
    bonf_significant = pvalues < bonf_alpha
    
    # Benjamini-Hochberg (FDR)
    sorted_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_idx]
    
    bh_critical = np.arange(1, n_tests + 1) / n_tests * 0.05
    bh_significant_sorted = sorted_pvals <= bh_critical
    
    # Find the largest k where p(k) <= k/m * alpha
    if bh_significant_sorted.any():
        max_k = np.max(np.where(bh_significant_sorted)[0])
        bh_threshold = sorted_pvals[max_k]
    else:
        bh_threshold = 0
    
    bh_significant = pvalues <= bh_threshold
    
    # Add to dataframe
    ushape_df['Bonferroni_sig'] = bonf_significant
    ushape_df['BH_FDR_sig'] = bh_significant
    
    print(f"  Number of tests: {n_tests}")
    print(f"  Bonferroni threshold: {bonf_alpha:.6f}")
    print(f"  BH-FDR threshold: {bh_threshold:.6f}")
    print(f"\n  Significant at α=0.05 (uncorrected): {(pvalues < 0.05).sum()}")
    print(f"  Significant after Bonferroni: {bonf_significant.sum()}")
    print(f"  Significant after BH-FDR: {bh_significant.sum()}")
    
    if bh_significant.sum() > 0:
        print("\n  Patterns surviving FDR correction:")
        for _, row in ushape_df[ushape_df['BH_FDR_sig']].iterrows():
            print(f"    {row['Residual']} vs {row['Covariate']}: p = {row['P_value']:.4f}")
    
    ushape_df.to_csv(RESULTS / "test2_ushape_corrected.csv", index=False)
    
    return ushape_df


def sensitivity_analysis(df):
    """
    Sensitivity analysis: How robust are findings to outlier removal?
    """
    print("\n" + "=" * 60)
    print("ROBUSTNESS TEST 4: SENSITIVITY TO OUTLIERS")
    print("=" * 60)
    
    ratio_cols = ['HOMA_IR', 'LAR', 'Resistin_Adiponectin', 
                  'MCP1_Adiponectin', 'Leptin_BMI', 'Glucose_Insulin']
    residual_cols = [col for col in df.columns if col.endswith('_resid')]
    
    results = []
    
    for resid_col in residual_cols:
        original = df[resid_col].dropna()
        
        # Identify outliers (>3 SD from mean)
        mean, std = original.mean(), original.std()
        outliers = (np.abs(original - mean) > 3 * std)
        n_outliers = outliers.sum()
        
        # Test with and without outliers
        healthy_full = df.loc[df['Cancer'] == 0, resid_col].dropna()
        cancer_full = df.loc[df['Cancer'] == 1, resid_col].dropna()
        _, pval_full = stats.mannwhitneyu(healthy_full, cancer_full)
        
        # Remove outliers
        df_clean = df[~outliers.reindex(df.index, fill_value=False)]
        healthy_clean = df_clean.loc[df_clean['Cancer'] == 0, resid_col].dropna()
        cancer_clean = df_clean.loc[df_clean['Cancer'] == 1, resid_col].dropna()
        
        if len(healthy_clean) > 0 and len(cancer_clean) > 0:
            _, pval_clean = stats.mannwhitneyu(healthy_clean, cancer_clean)
        else:
            pval_clean = np.nan
        
        results.append({
            'Residual': resid_col.replace('_resid', ''),
            'N_outliers': n_outliers,
            'Pct_outliers': 100 * n_outliers / len(original),
            'P_value_full': pval_full,
            'P_value_no_outliers': pval_clean,
            'Conclusion_stable': (pval_full < 0.05) == (pval_clean < 0.05)
        })
        
        print(f"\n{resid_col.replace('_resid', '')}:")
        print(f"  Outliers (>3 SD): {n_outliers} ({100*n_outliers/len(original):.1f}%)")
        print(f"  p-value (full): {pval_full:.4f}")
        print(f"  p-value (no outliers): {pval_clean:.4f}")
        print(f"  Conclusion stable: {'Yes' if (pval_full < 0.05) == (pval_clean < 0.05) else 'No'}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS / "robustness_sensitivity.csv", index=False)
    
    n_stable = results_df['Conclusion_stable'].sum()
    print(f"\nSummary: {n_stable}/{len(residual_cols)} findings stable after outlier removal")
    
    return results_df


def generate_robustness_report():
    """Generate final robustness report."""
    
    report = []
    report.append("=" * 70)
    report.append("ROBUSTNESS ANALYSIS REPORT")
    report.append("Paper 2: The Revelatory Division")
    report.append("=" * 70)
    report.append("")
    
    # Load all results
    try:
        bootstrap = pd.read_csv(RESULTS / "robustness_bootstrap.csv")
        ushape = pd.read_csv(RESULTS / "test2_ushape_corrected.csv")
        sensitivity = pd.read_csv(RESULTS / "robustness_sensitivity.csv")
        
        report.append("1. BOOTSTRAP ANALYSIS")
        report.append("-" * 40)
        n_sig = bootstrap['CI_excludes_zero'].sum()
        report.append(f"   Residuals with significant group differences: {n_sig}/6")
        report.append("")
        
        report.append("2. MULTIPLE TESTING CORRECTION")
        report.append("-" * 40)
        n_fdr = ushape['BH_FDR_sig'].sum() if 'BH_FDR_sig' in ushape.columns else 0
        report.append(f"   U-shapes surviving FDR correction: {n_fdr}")
        report.append("")
        
        report.append("3. SENSITIVITY ANALYSIS")
        report.append("-" * 40)
        n_stable = sensitivity['Conclusion_stable'].sum()
        report.append(f"   Findings stable after outlier removal: {n_stable}/6")
        report.append("")
        
        # Overall robustness score
        robustness_score = (n_sig > 2) + (n_fdr > 0) + (n_stable >= 4)
        
        report.append("=" * 70)
        report.append("OVERALL ROBUSTNESS ASSESSMENT")
        report.append("=" * 70)
        
        if robustness_score >= 2:
            report.append("Findings are ROBUST to statistical scrutiny.")
        elif robustness_score == 1:
            report.append("Findings show MODERATE robustness.")
        else:
            report.append("Findings are NOT ROBUST and should be interpreted with caution.")
        
    except FileNotFoundError:
        report.append("Some result files not found. Run all analysis scripts first.")
    
    report.append("")
    report_text = "\n".join(report)
    
    with open(RESULTS / "robustness_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    
    return report_text


def main():
    """Main execution function."""
    print("=" * 60)
    print("PAPER 2: ROBUSTNESS TESTING")
    print("Validating the Revelatory Division Findings")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Run robustness tests
    bootstrap_results = bootstrap_group_differences(df, n_bootstrap=1000)
    perm_score, perm_scores, perm_pval = permutation_test_classification(df, n_permutations=100)
    ushape_corrected = multiple_testing_correction(df)
    sensitivity_results = sensitivity_analysis(df)
    
    # Generate report
    generate_robustness_report()
    
    print("\n✓ Robustness testing complete!")
    print(f"  All results saved to: {RESULTS}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
