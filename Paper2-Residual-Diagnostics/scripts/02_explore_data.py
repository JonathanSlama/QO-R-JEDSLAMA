#!/usr/bin/env python3
"""
02_explore_data.py
==================
Initial exploratory data analysis of the Breast Cancer Coimbra dataset.

Author: Jonathan Édouard Slama
Email: jonathan@metafund.in
Date: December 2025

This script performs:
1. Data loading and validation
2. Distribution analysis for each variable
3. Correlation analysis
4. Group comparisons (cancer vs. healthy)
5. Visualization generation
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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "figures"

# Ensure figures directory exists
FIGURES.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load the dataset from CSV."""
    data_path = DATA_RAW / "breast_cancer_coimbra_full.csv"
    
    if not data_path.exists():
        print(f"ERROR: Data not found at {data_path}")
        print("Run 01_download_data.py first.")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    
    # Rename target for clarity
    df['Cancer'] = df['Classification'].map({1: 0, 2: 1})  # 0=Healthy, 1=Cancer
    
    print(f"✓ Data loaded: {len(df)} samples, {len(df.columns)} columns")
    return df


def analyze_distributions(df):
    """Analyze distribution of each variable."""
    print("\n" + "=" * 60)
    print("DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    features = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 
                'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']
    
    results = []
    
    for feat in features:
        if feat not in df.columns:
            continue
            
        data = df[feat].dropna()
        
        # Normality test
        stat, pval = stats.shapiro(data)
        is_normal = pval > 0.05
        
        # Skewness and kurtosis
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)
        
        results.append({
            'Variable': feat,
            'Mean': data.mean(),
            'Std': data.std(),
            'Median': data.median(),
            'Skewness': skew,
            'Kurtosis': kurt,
            'Shapiro p': pval,
            'Normal?': 'Yes' if is_normal else 'No'
        })
        
        print(f"\n{feat}:")
        print(f"  Mean ± SD: {data.mean():.2f} ± {data.std():.2f}")
        print(f"  Median: {data.median():.2f}")
        print(f"  Range: [{data.min():.2f}, {data.max():.2f}]")
        print(f"  Skewness: {skew:.2f}, Kurtosis: {kurt:.2f}")
        print(f"  Normality (Shapiro): p = {pval:.4f} → {'Normal' if is_normal else 'Non-normal'}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(DATA_PROCESSED / "distribution_analysis.csv", index=False)
    print(f"\n✓ Results saved to {DATA_PROCESSED / 'distribution_analysis.csv'}")
    
    return results_df


def analyze_group_differences(df):
    """Compare variables between cancer and healthy groups."""
    print("\n" + "=" * 60)
    print("GROUP COMPARISON: Cancer vs. Healthy")
    print("=" * 60)
    
    features = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 
                'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']
    
    healthy = df[df['Cancer'] == 0]
    cancer = df[df['Cancer'] == 1]
    
    print(f"\nSample sizes: Healthy = {len(healthy)}, Cancer = {len(cancer)}")
    
    results = []
    
    for feat in features:
        if feat not in df.columns:
            continue
            
        h_data = healthy[feat].dropna()
        c_data = cancer[feat].dropna()
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pval = stats.mannwhitneyu(h_data, c_data, alternative='two-sided')
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(h_data), len(c_data)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)
        
        # T-test for comparison
        t_stat, t_pval = stats.ttest_ind(h_data, c_data)
        
        results.append({
            'Variable': feat,
            'Healthy_Mean': h_data.mean(),
            'Cancer_Mean': c_data.mean(),
            'Difference': c_data.mean() - h_data.mean(),
            'Pct_Change': 100 * (c_data.mean() - h_data.mean()) / h_data.mean(),
            'U_statistic': u_stat,
            'U_pvalue': u_pval,
            'Effect_size': effect_size,
            'Significant': u_pval < 0.05
        })
        
        sig = "***" if u_pval < 0.001 else "**" if u_pval < 0.01 else "*" if u_pval < 0.05 else ""
        print(f"\n{feat}: {sig}")
        print(f"  Healthy: {h_data.mean():.2f} ± {h_data.std():.2f}")
        print(f"  Cancer:  {c_data.mean():.2f} ± {c_data.std():.2f}")
        print(f"  Difference: {c_data.mean() - h_data.mean():+.2f} ({100*(c_data.mean()-h_data.mean())/h_data.mean():+.1f}%)")
        print(f"  Mann-Whitney p = {u_pval:.4f}, Effect size = {effect_size:.3f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(DATA_PROCESSED / "group_comparison.csv", index=False)
    print(f"\n✓ Results saved to {DATA_PROCESSED / 'group_comparison.csv'}")
    
    # Summary of significant differences
    sig_vars = results_df[results_df['Significant']]['Variable'].tolist()
    print(f"\n📊 Significant differences (p < 0.05): {', '.join(sig_vars)}")
    
    return results_df


def analyze_correlations(df):
    """Compute and visualize correlation matrix."""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    
    features = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 
                'Leptin', 'Adiponectin', 'Resistin', 'MCP.1', 'Cancer']
    
    # Filter to available features
    features = [f for f in features if f in df.columns]
    
    corr_matrix = df[features].corr()
    
    # Save correlation matrix
    corr_matrix.to_csv(DATA_PROCESSED / "correlation_matrix.csv")
    
    # Find strongest correlations with Cancer
    cancer_corrs = corr_matrix['Cancer'].drop('Cancer').sort_values(key=abs, ascending=False)
    
    print("\nCorrelations with Cancer (sorted by magnitude):")
    for var, corr in cancer_corrs.items():
        print(f"  {var}: r = {corr:+.3f}")
    
    return corr_matrix


def create_visualizations(df):
    """Generate exploratory visualizations."""
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    features = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 
                'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']
    features = [f for f in features if f in df.columns]
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Distribution plots by group
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, feat in enumerate(features):
        ax = axes[i]
        
        # KDE plots for each group
        for cancer, color, label in [(0, 'green', 'Healthy'), (1, 'red', 'Cancer')]:
            data = df[df['Cancer'] == cancer][feat].dropna()
            data.plot.kde(ax=ax, color=color, label=label, alpha=0.7)
        
        ax.set_xlabel(feat)
        ax.set_ylabel('Density')
        ax.set_title(f'{feat} Distribution')
        ax.legend()
    
    plt.tight_layout()
    fig.savefig(FIGURES / 'fig01_distributions_by_group.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Figure 1: Distributions by group saved")
    
    # 2. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[features + ['Cancer']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Correlation Matrix')
    plt.tight_layout()
    fig.savefig(FIGURES / 'fig02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Figure 2: Correlation heatmap saved")
    
    # 3. Box plots comparing groups
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, feat in enumerate(features):
        ax = axes[i]
        df.boxplot(column=feat, by='Cancer', ax=ax)
        ax.set_xlabel('Group (0=Healthy, 1=Cancer)')
        ax.set_ylabel(feat)
        ax.set_title(feat)
    
    plt.suptitle('Feature Distributions by Group', fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES / 'fig03_boxplots_by_group.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Figure 3: Box plots by group saved")
    
    # 4. Pairplot for key variables
    key_vars = ['Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Cancer']
    key_vars = [v for v in key_vars if v in df.columns]
    
    g = sns.pairplot(df[key_vars], hue='Cancer', 
                     palette={0: 'green', 1: 'red'},
                     diag_kind='kde', 
                     plot_kws={'alpha': 0.6})
    g.fig.suptitle('Pairwise Relationships (Key Variables)', y=1.02)
    g.savefig(FIGURES / 'fig04_pairplot_key_variables.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Figure 4: Pairplot saved")
    
    print(f"\n✓ All figures saved to {FIGURES}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("PAPER 2: EXPLORATORY DATA ANALYSIS")
    print("The Revelatory Division - Residual Diagnostics")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Analyses
    dist_results = analyze_distributions(df)
    group_results = analyze_group_differences(df)
    corr_matrix = analyze_correlations(df)
    
    # Visualizations
    create_visualizations(df)
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPLORATORY ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nKey Findings:")
    print("  1. Most variables are non-normally distributed (right-skewed)")
    print("  2. Several biomarkers differ significantly between groups")
    print("  3. Strong correlations exist among metabolic markers")
    print("\nNext step: Run 03_compute_ratios.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
