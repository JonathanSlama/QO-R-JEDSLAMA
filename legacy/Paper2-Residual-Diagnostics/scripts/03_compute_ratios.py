#!/usr/bin/env python3
"""
03_compute_ratios.py
====================
Compute clinically relevant biomarker ratios and prepare for residual analysis.

Author: Jonathan Édouard Slama
Email: jonathan@metafund.in
Date: December 2025

Ratios computed:
1. HOMA-IR: (Glucose × Insulin) / 405 - Insulin resistance index
2. Leptin/Adiponectin: Metabolic dysfunction marker
3. Resistin/Adiponectin: Inflammatory marker
4. MCP.1/Adiponectin: Inflammatory marker
5. Leptin/BMI: Body composition adjustment
6. Glucose/BMI: Metabolic rate proxy

The Revelatory Division Concept:
- Each ratio is a "quotient" (Q) - what we typically analyze
- The residuals from ratio models are the "reliquat" (R) - what we usually discard
- This script prepares both Q and R for analysis
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


def load_data():
    """Load the dataset."""
    data_path = DATA_RAW / "breast_cancer_coimbra_full.csv"
    
    if not data_path.exists():
        print(f"ERROR: Data not found at {data_path}")
        print("Run 01_download_data.py first.")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    df['Cancer'] = df['Classification'].map({1: 0, 2: 1})
    
    print(f"✓ Data loaded: {len(df)} samples")
    return df


def compute_ratios(df):
    """
    Compute clinically relevant biomarker ratios.
    
    These ratios represent the "Quotient" (Q) in our framework.
    """
    print("\n" + "=" * 60)
    print("COMPUTING BIOMARKER RATIOS (Quotients)")
    print("=" * 60)
    
    ratios = pd.DataFrame(index=df.index)
    
    # 1. HOMA-IR: Homeostatic Model Assessment of Insulin Resistance
    #    Standard formula: (Glucose [mg/dL] × Insulin [µU/mL]) / 405
    ratios['HOMA_IR'] = (df['Glucose'] * df['Insulin']) / 405
    print("\n1. HOMA-IR = (Glucose × Insulin) / 405")
    print(f"   Range: [{ratios['HOMA_IR'].min():.2f}, {ratios['HOMA_IR'].max():.2f}]")
    print(f"   Mean ± SD: {ratios['HOMA_IR'].mean():.2f} ± {ratios['HOMA_IR'].std():.2f}")
    
    # Note: Dataset already includes HOMA - we compute our own for transparency
    if 'HOMA' in df.columns:
        corr = ratios['HOMA_IR'].corr(df['HOMA'])
        print(f"   Correlation with dataset HOMA: r = {corr:.4f}")
    
    # 2. Leptin/Adiponectin Ratio (LAR)
    #    Higher values associated with metabolic dysfunction and cancer risk
    ratios['LAR'] = df['Leptin'] / df['Adiponectin']
    print("\n2. LAR (Leptin/Adiponectin)")
    print(f"   Range: [{ratios['LAR'].min():.2f}, {ratios['LAR'].max():.2f}]")
    print(f"   Mean ± SD: {ratios['LAR'].mean():.2f} ± {ratios['LAR'].std():.2f}")
    
    # 3. Resistin/Adiponectin Ratio
    #    Inflammatory marker
    ratios['Resistin_Adiponectin'] = df['Resistin'] / df['Adiponectin']
    print("\n3. Resistin/Adiponectin")
    print(f"   Range: [{ratios['Resistin_Adiponectin'].min():.2f}, {ratios['Resistin_Adiponectin'].max():.2f}]")
    print(f"   Mean ± SD: {ratios['Resistin_Adiponectin'].mean():.2f} ± {ratios['Resistin_Adiponectin'].std():.2f}")
    
    # 4. MCP-1/Adiponectin Ratio
    #    Inflammatory marker
    ratios['MCP1_Adiponectin'] = df['MCP.1'] / df['Adiponectin']
    print("\n4. MCP-1/Adiponectin")
    print(f"   Range: [{ratios['MCP1_Adiponectin'].min():.2f}, {ratios['MCP1_Adiponectin'].max():.2f}]")
    print(f"   Mean ± SD: {ratios['MCP1_Adiponectin'].mean():.2f} ± {ratios['MCP1_Adiponectin'].std():.2f}")
    
    # 5. Leptin/BMI - Body composition adjustment
    ratios['Leptin_BMI'] = df['Leptin'] / df['BMI']
    print("\n5. Leptin/BMI")
    print(f"   Range: [{ratios['Leptin_BMI'].min():.2f}, {ratios['Leptin_BMI'].max():.2f}]")
    print(f"   Mean ± SD: {ratios['Leptin_BMI'].mean():.2f} ± {ratios['Leptin_BMI'].std():.2f}")
    
    # 6. Glucose/Insulin - Simplified insulin sensitivity
    ratios['Glucose_Insulin'] = df['Glucose'] / df['Insulin']
    print("\n6. Glucose/Insulin (inverse of insulin effect)")
    print(f"   Range: [{ratios['Glucose_Insulin'].min():.2f}, {ratios['Glucose_Insulin'].max():.2f}]")
    print(f"   Mean ± SD: {ratios['Glucose_Insulin'].mean():.2f} ± {ratios['Glucose_Insulin'].std():.2f}")
    
    # 7. Log-transformed ratios (for normalization)
    ratios['log_LAR'] = np.log1p(ratios['LAR'])
    ratios['log_HOMA_IR'] = np.log1p(ratios['HOMA_IR'])
    
    return ratios


def analyze_ratio_distributions(ratios, df):
    """Analyze distribution properties of computed ratios."""
    print("\n" + "=" * 60)
    print("RATIO DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    ratio_cols = ['HOMA_IR', 'LAR', 'Resistin_Adiponectin', 
                  'MCP1_Adiponectin', 'Leptin_BMI', 'Glucose_Insulin']
    
    results = []
    
    for ratio in ratio_cols:
        data = ratios[ratio].dropna()
        
        # Normality test
        stat, pval = stats.shapiro(data)
        
        # Skewness
        skew = stats.skew(data)
        
        # Compare groups
        healthy = ratios.loc[df['Cancer'] == 0, ratio].dropna()
        cancer = ratios.loc[df['Cancer'] == 1, ratio].dropna()
        u_stat, u_pval = stats.mannwhitneyu(healthy, cancer, alternative='two-sided')
        
        results.append({
            'Ratio': ratio,
            'Mean': data.mean(),
            'Std': data.std(),
            'Skewness': skew,
            'Shapiro_p': pval,
            'Normal': pval > 0.05,
            'Healthy_Mean': healthy.mean(),
            'Cancer_Mean': cancer.mean(),
            'U_pvalue': u_pval,
            'Significant': u_pval < 0.05
        })
        
        sig = "***" if u_pval < 0.001 else "**" if u_pval < 0.01 else "*" if u_pval < 0.05 else ""
        print(f"\n{ratio}: {sig}")
        print(f"  Shapiro p = {pval:.4f} → {'Normal' if pval > 0.05 else 'Non-normal'}")
        print(f"  Skewness: {skew:.2f}")
        print(f"  Healthy: {healthy.mean():.3f} ± {healthy.std():.3f}")
        print(f"  Cancer:  {cancer.mean():.3f} ± {cancer.std():.3f}")
        print(f"  Group difference: p = {u_pval:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(DATA_PROCESSED / "ratio_analysis.csv", index=False)
    
    return results_df


def compute_ratio_residuals(ratios, df):
    """
    Compute residuals from ratio-based models.
    
    These residuals represent the "Reliquat" (R) in our framework.
    
    For each ratio, we:
    1. Fit a simple model: Ratio ~ f(covariates)
    2. Extract residuals: observed - predicted
    3. Analyze residual structure
    """
    print("\n" + "=" * 60)
    print("COMPUTING RESIDUALS (Reliquats)")
    print("=" * 60)
    
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    residuals = pd.DataFrame(index=df.index)
    
    # Covariates for adjustment
    covariates = ['Age', 'BMI']
    X_cov = df[covariates].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cov)
    
    ratio_cols = ['HOMA_IR', 'LAR', 'Resistin_Adiponectin', 
                  'MCP1_Adiponectin', 'Leptin_BMI', 'Glucose_Insulin']
    
    for ratio in ratio_cols:
        y = ratios[ratio].values
        
        # Handle any NaN
        mask = ~np.isnan(y)
        
        # Fit linear regression: Ratio ~ Age + BMI
        model = LinearRegression()
        model.fit(X_scaled[mask], y[mask])
        
        # Predictions and residuals
        y_pred = np.full_like(y, np.nan)
        y_pred[mask] = model.predict(X_scaled[mask])
        
        resid = y - y_pred
        residuals[f'{ratio}_resid'] = resid
        
        print(f"\n{ratio}:")
        print(f"  Model: {ratio} ~ Age + BMI")
        print(f"  R² = {model.score(X_scaled[mask], y[mask]):.4f}")
        print(f"  Residual mean: {np.nanmean(resid):.6f} (should be ~0)")
        print(f"  Residual std: {np.nanstd(resid):.4f}")
    
    return residuals


def visualize_ratios(ratios, df):
    """Generate visualizations for computed ratios."""
    print("\n" + "=" * 60)
    print("GENERATING RATIO VISUALIZATIONS")
    print("=" * 60)
    
    ratio_cols = ['HOMA_IR', 'LAR', 'Resistin_Adiponectin', 
                  'MCP1_Adiponectin', 'Leptin_BMI', 'Glucose_Insulin']
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Ratio distributions by group
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, ratio in enumerate(ratio_cols):
        ax = axes[i]
        
        for cancer, color, label in [(0, 'green', 'Healthy'), (1, 'red', 'Cancer')]:
            data = ratios.loc[df['Cancer'] == cancer, ratio].dropna()
            ax.hist(data, bins=20, alpha=0.5, color=color, label=label, density=True)
        
        ax.set_xlabel(ratio)
        ax.set_ylabel('Density')
        ax.set_title(f'{ratio} by Group')
        ax.legend()
    
    plt.suptitle('Biomarker Ratios: Distribution by Cancer Status', fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES / 'fig05_ratio_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Figure 5: Ratio distributions saved")
    
    # 2. Ratio correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = ratios[ratio_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Correlation Between Biomarker Ratios')
    plt.tight_layout()
    fig.savefig(FIGURES / 'fig06_ratio_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Figure 6: Ratio correlations saved")


def save_processed_data(df, ratios, residuals):
    """Save all processed data for subsequent analysis."""
    
    # Combine everything
    full_data = pd.concat([df, ratios, residuals], axis=1)
    
    # Save
    output_path = DATA_PROCESSED / "data_with_ratios_and_residuals.csv"
    full_data.to_csv(output_path, index=False)
    print(f"\n✓ Full processed data saved to: {output_path}")
    
    # Save just ratios
    ratios.to_csv(DATA_PROCESSED / "computed_ratios.csv", index=False)
    
    # Save just residuals
    residuals.to_csv(DATA_PROCESSED / "computed_residuals.csv", index=False)
    
    return full_data


def main():
    """Main execution function."""
    print("=" * 60)
    print("PAPER 2: COMPUTE BIOMARKER RATIOS")
    print("The Revelatory Division - Quotients and Reliquats")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Compute ratios (Quotients)
    ratios = compute_ratios(df)
    
    # Analyze ratio distributions
    ratio_analysis = analyze_ratio_distributions(ratios, df)
    
    # Compute residuals (Reliquats)
    residuals = compute_ratio_residuals(ratios, df)
    
    # Visualizations
    visualize_ratios(ratios, df)
    
    # Save processed data
    full_data = save_processed_data(df, ratios, residuals)
    
    # Summary
    print("\n" + "=" * 60)
    print("RATIO COMPUTATION COMPLETE")
    print("=" * 60)
    print("\nComputed:")
    print(f"  - 6 biomarker ratios (Quotients)")
    print(f"  - 6 residual series (Reliquats)")
    print("\nSignificant group differences in ratios:")
    sig_ratios = ratio_analysis[ratio_analysis['Significant']]['Ratio'].tolist()
    for r in sig_ratios:
        print(f"  - {r}")
    print("\nNext step: Run 04_residual_analysis.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
