#!/usr/bin/env python3
"""
Paper 2 Extended: Chronic Kidney Disease Analysis
The Revelatory Division - Testing Residual Diagnostics

Analyzes whether residuals from eGFR, ACR, and BUN/Cr contain
diagnostic information for CKD progression.

Author: Jonathan Édouard Slama
Email: jonathan@metafund.in
Affiliation: Metafund Research Division, Strasbourg, France
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
PROCESSED_DATA = PROJECT_DIR / "data" / "processed"
RESULTS = PROJECT_DIR / "results"
FIGURES = PROJECT_DIR / "figures"

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

def load_data() -> pd.DataFrame:
    filepath = PROCESSED_DATA / "nhanes_with_ratios.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Data not found: {filepath}")
    return pd.read_csv(filepath)

# =============================================================================
# CKD ANALYSIS
# =============================================================================

def analyze_ckd_stages(df: pd.DataFrame) -> dict:
    """Analyze CKD stage distribution."""
    
    print("\n" + "=" * 60)
    print("CKD STAGE ANALYSIS")
    print("=" * 60)
    
    results = {}
    
    if 'eGFR' in df.columns:
        # eGFR distribution
        egfr = df['eGFR'].dropna()
        print(f"\neGFR Statistics:")
        print(f"  Mean: {egfr.mean():.1f} mL/min/1.73m²")
        print(f"  Median: {egfr.median():.1f}")
        print(f"  Range: [{egfr.min():.1f}, {egfr.max():.1f}]")
        
        # CKD staging
        df['CKD_Stage_Num'] = pd.cut(df['eGFR'],
                                      bins=[0, 15, 30, 45, 60, 90, 200],
                                      labels=[5, 4, 3.5, 3, 2, 1])
        
        # Stage distribution
        print("\nCKD Stage Distribution (by eGFR):")
        stage_counts = df['CKD_Stage_Num'].value_counts().sort_index()
        for stage, count in stage_counts.items():
            pct = 100 * count / len(df['CKD_Stage_Num'].dropna())
            print(f"  Stage {stage}: {count} ({pct:.1f}%)")
        
        # Identify "gray zone" (eGFR 60-90)
        gray_zone = (df['eGFR'] >= 60) & (df['eGFR'] < 90)
        gray_pct = 100 * gray_zone.sum() / df['eGFR'].notna().sum()
        print(f"\n→ Gray zone (eGFR 60-90): {gray_pct:.1f}%")
        print("  These patients are 'mildly decreased' - progression uncertain")
        
        results['gray_zone_pct'] = gray_pct
    
    if 'ACR' in df.columns:
        acr = df['ACR'].dropna()
        print(f"\nACR Statistics:")
        print(f"  Median: {acr.median():.1f} mg/g")
        
        # Albuminuria categories
        normal = (df['ACR'] < 30).sum()
        micro = ((df['ACR'] >= 30) & (df['ACR'] < 300)).sum()
        macro = (df['ACR'] >= 300).sum()
        total = df['ACR'].notna().sum()
        
        print(f"  Normal (<30): {100*normal/total:.1f}%")
        print(f"  Microalbuminuria (30-300): {100*micro/total:.1f}%")
        print(f"  Macroalbuminuria (>300): {100*macro/total:.1f}%")
    
    return results

def test_residual_by_ckd_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Test residual differences across CKD stages."""
    
    print("\n" + "=" * 60)
    print("TEST 1: RESIDUALS BY CKD STAGE")
    print("=" * 60)
    
    results = []
    
    # Create CKD categories
    df['CKD_Category'] = pd.cut(df['eGFR'],
                                bins=[0, 60, 90, 200],
                                labels=['CKD ≥3', 'Mild decrease', 'Normal'])
    
    residual_cols = [c for c in df.columns if c.endswith('_Residual')]
    
    for resid_col in residual_cols:
        if resid_col not in df.columns:
            continue
        
        # Get data by category
        categories = ['Normal', 'Mild decrease', 'CKD ≥3']
        cat_data = {}
        
        for cat in categories:
            mask = (df['CKD_Category'] == cat) & df[resid_col].notna()
            cat_data[cat] = df.loc[mask, resid_col].values
        
        if any(len(cat_data.get(c, [])) < 30 for c in categories):
            continue
        
        # Kruskal-Wallis test
        valid_cats = [cat_data[c] for c in categories if len(cat_data.get(c, [])) > 0]
        if len(valid_cats) >= 2:
            h_stat, kw_p = stats.kruskal(*valid_cats)
        else:
            continue
        
        ratio_name = resid_col.replace('_Residual', '')
        
        results.append({
            'Ratio': ratio_name,
            'N_Normal': len(cat_data.get('Normal', [])),
            'N_Mild': len(cat_data.get('Mild decrease', [])),
            'N_CKD': len(cat_data.get('CKD ≥3', [])),
            'Mean_Normal': np.mean(cat_data.get('Normal', [np.nan])),
            'Mean_Mild': np.mean(cat_data.get('Mild decrease', [np.nan])),
            'Mean_CKD': np.mean(cat_data.get('CKD ≥3', [np.nan])),
            'KW_H': h_stat,
            'KW_p': kw_p,
            'Significant': kw_p < 0.05
        })
        
        if kw_p < 0.05:
            print(f"\n{ratio_name}: *** SIGNIFICANT ***")
            print(f"  H = {h_stat:.2f}, p = {kw_p:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS / "kidney_test1_residuals_by_stage.csv", index=False)
    
    n_sig = results_df['Significant'].sum() if len(results_df) > 0 else 0
    print(f"\n→ {n_sig}/{len(results_df)} residuals differ across CKD stages")
    
    return results_df

def test_acr_residual_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Test for patterns in ACR residuals."""
    
    print("\n" + "=" * 60)
    print("TEST 2: ACR RESIDUAL PATTERNS")
    print("=" * 60)
    
    if 'ACR_Residual' not in df.columns:
        print("ACR residuals not available")
        return pd.DataFrame()
    
    results = []
    
    # Test ACR residuals vs various covariates
    covariates = ['Age', 'BMI', 'SystolicBP', 'Glucose', 'HbA1c']
    covariates = [c for c in covariates if c in df.columns]
    
    for cov in covariates:
        mask = df[['ACR_Residual', cov]].notna().all(axis=1)
        if mask.sum() < 100:
            continue
        
        x = df.loc[mask, cov].values
        y = df.loc[mask, 'ACR_Residual'].values
        
        # Standardize
        x_std = (x - x.mean()) / x.std()
        
        # Fit models
        lin_coef = np.polyfit(x_std, y, 1)
        y_lin = np.polyval(lin_coef, x_std)
        ss_lin = np.sum((y - y_lin) ** 2)
        
        quad_coef = np.polyfit(x_std, y, 2)
        y_quad = np.polyval(quad_coef, x_std)
        ss_quad = np.sum((y - y_quad) ** 2)
        
        # F-test
        n = len(y)
        if ss_quad > 0:
            f_stat = ((ss_lin - ss_quad) / 1) / (ss_quad / (n - 3))
            f_p = 1 - stats.f.cdf(f_stat, 1, n - 3)
        else:
            f_stat, f_p = np.nan, np.nan
        
        a = quad_coef[0]
        shape = "U-shape" if a > 0 else "Inverted-U"
        
        results.append({
            'Covariate': cov,
            'N': n,
            'Quad_Coef': a,
            'F_Stat': f_stat,
            'F_p': f_p,
            'Shape': shape,
            'Significant': f_p < 0.05 if not np.isnan(f_p) else False
        })
        
        if not np.isnan(f_p) and f_p < 0.05:
            print(f"\nACR Residual vs {cov}: *** {shape} ***")
            print(f"  a = {a:.4f}, F = {f_stat:.2f}, p = {f_p:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS / "kidney_test2_acr_patterns.csv", index=False)
    
    return results_df

def test_ckd_progression_prediction(df: pd.DataFrame) -> dict:
    """Test if residuals predict CKD in gray zone."""
    
    print("\n" + "=" * 60)
    print("TEST 3: PREDICTING CKD IN GRAY ZONE")
    print("=" * 60)
    
    # Focus on gray zone (eGFR 60-90)
    mask = (df['eGFR'] >= 60) & (df['eGFR'] < 90)
    df_gray = df[mask].copy()
    
    print(f"\nAnalyzing {len(df_gray)} participants in gray zone (eGFR 60-90)")
    
    # Outcome: elevated ACR (proxy for early CKD)
    if 'ACR' not in df.columns:
        print("ACR not available for outcome")
        return {}
    
    df_gray['ElevatedACR'] = (df_gray['ACR'] >= 30).astype(int)
    outcome_prev = 100 * df_gray['ElevatedACR'].mean()
    print(f"Elevated ACR (≥30) prevalence in gray zone: {outcome_prev:.1f}%")
    
    if df_gray['ElevatedACR'].sum() < 20:
        print("Not enough cases")
        return {}
    
    results = {}
    
    # Get features
    ratio_cols = ['HOMA_IR', 'TG_HDL', 'BUN_Cr']
    ratio_cols = [c for c in ratio_cols if c in df_gray.columns]
    
    resid_cols = [f"{c}_Residual" for c in ratio_cols if f"{c}_Residual" in df_gray.columns]
    
    # Model with ratios
    if ratio_cols:
        mask_complete = df_gray[ratio_cols + ['ElevatedACR']].notna().all(axis=1)
        if mask_complete.sum() >= 50:
            X = df_gray.loc[mask_complete, ratio_cols].values
            y = df_gray.loc[mask_complete, 'ElevatedACR'].values
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LogisticRegression(max_iter=1000, random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            try:
                auc = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
                results['auc_ratios'] = auc.mean()
                print(f"\nRatios only: AUC = {auc.mean():.3f} ± {auc.std():.3f}")
            except:
                pass
    
    # Model with residuals
    if resid_cols:
        mask_complete = df_gray[resid_cols + ['ElevatedACR']].notna().all(axis=1)
        if mask_complete.sum() >= 50:
            X = df_gray.loc[mask_complete, resid_cols].values
            y = df_gray.loc[mask_complete, 'ElevatedACR'].values
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LogisticRegression(max_iter=1000, random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            try:
                auc = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
                results['auc_residuals'] = auc.mean()
                print(f"Residuals only: AUC = {auc.mean():.3f} ± {auc.std():.3f}")
            except:
                pass
    
    return results

def generate_visualizations(df: pd.DataFrame):
    """Generate kidney analysis figures."""
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # eGFR distribution
    ax = axes[0]
    if 'eGFR' in df.columns:
        egfr = df['eGFR'].dropna()
        egfr = egfr[egfr < 150]
        
        ax.hist(egfr, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=60, color='red', linestyle='--', linewidth=2, label='CKD threshold (60)')
        ax.axvline(x=90, color='orange', linestyle='--', linewidth=2, label='Normal threshold (90)')
        ax.axvspan(60, 90, alpha=0.2, color='yellow', label='Gray zone')
        ax.set_xlabel('eGFR (mL/min/1.73m²)')
        ax.set_ylabel('Count')
        ax.set_title('eGFR Distribution with Clinical Thresholds')
        ax.legend()
    
    # ACR distribution (log scale)
    ax = axes[1]
    if 'ACR' in df.columns:
        acr = df['ACR'].dropna()
        acr = acr[acr > 0]
        
        ax.hist(np.log10(acr), bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=np.log10(30), color='orange', linestyle='--', linewidth=2, label='Microalbuminuria (30)')
        ax.axvline(x=np.log10(300), color='red', linestyle='--', linewidth=2, label='Macroalbuminuria (300)')
        ax.set_xlabel('log₁₀(ACR) [mg/g]')
        ax.set_ylabel('Count')
        ax.set_title('ACR Distribution (log scale)')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES / "kidney_fig01_egfr_acr_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Figure 1: eGFR and ACR distributions saved")

def generate_report(stage_results: dict, resid_results: pd.DataFrame,
                   acr_results: pd.DataFrame, pred_results: dict):
    """Generate summary report."""
    
    report = []
    report.append("=" * 70)
    report.append("PAPER 2 EXTENDED: CHRONIC KIDNEY DISEASE ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")
    
    report.append("1. CKD STAGE ANALYSIS")
    report.append("-" * 40)
    if 'gray_zone_pct' in stage_results:
        report.append(f"   Gray zone (eGFR 60-90): {stage_results['gray_zone_pct']:.1f}%")
    report.append("")
    
    report.append("2. RESIDUALS BY CKD STAGE")
    report.append("-" * 40)
    if len(resid_results) > 0:
        n_sig = resid_results['Significant'].sum()
        report.append(f"   Significant differences: {n_sig}/{len(resid_results)}")
    report.append("")
    
    report.append("3. ACR RESIDUAL PATTERNS")
    report.append("-" * 40)
    if len(acr_results) > 0:
        n_sig = acr_results['Significant'].sum()
        report.append(f"   Non-linear patterns: {n_sig}/{len(acr_results)}")
    report.append("")
    
    report.append("4. PREDICTION IN GRAY ZONE")
    report.append("-" * 40)
    if pred_results:
        for key, val in pred_results.items():
            report.append(f"   {key}: {val:.3f}")
    report.append("")
    
    # Save
    report_text = "\n".join(report)
    with open(RESULTS / "kidney_analysis_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + report_text)

def main():
    print("=" * 60)
    print("PAPER 2 EXTENDED: CHRONIC KIDNEY DISEASE ANALYSIS")
    print("=" * 60)
    
    df = load_data()
    print(f"✓ Loaded {len(df)} participants")
    
    stage_results = analyze_ckd_stages(df)
    resid_results = test_residual_by_ckd_stage(df)
    acr_results = test_acr_residual_patterns(df)
    pred_results = test_ckd_progression_prediction(df)
    
    generate_visualizations(df)
    generate_report(stage_results, resid_results, acr_results, pred_results)
    
    print("\n✓ Kidney disease analysis complete!")
    print("Next: Run 05_cardiovascular.py")

if __name__ == "__main__":
    main()
