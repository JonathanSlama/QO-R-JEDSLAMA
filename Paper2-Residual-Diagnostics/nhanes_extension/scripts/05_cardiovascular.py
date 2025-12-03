#!/usr/bin/env python3
"""
Paper 2 Extended: Cardiovascular Risk Analysis
The Revelatory Division - Testing Residual Diagnostics

Analyzes whether residuals from TG/HDL, TC/HDL, and LDL/HDL contain
diagnostic information for CVD risk stratification.

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
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
PROCESSED_DATA = PROJECT_DIR / "data" / "processed"
RESULTS = PROJECT_DIR / "results"
FIGURES = PROJECT_DIR / "figures"

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

def load_data() -> pd.DataFrame:
    filepath = PROCESSED_DATA / "nhanes_with_ratios.csv"
    return pd.read_csv(filepath)

def analyze_cv_risk_groups(df: pd.DataFrame) -> dict:
    """Analyze CV risk group distribution."""
    
    print("\n" + "=" * 60)
    print("CARDIOVASCULAR RISK ANALYSIS")
    print("=" * 60)
    
    results = {}
    
    # CVD prevalence
    if 'HasCVD' in df.columns:
        cvd_prev = 100 * df['HasCVD'].mean()
        print(f"\nCVD prevalence: {cvd_prev:.1f}%")
        results['cvd_prevalence'] = cvd_prev
    
    # TG/HDL distribution (insulin resistance proxy)
    if 'TG_HDL' in df.columns:
        tg_hdl = df['TG_HDL'].dropna()
        print(f"\nTG/HDL Ratio:")
        print(f"  Mean: {tg_hdl.mean():.2f}")
        print(f"  Median: {tg_hdl.median():.2f}")
        
        high_tg_hdl = (tg_hdl > 3.5).sum() / len(tg_hdl) * 100
        print(f"  High (>3.5): {high_tg_hdl:.1f}%")
        results['high_tg_hdl_pct'] = high_tg_hdl
    
    # Intermediate risk (TG/HDL 2-4)
    if 'TG_HDL' in df.columns:
        intermediate = ((df['TG_HDL'] >= 2) & (df['TG_HDL'] < 4)).sum()
        intermediate_pct = 100 * intermediate / df['TG_HDL'].notna().sum()
        print(f"\n→ Intermediate zone (TG/HDL 2-4): {intermediate_pct:.1f}%")
        results['intermediate_zone_pct'] = intermediate_pct
    
    return results

def test_residuals_by_cvd_status(df: pd.DataFrame) -> pd.DataFrame:
    """Test residual differences between CVD and non-CVD."""
    
    print("\n" + "=" * 60)
    print("TEST 1: RESIDUALS BY CVD STATUS")
    print("=" * 60)
    
    if 'HasCVD' not in df.columns:
        print("CVD status not available")
        return pd.DataFrame()
    
    results = []
    
    residual_cols = [c for c in df.columns if c.endswith('_Residual')]
    
    for resid_col in residual_cols:
        mask_cvd = (df['HasCVD'] == 1) & df[resid_col].notna()
        mask_no_cvd = (df['HasCVD'] == 0) & df[resid_col].notna()
        
        cvd_data = df.loc[mask_cvd, resid_col].values
        no_cvd_data = df.loc[mask_no_cvd, resid_col].values
        
        if len(cvd_data) < 30 or len(no_cvd_data) < 30:
            continue
        
        # Mann-Whitney U test
        stat, p = stats.mannwhitneyu(cvd_data, no_cvd_data, alternative='two-sided')
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(cvd_data), len(no_cvd_data)
        r = 1 - (2 * stat) / (n1 * n2)
        
        # Cohen's d
        pooled_std = np.sqrt(((n1-1)*np.var(cvd_data) + (n2-1)*np.var(no_cvd_data)) / (n1+n2-2))
        cohens_d = (np.mean(cvd_data) - np.mean(no_cvd_data)) / pooled_std if pooled_std > 0 else 0
        
        ratio_name = resid_col.replace('_Residual', '')
        
        results.append({
            'Ratio': ratio_name,
            'N_CVD': n1,
            'N_NoCVD': n2,
            'Mean_CVD': np.mean(cvd_data),
            'Mean_NoCVD': np.mean(no_cvd_data),
            'Cohen_d': cohens_d,
            'Rank_Biserial': r,
            'MW_p': p,
            'Significant': p < 0.05
        })
        
        if p < 0.05:
            print(f"\n{ratio_name}: *** SIGNIFICANT ***")
            print(f"  CVD: {np.mean(cvd_data):.3f} ± {np.std(cvd_data):.3f}")
            print(f"  No CVD: {np.mean(no_cvd_data):.3f} ± {np.std(no_cvd_data):.3f}")
            print(f"  Cohen's d = {cohens_d:.3f}, p = {p:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS / "cv_test1_residuals_by_status.csv", index=False)
    
    n_sig = results_df['Significant'].sum() if len(results_df) > 0 else 0
    print(f"\n→ {n_sig}/{len(results_df)} residuals differ between CVD/no-CVD")
    
    return results_df

def test_ushapes_cv_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Test for U-shapes in CV-related residuals."""
    
    print("\n" + "=" * 60)
    print("TEST 2: U-SHAPES IN CV RESIDUALS")
    print("=" * 60)
    
    results = []
    
    cv_residuals = ['TG_HDL_Residual', 'TC_HDL_Residual', 'LDL_HDL_Residual', 'TyG_Residual']
    cv_residuals = [c for c in cv_residuals if c in df.columns]
    
    covariates = ['Age', 'BMI', 'SystolicBP', 'Glucose']
    covariates = [c for c in covariates if c in df.columns]
    
    for resid_col in cv_residuals:
        for cov in covariates:
            mask = df[[resid_col, cov]].notna().all(axis=1)
            if mask.sum() < 100:
                continue
            
            x = df.loc[mask, cov].values
            y = df.loc[mask, resid_col].values
            
            x_std = (x - x.mean()) / x.std()
            
            # Fit models
            lin_coef = np.polyfit(x_std, y, 1)
            y_lin = np.polyval(lin_coef, x_std)
            ss_lin = np.sum((y - y_lin) ** 2)
            
            quad_coef = np.polyfit(x_std, y, 2)
            y_quad = np.polyval(quad_coef, x_std)
            ss_quad = np.sum((y - y_quad) ** 2)
            
            n = len(y)
            if ss_quad > 0:
                f_stat = ((ss_lin - ss_quad) / 1) / (ss_quad / (n - 3))
                f_p = 1 - stats.f.cdf(f_stat, 1, n - 3)
            else:
                f_stat, f_p = np.nan, np.nan
            
            a = quad_coef[0]
            shape = "U-shape" if a > 0 else "Inverted-U"
            
            ratio_name = resid_col.replace('_Residual', '')
            
            results.append({
                'Ratio': ratio_name,
                'Covariate': cov,
                'N': n,
                'Quad_Coef': a,
                'F_Stat': f_stat,
                'F_p': f_p,
                'Shape': shape,
                'Significant': f_p < 0.05 if not np.isnan(f_p) else False
            })
            
            if not np.isnan(f_p) and f_p < 0.05:
                print(f"\n{ratio_name} vs {cov}: *** {shape} ***")
                print(f"  a = {a:.4f}, p = {f_p:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS / "cv_test2_ushapes.csv", index=False)
    
    return results_df

def test_cvd_prediction(df: pd.DataFrame) -> dict:
    """Test CVD prediction with ratios vs residuals."""
    
    print("\n" + "=" * 60)
    print("TEST 3: CVD PREDICTION")
    print("=" * 60)
    
    if 'HasCVD' not in df.columns:
        return {}
    
    results = {}
    
    # Features
    ratio_cols = ['TG_HDL', 'TC_HDL', 'LDL_HDL', 'TyG']
    ratio_cols = [c for c in ratio_cols if c in df.columns]
    
    resid_cols = [f"{c}_Residual" for c in ratio_cols if f"{c}_Residual" in df.columns]
    
    # Model with ratios
    if ratio_cols:
        all_cols = ratio_cols + ['HasCVD']
        mask = df[all_cols].notna().all(axis=1)
        
        if mask.sum() >= 100 and df.loc[mask, 'HasCVD'].sum() >= 20:
            X = df.loc[mask, ratio_cols].values
            y = df.loc[mask, 'HasCVD'].values
            
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
        all_cols = resid_cols + ['HasCVD']
        mask = df[all_cols].notna().all(axis=1)
        
        if mask.sum() >= 100 and df.loc[mask, 'HasCVD'].sum() >= 20:
            X = df.loc[mask, resid_cols].values
            y = df.loc[mask, 'HasCVD'].values
            
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
    
    # Combined
    all_features = ratio_cols + resid_cols
    if all_features:
        all_cols = all_features + ['HasCVD']
        mask = df[all_cols].notna().all(axis=1)
        
        if mask.sum() >= 100 and df.loc[mask, 'HasCVD'].sum() >= 20:
            X = df.loc[mask, all_features].values
            y = df.loc[mask, 'HasCVD'].values
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LogisticRegression(max_iter=1000, random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            try:
                auc = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
                results['auc_combined'] = auc.mean()
                print(f"Combined: AUC = {auc.mean():.3f} ± {auc.std():.3f}")
            except:
                pass
    
    if 'auc_ratios' in results and 'auc_combined' in results:
        improvement = results['auc_combined'] - results['auc_ratios']
        print(f"\n→ Improvement: {improvement:+.3f} AUC")
    
    return results

def generate_visualizations(df: pd.DataFrame):
    """Generate CV analysis figures."""
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # TG/HDL by CVD status
    ax = axes[0]
    if 'TG_HDL' in df.columns and 'HasCVD' in df.columns:
        for status, label in [(0, 'No CVD'), (1, 'CVD')]:
            data = df.loc[df['HasCVD'] == status, 'TG_HDL'].dropna()
            data = data[data < 15]  # Remove outliers
            ax.hist(data, bins=50, alpha=0.5, label=label, density=True)
        
        ax.axvline(x=3.5, color='red', linestyle='--', label='High risk threshold')
        ax.set_xlabel('TG/HDL Ratio')
        ax.set_ylabel('Density')
        ax.set_title('TG/HDL Distribution by CVD Status')
        ax.legend()
    
    # TG/HDL residuals by CVD status
    ax = axes[1]
    if 'TG_HDL_Residual' in df.columns and 'HasCVD' in df.columns:
        for status, label in [(0, 'No CVD'), (1, 'CVD')]:
            data = df.loc[df['HasCVD'] == status, 'TG_HDL_Residual'].dropna()
            ax.hist(data, bins=50, alpha=0.5, label=label, density=True)
        
        ax.axvline(x=0, color='gray', linestyle='--')
        ax.set_xlabel('TG/HDL Residual')
        ax.set_ylabel('Density')
        ax.set_title('TG/HDL Residuals by CVD Status')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES / "cv_fig01_tg_hdl_by_status.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Figure saved")

def main():
    print("=" * 60)
    print("PAPER 2 EXTENDED: CARDIOVASCULAR RISK ANALYSIS")
    print("=" * 60)
    
    df = load_data()
    print(f"✓ Loaded {len(df)} participants")
    
    cv_results = analyze_cv_risk_groups(df)
    resid_results = test_residuals_by_cvd_status(df)
    ushape_results = test_ushapes_cv_residuals(df)
    pred_results = test_cvd_prediction(df)
    
    generate_visualizations(df)
    
    # Save summary
    summary = {
        'cv_analysis': cv_results,
        'n_sig_residuals': resid_results['Significant'].sum() if len(resid_results) > 0 else 0,
        'n_sig_ushapes': ushape_results['Significant'].sum() if len(ushape_results) > 0 else 0,
        'prediction': pred_results
    }
    
    print("\n✓ Cardiovascular analysis complete!")
    print("Next: Run 06_diabetes.py")

if __name__ == "__main__":
    main()
