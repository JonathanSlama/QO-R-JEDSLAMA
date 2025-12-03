#!/usr/bin/env python3
"""
Paper 2 Extended: Diabetes/Prediabetes Analysis
The Revelatory Division - Testing Residual Diagnostics

Extension of Paper 2 original analysis to NHANES data.
Tests HOMA-IR, TyG, and QUICKI residuals for diabetes prediction.

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

def analyze_glycemic_status(df: pd.DataFrame) -> dict:
    """Analyze glycemic status distribution."""
    
    print("\n" + "=" * 60)
    print("GLYCEMIC STATUS ANALYSIS")
    print("=" * 60)
    
    results = {}
    
    # HbA1c distribution
    if 'HbA1c' in df.columns:
        hba1c = df['HbA1c'].dropna()
        print(f"\nHbA1c Statistics:")
        print(f"  Mean: {hba1c.mean():.2f}%")
        print(f"  Median: {hba1c.median():.2f}%")
        
        # Categories
        normal = (hba1c < 5.7).sum() / len(hba1c) * 100
        prediab = ((hba1c >= 5.7) & (hba1c < 6.5)).sum() / len(hba1c) * 100
        diab = (hba1c >= 6.5).sum() / len(hba1c) * 100
        
        print(f"\nGlycemic Categories (by HbA1c):")
        print(f"  Normal (<5.7%): {normal:.1f}%")
        print(f"  Prediabetes (5.7-6.4%): {prediab:.1f}%")
        print(f"  Diabetes (≥6.5%): {diab:.1f}%")
        
        results['prediabetes_pct'] = prediab
        results['diabetes_pct'] = diab
        
        print(f"\n→ Prediabetes zone: {prediab:.1f}% of population")
        print("  These individuals may or may not progress to diabetes")
    
    # HOMA-IR distribution
    if 'HOMA_IR' in df.columns:
        homa = df['HOMA_IR'].dropna()
        print(f"\nHOMA-IR Statistics:")
        print(f"  Mean: {homa.mean():.2f}")
        print(f"  Median: {homa.median():.2f}")
        
        # Categories
        normal = (homa < 1.0).sum() / len(homa) * 100
        borderline = ((homa >= 1.0) & (homa < 2.5)).sum() / len(homa) * 100
        ir = (homa >= 2.5).sum() / len(homa) * 100
        
        print(f"\nHOMA-IR Categories:")
        print(f"  Normal (<1.0): {normal:.1f}%")
        print(f"  Borderline (1.0-2.5): {borderline:.1f}%")
        print(f"  Insulin Resistant (≥2.5): {ir:.1f}%")
        
        results['ir_pct'] = ir
    
    return results

def test_residuals_by_glycemic_status(df: pd.DataFrame) -> pd.DataFrame:
    """Test residual differences across glycemic categories."""
    
    print("\n" + "=" * 60)
    print("TEST 1: RESIDUALS BY GLYCEMIC STATUS")
    print("=" * 60)
    
    # Create glycemic categories
    if 'HbA1c' not in df.columns:
        print("HbA1c not available")
        return pd.DataFrame()
    
    df['GlycemicCategory'] = pd.cut(df['HbA1c'],
                                     bins=[0, 5.7, 6.5, 20],
                                     labels=['Normal', 'Prediabetes', 'Diabetes'])
    
    results = []
    
    residual_cols = [c for c in df.columns if c.endswith('_Residual')]
    
    for resid_col in residual_cols:
        categories = ['Normal', 'Prediabetes', 'Diabetes']
        cat_data = {}
        
        for cat in categories:
            mask = (df['GlycemicCategory'] == cat) & df[resid_col].notna()
            cat_data[cat] = df.loc[mask, resid_col].values
        
        if any(len(cat_data.get(c, [])) < 30 for c in categories):
            continue
        
        # Kruskal-Wallis
        valid_cats = [cat_data[c] for c in categories if len(cat_data.get(c, [])) > 0]
        h_stat, kw_p = stats.kruskal(*valid_cats)
        
        # Effect size
        n_total = sum(len(cat_data[c]) for c in categories)
        eta_sq = (h_stat - len(valid_cats) + 1) / (n_total - len(valid_cats))
        
        ratio_name = resid_col.replace('_Residual', '')
        
        results.append({
            'Ratio': ratio_name,
            'N_Normal': len(cat_data.get('Normal', [])),
            'N_Prediab': len(cat_data.get('Prediabetes', [])),
            'N_Diabetes': len(cat_data.get('Diabetes', [])),
            'Mean_Normal': np.mean(cat_data.get('Normal', [np.nan])),
            'Mean_Prediab': np.mean(cat_data.get('Prediabetes', [np.nan])),
            'Mean_Diabetes': np.mean(cat_data.get('Diabetes', [np.nan])),
            'KW_H': h_stat,
            'KW_p': kw_p,
            'Eta_Squared': eta_sq,
            'Significant': kw_p < 0.05
        })
        
        if kw_p < 0.05:
            print(f"\n{ratio_name}: *** SIGNIFICANT ***")
            print(f"  H = {h_stat:.2f}, p = {kw_p:.4f}, η² = {eta_sq:.3f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS / "diabetes_test1_residuals_by_status.csv", index=False)
    
    n_sig = results_df['Significant'].sum() if len(results_df) > 0 else 0
    print(f"\n→ {n_sig}/{len(results_df)} residuals differ across glycemic categories")
    
    return results_df

def test_ushapes_in_prediabetes(df: pd.DataFrame) -> pd.DataFrame:
    """Test for U-shapes in prediabetic population."""
    
    print("\n" + "=" * 60)
    print("TEST 2: U-SHAPES IN PREDIABETES ZONE")
    print("=" * 60)
    
    # Focus on prediabetes (HbA1c 5.7-6.4%)
    mask = (df['HbA1c'] >= 5.7) & (df['HbA1c'] < 6.5)
    df_prediab = df[mask].copy()
    
    print(f"\nAnalyzing {len(df_prediab)} participants with prediabetes")
    
    results = []
    
    diabetes_residuals = ['HOMA_IR_Residual', 'QUICKI_Residual', 'G_I_Residual', 'TyG_Residual']
    diabetes_residuals = [c for c in diabetes_residuals if c in df.columns]
    
    covariates = ['Age', 'BMI', 'Triglycerides', 'HDL']
    covariates = [c for c in covariates if c in df.columns]
    
    for resid_col in diabetes_residuals:
        for cov in covariates:
            mask = df_prediab[[resid_col, cov]].notna().all(axis=1)
            if mask.sum() < 50:
                continue
            
            x = df_prediab.loc[mask, cov].values
            y = df_prediab.loc[mask, resid_col].values
            
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
    results_df.to_csv(RESULTS / "diabetes_test2_ushapes_prediabetes.csv", index=False)
    
    return results_df

def test_diabetes_progression_prediction(df: pd.DataFrame) -> dict:
    """Test if residuals predict diabetes in prediabetic zone."""
    
    print("\n" + "=" * 60)
    print("TEST 3: PREDICTING DIABETES IN PREDIABETES ZONE")
    print("=" * 60)
    
    # In cross-sectional data, we use current diabetes diagnosis as outcome
    # This is a proxy - ideally we'd have longitudinal data
    
    # Focus on prediabetes zone
    mask = (df['HbA1c'] >= 5.7) & (df['HbA1c'] < 6.5)
    df_prediab = df[mask].copy()
    
    print(f"\nAnalyzing {len(df_prediab)} participants with prediabetes")
    
    # Outcome: diagnosed diabetes (from questionnaire)
    if 'HasDiabetes' not in df.columns:
        print("Diabetes diagnosis not available")
        return {}
    
    outcome_prev = 100 * df_prediab['HasDiabetes'].mean()
    print(f"Diagnosed diabetes in prediabetes zone: {outcome_prev:.1f}%")
    
    if df_prediab['HasDiabetes'].sum() < 10:
        print("Not enough cases")
        return {}
    
    results = {}
    
    # Features
    ratio_cols = ['HOMA_IR', 'QUICKI', 'TyG', 'TG_HDL']
    ratio_cols = [c for c in ratio_cols if c in df_prediab.columns]
    
    resid_cols = [f"{c}_Residual" for c in ratio_cols if f"{c}_Residual" in df_prediab.columns]
    
    # Model with ratios
    if ratio_cols:
        all_cols = ratio_cols + ['HasDiabetes']
        mask_complete = df_prediab[all_cols].notna().all(axis=1)
        
        if mask_complete.sum() >= 50 and df_prediab.loc[mask_complete, 'HasDiabetes'].sum() >= 10:
            X = df_prediab.loc[mask_complete, ratio_cols].values
            y = df_prediab.loc[mask_complete, 'HasDiabetes'].values
            
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
        all_cols = resid_cols + ['HasDiabetes']
        mask_complete = df_prediab[all_cols].notna().all(axis=1)
        
        if mask_complete.sum() >= 50 and df_prediab.loc[mask_complete, 'HasDiabetes'].sum() >= 10:
            X = df_prediab.loc[mask_complete, resid_cols].values
            y = df_prediab.loc[mask_complete, 'HasDiabetes'].values
            
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
    """Generate diabetes analysis figures."""
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # HbA1c distribution
    ax = axes[0, 0]
    if 'HbA1c' in df.columns:
        hba1c = df['HbA1c'].dropna()
        hba1c = hba1c[hba1c < 15]
        
        ax.hist(hba1c, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=5.7, color='orange', linestyle='--', linewidth=2, label='Prediab (5.7)')
        ax.axvline(x=6.5, color='red', linestyle='--', linewidth=2, label='Diabetes (6.5)')
        ax.axvspan(5.7, 6.5, alpha=0.2, color='yellow', label='Prediabetes zone')
        ax.set_xlabel('HbA1c (%)')
        ax.set_ylabel('Count')
        ax.set_title('HbA1c Distribution')
        ax.legend()
    
    # HOMA-IR distribution
    ax = axes[0, 1]
    if 'HOMA_IR' in df.columns:
        homa = df['HOMA_IR'].dropna()
        homa = homa[homa < 20]
        
        ax.hist(homa, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=2.5, color='red', linestyle='--', linewidth=2, label='IR threshold (2.5)')
        ax.set_xlabel('HOMA-IR')
        ax.set_ylabel('Count')
        ax.set_title('HOMA-IR Distribution')
        ax.legend()
    
    # HOMA-IR residuals by glycemic status
    ax = axes[1, 0]
    if 'HOMA_IR_Residual' in df.columns and 'GlycemicCategory' in df.columns:
        for cat in ['Normal', 'Prediabetes', 'Diabetes']:
            data = df.loc[df['GlycemicCategory'] == cat, 'HOMA_IR_Residual'].dropna()
            if len(data) > 0:
                ax.hist(data, bins=30, alpha=0.5, label=cat, density=True)
        
        ax.axvline(x=0, color='gray', linestyle='--')
        ax.set_xlabel('HOMA-IR Residual')
        ax.set_ylabel('Density')
        ax.set_title('HOMA-IR Residuals by Glycemic Status')
        ax.legend()
    
    # U-shape in prediabetes
    ax = axes[1, 1]
    if 'HOMA_IR_Residual' in df.columns and 'HbA1c' in df.columns:
        mask = (df['HbA1c'] >= 5.7) & (df['HbA1c'] < 6.5)
        df_prediab = df[mask]
        
        if 'BMI' in df_prediab.columns:
            mask = df_prediab[['BMI', 'HOMA_IR_Residual']].notna().all(axis=1)
            x = df_prediab.loc[mask, 'BMI']
            y = df_prediab.loc[mask, 'HOMA_IR_Residual']
            
            if len(x) > 50:
                ax.scatter(x, y, alpha=0.3, s=20)
                
                z = np.polyfit(x, y, 2)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Quadratic (a={z[0]:.4f})')
                
                ax.set_xlabel('BMI')
                ax.set_ylabel('HOMA-IR Residual')
                ax.set_title('HOMA-IR Residual vs BMI (Prediabetes Zone)')
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES / "diabetes_fig01_glycemic_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Figure saved")

def main():
    print("=" * 60)
    print("PAPER 2 EXTENDED: DIABETES/PREDIABETES ANALYSIS")
    print("=" * 60)
    
    df = load_data()
    print(f"✓ Loaded {len(df)} participants")
    
    glycemic_results = analyze_glycemic_status(df)
    resid_results = test_residuals_by_glycemic_status(df)
    ushape_results = test_ushapes_in_prediabetes(df)
    pred_results = test_diabetes_progression_prediction(df)
    
    generate_visualizations(df)
    
    # Summary
    print("\n" + "=" * 60)
    print("DIABETES ANALYSIS SUMMARY")
    print("=" * 60)
    if 'prediabetes_pct' in glycemic_results:
        print(f"Prediabetes zone: {glycemic_results['prediabetes_pct']:.1f}%")
    if len(resid_results) > 0:
        print(f"Significant residual differences: {resid_results['Significant'].sum()}/{len(resid_results)}")
    if len(ushape_results) > 0:
        print(f"U-shapes in prediabetes: {ushape_results['Significant'].sum()}/{len(ushape_results)}")
    
    print("\n✓ Diabetes analysis complete!")
    print("Next: Run 07_metabolic_syndrome.py")

if __name__ == "__main__":
    main()
