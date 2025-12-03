#!/usr/bin/env python3
"""
Paper 2 Extended: Hepatic Fibrosis Analysis
The Revelatory Division - Testing Residual Diagnostics

Analyzes whether residuals from FIB-4, APRI, and NFS contain
diagnostic information, particularly in the "indeterminate" zone.

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

# Ensure directories exist
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

def load_data() -> pd.DataFrame:
    """Load data with computed ratios."""
    filepath = PROCESSED_DATA / "nhanes_with_ratios.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Data not found: {filepath}\nRun 02_compute_all_ratios.py first")
    return pd.read_csv(filepath)

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_fib4_zones(df: pd.DataFrame) -> dict:
    """Analyze the FIB-4 indeterminate zone."""
    
    print("\n" + "=" * 60)
    print("FIB-4 ZONE ANALYSIS")
    print("=" * 60)
    
    results = {}
    
    # Zone distribution
    if 'FIB4_Category' in df.columns:
        zone_counts = df['FIB4_Category'].value_counts()
        zone_pcts = 100 * zone_counts / zone_counts.sum()
        
        print("\nFIB-4 Zone Distribution:")
        for zone in ['Low Risk', 'Indeterminate', 'High Risk']:
            if zone in zone_counts.index:
                print(f"  {zone}: {zone_counts[zone]} ({zone_pcts[zone]:.1f}%)")
        
        results['zone_distribution'] = zone_pcts.to_dict()
        
        # Indeterminate zone size
        if 'Indeterminate' in zone_pcts.index:
            indet_pct = zone_pcts['Indeterminate']
            print(f"\n→ {indet_pct:.1f}% of participants are in the INDETERMINATE zone")
            print("  These patients need additional testing (biopsy, elastography)")
            results['indeterminate_pct'] = indet_pct
    
    return results

def test_residual_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """Test if residual distributions differ across FIB-4 zones."""
    
    print("\n" + "=" * 60)
    print("TEST 1: RESIDUAL DISTRIBUTIONS ACROSS FIB-4 ZONES")
    print("=" * 60)
    
    results = []
    
    residual_cols = [c for c in df.columns if c.endswith('_Residual')]
    
    for resid_col in residual_cols:
        if resid_col not in df.columns or 'FIB4_Category' not in df.columns:
            continue
        
        # Get data for each zone
        zones = ['Low Risk', 'Indeterminate', 'High Risk']
        zone_data = {}
        
        for zone in zones:
            mask = (df['FIB4_Category'] == zone) & df[resid_col].notna()
            zone_data[zone] = df.loc[mask, resid_col].values
        
        # Skip if not enough data
        if any(len(zone_data[z]) < 30 for z in zones if z in zone_data):
            continue
        
        # Kruskal-Wallis test (non-parametric ANOVA)
        valid_zones = [zone_data[z] for z in zones if len(zone_data.get(z, [])) > 0]
        if len(valid_zones) >= 2:
            h_stat, kw_p = stats.kruskal(*valid_zones)
        else:
            h_stat, kw_p = np.nan, np.nan
        
        # Pairwise comparisons
        if 'Low Risk' in zone_data and 'High Risk' in zone_data:
            _, lr_hr_p = stats.mannwhitneyu(zone_data['Low Risk'], zone_data['High Risk'])
        else:
            lr_hr_p = np.nan
        
        if 'Indeterminate' in zone_data and 'High Risk' in zone_data:
            _, ind_hr_p = stats.mannwhitneyu(zone_data['Indeterminate'], zone_data['High Risk'])
        else:
            ind_hr_p = np.nan
        
        # Effect size (eta-squared for Kruskal-Wallis)
        n_total = sum(len(zone_data[z]) for z in zones if z in zone_data)
        if not np.isnan(h_stat):
            eta_sq = (h_stat - len(valid_zones) + 1) / (n_total - len(valid_zones))
        else:
            eta_sq = np.nan
        
        ratio_name = resid_col.replace('_Residual', '')
        
        results.append({
            'Ratio': ratio_name,
            'Residual': resid_col,
            'N_Total': n_total,
            'N_LowRisk': len(zone_data.get('Low Risk', [])),
            'N_Indeterminate': len(zone_data.get('Indeterminate', [])),
            'N_HighRisk': len(zone_data.get('High Risk', [])),
            'Mean_LowRisk': np.mean(zone_data.get('Low Risk', [np.nan])),
            'Mean_Indeterminate': np.mean(zone_data.get('Indeterminate', [np.nan])),
            'Mean_HighRisk': np.mean(zone_data.get('High Risk', [np.nan])),
            'KW_H': h_stat,
            'KW_p': kw_p,
            'Eta_Squared': eta_sq,
            'LowRisk_vs_HighRisk_p': lr_hr_p,
            'Indeterminate_vs_HighRisk_p': ind_hr_p,
            'Significant': kw_p < 0.05 if not np.isnan(kw_p) else False
        })
        
        # Print significant results
        if not np.isnan(kw_p) and kw_p < 0.05:
            print(f"\n{ratio_name} Residuals: *** SIGNIFICANT ***")
            print(f"  Kruskal-Wallis H = {h_stat:.2f}, p = {kw_p:.4f}")
            print(f"  Effect size (η²) = {eta_sq:.3f}")
            print(f"  Mean by zone:")
            for zone in zones:
                if zone in zone_data and len(zone_data[zone]) > 0:
                    print(f"    {zone}: {np.mean(zone_data[zone]):.3f} ± {np.std(zone_data[zone]):.3f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS / "liver_test1_residual_distributions.csv", index=False)
    
    # Summary
    n_sig = results_df['Significant'].sum()
    print(f"\n→ {n_sig}/{len(results_df)} residuals show significant differences across FIB-4 zones")
    
    return results_df

def test_ushapes_in_indeterminate(df: pd.DataFrame) -> pd.DataFrame:
    """Test for U-shapes in residuals within the indeterminate zone."""
    
    print("\n" + "=" * 60)
    print("TEST 2: U-SHAPES IN THE INDETERMINATE ZONE")
    print("=" * 60)
    
    # Focus on indeterminate zone
    mask = df['FIB4_Category'] == 'Indeterminate'
    df_indet = df[mask].copy()
    
    print(f"\nAnalyzing {len(df_indet)} participants in indeterminate zone")
    
    results = []
    
    residual_cols = [c for c in df.columns if c.endswith('_Residual')]
    covariates = ['Age', 'BMI', 'Glucose', 'Triglycerides', 'HDL']
    covariates = [c for c in covariates if c in df.columns]
    
    for resid_col in residual_cols:
        for cov in covariates:
            # Get complete cases
            mask = df_indet[[resid_col, cov]].notna().all(axis=1)
            if mask.sum() < 50:
                continue
            
            x = df_indet.loc[mask, cov].values
            y = df_indet.loc[mask, resid_col].values
            
            # Standardize x for numerical stability
            x_std = (x - x.mean()) / x.std()
            
            # Fit linear model
            lin_coef = np.polyfit(x_std, y, 1)
            y_lin = np.polyval(lin_coef, x_std)
            ss_lin = np.sum((y - y_lin) ** 2)
            
            # Fit quadratic model
            quad_coef = np.polyfit(x_std, y, 2)
            y_quad = np.polyval(quad_coef, x_std)
            ss_quad = np.sum((y - y_quad) ** 2)
            
            # F-test for model comparison
            n = len(y)
            df_lin = n - 2
            df_quad = n - 3
            
            if ss_quad > 0 and df_quad > 0:
                f_stat = ((ss_lin - ss_quad) / 1) / (ss_quad / df_quad)
                f_p = 1 - stats.f.cdf(f_stat, 1, df_quad)
            else:
                f_stat, f_p = np.nan, np.nan
            
            # Determine shape
            a = quad_coef[0]  # Quadratic coefficient
            if a > 0:
                shape = "U-shape"
            else:
                shape = "Inverted-U"
            
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
            
            # Print significant results
            if not np.isnan(f_p) and f_p < 0.05:
                print(f"\n{ratio_name} vs {cov}: *** {shape} DETECTED ***")
                print(f"  Quadratic coefficient = {a:.4f}")
                print(f"  F = {f_stat:.2f}, p = {f_p:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS / "liver_test2_ushapes_indeterminate.csv", index=False)
    
    # Summary
    n_sig = results_df['Significant'].sum()
    n_ushape = ((results_df['Significant']) & (results_df['Shape'] == 'U-shape')).sum()
    print(f"\n→ {n_sig} significant non-linear patterns found")
    print(f"→ {n_ushape} are U-shapes (consistent with opposing mechanisms)")
    
    return results_df

def test_residual_classification(df: pd.DataFrame) -> dict:
    """Test if residuals can classify within indeterminate zone."""
    
    print("\n" + "=" * 60)
    print("TEST 3: RESIDUAL CLASSIFICATION IN INDETERMINATE ZONE")
    print("=" * 60)
    
    # We need a ground truth for fibrosis
    # In NHANES, we can use surrogate markers:
    # - GGT elevation
    # - ALT > 40
    # - Diagnosed liver condition
    
    # Create a "probable advanced fibrosis" proxy
    df = df.copy()
    
    # Proxy for advanced liver disease (in absence of biopsy)
    # High GGT + elevated ALT + elevated AST
    if all(v in df.columns for v in ['GGT', 'ALT', 'AST']):
        df['ProbableAdvancedLiver'] = (
            (df['GGT'] > 60) &  # Elevated GGT
            (df['ALT'] > 40) &   # Elevated ALT
            (df['AST'] > 40)     # Elevated AST
        ).astype(int)
    else:
        print("Cannot create liver outcome proxy - missing variables")
        return {}
    
    # Focus on indeterminate zone
    mask = (df['FIB4_Category'] == 'Indeterminate') & df['ProbableAdvancedLiver'].notna()
    df_indet = df[mask].copy()
    
    print(f"\nAnalyzing {len(df_indet)} participants in indeterminate zone")
    
    # Outcome prevalence
    outcome_prev = 100 * df_indet['ProbableAdvancedLiver'].mean()
    print(f"'Probable advanced liver disease' prevalence: {outcome_prev:.1f}%")
    
    if df_indet['ProbableAdvancedLiver'].sum() < 20:
        print("Not enough cases for classification analysis")
        return {}
    
    results = {}
    
    # Get residual columns
    residual_cols = [c for c in df.columns if c.endswith('_Residual') and 
                     c.replace('_Residual', '') in ['HOMA_IR', 'TG_HDL', 'QUICKI', 'DeRitis', 'BUN_Cr']]
    
    # Get ratio columns (for comparison)
    ratio_cols = ['HOMA_IR', 'TG_HDL', 'QUICKI', 'DeRitis', 'BUN_Cr']
    ratio_cols = [c for c in ratio_cols if c in df.columns]
    
    # Model 1: Ratios only
    X_cols_ratios = [c for c in ratio_cols if df_indet[c].notna().sum() > len(df_indet) * 0.5]
    
    if X_cols_ratios:
        mask_complete = df_indet[X_cols_ratios + ['ProbableAdvancedLiver']].notna().all(axis=1)
        X = df_indet.loc[mask_complete, X_cols_ratios].values
        y = df_indet.loc[mask_complete, 'ProbableAdvancedLiver'].values
        
        if len(y) >= 50 and y.sum() >= 10:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LogisticRegression(max_iter=1000, random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            try:
                auc_ratios = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
                results['auc_ratios'] = auc_ratios.mean()
                results['auc_ratios_std'] = auc_ratios.std()
                print(f"\nRatios only: AUC = {auc_ratios.mean():.3f} ± {auc_ratios.std():.3f}")
            except Exception as e:
                print(f"Ratios model failed: {e}")
    
    # Model 2: Residuals only
    X_cols_resid = [c for c in residual_cols if c in df_indet.columns and 
                    df_indet[c].notna().sum() > len(df_indet) * 0.5]
    
    if X_cols_resid:
        mask_complete = df_indet[X_cols_resid + ['ProbableAdvancedLiver']].notna().all(axis=1)
        X = df_indet.loc[mask_complete, X_cols_resid].values
        y = df_indet.loc[mask_complete, 'ProbableAdvancedLiver'].values
        
        if len(y) >= 50 and y.sum() >= 10:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LogisticRegression(max_iter=1000, random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            try:
                auc_resid = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
                results['auc_residuals'] = auc_resid.mean()
                results['auc_residuals_std'] = auc_resid.std()
                print(f"Residuals only: AUC = {auc_resid.mean():.3f} ± {auc_resid.std():.3f}")
            except Exception as e:
                print(f"Residuals model failed: {e}")
    
    # Model 3: Combined
    X_cols_all = X_cols_ratios + X_cols_resid
    X_cols_all = list(set(X_cols_all))  # Remove duplicates
    
    if len(X_cols_all) > 0:
        mask_complete = df_indet[X_cols_all + ['ProbableAdvancedLiver']].notna().all(axis=1)
        X = df_indet.loc[mask_complete, X_cols_all].values
        y = df_indet.loc[mask_complete, 'ProbableAdvancedLiver'].values
        
        if len(y) >= 50 and y.sum() >= 10:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LogisticRegression(max_iter=1000, random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            try:
                auc_combined = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
                results['auc_combined'] = auc_combined.mean()
                results['auc_combined_std'] = auc_combined.std()
                print(f"Combined: AUC = {auc_combined.mean():.3f} ± {auc_combined.std():.3f}")
            except Exception as e:
                print(f"Combined model failed: {e}")
    
    # Improvement analysis
    if 'auc_ratios' in results and 'auc_combined' in results:
        improvement = results['auc_combined'] - results['auc_ratios']
        print(f"\n→ Improvement from adding residuals: {improvement:+.3f} AUC")
    
    return results

def generate_visualizations(df: pd.DataFrame):
    """Generate figures for liver analysis."""
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Figure 1: FIB-4 distribution with zones
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # FIB-4 histogram with zone markers
    ax = axes[0]
    fib4_data = df['FIB4'].dropna()
    fib4_data = fib4_data[fib4_data < 10]  # Remove extreme outliers for visualization
    
    ax.hist(fib4_data, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=1.3, color='green', linestyle='--', linewidth=2, label='Low/Indet threshold (1.3)')
    ax.axvline(x=2.67, color='red', linestyle='--', linewidth=2, label='Indet/High threshold (2.67)')
    ax.set_xlabel('FIB-4 Score')
    ax.set_ylabel('Count')
    ax.set_title('FIB-4 Distribution with Clinical Thresholds')
    ax.legend()
    
    # Zone pie chart
    ax = axes[1]
    if 'FIB4_Category' in df.columns:
        zone_counts = df['FIB4_Category'].value_counts()
        colors = {'Low Risk': '#2ecc71', 'Indeterminate': '#f39c12', 'High Risk': '#e74c3c'}
        ax.pie(zone_counts, labels=zone_counts.index, autopct='%1.1f%%',
               colors=[colors.get(z, 'gray') for z in zone_counts.index],
               explode=[0.05 if z == 'Indeterminate' else 0 for z in zone_counts.index])
        ax.set_title('FIB-4 Risk Categories')
    
    plt.tight_layout()
    plt.savefig(FIGURES / "liver_fig01_fib4_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Figure 1: FIB-4 distribution saved")
    
    # Figure 2: Residual distributions by FIB-4 zone
    residual_cols = ['HOMA_IR_Residual', 'TG_HDL_Residual', 'DeRitis_Residual']
    residual_cols = [c for c in residual_cols if c in df.columns]
    
    if residual_cols and 'FIB4_Category' in df.columns:
        fig, axes = plt.subplots(1, len(residual_cols), figsize=(5*len(residual_cols), 5))
        if len(residual_cols) == 1:
            axes = [axes]
        
        for ax, col in zip(axes, residual_cols):
            for zone in ['Low Risk', 'Indeterminate', 'High Risk']:
                data = df.loc[df['FIB4_Category'] == zone, col].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=30, alpha=0.5, label=zone, density=True)
            
            ax.set_xlabel(col.replace('_Residual', ' Residual'))
            ax.set_ylabel('Density')
            ax.set_title(f'{col.replace("_Residual", "")} Residuals by FIB-4 Zone')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(FIGURES / "liver_fig02_residuals_by_zone.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Figure 2: Residuals by zone saved")
    
    # Figure 3: U-shape exploration in indeterminate zone
    if 'FIB4_Category' in df.columns and 'HOMA_IR_Residual' in df.columns:
        df_indet = df[df['FIB4_Category'] == 'Indeterminate'].copy()
        
        if len(df_indet) > 50:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # HOMA_IR residual vs Age
            ax = axes[0]
            mask = df_indet[['Age', 'HOMA_IR_Residual']].notna().all(axis=1)
            x = df_indet.loc[mask, 'Age']
            y = df_indet.loc[mask, 'HOMA_IR_Residual']
            
            ax.scatter(x, y, alpha=0.3, s=20)
            
            # Fit quadratic
            if len(x) > 20:
                z = np.polyfit(x, y, 2)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Quadratic fit (a={z[0]:.4f})')
            
            ax.set_xlabel('Age')
            ax.set_ylabel('HOMA-IR Residual')
            ax.set_title('HOMA-IR Residual vs Age (Indeterminate Zone)')
            ax.legend()
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # HOMA_IR residual vs BMI
            ax = axes[1]
            if 'BMI' in df_indet.columns:
                mask = df_indet[['BMI', 'HOMA_IR_Residual']].notna().all(axis=1)
                x = df_indet.loc[mask, 'BMI']
                y = df_indet.loc[mask, 'HOMA_IR_Residual']
                
                ax.scatter(x, y, alpha=0.3, s=20)
                
                if len(x) > 20:
                    z = np.polyfit(x, y, 2)
                    p = np.poly1d(z)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Quadratic fit (a={z[0]:.4f})')
                
                ax.set_xlabel('BMI')
                ax.set_ylabel('HOMA-IR Residual')
                ax.set_title('HOMA-IR Residual vs BMI (Indeterminate Zone)')
                ax.legend()
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(FIGURES / "liver_fig03_ushapes_indeterminate.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ Figure 3: U-shapes in indeterminate zone saved")

def generate_report(zone_results: dict, dist_results: pd.DataFrame, 
                   ushape_results: pd.DataFrame, class_results: dict):
    """Generate summary report."""
    
    report = []
    report.append("=" * 70)
    report.append("PAPER 2 EXTENDED: HEPATIC FIBROSIS ANALYSIS REPORT")
    report.append("The Revelatory Division - Testing Residual Diagnostics")
    report.append("=" * 70)
    report.append("")
    
    report.append("1. FIB-4 ZONE ANALYSIS")
    report.append("-" * 40)
    if 'indeterminate_pct' in zone_results:
        report.append(f"   Indeterminate zone: {zone_results['indeterminate_pct']:.1f}% of population")
        report.append("   → These patients need additional testing")
    report.append("")
    
    report.append("2. RESIDUAL DISTRIBUTION DIFFERENCES")
    report.append("-" * 40)
    if len(dist_results) > 0:
        n_sig = dist_results['Significant'].sum()
        report.append(f"   Significant differences: {n_sig}/{len(dist_results)} residuals")
        
        sig_results = dist_results[dist_results['Significant']]
        for _, row in sig_results.iterrows():
            report.append(f"   • {row['Ratio']}: p = {row['KW_p']:.4f}, η² = {row['Eta_Squared']:.3f}")
    report.append("")
    
    report.append("3. U-SHAPES IN INDETERMINATE ZONE")
    report.append("-" * 40)
    if len(ushape_results) > 0:
        n_sig = ushape_results['Significant'].sum()
        n_ushape = ((ushape_results['Significant']) & (ushape_results['Shape'] == 'U-shape')).sum()
        report.append(f"   Significant non-linear patterns: {n_sig}")
        report.append(f"   U-shapes detected: {n_ushape}")
    report.append("")
    
    report.append("4. CLASSIFICATION IN INDETERMINATE ZONE")
    report.append("-" * 40)
    if class_results:
        if 'auc_ratios' in class_results:
            report.append(f"   Ratios only: AUC = {class_results['auc_ratios']:.3f}")
        if 'auc_residuals' in class_results:
            report.append(f"   Residuals only: AUC = {class_results['auc_residuals']:.3f}")
        if 'auc_combined' in class_results:
            report.append(f"   Combined: AUC = {class_results['auc_combined']:.3f}")
        
        if 'auc_ratios' in class_results and 'auc_combined' in class_results:
            improvement = class_results['auc_combined'] - class_results['auc_ratios']
            report.append(f"   → Improvement: {improvement:+.3f} AUC")
    report.append("")
    
    report.append("=" * 70)
    report.append("CONCLUSION")
    report.append("=" * 70)
    
    # Determine overall conclusion
    findings = 0
    if len(dist_results) > 0 and dist_results['Significant'].sum() > 0:
        findings += 1
    if len(ushape_results) > 0 and ushape_results['Significant'].sum() > 0:
        findings += 1
    if class_results.get('auc_combined', 0) > class_results.get('auc_ratios', 0):
        findings += 1
    
    if findings >= 2:
        conclusion = "HYPOTHESIS SUPPORTED"
        detail = "Residuals contain diagnostic information in the FIB-4 indeterminate zone"
    elif findings == 1:
        conclusion = "PARTIAL SUPPORT"
        detail = "Some evidence for residual information, but findings are mixed"
    else:
        conclusion = "HYPOTHESIS NOT SUPPORTED"
        detail = "No strong evidence that residuals add diagnostic value for liver fibrosis"
    
    report.append(f"   {conclusion}")
    report.append(f"   {detail}")
    report.append("")
    report.append("Note: This analysis uses NHANES data without biopsy-confirmed fibrosis.")
    report.append("Findings should be validated with imaging or histological data.")
    report.append("")
    
    # Save report
    report_text = "\n".join(report)
    with open(RESULTS / "liver_analysis_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + report_text)

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""
    
    print("=" * 60)
    print("PAPER 2 EXTENDED: HEPATIC FIBROSIS ANALYSIS")
    print("The Revelatory Division - Testing Residual Diagnostics")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"✓ Loaded {len(df)} participants")
    
    # Check for required variables
    if 'FIB4' not in df.columns:
        print("✗ FIB-4 not found. Run 02_compute_all_ratios.py first.")
        return
    
    # Run analyses
    zone_results = analyze_fib4_zones(df)
    dist_results = test_residual_distributions(df)
    ushape_results = test_ushapes_in_indeterminate(df)
    class_results = test_residual_classification(df)
    
    # Generate visualizations
    generate_visualizations(df)
    
    # Generate report
    generate_report(zone_results, dist_results, ushape_results, class_results)
    
    print()
    print("=" * 60)
    print("✓ Hepatic fibrosis analysis complete!")
    print(f"  Results: {RESULTS}")
    print(f"  Figures: {FIGURES}")
    print()
    print("Next: Run 04_kidney_disease.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
