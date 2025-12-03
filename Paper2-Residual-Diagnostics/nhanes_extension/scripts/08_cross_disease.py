#!/usr/bin/env python3
"""
Paper 2 Extended: Cross-Disease Pattern Analysis
The Revelatory Division - Testing Residual Diagnostics

This script analyzes patterns that emerge ACROSS disease categories,
testing whether the Revelatory Division principle is universal or
disease-specific.

Key Questions:
--------------
1. Do the same residuals show significance across multiple diseases?
2. Are U-shape patterns consistent across disease categories?
3. Do residual correlations reveal shared pathophysiology?
4. Is there a "universal" residual signature of metabolic dysfunction?

QO+R Framework Connection:
--------------------------
If the QO+R principle is truly universal (as suggested by its appearance
in both astrophysics and biology), we should observe:
- Consistent residual patterns across diseases with shared mechanisms
- U-shaped signatures at regime transitions
- Residual correlations that reveal hidden mechanistic links

Author: Jonathan Édouard Slama
Email: jonathan@metafund.in
Affiliation: Metafund Research Division, Strasbourg, France
ORCID: 0009-0002-1292-4350
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.cluster import hierarchy
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

# Disease category definitions
DISEASE_OUTCOMES = {
    'Liver': {
        'outcome_col': 'FIB4_Category',
        'positive_value': 'High Risk',
        'description': 'Hepatic Fibrosis (FIB-4 High Risk)'
    },
    'Kidney': {
        'outcome_col': 'HasCKD',
        'positive_value': 1,
        'description': 'Chronic Kidney Disease (eGFR<60 or ACR>30)'
    },
    'Cardiovascular': {
        'outcome_col': 'HasCVD',
        'positive_value': 1,
        'description': 'Cardiovascular Disease (self-reported)'
    },
    'Diabetes': {
        'outcome_col': 'HasDiabetes',
        'positive_value': 1,
        'description': 'Diabetes (HbA1c≥6.5% or diagnosed)'
    },
    'MetS': {
        'outcome_col': 'HasMetS',
        'positive_value': 1,
        'description': 'Metabolic Syndrome (≥3 components)'
    }
}

def load_data() -> pd.DataFrame:
    """Load processed NHANES data with computed ratios."""
    filepath = PROCESSED_DATA / "nhanes_with_ratios.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Data not found: {filepath}")
    return pd.read_csv(filepath)

# =============================================================================
# CROSS-DISEASE ANALYSIS FUNCTIONS
# =============================================================================

def analyze_residual_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze correlations between residuals from different ratio categories.
    
    Hypothesis: If residuals capture shared underlying dysfunction,
    they should correlate across disease categories.
    """
    
    print("\n" + "=" * 60)
    print("CROSS-DISEASE RESIDUAL CORRELATIONS")
    print("=" * 60)
    
    # Select residuals from each disease category
    residual_groups = {
        'Liver': ['FIB4_Residual', 'APRI_Residual', 'DeRitis_Residual'],
        'Kidney': ['eGFR_Residual', 'ACR_Residual', 'BUN_Cr_Residual'],
        'CV': ['TG_HDL_Residual', 'TC_HDL_Residual', 'LDL_HDL_Residual'],
        'Diabetes': ['HOMA_IR_Residual', 'QUICKI_Residual', 'TyG_Residual'],
        'MetS': ['MetS_Count_Residual', 'WHtR_Residual', 'VAI_Residual']
    }
    
    # Get available residuals
    available_residuals = []
    residual_categories = {}
    
    for category, residuals in residual_groups.items():
        for resid in residuals:
            if resid in df.columns:
                available_residuals.append(resid)
                residual_categories[resid] = category
    
    print(f"\nAnalyzing {len(available_residuals)} residuals across categories")
    
    if len(available_residuals) < 2:
        print("Not enough residuals available")
        return pd.DataFrame()
    
    # Compute correlation matrix
    resid_df = df[available_residuals].dropna()
    
    if len(resid_df) < 100:
        print("Not enough complete cases")
        return pd.DataFrame()
    
    print(f"Complete cases: {len(resid_df)}")
    
    corr_matrix = resid_df.corr(method='spearman')
    
    # Identify significant cross-category correlations
    results = []
    
    for i, resid1 in enumerate(available_residuals):
        for j, resid2 in enumerate(available_residuals):
            if i >= j:
                continue
            
            cat1 = residual_categories[resid1]
            cat2 = residual_categories[resid2]
            
            # Focus on cross-category correlations
            if cat1 == cat2:
                continue
            
            r = corr_matrix.loc[resid1, resid2]
            
            # Test significance
            x = df[resid1].dropna()
            y = df[resid2].dropna()
            common_idx = x.index.intersection(y.index)
            
            if len(common_idx) >= 100:
                rho, p = stats.spearmanr(x.loc[common_idx], y.loc[common_idx])
                
                results.append({
                    'Residual_1': resid1,
                    'Category_1': cat1,
                    'Residual_2': resid2,
                    'Category_2': cat2,
                    'Spearman_r': rho,
                    'p_value': p,
                    'N': len(common_idx),
                    'Significant': p < 0.001,  # Stricter threshold for multiple comparisons
                    'Strong': abs(rho) > 0.3
                })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        results_df = results_df.sort_values('Spearman_r', key=abs, ascending=False)
        results_df.to_csv(RESULTS / "cross_disease_correlations.csv", index=False)
        
        # Report top correlations
        print("\nTop Cross-Category Correlations:")
        print("-" * 50)
        
        top_results = results_df.head(10)
        for _, row in top_results.iterrows():
            sig = "***" if row['Significant'] else ""
            print(f"  {row['Residual_1']} ↔ {row['Residual_2']}: "
                  f"r = {row['Spearman_r']:.3f} {sig}")
        
        # Count significant correlations
        n_sig = results_df['Significant'].sum()
        n_strong = (results_df['Significant'] & results_df['Strong']).sum()
        
        print(f"\n→ Significant correlations: {n_sig}/{len(results_df)}")
        print(f"→ Strong correlations (|r|>0.3): {n_strong}/{len(results_df)}")
    
    return results_df

def analyze_universal_residual_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test which residuals show significance across MULTIPLE diseases.
    
    A residual that predicts multiple diseases suggests it captures
    a fundamental aspect of metabolic dysfunction.
    """
    
    print("\n" + "=" * 60)
    print("UNIVERSAL RESIDUAL PATTERNS")
    print("=" * 60)
    
    residual_cols = [c for c in df.columns if c.endswith('_Residual')]
    
    results = []
    
    for resid_col in residual_cols:
        if df[resid_col].notna().sum() < 500:
            continue
        
        disease_results = {}
        
        for disease, config in DISEASE_OUTCOMES.items():
            outcome_col = config['outcome_col']
            positive_value = config['positive_value']
            
            if outcome_col not in df.columns:
                continue
            
            # Create binary outcome
            if df[outcome_col].dtype == 'object':
                outcome = (df[outcome_col] == positive_value).astype(int)
            else:
                outcome = (df[outcome_col] == positive_value).astype(int)
            
            # Get complete cases
            mask = df[resid_col].notna() & outcome.notna()
            
            if mask.sum() < 200:
                continue
            
            resid_data = df.loc[mask, resid_col]
            outcome_data = outcome.loc[mask]
            
            # Test association
            case_resid = resid_data[outcome_data == 1]
            control_resid = resid_data[outcome_data == 0]
            
            if len(case_resid) < 30 or len(control_resid) < 30:
                continue
            
            stat, p = stats.mannwhitneyu(case_resid, control_resid, alternative='two-sided')
            
            # Effect size
            n1, n2 = len(case_resid), len(control_resid)
            r = 1 - (2 * stat) / (n1 * n2)  # Rank-biserial correlation
            
            disease_results[disease] = {
                'p_value': p,
                'effect_size': r,
                'significant': p < 0.01
            }
        
        if disease_results:
            n_diseases_tested = len(disease_results)
            n_significant = sum(1 for d in disease_results.values() if d['significant'])
            
            # Calculate mean effect size across diseases
            effect_sizes = [d['effect_size'] for d in disease_results.values()]
            mean_effect = np.mean(effect_sizes)
            
            ratio_name = resid_col.replace('_Residual', '')
            
            row = {
                'Ratio': ratio_name,
                'Residual': resid_col,
                'N_Diseases_Tested': n_diseases_tested,
                'N_Significant': n_significant,
                'Pct_Significant': 100 * n_significant / n_diseases_tested,
                'Mean_Effect_Size': mean_effect,
                'Universal': n_significant >= 3  # Significant for ≥3 diseases
            }
            
            # Add per-disease results
            for disease, vals in disease_results.items():
                row[f'{disease}_p'] = vals['p_value']
                row[f'{disease}_r'] = vals['effect_size']
                row[f'{disease}_sig'] = vals['significant']
            
            results.append(row)
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        results_df = results_df.sort_values('N_Significant', ascending=False)
        results_df.to_csv(RESULTS / "universal_residual_patterns.csv", index=False)
        
        # Report universal residuals
        print("\nResiduals Significant Across Multiple Diseases:")
        print("-" * 50)
        
        for _, row in results_df.head(10).iterrows():
            univ = "*** UNIVERSAL ***" if row['Universal'] else ""
            print(f"  {row['Ratio']}: {row['N_Significant']}/{row['N_Diseases_Tested']} diseases "
                  f"(mean r = {row['Mean_Effect_Size']:.3f}) {univ}")
        
        n_universal = results_df['Universal'].sum()
        print(f"\n→ Universal residuals (≥3 diseases): {n_universal}/{len(results_df)}")
    
    return results_df

def analyze_ushape_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test whether U-shape patterns are consistent across diseases.
    
    If the QO+R framework is correct, U-shapes should appear where
    competing mechanisms create regime transitions.
    """
    
    print("\n" + "=" * 60)
    print("U-SHAPE CONSISTENCY ACROSS DISEASES")
    print("=" * 60)
    
    # Key residuals to test
    residuals = ['HOMA_IR_Residual', 'TG_HDL_Residual', 'FIB4_Residual', 
                 'eGFR_Residual', 'VAI_Residual']
    residuals = [r for r in residuals if r in df.columns]
    
    # Common covariates
    covariates = ['Age', 'BMI']
    covariates = [c for c in covariates if c in df.columns]
    
    results = []
    
    for resid_col in residuals:
        for cov in covariates:
            # Test in each disease subgroup
            for disease, config in DISEASE_OUTCOMES.items():
                outcome_col = config['outcome_col']
                
                if outcome_col not in df.columns:
                    continue
                
                # Test in cases
                if df[outcome_col].dtype == 'object':
                    mask_case = (df[outcome_col] == config['positive_value'])
                else:
                    mask_case = (df[outcome_col] == config['positive_value'])
                
                for group, mask in [('Cases', mask_case), ('Controls', ~mask_case)]:
                    df_sub = df[mask]
                    
                    # Get complete cases
                    complete_mask = df_sub[[resid_col, cov]].notna().all(axis=1)
                    
                    if complete_mask.sum() < 100:
                        continue
                    
                    x = df_sub.loc[complete_mask, cov].values
                    y = df_sub.loc[complete_mask, resid_col].values
                    
                    # Standardize
                    x_std = (x - x.mean()) / x.std()
                    
                    # Fit quadratic
                    quad_coef = np.polyfit(x_std, y, 2)
                    y_quad = np.polyval(quad_coef, x_std)
                    ss_quad = np.sum((y - y_quad) ** 2)
                    
                    # Fit linear
                    lin_coef = np.polyfit(x_std, y, 1)
                    y_lin = np.polyval(lin_coef, x_std)
                    ss_lin = np.sum((y - y_lin) ** 2)
                    
                    # F-test
                    n = len(y)
                    if ss_quad > 0:
                        f_stat = ((ss_lin - ss_quad) / 1) / (ss_quad / (n - 3))
                        f_p = 1 - stats.f.cdf(f_stat, 1, n - 3)
                    else:
                        f_stat, f_p = np.nan, np.nan
                    
                    a = quad_coef[0]
                    
                    results.append({
                        'Residual': resid_col.replace('_Residual', ''),
                        'Covariate': cov,
                        'Disease': disease,
                        'Group': group,
                        'N': n,
                        'Quad_Coef': a,
                        'F_p': f_p,
                        'U_Shape': a > 0,
                        'Significant': f_p < 0.05 if not np.isnan(f_p) else False
                    })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        results_df.to_csv(RESULTS / "ushape_consistency.csv", index=False)
        
        # Analyze consistency
        print("\nU-Shape Consistency Summary:")
        print("-" * 50)
        
        # Group by residual-covariate pair
        for resid in residuals:
            resid_name = resid.replace('_Residual', '')
            subset = results_df[results_df['Residual'] == resid_name]
            
            if len(subset) == 0:
                continue
            
            n_sig = subset['Significant'].sum()
            n_ushape = (subset['Significant'] & subset['U_Shape']).sum()
            
            print(f"\n  {resid_name}:")
            print(f"    Significant quadratic: {n_sig}/{len(subset)}")
            print(f"    U-shapes (a > 0): {n_ushape}/{n_sig if n_sig > 0 else 1}")
            
            # Check consistency of direction
            if n_sig > 0:
                sig_subset = subset[subset['Significant']]
                all_ushape = sig_subset['U_Shape'].all()
                all_inverted = (~sig_subset['U_Shape']).all()
                
                if all_ushape:
                    print(f"    → CONSISTENT U-shape across diseases")
                elif all_inverted:
                    print(f"    → CONSISTENT inverted-U across diseases")
                else:
                    print(f"    → Mixed patterns")
    
    return results_df

def analyze_disease_comorbidity_patterns(df: pd.DataFrame) -> dict:
    """
    Analyze how residual patterns relate to disease comorbidity.
    
    Hypothesis: Patients with multiple diseases may show distinct
    residual signatures compared to single-disease patients.
    """
    
    print("\n" + "=" * 60)
    print("COMORBIDITY AND RESIDUAL PATTERNS")
    print("=" * 60)
    
    # Count diseases per person
    disease_cols = ['HasCVD', 'HasDiabetes', 'HasMetS', 'HasCKD']
    disease_cols = [c for c in disease_cols if c in df.columns]
    
    if len(disease_cols) < 2:
        print("Not enough disease indicators")
        return {}
    
    df['Disease_Count'] = df[disease_cols].sum(axis=1)
    
    # Distribution
    count_dist = df['Disease_Count'].value_counts().sort_index()
    print("\nDisease Comorbidity Distribution:")
    print("-" * 40)
    for count, n in count_dist.items():
        pct = 100 * n / len(df)
        print(f"  {int(count)} diseases: {n:,} ({pct:.1f}%)")
    
    results = {}
    
    # Test if residuals differ by comorbidity count
    residuals = ['HOMA_IR_Residual', 'TG_HDL_Residual']
    residuals = [r for r in residuals if r in df.columns]
    
    for resid_col in residuals:
        count_data = {}
        for count in range(5):
            mask = (df['Disease_Count'] == count) & df[resid_col].notna()
            count_data[count] = df.loc[mask, resid_col].values
        
        valid_counts = [c for c in range(5) if len(count_data.get(c, [])) >= 30]
        
        if len(valid_counts) >= 3:
            valid_data = [count_data[c] for c in valid_counts]
            h_stat, p = stats.kruskal(*valid_data)
            
            ratio_name = resid_col.replace('_Residual', '')
            results[ratio_name] = {'H': h_stat, 'p': p}
            
            print(f"\n{ratio_name} residual by disease count:")
            print(f"  Kruskal-Wallis H = {h_stat:.2f}, p = {p:.4f}")
            
            if p < 0.05:
                print(f"  *** SIGNIFICANT ***")
                for count in valid_counts:
                    mean = np.mean(count_data[count])
                    print(f"    {count} diseases: mean = {mean:.3f}")
    
    return results

def generate_cross_disease_visualizations(df: pd.DataFrame, 
                                          corr_results: pd.DataFrame,
                                          universal_results: pd.DataFrame):
    """Generate cross-disease analysis figures."""
    
    print("\n" + "=" * 60)
    print("GENERATING CROSS-DISEASE VISUALIZATIONS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Figure 1: Residual correlation heatmap
    ax = axes[0, 0]
    
    residuals = ['HOMA_IR_Residual', 'TG_HDL_Residual', 'FIB4_Residual', 
                 'eGFR_Residual', 'VAI_Residual', 'QUICKI_Residual']
    residuals = [r for r in residuals if r in df.columns]
    
    if len(residuals) >= 2:
        resid_df = df[residuals].dropna()
        if len(resid_df) >= 100:
            corr = resid_df.corr(method='spearman')
            
            # Clean labels
            labels = [r.replace('_Residual', '') for r in residuals]
            
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                       xticklabels=labels, yticklabels=labels,
                       center=0, vmin=-0.5, vmax=0.5, ax=ax)
            ax.set_title('Cross-Disease Residual Correlations')
    
    # Figure 2: Universal residuals bar chart
    ax = axes[0, 1]
    
    if len(universal_results) > 0:
        top_results = universal_results.head(10)
        
        colors = ['#e74c3c' if univ else '#3498db' 
                  for univ in top_results['Universal']]
        
        bars = ax.barh(range(len(top_results)), top_results['N_Significant'],
                       color=colors, edgecolor='black')
        
        ax.set_yticks(range(len(top_results)))
        ax.set_yticklabels(top_results['Ratio'])
        ax.set_xlabel('Number of Diseases (out of 5)')
        ax.set_title('Residuals Significant Across Multiple Diseases')
        ax.axvline(x=3, color='red', linestyle='--', label='Universal threshold')
        ax.legend()
    
    # Figure 3: Disease comorbidity vs. HOMA-IR residual
    ax = axes[1, 0]
    
    if 'Disease_Count' in df.columns and 'HOMA_IR_Residual' in df.columns:
        # Box plot
        data_by_count = []
        labels = []
        
        for count in range(5):
            data = df.loc[df['Disease_Count'] == count, 'HOMA_IR_Residual'].dropna()
            if len(data) >= 30:
                data_by_count.append(data)
                labels.append(str(int(count)))
        
        if data_by_count:
            bp = ax.boxplot(data_by_count, labels=labels, patch_artist=True)
            
            colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(data_by_count)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Number of Diseases')
            ax.set_ylabel('HOMA-IR Residual')
            ax.set_title('HOMA-IR Residual by Disease Comorbidity')
    
    # Figure 4: Scatter of TG/HDL vs HOMA-IR residuals, colored by disease count
    ax = axes[1, 1]
    
    if all(c in df.columns for c in ['HOMA_IR_Residual', 'TG_HDL_Residual', 'Disease_Count']):
        mask = df[['HOMA_IR_Residual', 'TG_HDL_Residual', 'Disease_Count']].notna().all(axis=1)
        df_plot = df[mask].sample(min(2000, mask.sum()), random_state=42)
        
        scatter = ax.scatter(df_plot['HOMA_IR_Residual'], 
                            df_plot['TG_HDL_Residual'],
                            c=df_plot['Disease_Count'],
                            cmap='Reds', alpha=0.5, s=20)
        
        plt.colorbar(scatter, ax=ax, label='Disease Count')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('HOMA-IR Residual')
        ax.set_ylabel('TG/HDL Residual')
        ax.set_title('Residual Space Colored by Comorbidity')
    
    plt.tight_layout()
    plt.savefig(FIGURES / "cross_disease_fig01_overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Cross-disease figure saved")

def generate_cross_disease_report(corr_results: pd.DataFrame,
                                  universal_results: pd.DataFrame,
                                  ushape_results: pd.DataFrame,
                                  comorbidity_results: dict):
    """Generate comprehensive cross-disease report."""
    
    report = []
    report.append("=" * 70)
    report.append("PAPER 2 EXTENDED: CROSS-DISEASE PATTERN ANALYSIS")
    report.append("The Revelatory Division - Testing Residual Diagnostics")
    report.append("=" * 70)
    report.append("")
    report.append("Author: Jonathan Édouard Slama")
    report.append("Affiliation: Metafund Research Division, Strasbourg, France")
    report.append("")
    
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 40)
    report.append("")
    report.append("This analysis tests whether the Revelatory Division principle")
    report.append("generates UNIVERSAL patterns across disease categories, or")
    report.append("whether findings are disease-specific.")
    report.append("")
    
    # Cross-correlations
    report.append("1. CROSS-DISEASE RESIDUAL CORRELATIONS")
    report.append("-" * 40)
    if len(corr_results) > 0:
        n_sig = corr_results['Significant'].sum()
        n_strong = (corr_results['Significant'] & corr_results['Strong']).sum()
        report.append(f"   Cross-category correlations tested: {len(corr_results)}")
        report.append(f"   Significant (p < 0.001): {n_sig}")
        report.append(f"   Strong (|r| > 0.3): {n_strong}")
        report.append("")
        report.append("   Interpretation: Correlated residuals suggest shared")
        report.append("   underlying mechanisms not captured by individual ratios.")
    report.append("")
    
    # Universal residuals
    report.append("2. UNIVERSAL RESIDUAL PATTERNS")
    report.append("-" * 40)
    if len(universal_results) > 0:
        n_universal = universal_results['Universal'].sum()
        report.append(f"   Residuals tested: {len(universal_results)}")
        report.append(f"   Universal (≥3 diseases): {n_universal}")
        report.append("")
        
        if n_universal > 0:
            universal_residuals = universal_results[universal_results['Universal']]
            report.append("   Universal residuals:")
            for _, row in universal_residuals.iterrows():
                report.append(f"     - {row['Ratio']}: {row['N_Significant']}/5 diseases")
        report.append("")
        report.append("   Interpretation: Universal residuals may capture fundamental")
        report.append("   metabolic dysfunction that manifests across disease categories.")
    report.append("")
    
    # U-shape consistency
    report.append("3. U-SHAPE PATTERN CONSISTENCY")
    report.append("-" * 40)
    if len(ushape_results) > 0:
        n_sig = ushape_results['Significant'].sum()
        n_ushape = (ushape_results['Significant'] & ushape_results['U_Shape']).sum()
        report.append(f"   Significant quadratic patterns: {n_sig}/{len(ushape_results)}")
        report.append(f"   U-shapes (a > 0): {n_ushape}/{n_sig if n_sig > 0 else 1}")
        report.append("")
        report.append("   Interpretation: Consistent U-shapes across diseases would")
        report.append("   support the QO+R framework's prediction of regime transitions.")
    report.append("")
    
    # Comorbidity
    report.append("4. COMORBIDITY PATTERNS")
    report.append("-" * 40)
    if comorbidity_results:
        for ratio, vals in comorbidity_results.items():
            sig = "***" if vals['p'] < 0.05 else ""
            report.append(f"   {ratio}: H = {vals['H']:.2f}, p = {vals['p']:.4f} {sig}")
        report.append("")
        report.append("   Interpretation: Residuals that differ by comorbidity count")
        report.append("   may reflect cumulative metabolic dysfunction.")
    report.append("")
    
    # Conclusions
    report.append("=" * 70)
    report.append("CONCLUSIONS")
    report.append("=" * 70)
    report.append("")
    report.append("The cross-disease analysis tests the core prediction of the")
    report.append("Revelatory Division: that residuals reveal UNIVERSAL patterns")
    report.append("of metabolic dysfunction, not just disease-specific noise.")
    report.append("")
    report.append("If confirmed, this suggests:")
    report.append("  1. Clinical ratios share a common mathematical structure")
    report.append("  2. Residuals capture shared pathophysiology")
    report.append("  3. The QO+R framework may have broad applicability")
    report.append("")
    report.append("Limitations:")
    report.append("  - Cross-sectional data cannot prove causation")
    report.append("  - Shared confounders may drive apparent correlations")
    report.append("  - Validation in independent datasets required")
    report.append("")
    
    # Save
    report_text = "\n".join(report)
    with open(RESULTS / "cross_disease_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + report_text)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    
    print("=" * 70)
    print("PAPER 2 EXTENDED: CROSS-DISEASE PATTERN ANALYSIS")
    print("The Revelatory Division - Testing Residual Diagnostics")
    print("=" * 70)
    print()
    print("Author: Jonathan Édouard Slama")
    print("Affiliation: Metafund Research Division, Strasbourg, France")
    print()
    
    # Load data
    print("Loading data...")
    df = load_data()
    print(f"✓ Loaded {len(df):,} participants")
    
    # Run analyses
    corr_results = analyze_residual_correlations(df)
    universal_results = analyze_universal_residual_patterns(df)
    ushape_results = analyze_ushape_consistency(df)
    comorbidity_results = analyze_disease_comorbidity_patterns(df)
    
    # Generate outputs
    generate_cross_disease_visualizations(df, corr_results, universal_results)
    generate_cross_disease_report(corr_results, universal_results, 
                                  ushape_results, comorbidity_results)
    
    print()
    print("=" * 70)
    print("✓ Cross-disease analysis complete!")
    print(f"  Results saved to: {RESULTS}")
    print(f"  Figures saved to: {FIGURES}")
    print()
    print("Next step: Run 09_summary_report.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
