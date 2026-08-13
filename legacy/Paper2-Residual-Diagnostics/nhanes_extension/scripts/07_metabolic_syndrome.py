#!/usr/bin/env python3
"""
Paper 2 Extended: Metabolic Syndrome Analysis
The Revelatory Division - Testing Residual Diagnostics

Analyzes whether residuals from metabolic syndrome components
contain diagnostic information for MetS and its progression.

Background:
-----------
Metabolic Syndrome (MetS) is defined by the presence of ≥3 of 5 criteria:
1. Elevated waist circumference (>102 cm men, >88 cm women)
2. Elevated triglycerides (≥150 mg/dL)
3. Low HDL (<40 mg/dL men, <50 mg/dL women)
4. Elevated blood pressure (≥130/85 mmHg)
5. Elevated fasting glucose (≥100 mg/dL)

QO+R Interpretation:
-------------------
Each MetS component represents a balance between regulatory mechanisms:
- Waist: Energy storage (Q) vs. expenditure (R)
- TG/HDL: Lipid production vs. clearance
- BP: Vasoconstriction vs. vasodilation
- Glucose: Hepatic production vs. peripheral uptake

The residuals from component-based models may reveal:
- Which individuals at "borderline" counts (2-3 components) will progress
- Hidden metabolic dysfunction not captured by binary criteria
- Subgroups with different underlying pathophysiology

Author: Jonathan Édouard Slama
Email: jonathan@metafund.in
Affiliation: Metafund Research Division, Strasbourg, France
ORCID: 0009-0002-1292-4350
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
    """Load processed NHANES data with computed ratios."""
    filepath = PROCESSED_DATA / "nhanes_with_ratios.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Data not found: {filepath}\nRun 02_compute_all_ratios.py first")
    return pd.read_csv(filepath)

# =============================================================================
# METABOLIC SYNDROME ANALYSIS
# =============================================================================

def analyze_mets_distribution(df: pd.DataFrame) -> dict:
    """
    Analyze the distribution of metabolic syndrome components.
    
    This establishes the baseline prevalence and identifies the
    "gray zone" (2-3 components) where diagnostic uncertainty is highest.
    """
    
    print("\n" + "=" * 60)
    print("METABOLIC SYNDROME DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    results = {}
    
    if 'MetS_Count' not in df.columns:
        print("MetS component count not available")
        return results
    
    # Component count distribution
    count_dist = df['MetS_Count'].value_counts().sort_index()
    print("\nMetabolic Syndrome Component Distribution:")
    print("-" * 40)
    
    for count, n in count_dist.items():
        pct = 100 * n / df['MetS_Count'].notna().sum()
        status = "MetS" if count >= 3 else "No MetS"
        print(f"  {int(count)} components: {n:,} ({pct:.1f}%) - {status}")
    
    # MetS prevalence
    if 'HasMetS' in df.columns:
        mets_prev = 100 * df['HasMetS'].mean()
        print(f"\nOverall MetS prevalence: {mets_prev:.1f}%")
        results['mets_prevalence'] = mets_prev
    
    # Gray zone: 2-3 components (diagnostic uncertainty)
    gray_zone_mask = (df['MetS_Count'] >= 2) & (df['MetS_Count'] <= 3)
    gray_zone_pct = 100 * gray_zone_mask.sum() / df['MetS_Count'].notna().sum()
    
    print(f"\n→ Gray zone (2-3 components): {gray_zone_pct:.1f}%")
    print("  These individuals are at the diagnostic boundary")
    print("  Some with 2 components will progress; some with 3 may regress")
    
    results['gray_zone_pct'] = gray_zone_pct
    results['n_gray_zone'] = gray_zone_mask.sum()
    
    # Individual component prevalence
    print("\nIndividual Component Prevalence:")
    print("-" * 40)
    
    components = ['MetS_Waist', 'MetS_TG', 'MetS_HDL', 'MetS_BP', 'MetS_Glucose']
    for comp in components:
        if comp in df.columns:
            prev = 100 * df[comp].mean()
            name = comp.replace('MetS_', '')
            print(f"  {name}: {prev:.1f}%")
            results[f'{comp}_prev'] = prev
    
    return results

def test_residuals_by_mets_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test whether residual distributions differ between MetS and non-MetS.
    
    Hypothesis: If the Revelatory Division applies, residuals should show
    systematic differences between groups, even after controlling for
    the variables used to compute the ratios.
    """
    
    print("\n" + "=" * 60)
    print("TEST 1: RESIDUALS BY METABOLIC SYNDROME STATUS")
    print("=" * 60)
    
    if 'HasMetS' not in df.columns:
        print("MetS status not available")
        return pd.DataFrame()
    
    results = []
    
    # Get all residual columns
    residual_cols = [c for c in df.columns if c.endswith('_Residual')]
    
    for resid_col in residual_cols:
        # Get data for each group
        mask_mets = (df['HasMetS'] == 1) & df[resid_col].notna()
        mask_no_mets = (df['HasMetS'] == 0) & df[resid_col].notna()
        
        mets_data = df.loc[mask_mets, resid_col].values
        no_mets_data = df.loc[mask_no_mets, resid_col].values
        
        if len(mets_data) < 50 or len(no_mets_data) < 50:
            continue
        
        # Statistical tests
        # Mann-Whitney U (non-parametric)
        mw_stat, mw_p = stats.mannwhitneyu(mets_data, no_mets_data, alternative='two-sided')
        
        # Kolmogorov-Smirnov (distribution shape)
        ks_stat, ks_p = stats.ks_2samp(mets_data, no_mets_data)
        
        # Effect size (Cohen's d)
        n1, n2 = len(mets_data), len(no_mets_data)
        pooled_std = np.sqrt(((n1-1)*np.var(mets_data) + (n2-1)*np.var(no_mets_data)) / (n1+n2-2))
        cohens_d = (np.mean(mets_data) - np.mean(no_mets_data)) / pooled_std if pooled_std > 0 else 0
        
        # Levene's test (variance equality)
        lev_stat, lev_p = stats.levene(mets_data, no_mets_data)
        
        ratio_name = resid_col.replace('_Residual', '')
        
        results.append({
            'Ratio': ratio_name,
            'Residual': resid_col,
            'N_MetS': n1,
            'N_NoMetS': n2,
            'Mean_MetS': np.mean(mets_data),
            'Mean_NoMetS': np.mean(no_mets_data),
            'Std_MetS': np.std(mets_data),
            'Std_NoMetS': np.std(no_mets_data),
            'Cohen_d': cohens_d,
            'MW_p': mw_p,
            'KS_p': ks_p,
            'Levene_p': lev_p,
            'Location_Diff': mw_p < 0.05,
            'Shape_Diff': ks_p < 0.05,
            'Variance_Diff': lev_p < 0.05
        })
        
        # Report significant findings
        if mw_p < 0.05:
            print(f"\n{ratio_name}: *** SIGNIFICANT LOCATION DIFFERENCE ***")
            print(f"  MetS: {np.mean(mets_data):.3f} ± {np.std(mets_data):.3f}")
            print(f"  No MetS: {np.mean(no_mets_data):.3f} ± {np.std(no_mets_data):.3f}")
            print(f"  Cohen's d = {cohens_d:.3f}, p = {mw_p:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS / "mets_test1_residuals_by_status.csv", index=False)
    
    # Summary
    n_loc = results_df['Location_Diff'].sum() if len(results_df) > 0 else 0
    n_shape = results_df['Shape_Diff'].sum() if len(results_df) > 0 else 0
    print(f"\n→ Summary:")
    print(f"  Location differences: {n_loc}/{len(results_df)} residuals")
    print(f"  Shape differences: {n_shape}/{len(results_df)} residuals")
    
    return results_df

def test_residuals_by_component_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test residual patterns across MetS component counts (0-5).
    
    Key question: Do residuals show a gradient across component counts,
    or do they show non-linear (U-shaped) patterns?
    """
    
    print("\n" + "=" * 60)
    print("TEST 2: RESIDUALS BY COMPONENT COUNT")
    print("=" * 60)
    
    if 'MetS_Count' not in df.columns:
        print("MetS component count not available")
        return pd.DataFrame()
    
    results = []
    
    residual_cols = [c for c in df.columns if c.endswith('_Residual')]
    
    for resid_col in residual_cols:
        # Get data for each component count
        count_data = {}
        for count in range(6):
            mask = (df['MetS_Count'] == count) & df[resid_col].notna()
            count_data[count] = df.loc[mask, resid_col].values
        
        # Need at least 30 in each group for reliable testing
        valid_counts = [c for c in range(6) if len(count_data.get(c, [])) >= 30]
        
        if len(valid_counts) < 3:
            continue
        
        # Kruskal-Wallis test (non-parametric ANOVA)
        valid_data = [count_data[c] for c in valid_counts]
        h_stat, kw_p = stats.kruskal(*valid_data)
        
        # Jonckheere-Terpstra trend test (monotonic trend)
        # Approximation using Spearman correlation
        all_x = []
        all_y = []
        for count in valid_counts:
            all_x.extend([count] * len(count_data[count]))
            all_y.extend(count_data[count])
        
        spearman_r, spearman_p = stats.spearmanr(all_x, all_y)
        
        # Test for quadratic (U-shape) pattern
        x_vals = np.array(all_x)
        y_vals = np.array(all_y)
        
        # Standardize for numerical stability
        x_std = (x_vals - x_vals.mean()) / x_vals.std()
        
        # Linear fit
        lin_coef = np.polyfit(x_std, y_vals, 1)
        y_lin = np.polyval(lin_coef, x_std)
        ss_lin = np.sum((y_vals - y_lin) ** 2)
        
        # Quadratic fit
        quad_coef = np.polyfit(x_std, y_vals, 2)
        y_quad = np.polyval(quad_coef, x_std)
        ss_quad = np.sum((y_vals - y_quad) ** 2)
        
        # F-test for quadratic improvement
        n = len(y_vals)
        if ss_quad > 0:
            f_stat = ((ss_lin - ss_quad) / 1) / (ss_quad / (n - 3))
            f_p = 1 - stats.f.cdf(f_stat, 1, n - 3)
        else:
            f_stat, f_p = np.nan, np.nan
        
        a = quad_coef[0]  # Quadratic coefficient
        shape = "U-shape" if a > 0 else "Inverted-U" if a < 0 else "Linear"
        
        ratio_name = resid_col.replace('_Residual', '')
        
        results.append({
            'Ratio': ratio_name,
            'N_Total': n,
            'KW_H': h_stat,
            'KW_p': kw_p,
            'Spearman_r': spearman_r,
            'Spearman_p': spearman_p,
            'Quad_Coef': a,
            'F_Stat': f_stat,
            'F_p': f_p,
            'Pattern': shape,
            'Group_Diff': kw_p < 0.05,
            'Monotonic': (spearman_p < 0.05) and (abs(spearman_r) > 0.1),
            'NonLinear': f_p < 0.05 if not np.isnan(f_p) else False
        })
        
        # Report interesting findings
        if kw_p < 0.05:
            print(f"\n{ratio_name}:")
            print(f"  Group differences: H = {h_stat:.2f}, p = {kw_p:.4f}")
            print(f"  Trend: r = {spearman_r:.3f} ({'increasing' if spearman_r > 0 else 'decreasing'})")
            if f_p < 0.05:
                print(f"  *** {shape} DETECTED *** (a = {a:.4f}, p = {f_p:.4f})")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS / "mets_test2_residuals_by_count.csv", index=False)
    
    return results_df

def test_gray_zone_discrimination(df: pd.DataFrame) -> dict:
    """
    Test if residuals can discriminate within the MetS gray zone (2-3 components).
    
    Clinical relevance: Patients with 2-3 components are at the diagnostic
    boundary. If residuals can identify who will progress vs. regress,
    this has immediate clinical utility.
    """
    
    print("\n" + "=" * 60)
    print("TEST 3: DISCRIMINATION IN GRAY ZONE (2-3 COMPONENTS)")
    print("=" * 60)
    
    if 'MetS_Count' not in df.columns:
        print("MetS component count not available")
        return {}
    
    # Focus on gray zone
    gray_mask = (df['MetS_Count'] >= 2) & (df['MetS_Count'] <= 3)
    df_gray = df[gray_mask].copy()
    
    print(f"\nAnalyzing {len(df_gray)} participants in gray zone")
    
    # Define outcome: MetS (3 components) vs. sub-threshold (2 components)
    df_gray['MetS_Outcome'] = (df_gray['MetS_Count'] >= 3).astype(int)
    
    outcome_prev = 100 * df_gray['MetS_Outcome'].mean()
    print(f"MetS (≥3 components) in gray zone: {outcome_prev:.1f}%")
    
    results = {}
    
    # Get feature sets
    ratio_cols = ['HOMA_IR', 'TG_HDL', 'TC_HDL', 'TyG', 'WHtR', 'VAI']
    ratio_cols = [c for c in ratio_cols if c in df_gray.columns]
    
    resid_cols = [f"{c}_Residual" for c in ratio_cols if f"{c}_Residual" in df_gray.columns]
    
    print(f"\nUsing {len(ratio_cols)} ratios and {len(resid_cols)} residuals")
    
    # Model 1: Ratios only
    if ratio_cols:
        all_cols = ratio_cols + ['MetS_Outcome']
        mask_complete = df_gray[all_cols].notna().all(axis=1)
        
        if mask_complete.sum() >= 100:
            X = df_gray.loc[mask_complete, ratio_cols].values
            y = df_gray.loc[mask_complete, 'MetS_Outcome'].values
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LogisticRegression(max_iter=1000, random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            try:
                auc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
                results['auc_ratios'] = auc_scores.mean()
                results['auc_ratios_std'] = auc_scores.std()
                print(f"\nRatios only: AUC = {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
            except Exception as e:
                print(f"Ratio model failed: {e}")
    
    # Model 2: Residuals only
    if resid_cols:
        all_cols = resid_cols + ['MetS_Outcome']
        mask_complete = df_gray[all_cols].notna().all(axis=1)
        
        if mask_complete.sum() >= 100:
            X = df_gray.loc[mask_complete, resid_cols].values
            y = df_gray.loc[mask_complete, 'MetS_Outcome'].values
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LogisticRegression(max_iter=1000, random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            try:
                auc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
                results['auc_residuals'] = auc_scores.mean()
                results['auc_residuals_std'] = auc_scores.std()
                print(f"Residuals only: AUC = {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
            except Exception as e:
                print(f"Residual model failed: {e}")
    
    # Model 3: Combined
    all_features = ratio_cols + resid_cols
    if all_features:
        all_cols = all_features + ['MetS_Outcome']
        mask_complete = df_gray[all_cols].notna().all(axis=1)
        
        if mask_complete.sum() >= 100:
            X = df_gray.loc[mask_complete, all_features].values
            y = df_gray.loc[mask_complete, 'MetS_Outcome'].values
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LogisticRegression(max_iter=1000, random_state=42, penalty='l2')
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            try:
                auc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
                results['auc_combined'] = auc_scores.mean()
                results['auc_combined_std'] = auc_scores.std()
                print(f"Combined: AUC = {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
            except Exception as e:
                print(f"Combined model failed: {e}")
    
    # Calculate improvement
    if 'auc_ratios' in results and 'auc_combined' in results:
        improvement = results['auc_combined'] - results['auc_ratios']
        results['improvement'] = improvement
        print(f"\n→ Improvement from adding residuals: {improvement:+.3f} AUC")
    
    return results

def generate_visualizations(df: pd.DataFrame):
    """Generate figures for metabolic syndrome analysis."""
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Figure 1a: Component count distribution
    ax = axes[0, 0]
    if 'MetS_Count' in df.columns:
        counts = df['MetS_Count'].value_counts().sort_index()
        colors = ['#2ecc71', '#2ecc71', '#f39c12', '#f39c12', '#e74c3c', '#e74c3c']
        ax.bar(counts.index, counts.values, color=colors[:len(counts)], edgecolor='black')
        ax.axvline(x=2.5, color='red', linestyle='--', linewidth=2, label='MetS threshold')
        ax.set_xlabel('Number of MetS Components')
        ax.set_ylabel('Count')
        ax.set_title('Metabolic Syndrome Component Distribution')
        ax.legend()
    
    # Figure 1b: HOMA-IR residuals by MetS status
    ax = axes[0, 1]
    if 'HOMA_IR_Residual' in df.columns and 'HasMetS' in df.columns:
        for status, label, color in [(0, 'No MetS', '#2ecc71'), (1, 'MetS', '#e74c3c')]:
            data = df.loc[df['HasMetS'] == status, 'HOMA_IR_Residual'].dropna()
            ax.hist(data, bins=50, alpha=0.5, label=label, color=color, density=True)
        
        ax.axvline(x=0, color='gray', linestyle='--')
        ax.set_xlabel('HOMA-IR Residual')
        ax.set_ylabel('Density')
        ax.set_title('HOMA-IR Residuals by MetS Status')
        ax.legend()
    
    # Figure 1c: Residual means by component count
    ax = axes[1, 0]
    if 'HOMA_IR_Residual' in df.columns and 'MetS_Count' in df.columns:
        means = []
        stds = []
        counts_list = []
        
        for count in range(6):
            data = df.loc[df['MetS_Count'] == count, 'HOMA_IR_Residual'].dropna()
            if len(data) >= 30:
                means.append(data.mean())
                stds.append(data.std() / np.sqrt(len(data)))  # SEM
                counts_list.append(count)
        
        if means:
            ax.errorbar(counts_list, means, yerr=stds, fmt='o-', capsize=5, 
                       color='#3498db', markersize=8, linewidth=2)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.axvspan(1.5, 3.5, alpha=0.2, color='yellow', label='Gray zone')
            ax.set_xlabel('Number of MetS Components')
            ax.set_ylabel('Mean HOMA-IR Residual (± SEM)')
            ax.set_title('HOMA-IR Residual Trend Across MetS Severity')
            ax.legend()
    
    # Figure 1d: TG/HDL residual vs HOMA-IR residual, colored by MetS
    ax = axes[1, 1]
    if all(c in df.columns for c in ['HOMA_IR_Residual', 'TG_HDL_Residual', 'HasMetS']):
        for status, label, color in [(0, 'No MetS', '#2ecc71'), (1, 'MetS', '#e74c3c')]:
            mask = (df['HasMetS'] == status) & df['HOMA_IR_Residual'].notna() & df['TG_HDL_Residual'].notna()
            x = df.loc[mask, 'HOMA_IR_Residual']
            y = df.loc[mask, 'TG_HDL_Residual']
            ax.scatter(x, y, alpha=0.3, s=10, label=label, color=color)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('HOMA-IR Residual')
        ax.set_ylabel('TG/HDL Residual')
        ax.set_title('Residual Space: HOMA-IR vs TG/HDL')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES / "mets_fig01_overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Figure 1: MetS overview saved")

def generate_report(dist_results: dict, status_results: pd.DataFrame,
                   count_results: pd.DataFrame, gray_results: dict):
    """Generate summary report for metabolic syndrome analysis."""
    
    report = []
    report.append("=" * 70)
    report.append("PAPER 2 EXTENDED: METABOLIC SYNDROME ANALYSIS REPORT")
    report.append("The Revelatory Division - Testing Residual Diagnostics")
    report.append("=" * 70)
    report.append("")
    
    report.append("Author: Jonathan Édouard Slama")
    report.append("Affiliation: Metafund Research Division, Strasbourg, France")
    report.append("")
    
    report.append("1. METABOLIC SYNDROME DISTRIBUTION")
    report.append("-" * 40)
    if 'mets_prevalence' in dist_results:
        report.append(f"   Overall MetS prevalence: {dist_results['mets_prevalence']:.1f}%")
    if 'gray_zone_pct' in dist_results:
        report.append(f"   Gray zone (2-3 components): {dist_results['gray_zone_pct']:.1f}%")
    report.append("")
    
    report.append("2. RESIDUALS BY MetS STATUS")
    report.append("-" * 40)
    if len(status_results) > 0:
        n_loc = status_results['Location_Diff'].sum()
        n_shape = status_results['Shape_Diff'].sum()
        report.append(f"   Location differences: {n_loc}/{len(status_results)}")
        report.append(f"   Shape differences: {n_shape}/{len(status_results)}")
        
        # Top findings
        sig_results = status_results[status_results['Location_Diff']].nlargest(3, 'Cohen_d')
        if len(sig_results) > 0:
            report.append("   Top findings:")
            for _, row in sig_results.iterrows():
                report.append(f"     - {row['Ratio']}: d = {row['Cohen_d']:.3f}, p = {row['MW_p']:.4f}")
    report.append("")
    
    report.append("3. RESIDUALS BY COMPONENT COUNT")
    report.append("-" * 40)
    if len(count_results) > 0:
        n_diff = count_results['Group_Diff'].sum()
        n_nonlin = count_results['NonLinear'].sum()
        report.append(f"   Group differences: {n_diff}/{len(count_results)}")
        report.append(f"   Non-linear patterns: {n_nonlin}/{len(count_results)}")
    report.append("")
    
    report.append("4. GRAY ZONE DISCRIMINATION")
    report.append("-" * 40)
    if gray_results:
        if 'auc_ratios' in gray_results:
            report.append(f"   Ratios only: AUC = {gray_results['auc_ratios']:.3f}")
        if 'auc_residuals' in gray_results:
            report.append(f"   Residuals only: AUC = {gray_results['auc_residuals']:.3f}")
        if 'auc_combined' in gray_results:
            report.append(f"   Combined: AUC = {gray_results['auc_combined']:.3f}")
        if 'improvement' in gray_results:
            report.append(f"   Improvement: {gray_results['improvement']:+.3f} AUC")
    report.append("")
    
    report.append("=" * 70)
    report.append("INTERPRETATION")
    report.append("=" * 70)
    report.append("")
    report.append("The metabolic syndrome represents a cluster of interrelated risk factors.")
    report.append("The 'gray zone' (2-3 components) is clinically important because:")
    report.append("  - These patients are at the diagnostic boundary")
    report.append("  - Standard criteria give ambiguous guidance")
    report.append("  - Progression risk varies substantially")
    report.append("")
    report.append("If residuals improve discrimination in the gray zone, it suggests that:")
    report.append("  - The standard component count misses hidden metabolic dysfunction")
    report.append("  - Residual patterns may identify different MetS subtypes")
    report.append("  - Personalized risk stratification may be possible")
    report.append("")
    
    # Save report
    report_text = "\n".join(report)
    with open(RESULTS / "mets_analysis_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + report_text)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    
    print("=" * 70)
    print("PAPER 2 EXTENDED: METABOLIC SYNDROME ANALYSIS")
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
    dist_results = analyze_mets_distribution(df)
    status_results = test_residuals_by_mets_status(df)
    count_results = test_residuals_by_component_count(df)
    gray_results = test_gray_zone_discrimination(df)
    
    # Generate outputs
    generate_visualizations(df)
    generate_report(dist_results, status_results, count_results, gray_results)
    
    print()
    print("=" * 70)
    print("✓ Metabolic syndrome analysis complete!")
    print(f"  Results saved to: {RESULTS}")
    print(f"  Figures saved to: {FIGURES}")
    print()
    print("Next step: Run 08_cross_disease.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
