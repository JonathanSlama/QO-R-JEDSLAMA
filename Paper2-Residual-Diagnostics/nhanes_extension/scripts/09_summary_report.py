#!/usr/bin/env python3
"""
Paper 2 Extended: Final Summary Report Generator
The Revelatory Division - Multi-Disease Validation

This script compiles all analysis results into a comprehensive
summary report suitable for manuscript preparation.

Purpose:
--------
1. Aggregate results from all disease-specific analyses
2. Synthesize cross-disease patterns
3. Generate publication-ready summary statistics
4. Assess overall support for the Revelatory Division hypothesis

Author: Jonathan Édouard Slama
Email: jonathan@metafund.in
Affiliation: Metafund Research Division, Strasbourg, France
ORCID: 0009-0002-1292-4350
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
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
MANUSCRIPT = PROJECT_DIR / "manuscript"

MANUSCRIPT.mkdir(parents=True, exist_ok=True)

# =============================================================================
# RESULT AGGREGATION FUNCTIONS
# =============================================================================

def load_all_results() -> dict:
    """Load all analysis result files."""
    
    print("Loading analysis results...")
    
    results = {}
    
    # Define expected result files
    result_files = {
        # Ratio computation
        'ratio_summary': 'ratio_summary.csv',
        'residual_stats': 'residual_computation_stats.csv',
        
        # Liver analysis
        'liver_test1': 'liver_test1_residual_distributions.csv',
        'liver_test2': 'liver_test2_ushapes_indeterminate.csv',
        
        # Kidney analysis
        'kidney_test1': 'kidney_test1_residuals_by_stage.csv',
        'kidney_test2': 'kidney_test2_acr_patterns.csv',
        
        # CV analysis
        'cv_test1': 'cv_test1_residuals_by_status.csv',
        'cv_test2': 'cv_test2_ushapes.csv',
        
        # Diabetes analysis
        'diabetes_test1': 'diabetes_test1_residuals_by_status.csv',
        'diabetes_test2': 'diabetes_test2_ushapes_prediabetes.csv',
        
        # MetS analysis
        'mets_test1': 'mets_test1_residuals_by_status.csv',
        'mets_test2': 'mets_test2_residuals_by_count.csv',
        
        # Cross-disease analysis
        'cross_correlations': 'cross_disease_correlations.csv',
        'universal_patterns': 'universal_residual_patterns.csv',
        'ushape_consistency': 'ushape_consistency.csv',
    }
    
    for key, filename in result_files.items():
        filepath = RESULTS / filename
        if filepath.exists():
            try:
                results[key] = pd.read_csv(filepath)
                print(f"  ✓ Loaded {filename}")
            except Exception as e:
                print(f"  ✗ Failed to load {filename}: {e}")
        else:
            print(f"  - Not found: {filename}")
    
    return results

def compute_summary_statistics(results: dict) -> dict:
    """Compute summary statistics across all analyses."""
    
    summary = {}
    
    # Count significant findings per disease
    disease_findings = {
        'Liver': {'test1': 'liver_test1', 'test2': 'liver_test2'},
        'Kidney': {'test1': 'kidney_test1', 'test2': 'kidney_test2'},
        'CV': {'test1': 'cv_test1', 'test2': 'cv_test2'},
        'Diabetes': {'test1': 'diabetes_test1', 'test2': 'diabetes_test2'},
        'MetS': {'test1': 'mets_test1', 'test2': 'mets_test2'},
    }
    
    for disease, tests in disease_findings.items():
        summary[disease] = {}
        
        # Test 1: Residual differences
        if tests['test1'] in results:
            df = results[tests['test1']]
            if 'Significant' in df.columns:
                summary[disease]['test1_sig'] = df['Significant'].sum()
                summary[disease]['test1_total'] = len(df)
            elif 'Location_Diff' in df.columns:
                summary[disease]['test1_sig'] = df['Location_Diff'].sum()
                summary[disease]['test1_total'] = len(df)
        
        # Test 2: U-shapes
        if tests['test2'] in results:
            df = results[tests['test2']]
            if 'Significant' in df.columns:
                summary[disease]['test2_sig'] = df['Significant'].sum()
                summary[disease]['test2_total'] = len(df)
                if 'Shape' in df.columns:
                    ushapes = (df['Significant'] & (df['Shape'] == 'U-shape')).sum()
                    summary[disease]['ushapes'] = ushapes
    
    # Cross-disease patterns
    if 'universal_patterns' in results:
        df = results['universal_patterns']
        if 'Universal' in df.columns:
            summary['universal_residuals'] = df['Universal'].sum()
            summary['total_residuals'] = len(df)
    
    if 'cross_correlations' in results:
        df = results['cross_correlations']
        if 'Significant' in df.columns:
            summary['cross_correlations_sig'] = df['Significant'].sum()
            summary['cross_correlations_total'] = len(df)
    
    return summary

def assess_hypothesis_support(summary: dict) -> dict:
    """
    Assess the overall level of support for the Revelatory Division hypothesis.
    
    Criteria:
    ---------
    H1: Residuals show non-random structure (distribution differences)
    H2: U-shaped patterns exist (regime transitions)
    H3: Patterns are consistent across diseases (universality)
    """
    
    assessment = {
        'H1_distribution': {'support': 0, 'evidence': []},
        'H2_ushapes': {'support': 0, 'evidence': []},
        'H3_universality': {'support': 0, 'evidence': []},
    }
    
    # H1: Distribution differences
    total_sig_distributions = 0
    total_tests = 0
    
    for disease in ['Liver', 'Kidney', 'CV', 'Diabetes', 'MetS']:
        if disease in summary:
            if 'test1_sig' in summary[disease]:
                total_sig_distributions += summary[disease]['test1_sig']
                total_tests += summary[disease]['test1_total']
                
                if summary[disease]['test1_sig'] > 0:
                    pct = 100 * summary[disease]['test1_sig'] / summary[disease]['test1_total']
                    assessment['H1_distribution']['evidence'].append(
                        f"{disease}: {summary[disease]['test1_sig']}/{summary[disease]['test1_total']} ({pct:.0f}%)"
                    )
    
    if total_tests > 0:
        pct = 100 * total_sig_distributions / total_tests
        if pct >= 50:
            assessment['H1_distribution']['support'] = 'STRONG'
        elif pct >= 25:
            assessment['H1_distribution']['support'] = 'MODERATE'
        elif pct >= 10:
            assessment['H1_distribution']['support'] = 'WEAK'
        else:
            assessment['H1_distribution']['support'] = 'NOT SUPPORTED'
    
    # H2: U-shapes
    total_ushapes = 0
    total_nonlinear = 0
    
    for disease in ['Liver', 'Kidney', 'CV', 'Diabetes', 'MetS']:
        if disease in summary:
            if 'test2_sig' in summary[disease]:
                total_nonlinear += summary[disease]['test2_sig']
                if 'ushapes' in summary[disease]:
                    total_ushapes += summary[disease]['ushapes']
                    assessment['H2_ushapes']['evidence'].append(
                        f"{disease}: {summary[disease].get('ushapes', 0)} U-shapes"
                    )
    
    if total_nonlinear >= 10:
        assessment['H2_ushapes']['support'] = 'STRONG'
    elif total_nonlinear >= 5:
        assessment['H2_ushapes']['support'] = 'MODERATE'
    elif total_nonlinear >= 2:
        assessment['H2_ushapes']['support'] = 'WEAK'
    else:
        assessment['H2_ushapes']['support'] = 'NOT SUPPORTED'
    
    # H3: Universality
    if 'universal_residuals' in summary and 'total_residuals' in summary:
        n_universal = summary['universal_residuals']
        if n_universal >= 5:
            assessment['H3_universality']['support'] = 'STRONG'
        elif n_universal >= 3:
            assessment['H3_universality']['support'] = 'MODERATE'
        elif n_universal >= 1:
            assessment['H3_universality']['support'] = 'WEAK'
        else:
            assessment['H3_universality']['support'] = 'NOT SUPPORTED'
        
        assessment['H3_universality']['evidence'].append(
            f"{n_universal} residuals significant across ≥3 diseases"
        )
    
    if 'cross_correlations_sig' in summary:
        assessment['H3_universality']['evidence'].append(
            f"{summary['cross_correlations_sig']} significant cross-disease correlations"
        )
    
    # Overall assessment
    support_levels = [
        assessment['H1_distribution']['support'],
        assessment['H2_ushapes']['support'],
        assessment['H3_universality']['support']
    ]
    
    strong_count = support_levels.count('STRONG')
    moderate_count = support_levels.count('MODERATE')
    
    if strong_count >= 2:
        assessment['overall'] = 'STRONG SUPPORT for Revelatory Division'
    elif strong_count >= 1 or moderate_count >= 2:
        assessment['overall'] = 'MODERATE SUPPORT for Revelatory Division'
    elif moderate_count >= 1:
        assessment['overall'] = 'WEAK/PARTIAL SUPPORT for Revelatory Division'
    else:
        assessment['overall'] = 'INSUFFICIENT SUPPORT for Revelatory Division'
    
    return assessment

def generate_final_report(results: dict, summary: dict, assessment: dict):
    """Generate the final comprehensive report."""
    
    report = []
    
    # Header
    report.append("=" * 80)
    report.append("PAPER 2 EXTENDED: FINAL SUMMARY REPORT")
    report.append("The Revelatory Division: Multi-Disease Validation Using NHANES")
    report.append("=" * 80)
    report.append("")
    report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("Author: Jonathan Édouard Slama")
    report.append("Affiliation: Metafund Research Division, Strasbourg, France")
    report.append("Email: jonathan@metafund.in")
    report.append("ORCID: 0009-0002-1292-4350")
    report.append("")
    
    # Executive Summary
    report.append("=" * 80)
    report.append("EXECUTIVE SUMMARY")
    report.append("=" * 80)
    report.append("")
    report.append(f"Overall Assessment: {assessment['overall']}")
    report.append("")
    report.append("The Revelatory Division hypothesis posits that residuals from")
    report.append("clinical biomarker ratios contain diagnostic information that")
    report.append("becomes visible through non-linear (U-shaped) patterns, particularly")
    report.append("in diagnostic 'gray zones' where ratios alone are ambiguous.")
    report.append("")
    
    # Hypothesis Testing Results
    report.append("=" * 80)
    report.append("HYPOTHESIS TESTING SUMMARY")
    report.append("=" * 80)
    report.append("")
    
    for h_key, h_name in [('H1_distribution', 'H1: Residuals show non-random structure'),
                          ('H2_ushapes', 'H2: U-shaped patterns exist'),
                          ('H3_universality', 'H3: Patterns are universal across diseases')]:
        h_data = assessment[h_key]
        report.append(f"{h_name}")
        report.append(f"   Support Level: {h_data['support']}")
        report.append("   Evidence:")
        for ev in h_data['evidence']:
            report.append(f"     - {ev}")
        report.append("")
    
    # Disease-Specific Results
    report.append("=" * 80)
    report.append("DISEASE-SPECIFIC RESULTS")
    report.append("=" * 80)
    report.append("")
    
    disease_names = {
        'Liver': 'Hepatic Fibrosis (FIB-4, APRI, NFS)',
        'Kidney': 'Chronic Kidney Disease (eGFR, ACR)',
        'CV': 'Cardiovascular Risk (TG/HDL, TC/HDL)',
        'Diabetes': 'Diabetes/Prediabetes (HOMA-IR, TyG)',
        'MetS': 'Metabolic Syndrome (ATP III criteria)'
    }
    
    for disease, full_name in disease_names.items():
        report.append(f"--- {full_name} ---")
        
        if disease in summary:
            if 'test1_sig' in summary[disease]:
                report.append(f"   Residual distribution differences: "
                            f"{summary[disease]['test1_sig']}/{summary[disease]['test1_total']}")
            if 'test2_sig' in summary[disease]:
                report.append(f"   Non-linear patterns: "
                            f"{summary[disease]['test2_sig']}/{summary[disease]['test2_total']}")
            if 'ushapes' in summary[disease]:
                report.append(f"   U-shapes detected: {summary[disease]['ushapes']}")
        else:
            report.append("   [Analysis not completed or no results available]")
        
        report.append("")
    
    # Cross-Disease Patterns
    report.append("=" * 80)
    report.append("CROSS-DISEASE PATTERNS")
    report.append("=" * 80)
    report.append("")
    
    if 'universal_residuals' in summary:
        report.append(f"Universal residuals (≥3 diseases): {summary['universal_residuals']}")
    if 'cross_correlations_sig' in summary:
        report.append(f"Significant cross-disease correlations: "
                     f"{summary['cross_correlations_sig']}/{summary.get('cross_correlations_total', '?')}")
    report.append("")
    
    # Methodological Notes
    report.append("=" * 80)
    report.append("METHODOLOGICAL NOTES")
    report.append("=" * 80)
    report.append("")
    report.append("Data Source: NHANES 2017-2018 / 2017-March 2020 (pre-pandemic)")
    report.append("Analysis Framework: The Revelatory Division (QO+R inspired)")
    report.append("")
    report.append("Statistical Tests:")
    report.append("   - Distribution differences: Mann-Whitney U, Kolmogorov-Smirnov")
    report.append("   - Non-linearity: F-test for quadratic vs. linear models")
    report.append("   - Classification: Logistic regression with 5-fold CV, AUC-ROC")
    report.append("   - Correlations: Spearman rank correlation")
    report.append("")
    report.append("Multiple Testing: Bonferroni and FDR corrections applied where noted")
    report.append("")
    
    # Limitations
    report.append("=" * 80)
    report.append("LIMITATIONS")
    report.append("=" * 80)
    report.append("")
    report.append("1. Cross-sectional design: Cannot establish temporal sequence")
    report.append("2. Self-reported outcomes: Some disease status from questionnaires")
    report.append("3. Proxy outcomes: No biopsy/imaging gold standards available")
    report.append("4. U.S. population: May not generalize globally")
    report.append("5. Single dataset: Requires external validation")
    report.append("6. Exploratory analysis: Multiple comparisons increase false positive risk")
    report.append("")
    
    # Conclusions
    report.append("=" * 80)
    report.append("CONCLUSIONS")
    report.append("=" * 80)
    report.append("")
    report.append("This analysis tested the Revelatory Division hypothesis across five")
    report.append("disease categories using NHANES data. The results indicate:")
    report.append("")
    
    # Determine conclusion based on assessment
    if 'STRONG' in assessment['overall']:
        report.append("✓ STRONG SUPPORT: Residuals contain structured diagnostic information")
        report.append("  across multiple disease categories, with consistent U-shaped patterns")
        report.append("  suggesting underlying mechanistic significance.")
    elif 'MODERATE' in assessment['overall']:
        report.append("✓ MODERATE SUPPORT: Residuals show non-random patterns in several")
        report.append("  disease categories, though evidence for U-shapes and universality")
        report.append("  is mixed. Further investigation warranted.")
    elif 'WEAK' in assessment['overall'] or 'PARTIAL' in assessment['overall']:
        report.append("~ PARTIAL SUPPORT: Some residuals show significant patterns, but")
        report.append("  findings are not consistent across diseases. The hypothesis")
        report.append("  may apply to specific contexts rather than universally.")
    else:
        report.append("✗ INSUFFICIENT SUPPORT: The analysis did not find strong evidence")
        report.append("  for the Revelatory Division hypothesis in this dataset.")
        report.append("  Residuals may not contain clinically useful information beyond")
        report.append("  what is captured by the original ratios.")
    
    report.append("")
    report.append("Regardless of hypothesis support level, this work demonstrates:")
    report.append("   - A systematic methodology for residual analysis")
    report.append("   - Testable predictions from the QO+R framework")
    report.append("   - Potential for cross-domain scientific inquiry")
    report.append("")
    
    # Future Directions
    report.append("=" * 80)
    report.append("FUTURE DIRECTIONS")
    report.append("=" * 80)
    report.append("")
    report.append("1. External validation: UK Biobank, Framingham Heart Study")
    report.append("2. Longitudinal analysis: NHANES mortality linkage")
    report.append("3. Mechanistic investigation: Identify biological correlates")
    report.append("4. Clinical utility: Develop residual-augmented scores")
    report.append("5. Theoretical refinement: Formalize QO+R biomedical framework")
    report.append("")
    
    # Footer
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    report.append("")
    report.append("This report was generated as part of the Paper 2 Extended analysis.")
    report.append("All data and code are available for reproducibility.")
    report.append("")
    report.append("Contact: jonathan@metafund.in")
    report.append("")
    
    # Save report
    report_text = "\n".join(report)
    
    # Save to results
    with open(RESULTS / "FINAL_SUMMARY_REPORT.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # Save to manuscript folder
    with open(MANUSCRIPT / "FINAL_SUMMARY_REPORT.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    
    return report_text

def generate_summary_table(results: dict, summary: dict):
    """Generate a summary table for manuscript."""
    
    rows = []
    
    diseases = ['Liver', 'Kidney', 'CV', 'Diabetes', 'MetS']
    
    for disease in diseases:
        if disease in summary:
            d = summary[disease]
            rows.append({
                'Disease Category': disease,
                'Distribution Differences': f"{d.get('test1_sig', 0)}/{d.get('test1_total', 0)}",
                'Non-linear Patterns': f"{d.get('test2_sig', 0)}/{d.get('test2_total', 0)}",
                'U-Shapes': d.get('ushapes', 0)
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(MANUSCRIPT / "Table1_summary_by_disease.csv", index=False)
    
    print("\n✓ Summary table saved to manuscript folder")
    
    return df

def generate_summary_figure(summary: dict, assessment: dict):
    """Generate a summary figure for manuscript."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Support by hypothesis
    ax = axes[0]
    
    hypotheses = ['H1\nDistribution', 'H2\nU-Shapes', 'H3\nUniversality']
    support_scores = []
    
    for h_key in ['H1_distribution', 'H2_ushapes', 'H3_universality']:
        support = assessment[h_key]['support']
        if support == 'STRONG':
            support_scores.append(3)
        elif support == 'MODERATE':
            support_scores.append(2)
        elif support == 'WEAK':
            support_scores.append(1)
        else:
            support_scores.append(0)
    
    colors = ['#e74c3c' if s == 0 else '#f39c12' if s == 1 else '#27ae60' if s == 2 else '#2ecc71' 
              for s in support_scores]
    
    bars = ax.bar(hypotheses, support_scores, color=colors, edgecolor='black')
    ax.set_ylabel('Support Level')
    ax.set_ylim(0, 3.5)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Not\nSupported', 'Weak', 'Moderate', 'Strong'])
    ax.set_title('Hypothesis Support Summary')
    ax.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='Moderate threshold')
    
    # Panel B: Findings by disease
    ax = axes[1]
    
    diseases = []
    test1_vals = []
    test2_vals = []
    
    for disease in ['Liver', 'Kidney', 'CV', 'Diabetes', 'MetS']:
        if disease in summary:
            diseases.append(disease)
            d = summary[disease]
            t1 = d.get('test1_total', 1)
            test1_vals.append(d.get('test1_sig', 0) / t1 * 100 if t1 > 0 else 0)
            t2 = d.get('test2_total', 1)
            test2_vals.append(d.get('test2_sig', 0) / t2 * 100 if t2 > 0 else 0)
    
    x = np.arange(len(diseases))
    width = 0.35
    
    ax.bar(x - width/2, test1_vals, width, label='Distribution Diff.', color='#3498db')
    ax.bar(x + width/2, test2_vals, width, label='Non-linear', color='#e74c3c')
    
    ax.set_ylabel('% Significant')
    ax.set_xticks(x)
    ax.set_xticklabels(diseases)
    ax.set_title('Significant Findings by Disease')
    ax.legend()
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(MANUSCRIPT / "Figure_summary.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES / "summary_figure.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Summary figure saved")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    
    print("=" * 80)
    print("PAPER 2 EXTENDED: FINAL SUMMARY REPORT GENERATOR")
    print("The Revelatory Division - Multi-Disease Validation")
    print("=" * 80)
    print()
    print("Author: Jonathan Édouard Slama")
    print("Affiliation: Metafund Research Division, Strasbourg, France")
    print()
    
    # Load results
    results = load_all_results()
    
    if not results:
        print("\n✗ No results found. Please run the analysis scripts first:")
        print("   python scripts/00_download_nhanes.py")
        print("   python scripts/01_merge_nhanes.py")
        print("   python scripts/02_compute_all_ratios.py")
        print("   python scripts/03_hepatic_fibrosis.py")
        print("   python scripts/04_kidney_disease.py")
        print("   python scripts/05_cardiovascular.py")
        print("   python scripts/06_diabetes.py")
        print("   python scripts/07_metabolic_syndrome.py")
        print("   python scripts/08_cross_disease.py")
        return
    
    # Compute summary statistics
    print("\nComputing summary statistics...")
    summary = compute_summary_statistics(results)
    
    # Assess hypothesis support
    print("Assessing hypothesis support...")
    assessment = assess_hypothesis_support(summary)
    
    # Generate outputs
    print("\nGenerating final report...")
    generate_final_report(results, summary, assessment)
    
    print("\nGenerating summary table...")
    generate_summary_table(results, summary)
    
    print("\nGenerating summary figure...")
    generate_summary_figure(summary, assessment)
    
    print()
    print("=" * 80)
    print("✓ FINAL SUMMARY COMPLETE!")
    print("=" * 80)
    print()
    print(f"Final report: {RESULTS / 'FINAL_SUMMARY_REPORT.txt'}")
    print(f"Summary table: {MANUSCRIPT / 'Table1_summary_by_disease.csv'}")
    print(f"Summary figure: {MANUSCRIPT / 'Figure_summary.png'}")
    print()
    print("The analysis pipeline is complete. Next steps:")
    print("  1. Review the final report")
    print("  2. Prepare manuscript draft")
    print("  3. Consider external validation datasets")
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
