#!/usr/bin/env python3
"""
Paper 2 Extended: Compute All Clinical Ratios
The Revelatory Division - Multi-Disease Validation

Computes biomarker ratios for:
1. Hepatic Fibrosis (FIB-4, APRI, NFS)
2. Chronic Kidney Disease (eGFR already computed, ACR, BUN/Cr)
3. Cardiovascular Risk (TG/HDL, TC/HDL, LDL/HDL)
4. Diabetes (HOMA-IR, TyG, QUICKI)
5. Metabolic Syndrome (component count)

Author: Jonathan Édouard Slama
Email: jonathan@metafund.in
Affiliation: Metafund Research Division, Strasbourg, France
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
PROCESSED_DATA = PROJECT_DIR / "data" / "processed"
RESULTS = PROJECT_DIR / "results"

# Upper limits of normal for liver enzymes (for APRI calculation)
AST_ULN = 40  # U/L
ALT_ULN = 40  # U/L

def load_data() -> pd.DataFrame:
    """Load merged NHANES data."""
    filepath = PROCESSED_DATA / "nhanes_merged.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Merged data not found: {filepath}\nRun 01_merge_nhanes.py first")
    return pd.read_csv(filepath)

# =============================================================================
# HEPATIC FIBROSIS RATIOS
# =============================================================================

def compute_liver_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute liver fibrosis ratios."""
    
    print("\n--- HEPATIC FIBROSIS RATIOS ---")
    
    # FIB-4 = (Age × AST) / (Platelets × √ALT)
    if all(v in df.columns for v in ['Age', 'AST', 'Platelets', 'ALT']):
        df['FIB4'] = (df['Age'] * df['AST']) / (df['Platelets'] * np.sqrt(df['ALT']))
        n_valid = df['FIB4'].notna().sum()
        print(f"  FIB-4: computed for {n_valid} participants")
        print(f"    Mean: {df['FIB4'].mean():.2f}, Median: {df['FIB4'].median():.2f}")
        
        # FIB-4 categories
        df['FIB4_Category'] = pd.cut(df['FIB4'],
                                     bins=[0, 1.3, 2.67, 100],
                                     labels=['Low Risk', 'Indeterminate', 'High Risk'])
    
    # APRI = (AST / ULN) × 100 / Platelets
    if all(v in df.columns for v in ['AST', 'Platelets']):
        df['APRI'] = ((df['AST'] / AST_ULN) * 100) / df['Platelets']
        n_valid = df['APRI'].notna().sum()
        print(f"  APRI: computed for {n_valid} participants")
        print(f"    Mean: {df['APRI'].mean():.2f}, Median: {df['APRI'].median():.2f}")
        
        # APRI categories
        df['APRI_Category'] = pd.cut(df['APRI'],
                                     bins=[0, 0.5, 1.5, 100],
                                     labels=['No Fibrosis', 'Indeterminate', 'Significant Fibrosis'])
    
    # AST/ALT Ratio (De Ritis ratio)
    if all(v in df.columns for v in ['AST', 'ALT']):
        df['DeRitis'] = df['AST'] / df['ALT']
        n_valid = df['DeRitis'].notna().sum()
        print(f"  De Ritis (AST/ALT): computed for {n_valid} participants")
    
    # NFS = -1.675 + 0.037×Age + 0.094×BMI + 1.13×IFG + 0.99×AST/ALT - 0.013×Plt - 0.66×Alb
    if all(v in df.columns for v in ['Age', 'BMI', 'AST', 'ALT', 'Platelets', 'Albumin']):
        # IFG (Impaired Fasting Glucose) = Glucose >= 100 or Diabetes
        if 'Glucose' in df.columns:
            df['IFG'] = ((df['Glucose'] >= 100) | (df.get('HasDiabetes', 0) == 1)).astype(int)
        else:
            df['IFG'] = df.get('HasDiabetes', 0)
        
        df['NFS'] = (-1.675 + 
                     0.037 * df['Age'] + 
                     0.094 * df['BMI'] + 
                     1.13 * df['IFG'] + 
                     0.99 * df['DeRitis'] - 
                     0.013 * df['Platelets'] - 
                     0.66 * df['Albumin'])
        n_valid = df['NFS'].notna().sum()
        print(f"  NFS: computed for {n_valid} participants")
        
        # NFS categories
        df['NFS_Category'] = pd.cut(df['NFS'],
                                    bins=[-100, -1.455, 0.676, 100],
                                    labels=['No Fibrosis', 'Indeterminate', 'Advanced Fibrosis'])
    
    # GGT/ALT ratio
    if all(v in df.columns for v in ['GGT', 'ALT']):
        df['GGT_ALT'] = df['GGT'] / df['ALT']
        print(f"  GGT/ALT: computed")
    
    return df

# =============================================================================
# KIDNEY DISEASE RATIOS
# =============================================================================

def compute_kidney_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute kidney disease ratios."""
    
    print("\n--- CHRONIC KIDNEY DISEASE RATIOS ---")
    
    # eGFR should already be computed, but verify
    if 'eGFR' in df.columns:
        n_valid = df['eGFR'].notna().sum()
        print(f"  eGFR: {n_valid} participants (pre-computed)")
    
    # ACR should already be computed
    if 'ACR' in df.columns:
        n_valid = df['ACR'].notna().sum()
        print(f"  ACR: {n_valid} participants (pre-computed)")
    
    # BUN/Creatinine ratio
    if all(v in df.columns for v in ['BUN', 'Creatinine']):
        df['BUN_Cr'] = df['BUN'] / df['Creatinine']
        n_valid = df['BUN_Cr'].notna().sum()
        print(f"  BUN/Creatinine: computed for {n_valid} participants")
        print(f"    Mean: {df['BUN_Cr'].mean():.1f}, Normal range: 10-20")
    
    # Albumin/Globulin ratio (if total protein available)
    # Globulin = Total Protein - Albumin
    # Note: NHANES may not have total protein directly
    
    # CKD composite
    if all(v in df.columns for v in ['eGFR', 'ACR']):
        # CKD defined as eGFR < 60 OR ACR > 30
        df['HasCKD'] = ((df['eGFR'] < 60) | (df['ACR'] > 30)).astype(int)
        ckd_prev = 100 * df['HasCKD'].mean()
        print(f"  CKD prevalence: {ckd_prev:.1f}%")
    
    return df

# =============================================================================
# CARDIOVASCULAR RATIOS
# =============================================================================

def compute_cv_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cardiovascular risk ratios."""
    
    print("\n--- CARDIOVASCULAR RATIOS ---")
    
    # TG/HDL ratio (insulin resistance proxy)
    if all(v in df.columns for v in ['Triglycerides', 'HDL']):
        df['TG_HDL'] = df['Triglycerides'] / df['HDL']
        n_valid = df['TG_HDL'].notna().sum()
        print(f"  TG/HDL: computed for {n_valid} participants")
        print(f"    Mean: {df['TG_HDL'].mean():.2f} (>3.5 suggests IR)")
        
        # High risk threshold
        df['TG_HDL_High'] = (df['TG_HDL'] > 3.5).astype(int)
    
    # TC/HDL ratio (atherogenic index)
    if all(v in df.columns for v in ['TotalCholesterol', 'HDL']):
        df['TC_HDL'] = df['TotalCholesterol'] / df['HDL']
        n_valid = df['TC_HDL'].notna().sum()
        print(f"  TC/HDL: computed for {n_valid} participants")
        print(f"    Mean: {df['TC_HDL'].mean():.2f} (<4.5 optimal)")
    
    # LDL/HDL ratio
    if all(v in df.columns for v in ['LDL', 'HDL']):
        df['LDL_HDL'] = df['LDL'] / df['HDL']
        n_valid = df['LDL_HDL'].notna().sum()
        print(f"  LDL/HDL: computed for {n_valid} participants")
        print(f"    Mean: {df['LDL_HDL'].mean():.2f} (<2.5 optimal)")
    
    # Non-HDL cholesterol
    if all(v in df.columns for v in ['TotalCholesterol', 'HDL']):
        df['NonHDL'] = df['TotalCholesterol'] - df['HDL']
        print(f"  Non-HDL: computed")
    
    # TyG index (for CV risk, not just diabetes)
    if all(v in df.columns for v in ['Triglycerides', 'Glucose']):
        df['TyG'] = np.log(df['Triglycerides'] * df['Glucose'] / 2)
        n_valid = df['TyG'].notna().sum()
        print(f"  TyG index: computed for {n_valid} participants")
        print(f"    Mean: {df['TyG'].mean():.2f}")
    
    return df

# =============================================================================
# DIABETES RATIOS
# =============================================================================

def compute_diabetes_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute diabetes/insulin resistance ratios."""
    
    print("\n--- DIABETES/INSULIN RESISTANCE RATIOS ---")
    
    # HOMA-IR = (Glucose × Insulin) / 405
    if all(v in df.columns for v in ['Glucose', 'Insulin']):
        df['HOMA_IR'] = (df['Glucose'] * df['Insulin']) / 405
        n_valid = df['HOMA_IR'].notna().sum()
        print(f"  HOMA-IR: computed for {n_valid} participants")
        print(f"    Mean: {df['HOMA_IR'].mean():.2f}, Median: {df['HOMA_IR'].median():.2f}")
        print(f"    (>2.5 suggests insulin resistance)")
        
        # HOMA-IR categories
        df['HOMA_Category'] = pd.cut(df['HOMA_IR'],
                                     bins=[0, 1.0, 2.5, 100],
                                     labels=['Normal', 'Borderline', 'Insulin Resistant'])
    
    # HOMA-B (Beta cell function)
    if all(v in df.columns for v in ['Glucose', 'Insulin']):
        df['HOMA_B'] = (360 * df['Insulin']) / (df['Glucose'] - 63)
        # Avoid division issues
        df.loc[df['Glucose'] <= 63, 'HOMA_B'] = np.nan
        print(f"  HOMA-B: computed")
    
    # QUICKI = 1 / (log(Insulin) + log(Glucose))
    if all(v in df.columns for v in ['Glucose', 'Insulin']):
        df['QUICKI'] = 1 / (np.log10(df['Insulin']) + np.log10(df['Glucose']))
        n_valid = df['QUICKI'].notna().sum()
        print(f"  QUICKI: computed for {n_valid} participants")
        print(f"    Mean: {df['QUICKI'].mean():.3f} (<0.339 suggests IR)")
    
    # Glucose/Insulin ratio
    if all(v in df.columns for v in ['Glucose', 'Insulin']):
        df['G_I'] = df['Glucose'] / df['Insulin']
        print(f"  Glucose/Insulin: computed")
    
    # TyG already computed in CV section
    
    # Matsuda Index (if we had OGTT data - we don't in NHANES fasting)
    # Skipping Matsuda
    
    # McAuley Index = exp(2.63 - 0.28 × ln(Insulin) - 0.31 × ln(TG))
    if all(v in df.columns for v in ['Insulin', 'Triglycerides']):
        df['McAuley'] = np.exp(2.63 - 0.28 * np.log(df['Insulin']) - 0.31 * np.log(df['Triglycerides']))
        print(f"  McAuley Index: computed")
    
    return df

# =============================================================================
# METABOLIC SYNDROME
# =============================================================================

def compute_mets_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute metabolic syndrome components and ratios."""
    
    print("\n--- METABOLIC SYNDROME ---")
    
    # ATP III Criteria (modified)
    # 1. Waist: >102 cm (M) or >88 cm (F)
    # 2. TG: ≥150 mg/dL
    # 3. HDL: <40 (M) or <50 (F)
    # 4. BP: ≥130/85 mmHg
    # 5. Glucose: ≥100 mg/dL
    
    mets_components = []
    
    # Waist criterion
    if all(v in df.columns for v in ['Waist', 'Sex']):
        df['MetS_Waist'] = ((df['Sex'] == 1) & (df['Waist'] > 102)) | \
                           ((df['Sex'] == 0) & (df['Waist'] > 88))
        df['MetS_Waist'] = df['MetS_Waist'].astype(int)
        mets_components.append('MetS_Waist')
        print(f"  Waist criterion: {100*df['MetS_Waist'].mean():.1f}% meet")
    
    # TG criterion
    if 'Triglycerides' in df.columns:
        df['MetS_TG'] = (df['Triglycerides'] >= 150).astype(int)
        mets_components.append('MetS_TG')
        print(f"  TG criterion: {100*df['MetS_TG'].mean():.1f}% meet")
    
    # HDL criterion
    if all(v in df.columns for v in ['HDL', 'Sex']):
        df['MetS_HDL'] = ((df['Sex'] == 1) & (df['HDL'] < 40)) | \
                         ((df['Sex'] == 0) & (df['HDL'] < 50))
        df['MetS_HDL'] = df['MetS_HDL'].astype(int)
        mets_components.append('MetS_HDL')
        print(f"  HDL criterion: {100*df['MetS_HDL'].mean():.1f}% meet")
    
    # BP criterion
    if all(v in df.columns for v in ['SystolicBP', 'DiastolicBP']):
        df['MetS_BP'] = ((df['SystolicBP'] >= 130) | (df['DiastolicBP'] >= 85)).astype(int)
        mets_components.append('MetS_BP')
        print(f"  BP criterion: {100*df['MetS_BP'].mean():.1f}% meet")
    
    # Glucose criterion
    if 'Glucose' in df.columns:
        df['MetS_Glucose'] = (df['Glucose'] >= 100).astype(int)
        mets_components.append('MetS_Glucose')
        print(f"  Glucose criterion: {100*df['MetS_Glucose'].mean():.1f}% meet")
    
    # Count components
    if mets_components:
        df['MetS_Count'] = df[mets_components].sum(axis=1)
        df['HasMetS'] = (df['MetS_Count'] >= 3).astype(int)
        mets_prev = 100 * df['HasMetS'].mean()
        print(f"\n  Metabolic Syndrome prevalence: {mets_prev:.1f}%")
        print(f"  Component distribution: {df['MetS_Count'].value_counts().sort_index().to_dict()}")
    
    # Waist/Height ratio
    if all(v in df.columns for v in ['Waist', 'Height']):
        df['WHtR'] = df['Waist'] / df['Height']
        print(f"  Waist/Height ratio: mean = {df['WHtR'].mean():.2f}")
    
    # Visceral Adiposity Index (VAI) - simplified
    # VAI = (Waist / 39.68 + 1.88×BMI) × (TG/1.03) × (1.31/HDL) for men
    # Different formula for women
    if all(v in df.columns for v in ['Waist', 'BMI', 'Triglycerides', 'HDL', 'Sex']):
        # Male formula
        vai_m = (df['Waist'] / (39.68 + 1.88 * df['BMI'])) * (df['Triglycerides'] / 1.03) * (1.31 / df['HDL'])
        # Female formula  
        vai_f = (df['Waist'] / (36.58 + 1.89 * df['BMI'])) * (df['Triglycerides'] / 0.81) * (1.52 / df['HDL'])
        
        df['VAI'] = np.where(df['Sex'] == 1, vai_m, vai_f)
        print(f"  VAI: computed")
    
    return df

# =============================================================================
# COMPUTE RESIDUALS
# =============================================================================

def compute_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute residuals for all ratios."""
    
    print("\n" + "=" * 60)
    print("COMPUTING RESIDUALS (RELIQUATS)")
    print("=" * 60)
    
    from sklearn.linear_model import LinearRegression
    
    # Define ratios and their covariates
    ratio_configs = {
        # Liver ratios
        'FIB4': ['Age', 'Sex', 'BMI'],
        'APRI': ['Age', 'Sex', 'BMI'],
        'NFS': ['Age', 'Sex'],  # BMI already in formula
        'DeRitis': ['Age', 'Sex', 'BMI'],
        
        # Kidney ratios
        'eGFR': ['Sex', 'BMI'],  # Age already in formula
        'ACR': ['Age', 'Sex', 'BMI'],
        'BUN_Cr': ['Age', 'Sex', 'BMI'],
        
        # CV ratios
        'TG_HDL': ['Age', 'Sex', 'BMI'],
        'TC_HDL': ['Age', 'Sex', 'BMI'],
        'LDL_HDL': ['Age', 'Sex', 'BMI'],
        'TyG': ['Age', 'Sex', 'BMI'],
        
        # Diabetes ratios
        'HOMA_IR': ['Age', 'Sex', 'BMI'],
        'QUICKI': ['Age', 'Sex', 'BMI'],
        'G_I': ['Age', 'Sex', 'BMI'],
        
        # MetS
        'MetS_Count': ['Age', 'Sex'],
        'WHtR': ['Age', 'Sex'],
        'VAI': ['Age', 'Sex', 'BMI'],
    }
    
    residual_stats = []
    
    for ratio, covariates in ratio_configs.items():
        if ratio not in df.columns:
            continue
        
        # Check covariates exist
        covariates = [c for c in covariates if c in df.columns]
        if not covariates:
            continue
        
        # Get complete cases
        vars_needed = [ratio] + covariates
        mask = df[vars_needed].notna().all(axis=1)
        n_valid = mask.sum()
        
        if n_valid < 100:
            print(f"  {ratio}: skipped (only {n_valid} complete cases)")
            continue
        
        # Fit regression
        X = df.loc[mask, covariates].values
        y = df.loc[mask, ratio].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Compute residuals
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Store residuals
        resid_col = f"{ratio}_Residual"
        df[resid_col] = np.nan
        df.loc[mask, resid_col] = residuals
        
        # Statistics
        r2 = model.score(X, y)
        resid_std = np.std(residuals)
        
        residual_stats.append({
            'Ratio': ratio,
            'Covariates': '+'.join(covariates),
            'N': n_valid,
            'R2': r2,
            'Residual_Std': resid_std
        })
        
        print(f"  {ratio} ~ {'+'.join(covariates)}: R²={r2:.3f}, N={n_valid}")
    
    # Save residual statistics
    stats_df = pd.DataFrame(residual_stats)
    stats_df.to_csv(RESULTS / "residual_computation_stats.csv", index=False)
    
    return df

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""
    
    print("=" * 60)
    print("PAPER 2 EXTENDED: COMPUTE ALL CLINICAL RATIOS")
    print("The Revelatory Division - Multi-Disease Validation")
    print("=" * 60)
    
    # Create results directory
    RESULTS.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading merged NHANES data...")
    df = load_data()
    print(f"✓ Loaded {len(df)} participants, {len(df.columns)} variables")
    
    # Compute all ratios
    df = compute_liver_ratios(df)
    df = compute_kidney_ratios(df)
    df = compute_cv_ratios(df)
    df = compute_diabetes_ratios(df)
    df = compute_mets_ratios(df)
    
    # Compute residuals
    df = compute_residuals(df)
    
    # Summary of new columns
    print("\n" + "=" * 60)
    print("SUMMARY OF COMPUTED RATIOS")
    print("=" * 60)
    
    ratio_cols = [c for c in df.columns if any(c.startswith(p) for p in 
                  ['FIB4', 'APRI', 'NFS', 'DeRitis', 'GGT_ALT',
                   'BUN_Cr', 'TG_HDL', 'TC_HDL', 'LDL_HDL', 'NonHDL', 'TyG',
                   'HOMA', 'QUICKI', 'G_I', 'McAuley',
                   'MetS', 'WHtR', 'VAI'])]
    
    print(f"\nTotal new ratio columns: {len(ratio_cols)}")
    
    # Count residual columns
    resid_cols = [c for c in df.columns if c.endswith('_Residual')]
    print(f"Residual columns: {len(resid_cols)}")
    
    # Save
    output_file = PROCESSED_DATA / "nhanes_with_ratios.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved data with ratios to: {output_file}")
    print(f"  Final dimensions: {len(df)} rows × {len(df.columns)} columns")
    
    # Save ratio summary
    ratio_summary = []
    for col in ratio_cols:
        if col.endswith('_Residual') or col.endswith('_Category'):
            continue
        if df[col].dtype in ['object', 'category']:
            continue
        ratio_summary.append({
            'Ratio': col,
            'N_Valid': df[col].notna().sum(),
            'Mean': df[col].mean(),
            'Std': df[col].std(),
            'Median': df[col].median(),
            'Min': df[col].min(),
            'Max': df[col].max()
        })
    
    summary_df = pd.DataFrame(ratio_summary)
    summary_df.to_csv(RESULTS / "ratio_summary.csv", index=False)
    print(f"✓ Saved ratio summary to: {RESULTS / 'ratio_summary.csv'}")
    
    print()
    print("=" * 60)
    print("✓ Ratio computation complete!")
    print()
    print("Next steps:")
    print("  python scripts/03_hepatic_fibrosis.py")
    print("  python scripts/04_kidney_disease.py")
    print("  python scripts/05_cardiovascular.py")
    print("  python scripts/06_diabetes.py")
    print("  python scripts/07_metabolic_syndrome.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
