#!/usr/bin/env python3
"""
Paper 2 Extended: NHANES Data Merge Script
The Revelatory Division - Multi-Disease Validation

Merges all downloaded NHANES files into a single analysis dataset.

Author: Jonathan Édouard Slama
Email: jonathan@metafund.in
Affiliation: Metafund Research Division, Strasbourg, France
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pyreadstat
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_DATA = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA = PROJECT_DIR / "data" / "processed"

# Variable mappings (standardize names across cycles)
VARIABLE_RENAMES = {
    # Demographics
    'RIDAGEYR': 'Age',
    'RIAGENDR': 'Sex',  # 1=Male, 2=Female
    'RIDRETH3': 'Race',
    'DMDEDUC2': 'Education',
    'INDFMPIR': 'PovertyRatio',
    
    # Liver enzymes
    'LBXSATSI': 'AST',
    'LBXSASSI': 'ALT', 
    'LBXSGTSI': 'GGT',
    'LBXSTB': 'Bilirubin',
    'LBXSAPSI': 'ALP',
    'LBXSAL': 'Albumin',
    
    # Kidney function
    'LBXSCR': 'Creatinine',
    'LBXSBU': 'BUN',
    'URXUMA': 'UrineAlbumin',
    'URXUCR': 'UrineCreatinine',
    
    # Complete blood count
    'LBXPLTSI': 'Platelets',
    'LBXWBCSI': 'WBC',
    'LBXRBCSI': 'RBC',
    'LBXHGB': 'Hemoglobin',
    
    # Lipids
    'LBXTC': 'TotalCholesterol',
    'LBDHDD': 'HDL',
    'LBXTR': 'Triglycerides',
    'LBDLDL': 'LDL',
    
    # Glucose/Insulin
    'LBXGLU': 'Glucose',
    'LBXIN': 'Insulin',
    'LBXGH': 'HbA1c',
    
    # Body measures
    'BMXBMI': 'BMI',
    'BMXWAIST': 'Waist',
    'BMXHT': 'Height',
    'BMXWT': 'Weight',
    
    # Blood pressure (use first reading)
    'BPXSY1': 'SystolicBP',
    'BPXDI1': 'DiastolicBP',
    
    # Questionnaire - Diabetes
    'DIQ010': 'DiabetesDiagnosis',  # 1=Yes, 2=No, 3=Borderline
    'DIQ050': 'TakingInsulin',
    'DIQ070': 'TakingDiabetesPills',
    
    # Questionnaire - Medical conditions
    'MCQ160F': 'HeartFailure',
    'MCQ160B': 'CoronaryHeartDisease',
    'MCQ160C': 'HeartAttack',
    'MCQ160E': 'Stroke',
    'MCQ220': 'Cancer',
    
    # Questionnaire - Kidney
    'KIQ022': 'WeakKidneys',
    'KIQ025': 'Dialysis',
    
    # Smoking
    'SMQ020': 'Smoked100Cigarettes',
    'SMQ040': 'CurrentSmoker',
    
    # Alcohol
    'ALQ130': 'DrinksPerDay',
}

def load_xpt(filepath: Path) -> pd.DataFrame:
    """Load SAS XPT file into DataFrame."""
    try:
        df, meta = pyreadstat.read_xport(str(filepath))
        return df
    except Exception as e:
        print(f"  Warning: Could not load {filepath.name}: {e}")
        return pd.DataFrame()

def merge_nhanes_files(raw_dir: Path, use_prepandemic: bool = True) -> pd.DataFrame:
    """Merge all NHANES files into single DataFrame."""
    
    # Determine which files to use
    if use_prepandemic:
        prefix = "P_"
        fallback_suffix = "_J"
    else:
        prefix = ""
        fallback_suffix = "_J"
    
    # Start with demographics (required)
    demo_file = raw_dir / f"{prefix}DEMO.XPT"
    if not demo_file.exists():
        demo_file = raw_dir / f"DEMO{fallback_suffix}.XPT"
    
    if not demo_file.exists():
        raise FileNotFoundError(f"Demographics file not found in {raw_dir}")
    
    print(f"Loading demographics: {demo_file.name}")
    df = load_xpt(demo_file)
    print(f"  Initial sample: {len(df)} participants")
    
    # Files to merge (in order of importance)
    merge_files = [
        "BIOPRO",  # Biochemistry (AST, ALT, etc.)
        "CBC",     # Complete blood count (Platelets)
        "TCHOL",   # Total cholesterol
        "HDL",     # HDL cholesterol
        "TRIGLY",  # Triglycerides, LDL
        "GLU",     # Fasting glucose
        "INS",     # Insulin
        "GHB",     # HbA1c
        "ALB_CR",  # Urine albumin/creatinine
        "BMX",     # Body measures
        "BPX",     # Blood pressure
        "DIQ",     # Diabetes questionnaire
        "MCQ",     # Medical conditions
        "KIQ_U",   # Kidney conditions
        "SMQ",     # Smoking
        "ALQ",     # Alcohol
    ]
    
    for file_key in merge_files:
        # Try pre-pandemic first, then fall back to 2017-2018
        filepath = raw_dir / f"{prefix}{file_key}.XPT"
        if not filepath.exists():
            filepath = raw_dir / f"{file_key}{fallback_suffix}.XPT"
        
        if not filepath.exists():
            print(f"  Skipping {file_key}: file not found")
            continue
        
        print(f"Merging {file_key}: {filepath.name}")
        temp_df = load_xpt(filepath)
        
        if temp_df.empty:
            continue
        
        # Merge on SEQN (participant ID)
        if 'SEQN' in temp_df.columns:
            # Get only new columns (avoid duplicates)
            existing_cols = set(df.columns)
            new_cols = ['SEQN'] + [c for c in temp_df.columns if c not in existing_cols]
            temp_df = temp_df[new_cols]
            
            df = df.merge(temp_df, on='SEQN', how='left')
            print(f"  Merged: {len(temp_df)} rows, now {len(df.columns)} columns")
    
    return df

def clean_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data and apply standard variable names."""
    
    # Rename variables
    rename_dict = {k: v for k, v in VARIABLE_RENAMES.items() if k in df.columns}
    df = df.rename(columns=rename_dict)
    
    print(f"\nRenamed {len(rename_dict)} variables to standard names")
    
    # Keep only renamed variables + SEQN
    keep_cols = ['SEQN'] + list(VARIABLE_RENAMES.values())
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]
    
    # Convert sex to binary (1=Male, 0=Female)
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({1: 1, 2: 0})  # 1=Male, 0=Female
        df['SexLabel'] = df['Sex'].map({1: 'Male', 0: 'Female'})
    
    # Clean diabetes diagnosis
    if 'DiabetesDiagnosis' in df.columns:
        # 1=Yes, 2=No, 3=Borderline -> 1=Yes, 0=No, 0.5=Borderline
        df['HasDiabetes'] = df['DiabetesDiagnosis'].map({1: 1, 2: 0, 3: 0})
        df['Prediabetes'] = df['DiabetesDiagnosis'].map({1: 0, 2: 0, 3: 1})
    
    # Clean medical conditions (1=Yes, 2=No -> 1=Yes, 0=No)
    binary_vars = ['HeartFailure', 'CoronaryHeartDisease', 'HeartAttack', 
                   'Stroke', 'Cancer', 'WeakKidneys', 'Dialysis']
    for var in binary_vars:
        if var in df.columns:
            df[var] = df[var].map({1: 1, 2: 0})
    
    # Create composite CVD variable
    cvd_vars = ['HeartFailure', 'CoronaryHeartDisease', 'HeartAttack', 'Stroke']
    cvd_vars = [v for v in cvd_vars if v in df.columns]
    if cvd_vars:
        df['HasCVD'] = df[cvd_vars].max(axis=1)
    
    return df

def add_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived clinical variables."""
    
    # Age groups
    if 'Age' in df.columns:
        df['AgeGroup'] = pd.cut(df['Age'], 
                                bins=[0, 40, 50, 60, 70, 100],
                                labels=['<40', '40-49', '50-59', '60-69', '70+'])
    
    # BMI categories
    if 'BMI' in df.columns:
        df['BMICategory'] = pd.cut(df['BMI'],
                                   bins=[0, 18.5, 25, 30, 100],
                                   labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # HbA1c categories
    if 'HbA1c' in df.columns:
        df['GlycemicStatus'] = pd.cut(df['HbA1c'],
                                       bins=[0, 5.7, 6.5, 100],
                                       labels=['Normal', 'Prediabetes', 'Diabetes'])
    
    # eGFR calculation (CKD-EPI 2021 equation without race)
    if all(v in df.columns for v in ['Creatinine', 'Age', 'Sex']):
        # Simplified CKD-EPI 2021
        df['eGFR'] = df.apply(lambda row: calculate_egfr(
            row['Creatinine'], row['Age'], row['Sex']
        ), axis=1)
        
        df['CKDStage'] = pd.cut(df['eGFR'],
                                bins=[0, 15, 30, 45, 60, 90, 200],
                                labels=['Stage 5', 'Stage 4', 'Stage 3b', 'Stage 3a', 'Stage 2', 'Stage 1'])
    
    # Urine ACR
    if all(v in df.columns for v in ['UrineAlbumin', 'UrineCreatinine']):
        df['ACR'] = df['UrineAlbumin'] / (df['UrineCreatinine'] / 100)  # mg/g
        df['Albuminuria'] = pd.cut(df['ACR'],
                                   bins=[0, 30, 300, 10000],
                                   labels=['Normal', 'Microalbuminuria', 'Macroalbuminuria'])
    
    return df

def calculate_egfr(creatinine: float, age: int, sex: int) -> float:
    """Calculate eGFR using CKD-EPI 2021 equation (without race)."""
    if pd.isna(creatinine) or pd.isna(age) or pd.isna(sex):
        return np.nan
    
    # CKD-EPI 2021 (race-free)
    if sex == 0:  # Female
        if creatinine <= 0.7:
            egfr = 142 * (creatinine / 0.7) ** -0.241 * (0.9938 ** age) * 1.012
        else:
            egfr = 142 * (creatinine / 0.7) ** -1.2 * (0.9938 ** age) * 1.012
    else:  # Male
        if creatinine <= 0.9:
            egfr = 142 * (creatinine / 0.9) ** -0.302 * (0.9938 ** age)
        else:
            egfr = 142 * (creatinine / 0.9) ** -1.2 * (0.9938 ** age)
    
    return egfr

def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal participants: {len(df)}")
    print(f"Total variables: {len(df.columns)}")
    
    # Demographics
    print("\n--- Demographics ---")
    if 'Age' in df.columns:
        print(f"Age: {df['Age'].mean():.1f} ± {df['Age'].std():.1f} years")
    if 'SexLabel' in df.columns:
        print(f"Sex: {df['SexLabel'].value_counts().to_dict()}")
    if 'BMI' in df.columns:
        print(f"BMI: {df['BMI'].mean():.1f} ± {df['BMI'].std():.1f}")
    
    # Key biomarkers
    print("\n--- Key Biomarkers (non-missing counts) ---")
    key_vars = ['AST', 'ALT', 'Platelets', 'Creatinine', 'Glucose', 
                'Insulin', 'HbA1c', 'HDL', 'Triglycerides', 'eGFR']
    for var in key_vars:
        if var in df.columns:
            n_valid = df[var].notna().sum()
            pct = 100 * n_valid / len(df)
            print(f"  {var}: {n_valid} ({pct:.1f}%)")
    
    # Disease prevalence
    print("\n--- Disease Prevalence ---")
    if 'HasDiabetes' in df.columns:
        prev = 100 * df['HasDiabetes'].mean()
        print(f"  Diabetes: {prev:.1f}%")
    if 'HasCVD' in df.columns:
        prev = 100 * df['HasCVD'].mean()
        print(f"  Cardiovascular Disease: {prev:.1f}%")
    if 'CKDStage' in df.columns:
        ckd = df['CKDStage'].isin(['Stage 3a', 'Stage 3b', 'Stage 4', 'Stage 5']).mean() * 100
        print(f"  CKD (Stage ≥3): {ckd:.1f}%")
    
    # Missing data
    print("\n--- Missing Data ---")
    missing = df.isnull().sum()
    high_missing = missing[missing > len(df) * 0.5]
    if len(high_missing) > 0:
        print("Variables with >50% missing:")
        for var, n in high_missing.items():
            print(f"  {var}: {100*n/len(df):.1f}% missing")

def main():
    """Main execution."""
    
    print("=" * 60)
    print("PAPER 2 EXTENDED: NHANES DATA MERGE")
    print("The Revelatory Division - Multi-Disease Validation")
    print("=" * 60)
    print()
    
    # Create output directory
    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    
    # Check for raw data
    if not RAW_DATA.exists():
        print(f"✗ Raw data directory not found: {RAW_DATA}")
        print("  Please run 00_download_nhanes.py first")
        return
    
    xpt_files = list(RAW_DATA.glob("*.XPT"))
    if len(xpt_files) == 0:
        print(f"✗ No XPT files found in {RAW_DATA}")
        print("  Please run 00_download_nhanes.py first")
        return
    
    print(f"Found {len(xpt_files)} XPT files in {RAW_DATA}")
    print()
    
    # Merge files
    print("=" * 60)
    print("MERGING NHANES FILES")
    print("=" * 60)
    
    df = merge_nhanes_files(RAW_DATA, use_prepandemic=True)
    print(f"\nMerged dataset: {len(df)} participants, {len(df.columns)} variables")
    
    # Clean and rename
    print()
    print("=" * 60)
    print("CLEANING AND STANDARDIZING")
    print("=" * 60)
    
    df = clean_and_rename(df)
    print(f"After cleaning: {len(df)} participants, {len(df.columns)} variables")
    
    # Add derived variables
    print()
    print("=" * 60)
    print("ADDING DERIVED VARIABLES")
    print("=" * 60)
    
    df = add_derived_variables(df)
    print(f"After derivation: {len(df)} participants, {len(df.columns)} variables")
    
    # Summary
    print_summary(df)
    
    # Save
    output_file = PROCESSED_DATA / "nhanes_merged.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved merged data to: {output_file}")
    
    # Also save variable list
    var_file = PROCESSED_DATA / "variable_list.txt"
    with open(var_file, 'w', encoding='utf-8') as f:
        f.write("NHANES Merged Dataset Variables\n")
        f.write("=" * 40 + "\n\n")
        for col in sorted(df.columns):
            n_valid = df[col].notna().sum()
            dtype = df[col].dtype
            f.write(f"{col}: {dtype} ({n_valid} valid)\n")
    
    print(f"✓ Saved variable list to: {var_file}")
    
    print()
    print("=" * 60)
    print("✓ Merge complete!")
    print()
    print("Next step: Run 02_compute_all_ratios.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
