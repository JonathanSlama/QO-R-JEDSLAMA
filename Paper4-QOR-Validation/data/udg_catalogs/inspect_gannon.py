#!/usr/bin/env python3
"""
Inspect Gannon+2024 UDG Spectroscopic Catalog
THE CRITICAL DATASET for Killer Prediction #1
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.dirname(__file__)
GANNON_PATH = os.path.join(DATA_DIR, "Gannon2024_udg_data.csv")

print("="*70)
print(" GANNON+2024 UDG SPECTROSCOPIC CATALOG")
print(" THE CRITICAL DATASET FOR QO+R KILLER PREDICTION #1")
print("="*70)

# Load data
df = pd.read_csv(GANNON_PATH)

print(f"\nSuccessfully loaded: {GANNON_PATH}")
print(f"N objects: {len(df)}")
print(f"N columns: {len(df.columns)}")

print("\n" + "="*70)
print(" ALL COLUMNS")
print("="*70)
for i, col in enumerate(df.columns):
    dtype = df[col].dtype
    n_valid = df[col].notna().sum()
    print(f"  {i+1:2d}. {col:<25} ({dtype}, {n_valid}/{len(df)} valid)")

print("\n" + "="*70)
print(" KEY COLUMNS FOR QO+R TEST")
print("="*70)

# Check for critical columns
critical = {
    'Environment': ['Env', 'Environment', 'env', 'environment'],
    'Distance': ['Dist', 'Distance', 'D', 'dist'],
    'Velocity': ['V', 'Vel', 'v', 'Vrec', 'V_rec', 'cz'],
    'Velocity Dispersion': ['sigma', 'Sigma', 'sigma_star', 'sig'],
    'Stellar Mass': ['Mstar', 'M_star', 'logM', 'log_Mstar', 'Mass'],
    'Age': ['Age', 'age', 'Age_Gyr'],
    'Metallicity': ['MH', 'FeH', 'Z', 'metallicity', '[M/H]', '[Fe/H]'],
    'Position RA': ['RA', 'ra', 'RAJ2000'],
    'Position DEC': ['DEC', 'Dec', 'dec', 'DEJ2000'],
}

for category, possible_names in critical.items():
    found = None
    for name in possible_names:
        matches = [c for c in df.columns if name.lower() in c.lower()]
        if matches:
            found = matches[0]
            break
    
    if found:
        n_valid = df[found].notna().sum()
        print(f"  {category:<20}: {found} ({n_valid}/{len(df)} valid)")
    else:
        print(f"  {category:<20}: NOT FOUND")

print("\n" + "="*70)
print(" ENVIRONMENT DISTRIBUTION")
print("="*70)

# Find environment column
env_cols = [c for c in df.columns if 'env' in c.lower()]
if env_cols:
    env_col = env_cols[0]
    print(f"Environment column: {env_col}")
    print(df[env_col].value_counts())
    print("\nLegend: 1=Cluster, 2=Group, 3=Field")
else:
    print("Environment column not found - checking all columns...")
    print(df.columns.tolist())

print("\n" + "="*70)
print(" FIRST 5 ROWS (KEY COLUMNS)")
print("="*70)

# Show first rows with key columns
key_cols = []
for col in df.columns:
    col_lower = col.lower()
    if any(k in col_lower for k in ['name', 'env', 'dist', 'sigma', 'mass', 'age', 'vel']):
        key_cols.append(col)

if key_cols:
    print(df[key_cols[:8]].head())
else:
    print(df.head())

print("\n" + "="*70)
print(" STATISTICS FOR NUMERIC COLUMNS")
print("="*70)

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols[:10]:
    valid = df[col].dropna()
    if len(valid) > 0:
        print(f"  {col:<25}: min={valid.min():.2f}, max={valid.max():.2f}, mean={valid.mean():.2f}")

print("\n" + "="*70)
print(" ASSESSMENT")
print("="*70)
print("""
This is THE critical dataset for Killer Prediction #1!

For UDG test we need:
1. Environment (cluster/group/field) → For U-shape analysis
2. Velocity dispersion → For dynamical mass / BTFR
3. Stellar mass → For baryonic mass
4. Distance → For physical size

QO+R PREDICTION: UDGs are R-dominated (gas-poor)
                 → Should show INVERTED U-shape (a < 0)
                 
Next: Run killer_prediction_1_udg.py
""")

# Save summary
summary = {
    'n_objects': len(df),
    'n_columns': len(df.columns),
    'columns': df.columns.tolist(),
}
print(f"\nColumns available: {df.columns.tolist()}")
