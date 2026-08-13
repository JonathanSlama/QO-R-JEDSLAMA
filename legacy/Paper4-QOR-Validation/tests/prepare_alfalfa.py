#!/usr/bin/env python3
"""
===============================================================================
PREPARE ALFALFA DATA - CORRECTED VERSION
===============================================================================
Paper 4 - QO+R Validation

CRITICAL FIX:
-------------
Previous version used scaling relation: M_star = 0.7 × log(M_HI) + 2.5
This created artificial Q-R correlation of +0.997 (WRONG!)

Corrected version uses:
- ALFALFA α.100 (Haynes et al. 2018) for M_HI
- Durbala et al. (2020) for M_star (SDSS photometry - INDEPENDENT!)
- Resulting Q-R correlation: -0.808 (physically correct)

Author: Jonathan Édouard Slama
Date: December 2025
===============================================================================
"""

import numpy as np
import pandas as pd
import os
import sys

# Try to import astropy for FITS reading
try:
    from astropy.table import Table
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    print("WARNING: astropy not installed. Install with: pip install astropy")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to BTFR directory (relative to Git folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
GIT_DIR = os.path.dirname(PROJECT_DIR)
BTFR_DIR = os.path.join(os.path.dirname(GIT_DIR), "BTFR")

# Input files (raw data)
ALFALFA_RAW = os.path.join(BTFR_DIR, "Test-Replicability", "data", "Alfalfa", "a100.code12.table2.190808.csv")
DURBALA_T2 = os.path.join(BTFR_DIR, "Test-Replicability", "data", "Alfalfa", "durbala2020-table2.21-Sep-2020.fits.gz")

# Output file
OUTPUT_FILE = os.path.join(BTFR_DIR, "Test-Replicability", "data", "alfalfa_corrected.csv")

# Parameters
MIN_DISTANCE_MPC = 5
MAX_DISTANCE_MPC = 200
MIN_SNR = 6.5
HICODE_QUALITY = 1
MSTAR_COLUMN = 'logMstarTaylor'  # From Durbala 2020


def main():
    print("="*70)
    print(" ALFALFA DATA PREPARATION - CORRECTED VERSION")
    print("="*70)
    
    # Check files exist
    if not os.path.exists(ALFALFA_RAW):
        print(f"ERROR: ALFALFA file not found: {ALFALFA_RAW}")
        return False
    
    if not os.path.exists(DURBALA_T2):
        print(f"ERROR: Durbala 2020 file not found: {DURBALA_T2}")
        return False
    
    if not HAS_ASTROPY:
        print("ERROR: astropy required to read FITS file")
        return False
    
    # ==========================================================================
    # 1. Load ALFALFA α.100
    # ==========================================================================
    print("\n[1] Loading ALFALFA α.100...")
    alf_raw = pd.read_csv(ALFALFA_RAW)
    print(f"    Loaded: {len(alf_raw)} sources")
    
    # ==========================================================================
    # 2. Load Durbala 2020 Table 2
    # ==========================================================================
    print("\n[2] Loading Durbala 2020 Table 2...")
    durbala = Table.read(DURBALA_T2).to_pandas()
    durbala = durbala.rename(columns={'AGC': 'AGCNr'})
    print(f"    Loaded: {len(durbala)} galaxies")
    
    valid_mstar = durbala[MSTAR_COLUMN].notna().sum()
    print(f"    Valid {MSTAR_COLUMN}: {valid_mstar} ({100*valid_mstar/len(durbala):.1f}%)")
    
    # ==========================================================================
    # 3. Merge catalogs
    # ==========================================================================
    print("\n[3] Merging catalogs by AGC number...")
    merged = pd.merge(alf_raw, durbala, on='AGCNr', how='inner')
    print(f"    After merge: {len(merged)} galaxies")
    
    # ==========================================================================
    # 4. Quality filtering
    # ==========================================================================
    print("\n[4] Quality filtering...")
    n_start = len(merged)
    
    # HI quality
    merged = merged[merged['HIcode'] == HICODE_QUALITY].copy()
    print(f"    After HIcode=1: {len(merged)}")
    
    # Distance
    merged = merged[(merged['Dist'] > MIN_DISTANCE_MPC) & (merged['Dist'] < MAX_DISTANCE_MPC)].copy()
    print(f"    After distance [{MIN_DISTANCE_MPC}-{MAX_DISTANCE_MPC}] Mpc: {len(merged)}")
    
    # S/N
    merged = merged[merged['SNR'] > MIN_SNR].copy()
    print(f"    After S/N > {MIN_SNR}: {len(merged)}")
    
    # Valid M_star
    merged = merged[merged[MSTAR_COLUMN].notna()].copy()
    print(f"    After valid {MSTAR_COLUMN}: {len(merged)}")
    
    # Valid M_HI (column becomes logMH_x after merge)
    mhi_col = 'logMH_x' if 'logMH_x' in merged.columns else 'logMH'
    merged = merged[merged[mhi_col].notna()].copy()
    print(f"    After valid logMH: {len(merged)}")
    
    # ==========================================================================
    # 5. Compute derived properties
    # ==========================================================================
    print("\n[5] Computing derived properties...")
    
    df = pd.DataFrame()
    
    # Identifiers
    df['Galaxy'] = 'AGC' + merged['AGCNr'].astype(str)
    df['AGCNr'] = merged['AGCNr'].values
    
    # Coordinates
    df['RA'] = merged['RAdeg_HI'].values
    df['Dec'] = merged['DECdeg_HI'].values
    
    # Distance & velocity
    df['Distance_Mpc'] = merged['Dist'].values
    df['Vhelio'] = merged['Vhelio'].values
    
    # HI mass (from ALFALFA - radio 21cm)
    df['log_MHI'] = merged[mhi_col].values
    df['MHI'] = 10**df['log_MHI']
    
    # Stellar mass (from Durbala 2020 - SDSS photometry = INDEPENDENT!)
    df['log_Mstar'] = merged[MSTAR_COLUMN].values
    df['M_star'] = 10**df['log_Mstar']
    
    # Baryonic mass
    df['M_bar'] = df['M_star'] + 1.36 * df['MHI']
    df['log_Mbar'] = np.log10(df['M_bar'])
    
    # Velocity (W50/2 as proxy)
    df['W50'] = merged['W50'].values
    df['Vflat'] = df['W50'] / 2
    df['log_Vflat'] = np.log10(df['Vflat'].clip(lower=1))
    
    # Gas fraction (Q in QO+R)
    df['f_gas'] = (1.36 * df['MHI']) / df['M_bar']
    df['f_HI'] = df['MHI'] / df['M_bar']
    df['HI_to_star'] = df['MHI'] / df['M_star']
    
    # ==========================================================================
    # 6. Environment estimation
    # ==========================================================================
    print("\n[6] Estimating environments...")
    
    ra_rad = np.radians(df['RA'].values)
    dec_rad = np.radians(df['Dec'].values)
    sgb_approx = np.sin(dec_rad) * 0.5 + np.cos(dec_rad) * np.sin(ra_rad - np.radians(137)) * 0.866
    
    # Known clusters
    virgo_mask = ((df['RA'] > 180) & (df['RA'] < 195) &
                  (df['Dec'] > 5) & (df['Dec'] < 20) &
                  (df['Distance_Mpc'] > 12) & (df['Distance_Mpc'] < 25))
    
    coma_mask = ((df['RA'] > 190) & (df['RA'] < 200) &
                 (df['Dec'] > 25) & (df['Dec'] < 32) &
                 (df['Distance_Mpc'] > 85) & (df['Distance_Mpc'] < 115))
    
    # Density proxy
    density_from_sgb = 1 - np.abs(sgb_approx)
    density_from_dist = np.clip(df['Distance_Mpc'] / 100, 0, 1)
    
    cluster_boost = np.zeros(len(df))
    cluster_boost[virgo_mask] = 0.3
    cluster_boost[coma_mask] = 0.4
    
    density_proxy = 0.5 * density_from_sgb + 0.3 * density_from_dist + cluster_boost
    density_proxy = np.clip(density_proxy, 0, 1)
    
    np.random.seed(42)
    density_proxy += np.random.normal(0, 0.1, len(density_proxy))
    density_proxy = np.clip(density_proxy, 0, 1)
    
    df['density_proxy'] = density_proxy
    df['env_class'] = np.where(density_proxy < 0.33, 'void',
                      np.where(density_proxy < 0.67, 'field', 'cluster'))
    
    # ==========================================================================
    # 7. BTFR residuals
    # ==========================================================================
    print("\n[7] Computing BTFR residuals...")
    
    valid = df[['log_Mbar', 'log_Vflat']].dropna()
    coeffs = np.polyfit(valid['log_Mbar'], valid['log_Vflat'], 1)
    slope, intercept = coeffs[0], coeffs[1]
    
    df['btfr_pred'] = intercept + slope * df['log_Mbar']
    df['btfr_residual'] = df['log_Vflat'] - df['btfr_pred']
    
    print(f"    BTFR: log(Vflat) = {slope:.3f} × log(Mbar) + {intercept:.3f}")
    
    # Remove outliers
    n_before = len(df)
    df = df[np.abs(df['btfr_residual']) < 0.5].copy()
    print(f"    After outlier removal: {len(df)} ({n_before - len(df)} removed)")
    
    # ==========================================================================
    # 8. VALIDATION: Check Q-R independence
    # ==========================================================================
    print("\n" + "="*70)
    print(" VALIDATION: Q-R INDEPENDENCE")
    print("="*70)
    
    Q = df['f_gas'].values
    R = df['log_Mstar'].values
    corr_QR = np.corrcoef(Q, R)[0, 1]
    
    print(f"\n    Q (f_gas):     mean={Q.mean():.3f}, std={Q.std():.3f}")
    print(f"    R (log_Mstar): mean={R.mean():.2f}, std={R.std():.2f}")
    print(f"\n    *** Correlation Q-R = {corr_QR:+.3f} ***")
    
    if corr_QR > 0.9:
        print("\n    ERROR: Q-R correlation > 0.9 - variables not independent!")
        return False
    elif corr_QR > 0:
        print("\n    WARNING: Positive correlation (expected negative)")
    else:
        print("\n    Negative correlation - PHYSICALLY CORRECT")
    
    # ==========================================================================
    # 9. Save
    # ==========================================================================
    print("\n" + "="*70)
    print(" SAVING")
    print("="*70)
    
    output_cols = [
        'Galaxy', 'AGCNr', 'RA', 'Dec', 'Distance_Mpc', 'Vhelio',
        'log_MHI', 'MHI', 'log_Mstar', 'M_star', 'M_bar', 'log_Mbar',
        'W50', 'Vflat', 'log_Vflat',
        'f_gas', 'f_HI', 'HI_to_star',
        'density_proxy', 'env_class',
        'btfr_pred', 'btfr_residual'
    ]
    
    df_out = df[output_cols].copy()
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n    Saved: {OUTPUT_FILE}")
    print(f"       {len(df_out)} galaxies, {len(output_cols)} columns")
    
    # Summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    print(f"""
    Total galaxies: {len(df_out)}
    
    Q (f_gas) range: [{df_out['f_gas'].min():.3f}, {df_out['f_gas'].max():.3f}]
    R (log_Mstar) range: [{df_out['log_Mstar'].min():.2f}, {df_out['log_Mstar'].max():.2f}]
    Q-R correlation: {corr_QR:+.3f}
    
    Stratification:
      Q-dominated (f_gas >= 0.5): {(df_out['f_gas'] >= 0.5).sum()}
      R-dominated (f_gas < 0.3):  {(df_out['f_gas'] < 0.3).sum()}
    
    Environment:
      Void:    {(df_out['env_class'] == 'void').sum()}
      Field:   {(df_out['env_class'] == 'field').sum()}
      Cluster: {(df_out['env_class'] == 'cluster').sum()}
    """)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
