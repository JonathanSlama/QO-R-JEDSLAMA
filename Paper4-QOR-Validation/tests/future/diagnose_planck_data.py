#!/usr/bin/env python3
"""
Diagnostic script for Planck PSZ2 data quality issues.
Let's understand what's really in MSZ and Y5R500.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "level2_multicontext", "clusters", "COM_PCCS_SZ-Catalogs_vPR2", "HFI_PCCS_SZ-union_R2.08.fits.gz")

print("="*70)
print(" PLANCK PSZ2 DATA DIAGNOSTIC")
print("="*70)

with fits.open(DATA_PATH) as hdul:
    data = hdul[1].data

n_total = len(data)
print(f"\nTotal clusters: {n_total}")

# Extract key columns
MSZ = data['MSZ']
Y5R500 = data['Y5R500']
REDSHIFT = data['REDSHIFT']
SNR = data['SNR']

print("\n" + "="*70)
print(" RAW DATA STATISTICS")
print("="*70)

print(f"\nMSZ (Mass in 10^14 M_sun):")
print(f"  Min: {np.nanmin(MSZ):.4f}")
print(f"  Max: {np.nanmax(MSZ):.4f}")
print(f"  Mean: {np.nanmean(MSZ):.4f}")
print(f"  N(MSZ == 0): {np.sum(MSZ == 0)}")
print(f"  N(MSZ > 0): {np.sum(MSZ > 0)}")
print(f"  N(MSZ < 0): {np.sum(MSZ < 0)}")
print(f"  N(isnan): {np.sum(np.isnan(MSZ))}")

print(f"\nY5R500 (SZ signal in arcmin^2):")
print(f"  Min: {np.nanmin(Y5R500):.4f}")
print(f"  Max: {np.nanmax(Y5R500):.4f}")
print(f"  Mean: {np.nanmean(Y5R500):.4f}")
print(f"  N(Y5R500 > 0): {np.sum(Y5R500 > 0)}")

print(f"\nREDSHIFT:")
print(f"  Min: {np.nanmin(REDSHIFT):.4f}")
print(f"  Max: {np.nanmax(REDSHIFT):.4f}")
print(f"  N(z == -1): {np.sum(REDSHIFT == -1)}")
print(f"  N(z > 0): {np.sum(REDSHIFT > 0)}")
print(f"  N(0 < z < 1): {np.sum((REDSHIFT > 0) & (REDSHIFT < 1))}")

# Look at the "good" sample
print("\n" + "="*70)
print(" GOOD SAMPLE (z > 0.01, MSZ > 0.1, SNR > 4.5)")
print("="*70)

good_mask = (REDSHIFT > 0.01) & (REDSHIFT < 1.0) & (MSZ > 0.1) & (SNR > 4.5) & (Y5R500 > 0)
n_good = good_mask.sum()
print(f"\nN good: {n_good}")

MSZ_good = MSZ[good_mask]
Y_good = Y5R500[good_mask]
z_good = REDSHIFT[good_mask]

print(f"\nGood MSZ:")
print(f"  Min: {MSZ_good.min():.4f}")
print(f"  Max: {MSZ_good.max():.4f}")
print(f"  Median: {np.median(MSZ_good):.4f}")

print(f"\nGood Y5R500:")
print(f"  Min: {Y_good.min():.4f}")
print(f"  Max: {Y_good.max():.4f}")
print(f"  Median: {np.median(Y_good):.4f}")

print(f"\nGood z:")
print(f"  Min: {z_good.min():.4f}")
print(f"  Max: {z_good.max():.4f}")
print(f"  Median: {np.median(z_good):.4f}")

# Check the M-Y correlation
print("\n" + "="*70)
print(" M-Y CORRELATION CHECK")
print("="*70)

log_M = np.log10(MSZ_good)
log_Y = np.log10(Y_good)

from scipy import stats
slope, intercept, r, p, err = stats.linregress(log_M, log_Y)

print(f"\nlog(Y) = {slope:.4f} * log(M) + {intercept:.4f}")
print(f"Correlation r: {r:.4f}")
print(f"p-value: {p:.2e}")

# Expected: Y ∝ M^(5/3), so slope should be ~1.67
print(f"\nExpected slope: ~1.67 (self-similar)")
print(f"Observed slope: {slope:.4f}")

if slope < 0:
    print("\n WARNING: NEGATIVE SLOPE - This is UNPHYSICAL!")
    print("   Possible causes:")
    print("   1. MSZ and Y5R500 are not independent (MSZ derived from Y?)")
    print("   2. Selection effects dominating")
    print("   3. Data quality issues")

# Let's check if MSZ is derived from Y5R500
print("\n" + "="*70)
print(" CHECKING IF MSZ IS DERIVED FROM Y5R500")
print("="*70)

# If MSZ is derived from Y, they should have a tight correlation
# with slope = 3/5 = 0.6 (inverse of 5/3)
# i.e., M ∝ Y^(3/5) → log(M) = 0.6 * log(Y) + const

slope_MY, intercept_MY, r_MY, p_MY, err_MY = stats.linregress(log_Y, log_M)
print(f"\nlog(M) = {slope_MY:.4f} * log(Y) + {intercept_MY:.4f}")
print(f"Correlation r: {r_MY:.4f}")
print(f"If MSZ derived from Y via Y-M relation, slope should be ~0.6")

# Check a few specific clusters
print("\n" + "="*70)
print(" SAMPLE CLUSTERS (first 10 good ones)")
print("="*70)

print(f"\n{'Name':<20} {'z':>8} {'MSZ':>10} {'Y5R500':>12} {'log(M)':>8} {'log(Y)':>8}")
print("-"*70)

names = data['NAME'][good_mask]
for i in range(min(10, n_good)):
    name = names[i] if isinstance(names[i], str) else names[i].decode()
    print(f"{name:<20} {z_good[i]:>8.3f} {MSZ_good[i]:>10.3f} {Y_good[i]:>12.4f} {log_M[i]:>8.3f} {log_Y[i]:>8.3f}")

# Distribution plots
print("\n" + "="*70)
print(" CREATING DIAGNOSTIC PLOTS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: M-Y scatter
ax1 = axes[0, 0]
ax1.scatter(log_M, log_Y, alpha=0.5, s=10)
x_line = np.linspace(log_M.min(), log_M.max(), 100)
ax1.plot(x_line, slope * x_line + intercept, 'r-', label=f'Fit: slope={slope:.2f}')
ax1.plot(x_line, 1.67 * x_line + intercept - 1.67*np.mean(log_M) + np.mean(log_Y), 'g--', label='Expected: slope=1.67')
ax1.set_xlabel('log(MSZ / 10^14 M_sun)')
ax1.set_ylabel('log(Y5R500 / arcmin²)')
ax1.set_title('M-Y Relation')
ax1.legend()

# Plot 2: MSZ histogram
ax2 = axes[0, 1]
ax2.hist(MSZ_good, bins=50, edgecolor='black')
ax2.set_xlabel('MSZ (10^14 M_sun)')
ax2.set_ylabel('Count')
ax2.set_title('Mass Distribution')

# Plot 3: Y5R500 histogram
ax3 = axes[1, 0]
ax3.hist(Y_good, bins=50, edgecolor='black')
ax3.set_xlabel('Y5R500 (arcmin²)')
ax3.set_ylabel('Count')
ax3.set_title('SZ Signal Distribution')

# Plot 4: Redshift histogram
ax4 = axes[1, 1]
ax4.hist(z_good, bins=50, edgecolor='black')
ax4.set_xlabel('Redshift')
ax4.set_ylabel('Count')
ax4.set_title('Redshift Distribution')

plt.tight_layout()
plt.savefig('planck_diagnostic.png', dpi=150)
print("\nPlot saved: planck_diagnostic.png")

# Final diagnosis
print("\n" + "="*70)
print(" DIAGNOSIS")
print("="*70)

if abs(r_MY) > 0.9:
    print("\n ANALYSIS: MSZ appears to be DIRECTLY DERIVED from Y5R500!")
    print("   This means they are not independent variables.")
    print("   The 'residuals' we compute are just noise/scatter in the Y-M relation.")
    print("\n   SOLUTION: We need EXTERNAL mass estimates (e.g., from X-ray, weak lensing)")
    print("   Or: Use only Y5R500 as the observable and a different proxy for environment.")
elif slope < 0:
    print("\n ANALYSIS: Negative M-Y slope despite good data quality cuts.")
    print("   This might indicate selection effects or data issues.")
else:
    print("\n RESULT: Data appears reasonable for analysis.")

print("\n" + "="*70)
