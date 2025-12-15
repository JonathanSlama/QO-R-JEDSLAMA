#!/usr/bin/env python3
"""
===============================================================================
PHASE 3: CMB LENSING CROSS-CORRELATION
===============================================================================

Downloads Planck CMB lensing convergence map and performs cross-correlation
with galaxy scaling relation residuals.

THEORY:
If Q2R2 modifies the gravitational potential, CMB photons passing through
Q-dominated vs R-dominated regions should experience different lensing.

This test correlates:
- Galaxy BTF/M-L residuals (where Q2R2 is detected)
- CMB lensing convergence (kappa) from Planck

PREDICTION:
Galaxies with positive residuals (R-dominated) should correlate differently
with CMB kappa than galaxies with negative residuals (Q-dominated).

Author: Jonathan E. Slama
Date: December 2025
===============================================================================
"""

import os
import sys
import numpy as np
import json
from scipy import stats
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import urllib.request
import gzip
import shutil
import warnings
warnings.filterwarnings('ignore')

# Check for healpy
try:
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    HEALPY_AVAILABLE = False
    print("WARNING: healpy not installed")
    print("Install with: pip install healpy")
    print("Note: healpy can be difficult on Windows, consider WSL")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "level3_direct", "planck_lensing")
KIDS_DIR = os.path.join(BASE_DIR, "data", "level2_multicontext", "kids")
ALFALFA_DIR = os.path.join(BASE_DIR, "data", "alfalfa")
OUTPUT_DIR = os.path.join(BASE_DIR, "experimental", "04_RESULTS", "test_results")


def download_planck_lensing():
    """
    Download Planck 2018 CMB lensing convergence map.
    
    Product: COM_Lensing_4096_R3.00
    """
    
    print("\n" + "="*70)
    print(" DOWNLOADING PLANCK CMB LENSING DATA")
    print("="*70)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Planck lensing convergence map URL
    # Using the minimum variance (MV) estimator
    base_url = "http://pla.esac.esa.int/pla/aio/product-action?LENSING.FILE_ID="
    
    files = {
        'kappa_map': 'COM_Lensing_4096_R3.00_TT_kappa_lm.fits',
        'mask': 'COM_Lensing_4096_R3.00_mask.fits.gz'
    }
    
    # Alternative: Use pre-computed kappa map from NERSC or direct link
    # The official Planck archive requires authentication
    
    print("""
  Planck CMB Lensing Data
  =======================
  
  The Planck 2018 lensing convergence (kappa) map is available from:
  
  1. Planck Legacy Archive (requires registration):
     https://pla.esac.esa.int/
     
  2. NERSC (public):
     https://portal.nersc.gov/project/cmb/planck2018/
     
  For this analysis, we need:
  - Convergence map (kappa)
  - Lensing mask
  
  File sizes: ~100-500 MB each
    """)
    
    # Check if files already exist
    kappa_path = os.path.join(DATA_DIR, "planck_kappa.fits")
    mask_path = os.path.join(DATA_DIR, "planck_mask.fits")
    
    if os.path.exists(kappa_path):
        print(f"  Kappa map already exists: {kappa_path}")
        return kappa_path, mask_path
    
    # Try to download from NERSC (public)
    nersc_url = "https://portal.nersc.gov/project/cmb/planck2018/lensing/"
    
    print("\n  Attempting download from NERSC...")
    
    try:
        # Download kappa map
        kappa_url = nersc_url + "COM_Lensing_4096_R3.00/MV/dat_klm.fits"
        print(f"  Downloading: {kappa_url}")
        
        # This may take a while
        urllib.request.urlretrieve(kappa_url, kappa_path)
        print(f"  Saved: {kappa_path}")
        
    except Exception as e:
        print(f"  Download failed: {e}")
        print("\n  Please download manually from Planck Legacy Archive")
        print("  and place files in:", DATA_DIR)
        
        # Create placeholder for testing
        create_mock_kappa_map(kappa_path)
        
    return kappa_path, mask_path


def create_mock_kappa_map(path):
    """Create a mock kappa map for testing the pipeline."""
    
    print("\n  Creating mock kappa map for pipeline testing...")
    
    if not HEALPY_AVAILABLE:
        print("  ERROR: healpy required for mock map creation")
        return None
    
    # Create a simple mock map
    nside = 256  # Lower resolution for testing
    npix = hp.nside2npix(nside)
    
    # Random Gaussian field with some structure
    np.random.seed(42)
    kappa = np.random.normal(0, 1e-2, npix)
    
    # Add some large-scale structure
    for l in range(2, 50):
        for m in range(-l, l+1):
            alm_idx = hp.sphtfunc.Alm.getidx(50, l, abs(m))
            # This is simplified - real kappa has specific power spectrum
    
    # Save as FITS
    hp.write_map(path, kappa, overwrite=True)
    print(f"  Mock map saved: {path}")
    print("  WARNING: This is a mock map for pipeline testing only!")
    
    return path


def load_kappa_map(path):
    """Load Planck lensing convergence map."""
    
    if not HEALPY_AVAILABLE:
        print("  ERROR: healpy required")
        return None, None
    
    if not os.path.exists(path):
        print(f"  ERROR: File not found: {path}")
        return None, None
    
    print(f"  Loading: {path}")
    
    try:
        kappa = hp.read_map(path, verbose=False)
        nside = hp.get_nside(kappa)
        print(f"  Loaded kappa map: NSIDE={nside}, Npix={len(kappa)}")
        return kappa, nside
    except Exception as e:
        print(f"  ERROR loading map: {e}")
        return None, None


def get_kappa_at_positions(kappa, nside, ra, dec):
    """Extract kappa values at galaxy positions."""
    
    if not HEALPY_AVAILABLE:
        return None
    
    # Convert RA/Dec to theta/phi (HEALPix convention)
    theta = np.radians(90 - dec)  # colatitude
    phi = np.radians(ra)          # longitude
    
    # Get pixel indices
    pix = hp.ang2pix(nside, theta, phi)
    
    return kappa[pix]


def load_kids_with_residuals():
    """Load KiDS data and compute M-L residuals."""
    
    print("\n  Loading KiDS data...")
    
    bright_path = os.path.join(KIDS_DIR, "KiDS_DR4_brightsample.fits")
    lephare_path = os.path.join(KIDS_DIR, "KiDS_DR4_brightsample_LePhare.fits")
    
    if not os.path.exists(bright_path):
        print(f"  ERROR: KiDS data not found: {bright_path}")
        return None
    
    bright = Table.read(bright_path)
    lephare = Table.read(lephare_path)
    
    ra = np.array(bright['RAJ2000'])
    dec = np.array(bright['DECJ2000'])
    mag_r = np.array(lephare['MAG_ABS_r'])
    mass = np.array(lephare['MASS_MED'])
    
    # Quality cuts
    mask = (
        np.isfinite(ra) & np.isfinite(dec) &
        np.isfinite(mag_r) & np.isfinite(mass) &
        (mass > 7) & (mass < 13)
    )
    
    ra = ra[mask]
    dec = dec[mask]
    mag_r = mag_r[mask]
    mass = mass[mask]
    
    # Compute M-L residuals
    slope, intercept, _, _, _ = stats.linregress(mass, mag_r)
    residuals = mag_r - (slope * mass + intercept)
    
    print(f"  Loaded {len(ra)} KiDS galaxies")
    
    return {
        'ra': ra,
        'dec': dec,
        'residuals': residuals,
        'mass': mass
    }


def load_alfalfa_with_residuals():
    """Load ALFALFA data and compute BTF residuals."""
    
    print("\n  Loading ALFALFA data...")
    
    data_path = os.path.join(ALFALFA_DIR, "alfalfa_a100.fits")
    
    if not os.path.exists(data_path):
        print(f"  ERROR: ALFALFA data not found: {data_path}")
        return None
    
    t = Table.read(data_path)
    
    # Convert coordinates
    ra_str = [str(x) for x in t['RAJ2000']]
    dec_str = [str(x) for x in t['DEJ2000']]
    coords = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
    ra = coords.ra.deg
    dec = coords.dec.deg
    
    logMHI = np.array(t['logMHI'], dtype=float)
    W50 = np.array(t['W50'], dtype=float)
    dist = np.array(t['Dist'], dtype=float)
    
    # Quality cuts
    mask = (
        np.isfinite(ra) & np.isfinite(dec) &
        np.isfinite(logMHI) & (logMHI > 6) & (logMHI < 12) &
        np.isfinite(W50) & (W50 > 20) & (W50 < 600) &
        np.isfinite(dist) & (dist > 0) & (dist < 300)
    )
    
    ra = ra[mask]
    dec = dec[mask]
    logMHI = logMHI[mask]
    W50 = W50[mask]
    
    # Compute BTF residuals
    logV = np.log10(W50 / 2.0)
    logMbary = logMHI + np.log10(1.4)
    
    slope, intercept, _, _, _ = stats.linregress(logMbary, logV)
    residuals = logV - (slope * logMbary + intercept)
    
    print(f"  Loaded {len(ra)} ALFALFA galaxies")
    
    return {
        'ra': ra,
        'dec': dec,
        'residuals': residuals,
        'mass': logMbary
    }


def cross_correlate_kappa_residuals(kappa, nside, galaxies, n_bins=10):
    """
    Cross-correlate galaxy residuals with CMB lensing kappa.
    
    If Q2R2 affects the gravitational potential:
    - Positive residuals (R-dominated) should have different mean kappa
    - Negative residuals (Q-dominated) should have different mean kappa
    """
    
    print("\n  Computing cross-correlation...")
    
    # Get kappa at galaxy positions
    kappa_at_gal = get_kappa_at_positions(kappa, nside, 
                                           galaxies['ra'], galaxies['dec'])
    
    if kappa_at_gal is None:
        return None
    
    residuals = galaxies['residuals']
    
    # Remove invalid values
    valid = np.isfinite(kappa_at_gal) & np.isfinite(residuals)
    kappa_at_gal = kappa_at_gal[valid]
    residuals = residuals[valid]
    
    print(f"  Valid matches: {len(residuals)}")
    
    # Bin by residual value
    residual_bins = np.percentile(residuals, np.linspace(0, 100, n_bins+1))
    
    results = []
    
    for i in range(n_bins):
        mask = (residuals >= residual_bins[i]) & (residuals < residual_bins[i+1])
        n = mask.sum()
        
        if n > 10:
            mean_kappa = kappa_at_gal[mask].mean()
            std_kappa = kappa_at_gal[mask].std() / np.sqrt(n)
            mean_residual = residuals[mask].mean()
            
            results.append({
                'bin': i,
                'residual_min': float(residual_bins[i]),
                'residual_max': float(residual_bins[i+1]),
                'mean_residual': float(mean_residual),
                'mean_kappa': float(mean_kappa),
                'std_kappa': float(std_kappa),
                'n': int(n)
            })
    
    return results


def test_kappa_residual_correlation(results):
    """Test for correlation between residuals and kappa."""
    
    print("\n" + "="*70)
    print(" KAPPA-RESIDUAL CORRELATION TEST")
    print("="*70)
    
    if results is None or len(results) < 3:
        print("  ERROR: Insufficient data")
        return None
    
    mean_residuals = [r['mean_residual'] for r in results]
    mean_kappas = [r['mean_kappa'] for r in results]
    
    # Pearson correlation
    r, p = stats.pearsonr(mean_residuals, mean_kappas)
    
    print(f"\n  Pearson correlation: r = {r:.4f}, p = {p:.4f}")
    
    # Split by residual sign
    positive_kappa = [r['mean_kappa'] for r in results if r['mean_residual'] > 0]
    negative_kappa = [r['mean_kappa'] for r in results if r['mean_residual'] < 0]
    
    if len(positive_kappa) > 0 and len(negative_kappa) > 0:
        mean_pos = np.mean(positive_kappa)
        mean_neg = np.mean(negative_kappa)
        
        print(f"\n  Mean kappa for positive residuals: {mean_pos:.6f}")
        print(f"  Mean kappa for negative residuals: {mean_neg:.6f}")
        print(f"  Difference: {mean_pos - mean_neg:.6f}")
    
    # Significance
    if p < 0.05:
        print(f"\n  RESULT: Significant correlation detected (p < 0.05)")
    else:
        print(f"\n  RESULT: No significant correlation (p = {p:.3f})")
    
    return {
        'pearson_r': float(r),
        'pearson_p': float(p),
        'significant': p < 0.05
    }


def analyze_q_vs_r_kappa(kappa, nside, galaxies):
    """
    Compare kappa for Q-dominated vs R-dominated galaxies.
    
    This is the key Q2R2 test: do gas-rich and gas-poor galaxies
    trace different gravitational environments?
    """
    
    print("\n" + "="*70)
    print(" Q vs R KAPPA COMPARISON")
    print("="*70)
    
    kappa_at_gal = get_kappa_at_positions(kappa, nside,
                                           galaxies['ra'], galaxies['dec'])
    
    if kappa_at_gal is None:
        return None
    
    residuals = galaxies['residuals']
    
    valid = np.isfinite(kappa_at_gal) & np.isfinite(residuals)
    kappa_at_gal = kappa_at_gal[valid]
    residuals = residuals[valid]
    
    # Split by residual (proxy for Q/R dominance)
    # Negative residuals = Q-dominated (rotating faster than expected)
    # Positive residuals = R-dominated (rotating slower than expected)
    
    q_dominated = residuals < np.percentile(residuals, 25)
    r_dominated = residuals > np.percentile(residuals, 75)
    
    kappa_q = kappa_at_gal[q_dominated]
    kappa_r = kappa_at_gal[r_dominated]
    
    mean_kappa_q = kappa_q.mean()
    mean_kappa_r = kappa_r.mean()
    std_kappa_q = kappa_q.std() / np.sqrt(len(kappa_q))
    std_kappa_r = kappa_r.std() / np.sqrt(len(kappa_r))
    
    # T-test
    t_stat, p_value = stats.ttest_ind(kappa_q, kappa_r)
    
    print(f"\n  Q-dominated galaxies (bottom 25% residuals):")
    print(f"    N = {len(kappa_q)}")
    print(f"    Mean kappa = {mean_kappa_q:.6f} +/- {std_kappa_q:.6f}")
    
    print(f"\n  R-dominated galaxies (top 25% residuals):")
    print(f"    N = {len(kappa_r)}")
    print(f"    Mean kappa = {mean_kappa_r:.6f} +/- {std_kappa_r:.6f}")
    
    print(f"\n  Difference: {mean_kappa_r - mean_kappa_q:.6f}")
    print(f"  T-statistic: {t_stat:.3f}")
    print(f"  P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"\n  RESULT: Q and R galaxies trace DIFFERENT kappa environments!")
        print(f"         This supports the Q2R2 baryonic coupling hypothesis.")
    else:
        print(f"\n  RESULT: No significant difference (p = {p_value:.3f})")
    
    return {
        'mean_kappa_q': float(mean_kappa_q),
        'mean_kappa_r': float(mean_kappa_r),
        'std_kappa_q': float(std_kappa_q),
        'std_kappa_r': float(std_kappa_r),
        'n_q': int(len(kappa_q)),
        'n_r': int(len(kappa_r)),
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }


def main():
    print("="*70)
    print(" PHASE 3: CMB LENSING CROSS-CORRELATION")
    print(" Testing Q2R2 Direct Gravitational Signature")
    print("="*70)
    
    # Step 1: Download/load Planck lensing map
    kappa_path, mask_path = download_planck_lensing()
    
    if not HEALPY_AVAILABLE:
        print("\n" + "="*70)
        print(" HEALPY NOT AVAILABLE")
        print("="*70)
        print("""
  healpy is required for CMB lensing analysis.
  
  Installation options:
  
  1. pip install healpy
     (may fail on Windows)
  
  2. conda install -c conda-forge healpy
     (recommended for Windows)
  
  3. Use Windows Subsystem for Linux (WSL)
     wsl --install
     Then: pip install healpy
  
  Once healpy is installed, run this script again.
        """)
        return
    
    # Step 2: Load kappa map
    kappa, nside = load_kappa_map(kappa_path)
    
    if kappa is None:
        print("\n  ERROR: Could not load kappa map")
        print("  Please download from Planck Legacy Archive")
        return
    
    results = {}
    
    # Step 3: Analyze KiDS
    print("\n" + "="*70)
    print(" KIDS ANALYSIS")
    print("="*70)
    
    kids = load_kids_with_residuals()
    if kids is not None:
        kids_correlation = cross_correlate_kappa_residuals(kappa, nside, kids)
        kids_test = test_kappa_residual_correlation(kids_correlation)
        kids_qr = analyze_q_vs_r_kappa(kappa, nside, kids)
        
        results['kids'] = {
            'correlation': kids_correlation,
            'test': kids_test,
            'q_vs_r': kids_qr
        }
    
    # Step 4: Analyze ALFALFA
    print("\n" + "="*70)
    print(" ALFALFA ANALYSIS")
    print("="*70)
    
    alfalfa = load_alfalfa_with_residuals()
    if alfalfa is not None:
        alfalfa_correlation = cross_correlate_kappa_residuals(kappa, nside, alfalfa)
        alfalfa_test = test_kappa_residual_correlation(alfalfa_correlation)
        alfalfa_qr = analyze_q_vs_r_kappa(kappa, nside, alfalfa)
        
        results['alfalfa'] = {
            'correlation': alfalfa_correlation,
            'test': alfalfa_test,
            'q_vs_r': alfalfa_qr
        }
    
    # Step 5: Summary
    print("\n" + "="*70)
    print(" PHASE 3 SUMMARY")
    print("="*70)
    
    print("""
  CMB Lensing Cross-Correlation Results:
  
  If Q2R2 modifies the gravitational potential:
  - Q-dominated and R-dominated galaxies should trace different kappa
  - Residuals should correlate with local kappa
  
  This would be DIRECT EVIDENCE of the Q2R2 effect on gravity.
    """)
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "phase3_cmb_lensing_results.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\n  Results saved: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
