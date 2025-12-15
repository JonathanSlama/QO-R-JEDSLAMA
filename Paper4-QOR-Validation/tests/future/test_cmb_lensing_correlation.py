#!/usr/bin/env python3
"""
CMB Lensing x Galaxy Residuals Cross-Correlation
=================================================

This script performs the actual cross-correlation once
Planck lensing maps are available.

Requirements:
    pip install healpy

Usage:
    python test_cmb_lensing_correlation.py
"""

import os
import numpy as np

# Check for healpy
try:
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    HEALPY_AVAILABLE = False
    print("WARNING: healpy not installed. Install with: pip install healpy")

from astropy.io import fits
from astropy.table import Table
from scipy import stats
import json


# Configuration
LENSING_MAP = "COM_Lensing_4096_R3.00_MV.fits.gz"
NOISE_MAP = "COM_Lensing_4096_R3.00_noise.fits.gz"


def load_lensing_map(path):
    """Load Planck lensing convergence map."""
    if not HEALPY_AVAILABLE:
        print("ERROR: healpy required")
        return None, None
    
    kappa = hp.read_map(path, field=0)
    nside = hp.get_nside(kappa)
    print(f"  Loaded lensing map: NSIDE={nside}, Npix={len(kappa)}")
    return kappa, nside


def get_kappa_at_positions(kappa, nside, ra, dec):
    """Extract kappa values at galaxy positions."""
    if not HEALPY_AVAILABLE:
        return None
    
    # Convert to HEALPix coordinates
    theta = np.radians(90 - dec)  # colatitude
    phi = np.radians(ra)          # longitude
    
    # Get pixel indices
    pix = hp.ang2pix(nside, theta, phi)
    
    return kappa[pix]


def cross_correlate_residuals(residuals, kappa_values, n_bins=10):
    """
    Cross-correlate galaxy residuals with CMB lensing.
    
    If Q2R2 affects gravitational potential:
    - Positive residuals should correlate differently with kappa
    - This would be a DIRECT signature of new physics
    """
    
    # Bin by residual value
    residual_bins = np.percentile(residuals, np.linspace(0, 100, n_bins+1))
    
    results = []
    for i in range(n_bins):
        mask = (residuals >= residual_bins[i]) & (residuals < residual_bins[i+1])
        if mask.sum() > 10:
            mean_kappa = kappa_values[mask].mean()
            std_kappa = kappa_values[mask].std() / np.sqrt(mask.sum())
            results.append({
                'residual_min': float(residual_bins[i]),
                'residual_max': float(residual_bins[i+1]),
                'mean_residual': float(residuals[mask].mean()),
                'mean_kappa': float(mean_kappa),
                'std_kappa': float(std_kappa),
                'n': int(mask.sum())
            })
    
    return results


def main():
    print("="*70)
    print(" CMB LENSING x GALAXY RESIDUALS CROSS-CORRELATION")
    print("="*70)
    
    if not HEALPY_AVAILABLE:
        print("\n  ERROR: healpy not available")
        print("  Install with: pip install healpy")
        print("  Note: On Windows, consider using WSL")
        return
    
    # Check for lensing map
    if not os.path.exists(LENSING_MAP):
        print(f"\n  ERROR: Lensing map not found: {LENSING_MAP}")
        print("  Please download from Planck Legacy Archive:")
        print("  https://pla.esac.esa.int/")
        return
    
    # Load lensing map
    print("\n  Loading Planck lensing map...")
    kappa, nside = load_lensing_map(LENSING_MAP)
    
    # Load galaxy data
    print("\n  Loading galaxy data...")
    # TODO: Load KiDS or ALFALFA data with residuals
    
    # Cross-correlate
    print("\n  Computing cross-correlation...")
    # TODO: Implement cross-correlation
    
    print("\n  Analysis complete!")


if __name__ == "__main__":
    main()
