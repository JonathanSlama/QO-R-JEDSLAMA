#!/usr/bin/env python3
"""
Data Download Guide
===================
Instructions and utilities for obtaining observational data.

Author: Jonathan Édouard Slama
ORCID: 0009-0002-1292-4350
Date: 2024-12-06

NOTE: Most data must be downloaded manually from the sources below.
This script provides helper functions for data verification and preprocessing.
"""

import os
import sys

# =============================================================================
# DATA SOURCES
# =============================================================================

DATA_SOURCES = {
    'SPARC': {
        'url': 'http://astroweb.cwru.edu/SPARC/',
        'file': 'SPARC_Lelli2016c.mrt',
        'reference': 'Lelli, McGaugh & Schombert (2016), AJ 152, 157',
        'registration': False,
    },
    'ALFALFA': {
        'url': 'https://egg.astro.cornell.edu/alfalfa/data/',
        'file': 'a100.csv',
        'reference': 'Haynes et al. (2018), ApJ 861, 49',
        'registration': False,
    },
    'WALLABY': {
        'url': 'https://wallaby-survey.org/data/',
        'file': 'WALLABY_PDR2_SourceCatalogue.xml',
        'reference': 'Westmeier et al. (2022), PASA 39, e058',
        'registration': True,
    },
}

# =============================================================================
# FUNCTIONS
# =============================================================================

def print_download_instructions():
    """Print instructions for downloading each dataset."""
    
    print("="*70)
    print(" DATA DOWNLOAD INSTRUCTIONS")
    print(" Paper 4 - QO+R Validation")
    print("="*70)
    
    for name, info in DATA_SOURCES.items():
        print(f"\n{name}")
        print("-" * len(name))
        print(f"  URL: {info['url']}")
        print(f"  File: {info['file']}")
        print(f"  Reference: {info['reference']}")
        print(f"  Registration required: {'Yes' if info['registration'] else 'No'}")
    
    print("\n" + "="*70)
    print(" After downloading, place files in:")
    print("   data/observations/SPARC/")
    print("   data/observations/ALFALFA/")
    print("   data/observations/WALLABY/")
    print("="*70)


def verify_data_presence(base_dir):
    """Check which datasets are present."""
    
    print("\nVerifying data presence...")
    
    obs_dir = os.path.join(base_dir, "data", "observations")
    
    status = {}
    for name, info in DATA_SOURCES.items():
        data_path = os.path.join(obs_dir, name, info['file'])
        exists = os.path.exists(data_path)
        status[name] = exists
        symbol = "OK" if exists else "MISSING"
        print(f"  [{symbol}] {name}: {data_path}")
    
    return status


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print_download_instructions()
    
    # Get project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    print("\n")
    verify_data_presence(project_dir)
