#!/usr/bin/env python3
"""
Diagnostic script to explore the actual structure of TNG HDF5 files.
"""

import h5py
import os
import numpy as np

# Paths - relative to Git folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
GIT_DIR = os.path.dirname(PROJECT_DIR)
BTFR_BASE = os.path.join(os.path.dirname(GIT_DIR), "BTFR")
TNG_PATHS = {
    'TNG50': os.path.join(BTFR_BASE, "TNG/Data_TNG50"),
    'TNG100': os.path.join(BTFR_BASE, "TNG/Data"),
    'TNG300': os.path.join(BTFR_BASE, "TNG/Data_TNG300"),
}

def explore_hdf5(filepath, max_depth=3, prefix=""):
    """Recursively explore HDF5 structure."""
    
    def print_attrs(obj, name):
        """Print attributes of an object."""
        if obj.attrs:
            for key, val in obj.attrs.items():
                print(f"{prefix}    @{key}: {val}")
    
    def explore_group(group, depth=0):
        """Recursively explore a group."""
        if depth > max_depth:
            return
        
        indent = "  " * depth
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                print(f"{indent}[GROUP] {key}/")
                explore_group(item, depth + 1)
            elif isinstance(item, h5py.Dataset):
                shape = item.shape
                dtype = item.dtype
                print(f"{indent}[DATASET] {key}: shape={shape}, dtype={dtype}")
                # Show first few values for small datasets
                if item.size < 10:
                    print(f"{indent}  values: {item[:]}")
    
    with h5py.File(filepath, 'r') as f:
        print(f"\n{'='*60}")
        print(f"FILE: {os.path.basename(filepath)}")
        print(f"{'='*60}")
        
        # Top-level attributes
        if f.attrs:
            print("\nFile Attributes:")
            for key, val in f.attrs.items():
                print(f"  @{key}: {val}")
        
        # Structure
        print("\nStructure:")
        explore_group(f)


def main():
    print("=" * 70)
    print("TNG HDF5 FILE STRUCTURE DIAGNOSTIC")
    print("=" * 70)
    
    for sim_name, sim_path in TNG_PATHS.items():
        print(f"\n\n{'#'*70}")
        print(f"# {sim_name}")
        print(f"{'#'*70}")
        
        if not os.path.exists(sim_path):
            print(f"  Path not found: {sim_path}")
            continue
        
        # Find first HDF5 file
        files = sorted([f for f in os.listdir(sim_path) if f.endswith('.hdf5')])
        
        if not files:
            print(f"  No HDF5 files found")
            continue
        
        # Explore first file
        first_file = os.path.join(sim_path, files[0])
        print(f"\nExploring: {files[0]}")
        explore_hdf5(first_file)
        
        # Also check a middle file (might have different structure)
        if len(files) > 10:
            mid_file = os.path.join(sim_path, files[len(files)//2])
            print(f"\n\nAlso checking middle file: {files[len(files)//2]}")
            explore_hdf5(mid_file)


if __name__ == "__main__":
    main()
