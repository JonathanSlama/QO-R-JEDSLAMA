#!/usr/bin/env python3
"""
01_download_data.py
===================
Download the Breast Cancer Coimbra dataset from UCI ML Repository.

Author: Jonathan Édouard Slama
Email: jonathan@metafund.in
Date: December 2025

Dataset Reference:
    Patrício, M., Pereira, J., Crisóstomo, J., Matafome, P., Gomes, M., 
    Seiça, R., & Caramelo, F. (2018). 
    Using Resistin, glucose, age and BMI to predict the presence of breast cancer. 
    BMC Cancer, 18(1), 29.

License: CC BY 4.0
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Output directories
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def setup_directories():
    """Create data directories if they don't exist."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    print(f"✓ Data directories ready")


def download_breast_cancer_coimbra():
    """
    Download Breast Cancer Coimbra dataset from UCI ML Repository.
    
    Returns:
        tuple: (features DataFrame, targets DataFrame, metadata dict)
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print("ERROR: ucimlrepo not installed. Run: pip install ucimlrepo")
        sys.exit(1)
    
    print("Downloading Breast Cancer Coimbra dataset from UCI ML Repository...")
    print("Dataset ID: 451")
    print("Source: https://archive.ics.uci.edu/dataset/451/breast+cancer+coimbra")
    print()
    
    # Fetch dataset
    dataset = fetch_ucirepo(id=451)
    
    # Extract components
    X = dataset.data.features
    y = dataset.data.targets
    metadata = dataset.metadata
    variables = dataset.variables
    
    print(f"✓ Download complete")
    print(f"  - Features shape: {X.shape}")
    print(f"  - Targets shape: {y.shape}")
    print(f"  - Variables: {list(X.columns)}")
    print()
    
    return X, y, metadata, variables


def save_data(X, y, metadata, variables):
    """Save downloaded data to CSV files."""
    
    # Save features
    features_path = DATA_RAW / "breast_cancer_coimbra_features.csv"
    X.to_csv(features_path, index=False)
    print(f"✓ Features saved to: {features_path}")
    
    # Save targets
    targets_path = DATA_RAW / "breast_cancer_coimbra_targets.csv"
    y.to_csv(targets_path, index=False)
    print(f"✓ Targets saved to: {targets_path}")
    
    # Save combined dataset
    import pandas as pd
    combined = pd.concat([X, y], axis=1)
    combined_path = DATA_RAW / "breast_cancer_coimbra_full.csv"
    combined.to_csv(combined_path, index=False)
    print(f"✓ Combined dataset saved to: {combined_path}")
    
    # Save metadata
    metadata_path = DATA_RAW / "metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write("BREAST CANCER COIMBRA DATASET - METADATA\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Download date: {datetime.now().isoformat()}\n")
        f.write(f"Source: UCI ML Repository (ID: 451)\n")
        f.write(f"URL: https://archive.ics.uci.edu/dataset/451/breast+cancer+coimbra\n\n")
        
        f.write("DATASET DESCRIPTION:\n")
        f.write("-" * 30 + "\n")
        if hasattr(metadata, 'description') and metadata.get('description'):
            f.write(str(metadata.get('description', 'N/A')) + "\n\n")
        
        f.write("VARIABLES:\n")
        f.write("-" * 30 + "\n")
        f.write(variables.to_string() + "\n\n")
        
        f.write("CITATION:\n")
        f.write("-" * 30 + "\n")
        f.write("Patrício, M., Pereira, J., Crisóstomo, J., Matafome, P., Gomes, M.,\n")
        f.write("Seiça, R., & Caramelo, F. (2018).\n")
        f.write("Using Resistin, glucose, age and BMI to predict the presence of breast cancer.\n")
        f.write("BMC Cancer, 18(1), 29.\n")
        f.write("DOI: 10.1186/s12885-017-3877-1\n\n")
        
        f.write("LICENSE:\n")
        f.write("-" * 30 + "\n")
        f.write("CC BY 4.0 (Creative Commons Attribution 4.0 International)\n")
    
    print(f"✓ Metadata saved to: {metadata_path}")
    
    return combined_path


def print_data_summary(X, y):
    """Print summary statistics of the downloaded data."""
    import pandas as pd
    
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    
    # Target distribution
    print("\nTarget Variable (Classification):")
    print("-" * 30)
    target_counts = y.value_counts()
    print(f"  1 = Healthy controls: {target_counts.get(1, 0)}")
    print(f"  2 = Cancer patients:  {target_counts.get(2, 0)}")
    print(f"  Total: {len(y)}")
    
    # Feature statistics
    print("\nFeature Statistics:")
    print("-" * 30)
    print(X.describe().round(2).to_string())
    
    # Missing values
    print("\nMissing Values:")
    print("-" * 30)
    missing = X.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values ✓")
    else:
        print(missing[missing > 0])
    
    print("\n" + "=" * 60)


def main():
    """Main execution function."""
    print("=" * 60)
    print("PAPER 2: DATA DOWNLOAD SCRIPT")
    print("The Revelatory Division - Residual Diagnostics")
    print("=" * 60)
    print()
    
    # Setup
    setup_directories()
    
    # Download
    X, y, metadata, variables = download_breast_cancer_coimbra()
    
    # Save
    combined_path = save_data(X, y, metadata, variables)
    
    # Summary
    print_data_summary(X, y)
    
    print("\n✓ Data download complete!")
    print(f"  Data location: {DATA_RAW}")
    print("\nNext step: Run 02_explore_data.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
