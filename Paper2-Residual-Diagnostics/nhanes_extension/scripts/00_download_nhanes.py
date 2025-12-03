#!/usr/bin/env python3
"""
Paper 2 Extended: NHANES Data Download Script (Alternative Method)
The Revelatory Division - Multi-Disease Validation

Uses correct NHANES URL structure and alternative download methods.

Author: Jonathan Édouard Slama
Email: jonathan@metafund.in
Affiliation: Metafund Research Division, Strasbourg, France
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_DATA = PROJECT_DIR / "data" / "raw"

# CORRECT NHANES URL structure (verified December 2024)
# The files are actually at a different location than the documentation suggests
BASE_URLS = [
    "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018",
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles",
]

# Headers to mimic browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Referer': 'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx',
}

# Files needed - format: (key, filename variations to try)
NHANES_FILES = {
    "DEMO": ["DEMO_J.XPT", "DEMO_J.xpt", "P_DEMO.XPT"],
    "BIOPRO": ["BIOPRO_J.XPT", "BIOPRO_J.xpt"],
    "CBC": ["CBC_J.XPT", "CBC_J.xpt"],
    "TCHOL": ["TCHOL_J.XPT", "TCHOL_J.xpt"],
    "HDL": ["HDL_J.XPT", "HDL_J.xpt"],
    "TRIGLY": ["TRIGLY_J.XPT", "TRIGLY_J.xpt"],
    "GLU": ["GLU_J.XPT", "GLU_J.xpt"],
    "INS": ["INS_J.XPT", "INS_J.xpt"],
    "GHB": ["GHB_J.XPT", "GHB_J.xpt"],
    "ALB_CR": ["ALB_CR_J.XPT", "ALB_CR_J.xpt"],
    "BMX": ["BMX_J.XPT", "BMX_J.xpt"],
    "BPX": ["BPX_J.XPT", "BPX_J.xpt"],
    "DIQ": ["DIQ_J.XPT", "DIQ_J.xpt"],
    "MCQ": ["MCQ_J.XPT", "MCQ_J.xpt"],
    "KIQ_U": ["KIQ_U_J.XPT", "KIQ_U_J.xpt"],
    "SMQ": ["SMQ_J.XPT", "SMQ_J.xpt"],
    "ALQ": ["ALQ_J.XPT", "ALQ_J.xpt"],
}

def is_valid_xpt(filepath: Path) -> bool:
    """Check if file is a valid XPT (not HTML error page)."""
    if not filepath.exists():
        return False
    
    # Check size (real XPT files are typically > 50KB)
    if filepath.stat().st_size < 50000:
        return False
    
    # Check first bytes - XPT files start with specific header
    with open(filepath, 'rb') as f:
        header = f.read(100)
        # XPT files typically start with 'HEADER RECORD' or similar
        # HTML files start with '<!DOCTYPE' or '<html'
        if b'<!DOCTYPE' in header or b'<html' in header or b'<HTML' in header:
            return False
    
    return True

def try_download(url: str, filepath: Path) -> bool:
    """Attempt to download a file and verify it's valid."""
    try:
        session = requests.Session()
        session.headers.update(HEADERS)
        
        response = session.get(url, timeout=60, allow_redirects=True)
        
        if response.status_code != 200:
            return False
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'html' in content_type:
            return False
        
        # Check size
        if len(response.content) < 50000:
            return False
        
        # Check content - not HTML
        if b'<!DOCTYPE' in response.content[:500] or b'<html' in response.content[:500]:
            return False
        
        # Save file
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        return True
        
    except Exception as e:
        return False

def main():
    """Main download function."""
    
    print("=" * 70)
    print("PAPER 2 EXTENDED: NHANES DATA DOWNLOAD")
    print("The Revelatory Division - Multi-Disease Validation")
    print("=" * 70)
    print()
    print("Author: Jonathan Édouard Slama")
    print("Affiliation: Metafund Research Division, Strasbourg, France")
    print()
    
    # Create directory
    RAW_DATA.mkdir(parents=True, exist_ok=True)
    
    # Remove invalid files first
    print("Cleaning up invalid files...")
    for f in RAW_DATA.glob("*.XPT"):
        if not is_valid_xpt(f):
            print(f"  Removing invalid: {f.name}")
            f.unlink()
    for f in RAW_DATA.glob("*.xpt"):
        if not is_valid_xpt(f):
            print(f"  Removing invalid: {f.name}")
            f.unlink()
    print()
    
    # Try to download each file
    print("=" * 70)
    print("ATTEMPTING DOWNLOADS")
    print("=" * 70)
    
    successful = []
    failed = []
    
    for key, filenames in NHANES_FILES.items():
        print(f"\n{key}:")
        
        # Check if we already have a valid file
        for fn in filenames:
            fp = RAW_DATA / fn
            if is_valid_xpt(fp):
                size_mb = fp.stat().st_size / (1024 * 1024)
                print(f"  ✓ Already have valid: {fn} ({size_mb:.2f} MB)")
                successful.append(fn)
                break
        else:
            # Try each URL/filename combination
            downloaded = False
            for base_url in BASE_URLS:
                if downloaded:
                    break
                for fn in filenames:
                    url = f"{base_url}/{fn}"
                    filepath = RAW_DATA / filenames[0]  # Use first filename as standard
                    
                    print(f"  Trying: {url}")
                    
                    if try_download(url, filepath):
                        size_mb = filepath.stat().st_size / (1024 * 1024)
                        print(f"  ✓ Downloaded: {filepath.name} ({size_mb:.2f} MB)")
                        successful.append(filepath.name)
                        downloaded = True
                        break
                    
                    time.sleep(0.5)
            
            if not downloaded:
                print(f"  ✗ All attempts failed")
                failed.append(key)
    
    # Summary
    print()
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"\n✓ Successful: {len(successful)}/{len(NHANES_FILES)}")
    print(f"✗ Failed: {len(failed)}/{len(NHANES_FILES)}")
    
    if failed:
        print(f"\nFailed files: {', '.join(failed)}")
    
    # List valid files
    print()
    print("=" * 70)
    print("VALID FILES IN DATA DIRECTORY")
    print("=" * 70)
    
    valid_files = [f for f in RAW_DATA.glob("*.[Xx][Pp][Tt]") if is_valid_xpt(f)]
    total_size = sum(f.stat().st_size for f in valid_files)
    
    for f in sorted(valid_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.2f} MB")
    
    print(f"\nTotal: {len(valid_files)} valid files, {total_size / (1024*1024):.1f} MB")
    
    # Instructions if download failed
    if len(valid_files) < 10:
        print()
        print("=" * 70)
        print("MANUAL DOWNLOAD REQUIRED")
        print("=" * 70)
        print()
        print("The CDC is blocking automated downloads.")
        print("Please download files manually:")
        print()
        print("1. Go to: https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Demographics&CycleBeginYear=2017")
        print()
        print("2. Download each of these files (click the XPT link):")
        print("   - Demographics: DEMO_J.XPT")
        print("   - Standard Biochemistry: BIOPRO_J.XPT") 
        print("   - Complete Blood Count: CBC_J.XPT")
        print("   - Cholesterol - Total: TCHOL_J.XPT")
        print("   - Cholesterol - HDL: HDL_J.XPT")
        print("   - Triglycerides: TRIGLY_J.XPT")
        print("   - Plasma Fasting Glucose: GLU_J.XPT")
        print("   - Insulin: INS_J.XPT")
        print("   - Glycohemoglobin: GHB_J.XPT")
        print("   - Albumin & Creatinine - Urine: ALB_CR_J.XPT")
        print("   - Body Measures: BMX_J.XPT")
        print("   - Blood Pressure: BPX_J.XPT")
        print("   - Diabetes: DIQ_J.XPT")
        print("   - Medical Conditions: MCQ_J.XPT")
        print("   - Kidney Conditions - Urine: KIQ_U_J.XPT")
        print("   - Smoking: SMQ_J.XPT")
        print("   - Alcohol Use: ALQ_J.XPT")
        print()
        print(f"3. Save all files to: {RAW_DATA}")
        print()
        print("4. Then run: python scripts/01_merge_nhanes.py")
        print()
        print("Direct links to NHANES data pages:")
        print("  Demographics: https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.htm")
        print("  Laboratory:   https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory&CycleBeginYear=2017")
        print("  Examination:  https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&CycleBeginYear=2017")
        print("  Questionnaire: https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&CycleBeginYear=2017")
    else:
        print()
        print("=" * 70)
        print("✓ Download successful!")
        print()
        print("Next step: python scripts/01_merge_nhanes.py")
        print("=" * 70)

if __name__ == "__main__":
    main()
