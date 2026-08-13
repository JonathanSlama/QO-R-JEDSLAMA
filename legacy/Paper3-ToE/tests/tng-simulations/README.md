# IllustrisTNG Simulation Tests

## Data Download and Analysis Scripts

**Author:** Jonathan Edouard SLAMA  
**Contact:** jonathan@metafund.in

---

## Overview

This directory contains scripts to validate QO+R predictions using the IllustrisTNG cosmological simulations. The simulation data files (HDF5 format) are NOT included in the Git repository due to their large size.

---

## Prerequisites

1. **TNG Account:** Register at https://www.tng-project.org/users/register/
2. **API Key:** Obtain your API key from the TNG website
3. **Python packages:**
   ```bash
   pip install numpy h5py requests pandas matplotlib
   ```

---

## Data Download

### Step 1: Set API Key

```bash
# Linux/Mac
export TNG_API_KEY="your-api-key-here"

# Windows PowerShell
$env:TNG_API_KEY = "your-api-key-here"

# Windows CMD
set TNG_API_KEY=your-api-key-here
```

### Step 2: Download Group Catalogs

```bash
# TNG100-1 (recommended for first tests)
python download_tng_groupcat.py

# TNG300-1 (larger volume, more statistics)
python download_tng300.py

# TNG50-1 (highest resolution)
python download_tng50.py
```

### Expected Download Sizes

| Simulation | Snapshot 99 (z=0) | Time |
|------------|-------------------|------|
| TNG100-1 | ~2 GB | ~30 min |
| TNG300-1 | ~5 GB | ~1 hour |
| TNG50-1 | ~3 GB | ~45 min |

---

## Directory Structure After Download

```
tng-simulations/
|
|-- Data/                          # TNG100 data (gitignored)
|   +-- fof_subhalo_tab_099.*.hdf5
|
|-- Data_TNG300/                   # TNG300 data (gitignored)
|   +-- fof_subhalo_tab_099.*.hdf5
|
|-- Data_TNG50/                    # TNG50 data (gitignored)
|   +-- fof_subhalo_tab_099.*.hdf5
|
|-- download_tng_groupcat.py       # Download scripts
|-- download_tng300.py
|-- download_tng50.py
|
|-- test_qor_on_tng.py             # Main analysis
|-- tng_ushape_test.py             # U-shape test
|-- calibrate_qor_parameters.py    # Parameter calibration
+-- visualize_ushape.py            # Visualization
```

---

## Running the Tests

### Basic QO+R Test
```bash
python test_qor_on_tng.py
```

### U-Shape Analysis
```bash
python tng_ushape_test.py
```

### Full Analysis Pipeline
```bash
python tng_full_ushape_analysis.py
```

---

## Expected Results

If QO+R is correct, you should observe:

1. **U-shaped residuals** in BTFR when stratified by gas fraction
2. **Environment dependence** correlating with local density
3. **Consistency** with SPARC observational results

---

## Troubleshooting

### "API key not found"
Make sure `TNG_API_KEY` is set in your environment.

### "Connection timeout"
TNG servers can be slow. Try:
```python
requests.get(url, timeout=300)  # 5 minute timeout
```

### "HDF5 file corrupted"
Re-download the file:
```bash
rm Data/fof_subhalo_tab_099.X.hdf5
python download_tng_groupcat.py
```

---

## References

- Nelson et al. (2019), "The IllustrisTNG Simulations", ComAC, 6, 2
- Pillepich et al. (2018), "First results from IllustrisTNG", MNRAS, 475, 648
