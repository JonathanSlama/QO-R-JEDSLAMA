# Data Sources - Paper 3

Instructions for obtaining the simulation data used to validate the QO+R theory.

**Author:** Jonathan Edouard SLAMA  
**Contact:** jonathan@metafund.in

---

## IllustrisTNG Simulations

**Description:** Cosmological magnetohydrodynamical simulations of galaxy formation.

**Source:** https://www.tng-project.org/

**Registration:** Create an account at https://www.tng-project.org/users/register/

---

## Setup API Key

```bash
# Linux/Mac
export TNG_API_KEY="your-api-key-here"

# Windows PowerShell
$env:TNG_API_KEY = "your-api-key-here"
```

---

## TNG100-1 (Recommended)

**Size:** ~2 GB for group catalogs  
**Galaxies:** ~10^5

```bash
cd tests/tng-simulations/
python download_tng_groupcat.py
```

---

## TNG300-1 (Large Volume)

**Size:** ~5 GB for group catalogs  
**Galaxies:** ~10^6

```bash
python download_tng300.py
```

---

## TNG50-1 (High Resolution)

**Size:** ~3 GB for group catalogs  
**Best for:** Detailed rotation curves

```bash
python download_tng50.py
```

---

## Storage Requirements

| Simulation | Size | Purpose |
|------------|------|---------|
| TNG100 | 2 GB | Primary validation |
| TNG300 | 5 GB | Statistical power |
| TNG50 | 3 GB | High resolution |

**Minimum:** 2 GB (TNG100 only)  
**Full validation:** ~10 GB

---

## Reference

Nelson et al. (2019), "The IllustrisTNG Simulations", ComAC, 6, 2

---

## Note

The HDF5 data files are **excluded from Git** due to their size. Only the download scripts and analysis code are version-controlled.
