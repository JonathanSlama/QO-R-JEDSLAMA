# Data Sources - Paper 1

Instructions for obtaining the datasets used in this paper.

**Author:** Jonathan Edouard SLAMA  
**Contact:** jonathan@metafund.in

---

## 1. SPARC Database (Primary Dataset)

**Description:** Spitzer Photometry and Accurate Rotation Curves - 175 galaxies with high-quality rotation curves.

**Source:** http://astroweb.cwru.edu/SPARC/

**Download:**
```bash
cd data/
wget http://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt
```

**Reference:** Lelli, McGaugh, Schombert (2016), AJ, 152, 157

---

## 2. ALFALFA Survey (Replication Test)

**Description:** Arecibo Legacy Fast ALFA survey - HI 21cm observations of ~31,000 galaxies.

**Source:** http://egg.astro.cornell.edu/alfalfa/data/

**Download:**
```bash
cd data/
wget http://egg.astro.cornell.edu/alfalfa/data/a100.code12.table2.190808.csv
```

**Reference:** Haynes et al. (2018), ApJ, 861, 49

---

## 3. Little THINGS (Replication Test)

**Description:** Local Irregulars That Trace Luminosity Extremes - dwarf irregular galaxies.

**Source:** https://science.nrao.edu/science/surveys/littlethings

**Reference:** Hunter et al. (2012), AJ, 144, 134

---

## Quick Setup

```bash
#!/bin/bash
mkdir -p data
cd data

# SPARC (required, ~100 KB)
wget -q http://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt
echo "SPARC downloaded"

# ALFALFA (optional, ~5 MB)
wget -q http://egg.astro.cornell.edu/alfalfa/data/a100.code12.table2.190808.csv
echo "ALFALFA downloaded"
```

---

## Storage Requirements

| Dataset | Size | Required |
|---------|------|----------|
| SPARC | 100 KB | Yes |
| ALFALFA | 5 MB | Optional |
| Little THINGS | 2 MB | Optional |

**Minimum:** ~100 KB (SPARC only)
