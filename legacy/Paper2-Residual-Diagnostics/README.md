# Paper 2: Residual Structure in Clinical Biomarker Ratios

## Multi-Disease Validation Using NHANES 2017-2018 and Breast Cancer Coimbra

**Author:** Jonathan Edouard Slama
**Affiliation:** Metafund Research Division, Strasbourg, France
**Email:** jonathan@metafund.in
**ORCID:** 0009-0002-1292-4350

---

## Abstract

Clinical biomarker ratios (FIB-4, HOMA-IR, eGFR) are widely used for disease screening, yet 20-40% of patients fall into diagnostic "gray zones." We investigated whether residuals from these ratios contain systematic structure that could inform diagnosis.

### Key Results

| Metric | Value |
|--------|-------|
| **Significant residuals (H1)** | 72/85 (85%) |
| **U-shaped patterns (H2)** | 35 detected |
| **Cross-disease correlations (H3)** | 70/90 (78%) |
| **Universal residuals** | 12 (in 3+ diseases) |

### Disease-Specific Findings

| Category | H1 Support | U-shapes | Key Finding |
|----------|------------|----------|-------------|
| Hepatic Fibrosis | 76% | 26 | U-shapes in FIB-4 gray zone |
| Kidney Disease | 94% | 3 | Highest discrimination |
| Cardiovascular | 65% | 0 | No U-shapes |
| Diabetes | 88% | 6 | U-shapes in prediabetes |
| Metabolic Syndrome | 100% | 14 | All residuals significant |

---

## The Story

### What We Asked
Are residuals from clinical ratios truly random noise, or do they contain systematic structure?

### What We Found
1. **Residuals are NOT random** - 85% show significant disease-related structure
2. **U-shaped patterns exist** - Particularly in diagnostic gray zones (FIB-4 1.3-2.67)
3. **Patterns are universal** - 12 residuals significant across 3+ disease categories

### What This Means
The "unexplained variance" in clinical ratios may contain hidden diagnostic information, particularly for patients in gray zones where standard ratios are ambiguous.

### What This Does NOT Mean
- Clinical utility is NOT proven
- Predictive value is NOT established
- Biological mechanisms are NOT identified
- This should NOT be used clinically yet

---

## Repository Structure

```
Paper2-Residual-Diagnostics/
+-- manuscript/                      # Main manuscript folder
+-- figures/                         # Breast Cancer figures (10)
+-- data/
|   +-- raw/                        # Breast Cancer Coimbra data
|   +-- processed/                  # Computed ratios and residuals
+-- scripts/                        # Breast Cancer analysis (5 scripts)
+-- results/                        # Statistical outputs
+-- nhanes_extension/               # NHANES multi-disease analysis
|   +-- manuscript/
|   |   +-- paper2_residuals_v3.tex # * MAIN MANUSCRIPT (11 figures)
|   |   +-- *.pdf                   # Compiled versions
|   +-- figures/                    # NHANES figures (9)
|   +-- data/
|   |   +-- raw/                   # 17 NHANES .XPT files
|   |   +-- processed/             # Merged data with ratios
|   +-- scripts/                   # 10 analysis scripts
|   +-- results/                   # Disease-specific outputs
+-- README.md                       # This file
+-- README_REPRODUCIBILITY.md       # Detailed reproduction guide
+-- FIGURES_AUDIT.md               # Figure documentation
+-- requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
cd Paper2-Residual-Diagnostics
pip install -r requirements.txt

# 2. Run NHANES pipeline (main analysis)
cd nhanes_extension
python scripts/00_download_nhanes.py    # Download data (~15 min)
python scripts/01_merge_nhanes.py       # Merge files
python scripts/02_compute_all_ratios.py # Compute ratios + residuals
python scripts/03_hepatic_fibrosis.py   # Liver analysis
python scripts/04_kidney_disease.py     # Kidney analysis
python scripts/05_cardiovascular.py     # CV analysis
python scripts/06_diabetes.py           # Diabetes analysis
python scripts/07_metabolic_syndrome.py # MetS analysis
python scripts/08_cross_disease.py      # Cross-disease patterns
python scripts/09_summary_report.py     # Generate report

# 3. Compile manuscript
cd manuscript
pdflatex paper2_residuals_v3.tex
pdflatex paper2_residuals_v3.tex  # 2x for TOC
```

---

## Data Sources

| Dataset | N | Source | Access |
|---------|---|--------|--------|
| NHANES 2017-2018 | 9,254 | CDC | [Link](https://wwwn.cdc.gov/nchs/nhanes/) |
| Breast Cancer Coimbra | 116 | UCI ML Repository | [Link](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra) |

**All data is publicly available. No synthetic data used.**

---

## Connection to QO+R Framework

This paper is Part 2 of the QO+R research program:

| Paper | Domain | Key Finding |
|-------|--------|-------------|
| **Paper 1** | Astrophysics (BTFR) | U-shaped residuals in galaxy rotation curves |
| **Paper 2** | Medicine (Biomarkers) | U-shaped residuals in clinical ratio gray zones |
| Paper 3 | Climate (Future) | Planned |

The hypothesis: competing regulatory mechanisms leave mathematical signatures (U-shapes) in ratio residuals across domains.

---

## Manuscript

**Current version:** `nhanes_extension/manuscript/paper2_residuals_v3.tex`

### Figures included (11 total):

1. `fig01_distributions_by_group.png` - Breast Cancer distributions
2. `fig07_residual_distributions.png` - Breast Cancer residuals
3. `liver_fig01_fib4_distribution.png` - FIB-4 zones
4. `liver_fig02_residuals_by_zone.png` - Residuals by FIB-4 zone
5. `liver_fig03_ushapes_indeterminate.png` - Key finding: U-shapes
6. `kidney_fig01_egfr_acr_distribution.png` - Kidney markers
7. `cv_fig01_tg_hdl_by_status.png` - CV risk markers
8. `diabetes_fig01_glycemic_analysis.png` - Glycemic analysis
9. `mets_fig01_overview.png` - Metabolic syndrome
10. `cross_disease_fig01_overview.png` - Universal patterns
11. `summary_figure.png` - Overview

---

## Limitations

1. **Cross-sectional design** - No causation established
2. **Multiple comparisons** - False positives possible despite correction
3. **No external validation** - Patterns may be dataset-specific
4. **Proxy outcomes** - No gold standard (biopsy/imaging)
5. **U.S. population only** - May not generalize globally

---

## Citation

```bibtex
@article{slama2025residuals,
  author  = {Slama, Jonathan Edouard},
  title   = {Residual Structure in Clinical Biomarker Ratios:
             Evidence for Non-Random Patterns Across Disease Categories},
  journal = {arXiv preprint},
  year    = {2025}
}
```

---

## License

MIT License - See LICENSE file

---

*This is exploratory research generating hypotheses for future investigation.*
*Findings require clinical validation before any diagnostic application.*
