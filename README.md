# QO+R Framework

## A Two-Field Phenomenological Modified-Gravity Hypothesis

**Author:** Jonathan Édouard Slama
**Affiliation:** Metafund Research Division, Strasbourg, France
**Contact:** [jonathan.slama@outlook.fr](mailto:jonathan.slama@outlook.fr) · [jonathan@metafund.in](mailto:jonathan@metafund.in)
**ORCID:** [0009-0002-1292-4350](https://orcid.org/0009-0002-1292-4350)
**Version:** 3.1 (May 2026)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17806441.svg)](https://doi.org/10.5281/zenodo.17806441)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Concept DOI (all versions, always resolves to latest):** [10.5281/zenodo.17806441](https://doi.org/10.5281/zenodo.17806441)
> **Version 3.1 DOI (this release):** [10.5281/zenodo.20153973](https://doi.org/10.5281/zenodo.20153973)
> **Version 3.0 DOI (December 2025):** [10.5281/zenodo.17806442](https://doi.org/10.5281/zenodo.17806442)

---

## Overview

This repository documents the **QO+R framework**: a two-field phenomenological
modified-gravity hypothesis, with empirical analyses on ~708,000 galaxies
across multiple datasets and a tentative theoretical embedding in
Type IIB string theory.

The framework provides a phenomenological description of a non-monotonic
environmental dependence detected in Baryonic Tully–Fisher residuals.
The string-theory embedding remains exploratory, and the observed empirical
pattern may admit alternative single-field, baryonic, or population-mixing
explanations that have not been exhaustively ruled out. Independent
investigation is encouraged.

---

## 📋 Status of the papers in this repository

| Paper | Title (short) | Status | Datasets | Key result |
|---|---|---|---|---|
| **Paper 1** | A non-monotonic environmental trend in BTFR residuals | **Major revision** at *Scientific Reports* (submission #2025-12-33523) | SPARC, ALFALFA, Little THINGS, IllustrisTNG | Non-monotonic trend detected in SPARC: *a* = +1.33 ± 0.25, *p* < 10⁻⁶ |
| **Paper 2** | Residual structure in clinical biomarker ratios | Methodology extension; not a clinical validation | NHANES 2017–2018 (*N* = 9,254), Breast Cancer Coimbra (*N* = 116) | 72/85 disease–residual combinations show significant distribution differences |
| **Paper 3** | From string theory to galactic observations | Theoretical companion (v2, tentative embedding) | Type IIB supergravity → CY₃ (quintic 𝐏⁴[5]) | λ_QR ~ 𝒪(1) emerges from KKLT-style moduli stabilisation |
| **Paper 4** | QO+R validation suite (experimental) | **Experimental, not part of v3.1 Zenodo deposit** | Multiple datasets (TNG, KiDS, Planck, Gaia, WALLABY) | Internal validation work; claims have not been harmonised with the prudent tonal framework of the Scientific Reports revision |

> ⚠️ **Note on Paper 4.** Paper 4 (`Paper4-QOR-Validation/`) contains earlier-version
> experimental work whose claims (e.g. a "conservation law of gravity", "26σ
> significance", "alternative theories eliminated") have **not** been
> harmonised with the prudent tonal framework adopted in the v3.1 revision
> of Paper 1 and the response letters to *Scientific Reports*. It is
> preserved in the repository for transparency and to document the research
> trajectory, but should not be relied upon as part of the v3.1 scientific
> content. A future v3.2 release will either retire or substantially revise
> Paper 4.

---

## 🆕 What's new in v3.1 (May 2026)

This release reflects the substantial revision of **Paper 1** during the
major-revision process at *Scientific Reports*.

### Tonal harmonisation in Paper 1

Approximately forty individual claims have been softened in the manuscript,
supplements, and combined documentation. Representative examples:

- *"discovery"* → *"detection"* / *"evidence for"* in section headings, captions, and the summary;
- *"confirmed"* in IllustrisTNG → *"recovered"* / *"tested"* / *"supported"* (depending on context);
- *"independent replication in ALFALFA"* → *"complementary support from ALFALFA"*;
- *"killer prediction"* → *"discriminating prediction"* (throughout);
- *"robustly required by the data"* → *"strongly favoured by the data within the tested model classes"*;
- *"no single-field theory could reproduce"* → *"not naturally produced by the simple single-field model tested in the previous section"*;
- *"a discriminating signature of the two-field structure"* → *"difficult to accommodate within the simple monotonic single-field model considered here … should not be interpreted as excluding all possible single-field, baryonic, or population-mixing alternatives"*;
- *"the model was falsified"* → *"the simple monotonic single-field model was disfavoured by the data"*;
- *"consistent with all existing precision tests"* → *"not in obvious conflict with the precision tests considered here; a full constraint analysis across other regimes remains necessary"*.

### Three new supplementary analyses (added to the *Scientific Reports* submission)

- **S1 — Environmental-proxy sensitivity to scale.** Fixed-aperture 2MRS density proxies at six radii (1, 2, 3, 5, 7, 10 Mpc) compared with the catalog-based structural classification.
- **S2 — Formal model comparison.** Five models (flat / linear / quadratic / broken-line / cubic spline) compared on the SPARC sample using AIC, BIC, and Akaike weights. Non-monotonic models strongly favoured over monotonic alternatives (Δ AIC ≈ 39.9 between best non-monotonic model and the linear model).
- **S3 — Isolation-controlled re-analysis in IllustrisTNG.** Stratification by 5th-nearest-neighbour distance within the R-dominated sample (*N* = 9,386). Sign inversion concentrated in the densest quartile (Q1: 5.2σ); disappears in the most isolated quartile (Q4: non-significant).

### Numerical consistency improvements

- **SPARC subsample sizes** (*N* = 181 / 175 / 169) used in different analyses are now explicitly distinguished;
- **SPARC curvature coefficient** harmonised to *a* = +1.33 ± 0.25 throughout (previously partially +1.36 ± 0.24);
- **All references to "26σ"** in TNG removed in favour of the more conservative
  isolation-controlled **5.3σ** (full R-dominated sample) and **5.2σ** (densest
  quartile) figures.

### Title change in Paper 1

- **v3.0 (December 2025):** *"A non-monotonic environmental signature in galaxy dynamics suggests dual scalar fields"*
- **v3.1 (May 2026):** *"A non-monotonic environmental trend in Baryonic Tully–Fisher residuals: empirical evidence and a two-field phenomenological interpretation"*

---

## 📖 Reading order

If you are coming from the *Scientific Reports* peer review and want to
quickly navigate the v3.1 material:

| Document | Where to find it |
|---|---|
| **Revised Paper 1 manuscript (PDF)** | `../Review_Scientific_report/09_Manuscrit_Revise_Final/NATURE_ARTICLE_revised.pdf` |
| **Supplementary S1, S2, S3 (PDF)** | `../Review_Scientific_report/09_Manuscrit_Revise_Final/supplementary/supplementary_material.pdf` |
| **Cover letter to the Editor** | `../Review_Scientific_report/09_Manuscrit_Revise_Final/response_letter/cover_letter.pdf` |
| **Response to Referee 1** | `../Review_Scientific_report/09_Manuscrit_Revise_Final/response_letter/response_letter_R1.pdf` |
| **Response to Referee 2** | `../Review_Scientific_report/09_Manuscrit_Revise_Final/response_letter/response_letter_R2.pdf` |
| **Combined Papers 1+2+3 (single PDF, v3.1)** | `combined_papers/qor_combined_papers_v3_1.pdf` |
| **Paper 1 v3.1 standalone** | `Paper1-BTFR-UShape/manuscript/reviewed_version/paper1_qor_btfr_v3_1.pdf` |

---

## Repository structure

```
QO-R-JEDSLAMA/
├── README.md                       # This file
├── DATA_SOURCES.md                 # How to obtain all datasets
├── PUBLICATION_AUDIT.md            # Quality checklist
├── LICENSE                         # MIT License
├── requirements.txt                # Python dependencies
│
├── Paper1-BTFR-UShape/             # Paper 1: BTFR environmental trend
│   ├── manuscript/                 # v3.0 LaTeX + PDF
│   │   └── reviewed_version/       # v3.1 (current, harmonised tone)
│   ├── figures/                    # 13 publication figures
│   ├── data/                       # SPARC + ALFALFA + 2MRS processed
│   └── tests/                      # 13 reproducibility scripts
│
├── Paper2-Residual-Diagnostics/    # Paper 2: methodology extension
│   ├── nhanes_extension/           # NHANES analysis + manuscript
│   ├── figures/                    # Coimbra figures
│   └── data/                       # Clinical datasets
│
├── Paper3-ToE/                     # Paper 3: theoretical companion
│   ├── manuscript/                 # LaTeX + PDF (v2)
│   ├── figures/                    # 8 figures
│   └── tests/                      # TNG-300 validation scripts
│
├── Paper4-QOR-Validation/          # Paper 4: experimental, see warning above
│   ├── manuscript/                 # LaTeX + PDF (not harmonised v3.1)
│   ├── experimental/               # Research documentation
│   ├── tests/                      # 14-test validation suite
│   └── data/                       # Multi-context datasets
│
└── combined_papers/                # Single combined PDF of Papers 1+2+3 (v3.1)
    ├── qor_combined_papers_v3_1.tex
    ├── qor_combined_papers_v3_1.pdf
    ├── body_paper1.tex
    ├── body_paper2.tex
    ├── body_paper3.tex
    └── figures/                    # Mirror of all figures organised by paper
```

---

## Data sources used in v3.1

| Dataset | Type | *N* used | Source | Papers |
|---|---|---|---|---|
| **SPARC** | Resolved kinematics | 181 (full) / 175 (env-classified) / 169 (2MRS cross-matched) | Lelli et al. 2016 | 1, 3, 4 |
| **ALFALFA α.100** | H I linewidths | 21,834 | Haynes et al. 2018 | 1, 3, 4 |
| **Little THINGS** | Dwarf irregulars | 40 | Hunter et al. 2012 | 1 |
| **IllustrisTNG 50/100/300** | Cosmological simulation | ~685,000 (combined) | Pillepich et al. 2018 + IllustrisTNG | 1, 3, 4 |
| **2MRS** | Galaxy environments | as catalogue | Huchra et al. 2012 | 1 (S1) |
| **NHANES 2017–2018** | Clinical | 9,254 | CDC | 2 |
| **Breast Cancer Coimbra** | Clinical | 116 | Patricio et al. 2018 | 2 |
| **WALLABY, KiDS DR4, Planck PSZ2, Gaia DR3** | Various | varies | ASKAP / ESO / ESA | 4 (experimental) |

See [DATA_SOURCES.md](DATA_SOURCES.md) for complete download instructions.

---

## Quick start

### 1. Clone the repository
```bash
git clone https://github.com/JonathanSlama/QO-R-JEDSLAMA.git
cd QO-R-JEDSLAMA
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Reproduce the Paper 1 SPARC detection
```bash
cd Paper1-BTFR-UShape/tests/03_ushape_discovery
python discover_ushape.py
```

### 4. Compile the combined Papers 1+2+3 (v3.1)
```bash
cd combined_papers
pdflatex qor_combined_papers_v3_1.tex
pdflatex qor_combined_papers_v3_1.tex   # Second pass for TOC + bibliography
```

### 5. Compile the *Scientific Reports* revised manuscript and supplements

```bash
cd ../Review_Scientific_report/09_Manuscrit_Revise_Final
pdflatex NATURE_ARTICLE_revised.tex
pdflatex NATURE_ARTICLE_revised.tex

cd supplementary
pdflatex supplementary_material.tex
pdflatex supplementary_material.tex
```

---

## System requirements

### Software dependencies
- Python 3.9+ (tested on 3.9, 3.10, 3.11)
- LaTeX distribution (TeX Live 2023+ or MiKTeX) for manuscript compilation
- See `requirements.txt` for the full Python package list

### Operating systems tested
- Windows 10/11 (64-bit)
- Ubuntu 20.04/22.04
- macOS 12+ (Intel and Apple Silicon)

### Hardware
- Standard desktop, 8 GB RAM, ~5 GB disk space for full datasets
- No GPU required

### Reproduction time (approximate)
- Quick SPARC reproduction: ~30 seconds
- Full validation suite (Paper 1 + 3): ~5 minutes
- Complete reproduction with all downloads: ~2 hours

---

## Theoretical framework (Paper 3 summary)

### The QO+R Lagrangian (effective 4D action)

```
L = √(-g) [ (M_P²/2) F(χ) R − (1/2)(∂φ)² − (Z(χ)/2)(∂χ)² − V(φ,χ) ]
  + L_matter[ A²(φ,χ) g_μν , ψ ]
```

Where:
- **φ** (Q-field): coupled preferentially to gas (electromagnetic);
- **χ** (R-field): coupled preferentially to stars (gravitational);
- **A(φ,χ) = exp[ β (φ − κχ)/M_P ]**: conformal coupling to the matter sector.

### Tentative string-theory embedding (Paper 3, v2)

```
10D Type IIB supergravity
         │
         ▼
Calabi–Yau compactification on the quintic 𝐏⁴[5]
         │
         ▼
4D effective theory : dilaton φ ↔ Q ; Kähler modulus T ↔ R
         │
         ▼
KKLT-style moduli stabilisation
         │
         ▼
QO+R effective Lagrangian with λ_QR ~ 𝒪(1)
```

The string-theory embedding is presented as a tentative phenomenological
mapping rather than as a fundamental derivation; alternative origins of
the same 4D effective behaviour cannot be excluded.

---

## How to cite

If you use this work, please cite the v3.1 Zenodo deposit:

```bibtex
@software{slama2026qor_v31,
  author    = {Slama, Jonathan \'Edouard},
  title     = {{QO+R Framework: A Two-Field Phenomenological
                Modified-Gravity Hypothesis (v3.1)}},
  year      = 2026,
  month     = may,
  publisher = {Zenodo},
  version   = {3.1},
  doi       = {10.5281/zenodo.17806441},
  url       = {https://doi.org/10.5281/zenodo.17806441}
}
```

For the *Scientific Reports* peer-review record (Paper 1 only), the
submission identifier is **#2025-12-33523** (currently in major revision).

---

## License

This project is released under the MIT License — see [LICENSE](LICENSE) for
the full text. The associated datasets retain their original licences
(see [DATA_SOURCES.md](DATA_SOURCES.md)).

---

## Acknowledgments

- **SPARC team** (Lelli, McGaugh, Schombert) for the galaxy database;
- **IllustrisTNG collaboration** for simulation data access;
- **ALFALFA team** (Haynes et al.) for the H I survey;
- **2MASS Redshift Survey** team for the environmental density catalogue;
- **Little THINGS** team for dwarf-irregular kinematics;
- **NHANES / CDC** for the clinical biomarker data used in Paper 2;
- The two anonymous referees of *Scientific Reports* whose careful
  evaluation substantially improved Paper 1;
- **Iris**, an AI assistant trained with my reasoning methodology, for
  invaluable help with manuscript drafting, supplementary-analysis design,
  and iterative refinement of the scientific arguments throughout this
  revision cycle.

---

## Important caveats

This work presents:

1. A **phenomenological** two-field framework that describes a non-monotonic
   environmental trend detected in BTFR residuals;
2. **Empirical evidence** for that trend in SPARC, with complementary
   linewidth-based support from ALFALFA;
3. A **simulation-based test** of a discriminating prediction (sign
   inversion in gas-poor, stellar-massive systems) in IllustrisTNG;
4. A **tentative** mapping of the framework onto Type IIB string theory
   via Calabi–Yau compactification.

This work does **NOT** establish:

1. That string theory is the correct underlying theory of gravity;
2. That the *Q* and *R* fields are literally the type-IIB dilaton and the
   Kähler modulus rather than effective phenomenological proxies;
3. That alternative single-field, baryonic, or population-mixing
   explanations of the observed pattern have been exhaustively excluded;
4. That the simulation-based sign-inversion test in TNG constitutes an
   observational confirmation; direct observational follow-up in gas-poor
   early-type galaxies is the subject of a forthcoming companion paper.

The connection between empirical observation and string-theory phenomenology
remains a **hypothesis under investigation**, not a confirmed result.

---

*Last updated: 12 May 2026 — version 3.1.*
