# 🌌 QO+R Paper 1: U-Shaped BTFR Residuals

## Environmental Modulation of the Baryonic Tully-Fisher Relation

**Author:** Jonathan Édouard Slama  
**Affiliation:** Metafund Research Division  
**Contact:** jonathan@metafund.in  
**ORCID:** [0009-0002-1292-4350](https://orcid.org/0009-0002-1292-4350)  
**Version:** 3.0 (December 2025)  
**Status:** ✅ Ready for arXiv submission

---

## 🎯 Key Results

| Dataset | N | U-shape Coefficient (a) | p-value | Status |
|---------|---|------------------------|---------|--------|
| **SPARC** | 175 | +1.36 ± 0.24 | < 10⁻⁶ | ✅ Discovery |
| **ALFALFA** | 21,834 | +0.07 ± 0.03 | 0.0065 | ✅ **Independent Replication** |
| **Little THINGS** | 40 | +0.29 ± 0.32 | 0.19 | ⚠️ Consistent (small N) |
| **TNG100-1** | 53,363 | +1.95 ± 0.03 | < 10⁻⁶ | ✅ Simulation validation |

### 🔥 Killer Prediction Confirmed (TNG300)

| Category | N | a | Interpretation |
|----------|---|---|----------------|
| Gas-rich (Q-dom) | 444,374 | **+0.017** | Standard U-shape |
| Gas-poor + High M* | 8,779 | **-0.014** | **INVERTED U-shape!** |
| Extreme R-dom | 16,924 | **-0.019** | **Inversion confirmed** |

The inverted U-shape in gas-poor, high stellar mass systems is the **unique signature** of QO+R that distinguishes it from standard astrophysical explanations.

---

## 📖 Scientific Narrative (13 Figures)

The paper tells a complete scientific story in 5 acts:

| Act | Figures | Theme | Key Message |
|-----|---------|-------|-------------|
| **1. Failure** | fig01-02 | QO model fails | Science progresses through falsification |
| **2. Discovery** | fig03-04 | U-shape detected | From failure emerges structure |
| **3. Validation** | fig05-08 | Robustness & replication | Pattern is robust and replicated |
| **4. Constraints** | fig09-11 | Solar system & TNG | Model is consistent and testable |
| **5. Future** | fig12-13 | Theory & predictions | Toward fundamental connection |

---

## 📁 Repository Structure

```
Paper1-BTFR-UShape/
├── manuscript/
│   └── paper1_qor_btfr_v3.tex     ← Complete manuscript (13 figures)
│
├── figures/                        ← All 13 figures
│   ├── fig01_qo_only_failure.png
│   ├── fig02_forensic_analysis.png
│   ├── fig03_ushape_discovery.png
│   ├── fig04_calibration.png
│   ├── fig05_robustness.png
│   ├── fig06_replicability.png    ★ ALFALFA + Little THINGS
│   ├── fig07_microphysics.png
│   ├── fig08_q_hi_correlation.png
│   ├── fig09_solar_system.png
│   ├── fig10_tng_validation.png   ★ TNG100 validation
│   ├── fig11_tng_multiscale.png   ★ Killer prediction
│   ├── fig12_string_theory.png
│   └── fig13_predictions.png
│
├── tests/                          ← 13 analysis scripts
│   ├── 01_initial_qo_test/
│   ├── 02_forensic_analysis/
│   ├── 03_ushape_discovery/
│   ├── 04_calibration/
│   ├── 05_robustness/             ← Monte Carlo, Bootstrap, etc.
│   ├── 06_replicability/          ★ REAL ALFALFA data
│   ├── 07_microphysics/
│   ├── 08_microphysics_qhi/
│   ├── 09_solar_system/
│   ├── 10_tng_validation/         ★ REAL TNG data
│   ├── 11_tng_multiscale/         ★ Killer prediction
│   ├── 12_string_theory_link/
│   └── 13_predictions/
│
├── data/
│   └── sparc_with_environment.csv
│
├── README.md                       ← This file
├── README_REPRODUCIBILITY.md       ← Full reproducibility guide
├── FIGURES_AUDIT.md               ← Data provenance for all 13 figures
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone and install
```bash
git clone https://github.com/JonathanSlama/QO-R-JEDSLAMA.git
cd QO-R-JEDSLAMA/Paper1-BTFR-UShape
pip install -r requirements.txt
```

### 2. Run complete analysis pipeline
```bash
cd tests

# Act 1: The Failure
python 01_initial_qo_test/test_qo_only.py
python 02_forensic_analysis/forensic_analysis.py

# Act 2: The Discovery
python 03_ushape_discovery/discover_ushape.py
python 04_calibration/calibrate_params.py

# Act 3: Validation
python 05_robustness/robustness_tests.py
python 06_replicability/replicability_tests.py      # ★ ALFALFA
python 07_microphysics/microphysics_analysis.py
python 08_microphysics_qhi/q_hi_analysis.py

# Act 4: Constraints
python 09_solar_system/solar_system_constraints.py
python 10_tng_validation/tng_validation.py          # ★ TNG100
python 11_tng_multiscale/tng_multiscale.py          # ★ Killer prediction

# Act 5: Theory & Predictions
python 12_string_theory_link/string_theory_link.py
python 13_predictions/predictions.py
```

### 3. Compile manuscript
```bash
cd manuscript
pdflatex paper1_qor_btfr_v3.tex
pdflatex paper1_qor_btfr_v3.tex  # 2x for TOC and refs
```

---

## 📊 Data Sources (100% Real)

| Dataset | N | Source | Reference |
|---------|---|--------|-----------|
| SPARC | 175 | [astroweb.cwru.edu](http://astroweb.cwru.edu/SPARC/) | Lelli et al. (2016) |
| ALFALFA | 21,834 | [egg.astro.cornell.edu](http://egg.astro.cornell.edu/alfalfa/) | Haynes et al. (2018) |
| Little THINGS | 40 | VLA archive | Hunter et al. (2012) |
| TNG100-1 | 53,363 | [tng-project.org](https://www.tng-project.org/) | Pillepich et al. (2018) |
| TNG300-1 | 623,609 | [tng-project.org](https://www.tng-project.org/) | Pillepich et al. (2018) |

**⚠️ No synthetic data is used.** All scripts use only real observational and simulation data.

---

## 🔬 Scientific Summary

### The Story in Brief

1. **We started with a prediction** that a single scalar field would produce monotonic environmental dependence in BTFR residuals
2. **The prediction failed** — data showed the opposite of what was expected
3. **Forensic analysis revealed** a U-shaped pattern: both voids AND clusters show elevated residuals
4. **We reformulated** with two antagonistic fields (Q and R) that naturally produce the U-shape
5. **The pattern was replicated** on 21,834 independent ALFALFA galaxies
6. **A killer prediction emerged**: R-dominated systems should show INVERTED U-shape
7. **The prediction was confirmed** on 623,609 TNG300 galaxies

### What this PROVES

- ✅ The U-shape exists in SPARC (p < 10⁻⁶)
- ✅ The U-shape replicates in ALFALFA (p = 0.0065)
- ✅ The killer prediction is confirmed in TNG300
- ✅ QO+R satisfies solar system constraints

### What this does NOT prove

- ❌ The literal existence of Q and R fields
- ❌ A rigorous derivation from string theory
- ❌ Exclusion of all alternative explanations

See `README_REPRODUCIBILITY.md` for complete scientific discussion.

---

## 📝 Citation

```bibtex
@article{slama2025qor_btfr,
  author  = {Slama, Jonathan Édouard},
  title   = {Environmental Modulation of the Baryonic Tully-Fisher Relation:
             A Two-Field Scalar Approach Revealing a U-Shaped Residual Pattern},
  journal = {arXiv preprint},
  year    = {2025},
  note    = {Metafund Research Division}
}
```

---

## 🔗 Related Work

- **Paper 2:** QO+R Signatures in Human Health Data (NHANES)
- **Paper 3:** String Theory Embedding of the QO+R Framework

---

## 📜 License

MIT License - See LICENSE file.

---

*Last updated: December 3, 2025*
