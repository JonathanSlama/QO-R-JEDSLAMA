# AUDIT DES FIGURES - Paper 2 Residual Diagnostics
## Documentation complète des 11 figures

**Date:** 3 Décembre 2025  
**Auteur:** Jonathan Édouard Slama  
**Statut:** ✅ TOUTES LES FIGURES INCLUSES DANS LE MANUSCRIT v3

---

## RÉSUMÉ EXÉCUTIF

Le manuscrit `paper2_residuals_v3.tex` inclut **11 figures** couvrant deux datasets :
- **Breast Cancer Coimbra** (N=116) : 2 figures
- **NHANES 2017-2018** (N=9,254) : 9 figures

---

## LISTE COMPLÈTE DES FIGURES

| # | Fichier | Dossier | Section | Contenu |
|---|---------|---------|---------|---------|
| 1 | `fig01_distributions_by_group.png` | `figures/` | Results: BC | Distributions biomarqueurs |
| 2 | `fig07_residual_distributions.png` | `figures/` | Results: BC | Résidus par groupe |
| 3 | `liver_fig01_fib4_distribution.png` | `nhanes_extension/figures/` | Results: Liver | Distribution FIB-4 |
| 4 | `liver_fig02_residuals_by_zone.png` | `nhanes_extension/figures/` | Results: Liver | Résidus par zone |
| 5 | `liver_fig03_ushapes_indeterminate.png` | `nhanes_extension/figures/` | Results: Liver | ★ U-shapes zone grise |
| 6 | `kidney_fig01_egfr_acr_distribution.png` | `nhanes_extension/figures/` | Results: Kidney | eGFR et ACR |
| 7 | `cv_fig01_tg_hdl_by_status.png` | `nhanes_extension/figures/` | Results: CV | TG/HDL par statut |
| 8 | `diabetes_fig01_glycemic_analysis.png` | `nhanes_extension/figures/` | Results: Diabetes | Analyse glycémique |
| 9 | `mets_fig01_overview.png` | `nhanes_extension/figures/` | Results: MetS | Syndrome métabolique |
| 10 | `cross_disease_fig01_overview.png` | `nhanes_extension/figures/` | Results: Cross | Patterns universels |
| 11 | `summary_figure.png` | `nhanes_extension/figures/` | Summary | Vue d'ensemble |

---

## STRUCTURE NARRATIVE

### Partie 1 : Breast Cancer Coimbra (Étude préliminaire)
- **fig01** : Distributions montrent différences cancer/contrôle
- **fig07** : Résidus contiennent structure diagnostique
- **Conclusion** : Structure existe mais redondante avec ratios

### Partie 2 : NHANES Multi-Disease (Validation principale)

| Section | Figure | Finding clé |
|---------|--------|-------------|
| Liver | fig03-05 | ★ U-shapes dans zone grise FIB-4 |
| Kidney | fig06 | 94% des résidus significatifs |
| CV | fig07 | 65% significatifs, pas de U-shapes |
| Diabetes | fig08 | U-shapes en prédiabète |
| MetS | fig09 | 100% des résidus significatifs |
| Cross | fig10 | 12 résidus universels |
| Summary | fig11 | Vue d'ensemble H1-H2-H3 |

---

## RÉSULTATS CLÉS PAR FIGURE

### Figure 5 (liver_fig03) - FIGURE CLÉ ★
**U-shapes dans la zone indéterminée FIB-4**
- Patients aux deux extrêmes de la zone 1.3-2.67 montrent résidus élevés
- Suggère information sur risque de progression
- Parallel avec U-shapes BTFR du Paper 1

### Figure 10 (cross_disease_fig01) - RÉSIDUS UNIVERSELS
**12 résidus significatifs dans ≥3 catégories**
- TG/HDL_Residual : 5/5 maladies
- HOMA-IR_Residual : 4/5 maladies
- eGFR_Residual : 4/5 maladies

---

## PROVENANCE DES DONNÉES

| Dataset | Source | N | Référence |
|---------|--------|---|-----------|
| NHANES 2017-2018 | CDC | 9,254 | NHANES (2018) |
| Breast Cancer Coimbra | UCI ML Repository | 116 | Patrício et al. (2018) |

**Toutes les données sont publiques et réelles.**

---

## SCRIPTS ASSOCIÉS

| Figure | Script générateur |
|--------|-------------------|
| fig01-10 (BC) | `scripts/02_explore_data.py` → `scripts/05_statistical_tests.py` |
| liver_fig* | `nhanes_extension/scripts/03_hepatic_fibrosis.py` |
| kidney_fig* | `nhanes_extension/scripts/04_kidney_disease.py` |
| cv_fig* | `nhanes_extension/scripts/05_cardiovascular.py` |
| diabetes_fig* | `nhanes_extension/scripts/06_diabetes.py` |
| mets_fig* | `nhanes_extension/scripts/07_metabolic_syndrome.py` |
| cross_disease_fig* | `nhanes_extension/scripts/08_cross_disease.py` |
| summary_figure | `nhanes_extension/scripts/09_summary_report.py` |

---

## COMPILATION DU MANUSCRIT

```bash
cd Paper2-Residual-Diagnostics/nhanes_extension/manuscript

# Compilation (nécessite les deux dossiers figures accessibles)
pdflatex paper2_residuals_v3.tex
pdflatex paper2_residuals_v3.tex  # 2x pour TOC
```

**Note :** Le fichier .tex utilise `\graphicspath{{../figures/}{../nhanes_extension/figures/}}` pour accéder aux deux dossiers.

---

## CHECKLIST

- [x] 11 figures identifiées
- [x] Toutes les figures dans le manuscrit v3
- [x] Chemins graphicspath corrects
- [x] Labels LaTeX définis
- [x] Données 100% réelles (NHANES + UCI)
- [x] Scripts documentés

---

*Audit mis à jour : 2025-12-03*  
*Manuscrit : paper2_residuals_v3.tex*
