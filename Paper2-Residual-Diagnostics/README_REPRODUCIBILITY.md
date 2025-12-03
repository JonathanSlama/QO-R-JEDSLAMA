# 🔬 GUIDE DE REPRODUCTIBILITÉ - Paper 2 Residual Diagnostics

## Residual Structure in Clinical Biomarker Ratios

**Version:** 3.0 (Décembre 2025)  
**Auteur:** Jonathan Édouard Slama  
**Affiliation:** Metafund Research Division  
**ORCID:** 0009-0002-1292-4350

---

## 📋 TABLE DES MATIÈRES

1. [Vue d'ensemble](#vue-densemble)
2. [Données sources](#données-sources)
3. [Installation](#installation)
4. [Pipeline d'analyse](#pipeline-danalyse)
5. [Résultats attendus](#résultats-attendus)
6. [Ce que les résultats prouvent](#ce-que-les-résultats-prouvent)
7. [Limitations](#limitations)

---

## 🎯 VUE D'ENSEMBLE

### Objectif
Tester si les résidus des ratios biomarqueurs cliniques contiennent une structure statistique non-aléatoire liée au statut de maladie.

### Hypothèses testées
- **H1** : Les distributions de résidus diffèrent entre états pathologiques
- **H2** : Des patterns non-linéaires (U-shape) existent
- **H3** : Les patterns corrèlent entre catégories de maladies

### Résultats principaux
| Métrique | Valeur |
|----------|--------|
| Résidus significatifs (H1) | 72/85 (85%) |
| Patterns U-shape (H2) | 35 détectés |
| Corrélations cross-disease (H3) | 70/90 (78%) |
| Résidus universels | 12 (≥3 maladies) |

---

## 📊 DONNÉES SOURCES

### Dataset 1 : NHANES 2017-2018

| Composant | Fichier | Contenu |
|-----------|---------|---------|
| Demographics | DEMO_J.XPT | Âge, sexe, race |
| Body Measures | BMX_J.XPT | BMI, tour de taille |
| Blood Pressure | BPX_J.XPT | Pression artérielle |
| Biochemistry | BIOPRO_J.XPT | AST, ALT, albumine |
| Glucose | GLU_J.XPT | Glucose à jeun |
| Insulin | INS_J.XPT | Insuline à jeun |
| Lipids | TCHOL_J.XPT, TRIGLY_J.XPT, HDL_J.XPT | Cholestérol, TG, HDL |
| HbA1c | GHB_J.XPT | Hémoglobine glyquée |
| Kidney | ALB_CR_J.XPT | Albumine/créatinine urinaire |
| CBC | CBC_J.XPT | Plaquettes |

**Source :** https://wwwn.cdc.gov/nchs/nhanes/

### Dataset 2 : Breast Cancer Coimbra

| Variable | Description |
|----------|-------------|
| Age | Années |
| BMI | kg/m² |
| Glucose | mg/dL |
| Insulin | µU/mL |
| HOMA | Index de résistance |
| Leptin | ng/mL |
| Adiponectin | µg/mL |
| Resistin | ng/mL |
| MCP-1 | pg/mL |
| Classification | 1=Control, 2=Cancer |

**Source :** https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra

---

## 🔧 INSTALLATION

```bash
# Cloner le repository
cd Paper2-Residual-Diagnostics/nhanes_extension

# Installer les dépendances
pip install -r requirements.txt

# Dépendances principales :
# - pandas, numpy, scipy
# - matplotlib, seaborn
# - scikit-learn
# - pyreadstat (pour fichiers .XPT NHANES)
```

---

## 📈 PIPELINE D'ANALYSE

### Étape 1 : Téléchargement NHANES
```bash
python scripts/00_download_nhanes.py
```
**Durée :** ~15 minutes  
**Sortie :** `data/raw/*.XPT`

### Étape 2 : Fusion des données
```bash
python scripts/01_merge_nhanes.py
```
**Sortie :** `data/processed/nhanes_merged.csv`

### Étape 3 : Calcul des ratios et résidus
```bash
python scripts/02_compute_all_ratios.py
```
**Sortie :** `data/processed/nhanes_with_ratios.csv`

### Étapes 4-8 : Analyses par maladie
```bash
python scripts/03_hepatic_fibrosis.py   # → figures/liver_fig*
python scripts/04_kidney_disease.py     # → figures/kidney_fig*
python scripts/05_cardiovascular.py     # → figures/cv_fig*
python scripts/06_diabetes.py           # → figures/diabetes_fig*
python scripts/07_metabolic_syndrome.py # → figures/mets_fig*
```

### Étape 9 : Patterns cross-disease
```bash
python scripts/08_cross_disease.py
```
**Sortie :** `figures/cross_disease_fig01_overview.png`

### Étape 10 : Rapport final
```bash
python scripts/09_summary_report.py
```
**Sortie :** `results/FINAL_SUMMARY_REPORT.txt`, `figures/summary_figure.png`

---

## 📊 RÉSULTATS ATTENDUS

### Par catégorie de maladie

| Catégorie | H1 (Diff) | H2 (U-shape) | Principal finding |
|-----------|-----------|--------------|-------------------|
| Hepatic Fibrosis | 76% | 26 patterns | U-shapes en zone grise |
| Kidney Disease | 94% | 3 patterns | Plus forte discrimination |
| Cardiovascular | 65% | 0 patterns | Pas de U-shapes |
| Diabetes | 88% | 6 patterns | U-shapes en prédiabète |
| Metabolic Syndrome | 100% | 14 patterns | Tous significatifs |

### Figure clé : U-shapes en zone indéterminée FIB-4

Le finding le plus important est la présence de patterns U-shaped dans la zone indéterminée FIB-4 (1.3-2.67), où 20% des patients tombent. Ceci suggère que les résidus contiennent de l'information sur le risque de progression que le score FIB-4 seul ne capture pas.

---

## ✅ CE QUE LES RÉSULTATS PROUVENT

1. **Les résidus ne sont pas du bruit aléatoire**
   - 85% des combinaisons résidu-maladie montrent des différences significatives
   - Pattern cohérent à travers 5 catégories de maladies

2. **Des patterns non-linéaires existent**
   - 35 U-shapes détectés, principalement en fibrose hépatique
   - Concentration dans les zones diagnostiques grises

3. **Les patterns sont universels**
   - 12 résidus significatifs dans ≥3 maladies
   - Suggère des mécanismes de régulation systémiques

---

## ❌ CE QUE LES RÉSULTATS NE PROUVENT PAS

1. **Utilité clinique**
   - Aucune amélioration de classification démontrée
   - Dans Breast Cancer, résidus redondants avec ratios

2. **Valeur prédictive**
   - Étude transversale, pas de suivi longitudinal
   - Pas de validation sur outcomes cliniques

3. **Mécanismes biologiques**
   - Structure statistique ≠ signification biologique
   - Identification des processus sous-jacents nécessaire

4. **Généralisation**
   - Population U.S. uniquement (NHANES)
   - Validation externe requise (UK Biobank, etc.)

---

## ⚠️ LIMITATIONS

1. **Design transversal** : Pas de causalité établie
2. **Comparaisons multiples** : Risque de faux positifs malgré Bonferroni
3. **Outcomes proxy** : Pas de gold standard (biopsie, imagerie)
4. **Dépendance au modèle** : Résidus dépendent des covariables choisies
5. **Population unique** : NHANES = U.S., peut ne pas généraliser

---

## 🔗 CONNEXION AU FRAMEWORK QO+R

Ce travail est motivé par le Paper 1, où nous avons trouvé des U-shapes dans les résidus astrophysiques (BTFR). L'hypothèse est qu'un phénomène analogue pourrait exister en médecine :

- **Ratio (Q)** : Capture la dynamique primaire (FIB-4, HOMA-IR, eGFR)
- **Résidu (R)** : Révèle les mécanismes secondaires de régulation

Les U-shapes en fibrose hépatique pourraient refléter des mécanismes compétiteurs (inflammation vs fibrose) laissant des signatures mathématiques.

**Cette connexion est spéculative et nécessite validation.**

---

## 📝 CITATION

```bibtex
@article{slama2025residuals,
  author  = {Slama, Jonathan Édouard},
  title   = {Residual Structure in Clinical Biomarker Ratios: 
             Evidence for Non-Random Patterns Across Disease Categories},
  journal = {arXiv preprint},
  year    = {2025},
  note    = {Metafund Research Division}
}
```

---

*Guide mis à jour : 2025-12-03*
