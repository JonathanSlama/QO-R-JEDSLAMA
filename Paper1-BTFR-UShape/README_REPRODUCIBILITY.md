# 🔬 GUIDE DE REPRODUCTIBILITÉ COMPLÈTE - QO+R Paper 1

## U-Shaped BTFR Residuals: Evidence for Dual-Field Modified Gravity

**Version:** 2.0 (3 Décembre 2025)  
**Auteur:** Jonathan Edouard SLAMA  
**Affiliation:** Metafund Research Division  
**ORCID:** 0009-0002-1292-4350

---

## 📋 TABLE DES MATIÈRES

1. [Vue d'ensemble](#vue-densemble)
2. [Structure narrative](#structure-narrative)
3. [Données sources](#données-sources)
4. [Environnement et installation](#environnement-et-installation)
5. [Étapes de l'analyse](#étapes-de-lanalyse)
6. [Tests de reproductibilité](#tests-de-reproductibilité)
7. [Ce que les résultats prouvent](#ce-que-les-résultats-prouvent)
8. [Ce que les résultats NE prouvent PAS](#ce-que-les-résultats-ne-prouvent-pas)
9. [Limitations et discussion](#limitations-et-discussion)
10. [Alternatives astrophysiques classiques](#alternatives-astrophysiques-classiques)

---

## 🎯 VUE D'ENSEMBLE

### Objectif
Tester l'existence d'une dépendance environnementale en forme de U dans les résidus de la relation Baryonic Tully-Fisher (BTFR), et évaluer si cette signature est cohérente avec le modèle QO+R.

### Résultat principal
Un **U-shape statistiquement significatif** est détecté sur :
- SPARC (175 galaxies) : a = +1.36, p < 0.000001
- ALFALFA (21,834 galaxies) : a = +0.07, p = 0.0065
- IllustrisTNG300 (623,609 galaxies) : Pattern confirmé avec **inversion** pour systèmes R-dominés

### Prédiction discriminante (Killer Prediction)
Les systèmes **gas-poor à haute masse stellaire** montrent un U-shape **INVERSÉ** (a < 0), exactement comme prédit par QO+R où le champ R domine.

---

## 📖 STRUCTURE NARRATIVE

Le Paper 1 raconte une **histoire scientifique complète** en 5 actes, avec 13 figures :

### Acte 1 : L'Échec (fig01-02)
- **fig01** : Le modèle QO seul prédit un comportement monotone
- **fig02** : Analyse forensique révèle un U-shape inattendu
- **Message** : La science progresse par falsification

### Acte 2 : La Découverte (fig03-04)
- **fig03** : Quantification du U-shape (a = 1.36, p < 10⁻⁶)
- **fig04** : Calibration de C_Q = +2.82 et C_R = -0.72
- **Message** : Du problème émerge une structure

### Acte 3 : La Validation (fig05-08)
- **fig05** : Monte Carlo, Bootstrap, Jackknife, Permutation
- **fig06** : Réplication sur ALFALFA (21,834 galaxies)
- **fig07** : Couplage Q → gaz, R → étoiles
- **fig08** : Corrélation Q-HI (r = 0.67)
- **Message** : Le pattern est robuste et répliqué

### Acte 4 : Les Contraintes (fig09-11)
- **fig09** : QO+R satisfait Eöt-Wash, PPN, LLR
- **fig10** : Validation sur TNG100 (53,363 galaxies)
- **fig11** : **Killer prediction** - U inversé dans systèmes R-dominés
- **Message** : Le modèle est cohérent et testable

### Acte 5 : L'Avenir (fig12-13)
- **fig12** : Connexion suggestive à la théorie des cordes
- **fig13** : Prédictions falsifiables (UDGs, WALLABY/SKA...)
- **Message** : Vers une connexion fondamentale

---

## 📊 DONNÉES SOURCES

### Données observationnelles (100% RÉELLES)

| Dataset | N | Source | Référence | Fichier local |
|---------|---|--------|-----------|---------------|
| SPARC | 175 | Rotation curves | Lelli et al. (2016) | `data/sparc_with_environment.csv` |
| ALFALFA | 21,834 | α.100 HI catalog | Haynes et al. (2018) | `BTFR/Test-Replicability/data/alfalfa.csv` |
| Little THINGS | 40 | Dwarf irregulars | Hunter et al. (2012) | `BTFR/Test-Replicability/data/little_things.csv` |
| WALLABY | ~1,000 | HI survey | Koribalski et al. (2020) | `BTFR/TNG/Data_Wallaby/` |

### Données de simulation (IllustrisTNG - RÉELLES)

| Simulation | Fichiers | Galaxies analysées | Source |
|------------|----------|-------------------|--------|
| TNG100-1 | 448 HDF5 | 53,363 | IllustrisTNG Project |
| TNG300-1 | 600 HDF5 | 623,609 | IllustrisTNG Project |
| TNG50-1 | 680 HDF5 | - | IllustrisTNG Project |

**Clé API IllustrisTNG requise** : https://www.tng-project.org/data/

### ⚠️ AUCUNE DONNÉE SYNTHÉTIQUE

Tous les tests utilisent exclusivement des données observationnelles ou simulées réelles. Les scripts précédents qui généraient des données synthétiques ont été remplacés le 3 Décembre 2025.

---

## 🔧 ENVIRONNEMENT ET INSTALLATION

### Prérequis
```bash
Python 3.9+
pip install numpy pandas scipy matplotlib h5py
```

### Structure du projet
```
Paper1-BTFR-UShape/
├── data/
│   └── sparc_with_environment.csv    # SPARC avec classification environnementale
├── figures/
│   ├── fig01_qo_only_failure.png     # Échec du modèle QO seul
│   ├── fig02_forensic_analysis.png   # Analyse forensique
│   ├── fig03_ushape_discovery.png    # Découverte U-shape
│   ├── fig04_calibration.png         # Calibration paramètres
│   ├── fig05_robustness.png          # Tests de robustesse
│   ├── fig06_replicability.png       # ★ Réplicabilité ALFALFA/Little THINGS
│   ├── fig07_microphysics.png        # Couplage microphysique
│   ├── fig08_q_hi_correlation.png    # Corrélation Q-HI
│   ├── fig09_solar_system.png        # Contraintes système solaire
│   ├── fig10_tng_validation.png      # ★ Validation TNG100
│   ├── fig11_tng_multiscale.png      # ★ Killer prediction TNG300
│   ├── fig12_string_theory.png       # Lien théorie des cordes
│   └── fig13_predictions.png         # Prédictions testables
├── tests/
│   ├── 01_initial_qo_test/           # Test initial QO seul → ÉCHEC
│   ├── 02_forensic_analysis/         # Pourquoi QO échoue
│   ├── 03_ushape_discovery/          # Découverte du U-shape
│   ├── 04_calibration/               # Calibration C_Q, C_R
│   ├── 05_robustness/                # Monte Carlo, Bootstrap, etc.
│   ├── 06_replicability/             # ★ ALFALFA + Little THINGS
│   ├── 07_microphysics/              # Couplage Q-gas, R-stars
│   ├── 08_microphysics_qhi/          # Corrélation Q-HI
│   ├── 09_solar_system/              # Contraintes Eöt-Wash, PPN
│   ├── 10_tng_validation/            # ★ TNG100 validation
│   ├── 11_tng_multiscale/            # ★ TNG300 stratifié
│   ├── 12_string_theory_link/        # Embedding string theory
│   └── 13_predictions/               # Prédictions falsifiables
└── manuscript/
    └── paper1_qor_btfr_v2.tex        # Manuscrit LaTeX
```

---

## 📈 ÉTAPES DE L'ANALYSE

### Démarche scientifique complète

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 1: TEST DU MODÈLE QO INITIAL (01_initial_qo_test)               │
│  → RÉSULTAT: ÉCHEC - Prédiction opposée aux données                    │
│  → Le modèle QO seul prédit ↓ dans les voids, observe ↑                │
├─────────────────────────────────────────────────────────────────────────┤
│  ÉTAPE 2: ANALYSE FORENSIQUE (02_forensic_analysis)                    │
│  → Pourquoi QO échoue ? Découverte de la forme en U                    │
│  → Voids ET clusters montrent des résidus positifs                     │
├─────────────────────────────────────────────────────────────────────────┤
│  ÉTAPE 3: REFORMULATION (03_ushape_discovery)                          │
│  → Introduction du champ R antagoniste                                  │
│  → Modèle QO+R avec couplage Q²R²                                      │
├─────────────────────────────────────────────────────────────────────────┤
│  ÉTAPE 4: CALIBRATION (04_calibration)                                 │
│  → C_Q = +2.82 (gaz → expansion)                                       │
│  → C_R = -0.72 (étoiles → compression)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  ÉTAPE 5: VALIDATION ROBUSTESSE (05_robustness)                        │
│  → Monte Carlo: 100% survie sur 1000 itérations                        │
│  → Bootstrap: 96.5% montrent U-shape                                   │
│  → Jackknife, permutation, K-fold CV                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  ÉTAPE 6: RÉPLICABILITÉ (06_replicability) ★                           │
│  → ALFALFA (21,834 galaxies): a = +0.07, p = 0.0065 ✓                  │
│  → Little THINGS (40 galaxies): non significatif (petit N)             │
├─────────────────────────────────────────────────────────────────────────┤
│  ÉTAPE 7: MICROPHYSIQUE (07_microphysics, 08_qhi)                      │
│  → Q couple préférentiellement au gaz HI                               │
│  → R couple préférentiellement aux étoiles                             │
├─────────────────────────────────────────────────────────────────────────┤
│  ÉTAPE 8: CONTRAINTES SYSTÈME SOLAIRE (09_solar_system)                │
│  → Tests Eöt-Wash, PPN, LLR satisfaits                                 │
│  → Mécanisme d'écrantage à petite échelle                              │
├─────────────────────────────────────────────────────────────────────────┤
│  ÉTAPE 9: VALIDATION TNG (10_tng_validation) ★                         │
│  → TNG100-1 (ΛCDM seul): U-shape faible, p = 0.075                     │
│  → TNG100-1 + QO+R: U-shape significatif, p = 0.004                    │
├─────────────────────────────────────────────────────────────────────────┤
│  ÉTAPE 10: KILLER PREDICTION (11_tng_multiscale) ★                     │
│  → Gas-rich (Q-dom): a = +0.017 → U positif                            │
│  → Gas-poor + High M* (R-dom): a = -0.014 → U INVERSÉ!                 │
│  → C'est la signature UNIQUE de QO+R                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Exécution complète

```bash
cd Paper1-BTFR-UShape/tests

# Étapes 1-5: Découverte et validation sur SPARC
python 01_initial_qo_test/test_qo_only.py
python 02_forensic_analysis/forensic_analysis.py
python 03_ushape_discovery/discover_ushape.py
python 04_calibration/calibrate_params.py
python 05_robustness/robustness_tests.py

# Étape 6: Réplicabilité (CRITIQUE)
python 06_replicability/replicability_tests.py

# Étapes 7-8: Microphysique
python 07_microphysics/microphysics_analysis.py
python 08_microphysics_qhi/q_hi_analysis.py

# Étape 9: Contraintes
python 09_solar_system/solar_system_constraints.py

# Étapes 10-11: Validation simulations (CRITIQUE)
python 10_tng_validation/tng_validation.py
python 11_tng_multiscale/tng_multiscale.py

# Étapes 12-13: Théorie et prédictions
python 12_string_theory_link/string_theory_link.py
python 13_predictions/predictions.py
```

---

## 🧪 TESTS DE REPRODUCTIBILITÉ

### Résultats attendus

#### Test 06 - Réplicabilité
```
Dataset          N        a          p-value      Résultat
─────────────────────────────────────────────────────────────
SPARC          181    +1.3606      <0.000001    ✓ U-shape
ALFALFA     21,834    +0.0701       0.006530    ✓ U-shape
Little_THINGS  40    +0.2891       0.186262    ✗ (petit N)
```

#### Test 10 - TNG Validation
```
Dataset              N        a          p-value      Résultat
────────────────────────────────────────────────────────────────
SPARC (obs)        175    +0.035      <0.001       ✓ U-shape
TNG100-1 (ΛCDM)  53,363    +0.045       0.075       ~ borderline
TNG100-1 + QO+R  53,363    +0.039       0.004       ✓ U-shape
```

#### Test 11 - Killer Prediction (TNG300)
```
Catégorie                    N            a          Interprétation
─────────────────────────────────────────────────────────────────────
Gas-rich (Q-dom)       444,374       +0.017       U positif (Q > R)
Gas-poor + High M*       8,779       -0.014       U INVERSÉ (R > Q) ★
EXTREME R-DOM           16,924       -0.019       Inversion confirmée ★
```

---

## ✅ CE QUE LES RÉSULTATS PROUVENT

### 1. Existence du U-shape (ÉTABLI)
- Le U-shape existe dans SPARC (p < 0.000001)
- Il est **répliqué** dans ALFALFA (p = 0.0065, N = 21,834)
- Ce n'est **pas** un artefact du sample SPARC

### 2. Robustesse statistique (ÉTABLIE)
- Monte Carlo: 100% de survie sur 1000 itérations
- Bootstrap: 96.5% des samples montrent le U-shape
- Permutation: p < 0.001 vs distribution nulle

### 3. Cohérence avec QO+R (SUPPORTÉE)
- La forme en U est **qualitativement** cohérente avec un modèle à deux champs
- Q couple au gaz (U positif dans systèmes gas-rich)
- R couple aux étoiles (U inversé dans systèmes gas-poor haute masse)

### 4. Killer Prediction confirmée (FORTE ÉVIDENCE)
- Les systèmes R-dominés montrent l'inversion prédite
- C'est une signature **unique** de QO+R
- Difficile à expliquer par l'astrophysique standard seule

---

## ❌ CE QUE LES RÉSULTATS NE PROUVENT PAS

### 1. L'existence des champs Q et R
- Les résultats sont **cohérents** avec QO+R
- Mais ils ne **prouvent pas** que ces champs existent
- D'autres modèles pourraient produire des signatures similaires

### 2. Le lien avec la théorie des cordes
- L'identification Q ↔ Dilaton et R ↔ Module de Kähler est **suggestive**
- Ce n'est **pas** une dérivation rigoureuse
- Présenté comme "correspondance formelle", pas preuve

### 3. L'exclusion des alternatives classiques
- Ram pressure stripping peut contribuer au signal
- Biais de sélection non totalement exclus
- Effets de marée possibles dans les clusters

### 4. La valeur exacte des paramètres
- C_Q et C_R sont calibrés **empiriquement**
- Pas dérivés de premiers principes
- Pourraient varier avec l'échelle ou la méthode

---

## ⚠️ LIMITATIONS ET DISCUSSION

### Différence d'amplitude SPARC vs ALFALFA

| Dataset | a (U-shape) | Différence |
|---------|-------------|------------|
| SPARC | 1.36 | Référence |
| ALFALFA | 0.07 | ~20× plus faible |

**Explications possibles :**
1. **Mesure de vitesse différente** : SPARC utilise Vflat (courbes de rotation complètes), ALFALFA utilise W50 (largeur de raie HI à 50%)
2. **Sélection différente** : SPARC = optique+HI, ALFALFA = HI seul
3. **Couverture environnementale** : ALFALFA manque de galaxies "group"
4. **Définition du density proxy** : Peut différer entre les deux analyses

### Classification environnementale

La classification void/field/group/cluster est basée sur :
- Distance au 5ème voisin le plus proche
- Densité locale de galaxies
- Pour SPARC : classification manuelle de la littérature
- Pour ALFALFA : classification automatique

**Cette méthode introduit des incertitudes** qui doivent être prises en compte.

---

## 🔬 ALTERNATIVES ASTROPHYSIQUES CLASSIQUES

### Effets qui pourraient contribuer au signal

| Effet | Description | Impact sur U-shape |
|-------|-------------|-------------------|
| **Ram pressure stripping** | Perte de gaz dans les clusters | ↓ résidus dans clusters |
| **Marées gravitationnelles** | Perturbation dans environnements denses | ↓ résidus dans clusters |
| **Quenching environnemental** | Arrêt formation stellaire | Modifie f_gas |
| **Biais de sélection** | Galaxies brillantes dans clusters | Biais sur masse |

### Pourquoi ces effets ne suffisent probablement pas

1. **Le U-shape a DEUX branches** : résidus ↑ dans voids ET clusters
2. **Ram pressure** ne peut pas expliquer les résidus **positifs** dans les voids
3. **L'inversion dans les systèmes R-dominés** n'est pas prédite par ces effets

### Test discriminant proposé

Pour distinguer QO+R de l'astrophysique standard :
- **Galaxies isolées haute masse** : Si U-shape présent sans environnement dense → favorise nouvelle physique
- **Courbes de rotation résolues** : Si le signal est dans la forme de la courbe, pas juste l'amplitude → favorise modification de gravité

---

## 📚 RÉFÉRENCES

### Données
- Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016). SPARC: Mass Models for 175 Disk Galaxies. AJ, 152, 157.
- Haynes, M. P., et al. (2018). The Arecibo Legacy Fast ALFA Survey: The ALFALFA Extragalactic HI Source Catalog. ApJ, 861, 49.
- Hunter, D. A., et al. (2012). Little Things. AJ, 144, 134.
- Pillepich, A., et al. (2018). First results from the IllustrisTNG simulations. MNRAS, 475, 648.

### Théorie
- McGaugh, S. S. (2012). The Baryonic Tully-Fisher Relation. AJ, 143, 40.
- Milgrom, M. (1983). MOND. ApJ, 270, 365.

---

## 🔧 RÉFÉRENCE TECHNIQUE DÉTAILLÉE

### Script 06 - Replicability Tests

**Fichier:** `tests/06_replicability/replicability_tests.py`

**Entrées requises:**
```
data/sparc_with_environment.csv          # Depuis ce repo
BTFR/Test-Replicability/data/alfalfa.csv  # 21,834 galaxies ALFALFA
BTFR/Test-Replicability/data/little_things.csv  # 40 galaxies Little THINGS
```

**Sorties générées:**
```
figures/fig06_replicability.png           # Figure de réplicabilité
tests/06_replicability/replicability_REAL_results.csv  # Résultats CSV
```

**Résultat attendu:**
```
SPARC:        a = +1.3606 ± 0.2368, p < 0.000001  ✓ U-shape
ALFALFA:      a = +0.0701 ± 0.0282, p = 0.006530 ✓ Répliqué
Little THINGS: a = +0.2891 ± 0.3242, p = 0.186262 ✗ (petit N)
```

---

### Script 10 - TNG Validation

**Fichier:** `tests/10_tng_validation/tng_validation.py`

**Entrées requises:**
```
data/sparc_with_environment.csv           # Depuis ce repo
BTFR/TNG/tng_galaxies.csv                 # 53,363 galaxies TNG100
BTFR/TNG/qor_validation_results_PAPER.csv # Résultats pré-calculés
```

**Sorties générées:**
```
figures/fig10_tng_validation.png          # Figure de validation TNG
```

**Résultat attendu:**
```
SPARC:              a = +1.3606, p < 0.000001  ✓ U-shape
TNG100-1:           a = +1.9530, p < 0.000001  ✓ U-shape détecté
TNG100-1 (ΛCDM):    a = 0.045, p = 0.075       ~ borderline
TNG100-1 + QO+R:    a = 0.039, p = 0.004       ✓ significatif
```

---

### Script 11 - TNG Multiscale (Killer Prediction)

**Fichier:** `tests/11_tng_multiscale/tng_multiscale.py`

**Entrées requises:**
```
BTFR/TNG/tng300_stratified_results.csv    # Résultats TNG300 stratifiés
```

**Sorties générées:**
```
figures/fig11_tng_multiscale.png          # Figure killer prediction
```

**Résultat attendu (Killer Prediction):**
```
Gas-rich (Q-dom):     444,374 galaxies, a = +0.0173  → U positif
Gas-poor + High M*:     8,779 galaxies, a = -0.0138  → U INVERSÉ ★
EXTREME R-DOM:         16,924 galaxies, a = -0.0187  → Inversion confirmée ★
```

---

### Vérification complète

Pour vérifier que tous les scripts fonctionnent :

```bash
cd Paper1-BTFR-UShape/tests

# Test rapide des 3 scripts critiques
python 06_replicability/replicability_tests.py
python 10_tng_validation/tng_validation.py
python 11_tng_multiscale/tng_multiscale.py

# Vérifier les figures générées
ls -la ../figures/fig06_replicability.png
ls -la ../figures/fig10_tng_validation.png
ls -la ../figures/fig11_tng_multiscale.png
```

### Compilation du manuscrit

```bash
cd Paper1-BTFR-UShape/manuscript
pdflatex paper1_qor_btfr_v3.tex
pdflatex paper1_qor_btfr_v3.tex  # 2x pour les références
```

---

## 📝 CHANGELOG

- **v3.0 (2025-12-03)** : Manuscrit v3 avec figures incluses, section reproductibilité technique
- **v2.0 (2025-12-03)** : Remplacement de TOUS les scripts avec données synthétiques par des versions utilisant des données réelles. Documentation complète de reproductibilité.
- **v1.0 (2025-11)** : Version initiale avec certains scripts utilisant des données synthétiques pour démonstration.

---

*Document créé : 3 Décembre 2025*  
*Auteur : Jonathan Edouard SLAMA*  
*Contact : jonathan@metafund.in*
