# AUDIT DES FIGURES - Paper 1 QO+R BTFR
## Documentation complète des 13 figures

**Date:** 3 Décembre 2025  
**Auteur:** Jonathan Édouard Slama  
**Statut:** ✅ TOUTES LES FIGURES INCLUSES DANS LE MANUSCRIT v3

---

## RÉSUMÉ EXÉCUTIF

Le manuscrit `paper1_qor_btfr_v3.tex` inclut les **13 figures** qui racontent l'histoire scientifique complète, de l'échec initial à la découverte et aux prédictions.

| Figure | Fichier | Section | Données | Statut |
|--------|---------|---------|---------|--------|
| fig01 | `fig01_qo_only_failure.png` | 3 (Failure) | SPARC réel | ✅ |
| fig02 | `fig02_forensic_analysis.png` | 3 (Failure) | SPARC réel | ✅ |
| fig03 | `fig03_ushape_discovery.png` | 4 (Discovery) | SPARC réel | ✅ |
| fig04 | `fig04_calibration.png` | 4 (Discovery) | SPARC réel | ✅ |
| fig05 | `fig05_robustness.png` | 5 (Robustness) | SPARC réel | ✅ |
| fig06 | `fig06_replicability.png` | 6 (Replication) | ALFALFA + Little THINGS réels | ✅ |
| fig07 | `fig07_microphysics.png` | 7 (Microphysics) | SPARC réel | ✅ |
| fig08 | `fig08_q_hi_correlation.png` | 7 (Microphysics) | SPARC réel | ✅ |
| fig09 | `fig09_solar_system.png` | 8 (Constraints) | Théorique | ✅ |
| fig10 | `fig10_tng_validation.png` | 9 (Simulations) | TNG100 réel | ✅ |
| fig11 | `fig11_tng_multiscale.png` | 9 (Simulations) | TNG300 réel | ✅ |
| fig12 | `fig12_string_theory.png` | 10 (Theory) | Théorique | ✅ |
| fig13 | `fig13_predictions.png` | 11 (Predictions) | Théorique | ✅ |

---

## STRUCTURE NARRATIVE

Le manuscrit suit une progression logique en 5 actes :

### Acte 1 : L'Échec (Section 3)
**Message :** La science progresse par falsification

| Figure | Rôle | Message clé |
|--------|------|-------------|
| fig01 | Échec du modèle QO | Le modèle à un seul champ prédit un comportement monotone, mais les données montrent autre chose |
| fig02 | Analyse forensique | Découverte inattendue : les résidus forment un U, pas une droite |

### Acte 2 : La Découverte (Section 4)
**Message :** Du problème émerge une structure

| Figure | Rôle | Message clé |
|--------|------|-------------|
| fig03 | Quantification du U-shape | Coefficient quadratique a = 1.36, p < 10⁻⁶ |
| fig04 | Calibration | Deux champs avec couplages opposés : C_Q = +2.82, C_R = -0.72 |

### Acte 3 : La Validation (Sections 5-7)
**Message :** Le pattern est robuste et répliqué

| Figure | Rôle | Message clé |
|--------|------|-------------|
| fig05 | Tests de robustesse | Monte Carlo, Bootstrap, Jackknife, Permutation : tous passent |
| fig06 | Réplication indépendante | ALFALFA (21,834 galaxies) confirme le U-shape |
| fig07 | Couplage microphysique | Q → gaz, R → étoiles |
| fig08 | Corrélation Q-HI | Le champ Q suit la masse de gaz HI |

### Acte 4 : Les Contraintes et Prédictions (Sections 8-9)
**Message :** Le modèle est cohérent et testable

| Figure | Rôle | Message clé |
|--------|------|-------------|
| fig09 | Contraintes système solaire | QO+R satisfait Eöt-Wash, PPN, LLR via écrantage |
| fig10 | Validation TNG100 | Simulations ΛCDM montrent un signal faible |
| fig11 | Killer prediction | U-shape INVERSÉ dans systèmes gas-poor haute masse |

### Acte 5 : La Théorie et l'Avenir (Sections 10-11)
**Message :** Vers une connexion fondamentale

| Figure | Rôle | Message clé |
|--------|------|-------------|
| fig12 | Lien théorie des cordes | Correspondance suggestive Q ↔ Dilaton, R ↔ Kähler |
| fig13 | Prédictions falsifiables | UDGs, elliptiques isolées, WALLABY/SKA, courbes résolues |

---

## DÉTAIL PAR FIGURE

### Figure 1 : Échec du modèle QO
- **Fichier :** `fig01_qo_only_failure.png`
- **Script :** `tests/01_initial_qo_test/test_qo_only.py`
- **Données :** SPARC (175 galaxies) - RÉEL
- **Section LaTeX :** 3 - Results I: The Failed Prediction
- **Label :** `\label{fig:qo_failure}`
- **Message :** Le modèle à un seul champ prédit un comportement monotone mais les données montrent un comportement non-monotone

### Figure 2 : Analyse forensique
- **Fichier :** `fig02_forensic_analysis.png`
- **Script :** `tests/02_forensic_analysis/forensic_analysis.py`
- **Données :** SPARC (175 galaxies) - RÉEL
- **Section LaTeX :** 3 - Results I: The Failed Prediction
- **Label :** `\label{fig:forensic}`
- **Message :** Les résidus forment un U-shape : voids ET clusters montrent des résidus positifs

### Figure 3 : Découverte du U-shape
- **Fichier :** `fig03_ushape_discovery.png`
- **Script :** `tests/03_ushape_discovery/discover_ushape.py`
- **Données :** SPARC (175 galaxies) - RÉEL
- **Section LaTeX :** 4 - Results II: Discovery of the U-Shape
- **Label :** `\label{fig:ushape}`
- **Résultats :** a = 1.36 ± 0.24, z = 5.75, p < 10⁻⁶

### Figure 4 : Calibration des paramètres
- **Fichier :** `fig04_calibration.png`
- **Script :** `tests/04_calibration/calibrate_params.py`
- **Données :** SPARC (175 galaxies) - RÉEL
- **Section LaTeX :** 4 - Results II: Discovery of the U-Shape
- **Label :** `\label{fig:calibration}`
- **Résultats :** C_Q = +2.82 ± 0.15, C_R = -0.72 ± 0.08

### Figure 5 : Tests de robustesse
- **Fichier :** `fig05_robustness.png`
- **Script :** `tests/05_robustness/robustness_tests.py`
- **Données :** SPARC (175 galaxies) - RÉEL
- **Section LaTeX :** 5 - Results III: Robustness Testing
- **Label :** `\label{fig:robustness}`
- **Tests :**
  - Monte Carlo : 100% survie (1000 itérations)
  - Bootstrap : 96.5% U-shape, 95% CI exclut zéro
  - Jackknife : 4/4 environnements stables
  - Permutation : p < 0.001

### Figure 6 : Réplicabilité ★ CRITIQUE
- **Fichier :** `fig06_replicability.png`
- **Script :** `tests/06_replicability/replicability_tests.py`
- **Données :** ALFALFA (21,834) + Little THINGS (40) - RÉEL
- **Section LaTeX :** 6 - Results IV: Independent Replication
- **Label :** `\label{fig:replication}`
- **Résultats :**
  - SPARC : a = +1.36, p < 10⁻⁶
  - ALFALFA : a = +0.07, p = 0.0065 ✓ RÉPLIQUÉ
  - Little THINGS : a = +0.29, p = 0.19 (petit N)

### Figure 7 : Couplage microphysique
- **Fichier :** `fig07_microphysics.png`
- **Script :** `tests/07_microphysics/microphysics_analysis.py`
- **Données :** SPARC (175 galaxies) - RÉEL
- **Section LaTeX :** 7 - Results V: Microphysical Coupling
- **Label :** `\label{fig:microphysics}`
- **Message :** Q couple au gaz, R couple aux étoiles

### Figure 8 : Corrélation Q-HI
- **Fichier :** `fig08_q_hi_correlation.png`
- **Script :** `tests/08_microphysics_qhi/q_hi_analysis.py`
- **Données :** SPARC (175 galaxies) - RÉEL
- **Section LaTeX :** 7 - Results V: Microphysical Coupling
- **Label :** `\label{fig:q_hi}`
- **Résultats :** Corrélation Q-HI : r = 0.67

### Figure 9 : Contraintes système solaire
- **Fichier :** `fig09_solar_system.png`
- **Script :** `tests/09_solar_system/solar_system_constraints.py`
- **Données :** Théorique (calculs analytiques)
- **Section LaTeX :** 8 - Results VI: Solar System Constraints
- **Label :** `\label{fig:solar_system}`
- **Contraintes satisfaites :**
  - Eötvös : η < 10⁻¹³ ✓
  - PPN γ : |γ-1| < 2×10⁻⁵ ✓
  - Nordtvedt : supprimé ✓

### Figure 10 : Validation TNG ★ CRITIQUE
- **Fichier :** `fig10_tng_validation.png`
- **Script :** `tests/10_tng_validation/tng_validation.py`
- **Données :** TNG100-1 (53,363 galaxies) - RÉEL
- **Section LaTeX :** 9 - Results VII: Simulation Validation
- **Label :** `\label{fig:tng_validation}`
- **Résultats :**
  - ΛCDM seul : a = 0.045, p = 0.075 (borderline)
  - Avec QO+R : a = 0.039, p = 0.004 (significatif)

### Figure 11 : Killer Prediction ★ CRITIQUE
- **Fichier :** `fig11_tng_multiscale.png`
- **Script :** `tests/11_tng_multiscale/tng_multiscale.py`
- **Données :** TNG300-1 (623,609 galaxies) - RÉEL
- **Section LaTeX :** 9 - Results VII: Simulation Validation
- **Label :** `\label{fig:killer}`
- **Résultats :**
  - Gas-rich : a = +0.017 → U positif
  - Gas-poor + High M* : a = -0.014 → U INVERSÉ ★
  - Extreme R-dom : a = -0.019 → Inversion confirmée ★

### Figure 12 : Lien théorie des cordes
- **Fichier :** `fig12_string_theory.png`
- **Script :** `tests/12_string_theory_link/string_theory_link.py`
- **Données :** Théorique (diagrammes conceptuels)
- **Section LaTeX :** 10 - Theoretical Framework
- **Label :** `\label{fig:string_theory}`
- **Message :** Correspondance suggestive Q ↔ Dilaton, R ↔ Kähler modulus

### Figure 13 : Prédictions falsifiables
- **Fichier :** `fig13_predictions.png`
- **Script :** `tests/13_predictions/predictions.py`
- **Données :** Théorique (projections)
- **Section LaTeX :** 11 - Falsifiable Predictions
- **Label :** `\label{fig:predictions}`
- **Prédictions :**
  1. UDGs : résidus élevés partout
  2. Elliptiques isolées : U inversé
  3. WALLABY/SKA : détection à >10σ
  4. Courbes résolues : effet sur la forme

---

## COMMANDES DE COMPILATION

```bash
cd Git/Paper1-BTFR-UShape/manuscript

# Compilation complète
pdflatex paper1_qor_btfr_v3.tex
pdflatex paper1_qor_btfr_v3.tex  # 2x pour TOC et références
```

---

## CHECKLIST DE VÉRIFICATION

- [x] 13 figures incluses dans le LaTeX
- [x] Toutes les figures utilisent des données réelles (sauf théoriques)
- [x] Labels LaTeX définis pour chaque figure
- [x] Références croisées dans le texte
- [x] Scripts documentés pour chaque figure
- [x] Structure narrative cohérente (échec → découverte → validation → prédictions)

---

*Audit mis à jour : 2025-12-03*  
*Manuscrit : paper1_qor_btfr_v3.tex (version complète avec 13 figures)*
