# TOE Échelle Stellaire - Wide Binaries (Gaia DR3)

**Document Type:** Théorie Pré-Test  
**Author:** Jonathan Édouard Slama  
**Institution:** Metafund Research Division, Strasbourg, France  
**Date:** December 14, 2025  
**Status:** FORMALISÉ - Prêt pour test

---

## 1) Contexte : Descente vers l'échelle stellaire

| Échelle | Distance | λ_QR | Status |
|---------|----------|------|--------|
| Galactique | 10²¹ m | 0.94 | ✅ Confirmé |
| Groupes | 10²² m | 1.67 | ⚠️ Partiellement supporté |
| Clusters | 10²³ m | 1.23 | ⚠️ Screened |
| **Wide Binaries** | **10¹³-10¹⁶ m** | — | ⬜ À tester |

Les Wide Binaries (WB) offrent un test unique :
- Séparations 10³-10⁵ AU (~10¹⁴-10¹⁶ m)
- Régime **sub-galactique** où le screening devrait être maximal
- Mais possibilité de signal résiduel dans les environnements **ultra-dilués**

---

## 2) Prédictions TOE à cette échelle

### 2.1 Couplage attendu

$$\lambda_{QR}^{WB} \sim \mathcal{O}(1)$$

Le couplage fondamental ne dépend pas de l'échelle.

### 2.2 Amplitude attendue

**Très faible** si l'écran (confinement Yukawa + caméléon) est efficace au voisinage stellaire :

$$|\alpha_{WB}| \ll |\alpha_{gal}| \approx 0.05$$

Signal proche de 0 dans les régions denses ; possibilité d'un **excès faiblement positif** dans les environnements ultra-dilués (halo galactique, haute latitude).

### 2.3 Observable analogue

$$\Delta_{WB} \equiv \log a_{obs} - \log a_{Kep}(M_{tot}, r_\perp)$$

où :
- $a_{obs}$ = demi-grand axe observé (ou séparation projetée $r_\perp$)
- $a_{Kep}$ = demi-grand axe Képlérien attendu pour la masse totale
- $M_{tot} = M_1 + M_2$ estimé par couleurs/HRD

### 2.4 Modèle TOE

$$\Delta_{WB} = \alpha_{WB} \cos\left(\mathcal{P}(\rho_\star)\right), \quad \alpha_{WB} \propto A_Q A_R$$

avec $\rho_\star$ = densité stellaire locale.

### 2.5 Critère de falsification

**FALSIFIÉ si :**
- $\alpha_{WB}$ reste compatible 0 pour **tous** les quantiles de $\rho_\star$
- ET les coupes "environnements ultra-dilués" n'augmentent pas significativement le signal

---

## 3) Données & Features

### 3.1 Catalogue Wide Binaries

**Source :** Gaia DR3 wide binary catalogs
- El-Badry et al. (2021) : ~1.3 million pairs
- Hartman & Lépine (2020) : ~100k high-confidence pairs

**Critères de sélection :**
- Parallaxe/PM cohérents (Δπ/π < 0.1, Δμ < 2 mas/yr)
- Séparation projetée : 10³ - 10⁵ AU
- RUWE < 1.4 (qualité astrométrique)
- Exclusion des triples connus/suspects

### 3.2 Variables

| Variable | Source | Description |
|----------|--------|-------------|
| $r_\perp$ | Gaia DR3 | Séparation projetée (AU) |
| $M_1, M_2$ | Couleurs + HRD | Masses stellaires |
| $M_{tot}$ | Dérivé | Masse totale du système |
| $\rho_\star$ | Counts-in-cylinder | Densité stellaire locale |
| $|b|$ | Coordonnées | Latitude galactique |
| RUWE | Gaia DR3 | Qualité astrométrique |

### 3.3 Environnement $\rho_\star$

Counts-in-cylinder depuis le champ Gaia local :
- R = 20-50 pc
- $|\Delta v| < 3$ km/s (vitesse radiale si disponible)
- Normalisé par la complétude locale

---

## 4) Pipeline d'analyse (miroir de GRP/CLU)

### Étape 1 : Qualité + mass-matching

- Binning par $M_{tot}$ et $r_\perp$
- Construction de $a_{Kep}$ de référence (régression orthogonale 2D)

### Étape 2 : Calcul de Δ_WB

$$\Delta_{WB} = \log a_{obs} - f(M_{tot}, r_\perp)$$

Résidu orthogonal pour éliminer les corrélations spurious.

### Étape 3 : Densité locale $\rho_\star$

- Quantiles Q20/Q80 pour définir low/high env
- Ou variable continue pour le fit

### Étape 4 : Fit TOE vs NULL

$$\Delta_{WB} = \alpha \cos\left(\mathcal{P}(\rho_\star; \rho_c, \Delta)\right) \quad \text{vs} \quad \Delta_{WB} = a + b\rho_\star$$

→ AICc, hold-out, bootstrap, permutation (identiques à v3)

---

## 5) Tests pré-enregistrés

| # | Test | Seuil | Interprétation |
|---|------|-------|----------------|
| T1 | Phase-flip low→high $\rho_\star$ | Détecté | Attendu : atténuation forte |
| T2 | Corrélation $\Delta_{WB}$–isolement | $\rho_\star^{-1} > 0$ en dilué | Signal résiduel |
| T3 | $\lambda_{QR}$ dans $[1/3, 3]$ | Cohérence | Multi-échelle |
| T4 | Robustesse RUWE < 1.2, \|b\| > 20° | Stable | Pas d'artefact |

---

## 6) Garde-fous systématiques

### 6.1 Biais de projection $r_\perp$ vs $r_{3D}$

- Simulé par tirages géométriques
- Correction moyenne appliquée aux bins

### 6.2 Multiples non résolus / triples

- Coupes sévères RUWE + astrométrie accélérée
- Test "exclusion 5% paires les plus suspectes"

### 6.3 Champ Galactique

- Contrôle par latitude $|b|$ et extinction
- Répéter l'analyse séparément pour $|b| > 40°$

### 6.4 Zéro-point parallaxe

- Appliquer zpt DR3
- Re-faire le fit avec ±10 μas pour tester la stabilité

---

## 7) Prédictions quantitatives

### 7.1 Si screening efficace (attendu)

- $\alpha_{WB} \approx 0$ (non détectable)
- $\lambda_{QR}$ mal contraint mais compatible O(1)
- ΔAICc < 0 (NULL préféré)
- p_perm ~ 1.0

### 7.2 Si signal résiduel détecté (bonus)

- $|\alpha_{WB}| > 0$ dans le quantile Q10 de $\rho_\star$
- Excès possible pour $|b| > 50°$ (halo galactique)
- Ce serait une **découverte majeure**

---

## 8) Structure de dépôt

```
10_WIDE_BINARIES/
  data/              # caches Gaia WB + champs locaux
  results/           # JSON outputs
  wb001_toe_test.py  # pipeline complet (v1)
  WB_THEORY.md       # ce document
```

---

## 9) Comparaison multi-échelle attendue

| Échelle | λ_QR | Amplitude α | Prédiction |
|---------|------|-------------|------------|
| Galactique | 0.94 | ~0.05 | Fort ✅ |
| Groupes | 1.67 | ~0.03 | Modéré ⚠️ |
| Clusters | 1.23 | ~0.04 | Screened ⚠️ |
| **Wide Binaries** | ~1? | **~0** | **Screened** |

---

**Document Status:** THÉORIE FORMALISÉE  
**Prêt pour test:** OUI  
**Critères verrouillés:** December 14, 2025
