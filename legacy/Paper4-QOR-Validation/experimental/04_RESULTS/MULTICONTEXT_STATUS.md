# Résultats Multi-Contextes : Bilan et Prochains Tests

**Date:** 7 décembre 2025  
**Status:** En cours - 2 succès, 1 tendance, 1 non-détection

---

## Résumé des tests effectués

### ✅ SUCCÈS : Galaxies individuelles

| Contexte | Observable | N | U-shape | Sign Inversion | σ |
|----------|------------|---|---------|----------------|---|
| **ALFALFA** | Courbes de rotation (v_rot) | 19,222 | ✅ | ✅ | 5σ+ |
| **KiDS DR4** | Relation M/L optique | 663,323 | ✅ | ✅ | 5.2σ |

**Caractéristiques communes :**
- Échelle : galaxies individuelles
- Baryons : 15-50% de la masse dynamique
- Environnement : densité locale de galaxies voisines

### ⚠️ TENDANCE : Planck × MCXC

| Métrique | Valeur |
|----------|--------|
| N clusters | 551 |
| Corrélation M-Y (corrigée) | r = +0.79 ✅ |
| U-shape | a = +0.020 ± 0.015 (1.3σ) |

**Limite :** Échantillon trop petit pour conclusion statistique

### ❌ NON-DÉTECTION : ACT-DR5 MCMF

| Métrique | Valeur |
|----------|--------|
| N clusters | 6,148 |
| Corrélation M-L | r = +0.57 ✅ |
| U-shape (densité 3D) | a = -0.60 (13σ) ∩-shape |
| U-shape (L/M ratio) | a = -0.08 (1.9σ) ~linéaire |

**Observations :**
- Signal fortement corrélé avec SNR → biais de sélection probable
- Relation résidus vs L/M est linéaire, pas parabolique
- Pas de sign inversion détecté

---

## Interprétations possibles

### Hypothèse 1 : Effet d'échelle (physique)
Les clusters sont dominés par la matière noire (85%+). Si Q²R² couple principalement aux baryons, le signal serait dilué.

**Test :** Chercher le signal dans les sous-structures baryoniques des clusters (BCGs, galaxies membres)

### Hypothèse 2 : Proxy inadéquat
La "densité de clusters voisins" n'est pas le bon environnement. Les clusters SONT les pics de densité.

**Test :** Utiliser d'autres définitions d'environnement (distance au filament le plus proche, densité de matière noire simulée)

### Hypothèse 3 : Biais de sélection
Le signal SZ crée des biais complexes (Malmquist, confusion, complétude variable).

**Test :** Utiliser des catalogues X-ray purs (eROSITA) ou optiques (redMaPPer)

### Hypothèse 4 : Statistique insuffisante pour les clusters
6,148 clusters vs 663,323 galaxies = facteur 100 de moins.

**Test :** Combiner plusieurs catalogues de clusters

---

## Plan de tests complémentaires

### A. CLUSTERS : Tests additionnels

#### A1. Autres catalogues de clusters SZ

| Catalogue | N | Source | Status |
|-----------|---|--------|--------|
| ACT-DR6 | 10,040 | LAMBDA (2025) | 📥 À télécharger |
| SPT-SZ + SPT-ECS | ~1,200 | SPT Collaboration | 📥 À télécharger |
| Planck PSZ2 complet | 1,653 | ESA | ✅ Disponible |

#### A2. Catalogues X-ray (masses indépendantes du SZ)

| Catalogue | N | Source | Avantage |
|-----------|---|--------|----------|
| **eROSITA eRASS1** | ~12,000 | MPE (2024) | Masses X-ray pures |
| MCXC-II | 2,221 | VizieR | Extension de MCXC |
| XXL | ~365 | XMM-Newton | Haute qualité |

#### A3. Catalogues optiques (pas de biais SZ/X-ray)

| Catalogue | N | Source | Avantage |
|-----------|---|--------|----------|
| **redMaPPer SDSS** | ~26,000 | SDSS DR8 | Richesse optique pure |
| **redMaPPer DES** | ~75,000 | DES Y3 | Plus profond |
| CAMIRA HSC | ~2,000 | HSC-SSP | Haute qualité |

#### A4. Tests dans les clusters (sous-structures)

| Test | Description |
|------|-------------|
| BCGs | Relation M*/L des galaxies centrales |
| Galaxies membres | Distribution de masses stellaires |
| Profils de gaz | Gradient de fraction de gaz vs rayon |

### B. AUTRES ÉCHELLES

#### B1. Groupes de galaxies (intermédiaire)

| Catalogue | N | Masse typique | Source |
|-----------|---|---------------|--------|
| **GAMA groups** | ~24,000 | 10¹²-10¹⁴ M☉ | GAMA DR4 |
| **Yang groups (SDSS)** | ~470,000 | 10¹¹-10¹⁵ M☉ | Yang+2007 |
| **2MRS groups** | ~30,000 | 10¹²-10¹⁴ M☉ | Tully+2015 |

#### B2. Échelle cosmologique

| Test | Observable | Source |
|------|------------|--------|
| **Lensing CMB × clusters** | Corrélation κ-Y | Planck + ACT |
| **BAO residuals** | Oscillations acoustiques | DESI/SDSS |
| **Weak lensing cosmic shear** | Cisaillement vs environnement | KiDS/DES/HSC |

#### B3. Galaxies : autres surveys

| Survey | N galaxies | Spécificité |
|--------|------------|-------------|
| **DESI** | ~40M | Spectro, z précis |
| **WALLABY** | ~500k (prévu) | HI 21cm, f_gas direct |
| **MaNGA** | ~10,000 | Cinématique spatiale |
| **SAMI** | ~3,000 | Cinématique + environnement |

### C. TESTS DE ROBUSTESSE

| Test | Objectif |
|------|----------|
| Bootstrap | Stabilité statistique |
| Jackknife spatial | Effet de régions du ciel |
| Mock catalogs | Comparaison avec simulations sans Q²R² |
| Bins de redshift | Évolution avec le temps cosmique |
| Bins de masse | Dépendance en échelle |

---

## Priorités recommandées

### Court terme (décembre 2025)

1. **eROSITA eRASS1** (~12,000 clusters X-ray)
   - Masses indépendantes du SZ
   - Pas de biais Malmquist type Planck
   - Données publiques depuis 2024

2. **redMaPPer DES** (~75,000 clusters optiques)
   - Plus grand catalogue optique
   - Richesse = proxy de masse indépendant
   - Environnement bien défini

3. **GAMA groups** (~24,000 groupes)
   - Échelle intermédiaire galaxies/clusters
   - Masses dynamiques
   - Environnement spectroscopique

### Moyen terme (Q1 2026)

4. **ACT-DR6** (10,040 clusters SZ)
5. **Yang groups SDSS** (470,000 groupes)
6. **Bins de redshift dans KiDS** (évolution)

### Long terme (2026+)

7. WALLABY (HI direct)
8. DESI (spectroscopie massive)
9. CMB lensing cross-correlations

---

## Scripts à créer

```
tests/future/
├── download_erosita_erass1.py      # Clusters X-ray
├── download_redmapper_des.py       # Clusters optiques
├── download_gama_groups.py         # Groupes
├── download_yang_groups.py         # Groupes SDSS
├── download_act_dr6.py             # Clusters SZ (quand disponible)
├── test_cluster_erosita.py         # Test eROSITA
├── test_cluster_redmapper.py       # Test redMaPPer
├── test_groups_gama.py             # Test groupes
├── test_kids_zbins.py              # KiDS en bins de z
└── compare_all_contexts.py         # Synthèse multi-échelles
```

---

## Conclusion provisoire

**Ce qu'on sait :**
- Q²R² confirmé à l'échelle des galaxies (2 contextes, >5σ)
- Non détecté dans ACT-DR5 MCMF (clusters SZ)
- Tendance positive dans Planck × MCXC (besoin plus de données)

**Ce qu'on ne sait pas encore :**
- Le signal existe-t-il dans les clusters X-ray ou optiques ?
- Quelle est la dépendance en échelle (groupes vs clusters) ?
- Le signal évolue-t-il avec le redshift ?

**Prochaine étape immédiate :** Tester eROSITA eRASS1 et redMaPPer DES

---

**Author:** Jonathan Édouard Slama
**Date:** December 7, 2025
