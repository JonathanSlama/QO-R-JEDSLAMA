# FIL-001 : Test TOE sur les Filaments Cosmiques

**Document Type:** Théorie et Motivation  
**Author:** Jonathan Édouard Slama  
**Institution:** Metafund Research Division, Strasbourg, France  
**Date:** December 15, 2025  
**Status:** PRÉ-ENREGISTRÉ

---

## 1) Pourquoi Tester les Filaments Cosmiques ?

### 1.1 Le Gap dans la Validation Multi-Échelle

Notre validation actuelle couvre :

```
... → Clusters (10²³ m) → [GAP] → Cosmique (10²⁵-10²⁷ m)
```

Les **filaments cosmiques** se situent à **~10²⁴ m**, exactement dans ce gap :

| Échelle | Distance | λ_QR | Status |
|---------|----------|------|--------|
| Clusters | 10²³ m | 1.23 | ⚠️ Screened |
| **Filaments** | **10²⁴ m** | **?** | ⬜ À TESTER |
| Cosmique (CMB) | 10²⁵-10²⁷ m | ~1.1 | ⬜ Data-limited |

### 1.2 Pourquoi les Filaments sont Critiques

Les filaments cosmiques ont des propriétés **uniques** :

1. **Densité intermédiaire** : ρ ~ 10-100 × ρ_moyen (entre vides et clusters)
2. **Structures les plus grandes** : Longueurs 10-100 Mpc
3. **Tracent la toile cosmique** : Connectent clusters, contiennent galaxies
4. **Régime non-linéaire** : Formation de structures en cours

### 1.3 Ce que Prédit QO+R

Dans le cadre QO+R avec screening chameleon :

À **densité intermédiaire** (filaments), on attend :
- **Screening partiel** : α intermédiaire entre clusters et vides
- **Signal Δ(ρ) détectable** : Comme dans les UDGs
- **λ_QR ~ O(1)** : Continuité avec autres échelles

**Prédiction quantitative :**
$$\lambda_{QR}^{FIL} \approx 1.0 - 1.3 \quad (\text{entre clusters et cosmique})$$

---

## 2) Ce qu'on Cherche

### 2.1 Observable Principal : Profils de Densité

Les filaments montrent des **profils de densité transverses** :

$$\rho(r_\perp) = \rho_0 \times f(r_\perp / r_{fil})$$

où r_⊥ est la distance au centre du filament.

### 2.2 Test QO+R : Excès de Vitesse

Les galaxies dans les filaments ont des vitesses :
- **Infall** vers le filament (composante radiale)
- **Streaming** le long du filament (composante longitudinale)

**Observable TOE :**
$$\Delta_{FIL} = \log\left(\frac{v_{obs}}{v_{pred}}\right)$$

où v_pred vient du modèle ΛCDM standard.

### 2.3 Attendu Théorique

| Scénario | ⟨Δ_FIL⟩ | Interprétation |
|----------|---------|----------------|
| **Signal QO+R** | 0.05-0.15 | Excès de velocités dans les filaments |
| Pas de signal | ≈ 0 | ΛCDM suffit |
| Signal inverse | < 0 | Problème avec le modèle |

---

## 3) Ce que ça Prouverait

### 3.1 Si Signal Détecté (⟨Δ⟩ > 0.05, γ < 0)

✅ **Extension du screening QO+R** aux échelles cosmiques

✅ **Continuité avec UDGs** : Même signature Δ(ρ) à plus grande échelle

✅ **Alternative à la matière noire** dans les filaments

### 3.2 Si Pas de Signal (⟨Δ⟩ ≈ 0)

⚠️ **Screening efficace** même à densité intermédiaire

⚠️ **Besoin d'aller aux vides cosmiques** pour voir le signal

### 3.3 Si Incohérence

❌ **Problème de transition** Clusters → Cosmique

---

## 4) Données Disponibles

### 4.1 Catalogues de Filaments

| Source | Données | Couverture |
|--------|---------|------------|
| **DisPerSE** | Filaments SDSS | z < 0.1 |
| **SDSS DR16** | Galaxies + redshifts | ~10⁶ galaxies |
| **DESI Y1** | Spectres + velocités | ~10⁷ objets |
| **Tempel+14** | Catalogue filaments | SDSS footprint |

### 4.2 Observables Clés

Pour chaque filament :
- **Galaxies membres** : Positions, z_spec
- **Profil ρ(r_⊥)** : Densité transverse
- **Champ de vitesses** : v_pec des galaxies
- **Environnement** : Distance aux clusters

### 4.3 Méthode de Détection

1. **Identifier filaments** via DisPerSE sur champ de densité
2. **Assigner galaxies** : Distance au squelette < r_max
3. **Calculer profils** : ρ(r_⊥), v(r_⊥)
4. **Comparer à ΛCDM** : Simulations N-body

---

## 5) Critères Pré-enregistrés

### 5.1 Échantillon

| Critère | Seuil | Justification |
|---------|-------|---------------|
| N_filaments | ≥ 100 | Statistique robuste |
| N_galaxies/filament | ≥ 20 | Profils résolus |
| z_range | 0.01 - 0.15 | Spectro SDSS/DESI |

### 5.2 Test Principal

**Corrélation Δ vs log(ρ)** comme pour UDGs :

**H0 :** r(Δ, log ρ) = 0
**H1 :** r(Δ, log ρ) < 0 (anti-corrélation)

Seuil : p < 0.01

### 5.3 Critères de Succès

| Métrique | Seuil | Interprétation |
|----------|-------|----------------|
| r(Δ, log ρ) | < -0.3 | Signal significatif |
| γ | CI exclut 0 | Screening confirmé |
| λ_QR | ∈ [0.5, 2.0] | Cohérent multi-échelle |

---

## 6) Méthodologie

### 6.1 Approche Simplifiée (V1)

Utiliser les **stacked profiles** publiés :

1. Charger profils ρ(r_⊥) de la littérature
2. Calculer densité moyenne par bin
3. Extraire velocités peculiaires des galaxies
4. Calculer excès vs prédiction ΛCDM
5. Fit TOE sur Δ(ρ)

### 6.2 Approche Complète (V2)

Pipeline full sur données SDSS/DESI :

1. Télécharger catalogue galaxies spectro
2. Appliquer DisPerSE pour identifier filaments
3. Assigner galaxies aux filaments
4. Calculer profils observés
5. Comparer à mocks ΛCDM
6. Fit TOE hiérarchique

### 6.3 Données Publiques Utilisables

| Référence | Données | Accès |
|-----------|---------|-------|
| Tempel+14 | Filaments SDSS DR10 | VizieR |
| Chen+16 | Profils stacked | Tables papier |
| Kraljic+18 | Filaments GAMA | Public |
| Malavasi+20 | Velocités filaments | VIPERS |

---

## 7) Résultats Attendus

### 7.1 Scénario Principal (Signal Détecté)

```
⟨Δ_FIL⟩ = 0.08 ± 0.03
r(Δ, log ρ) = -0.45, p = 0.002
γ = -0.12 ± 0.04 (CI exclut 0)
λ_QR = 1.15 ± 0.35
VERDICT : QO+R SUPPORTÉ DANS LES FILAMENTS
```

### 7.2 Scénario Alternatif (Pas de Signal)

```
⟨Δ_FIL⟩ = 0.02 ± 0.04
r(Δ, log ρ) = -0.10, p = 0.30
VERDICT : SCREENING EFFICACE À CETTE ÉCHELLE
```

---

## 8) Intégration Multi-Échelle

### 8.1 Position dans le Schéma

```
              GALAXIES              COSMIQUE
                 ↓                      ↓
    UDG ── Gal ── Grp ── Clu ── FIL ── CMB
   10²⁰  10²¹  10²²  10²³  10²⁴  10²⁷
                          │       │
                          └───────┘
                      TRANSITION ?
```

### 8.2 Test de Continuité

Si FIL confirme le signal Δ(ρ) :
- **Continuité UDG → FIL** : Même physique de screening
- **λ_QR(FIL) ≈ 1.0-1.3** : Entre clusters et cosmique

---

## 9) Comparaison avec UDG-001

| Aspect | UDG-001 | FIL-001 |
|--------|---------|---------|
| Échelle | 10²⁰ m | 10²⁴ m |
| Densité | 10⁻²⁶ g/cm³ | 10⁻²⁸ g/cm³ |
| N objets | 23 | ~100+ filaments |
| Observable | σ_los | v_pec |
| Signal attendu | γ < 0 ✅ | γ < 0 ? |

**Hypothèse :** Si le screening dépend de la densité, on devrait voir un signal **plus fort** dans les filaments (densité plus faible).

---

## 10) Défis et Limitations

### 10.1 Difficultés

| Défi | Impact | Mitigation |
|------|--------|------------|
| Définition filaments | Dépend de l'algorithme | Tester DisPerSE vs MST |
| Projection | Contamination LOS | Coupes en z |
| Prédiction ΛCDM | Incertitudes simulations | Plusieurs mocks |
| Sélection galaxies | Biais Malmquist | Volume-limited |

### 10.2 Approche Conservatrice

Commencer par les **données publiées** (profils stacked) avant d'attaquer les catalogues bruts.

---

## 11) Fichiers à Créer

```
14_COSMIC_FILAMENTS/
├── FIL_THEORY.md           # Ce document
├── fil001_toe_test.py      # Script principal
├── data/
│   └── filament_profiles.csv
├── results/
│   └── fil001_results.json
└── figures/
    └── fil_delta_vs_rho.pdf
```

---

**Document Status:** THÉORIE PRÉ-ENREGISTRÉE  
**Prêt pour implémentation:** OUI  
**Date de verrouillage:** December 15, 2025
