# GC-001 : Test TOE sur les Amas Globulaires

**Document Type:** Théorie et Motivation  
**Author:** Jonathan Édouard Slama  
**Institution:** Metafund Research Division, Strasbourg, France  
**Date:** December 15, 2025  
**Status:** PRÉ-ENREGISTRÉ

---

## 1) Pourquoi Tester les Amas Globulaires ?

### 1.1 Le Gap dans la Validation Multi-Échelle

Notre validation actuelle couvre :

```
WB (10¹³-10¹⁶ m) → [GAP] → UDG (10²⁰ m) → Galactique (10²¹ m) → ...
```

Les **amas globulaires** (GC) se situent à **~10¹⁸ m**, exactement dans ce gap :

| Échelle | Distance | λ_QR | Status |
|---------|----------|------|--------|
| Wide Binaries | 10¹³-10¹⁶ m | 1.71 | ✅ |
| **Globular Clusters** | **10¹⁸ m** | **?** | ⬜ À TESTER |
| UDGs | 10²⁰ m | 0.83 | ✅ |
| Galactique | 10²¹ m | 0.94 | ✅ |

### 1.2 Pourquoi les GC sont Critiques

Les amas globulaires ont des propriétés **uniques** :

1. **Densité stellaire extrême** : ρ ~ 10³-10⁵ M☉/pc³ (vs ~0.1 M☉/pc³ voisinage solaire)
2. **Systèmes auto-gravitants** : dynamique dominée par gravité
3. **Pas de matière noire** : ratio M/L cohérent avec populations stellaires
4. **Données de haute qualité** : dispersions de vitesse précises (Gaia, spectro)

### 1.3 Ce que Prédit QO+R

Dans le cadre QO+R avec screening chameleon :

$$\alpha(\rho) = \alpha_0 \times \frac{1}{1 + \kappa \rho}$$

À **très haute densité** (GC), on attend :
- **Screening quasi-total** : α → 0
- **λ_QR proche de WB** : car même régime de haute densité
- **Pas d'anomalie gravitationnelle** : M_dyn ≈ M_baryon

**Prédiction quantitative :**
$$\lambda_{QR}^{GC} \approx 1.5 - 1.8 \quad (\text{proche WB } 1.71)$$

---

## 2) Ce qu'on Cherche

### 2.1 Observable Principal

$$\Delta_{GC} = \log\left(\frac{\sigma_{obs}}{\sigma_{bar}}\right)$$

où :
- σ_obs = dispersion de vitesse observée (centrale ou intégrée)
- σ_bar = dispersion prédite par la masse stellaire (viriel/King model)

### 2.2 Attendu Théorique

| Scénario | ⟨Δ_GC⟩ | λ_QR | Interprétation |
|----------|--------|------|----------------|
| **Screening total** | ≈ 0 | ~1.7 | GR standard, QO+R invisible |
| Screening partiel | 0.05-0.10 | ~1.2 | Signal faible détectable |
| Pas de screening | > 0.15 | ~0.9 | Signal fort (improbable) |

**Hypothèse nulle :** ⟨Δ_GC⟩ = 0 (pas d'excès de dispersion)

### 2.3 Test de la Transition WB → Galactique

Si λ_QR(GC) ≈ 1.7 (proche WB) :
- Confirme le **régime haute densité** (λ_QR ~ 1.7)
- Transition vers régime basse densité (λ_QR ~ 0.9) se fait **après** 10¹⁸ m

Si λ_QR(GC) ≈ 0.9 (proche galactique) :
- Transition se fait **avant** 10¹⁸ m
- Gap réel est plus petit qu'anticipé

---

## 3) Ce que ça Prouverait

### 3.1 Si Screening Confirmé (⟨Δ⟩ ≈ 0, λ_QR ~ 1.7)

✅ **Cohérence du modèle chameleon** : Le screening fonctionne comme prédit à haute densité

✅ **Pas de "cinquième force" visible** dans les GC : Cohérent avec tests de gravité locaux

✅ **Transition WB-Galactique localisée** : Entre 10¹⁸ et 10²⁰ m

### 3.2 Si Signal Détecté (⟨Δ⟩ > 0.05)

⚠️ **Screening moins efficace qu'attendu** : Révision du paramètre κ

⚠️ **Possible contribution non-baryonique** : MOND-like ou autres

### 3.3 Si Incohérence (λ_QR très différent)

❌ **Falsification partielle** : Le modèle ne capture pas la physique des GC

---

## 4) Données Disponibles

### 4.1 Sources de Données

| Source | N GC | Données | Qualité |
|--------|------|---------|---------|
| **Harris 2010** | 157 | σ_0, r_c, r_h, M_V | Catalogue de référence |
| **Baumgardt+18** | 112 | σ_los(r), M_dyn | Profils résolus |
| **Baumgardt+19** | 156 | Masses dynamiques | HST + Gaia |
| **Vasiliev+21** | 170 | Proper motions | Gaia EDR3 |

### 4.2 Observables Clés

Pour chaque GC :
- **σ_0** : Dispersion centrale (km/s)
- **r_c** : Rayon du cœur (pc)
- **r_h** : Rayon de demi-lumière (pc)
- **M_V** : Magnitude absolue → M_* via M/L
- **[Fe/H]** : Métallicité → correction M/L

### 4.3 Calcul de σ_bar

Modèle de King ou Plummer :

$$\sigma_{bar}^2 = \frac{G M_*}{r_h} \times \eta$$

où η ≈ 0.4 (facteur géométrique dépendant du profil)

---

## 5) Critères Pré-enregistrés

### 5.1 Échantillon

| Critère | Seuil | Justification |
|---------|-------|---------------|
| N_GC | ≥ 50 | Puissance statistique |
| Données σ | Spectroscopiques | Pas seulement photométriques |
| Qualité | err(σ)/σ < 20% | Précision suffisante |

### 5.2 Test Principal

**H0 :** r(Δ, log ρ) = 0 (pas de corrélation)
**H1 :** r(Δ, log ρ) ≠ 0

Seuil : p < 0.05

### 5.3 Critères de Succès

| Métrique | Seuil | Interprétation |
|----------|-------|----------------|
| ⟨Δ_GC⟩ | < 0.05 | Screening attendu |
| λ_QR | ∈ [1.3, 2.1] | Cohérent WB/Groups |
| p(r) | > 0.05 | Pas de signal Δ-ρ |

---

## 6) Méthodologie

### 6.1 Pipeline

```
1. Charger catalogue Harris/Baumgardt
2. Calculer M_* depuis M_V et M/L(metallicité)
3. Calculer σ_bar via modèle King/Plummer
4. Calculer Δ_GC = log(σ_obs/σ_bar)
5. Calculer densité ρ = M/(4/3 π r_h³)
6. Fit TOE : Δ = α × C(ρ, λ_QR, ρ_c, δ)
7. Bootstrap + permutation tests
```

### 6.2 Modèle Statistique

Identique à UDG V3.1 :
- Likelihood Student-t (robuste)
- Priors informatifs depuis multi-échelle
- Effets aléatoires optionnels (GC du bulge vs halo)

---

## 7) Résultats Attendus

### 7.1 Scénario Principal (Screening)

```
⟨Δ_GC⟩ = 0.02 ± 0.05
r(Δ, log ρ) = -0.05, p = 0.60
λ_QR = 1.65 ± 0.30
VERDICT : SCREENING CONFIRMÉ
```

### 7.2 Scénario Alternatif (Signal Faible)

```
⟨Δ_GC⟩ = 0.08 ± 0.04
r(Δ, log ρ) = -0.25, p = 0.04
λ_QR = 1.35 ± 0.25
VERDICT : SIGNAL MARGINAL
```

---

## 8) Intégration Multi-Échelle

### 8.1 Position dans le Schéma

```
                    HAUTE DENSITÉ          BASSE DENSITÉ
                         ↓                      ↓
                    λ_QR ~ 1.7              λ_QR ~ 0.9
                         
    WB ──── GC ────────────────────── UDG ── Gal ── Grp ── Clu
   10¹³   10¹⁸                       10²⁰  10²¹  10²²  10²³
    │       │                          │      │
    └───────┴──────── TRANSITION ──────┴──────┘
                     (10¹⁸ - 10²⁰ m)
```

### 8.2 Prédiction pour la Transition

La transition λ_QR ~ 1.7 → λ_QR ~ 0.9 devrait se produire entre GC et UDG.

**Si GC confirme λ_QR ~ 1.7** : La transition est bien entre 10¹⁸ et 10²⁰ m.

---

## 9) Fichiers à Créer

```
13_GLOBULAR_CLUSTERS/
├── GC_THEORY.md           # Ce document
├── gc001_toe_test.py      # Script principal
├── results/
│   └── gc001_results.json
└── figures/
    └── gc_delta_vs_rho.pdf
```

---

**Document Status:** THÉORIE PRÉ-ENREGISTRÉE  
**Prêt pour implémentation:** OUI  
**Date de verrouillage:** December 15, 2025
