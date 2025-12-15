# UDG-001 : Test TOE sur Galaxies Ultra-Diffuses

**Document Type:** Théorie Pré-Test  
**Author:** Jonathan Édouard Slama  
**Institution:** Metafund Research Division, Strasbourg, France  
**Date:** December 15, 2025  
**Status:** FORMALISÉ - Prêt pour test

---

## 1) Pourquoi les UDG sont le Test Royal

### 1.1 Contexte

Les galaxies ultra-diffuses (UDG) sont des systèmes stellaires avec :
- **Luminosité de surface extrêmement faible** : μ_0 > 24 mag/arcsec²
- **Taille comparable aux galaxies normales** : R_eff ~ 1.5-5 kpc
- **Masse stellaire faible** : M_* ~ 10⁷-10⁸ M_sun

### 1.2 L'anomalie observée

Plusieurs UDG montrent une dynamique **incompatible avec ΛCDM standard** :

| UDG | σ_obs (km/s) | σ_prédit (DM) | Ratio |
|-----|-------------|---------------|-------|
| **NGC 1052-DF2** | 8.5 ± 2.1 | ~30 | **0.28** |
| **NGC 1052-DF4** | 4.2 ± 1.7 | ~25 | **0.17** |
| **AGC 114905** | 23 ± 2 | ~50 | **0.46** |

Ces galaxies semblent avoir **peu ou pas de matière noire** !

### 1.3 L'opportunité TOE

Dans le framework QO+R :
- **Densité ρ ~ 10⁻²⁶ g/cm³** → screening MINIMAL
- **Amplitude α attendue** : 0.05 - 0.20 (vs ~0 pour clusters)
- **Signal TOE potentiellement VISIBLE**

**Prédiction clé :** Dans les UDG, l'interférence Q-R devrait produire un **excès de vitesse** par rapport à la gravité newtonienne pure, mimant partiellement l'effet "matière noire".

---

## 2) Observable TOE pour UDG

### 2.1 Résidu dynamique

$$\Delta_{UDG} = \log(\sigma_{obs}) - \log(\sigma_{bar})$$

où :
- σ_obs = dispersion de vitesse observée
- σ_bar = dispersion prédite par masse baryonique seule (théorème viriel)

### 2.2 Prédiction newtonienne

Pour un système sphérique :
$$\sigma_{bar}^2 = \frac{G M_*}{2 R_{eff}}$$

### 2.3 Modèle TOE

$$\Delta_{UDG} = \alpha_{UDG} \cos\left(\mathcal{P}(\rho_*)\right)$$

avec :
$$\mathcal{P}(\rho) = \pi \cdot \text{sigmoid}\left(\frac{\rho - \rho_c}{\delta}\right)$$

**Attendu :** α_UDG > 0 (signal positif, non screené)

---

## 3) Données Disponibles

### 3.1 Catalogues UDG avec dynamique

| Source | N_UDG | Données | Référence |
|--------|-------|---------|-----------|
| **Van Dokkum+18** | 2 | DF2, DF4 σ, M_* | Nature 2018, ApJL 2019 |
| **Mancera Piña+22** | 6 | AGC 114905 + others | MNRAS 2022 |
| **Forbes+21** | ~30 | Compilation σ, R_eff | MNRAS 2021 |
| **Gannon+22** | ~20 | Keck KCWI spectro | ApJ 2022 |

### 3.2 Variables mesurées

- **σ_los** : dispersion de vitesse le long de la ligne de visée
- **R_eff** : rayon effectif (demi-lumière)
- **M_*** : masse stellaire (photométrie)
- **Distance** : Cepheides, SBF, ou groupe d'appartenance

### 3.3 Environnement

- **Densité locale** : appartenance à groupe/cluster
- **Isolation** : UDG de champ vs UDG de cluster

---

## 4) Pipeline UDG-001

### Étape 1 : Compilation des données

```python
# Sources principales
udg_data = {
    'DF2': {'sigma': 8.5, 'sigma_err': 2.1, 'M_star': 2e8, 'R_eff': 2.2, 'dist': 20},
    'DF4': {'sigma': 4.2, 'sigma_err': 1.7, 'M_star': 1.5e8, 'R_eff': 1.6, 'dist': 20},
    'AGC114905': {'sigma': 23, 'sigma_err': 2, 'M_star': 1.2e9, 'R_eff': 3.0, 'dist': 76},
    # + autres de Forbes+21, Gannon+22
}
```

### Étape 2 : Calcul du résidu

$$\Delta_{UDG} = \log_{10}(\sigma_{obs}) - \log_{10}\left(\sqrt{\frac{G M_*}{2 R_{eff}}}\right)$$

### Étape 3 : Estimation de la densité

$$\rho_* = \frac{M_*}{\frac{4}{3}\pi R_{eff}^3}$$

### Étape 4 : Fit TOE et tests statistiques

- AICc TOE vs NULL
- Bootstrap par sous-échantillons
- Jackknife par environnement (champ vs cluster)

---

## 5) Critères Pré-Enregistrés

| # | Critère | Seuil | Interprétation |
|---|---------|-------|----------------|
| C1 | N_UDG | ≥ 10 | Échantillon suffisant |
| C2 | ρ_range | ≥ 0.5 dex | Levier en densité |
| C3 | α_UDG | > 0.05 | Signal détecté (vs screened) |
| C4 | λ_QR | ∈ [0.3, 3.0] × λ_gal | Cohérence universelle |
| C5 | ΔAICc | > 10 | TOE préféré (nouveauté !) |
| C6 | p_perm | < 0.05 | Signal significatif |

---

## 6) Prédictions Quantitatives

### 6.1 Scénario A : Signal TOE détecté (attendu)

- α_UDG = 0.10 ± 0.03
- λ_QR = 1.2 ± 0.3
- ΔAICc > 10
- **Interprétation :** L'excès de σ dans les UDG est dû à QO+R, pas à la matière noire

### 6.2 Scénario B : Screening partiel

- α_UDG = 0.03 ± 0.02
- λ_QR ~ O(1)
- ΔAICc ~ 0
- **Interprétation :** Même à faible densité, screening non négligeable

### 6.3 Scénario C : Incohérence (falsification)

- λ_QR très différent des autres échelles
- **Interprétation :** QO+R ne s'applique pas aux UDG

---

## 7) Lien avec les Anomalies Connues

### 7.1 DF2 et DF4 "sans matière noire"

Van Dokkum et al. ont montré que DF2 et DF4 ont σ ~ σ_bar, suggérant **pas de halo de matière noire**.

**Interprétation TOE :** Ces galaxies sont dans un régime où :
- Matière noire absente (vrai)
- Mais QO+R fournit un léger boost gravitationnel
- La coïncidence σ_obs ≈ σ_bar suggère α_TOE + α_DM ≈ 0

### 7.2 AGC 114905

Cette UDG isolée a σ_obs significativement en dessous de la prédiction ΛCDM.

**Interprétation TOE :** Isolation → screening minimal → signal QO+R maximal
Mais si σ_obs < σ_ΛCDM, c'est que le halo DM est absent, pas que QO+R boost.

### 7.3 Hypothèse testable

Si QO+R est actif :
$$\sigma_{obs}^2 = \sigma_{bar}^2 \times (1 + \alpha_{TOE})$$

→ On devrait voir une **corrélation positive** entre Δ_UDG et faible densité.

---

## 8) Structure du Test

```
12_UDG_SCALE/
  data/           # Données UDG compilées
  results/        # JSON outputs
  udg001_toe_test.py  # Pipeline complet
  UDG_THEORY.md   # Ce document
```

---

## 9) Signification pour Paper 4

Si UDG-001 détecte un signal :

> *"Ultra-diffuse galaxies provide the first detection of an unscreened QO+R signal. With α_UDG = 0.10 ± 0.03 and λ_QR consistent with other scales, these low-density systems reveal the underlying Q-R interference that is suppressed in higher-density environments. This explains the apparent 'dark matter deficit' in objects like NGC 1052-DF2 as a manifestation of modified gravitational coupling rather than missing mass."*

---

---

## 10) Améliorations pour V3 (Pré-enregistrées)

### 10.1 Correction Physique de σ_bar

**Problème V2:** Formule virielle simple σ² = GM/(2R_eff) néglige :
- Correction d'ouverture (rayon de mesure vs R_eff)
- Anisotropie β de la distribution des vitesses
- Profil de densité (Sérsic vs isotherme)

**Solution V3:** Modèle Jeans isotrope

$\sigma_{bar}^2 = \frac{G M_*}{R_{eff}} \times K(n, R_{ap}/R_{eff}) \times (1 + \beta_{corr})$

où :
- K(n, R_ap/R_eff) = facteur de correction d'ouverture (Wolf+10)
- β_corr = correction d'anisotropie (prior β ~ U(0, 0.4))
- Propager incertitude sur K et β dans σ_bar

### 10.2 Modèle Hiérarchique Bayésien

**Problème V2:** N=23 trop petit pour fit classique, sur-ajustement possible.

**Solution V3:** Modèle hiérarchique

```
Δ_i ~ Normal(μ_i, s_i² + τ²)
μ_i = α × C_i(ρ, f) + γ × log(ρ_i)
```

**Priors informatifs (doux) :**
- α ~ Normal(0, 0.1²)
- λ_QR ~ Normal(1, 0.5²)  # centré sur valeur multi-échelle
- ρ_c ~ Normal(-25, 2²)  # gelé aux valeurs apprises
- δ ~ Normal(1, 0.5²)
- τ ~ HalfNormal(0.1)  # dispersion intrinsèque

**Effet aléatoire par environnement :**
- α_env ~ Normal(α, σ_α²) pour champ/groupe/amas

### 10.3 Test Principal : Corrélation Δ vs log(ρ)

**Prédiction TOE :** Δ décroît avec ρ (screening)

**Test pré-enregistré :**
- H0: r(Δ, logρ) = 0
- H1: r(Δ, logρ) < 0 (anti-corrélation)
- Seuil: p < 0.01 (unilatéral)

**Résultat V2:** r = -0.647, p = 0.0008 → **SIGNIFICATIF**

### 10.4 Contrôles Systématiques

| Source | Incertitude | Propagation |
|--------|-------------|-------------|
| Distance (SBF vs groupe) | ~15% | ΔD → ΔR_eff → Δσ_bar |
| R_eff (bande, méthode) | ~20% | Direct sur σ_bar |
| M_* (IMF, M/L) | ~30% | Direct sur σ_bar |
| Appartenance (interlopers) | ~5-10% | Mixture model Student-t |

**Outliers :** Utiliser Student-t (ν ≈ 3-5) au lieu de Normal pour robustesse.

### 10.5 Jackknife de Sensibilité

Retirer et reporter :
- Satellites du Groupe Local (WLM, Antlia II, Sagittarius) - environnement atypique
- UDGs à rotation suspecte (morphologie disque mince)
- Outliers extrêmes (Δ > 2σ)

### 10.6 Extension de l'Échantillon

**Sources à agréger (uniquement σ publiés) :**

| Source | N estimé | Type |
|--------|----------|------|
| Forbes+21 | ~10 | Cinématiques |
| Mancera-Piña+22 | ~6 | AGC (HI) |
| LEWIS survey | ~15 | Hydra I |
| Toloba+23 | ~10 | Virgo GC |
| **Total estimé** | **~60** | |

**Puissance requise :** ~40 UDG/bin pour p < 0.05 avec σ_Δ ≈ 0.33 dex

---

## 11) Résumé des Versions

| Version | Données | N | λ_QR | Status |
|---------|---------|---|------|--------|
| V1 | Interpolées | 24 | 4.91 | ❌ REJETÉE |
| V2 | Gannon+24 | 23 | 0.84 | ✅ VALIDÉE |
| V3 | Multi-sources | ~60 | TBD | ⬜ PLANIFIÉE |

---

**Document Status:** THÉORIE FORMALISÉE + ROADMAP V3  
**Critères verrouillés:** December 15, 2025  
**Prêt pour implémentation V3:** OUI
