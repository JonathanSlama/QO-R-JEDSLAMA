# TOE Échelle Clusters - Formalisation Théorique

**Document Type:** Théorie Pré-Test  
**Author:** Jonathan Édouard Slama  
**Institution:** Metafund Research Division, Strasbourg, France  
**Date:** December 14, 2025  
**Status:** FORMALISÉ - Prêt pour test

---

## 1) Contexte : Transition Groupes → Clusters

| Échelle | Distance | N testé | λ_QR | Status |
|---------|----------|---------|------|--------|
| Galactique | 10²¹ m | 19,222 | 0.94 | ✅ Confirmé |
| Groupes | 10²² m | 37,359 | 1.67 | ⚠️ Partiellement supporté |
| **Clusters** | **10²³ m** | — | — | ⬜ À tester |

---

## 2) Lagrangien TOE « Clusters »

On adapte l'action 4D aux clusters en remplaçant les densités baryoniques :

$$S_{QO+R}^{(cl)} = \int d^4x \sqrt{-g} \left[ \frac{M_{Pl}^2}{2} R - \mathcal{L}_Q - \mathcal{L}_R - V(Q,R) - \mathcal{L}_{int} - \mathcal{L}_{matter} \right]$$

### Champs effectifs pour clusters :

- **Q_cluster** : mode *dilaton-like* couplé à l'**ICM** (Intra-Cluster Medium)
  - $\mathcal{J}_{ICM}^{(cl)} \propto \rho_{ICM}^{(cl)}$ (gaz chaud X-ray/SZ)

- **R_cluster** : mode *Kähler-like* couplé au **contenu stellaire**
  - $\mathcal{J}_\star^{(cl)} \propto \rho_{BCG}^{(cl)} + \rho_{sat}^{(cl)}$ (BCG + satellites)

### Mapping des variables :

| Groupes | Clusters | Observable |
|---------|----------|------------|
| $\rho_{HI}^{(g)}$ | $\rho_{ICM}^{(cl)}$ | Signal SZ (Y) ou Lx |
| $\rho_\star^{(g)}$ | $\rho_{BCG+sat}^{(cl)}$ | Richesse optique |
| $f_{HI}^{(g)}$ | $f_{ICM}^{(cl)}$ | Fraction gaz/total |
| $\sigma_v$ | $\sigma_v$ ou $T_X$ | Dispersion ou température |

---

## 3) Observable : Fraction de Gaz ICM

$$f_{ICM} = \frac{M_{ICM}}{M_{ICM} + M_\star} \approx \frac{Y_{SZ}}{Y_{SZ} + \lambda \cdot \text{Richness}}$$

Pour les clusters, $f_{ICM}$ est typiquement **plus élevé** que $f_{HI}$ dans les groupes :
- Clusters : $f_{ICM} \sim 0.1 - 0.15$ (en fraction de masse totale)
- Mais en fraction baryonique : $f_{ICM}^{bar} \sim 0.8 - 0.9$

**→ Clusters sont Q-dominés** (gaz chaud domine)

---

## 4) Prédictions Quantitatives

### 4.1 Forme prédite

$$\Delta_{cl}(\rho_{cl}, f_{ICM}) \approx \alpha_{cl} \left[ C_Q(\rho_{cl}) \, \rho_{ICM} - |C_R(\rho_{cl})| \, \rho_\star \right] \cos\left( \mathcal{P}(\rho_{cl}) \right)$$

### 4.2 Amplitude attendue

Le screening est **plus fort** à l'échelle clusters :

$$|\Delta_{cl}| \sim (0.1 - 0.3) \times |\Delta_{group}| \sim (0.03 - 0.1) \times |\Delta_{gal}|$$

### 4.3 λ_QR attendu

Cohérence multi-échelle :

$$\lambda_{QR}^{cl} \in \left[\frac{1}{3}, 3\right] \times \lambda_{QR}^{gal} = [0.31, 2.82]$$

### 4.4 Signe attendu

Comme les clusters sont **Q-dominés** (gaz chaud) :
- Régime faible densité : $\cos \mathcal{P} \approx +1$ → $\Delta_{cl} > 0$
- Régime forte densité : $\cos \mathcal{P} \approx -1$ → $\Delta_{cl} < 0$

**→ Pattern similaire à ALFALFA (HI-selected)**

---

## 5) Données Disponibles

### 5.1 Résultats précédents (Planck × MCXC)

```
N = 551 clusters
a = 0.020 ± 0.015 → 1.3σ (non significatif)
Problème : Échantillon trop petit
```

### 5.2 Catalogues SZ disponibles

| Catalogue | N clusters | Année | Significativité attendue |
|-----------|------------|-------|--------------------------|
| Planck PSZ2 | 1,653 | 2015 | ~2σ |
| Planck × MCXC | 551 | 2015 | 1.3σ (actuel) |
| **ACT-DR5 MCMF** | **6,237** | 2024 | **~4.4σ** |
| ACT-DR6 | 10,040 | 2025 | ~5.7σ |

### 5.3 Choix : ACT-DR5 MCMF

- **VizieR** : J/A+A/690/A322 (Klein et al. 2024)
- **Confirmé optiquement** via MCMF
- **Disponible maintenant**

---

## 6) Critères de Falsification (Pré-Enregistrés)

| # | Critère | Seuil |
|---|---------|-------|
| 1 | Pas de flip U ↔ inverted-U sur > 1.5 dex | FALSIFIÉ |
| 2 | Décorrélation $\Delta_{cl} - f_{ICM}$ nulle (p > 0.05) | FALSIFIÉ |
| 3 | $\lambda_{QR}^{cl} \notin [0.31, 2.82]$ | FALSIFIÉ |
| 4 | Amplitude $|\Delta_{cl}| < 0.03 \times |\Delta_{gal}|$ | FALSIFIÉ |
| 5 | Permutation p > 0.05 | FALSIFIÉ |

---

## 7) Protocole de Test

### Étape 1 : Acquisition données ACT-DR5
- Télécharger depuis VizieR (J/A+A/690/A322)
- Colonnes : Y_SZ, M_500, z, RA, Dec, Richness

### Étape 2 : Calcul des variables
- $f_{ICM}$ = proxy depuis Y_SZ / (Y_SZ + Richness)
- $\rho_{cl}$ = densité locale (n clusters voisins dans R Mpc)
- $\Delta_{cl}$ = résidu M_SZ vs observable (mass-matched)

### Étape 3 : Fit TOE
- Même équation que groupes
- Paramètres libres : α, λ_QR, ρ_c, Δ
- β_Q = 1.0, β_R = 0.5 (fixés)

### Étape 4 : Validation
- Bootstrap stratifié (z, richesse)
- Permutation test
- Hold-out 20%
- Jackknife

---

## 8) Comparaison Attendue Multi-Échelle

| Échelle | λ_QR | Amplitude | Screening |
|---------|------|-----------|-----------|
| Galactique | 0.94 | 0.05 dex | Faible |
| Groupes | 1.67 | 0.03 dex | Modéré |
| **Clusters** | ~1-2 | ~0.01 dex | **Fort** |

---

**Document Status:** THÉORIE FORMALISÉE  
**Prêt pour test:** OUI  
**Critères verrouillés:** December 14, 2025
