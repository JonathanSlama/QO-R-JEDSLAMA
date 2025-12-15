# GRP-001 Résultats - Analyse Critique

**Date:** December 14, 2025  
**Version:** v2 (vraies données Tempel)

---

## Résultats Clés

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| N clean | 18,558 | ✅ Excellent échantillon |
| ρ_g range | 1.83 dex | ✅ > 1.5 dex requis |
| λ_QR | 1.864 | ✅ Dans [0.33, 3.00] × λ_gal |
| Phase flip | OUI | ✅ Δ_g change de signe |
| r(Δ_g, f_HI) | 0.018 (p=0.015) | ✅ Positif et significatif |
| **ΔAICc** | **-64.4** | ❌ NULL préféré |
| Hold-out r | 0.034 (p=0.038) | ⚠️ Signal faible |

---

## Ce qui est SOLIDE ✅

1. **4/4 tests de falsification passés**
   - Flip de phase détecté
   - Corrélation Δ_g - f_HI positive
   - λ_QR dans la fenêtre théorique
   - Amplitude cohérente

2. **Pipeline robuste**
   - AICc (pas AIC)
   - Hold-out 20% blinding
   - β_Q, β_R gelés
   - Régression pondérée

3. **Données réelles**
   - 88,662 groupes Tempel
   - 48.9% cross-matchés ALFALFA
   - Pas de mock

---

## Ce qui COINCE ❌

### 1. Préférence AICc inversée

```
ΔAICc = AICc_null - AICc_TOE = -64.4
```

**Le modèle NULL est préféré de 64 points AICc !**

Cela signifie : les 4 paramètres TOE (α, λ_QR, ρ_c, Δ) n'apportent pas assez de pouvoir prédictif par rapport à un simple modèle linéaire.

**Pourquoi ?**
- Δ_g tel que défini n'extrait pas assez de signal
- Le fit virial simple (R² = 0.21) laisse beaucoup de variance non expliquée
- Le bruit domine le signal TOE

### 2. Paramètres aux bornes

```
ρ_c = -3.0 (borne inférieure)
Δ = 3.0 (borne supérieure)
```

Les paramètres de phase sont aux **bornes** → le fit n'a pas convergé proprement.

### 3. Signal hold-out faible

```
r_holdout = 0.034
```

Effet minuscule. Le modèle TOE n'explique que ~0.1% de la variance sur données non vues.

---

## Corrections Proposées (Jon)

### Priorité 1 : Redéfinir Δ_g

**Problème :** σ_v résiduel vs M_* simple n'est pas assez propre.

**Solution :** Mass-matching strict
- Résidu σ_v à M_b fixé (bins de 0.1 dex)
- Contrôle par N_gal et z
- Luminosity/group-richness fixées

### Priorité 2 : Régression robuste

**Problème :** Erreurs non propagées correctement.

**Solution :**
- EIV (Errors-In-Variables) sur σ(σ_v) et σ(f_HI)
- Bootstrap stratifié (N_gal, z)
- Stabiliser AICc

### Priorité 3 : Phase continue

**Problème :** P(ρ) avec sigmoid simple → bornes atteintes.

**Solution :**
- Spline monotone pour P(ρ)
- Test de flip continu (pas juste low/high)
- Binning adaptatif en quantiles

### Priorité 4 : Jackknife

**Problème :** Biais de complétude (Malmquist, fibre collisions).

**Solution :**
- Jackknife par richesse (N_gal ≥ 3, 5, 10)
- Jackknife par distance

### Priorité 5 : Permutation test

**Problème :** ΔAICc peut être biaisé.

**Solution :**
- Shuffle Δ_g 1000× 
- p-value empirique pour préférence modèle

### Priorité 6 : Proxy f_HI robuste

**Problème :** 48.9% matchés ALFALFA → biais vers gas-rich.

**Solution :**
- Proxy via couleur g-r ou SFR/UV
- Ne pas biaiser sur groupes riches en gaz

---

## Mini-Checklist v3

| # | Item | Status |
|---|------|--------|
| 1 | Δ_g = résidu σ_v\|M_b avec matching (± 0.1 dex) en M_*, N_gal, z | ⬜ |
| 2 | Pondération par σ(σ_v) et σ(f_HI) + EIV | ⬜ |
| 3 | P(ρ) lisse (spline) + test flip continu | ⬜ |
| 4 | Bootstrap (1000) + jackknife (N_gal ≥ 3,5,10) | ⬜ |
| 5 | Permutation test ΔAICc + blind 20% fixé a priori | ⬜ |
| 6 | Annexe "systématiques" (Tempel finder, complétude ALFALFA) | ⬜ |

---

## Conclusion Provisoire

### Ce qu'on peut dire :

> **"Signal cohérent avec la TOE (4/4 falsifs passés), mais pas encore convaincant en sélection de modèle (ΔAICc < 0)."**

### Ce qu'on NE peut PAS dire :

- ❌ "TOE confirmée à l'échelle groupes"
- ❌ "TOE falsifiée à l'échelle groupes"

### Prochaine étape :

Implémenter les corrections v3 pour voir si ΔAICc devient positif.

Si après corrections :
- **ΔAICc > 10** → TOE supportée
- **ΔAICc ∈ [-10, 10]** → Inconclusif
- **ΔAICc < -10 ET falsifs échouent** → TOE falsifiée

---

## Valeurs pour Trace

```json
{
  "n_clean": 18558,
  "rho_range_dex": 1.83,
  "lambda_QR": 1.864,
  "delta_aicc": -64.4,
  "r_holdout": 0.034,
  "p_holdout": 0.038,
  "falsif_passed": "4/4",
  "verdict": "signal cohérent, sélection modèle non convaincante"
}
```

---

**Status:** En attente de v3 avec corrections  
**Dernière mise à jour:** December 14, 2025
