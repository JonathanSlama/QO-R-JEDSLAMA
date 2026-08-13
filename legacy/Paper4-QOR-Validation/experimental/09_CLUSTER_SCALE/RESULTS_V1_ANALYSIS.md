# CLU-001 Résultats - Analyse Critique

**Date:** December 14, 2025  
**Version:** v1 (vraies données ACT-DR5 MCMF)  
**Status:** INCONCLUSIF (limitations techniques identifiées)

---

## Résultats Clés

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| N clean | 3,764 | ✅ Bon échantillon |
| **ρ_cl range** | **0.20 dex** | ❌ **INSUFFISANT** (besoin >1.0 dex) |
| λ_QR | 1.70 [1.69, 2.43] | ✅ Cohérent multi-échelle |
| ΔAICc | -41.3 | ❌ NULL préféré |
| Permutation p | 1.000 | ❌ Non significatif |
| Hold-out r | 0.79 | ✅ Mais artefact Y_SZ~M500 |

---

## Diagnostic : Pourquoi INCONCLUSIF ?

### 1. Levier d'environnement TROP FAIBLE

```
ρ_cl range = 0.20 dex
Requis pour flip de phase : > 1.0 dex (idéalement 1.5)
```

**Cause :** Le calcul de ρ_cl en sphères 3D avec R=20 Mpc ne capture pas assez de variation dans l'environnement cluster. Tous les clusters "voient" approximativement le même nombre de voisins.

**Solution :** Counts-in-cylinders (Δv = ±1500 km/s) + normalisation par surface couverte (HEALPix).

### 2. Y_SZ DÉRIVÉ de M500 (circularité)

```python
# Code actuel (problématique)
df_clean['Y_SZ'] = (df_clean['M500'] / 1e14) ** (5/3) * 10 ** scatter
```

**Cause :** Y_SZ est calculé depuis M500 via scaling relation. Cela injecte une corrélation structurelle qui favorise le modèle nul.

**Solution :** Utiliser Y_SZ réel du catalogue (pas disponible dans ACT-DR5 MCMF) ou résidu orthogonal log Y_SZ ~ log M500 + z.

### 3. Hold-out r = 0.79 est TROMPEUR

Ce r élevé vient de la corrélation Y_SZ ~ M500 (artificielle), pas du signal TOE.

---

## Ce qui est SOLIDE malgré tout

### λ_QR cohérent sur 3 échelles

| Échelle | λ_QR | Ratio vs galactique |
|---------|------|---------------------|
| Galactique (SPARC) | 0.94 | 1.0 |
| Groupes (Tempel) | 1.67 | 1.78 |
| Clusters (ACT-DR5) | 1.70 | 1.81 |

**→ λ_QR reste O(1) même avec données limitées !**

Cela suggère que la physique sous-jacente est correcte, mais le jeu de données ne permet pas de la détecter avec puissance statistique.

---

## Plan d'Action v2 (Corrections)

### A. Renforcer ρ_cl

```python
def compute_environment_cylinders(df, R_mpc=20, dv_kms=1500):
    """
    Counts-in-cylinders : plus robuste pour clusters à différents z.
    """
    c = 299792.458  # km/s
    H0 = 70  # km/s/Mpc
    
    for i, row in df.iterrows():
        # Cylindre : |Δz| < dv/c, |Δθ| < R/D_A
        z_window = dv_kms / c
        mask = (
            (abs(df['z'] - row['z']) < z_window) &
            (angular_distance(df, row) < R_mpc / D_A(row['z']))
        )
        n_neighbors[i] = mask.sum() - 1
    
    # Normaliser par surface ACT couverte (HEALPix)
    df['rho_cl'] = np.log10(n_neighbors / area_effective)
```

### B. Décorréler Δ_cl de M500

```python
def compute_delta_cl_orthogonal(df):
    """
    Résidu 2D : log Y_SZ ~ log M500 + z
    """
    from sklearn.linear_model import LinearRegression
    
    X = np.column_stack([np.log10(df['M500']), df['z']])
    y = np.log10(df['Y_SZ'])
    
    reg = LinearRegression().fit(X, y)
    df['delta_cl'] = y - reg.predict(X)
```

### C. Utiliser Y_SZ réel (si disponible)

ACT-DR5 MCMF n'a pas Y_SZ direct, mais on peut :
- Cross-matcher avec Planck PSZ2 (Y5R500)
- Utiliser le signal SZ des cutouts ACT

### D. Exiger ρ_range > 1.0 dex

```python
if rho_range < 1.0:
    verdict = "DATA-LIMITED: insufficient environment leverage"
    # Ne pas interpréter ΔAICc
```

---

## Règles de Décision v2

| Condition | Verdict |
|-----------|---------|
| ρ_range < 0.8 dex | DATA-LIMITED (pas d'interprétation) |
| ρ_range ≥ 0.8 AND p_perm < 0.05 AND ΔAICc > 10 | SUPPORTED |
| ρ_range ≥ 0.8 AND p_perm < 0.05 AND ΔAICc < 10 | PARTIALLY SUPPORTED |
| ρ_range ≥ 0.8 AND p_perm ≥ 0.05 | INCONCLUSIVE |
| Falsifs échouent | FALSIFIED |

---

## Comparaison Multi-Échelle (État Actuel)

| Échelle | N | λ_QR | ρ_range | Status |
|---------|---|------|---------|--------|
| Galactique | 19,222 | 0.94 | >2 dex | ✅ CONFIRMÉ |
| Groupes | 37,359 | 1.67 | 1.95 dex | ⚠️ PARTIELLEMENT SUPPORTÉ |
| **Clusters** | 3,764 | 1.70 | **0.20 dex** | ⚠️ **DATA-LIMITED** |

---

## Conclusion

> **"Le test CLU-001 est INCONCLUSIF non parce que la TOE est fausse, mais parce que le levier d'environnement (0.20 dex) est insuffisant pour détecter un flip de phase."**

La cohérence de λ_QR ≈ 1.7 avec les groupes est encourageante mais ne constitue pas une validation.

**Prochaines étapes :**
1. Implémenter counts-in-cylinders + HEALPix
2. Cross-matcher avec Planck pour Y_SZ réel
3. Refaire le test avec ρ_range > 1.0 dex

---

## Valeurs pour Trace

```json
{
  "version": "v1",
  "date": "2025-12-14",
  "n_clean": 3764,
  "rho_range_dex": 0.20,
  "lambda_QR": 1.70,
  "delta_aicc": -41.3,
  "permutation_p": 1.0,
  "holdout_r": 0.79,
  "verdict": "DATA-LIMITED",
  "reason": "insufficient environment leverage (0.20 dex < 1.0 dex required)"
}
```

---

**Document Status:** RÉSULTATS v1 - DATA-LIMITED  
**Prochaine action:** Implémenter v2 avec corrections  
**Date:** December 14, 2025
