# Diagnostic Planck PSZ2 - Conclusion

**Date:** 7 décembre 2025  
**Script:** `diagnose_planck_data.py`

---

## Résultat du diagnostic

### Hypothèse initiale (RÉFUTÉE)
> "MSZ est probablement dérivé de Y5R500"

**Réfutation:** Si MSZ était dérivé de Y, on attendrait :
- Corrélation forte : r > 0.9
- Pente log(M) vs log(Y) : ~ 0.6

**Observé:**
- Corrélation faible : r = -0.196
- Pente : -0.104

→ MSZ et Y5R500 sont **quasi-indépendants** dans ce catalogue !

---

## Le vrai problème

### Observation
```
log(Y) = -0.373 * log(M) + 0.712
Corrélation: r = -0.196 (très faible)
```

La relation M-Y **attendue** :
```
Y ∝ M^(5/3) → log(Y) = 1.67 * log(M) + const
Corrélation attendue: r ~ +0.8
```

### Interprétation

Il y a un **énorme scatter** dans les données. Des clusters de masse similaire ont des signaux Y très différents :

| Cluster | z | MSZ (10¹⁴ M☉) | Y5R500 (arcmin²) |
|---------|---|---------------|------------------|
| G000.77-35.69 | 0.34 | 6.33 | 1.61 |
| G002.82+39.23 | 0.15 | 5.74 | 9.70 |

Même masse, Y diffère d'un facteur **6** !

---

## Causes possibles

### 1. Biais de sélection en redshift

À différents z, les critères de détection SZ sélectionnent différents types de clusters :
- **Bas z** : On détecte tout (même faible signal Y)
- **Haut z** : On ne détecte que les clusters avec fort Y/M ratio

Cela pourrait créer une anti-corrélation artificielle.

### 2. Hétérogénéité des sources de MSZ

Le catalogue PSZ2 combine des masses de sources différentes :
- Masses SZ (dérivées de Y avec relation d'échelle)
- Masses X-ray (hydrostatiques)
- Masses optiques (richesse)

Si MSZ vient de sources mixtes, la relation Y-MSZ sera bruitée.

### 3. Scatter intrinsèque de la relation M-Y

La relation M-Y a un scatter intrinsèque de ~20-30%. Avec N=1093, on devrait quand même voir une corrélation positive... à moins que le scatter soit dominé par des systématiques.

---

## Conclusion pour le test Q²R²

### ❌ Test M-Y impossible

On ne peut PAS tester la relation M-Y (et donc les résidus M-Y vs environnement) si la relation de base n'existe pas dans les données.

### Alternatives

1. **Utiliser seulement Y vs environnement**
   - Ignorer la masse
   - Tester si Y dépend de l'environnement directement

2. **Chercher un catalogue avec masses indépendantes**
   - Cross-match PSZ2 avec MCXC (X-ray)
   - Utiliser masses weak lensing (Planck WL, DES, KiDS)

3. **Sous-échantillon homogène**
   - Sélectionner clusters avec REDSHIFT_ID cohérent
   - Ou seulement ceux avec masses X-ray (MCXC match)

---

## Données brutes

```
Total clusters: 1653
With valid z & M: 1093 (66%)

MSZ range: 0.79 - 16.12 × 10¹⁴ M☉
Y5R500 range: 0.52 - 231.66 arcmin²
z range: 0.01 - 0.97 (median 0.22)

M-Y correlation: r = -0.196
Expected: r ~ +0.8
```

---

## Recommandation

**Ne pas utiliser Planck PSZ2 seul pour le test Level 2.**

Les données ne montrent pas la relation M-Y attendue, ce qui rend impossible l'analyse des résidus.

Options :
1. Cross-match avec masses X-ray (MCXC)
2. Utiliser un proxy différent (Y seul vs environnement)
3. Attendre de meilleures données (eROSITA clusters)

---

**Author:** Jonathan Édouard Slama
**Date:** December 7, 2025
