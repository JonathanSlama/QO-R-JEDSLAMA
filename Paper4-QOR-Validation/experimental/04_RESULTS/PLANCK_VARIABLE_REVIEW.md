# Planck PSZ2 Catalog - Variable Review

**Date:** 7 décembre 2025  
**Objectif:** Définir correctement les variables pour le test Q²R² sur amas de galaxies

---

## Catalogue disponible

### HFI_PCCS_SZ-union_R2.08.fits.gz (PSZ2 Union)

**30 colonnes, 1653 amas**

| Colonne | Description | Unité | Usage |
|---------|-------------|-------|-------|
| `INDEX` | Index interne | - | - |
| `NAME` | Nom PSZ2 | - | Identification |
| `GLON` | Longitude galactique | deg | Position |
| `GLAT` | Latitude galactique | deg | Position |
| `RA` | Ascension droite | deg | **Position** |
| `DEC` | Déclinaison | deg | **Position** |
| `POS_ERR` | Erreur de position | arcmin | Qualité |
| `SNR` | Signal/Bruit | - | **Qualité** |
| `PIPELINE` | Pipeline de détection | - | Méta |
| `PIPE_DET` | Flag de détection | - | Méta |
| `PCCS2` | Cross-match PCCS2 | - | Méta |
| `PSZ` | Cross-match PSZ1 | - | Méta |
| `IR_FLAG` | Flag contamination IR | - | Qualité |
| `Q_NEURAL` | Score qualité neural | - | Qualité |
| `Y5R500` | Signal SZ (Y × θ²) | arcmin² | **Observable SZ** |
| `Y5R500_ERR` | Erreur sur Y5R500 | arcmin² | Erreur |
| `VALIDATION` | Status validation | - | Qualité |
| `REDSHIFT_ID` | Source du redshift | - | Méta |
| `REDSHIFT` | Redshift spectro/photo | - | **Distance** |
| `MSZ` | Masse M₅₀₀ | 10¹⁴ M☉ | **MASSE** ⭐ |
| `MSZ_ERR_UP` | Erreur haute sur MSZ | 10¹⁴ M☉ | Erreur |
| `MSZ_ERR_LOW` | Erreur basse sur MSZ | 10¹⁴ M☉ | Erreur |
| `MCXC` | Cross-match MCXC | - | X-ray |
| `REDMAPPER` | Cross-match RedMaPPer | - | Optique |
| `ACT` | Cross-match ACT | - | SZ |
| `SPT` | Cross-match SPT | - | SZ |
| `WISE_FLAG` | Flag WISE | - | IR |
| `AMI_EVIDENCE` | Evidence AMI | - | Radio |
| `COSMO` | Flag cosmologie | - | Qualité |
| `COMMENT` | Commentaires | - | - |

---

## Variables pour le test Q²R²

### Variable Q (proxy gaz)

**Choix:** Fraction de gaz estimée depuis Y_SZ et M_500

```python
# Relation self-similar: Y_SZ ∝ f_gas × M^(5/3) × E(z)^(2/3)
# Donc: f_gas ∝ Y_SZ / (M^(5/3) × E(z)^(2/3))

E_z = cosmo.efunc(z)  # H(z)/H0
ratio = Y5R500 / (MSZ**(5/3) * E_z**(2/3))
f_gas = 0.12 * (ratio / median(ratio))  # Normalisé à ~12%
```

**Justification physique:** Le signal SZ est proportionnel à l'intégrale de la pression électronique, donc à M_gas × T. Pour une masse donnée, un signal Y plus élevé indique plus de gaz chaud.

### Variable R (proxy matière)

**Choix:** Masse totale M₅₀₀

```python
M_500 = data['MSZ']  # en 10^14 M_sun
log_M = np.log10(M_500)
```

**Note:** MSZ est dérivée de Y_500 via la relation Y-M calibrée. Il y a donc une corrélation intrinsèque Y-M. Le test cherche les **résidus** de cette relation.

### Relation d'échelle : M-Y

**Définition:**
```python
# Relation self-similar attendue:
# Y_500 ∝ M_500^(5/3) × E(z)^(2/3)
# log(Y) = (5/3) × log(M) + const

log_Y = np.log10(Y5R500)
log_M = np.log10(MSZ)

slope, intercept, r, p, err = linregress(log_M, log_Y)
# Attendu: slope ~ 1.67
```

**Résidus:**
```python
log_Y_predicted = slope * log_M + intercept
Y_residual = log_Y - log_Y_predicted
```

Un résidu positif = amas avec **excès de signal SZ** pour sa masse (plus de gaz chaud)
Un résidu négatif = amas avec **déficit de signal SZ** pour sa masse (moins de gaz)

### Variable environnement

**Méthode:** Comptage de voisins dans un rayon physique

```python
# Pour chaque amas i:
d_A = angular_diameter_distance(z_i)
max_sep_deg = arctan(50 Mpc / d_A)

# Compter voisins avec |z - z_i| < 0.05 et sep < max_sep_deg
neighbors = count(...)

# Normaliser
env = (neighbors - mean) / std
```

---

## Prédiction Q²R² pour Planck

### Si String Theory est correct:

1. **U-shape dans les résidus Y-M vs environnement**
   - Amas dans environnements extrêmes : résidus Y extrêmes
   - Amas dans environnement moyen : résidus Y proches de zéro

2. **Inversion de signe entre Q-dom et R-dom**
   - Amas **riches en gaz** (Q-dom): a < 0 (U inversé) ou a > 0 ?
   - Amas **massifs, pauvres en gaz** (R-dom): signe opposé

### Différences avec les galaxies

| Aspect | Galaxies (rotation) | Amas (SZ) |
|--------|---------------------|-----------|
| Échelle | kpc | Mpc |
| Observable | v_rot | Y_SZ |
| Q proxy | f_gas (HI 21cm) | f_gas (Y/M ratio) |
| R proxy | M★ (photométrie) | M_500 (SZ-derived) |
| Environnement | Densité locale | Voisins proches |
| N typique | ~20,000 | ~500-1000 |

---

## Cuts de qualité recommandés

```python
mask = (
    np.isfinite(RA) & np.isfinite(DEC) &
    np.isfinite(REDSHIFT) & (REDSHIFT > 0.01) & (REDSHIFT < 1.0) &
    np.isfinite(Y5R500) & (Y5R500 > 0) &
    np.isfinite(MSZ) & (MSZ > 0) &
    (SNR > 4.5)  # Détection fiable
)
```

**Note:** Beaucoup d'amas n'ont pas de redshift spectroscopique. Vérifier `REDSHIFT_ID` pour la source.

---

## Résultat attendu du test v2

| Métrique | Valeur attendue | Signe d'erreur |
|----------|-----------------|----------------|
| Pente M-Y | ~ 1.67 | < 0 ou > 3 → erreur |
| Corrélation r | ~ +0.8 | Négative → erreur |
| N après cuts | 400-800 | < 100 → problème |
| f_gas range | 0.08 - 0.18 | Constant → erreur |

---

## Limitations connues

1. **Corrélation intrinsèque Y-M:** MSZ est dérivée de Y, donc ils sont corrélés par construction. Les résidus mesurent les écarts à la relation moyenne, pas une indépendance.

2. **Échantillon limité:** ~1000 amas après cuts, vs ~20,000 galaxies. Moins de puissance statistique.

3. **Environnement mal défini:** Les amas sont eux-mêmes des pics de densité. L'environnement "autour" d'un amas est plus diffus.

4. **Biais de sélection SZ:** Favorise les amas massifs et à haut z (signal Y fort).

---

**Author:** Jonathan Édouard Slama
**Date:** December 7, 2025
