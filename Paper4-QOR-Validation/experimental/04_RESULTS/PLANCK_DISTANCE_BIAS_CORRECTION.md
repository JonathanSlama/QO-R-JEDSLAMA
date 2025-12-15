# Correction du biais de distance pour Planck PSZ2

**Date:** 7 décembre 2025  
**Problème:** Corrélation M-Y négative même avec masses MCXC indépendantes  
**Solution:** Correction du biais de distance angulaire

---

## Le problème découvert

### Observation initiale (PSZ2 seul)
```
M_SZ vs Y_SZ : r = -0.20 (attendu +0.80)
```
**Hypothèse :** MSZ dérivé de Y → circulaire

### Après cross-match MCXC
```
M_X vs Y_obs : r = -0.13 (toujours négatif !)
```
**Surprise :** Même avec des masses X-ray indépendantes, la corrélation reste négative !

---

## Diagnostic : Biais de distance angulaire

### Le signal SZ observé dépend de la distance

```
Y_obs [arcmin²] = Y_intrinsic [Mpc²] / D_A² [Mpc²/arcmin²]
```

Où D_A = distance angulaire.

### Effet combiné avec le biais de Malmquist

À haut redshift :
1. **Dilution du signal** : Y_obs ∝ 1/D_A² → signal plus faible
2. **Biais de sélection** : On ne détecte que les clusters les plus massifs

Résultat :
- Les clusters massifs sont préférentiellement à haut z
- Leurs Y_obs sont dilués par la distance
- → **Anti-corrélation artificielle M vs Y_obs**

### Preuve attendue

| Corrélation | Valeur attendue | Interprétation |
|-------------|-----------------|----------------|
| z vs M_X | Positive | Malmquist : massifs à haut z |
| z vs Y_obs | Négative | Dilution par distance |
| z vs Y_scaled | ~0 | Correction réussie |

---

## Solution : Correction de distance

### Formule de correction

```python
# Distance angulaire
D_A = cosmo.angular_diameter_distance(z)  # Mpc

# Conversion arcmin² → Mpc²
arcmin_to_rad = π / (180 × 60)
Y_true = Y_obs × (arcmin_to_rad)² × D_A²

# Correction E(z) pour relation self-similaire
E_z = H(z) / H_0 = sqrt(Ω_m(1+z)³ + Ω_Λ)
Y_scaled = Y_true / E(z)^(2/3)
```

### Relation M-Y self-similaire

La relation théorique (Kaiser 1986, Arnaud+2010) :

```
Y_500 × E(z)^(-2/3) × D_A² ∝ M_500^(5/3)
```

Donc :
```
log(Y_scaled) = (5/3) × log(M) + const
             ≈ 1.67 × log(M) + const
```

---

## Scripts créés

### Version 1 : `test_cluster_planck_mcxc.py`
- Cross-match PSZ2 × MCXC par nom
- Utilise Y_obs (non corrigé)
- Résultat : r = -0.13 (biais de distance)

### Version 2 : `test_cluster_planck_mcxc_v2.py`
- Même cross-match
- **Corrige Y pour la distance** : Y_scaled = Y_obs × D_A² / E(z)^(2/3)
- Résultat attendu : r > +0.5

---

## Structure du test v2

```
TEST 1: Y_obs vs M_X (non corrigé)
  → Devrait montrer r < 0 (confirme le biais)

TEST 2: Y_scaled vs M_X (corrigé)
  → Devrait montrer r > 0.3 (relation physique)

TEST 3: Corrélations avec z (diagnostic)
  → z vs M : positif (Malmquist)
  → z vs Y_obs : négatif (dilution)
  → z vs Y_scaled : ~0 (correction OK)

TEST 4: U-shape (si corrélation positive)
  → Résidus vs environnement
  → Q-dominé vs R-dominé
```

---

## Références

- **Kaiser (1986)** : Relation self-similaire Y-M
- **Arnaud et al. (2010)** : Calibration Y-M avec XMM-Newton
- **Planck Collaboration (2014)** : PSZ1 scaling relations
- **Piffaretti et al. (2011)** : Catalogue MCXC

---

## Leçons apprises

1. **Toujours vérifier les unités** : arcmin² vs Mpc²
2. **Le biais de distance est insidieux** : Il crée des anti-corrélations artificielles
3. **La physique guide la correction** : La relation self-similaire donne la formule exacte
4. **Tester les corrélations avec z** : Diagnostic simple mais puissant

---

**Author:** Jonathan Édouard Slama
**Date:** December 7, 2025
