# Stratégie pour augmenter l'échantillon de clusters : ACT-DR5 MCMF

**Date:** 7 décembre 2025  
**Objectif:** Obtenir un échantillon de clusters suffisant pour détecter le U-shape

---

## Résultat actuel avec Planck × MCXC

```
N = 551 clusters
a = 0.020 ± 0.015 → 1.3σ (non significatif)
```

**Problème:** Échantillon trop petit pour une détection significative.

---

## Analyse statistique

### Erreur sur le coefficient quadratique

L'erreur sur `a` dépend de la taille de l'échantillon :
```
σ_a ∝ 1/√N
```

### Calcul de la taille nécessaire

Pour atteindre 3σ avec a = 0.020 :
```
Besoin : σ_a < 0.0067
Facteur de réduction : 0.015 / 0.0067 = 2.24
Facteur sur N : 2.24² = 5.0
N requis : 551 × 5 ≈ 2,750 clusters
```

Pour atteindre 5σ :
```
Besoin : σ_a < 0.004
N requis : 551 × (0.015/0.004)² ≈ 7,750 clusters
```

---

## Catalogues SZ disponibles

| Catalogue | N clusters | Année | Notes |
|-----------|------------|-------|-------|
| Planck PSZ2 | 1,653 | 2015 | All-sky, SNR > 4.5 |
| ACT-DR5 | 4,195 | 2021 | Southern sky |
| **ACT-DR5 MCMF** | **6,237** | 2024 | Confirmé optique/IR |
| **ACT-DR6** | **10,040** | 2025 | Plus grand catalogue SZ ! |
| SPT-SZ | ~700 | 2015 | South Pole Telescope |
| SPT-ECS | ~500 | 2020 | Extended Cluster Survey |

---

## Choix : ACT-DR5 MCMF (disponible maintenant)

### Justification

1. **Taille** : 6,237 clusters (vs 551 actuellement, ×11 !)
2. **Publication récente** : 2024 (Klein et al., A&A 690, A322)
3. **Qualité** : Confirmé optiquement via MCMF
4. **Disponible** : VizieR J/A+A/690/A322

### Significativité attendue

```
N_actuel = 551, σ_a = 0.015
N_ACT-DR5 = 6,237

Facteur d'amélioration : √(6237/551) = 3.36
Nouvelle erreur : 0.015 / 3.36 = 0.0045

Si a ≈ 0.020 (comme observé) :
Significativité = 0.020 / 0.0045 ≈ 4.4σ
```

**→ Détection significative attendue !**

---

## Pour plus tard : ACT-DR6 (10,040 clusters)

Le catalogue ACT-DR6 (publié juillet 2025) contient **10,040 clusters**.
Il sera disponible sur LAMBDA prochainement.

Significativité attendue avec ACT-DR6 : **5.7σ**

---

## Caractéristiques ACT-DR6

### Couverture
- 16,293 deg² (40% du ciel)
- Déclinaison : -60° à +20°
- Chevauchement partiel avec Planck

### Données
- Fréquences : 98 GHz, 150 GHz
- Résolution : 1.4-2.2 arcmin
- Signal SZ intégré : Y₅₀₀
- Masses : M₅₀₀ via scaling relation
- Redshifts : Spectro + photo-z

### Sous-échantillons
- 1,171 clusters à z > 1
- 123 clusters à z > 1.5
- Complétude 90% pour SNR > 5

---

## Plan d'implémentation

### Étape 1 : Télécharger ACT-DR6
```
Source : https://lambda.gsfc.nasa.gov/product/act/
Fichier : DR6 SZ Cluster Catalog
Format : FITS
```

### Étape 2 : Adapter le script
- Utiliser les colonnes ACT-DR6 (Y₅₀₀, M₅₀₀, z, RA, DEC)
- Appliquer la correction de distance (comme v2)
- Calculer l'environnement

### Étape 3 : Test U-shape
- Relation M-Y avec correction D_A²
- Résidus vs environnement
- Subdivision Q-dominé / R-dominé

---

## Comparaison des tests

| Test | N | σ_a attendu | Significativité si a=0.02 |
|------|---|-------------|---------------------------|
| Planck × MCXC | 551 | 0.015 | 1.3σ |
| + ACT-DR5 | ~3,000 | 0.006 | 3.3σ |
| **ACT-DR6 seul** | **10,040** | **0.0035** | **5.7σ** |
| ACT-DR6 + Planck | ~11,000 | 0.0033 | 6.0σ |

---

## Risques et limitations

### Risques
1. **Masses ACT-DR6** peuvent être dérivées de Y (comme PSZ2 MSZ)
   - Solution : Vérifier la documentation
   - Alternative : Cross-match avec MCXC-II (2,221 clusters X-ray)

2. **Environnement** difficile à définir pour clusters isolés
   - Solution : Utiliser densité de clusters voisins
   - Alternative : Proxy basé sur le redshift

### Limitations
- Couverture hémisphère Sud principalement
- Biais de sélection différent de Planck

---

## Référence

**ACT-DR6 Cluster Catalog:**
> "The Atacama Cosmology Telescope: DR6 Sunyaev-Zel'dovich Selected Galaxy Clusters Catalog"
> ACT-DES-HSC Collaboration (2025)
> arXiv:2507.21459

---

## Conclusion

**ACT-DR6 est le choix optimal** pour tester le U-shape dans les clusters :
- ×18 plus de clusters que notre échantillon actuel
- Significativité attendue : 5-6σ
- Données publiques et bien documentées

C'est potentiellement le **troisième contexte indépendant** pour confirmer String Theory après ALFALFA et KiDS !

---

**Author:** Jonathan Édouard Slama
**Date:** December 7, 2025
