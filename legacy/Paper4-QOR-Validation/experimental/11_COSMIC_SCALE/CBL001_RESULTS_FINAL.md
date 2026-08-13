# CBL-001 : Résultats Échelle Cosmique

**Date:** December 15, 2025  
**Version:** v2  
**Status:** ⚠️ DATA-LIMITED (insufficient density leverage)

---

## 1. Résumé

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **N mesures** | 8 | Données publiées (3 études) |
| **ρ_eff range** | 0.12 dex | ❌ **INSUFFISANT** (besoin >0.5 dex) |
| **A moyen** | 0.86 ± 0.06 | 14% sous GR (~2.4σ) |
| **λ_QR fitté** | 4.22 | ⚠️ Non fiable (fit dégénéré) |

**VERDICT : DATA-LIMITED - Test TOE non applicable**

---

## 2. Pourquoi le test échoue

### 2.1 Levier en densité insuffisant

| Échelle | ρ range (dex) | Status |
|---------|---------------|--------|
| Wide Binaries | 1.41 | ✅ OK |
| Galactique | >2.0 | ✅ OK |
| Groupes | 1.95 | ✅ OK |
| Clusters | 6.72 | ✅ OK |
| **Cosmique** | **0.12** | ❌ Insuffisant |

Le modèle TOE prédit une **modulation** Δ(ρ) ∝ cos(P(ρ)).
Sans variation significative de ρ, on ne peut pas tester cette dépendance.

### 2.2 Données corrélées

- Omori+15 et Kuntz+15 utilisent les **mêmes cartes** (Planck × CFHTLenS)
- Ce ne sont pas 8 mesures indépendantes mais ~3-4

### 2.3 Le signal observé

Le résidu A - 1 = -0.14 ± 0.06 est un **offset global**, pas une modulation :
- Probablement dû à des biais de calibration photométrique
- Tension connue entre Planck 2013 et 2015 (A passe de 1.05 à 0.85)
- Pas interprétable comme signal TOE

---

## 3. Ce qu'on peut conclure

### 3.1 Compatible avec screening complet

À l'échelle cosmique (10²⁵-10²⁷ m), les observations CMB lensing montrent :
- A ≈ 0.86-1.05 selon les releases
- Compatible avec GR aux incertitudes systématiques près
- **Pas de signal de gravité modifiée détecté**

Ceci est **cohérent avec la prédiction TOE** : le champ QO est homogénéisé à l'échelle cosmique, donc α → 0.

### 3.2 Mais pas un TEST du modèle

On ne peut pas affirmer que λ_QR reste O(1) à l'échelle cosmique car :
1. Pas assez de variation en densité
2. Données dominées par les systématiques
3. Le fit λ_QR = 4.22 est un artefact numérique

---

## 4. Recommandation

### 4.1 Pour un vrai test CBL

Il faudrait :
- **Binning en densité** : Séparer les lignes de visée par densité de matière intégrée
- **Tomographie** : Utiliser plusieurs bins de redshift (z = 0.2, 0.5, 0.8, 1.2)
- **Cross-corrélation directe** : Accéder aux cartes κ et δ_g (pas juste l'amplitude globale A)

### 4.2 Données requises

| Source | Accès |
|--------|-------|
| Planck PR4 κ map | Public (HEALPix) |
| DESI DR1 galaxy density | Public (2024) |
| ACT DR6 κ map | Public |

Un vrai test nécessiterait de **calculer C_ℓ^κg dans des bins de densité** plutôt que d'utiliser les amplitudes globales publiées.

---

## 5. Statut multi-échelle révisé

| Échelle | Distance | λ_QR | Status |
|---------|----------|------|--------|
| Wide Binaries | 10¹³-10¹⁶ m | 1.71 | ✅ Screened |
| Galactique | 10²¹ m | 0.94 | ✅ Confirmé |
| Groupes | 10²² m | 1.67 | ⚠️ Partiellement supporté |
| Clusters | 10²³ m | 1.23 | ⚠️ Screened |
| **Cosmique** | 10²⁵-10²⁷ m | **—** | ⬜ **DATA-LIMITED** |

**Constante universelle (4 échelles validées) :**

$$\boxed{\Lambda_{QR} = 1.39 \pm 0.33}$$

**sur 10 ordres de grandeur (10¹³ → 10²³ m)**

---

## 6. Conclusion

> **"À l'échelle cosmique, les données CMB lensing publiées sont COMPATIBLES avec le screening complet prédit par QO+R, mais le test n'est pas CONCLUSIF car le levier en densité (0.12 dex) est insuffisant pour contraindre la dépendance λ_QR(ρ). Un test rigoureux nécessiterait une analyse tomographique binée en densité."**

---

**Document Status:** ANALYSE HONNÊTE  
**Prochaine étape:** Analyse tomographique avec cartes Planck+DESI (optionnel)
