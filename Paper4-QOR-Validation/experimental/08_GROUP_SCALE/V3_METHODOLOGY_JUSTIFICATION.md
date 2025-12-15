# GRP-001 v3 : Justification des Corrections Méthodologiques

**Document Type:** Justification Pré-Implémentation  
**Author:** Jonathan Édouard Slama  
**Institution:** Metafund Research Division, Strasbourg, France  
**Date:** December 14, 2025

---

## Contexte

La version v2 du test GRP-001 a produit un résultat **intermédiaire** :
- ✅ 4/4 tests de falsification passés
- ❌ ΔAICc = -64.4 (modèle NULL préféré)
- ⚠️ Paramètres aux bornes, signal hold-out faible

Ce document justifie chaque correction proposée pour v3.

---

## Correction 1 : Mass-Matching Strict pour Δ_g

### Problème v2

```python
# v2 : Résidu global
slope, intercept = linregress(log_Mstar, log_sigma_v)
delta_g = log_sigma_v - (slope * log_Mstar + intercept)
```

Le résidu est calculé par rapport à une **relation moyenne globale**. Cela introduit :
- Biais de masse : les groupes massifs ont systématiquement plus de scatter
- Confusion avec N_gal : richesse corrèle avec masse et σ_v
- Effet de distance : complétude dépend de z

### Solution v3

**Mass-matching** : Pour chaque groupe, calculer le résidu par rapport à des groupes de masse **similaire** (± 0.1 dex).

```python
def compute_delta_g_matched(df, mass_bin_width=0.1):
    """
    Résidu σ_v avec mass-matching.
    
    Pour chaque groupe i:
    1. Sélectionner groupes j avec |log_M_j - log_M_i| < bin_width
    2. Calculer <log_σ_v> dans ce bin
    3. Δ_g,i = log_σ_v,i - <log_σ_v>_bin
    """
    delta_g = []
    for i, row in df.iterrows():
        mask = abs(df['log_Mstar'] - row['log_Mstar']) < mass_bin_width
        mean_sigma = df.loc[mask, 'log_sigma_v'].mean()
        delta_g.append(row['log_sigma_v'] - mean_sigma)
    return delta_g
```

**Pourquoi ça aide :**
- Élimine la tendance masse-σ_v (qui n'est pas le signal TOE)
- Isole la dépendance environnementale à masse fixée
- Réduit le bruit systématique

**Contrôles additionnels :**
- Matching aussi en N_gal (± 2 galaxies) et z (± 0.01)
- Vérification : <Δ_g> ≈ 0 par construction dans chaque bin

---

## Correction 2 : Régression Pondérée EIV

### Problème v2

```python
# v2 : Pondération simple inverse-variance sur Δ_g
weights = 1 / (delta_g_err**2 + 1e-6)
```

Ignore les erreurs sur les **variables indépendantes** (ρ_g, f_HI).

### Solution v3

**Errors-In-Variables (EIV)** : Propager les incertitudes sur toutes les variables.

```python
from scipy.odr import ODR, Model, RealData

def toe_model_odr(params, x):
    """Wrapper pour ODR."""
    rho_g, f_HI = x
    return toe_model((rho_g, f_HI), *params)

# Données avec erreurs
data = RealData(
    x=[rho_g, f_HI],
    y=delta_g,
    sx=[sigma_rho_g, sigma_f_HI],  # Erreurs sur X
    sy=sigma_delta_g               # Erreurs sur Y
)

model = Model(toe_model_odr)
odr = ODR(data, model, beta0=p0)
output = odr.run()
```

**Pourquoi ça aide :**
- f_HI a ~20-30% d'incertitude (cross-match ALFALFA incomplet)
- ρ_g dépend du choix de rayon (5 Mpc) → incertitude systématique
- EIV donne des erreurs sur les paramètres plus réalistes

**Estimation des erreurs :**
- σ(f_HI) ≈ 0.15 × f_HI (incertitude typique HI)
- σ(ρ_g) ≈ 0.2 dex (choix de rayon, Poisson)
- σ(Δ_g) déjà calculé depuis σ(σ_v)

---

## Correction 3 : Bootstrap Stratifié

### Problème v2

```python
# v2 : Hold-out simple 20%
holdout_mask = np.random.rand(n) < 0.20
```

Un seul split → variance d'estimation inconnue.

### Solution v3

**Bootstrap stratifié** par N_gal et z :

```python
def stratified_bootstrap(df, n_boot=1000, strata=['Ngal_bin', 'z_bin']):
    """
    Bootstrap en préservant la distribution des strates.
    """
    results = []
    
    for _ in range(n_boot):
        # Échantillonner avec remplacement dans chaque strate
        df_boot = df.groupby(strata).apply(
            lambda x: x.sample(n=len(x), replace=True)
        ).reset_index(drop=True)
        
        # Fit TOE
        popt, aicc = fit_toe(df_boot)
        results.append({'popt': popt, 'aicc': aicc})
    
    return results
```

**Pourquoi ça aide :**
- Intervalle de confiance sur λ_QR, α, etc.
- Stabilise AICc (médiane au lieu de point estimate)
- Détecte si le résultat dépend de quelques groupes influents

**Strates proposées :**
- N_gal : [3-5], [5-10], [10-20], [20+]
- z : [0.01-0.05], [0.05-0.10], [0.10-0.15]

---

## Correction 4 : Phase Lisse P(ρ) avec Spline

### Problème v2

```python
# v2 : Sigmoid simple
P = np.pi * sigmoid((rho_g - rho_c) / delta)
```

Deux problèmes :
1. Sigmoid impose une forme particulière
2. Paramètres ρ_c, Δ aux bornes → pas de convergence

### Solution v3

**Spline monotone** pour P(ρ) :

```python
from scipy.interpolate import PchipInterpolator

def phase_spline(rho_g, rho_nodes, P_nodes):
    """
    Phase P(ρ) interpolée par spline monotone.
    
    rho_nodes : points d'ancrage (ex: quantiles de ρ_g)
    P_nodes : valeurs de phase à ces points (à fitter)
    """
    # PCHIP garantit la monotonie
    spline = PchipInterpolator(rho_nodes, P_nodes)
    return spline(rho_g)
```

**Implémentation :**
1. Définir 5 nœuds aux quantiles 10%, 30%, 50%, 70%, 90% de ρ_g
2. Fitter les valeurs P_nodes (5 paramètres) avec contrainte 0 ≤ P ≤ π
3. La spline interpole entre les nœuds

**Pourquoi ça aide :**
- Plus flexible que sigmoid
- Évite les paramètres aux bornes
- Permet de visualiser où se produit le flip

**Alternative plus simple :**
Si trop de paramètres, garder sigmoid mais élargir les bornes :
- ρ_c ∈ [-4, 0] au lieu de [-3, 0]
- Δ ∈ [0.05, 5] au lieu de [0.1, 3]

---

## Correction 5 : Test de Flip Continu

### Problème v2

```python
# v2 : Test binaire low/high
delta_low = toe_model(rho_10th_percentile)
delta_high = toe_model(rho_90th_percentile)
flip = (delta_low * delta_high) < 0
```

Ne capture pas **où** le flip se produit.

### Solution v3

**Scan continu** de ρ_g pour trouver le point de flip :

```python
def find_flip_point(popt, rho_range, f_HI_median, n_points=100):
    """
    Trouve le ρ_g où Δ_g change de signe.
    """
    rho_scan = np.linspace(rho_range[0], rho_range[1], n_points)
    delta_scan = toe_model((rho_scan, np.full_like(rho_scan, f_HI_median)), *popt)
    
    # Chercher changement de signe
    sign_changes = np.where(np.diff(np.sign(delta_scan)))[0]
    
    if len(sign_changes) > 0:
        # Interpoler pour trouver ρ_flip
        idx = sign_changes[0]
        rho_flip = np.interp(0, [delta_scan[idx], delta_scan[idx+1]], 
                               [rho_scan[idx], rho_scan[idx+1]])
        return rho_flip
    return None
```

**Pourquoi ça aide :**
- Vérifie que le flip est **dans** la plage de données (pas aux bornes)
- Donne ρ_c empirique à comparer avec le fit
- Plus informatif pour la physique

---

## Correction 6 : Jackknife par Richesse et Distance

### Problème v2

Pas de test de robustesse aux sous-populations.

### Solution v3

**Jackknife** en retirant successivement des sous-groupes :

```python
def jackknife_robustness(df, popt_full):
    """
    Teste la stabilité en retirant des sous-populations.
    """
    results = []
    
    # Par richesse
    for Ngal_cut in [3, 5, 10]:
        df_sub = df[df['Ngal'] >= Ngal_cut]
        popt, aicc = fit_toe(df_sub)
        results.append({
            'cut': f'Ngal >= {Ngal_cut}',
            'n': len(df_sub),
            'lambda_QR': popt[1],
            'delta_aicc': aicc - aicc_null
        })
    
    # Par distance
    for dist_cut in [100, 200, 300]:  # Mpc
        df_sub = df[df['Dist.c'] < dist_cut]
        popt, aicc = fit_toe(df_sub)
        results.append({
            'cut': f'Dist < {dist_cut} Mpc',
            'n': len(df_sub),
            'lambda_QR': popt[1],
            'delta_aicc': aicc - aicc_null
        })
    
    return results
```

**Pourquoi ça aide :**
- Groupes pauvres (N_gal < 5) ont σ_v mal défini → bruit
- Groupes lointains ont biais Malmquist
- Si λ_QR varie beaucoup → signal fragile

**Critère de robustesse :**
- λ_QR doit rester dans [0.5, 2] × λ_QR_full pour tous les cuts
- ΔAICc doit garder le même signe

---

## Correction 7 : Permutation Test pour ΔAICc

### Problème v2

ΔAICc = -64.4, mais est-ce significatif ?

### Solution v3

**Permutation test** : Shuffle Δ_g et recalculer ΔAICc.

```python
def permutation_test_aicc(df, n_perm=1000):
    """
    Test de permutation pour ΔAICc.
    
    H0 : Pas de relation TOE (Δ_g indépendant de ρ_g, f_HI)
    """
    delta_aicc_obs = fit_toe(df)['delta_aicc']
    
    delta_aicc_perm = []
    for _ in range(n_perm):
        df_perm = df.copy()
        df_perm['delta_g'] = np.random.permutation(df['delta_g'])
        
        result = fit_toe(df_perm)
        delta_aicc_perm.append(result['delta_aicc'])
    
    # p-value : fraction où permutation fait mieux
    p_value = np.mean(np.array(delta_aicc_perm) >= delta_aicc_obs)
    
    return {
        'delta_aicc_obs': delta_aicc_obs,
        'delta_aicc_perm_mean': np.mean(delta_aicc_perm),
        'delta_aicc_perm_std': np.std(delta_aicc_perm),
        'p_value': p_value
    }
```

**Pourquoi ça aide :**
- Donne une p-value empirique pour la préférence de modèle
- Indépendant des hypothèses de distribution
- Si p < 0.05 → TOE significativement meilleur que chance

---

## Correction 8 : Proxy f_HI Robuste

### Problème v2

48.9% des groupes matchés ALFALFA → biais vers gas-rich.

### Solution v3

**Proxy empirique** quand HI manque :

```python
def estimate_f_HI_proxy(df):
    """
    Estime f_HI via couleur g-r quand ALFALFA manque.
    
    Relation empirique : galaxies bleues → gas-rich
    """
    # Couleur moyenne du groupe
    df['gr_color'] = df['gmag'] - df['rmag']  # À calculer depuis membres
    
    # Relation empirique (à calibrer sur groupes avec ALFALFA)
    # f_HI ≈ a × (g-r) + b
    df_calib = df[df['f_HI'].notna()]
    slope, intercept, _, _, _ = linregress(df_calib['gr_color'], df_calib['f_HI'])
    
    # Appliquer aux groupes sans ALFALFA
    mask_missing = df['f_HI'].isna()
    df.loc[mask_missing, 'f_HI'] = slope * df.loc[mask_missing, 'gr_color'] + intercept
    df.loc[mask_missing, 'f_HI_source'] = 'color_proxy'
    
    return df
```

**Pourquoi ça aide :**
- Augmente la couverture (100% au lieu de 49%)
- Réduit le biais de sélection gas-rich
- Permet de tester si le résultat dépend du proxy

**Validation :**
- Comparer résultats avec/sans proxy
- λ_QR doit être consistant

---

## Résumé des Corrections v3

| # | Correction | Raison | Impact attendu |
|---|------------|--------|----------------|
| 1 | Mass-matching Δ_g | Isoler signal environnemental | Réduire bruit systématique |
| 2 | EIV (erreurs sur X) | Propager incertitudes f_HI, ρ_g | Erreurs paramètres réalistes |
| 3 | Bootstrap stratifié | Intervalles de confiance | Stabiliser AICc |
| 4 | Spline P(ρ) | Éviter bornes, flexibilité | Meilleure convergence |
| 5 | Flip continu | Localiser transition | Interprétation physique |
| 6 | Jackknife | Robustesse aux cuts | Détecter fragilités |
| 7 | Permutation test | p-value empirique ΔAICc | Signification statistique |
| 8 | Proxy f_HI couleur | Couverture 100% | Réduire biais sélection |

---

## Critères de Succès v3

### ΔAICc

| Résultat | Interprétation |
|----------|----------------|
| ΔAICc > 10 | ✅ TOE fortement préférée |
| ΔAICc ∈ [2, 10] | ⚠️ TOE légèrement préférée |
| ΔAICc ∈ [-2, 2] | 🔶 Inconclusif |
| ΔAICc < -2 | ❌ NULL préféré |

### Permutation p-value

| Résultat | Interprétation |
|----------|----------------|
| p < 0.01 | ✅ Très significatif |
| p ∈ [0.01, 0.05] | ⚠️ Significatif |
| p > 0.05 | ❌ Non significatif |

### Robustesse Jackknife

| Résultat | Interprétation |
|----------|----------------|
| λ_QR stable (±50%) | ✅ Robuste |
| λ_QR varie (±100%) | ⚠️ Sensible aux cuts |
| λ_QR change de signe | ❌ Fragile |

---

## Ordre d'Implémentation

1. **Mass-matching Δ_g** (priorité haute - change la définition de base)
2. **Élargir bornes sigmoid** (quick fix avant spline)
3. **Bootstrap 1000** (intervalles de confiance)
4. **Permutation test** (p-value ΔAICc)
5. **Jackknife** (robustesse)
6. **EIV** (erreurs propagées)
7. **Proxy f_HI** (couverture)
8. **Spline P(ρ)** (si sigmoid ne suffit pas)

---

**Document Status:** JUSTIFICATION PRÉ-IMPLÉMENTATION  
**Prêt pour implémentation:** OUI  
**Date:** December 14, 2025
