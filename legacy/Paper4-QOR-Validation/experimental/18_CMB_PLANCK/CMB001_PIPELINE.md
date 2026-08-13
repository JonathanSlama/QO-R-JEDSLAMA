# CMB-001 : Validation Cosmologique QO+R avec Planck 2018

**Document Type:** Test Cosmologique  
**Author:** Jonathan Édouard Slama  
**Institution:** Metafund Research Division, Strasbourg, France  
**Date:** December 15, 2025  
**Status:** PRÉ-ENREGISTRÉ - Pipeline défini

---

## 1) Idée Clé : La Transition de Screening

### 1.1 Le Calcul Fondamental

Dans QO+R, le couplage effectif décroît avec la densité :

```
α_eff(ρ) = α₀ / (1 + (ρ/ρ_c)^δ)

avec ρ_c ~ 10⁻²⁵ g/cm³
```

### 1.2 Densité de Matière Aujourd'hui

```
ρ_m,0 = Ω_m × ρ_c,0 ≈ 2.7 × 10⁻³⁰ g/cm³

(avec h ≃ 0.67, Ω_m ≃ 0.315)
```

### 1.3 Redshift de Transition

Le seuil de désécranage survient quand ρ(a) = ρ_c :

```
ρ(a) = ρ_m,0 / a³ = ρ_c

⟹ a_t = (ρ_m,0 / ρ_c)^(1/3)
      = (2.7 × 10⁻³⁰ / 10⁻²⁵)^(1/3)
      = (2.7 × 10⁻⁵)^(1/3)
      ≈ 0.03

⟹ z_t = 1/a_t - 1 ≈ 32
```

### 1.4 Conséquence Fondamentale

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   À la recombinaison (z ~ 1100) :                                          │
│                                                                             │
│   ρ(z=1100) = ρ_m,0 × (1+1100)³ ≈ 3.6 × 10⁻²¹ g/cm³                       │
│                                                                             │
│   ρ(z=1100) / ρ_c = 3.6 × 10⁻²¹ / 10⁻²⁵ = 3.6 × 10⁴ >> 1                  │
│                                                                             │
│   ⟹ SCREENING TOTAL au CMB primaire !                                      │
│   ⟹ TT/TE/EE de Planck quasi inchangés par rapport à ΛCDM                 │
│                                                                             │
│   Les contraintes viennent surtout de :                                     │
│   • Lensing CMB (z ~ 0.5-3)                                                │
│   • ISW tardif (multipôles bas, z < 2)                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2) Paramétrisation dans un Code Boltzmann

### 2.1 Équations de Perturbations (Quasi-Statique)

On code QO+R comme une MG "douce" (quasi-statique, sans glissement exotique) :

```
k² Ψ = -4πG a² μ(a) ρΔ

k² (Φ + Ψ) = -8πG a² Σ(a) ρΔ
```

où :
- Ψ = potentiel newtonien
- Φ = potentiel de courbure
- μ(a) = modification de la croissance des structures
- Σ(a) = modification du lensing

### 2.2 Fonctions QO+R (Indépendantes de k)

```
μ(a) = 1 + 2β² f(a)

Σ(a) = 1 + β² f(a)
```

avec la fonction de transition :

```
f(a) = 1 / (1 + (a_t/a)^(3δ))

a_t ≃ 0.03,  δ ≃ 1
```

### 2.3 Comportement

| Époque | a | f(a) | μ(a) | Σ(a) |
|--------|---|------|------|------|
| Recombinaison | 9×10⁻⁴ | ~10⁻⁵ | ~1 | ~1 |
| z = 30 (transition) | 0.03 | 0.5 | 1+β² | 1+0.5β² |
| Aujourd'hui | 1 | ~1 | 1+2β² | 1+β² |

**Le CMB primaire voit μ ≈ Σ ≈ 1 (GR standard) !**

---

## 3) Pipeline Concret

### 3.1 Option A : CAMB + MGCAMB (Cobaya)

**Installation :**
```bash
pip install cobaya camb
# MGCAMB est un patch de CAMB
git clone https://github.com/sfu-cosmo/MGCAMB
```

**Configuration YAML :**
```yaml
theory:
  mgcamb:
    MG_flag: 3  # time-dependent μ-Σ
    
params:
  # Paramètres QO+R
  beta:
    prior:
      min: 0
      max: 0.5
    ref: 0.1
    proposal: 0.02
    
  a_t:
    prior:
      min: 0.01
      max: 0.06
    ref: 0.03
    proposal: 0.005
    
  delta:
    prior:
      min: 0.5
      max: 2.0
    ref: 1.0
    proposal: 0.1
    
  # Relations dérivées
  mu_0:
    value: "lambda beta: 2*beta**2"
  Sigma_0:
    value: "lambda beta: beta**2"
    
likelihood:
  planck_2018_lowl.TT:
  planck_2018_lowl.EE:
  planck_2018_highl_plik.TTTEEE:
  planck_2018_lensing.clik:
  bao.sdss_dr12_consensus_final:
```

### 3.2 Option B : CLASS + hi_class (MontePython)

**Installation :**
```bash
git clone https://github.com/miguelzuma/hi_class_public
cd hi_class_public && make
```

**Configuration :**
```ini
# hi_class.ini
gravity_model = mgcamb_like
mg_flag = 3

# QO+R specific
MG_mu = 1 + mu_0 * f(a)
MG_Sigma = 1 + Sigma_0 * f(a)
```

### 3.3 Jeux de Données

| Dataset | Contrainte principale |
|---------|----------------------|
| Planck 2018 TT,TE,EE | Spectre primaire |
| Planck 2018 lowE | Réionisation |
| Planck 2018 lensing | Σ(a) |
| BAO (BOSS, 6dF, MGS) | Casser dégénérescences |

---

## 4) Critères de Validation

### 4.1 Compatibilité avec ΛCDM

```
Δχ²_MG - Δχ²_ΛCDM ≤ 2
```

### 4.2 Postérieurs Compatibles avec 0

```
μ₀ = 2β² : posterior includes 0
Σ₀ = β²  : posterior includes 0
```

### 4.3 Contraintes Attendues

| Paramètre | Prior | Posterior attendu |
|-----------|-------|-------------------|
| β | [0, 0.5] | Compatible 0, < 0.2 |
| a_t | [0.01, 0.06] | ~0.03 ± 0.01 |
| δ | [0.5, 2] | ~1 ± 0.3 |
| μ₀ | [0, 0.5] | < 0.1 (2σ) |
| Σ₀ | [0, 0.25] | < 0.1 (2σ) |

---

## 5) Ce que ça Prouve

### 5.1 Si Compatible (Attendu)

```
✅ QO+R est compatible avec Planck 2018
✅ Le screening à z ~ 1100 protège le CMB primaire
✅ Les contraintes ISW/lensing sont satisfaites
✅ La transition à z_t ~ 30 est cohérente avec les fits locaux
```

### 5.2 Pourquoi c'est Important

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   La transition à z ~ 30 explique NATURELLEMENT pourquoi :                 │
│                                                                             │
│   • On détecte un signal QO+R aux faibles densités locales                 │
│     (UDG, filaments : ρ < ρ_c)                                             │
│                                                                             │
│   • On ne viole PAS les contraintes Planck                                 │
│     (CMB : ρ >> ρ_c → screening total)                                     │
│                                                                             │
│   C'est la MÊME physique, vue à différentes époques !                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6) Timeline et Livrables

### 6.1 Étapes

| Étape | Délai | Livrable |
|-------|-------|----------|
| Setup MGCAMB/Cobaya | 2-3 jours | Environment fonctionnel |
| Implémentation f(a) | 1-2 jours | Module QO+R |
| Run MCMC | 3-5 jours | Chains |
| Analyse postérieurs | 1-2 jours | Plots + table Δχ² |

### 6.2 Output pour Paper 4

- Table : Δχ² QO+R vs ΛCDM
- Figure : Postérieurs (β, a_t, δ) 
- Figure : μ(a), Σ(a) avec bandes d'erreur
- Phrase clé : "At recombination, ρ >> ρ_c ⟹ GR recovered; QO+R deviations only affect late ISW and lensing."

---

**Document Status:** PIPELINE DÉFINI  
**Prêt pour implémentation:** OUI
