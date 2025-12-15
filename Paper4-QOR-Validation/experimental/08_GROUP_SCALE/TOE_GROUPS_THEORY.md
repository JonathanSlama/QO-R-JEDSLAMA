# TOE Échelle Groupes - Formalisation Théorique

**Document Type:** Théorie Pré-Test  
**Author:** Jonathan Édouard Slama  
**Institution:** Metafund Research Division, Strasbourg, France  
**Date:** December 14, 2025  
**Status:** FORMALISÉ - Prêt pour test

---

## 1) Lagrangien TOE « Groupes »

On part de l'action 4D déjà posée et on la **conditionne par la densité de groupe** ρ_g (milieu intermédiaire entre isolé et amas) :

$$S_{QO+R}^{(g)} = \int d^4x \sqrt{-g} \left[ \frac{M_{Pl}^2}{2} R - \mathcal{L}_Q - \mathcal{L}_R - V(Q,R) - \mathcal{L}_{int} - \mathcal{L}_{matter} \right]$$

### Champs effectifs et environnement :

$$\mathcal{L}_Q = \frac{1}{2} g^{\mu\nu} \partial_\mu Q \, \partial_\nu Q, \quad \mathcal{L}_R = \frac{1}{2} g^{\mu\nu} \partial_\mu R \, \partial_\nu R$$

$$V(Q,R) = \frac{1}{2} m_Q^2(\rho_g) Q^2 + \frac{1}{2} m_R^2(\rho_g) R^2 + \lambda_{QR} Q^2 R^2$$

$$\mathcal{L}_{int} = C_Q(\rho_g) \, Q \, \mathcal{J}_{HI}^{(g)} + C_R(\rho_g) \, R \, \mathcal{J}_\star^{(g)}$$

### Définitions des champs :

- **Q_group** : mode *dilaton-like* couplé au **gaz neutre** du groupe
  - $\mathcal{J}_{HI}^{(g)} \propto \rho_{HI}^{(g)}$

- **R_group** : mode *Kähler-like* couplé au **contenu stellaire**
  - $\mathcal{J}_\star^{(g)} \propto \rho_\star^{(g)}$

### Dépendances environnementales (screening Yukawa/caméléon) :

$$m_Q(\rho_g) = m_{Q,0} \left( \frac{\rho_g}{\rho_0} \right)^{\gamma_Q}, \quad m_R(\rho_g) = m_{R,0} \left( \frac{\rho_g}{\rho_0} \right)^{\gamma_R}$$

$$C_Q(\rho_g) = C_{Q,0} \left( \frac{\rho_g}{\rho_0} \right)^{-\beta_Q}, \quad C_R(\rho_g) = C_{R,0} \left( \frac{\rho_g}{\rho_0} \right)^{-\beta_R}$$

### Priors physiques (groupes) — à tester mais *a priori* réalistes :

- $\lambda_{QR} \sim \mathcal{O}(1)$ (log-normal, facteur ×3 max vs galactique)
- $\beta_Q \in [0.8, 1.2]$, $\beta_R \in [0.3, 0.7]$ (le gaz se "désécrante" plus vite que les étoiles)
- $\gamma_Q, \gamma_R \in [0.2, 0.6]$
- Confinement : $m_{Q,R}(\rho_g) \, R_g \gtrsim 3$ (avec $R_g$ rayon de groupe), évitant tout rayonnement dipolaire notable

### Battement métrique lent (Ω) inchangé, forcé par Q et R :

$$\Box \Omega + \mu_\Omega^2 \Omega = \kappa_Q Q + \kappa_R R, \quad \omega \ll 10^{-9} \text{Hz}$$

---

## 2) Observables & Prédictions Quantitatives (Groupes)

### Résidu dynamique de groupe (analogue "BTFR de groupe") :

Au choix selon données :
- (i) $\Delta_{\sigma_v}$ : résidu de dispersion de vitesses à masse baryonique fixée
- (ii) $\Delta_{g_{sat}}$ : résidu d'accélération moyenne des satellites
- (iii) $\Delta_{lens}$ : biais de masse par *weak lensing* vs baryonique

### Forme prédite (interférence Q/R + phase environnementale) :

$$\boxed{\Delta_g(\rho_g, f_{HI}^{(g)}) \approx \alpha \left[ C_Q(\rho_g) \, \rho_{HI}^{(g)} - |C_R(\rho_g)| \, \rho_\star^{(g)} \right] \cos\left( \mathcal{P}(\rho_g) \right)}$$

Où :
- $f_{HI}^{(g)} = \rho_{HI}^{(g)} / (\rho_{HI}^{(g)} + \rho_\star^{(g)})$
- $\mathcal{P}(\rho_g)$ est la fonction de phase topologique liée au mécanisme type "Möbius"/écrantage

### Ansatz minimal pour la phase :

$$\mathcal{P}(\rho_g) = \pi \cdot s\left( \frac{\rho_g - \rho_c}{\Delta} \right), \quad s(x) \in [0, 1]$$

(sigmoïde centrée sur $\rho_c$ ; $\Delta/\rho_c \sim 0.5$)

### Signatures testables (numériques) :

| Prédiction | Valeur attendue |
|------------|-----------------|
| **Amplitude** (groupes) | 30%–60% de l'amplitude galactique : $|\Delta_g| \sim (0.3 \text{ à } 0.6) |\Delta_{gal}|$ |
| **Signe régime pauvre** | Densité faible + riche en HI : $\cos \mathcal{P} \approx +1 \Rightarrow \Delta_g > 0$ (branche U) |
| **Signe régime dense** | Cœur de groupe + dominance stellaire : $\cos \mathcal{P} \approx -1 \Rightarrow \Delta_g < 0$ (branche inverted-U) |
| **Scaling au gaz** | À $\rho_g$ fixé : $\partial \Delta_g / \partial f_{HI}^{(g)} > 0$ |
| **Invariance λ_QR** | Le meilleur ajustement doit rester dans ×3 du galactique |

---

## 3) Critères de Falsification (clairs, *a priori*)

**Un seul de ces points invalide l'hypothèse « Groupes » du QO+R :**

| # | Critère de falsification |
|---|--------------------------|
| 1 | Aucun flip U → inverted-U (ou inverse) en balayant $\rho_g$ sur > 1.5 dex |
| 2 | Décorrélation forte $\Delta_g - f_{HI}^{(g)}$ nulle (p > 0.05) après contrôle de masse/métallicité |
| 3 | $\lambda_{QR}^{requis} \notin [\frac{1}{3}, 3] \times \lambda_{QR}^{gal}$ |
| 4 | Amplitude $|\Delta_g| < 0.1 |\Delta_{gal}|$ ou signe opposé aux prédictions aux deux extrêmes de densité |
| 5 | Screening violé : nécessité d'un libre parcours des champs $\gtrsim R_g$ (i.e. $m_{Q,R} R_g \ll 1$) pour expliquer les données |

---

## 4) Protocole "Tempel + 2017" (Priorité 1)

### Données :
SDSS group catalog (Tempel), croisé ALFALFA (HI) + photométrie (stellaires)

### Étapes (pré-enregistrées, pas d'itérations ad hoc) :

1. **Construction des métriques de groupe** : $\rho_g$ (densité nombre/masse dans $R_{200}$), $f_{HI}^{(g)}$, $\rho_\star^{(g)}$

2. **Définition du résidu $\Delta_g$** :
   - (A) résidu $\sigma_v$ vs $M_b$ (contrôle par *matching* en $M_b$, $z$)
   - (B) variante : résidu du *stacked g* des satellites

3. **Binning** en $\rho_g$ (≥ 6 bins réguliers en log) et en tertiles de $f_{HI}^{(g)}$

4. **Fit du modèle** :

$$\Delta_g = \alpha \left[ C_{Q,0} \left(\frac{\rho_g}{\rho_0}\right)^{-\beta_Q} \rho_{HI}^{(g)} - |C_{R,0}| \left(\frac{\rho_g}{\rho_0}\right)^{-\beta_R} \rho_\star^{(g)} \right] \cos\left( \pi \, s\left(\frac{\rho_g - \rho_c}{\Delta}\right) \right)$$

   **Paramètres libres** : $\alpha$, $\lambda_{QR}$ ($\propto C_{Q,0} C_{R,0}$), $\rho_c$, $\Delta$
   
   **Priors** : $\lambda_{QR}$ log-normal ($\mu = 0$, $\sigma = \ln 3$)

5. **Tests** :
   - Monotonicité de $s(\cdot)$ (logistic vs tanh)
   - Robustesse par *jackknife* de groupes massifs
   - **Pré-enregistré** : métriques, bins, covariables ($z$, $M_b$, métallicité), seuils de rejet définis ici

### Résultats attendus :
- Courbe $\Delta_g(\rho_g)$ en U qui s'inverse à $\rho_g \simeq \rho_c$
- Séparation nette des tertiles de $f_{HI}^{(g)}$ (pente positive)

---

## 5) Feuille de Route (Groupes → Clusters)

| ID | Tâche | Description |
|----|-------|-------------|
| **G-01** | Exécuter Tempel+ALFALFA | Pipeline ci-dessus |
| **G-02** | Réanalyse | Définition alternative de $\rho_g$ (completeness/segmentation) |
| **G-03** | Sanity check | Remplacer $\cos \mathcal{P}$ par une constante → doit dégrader l'AIC/BIC de > 10 |
| **C-01** | Enchaînement Clusters | Porter la même forme sur Planck-SZ en posant $\rho_g \to \rho_c^{cl}$, $f_{HI} \to f_{ICM}$ (gaz chaud), avec amplitude attendue **plus faible** (écrantage fort) |

---

## 6) Comparaison avec Échelle Galactique

| Paramètre | Galactique | Groupes (prédit) | Ratio attendu |
|-----------|------------|------------------|---------------|
| $\lambda_{QR}$ | ~1 (4.1σ) | 0.3 – 3 | ×0.3 à ×3 |
| Amplitude $|\Delta|$ | ~0.05 dex | 0.015 – 0.03 dex | ×0.3 à ×0.6 |
| Signe flip | Oui (ALFALFA vs KiDS) | Oui (ρ_g faible vs élevé) | Consistant |
| Screening | Faible | Modéré | Attendu |

---

**Document Status:** THÉORIE FORMALISÉE  
**Prêt pour test:** OUI  
**Critères verrouillés:** December 14, 2025
