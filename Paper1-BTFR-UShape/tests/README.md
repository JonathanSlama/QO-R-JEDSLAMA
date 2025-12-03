# Paper1-BTFR-UShape - Structure des Tests
# ==========================================

Ce dossier contient les 14 étapes de validation du modèle QO+R,
organisées chronologiquement selon la progression scientifique réelle.

## Structure

```
tests/
├── 01_initial_qo_test/          # Test Q seul → ÉCHEC
├── 02_forensic_analysis/        # Diagnostic de l'échec
├── 03_ushape_discovery/         # Découverte forme en U
├── 04_calibration/              # Calibration C_Q, C_R
├── 05_robustness/               # Monte Carlo, Bootstrap, Jackknife
│   ├── monte_carlo_hardcore.py
│   ├── bootstrap_analysis.py
│   ├── jackknife_test.py
│   └── cross_validation.py
├── 06_replicability/            # ALFALFA, Little THINGS
├── 07_microphysics/             # Couplages différenciés
├── 08_microphysics_qhi/         # Corrélation Q-HI
├── 09_solar_system/             # Contraintes Eöt-Wash, PPN
├── 10_tng_validation/           # TNG100 confirmation
├── 11_tng_multiscale/           # TNG50/100/300, λ_QR stable
├── 12_string_theory_link/       # Q=dilaton, R=Kähler (teaser Paper3)
└── 13_predictions/              # UDGs, Lensing, z-evolution
```

## Exécution

Chaque dossier contient un script principal qui génère les figures correspondantes.
Les résultats sont sauvegardés dans `../figures/`.

```bash
cd tests/05_robustness
python monte_carlo_hardcore.py
```
