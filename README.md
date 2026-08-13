# QO+R — Environmental trends in Baryonic Tully–Fisher residuals

**Author:** Jonathan Édouard Slama · Metafund Research Division, Strasbourg, France
**Contact:** jonathan.slama@outlook.fr · **ORCID:** [0009-0002-1292-4350](https://orcid.org/0009-0002-1292-4350)
**Version:** 4.0 (August 2026)

> **Concept DOI** (all versions, always resolves to latest): [10.5281/zenodo.17806441](https://doi.org/10.5281/zenodo.17806441)

## What this repository contains

Analysis code and derived data for the paper:

> **Slama, J. E.** *A non-monotonic environmental trend in Baryonic Tully–Fisher residuals: empirical evidence and a two-field phenomenological interpretation.* Scientific Reports (2026, accepted).

The paper reports a statistically significant non-monotonic (U-shaped) dependence of Baryonic Tully–Fisher Relation residuals on environment in the SPARC sample, a qualitatively similar trend in ALFALFA, and a simulation-based test in IllustrisTNG in which the environmental response inverts sign between gas-rich and gas-poor populations — including under a multivariate treatment where environment is the primary variable and gas fraction and stellar mass are statistical controls.

**What this work is:** a phenomenological consistency test. The observed pattern is of the kind a two-field antagonistic structure would produce, and is not of the kind produced by the simple monotonic single-field model tested.

**What this work is not:** a quantitative test of the two-field framework (no forward model links the field-theoretic parameters to the fitted observables), a measurement of any coupling constant, or a derivation from fundamental theory.

## Repository layout

```
Paper1-BTFR-UShape/     Published analysis: data, scripts, figures, manuscript
├── data/               SPARC + environment, TNG catalogues (derived tables)
├── scripts/            Analysis scripts, incl. multivariate_tng_analysis.py
├── figures/            Publication figures
├── manuscript/         LaTeX sources
└── tests/              Statistical robustness tests
legacy/                 Historical exploratory documents (Dec 2025) — superseded; see legacy/README.md
```

## Reproducing the main results

```bash
pip install -r requirements.txt
cd Paper1-BTFR-UShape/scripts
python multivariate_tng_analysis.py        # Supplementary S4: nested multivariate models
python regenerate_figure2_inversion.py     # Figure 2: sign inversion (TNG300)
python regenerate_figure3_robustness.py    # Figure 3: statistical robustness (SPARC)
```

Environment: Python 3, NumPy, SciPy, pandas, statsmodels, scikit-learn, Matplotlib.

## Data sources

SPARC (Lelli, McGaugh & Schombert 2016) · ALFALFA α.100 (Haynes et al. 2018) · Little THINGS (Hunter et al. 2012) · IllustrisTNG (Pillepich et al. 2018; Nelson et al. 2019) · 2MASS Redshift Survey (Huchra et al. 2012). See `DATA_SOURCES.md`.

## Citation

See `CITATION.cff`. Please cite the Scientific Reports paper and/or the Zenodo deposit.

## License

MIT — see `LICENSE`.
