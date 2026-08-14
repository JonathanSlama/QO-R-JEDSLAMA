# Paper 1: a non-monotonic environmental trend in BTFR residuals

**Author:** Jonathan Édouard Slama · Metafund Research Division, Strasbourg, France
**Contact:** jonathan.slama@outlook.fr · **ORCID:** [0009-0002-1292-4350](https://orcid.org/0009-0002-1292-4350)
**Version:** 4.0 (August 2026)

---

## Two documents, and the difference matters

This directory contains two texts covering the same work. They are not
interchangeable.

**`manuscript/` — the article.** The peer-reviewed version, accepted by
*Scientific Reports* (2026). This is the paper of record. It is shorter, more
cautious, and states its limitations explicitly. **Where the two documents
disagree, the article prevails.**

**`research_document/` — the research narrative.** A longer account of how the
work was actually done, in the order it happened, including the hypotheses that
were abandoned. Its second chapter, *Corrections following peer review*, records
the claims withdrawn between December 2025 and July 2026 and why. It exists
because a reader who encounters the December 2025 material, still archived under
`legacy/`, deserves to know which parts survived.

All numerical values in both documents must agree with
[`REFERENCE_VALUES.md`](REFERENCE_VALUES.md), which records each value together
with the script that produces it.

---

## Main results

| Dataset | N | Curvature coefficient a | Significance |
|---|---|---|---|
| SPARC, environmentally classified | 175 | +1.33 ± 0.25 | 5.26σ, p < 10⁻⁶ |
| SPARC, full quality-cut sample | 181 | +1.36 ± 0.24 | p < 10⁻⁶ |
| ALFALFA | 21,834 | +0.070 ± 0.028 | p = 0.0065 |
| Little THINGS | 40 | +0.29 ± 0.32 | p = 0.39, not conclusive |

The two SPARC values are both correct and differ only by the six galaxies
lacking a documented environmental class. Quoting either without stating the
sample invites confusion.

### Sign inversion in IllustrisTNG

| Population | N | a |
|---|---|---|
| Gas-rich | 444,374 | +0.017 ± 0.008 |
| Gas-poor, high stellar mass | 8,779 | −0.014 ± 0.001 |
| Extreme R-dominated | 16,924 | −0.019 ± 0.003 |

Controlling for isolation, the inversion is strongest in the densest quartile
(−0.048, 5.2σ) and absent in the most isolated (+0.003, not significant).

In a multivariate treatment with environment as the primary variable and gas
fraction and stellar mass as controls, the curvature survives but is attenuated
by roughly a factor of eight, from +0.079 to +0.0095. A substantial part of the
raw signal is carried by the covariance between environment and internal galaxy
properties.

---

## What this work claims, and what it does not

It reports a non-monotonic environmental trend and its qualitative consistency
with a two-field phenomenological description. It is a **consistency test**.

It does **not** constitute a quantitative test of that description: no forward
model links the field-theoretic parameters to the fitted curvature, and no
coupling constant is measured. The connection to moduli physics suggested in the
December 2025 documents is withdrawn; see `research_document/`, chapter 2.

Two limitations are load-bearing and stated in the article itself. The amplitude
and significance of the SPARC result are conditional on the catalogue-based
environmental classification: continuous density estimators built from 2MRS give
curvatures consistent with zero. And the environmental variables used in SPARC,
ALFALFA and IllustrisTNG are not equivalent, so the agreement across datasets is
qualitative.

---

## Layout

```
manuscript/          the accepted article and its Supplementary Information
research_document/   the research narrative v4.0, its figures and changelog
data/                SPARC with environment, TNG derived tables
scripts/             analysis scripts (see scripts/README.md for expected values)
figures/             figures of the article
tests/               the analysis chain, step by step, as originally run
REFERENCE_VALUES.md  single source of truth for every reported number
```

## Reproducing

```bash
cd scripts
python multivariate_tng_analysis.py     # Supplementary S4
python regenerate_figure2_inversion.py  # Figure 2
python regenerate_figure3_robustness.py # Figure 3
```

Expected outputs are listed in `scripts/README.md`. Any deviation means the
input data or the software environment has changed.

## Citing

Cite the article, and the Zenodo deposit for code and data. See `CITATION.cff`
at the repository root.
