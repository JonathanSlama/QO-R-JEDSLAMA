# Paper 1 v3.1 — Changelog
## Date : 2026-05-07
## Author : Jonathan Edouard Slama

This file documents the changes from version 3.0 (3 December 2025, DOI 10.5281/zenodo.17806442)
to version 3.1.

---

## Summary of changes

**v3.1 is a methodological clarification release. No scientific results have changed.**
All numerical values in tables and figures remain identical to v3.0.

The corrections concern:
1. The interpretive narrative of Section 9 (Simulation Validation)
2. The caption of Figure 10
3. The addition of a methodological note clarifying fit procedures

These corrections were prompted by an independent verification analysis
performed during the revision process of the companion submission to
Scientific Reports (manuscript #2025-12-33523).

---

## Detailed changes

### Change 1 — Section 9 narrative (page 16)

**v3.0 wording**:
> *"Standard physics produces a weak hint of the U-shape, but QO+R is needed to match observations."*

**v3.1 wording**:
> *"The TNG100-1 ΛCDM simulations produce a marginally significant positive
> quadratic coefficient (a = +0.045, p = 0.075), with amplitude slightly
> above the SPARC observational value (a_SPARC = +0.035). After applying
> the QO+R correction calibrated on SPARC, the TNG100-1 coefficient
> (a = +0.039, p = 0.004) aligns more closely with the SPARC value,
> demonstrating consistency between the framework and TNG simulations
> after calibration. We emphasize that this consistency check on TNG100-1
> is distinct from the discriminating test of the framework, which is
> the sign inversion in R-dominated populations (Section 9.1). The sign
> inversion is detected directly in TNG300-1 and does not depend on this
> calibration."*

**Rationale**:
Independent verification confirmed that ΛCDM alone produces a positive
quadratic coefficient (a = +0.045) slightly larger than the SPARC value
(a = +0.035). The QO+R correction, calibrated on SPARC, brings TNG100-1
closer to SPARC. The original wording "QO+R is needed to match observations"
could be interpreted as suggesting QO+R amplifies a sub-detected signal,
whereas the operational behavior is to align amplitude with SPARC after
calibration. The new wording is scientifically more precise.

The discriminating test of the framework remains the sign inversion in
R-dominated populations (Section 9.1), which is independent of this
calibration step.

### Change 2 — Figure 10 caption

The caption now reflects the corrected interpretation, explicitly noting
the SPARC reference value a = 0.035 and clarifying that the QO+R correction
"aligns" rather than "makes significant".

### Change 3 — Methodological note on fit procedures

A new subsection has been added between Figure 10 commentary and the
Killer Prediction section, transparently documenting that:
- TNG ΛCDM fit uses 15 quantile bins with σ-weighting
- TNG + QO+R fit uses 12 quantile bins without σ-weighting

Both methods are valid binned quadratic regression schemes, and we have
verified that all four combinations of binning preserve the qualitative
behavior (ΛCDM > SPARC > with-QO+R correction).

---

## What has NOT changed

The following remain identical to v3.0:

- All numerical values in tables and figures
- SPARC results (a = +1.36 ± 0.24, p < 10^-6, N=175)
- ALFALFA replication (a = +0.07 ± 0.03, p = 0.0065, N=21,834)
- Little THINGS (a = +0.29 ± 0.32, p = 0.19, N=40)
- TNG100-1 ΛCDM (a = +0.045, p = 0.075)
- TNG100-1 + QO+R (a = +0.039, p = 0.004)
- TNG300-1 killer prediction (sign inversion, 5 categories, all values unchanged)
- All theoretical sections (Lagrangian, screening, falsifiable predictions)
- All code, data, and reproducibility scripts

The discoveries reported in v3.0 are preserved in v3.1.

---

## Verification

The corrections in v3.1 are based on independent verification scripts
located in the companion Scientific Reports revision dossier:

- `verify_tng300_killer.py` — TNG300 5/5 categories reproduced (1.4-1.7% error)
- `verify_alfalfa_ushape.py` — ALFALFA reproduced to 4th decimal
- `verify_tng100_FINAL.py` — TNG100 LCDM + QO+R reproduced exactly
- `verify_little_things.py` — Little THINGS reproduced exactly
- `test_homogeneous_methods.py` — Method sensitivity test (4 fit schemes)
- `investigate_qor_formula.py` — Systematic test of 13 formula variants

All scripts and outputs are available upon request and will be incorporated
in the public GitHub repository alongside the v3.1 release.

---

## Citation

Cite v3.1 using the new DOI (to be assigned by Zenodo upon publication).
The original v3.0 DOI (10.5281/zenodo.17806442) remains available for
historical reference.

For citations of the QO+R framework as a whole, use the parent DOI
10.5281/zenodo.17806441 (resolves to latest version).

---

*Iris (assistant) + Jonathan Edouard Slama, 7 May 2026.*
