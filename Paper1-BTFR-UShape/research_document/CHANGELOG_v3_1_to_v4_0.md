# Changelog: narrative research document v3.1 to v4.0
August 2026

Version 4.0 brings the narrative research document into agreement with the
peer-reviewed article (Slama 2026, Scientific Reports) and adds a chapter
recording the claims withdrawn between December 2025 and July 2026.

Versions 3.0 and 3.1 remain frozen and unmodified under `legacy/`. Nothing has
been deleted; the history is meant to be readable.

---

## 1. New chapter: Corrections following peer review

Placed immediately after the introduction, so that a reader reaches it before
any result. Eight subsections, each stating what was claimed, what checking it
revealed, and what replaced it:

| Subsection | Outcome |
|---|---|
| The derivation from string theory | Withdrawn. Computed from the stabilisation potential the coupling is 10⁻¹¹ to 10⁻³, not O(1). The earlier agreement was presupposed. |
| The universal coupling constant | Withdrawn. Joint fit rejects universality (χ²/dof = 4.5, p = 0.001). Two of the author's own documents disagreed by a factor of 25. |
| The conservation law | Withdrawn. The required SO(2) symmetry is absent from the Lagrangian. |
| The significance of the sign inversion | 26σ replaced by 5.3σ / 5.2σ under isolation control. |
| What controlling for internal properties revealed | New. Curvature attenuated by a factor of eight; declared explicitly. |
| The Theory of Everything designation | Withdrawn. No General Relativity limit. |
| The extension to clinical data | Withdrawn. Exploratory, never validated. |
| What this leaves | The surviving empirical content, and how the errors arose. |

The chapter closes by naming the founding intuition, that a ratio yields both a
comparison and a remainder and that the remainder carries something like an
identity, and by explaining why naming it strengthens rather than weakens the
corrections.

## 2. Numerical corrections

| Where | Was | Now |
|---|---|---|
| Abstract, Figure 3 caption, regression equation, dataset table, summary | "175 galaxies ... a = 1.36 ± 0.24" | a = 1.33 ± 0.25 at 5.26σ for N = 175; the N = 181 value of 1.36 stated separately where used |
| Regression equation | z = 5.75 | σ = 5.26 (the previous value matched neither sample) |
| Little THINGS | p = 0.19 | p = 0.39 |
| TNG300, gas-poor low mass | σ_a = 0.012 | σ_a = 0.014 |
| TNG300, gas-poor high mass | σ_a = 0.003 | σ_a = 0.001 |
| TNG300, extreme R-dominated | σ_a = 0.004 | σ_a = 0.003 |

All values now agree with `../REFERENCE_VALUES.md`.

## 3. Rewritten sections

**Connection to fundamental physics.** Previously stopped at the resemblance
between the two-field structure and string compactifications, describing it as
suggestive and deferring to a companion Paper 3. Now states the outcome: the
coupling computed from the moduli stabilisation potential is eight to fifteen
orders of magnitude below the value the analogy was taken to explain. Figure 12
removed.

**Open questions.** The two questions concerning companion Papers 2 and 3 are
closed rather than left open, and replaced by the two that remain genuinely
open: robustness to the definition of environment, and the missing forward
model.

## 4. New sections

- **Is the quadratic form required by the data?** Model comparison across five
  functional forms. Non-monotonic dependence strongly preferred; the quadratic
  itself is slightly outperformed by spline and broken-line fits, and is
  retained for interpretability rather than fit quality.
- **How much does the result depend on how environment is defined?** The
  sensitivity analysis across 2MRS fixed-radius proxies, with the conclusion
  stated plainly: the amplitude and significance are conditional on the
  environmental definition adopted.
- **Controlling for isolation.** The d5NN-stratified test: inversion strongest
  in the densest quartile, absent in the most isolated.
- **Environment as the primary variable.** The nested multivariate regression on
  TNG100-1, with the eightfold attenuation and the continuous sign inversion.

## 5. Statements of status added

- The calibrated couplings C_Q and C_R are marked as sample-specific calibration
  outputs, absent from the article, and not measurements of physical constants.
- The screening density is marked as a free parameter fixed after the fact, and
  the solar system agreement as a requirement imposed rather than a test passed.
- A note in the introduction stating that this document is a research narrative,
  that the article is the paper of record, and that the article prevails where
  the two disagree.

## 6. Editorial

- Author email updated to jonathan.slama@outlook.fr
- Date and version updated to 4.0, August 2026
- Figure path repointed to the document's own `figures/` directory
- All em dashes replaced by hyphens
- Compiles without error or undefined reference; 35 pages

---

## Not done in this version

- Figures 3 and 4 still display values from the earlier analysis in their
  rendered panels. The captions are correct; the plotted annotations have not
  been regenerated.
- Sections on microphysical coupling have not been re-derived, only checked for
  withdrawn claims.
