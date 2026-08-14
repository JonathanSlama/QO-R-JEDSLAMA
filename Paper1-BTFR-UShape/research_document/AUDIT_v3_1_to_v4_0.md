# Audit of the narrative research document (v3.1 → v4.0)
Step 3 of PLAN_Research_Document_v4.md · carried out 14 August 2026

Every numerical value and every scientific claim in
`qor_btfr_research_narrative_v4.tex` (663 lines) was checked line by line
against `../REFERENCE_VALUES.md`.

Status: ⬜ to correct · ✅ corrected · ➖ correct as it stands

---

## A. Numerical discrepancies in the narrative document

| # | Line | What it says | What it should say | Status |
|---|---|---|---|---|
| A1 | 82 (abstract) | "Using **175** galaxies ... $a = 1.36 \pm 0.24$" | **Mispaired.** 1.36 ± 0.24 is the N = 181 full-sample fit. With N = 175 the value is 1.329 ± 0.253. Either quote 175 / 1.33 or 181 / 1.36 — never mixed. | ⬜ |
| A2 | 230 (fig. 3 caption) | "for **175** galaxies ... $a = 1.36 \pm 0.24$" | Same mispairing as A1. | ⬜ |
| A3 | 236 (equation) | $a = 1.36 \pm 0.24$, $z = 5.75$ | Value fine if N = 181 is stated. **But $z = 5.75$ does not match**: the reference significance is 5.26σ (N = 175). Recompute or state which sample. | ⬜ |
| A4 | 317 (table) | SPARC & **175** & +1.36 & 0.24 | Same mispairing. | ⬜ |
| A5 | 559 (summary) | $a = 1.36$ | Acceptable if the sample is stated; currently it is not. | ⬜ |
| A6 | 319 (table) | Little THINGS, p = **0.19** | Reference value is **p = 0.39**. | ⬜ |
| A7 | 440 (table) | Gas-poor low mass, $\sigma_a$ = **0.012** | CSV gives **0.0142**. | ⬜ |
| A8 | 441 (table) | Gas-poor high mass, $\sigma_a$ = **0.003** | CSV gives **0.00106**, i.e. ±0.001. | ⬜ |
| A9 | 442 (table) | Extreme R-dominated, $\sigma_a$ = **0.004** | CSV gives **0.00278**. | ⬜ |
| A10 | 264 | $C_Q = +2.82 \pm 0.15$ | Not reported anywhere in the article and absent from REFERENCE_VALUES. Either justify from a script, or drop. Note: an earlier framework document gave $C_Q = +2.28$ — the two disagree. | ⬜ |

Values verified as correct and needing no change: ALFALFA (+0.07 ± 0.03,
p = 0.0065, N = 21 834); TNG100 ΛCDM (a = 0.045, p = 0.075) and with correction
(a = 0.039, p = 0.004); all TNG300 central values; Eötvös and PPN limits;
N values throughout.

---

## B. Claims to rewrite

| # | Line | Claim | Action |
|---|---|---|---|
| B1 | 480–489 | §10.3 "Suggestive connection to string theory", calls `fig12_string_theory.png` | Rewrite: state what was attempted, that the first-principles computation gives 10⁻¹¹–10⁻³ rather than O(1), and that the line of inquiry is not pursued. Remove the figure and its call. Point to `legacy/Paper3-ToE` for the historical detail. **The document already says "suggestive, not derived" — the wording is honest; what is missing is the outcome.** | ⬜ |
| B2 | 489 | "A rigorous embedding of QO+R in string theory is the subject of Paper 3 in this series" | Paper 3 is withdrawn. Redirect to `legacy/` with a statement of why. | ⬜ |
| B3 | 571 | "Paper 2 of this series examines human health data for analogous signatures" | Withdrawn (exploratory, never validated). Remove or redirect to `legacy/`. | ⬜ |
| B4 | 572 | "Paper 3 explores the connection to string theory moduli" | Same as B2. | ⬜ |
| B5 | 82 (abstract) | "establishes the empirical foundation for a broader theoretical framework connecting galaxy dynamics to fundamental physics" | Overstated in light of the corrections. Reword to match the article's framing: a phenomenological consistency test. | ⬜ |

**Content absent from this document, and that is worth noting:** it contains no
26σ claim, no universal λ_QR, no conservation law, and no Theory-of-Everything
designation. Those live in the December 2025 companion documents, now in
`legacy/`. The narrative document is markedly more cautious than they are.

---

## C. Missing content to add

| # | What | Where |
|---|---|---|
| C1 | The corrections chapter (7 sub-sections) | New section after the Introduction — step 4 of the plan |
| C2 | The multivariate analysis (M0/M1/M2) and the eightfold attenuation | Section 9, after the TNG material |
| C3 | The isolation-controlled test (S3): 5.3σ / 5.2σ, effect vanishing in Q4 | Section 9 |
| C4 | The environmental-proxy sensitivity (S1) and its limitation | Section 4 or 11 |
| C5 | The model comparison (S2): spline and broken-line fit slightly better than the quadratic | Section 4 |
| C6 | An explicit statement that ρ₀ is adopted, not predicted | Section 8 |
| C7 | A pointer to the article as the version of record | Introduction |

---

## D. Discrepancies found **in the accepted article** — for the proof stage

These are not defects of the narrative document. They were found while building
the reference table, and are correctable when the proofs arrive. All follow the
same pattern: figure captions carrying values from an earlier analysis run,
while the figures themselves and the main text were updated.

| # | Where | Caption says | Source of the figure gives |
|---|---|---|---|
| D1 | Figure 3 caption | mean coefficient $a = 1.33$ | script gives **1.356** |
| D2 | Figure 3 caption | 95 % CI $[0.94, 1.75]$ | script gives **[0.9308, 1.7312]** |
| D3 | Figure 2 caption | low-mass $a = +0.052 \pm 0.012$ | CSV gives $\sigma_a$ = **0.0142** |
| D4 | Figure 2 caption | high-mass $a = -0.014 \pm 0.003$ | CSV gives $\sigma_a$ = **0.00106**; the article's own main text says **±0.001** |

D4 is the one to prioritise: the article contradicts itself between its main
text and its own figure caption, on the value that carries the discriminating
prediction.

**Action:** keep this table; submit these four corrections when Springer sends
the author proofs. No change is possible before then, and none is needed —
the science is unaffected.

---

## Summary

- 10 numerical discrepancies in the narrative document (A1–A10)
- 5 claims to rewrite (B1–B5)
- 7 additions to make (C1–C7)
- 4 corrections to request at proof stage on the article (D1–D4)

Next: step 4 — write the corrections chapter.
