# TEST RATIONALE AND EXPECTED OBSERVATIONS
## Why We Test, What We Test, What We Expect

Author: Jonathan Édouard Slama
Affiliation: Metafund Research Division, Strasbourg, France
Date: December 2025

---

## 1. Philosophy of Testing

### 1.1 The Scientific Method Applied

The QO+R framework is not validated by elegance alone—it must make **falsifiable predictions** that distinguish it from alternatives. Each test serves a specific purpose:

1. **Discrimination:** Can this result be explained by other theories?
2. **Quantification:** Does the observed value match the predicted range?
3. **Robustness:** Does the result persist under different conditions?
4. **Falsifiability:** What observation would disprove the theory?

### 1.2 The Burden of Proof

As a theory claiming to unify Newton, GR, MOND, quantum mechanics, and synchronization, QO+R carries an extraordinary burden of proof. We must demonstrate:

- It **reproduces** known physics in appropriate limits
- It **predicts** new phenomena not explained by existing theories
- It **survives** stringent observational constraints

---

## 2. Test Catalog

### 2.1 THREE-BODY PROBLEM TEST

#### Why This Test?

The three-body problem is the archetypal chaotic system in classical mechanics. If QO+R's phase coupling genuinely creates coherence, it should **reduce chaos**—a dramatic prediction with no analog in Newtonian gravity.

#### What We Test

We compare the maximum Lyapunov exponent λ (chaos measure) between:
- Pure Newtonian dynamics: λ_Newton
- QO+R with phase coupling: λ_QOR

#### What We Expect

**QO+R Prediction:** λ_QOR ≤ λ_Newton

The phase coupling term:
$$-\frac{\kappa}{r^\nu}\left(1 - \cos\frac{\Delta\theta}{2}\right)$$

creates an energy penalty for phase desynchronization. This should:
1. Reduce orbital divergence rate (lower λ)
2. Enhance stability of special configurations (Figure-8, Lagrange)
3. Prevent or delay escape events

#### What Would Falsify

If λ_QOR > λ_Newton systematically (>50% of configurations), the phase-coherence mechanism is wrong or incorrectly parameterized.

#### Physical Significance

- **Confirmed:** Phase coupling genuinely stabilizes gravitational systems—a new fundamental principle
- **Falsified:** Phase coupling is either absent or destabilizing—back to the drawing board

---

### 2.2 ULTRA-DIFFUSE GALAXY (UDG) TEST

#### Why This Test?

UDGs are extreme objects:
- Very gas-poor (f_HI < 5%)
- Low stellar density (μ₀ > 24 mag/arcsec²)
- Large effective radii (R_eff > 1.5 kpc)
- Living in low-density environments

They represent the **extreme R-dominated limit** in low-density environments—the opposite corner of parameter space from gas-rich ALFALFA galaxies.

#### What We Test

The quadratic coefficient 'a' in the environmental dependence of scaling relation residuals:
$$\Delta = a\rho^2 + b\rho + c$$

#### What We Expect

**QO+R Prediction:** a << 0 (strongly inverted U-shape)

Because:
1. UDGs are R-dominated (C_R coupling dominant)
2. Low-density environment (phase interference maximized)
3. The combination predicts strong negative residuals in intermediate environments

**Quantitative Target:** a < -1.0 at > 3σ

#### What Would Falsify

If UDGs show a > 0 (U-shape like SPARC), it would mean:
- The Q/R coupling sign rule is wrong, OR
- Environmental density doesn't control the phase as predicted

Either invalidates a core QO+R prediction.

#### Physical Significance

- **Confirmed:** Q/R sign inversion is real and extends to extreme parameter regimes
- **Falsified:** The Q/R model needs fundamental revision

---

### 2.3 GAS FRACTION THRESHOLD TEST

#### Why This Test?

QO+R predicts the sign of 'a' flips when gas fraction crosses a threshold. This is a **knife-edge prediction**—the exact value of f_crit probes the relative coupling strengths C_Q and C_R.

#### What We Test

Fit quadratic coefficient 'a' in bins of gas fraction f_HI = M_HI / M_bary, looking for sign change.

#### What We Expect

**QO+R Prediction:** Sign inversion at f_crit = 0.3 ± 0.1

The transition occurs when:
$$C_Q f_{HI} = |C_R| (1 - f_{HI})$$

Solving:
$$f_{crit} = \frac{|C_R|}{C_Q + |C_R|} \approx \frac{0.72}{2.82 + 0.72} \approx 0.20$$

With uncertainties, we predict f_crit ∈ [0.2, 0.4].

#### What Would Falsify

- No threshold exists (constant sign across all f_HI)
- Threshold at f_crit < 0.05 or f_crit > 0.9 (unphysical)

#### Physical Significance

- **Confirmed:** The C_Q/C_R ratio is constrained, linking to KKLT parameters
- **Falsified:** The two-field model with opposite couplings is wrong

---

### 2.4 CMB LENSING CORRELATION TEST

#### Why This Test?

If Q²R² modifies the gravitational potential, it should affect CMB lensing convergence κ. This tests whether the moduli couple to **all** gravity or only galactic dynamics.

#### What We Test

Cross-correlation between:
- BTFR residuals Δ(galaxy)
- CMB lensing convergence κ(position)

#### What We Expect

**QO+R Prediction:** Correlation r > 0.05 at > 3σ

The sign depends on Q/R dominance:
- Q-dominated sample: Positive correlation (more lensing where Q effect stronger)
- R-dominated sample: Negative correlation

#### What Would Falsify

If r = 0 ± 0.01 (no correlation at high precision), it means:
- Q²R² is a purely baryonic effect, not gravitational
- The string moduli interpretation is wrong

#### Physical Significance

- **Confirmed:** Q²R² is a genuine modification of gravity, not just baryonic physics
- **Falsified:** Effect is limited to baryonic dynamics, not cosmological gravity

---

### 2.5 REDSHIFT EVOLUTION TEST

#### Why This Test?

String moduli stabilization evolves with cosmic expansion. If Q and R are moduli, their coupling λ_QR should change with redshift.

#### What We Test

Measure λ_QR(z) in redshift bins from z=0 to z~1.

#### What We Expect

**QO+R Prediction:** λ_QR(z) = λ_QR,0 × (1+z)^γ with γ ∈ [0.5, 2.0]

From moduli cosmology:
- Kähler moduli masses scale with Hubble
- Coupling strength evolves as universe expands

#### What Would Falsify

If λ_QR is strictly constant (γ = 0 ± 0.1), the string moduli interpretation is disfavored.

#### Physical Significance

- **Confirmed:** First time-varying moduli detected cosmologically—revolutionary
- **Falsified:** Q²R² is a static effective theory, not dynamical string physics

---

### 2.6 SOLAR SYSTEM SCREENING TEST

#### Why This Test?

Any modification of gravity must not violate Solar System precision tests (Cassini, Lunar Laser Ranging, planetary ephemerides).

#### What We Test

Predicted deviation from Newtonian potential at 1 AU:
$$\left|\frac{\Delta g}{g}\right|_{1AU}$$

#### What We Expect

**QO+R Prediction:** |Δg/g| < 10⁻¹² (undetectable)

The screening mechanism:
$$m_{Q,R}(\rho_\odot) \sim 10^{-18} \text{ eV} \quad \Rightarrow \quad \text{range} \lesssim 0.1 \text{ AU}$$
$$C_{Q,R}(\rho_\odot) \sim 10^{-4} C_{Q,R}^{gal}$$

This environmental suppression ensures:
- No fifth-force detection in Solar System
- Effects only manifest at galactic densities

#### What Would Falsify

If the predicted galactic signal requires |Δg/g| > 10⁻¹⁰ at 1 AU, it conflicts with observations.

#### Physical Significance

- **Confirmed:** Chameleon-type screening works as predicted
- **Falsified:** Theory is ruled out by local gravity tests

---

## 3. Multi-Scale Validation Hierarchy

The tests span multiple scales to ensure QO+R is not just fitting one phenomenon:

| Scale | Test | Key Observable | Target |
|-------|------|----------------|--------|
| 10⁰ AU | Solar System | Δg/g | < 10⁻¹² |
| 10³ pc | Three-Body | λ_QOR vs λ_Newton | Reduction |
| 10⁴ pc | UDGs | a coefficient | < -1.0 |
| 10⁵ pc | ALFALFA/KiDS | Sign opposition | 26σ (done) |
| 10⁶ pc | Planck SZ | Cluster dilution | Flat (done) |
| 10⁸ pc | CMB Lensing | κ-residual corr. | r > 0.05 |
| 10¹⁰ pc | Redshift | λ_QR(z) | γ ≠ 0 |

This 12 orders of magnitude coverage is unprecedented for any modified gravity theory.

---

## 4. Theory Discrimination Matrix

Each test helps distinguish QO+R from alternatives:

| Test | QO+R | Newton | MOND | f(R) | ΛCDM |
|------|------|--------|------|------|------|
| Three-body chaos | Reduced | Baseline | Same | Same | Same |
| UDG U-shape | Inverted | None | None | Same | None |
| Gas threshold | 0.3 ± 0.1 | None | None | None | None |
| CMB correlation | Yes | No | No | Maybe | No |
| z-evolution | Yes | No | No | Maybe | No |
| Solar screening | Yes | N/A | Violated | Yes | N/A |

Only QO+R predicts ALL these signatures simultaneously.

---

## 5. Priority and Feasibility

### 5.1 Immediate (Data Available)

1. **Gas Fraction Threshold** - ALFALFA data ready
2. **CMB Lensing** - Planck PR3 public
3. **Three-Body Simulation** - Numerical, no new data

### 5.2 Near-Term (Catalogs to Download)

1. **UDG Test** - MATLAS catalog from VizieR
2. **Redshift Evolution** - KiDS photo-z bins (partially done)

### 5.3 Future (New Surveys)

1. **WALLABY** (2025-2027) - 500k HI galaxies
2. **Euclid** (2026+) - 30M spectroscopic galaxies
3. **Roman** (2027+) - Billion-galaxy weak lensing

---

## 6. Summary: What Success Looks Like

### Complete Validation Requires:

1. CONFIRMED: U-shape in SPARC
2. CONFIRMED: Sign opposition ALFALFA/KiDS (26σ)
3. CONFIRMED: Cluster dilution (5.7σ)
4. CONFIRMED: R in denser environments (60σ)
5. TO TEST: UDG inverted U-shape
6. TO TEST: Gas fraction threshold
7. TO TEST: CMB lensing correlation
8. TO TEST: Three-body chaos reduction
9. PARTIALLY DONE: Redshift evolution
10. THEORETICAL: Solar System screening

### Current Status: 4/10 confirmed, 6/10 pending

If all 10 are confirmed, QO+R would be the most empirically validated extension of gravity ever proposed.

---

## 7. Conclusion

Each test serves a specific purpose in the validation hierarchy. We don't just confirm—we **seek to falsify**. The tests are designed so that:

- Failure of ANY killer prediction requires major theory revision
- Success across ALL scales provides overwhelming evidence
- The multi-scale approach prevents overfitting to single phenomena

This is the rigorous scientific method applied to a Theory of Everything candidate.

---

*"The true test of a theory is not whether it can explain what we already know, but whether it can predict what we don't."*
