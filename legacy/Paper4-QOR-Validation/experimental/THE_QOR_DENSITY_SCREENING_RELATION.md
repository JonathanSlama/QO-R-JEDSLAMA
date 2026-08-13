# THE QO+R DENSITY-SCREENING RELATION
## A Generalized Gravity Law from QO+R Empirical Validation

**Author:** Jonathan Édouard Slama  
**Institution:** Metafund Research Division, Strasbourg, France  
**Document Type:** Theoretical Formalization  
**Date:** December 2025  
**Version:** 1.1

---

## Abstract

This document formalizes the QO+R Density-Screening Relation, an empirically-derived modification to the gravitational law that emerges from the QO+R (Quotient Ontologique + Reliquat) framework. The relation describes how effective gravitational strength varies with local baryonic density through a chameleon-type screening mechanism. Validated across 14 orders of magnitude in spatial scale (10^13 to 10^27 meters), the relation provides a unified description of gravitational phenomena from stellar systems to cosmic filaments while preserving compatibility with precision tests in laboratory and Solar System environments.

---

## 1. Statement of the Relation

### 1.1 The Complete Form

The QO+R Density-Screening Relation states:

```
G_eff(rho) = G_N * [1 + Lambda_QR * alpha_0 / (1 + (rho / rho_c)^delta)]
```

Where:
- G_eff is the effective gravitational constant at density rho
- G_N = 6.674 x 10^-11 m^3 kg^-1 s^-2 is Newton's constant
- Lambda_QR = 1.23 +/- 0.35 is the universal Q-R coupling constant
- alpha_0 = 0.05 is the bare moduli amplitude (vacuum value)
- rho_c = 10^-25 g/cm^3 is the critical screening density
- delta = 1 is the screening exponent

### 1.2 Limiting Behaviors

High density limit (rho >> rho_c):
```
G_eff -> G_N * [1 + Lambda_QR * alpha_0 * (rho_c/rho)^delta]
       -> G_N    (to within 10^-27 at laboratory densities)
```

Low density limit (rho << rho_c):
```
G_eff -> G_N * [1 + Lambda_QR * alpha_0]
       = G_N * [1 + 1.23 * 0.05]
       = 1.06 * G_N
```

This represents a maximum 6 percent enhancement of gravity in the most diffuse cosmic environments.

---

## 2. Key Parameters

| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Newton constant | G_N | 6.674 x 10^-11 | m^3 kg^-1 s^-2 |
| Coupling constant | Lambda_QR | 1.23 +/- 0.35 | dimensionless |
| Bare amplitude | alpha_0 | 0.05 +/- 0.02 | dimensionless |
| Critical density | rho_c | 10^-25 | g cm^-3 |
| Screening exponent | delta | 1.0 +/- 0.3 | dimensionless |
| Transition redshift | z_t | 32 | dimensionless |

---

## 3. Validation Summary

| Scale | Distance | Predicted | Observed |
|-------|----------|-----------|----------|
| Laboratory | 10^0 m | No signal | No signal |
| Solar System | 10^11 m | No signal | No signal |
| Wide Binaries | 10^14 m | No signal | No signal |
| Globular Clusters | 10^18 m | No signal | No signal |
| UDGs | 10^20 m | Strong signal | gamma = -0.16 |
| Galaxies | 10^21 m | Signal | > 10 sigma |
| Filaments | 10^24 m | Signal | gamma = -0.033 |

---

**See full document in manuscript/ folder for complete derivation.**

**Author:** Jonathan Édouard Slama  
**Institution:** Metafund Research Division, Strasbourg, France  
**Date:** December 2025
