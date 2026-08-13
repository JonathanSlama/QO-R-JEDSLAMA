#!/usr/bin/env python3
"""
regenerate_figure3_robustness_CLEAN_v2.py - Revision v2 / Referee 2 minor 5

Changements par rapport a la v1 :
  - layout 2 rangees x 3 colonnes (au lieu de 1 x 5) : demande explicite du referee.
    Ratio d'aspect 4.40 -> 1.57, donc bien moins de compression a l'insertion LaTeX.
  - polices augmentees 14/15/12 -> 17/19/16.
  - police effective sur le PDF imprime : ~3.9 pt -> ~6.3 pt.

Aucun changement scientifique : seuls figsize, rcParams et l'agencement des panneaux.
Valeurs reproduites a l'identique (MC 100%, CI [0.931, 1.731], jackknife 3/4,
RMSE 7.9%, permutation p = 0.0000).

Sortie : Review_Scientific_report_v2/11_Manuscrit_Revise_Final/figure3_robustness.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# === FONT SIZES (Referee 2, minor 5) ===
plt.rcParams.update({
    'font.size': 17,
    'axes.titlesize': 19,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
})

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPARC_CSV = PROJECT_ROOT / "data" / "sparc_with_environment.csv"
OUTPUT_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def quadratic(x, a, b, c):
    return a*x**2 + b*x + c


def linear(x, a, b):
    return a*x + b


def main():
    df = pd.read_csv(SPARC_CSV)
    a_btfr, b_btfr = -1.0577, 0.3442
    df['btfr_residual'] = df['log_Vflat'] - (a_btfr + b_btfr * df['log_Mbar'])
    print(f"[SPARC] Loaded {len(df)} galaxies")

    x = df['density_proxy'].values
    y = df['btfr_residual'].values
    n = len(df)

    np.random.seed(42)

    # ============ TEST 1: Monte Carlo ============
    print("Test 1: Monte Carlo (1000 iter)...")
    error_params = {'void': 0.07, 'field': 0.05, 'group': 0.055, 'cluster': 0.06}
    a_mc = []
    for _ in range(1000):
        y_pert = []
        for idx, row in df.iterrows():
            env = row['env_class'] if row['env_class'] in error_params else 'field'
            v_noise = np.random.normal(0, 0.03)
            dist_err = np.random.normal(0, error_params[env])
            y_pert.append(row['btfr_residual'] + v_noise - b_btfr * 2 * dist_err)
        try:
            popt, _ = curve_fit(quadratic, x, np.array(y_pert))
            a_mc.append(popt[0])
        except:
            continue
    a_mc = np.array(a_mc)
    mc_survival = np.mean(a_mc > 0) * 100
    print(f"   MC survival : {mc_survival:.1f}%")

    # ============ TEST 2: Bootstrap ============
    print("Test 2: Bootstrap (1000 iter)...")
    a_boot = []
    for _ in range(1000):
        idx = np.random.choice(n, size=n, replace=True)
        try:
            popt, _ = curve_fit(quadratic, x[idx], y[idx])
            a_boot.append(popt[0])
        except:
            continue
    a_boot = np.array(a_boot)
    boot_ushape = np.mean(a_boot > 0) * 100
    ci_low = np.percentile(a_boot, 2.5)
    ci_high = np.percentile(a_boot, 97.5)
    print(f"   Bootstrap CI : [{ci_low:.4f}, {ci_high:.4f}], {boot_ushape:.1f}% U-shape")

    # ============ TEST 3: Jackknife ============
    print("Test 3: Jackknife by environment...")
    jack = {}
    for env in df['env_class'].unique():
        df_j = df[df['env_class'] != env]
        x_j, y_j = df_j['density_proxy'].values, df_j['btfr_residual'].values
        try:
            popt, _ = curve_fit(quadratic, x_j, y_j)
            jack[env] = {'a': popt[0], 'ushape': popt[0] > 0, 'n': len(df_j)}
        except:
            jack[env] = None
    n_pass = sum(1 for r in jack.values() if r and r['ushape'])
    print(f"   Jackknife : {n_pass}/{len(jack)} stable")

    # ============ TEST 4: Cross-validation ============
    print("Test 4: Cross-validation (10-fold)...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    rmse_lin, rmse_quad = [], []
    for tr, te in kf.split(x):
        try:
            popt_l, _ = curve_fit(linear, x[tr], y[tr])
            rmse_lin.append(np.sqrt(np.mean((y[te] - linear(x[te], *popt_l))**2)))
        except: pass
        try:
            popt_q, _ = curve_fit(quadratic, x[tr], y[tr])
            rmse_quad.append(np.sqrt(np.mean((y[te] - quadratic(x[te], *popt_q))**2)))
        except: pass
    rmse_lin_m = np.mean(rmse_lin)
    rmse_quad_m = np.mean(rmse_quad)
    improvement = (rmse_lin_m - rmse_quad_m) / rmse_lin_m * 100
    print(f"   Cross-val: RMSE lin={rmse_lin_m:.4f}, quad={rmse_quad_m:.4f} ({improvement:.1f}% improvement)")

    # ============ TEST 5: Permutation ============
    print("Test 5: Permutation (1000 iter)...")
    popt_orig, _ = curve_fit(quadratic, x, y)
    a_orig = popt_orig[0]
    a_perm = []
    for _ in range(1000):
        x_sh = np.random.permutation(x)
        try:
            popt, _ = curve_fit(quadratic, x_sh, y)
            a_perm.append(popt[0])
        except: continue
    a_perm = np.array(a_perm)
    p_perm = np.mean(np.abs(a_perm) >= np.abs(a_orig))
    print(f"   Permutation : p = {p_perm:.4f}")

    # ============ FIGURE : 2 rangees x 3 colonnes (Referee 2, minor 5) ============
    fig, axes_grid = plt.subplots(2, 3, figsize=(16.5, 10.5))
    axes = axes_grid.flatten()
    axes_grid[1, 2].axis('off')   # 5 panneaux dans une grille 2x3 : on masque la cellule inutilisee

    # Panel A: Monte Carlo
    ax = axes[0]
    ax.hist(a_mc, bins=40, color='steelblue', edgecolor='black', alpha=0.75)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='$a = 0$')
    ax.axvline(np.mean(a_mc), color='green', linewidth=2, label=f'Mean = {np.mean(a_mc):.3f}')
    ax.set_xlabel('Quadratic Coefficient $a$')
    ax.set_ylabel('Frequency')
    ax.set_title(f'A) Monte Carlo\n{mc_survival:.0f}% U-shape', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel B: Bootstrap
    ax = axes[1]
    ax.hist(a_boot, bins=40, color='coral', edgecolor='black', alpha=0.75)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvspan(ci_low, ci_high, alpha=0.3, color='green',
              label=f'95% CI:\n[{ci_low:.3f}, {ci_high:.3f}]')
    ax.set_xlabel('Quadratic Coefficient $a$')
    ax.set_ylabel('Frequency')
    ax.set_title(f'B) Bootstrap\n{boot_ushape:.1f}% U-shape', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel C: Jackknife
    ax = axes[2]
    envs = list(jack.keys())
    a_vals = [jack[e]['a'] if jack[e] else 0 for e in envs]
    colors = ['green' if (jack[e] and jack[e]['ushape']) else 'red' for e in envs]
    bars = ax.bar(range(len(envs)), a_vals, color=colors, edgecolor='black', alpha=0.75)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels([f'w/o\n{e}' for e in envs], rotation=0, ha='center')
    ax.set_ylabel('Quadratic Coefficient $a$')
    ax.set_title(f'C) Jackknife\n{n_pass}/{len(envs)} pass', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel D: Cross-validation
    ax = axes[3]
    models = ['Linear', 'Quadratic']
    rmse_vals = [rmse_lin_m, rmse_quad_m]
    colors_d = ['#3498db', '#e74c3c']
    bars = ax.bar(models, rmse_vals, color=colors_d, edgecolor='black', alpha=0.85)
    ax.set_ylabel('RMSE')
    ax.set_title(f'D) Cross-validation\n{improvement:.1f}% improvement', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel E: Permutation
    ax = axes[4]
    ax.hist(a_perm, bins=40, color='gray', edgecolor='black', alpha=0.75, label='Permuted')
    ax.axvline(a_orig, color='red', linewidth=3, label=f'Original\n$a={a_orig:.3f}$')
    ax.set_xlabel('Quadratic Coefficient $a$')
    ax.set_ylabel('Frequency')
    ax.set_title(f'E) Permutation\np = {p_perm:.4f}', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "figure3_robustness.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n[OK] Saved: {output_path}")
    print("     layout : 2 rangees x 3 colonnes (5 panneaux occupes)")
    print("     fonts  : axes=18, titles=19, ticks=16")


if __name__ == "__main__":
    main()

