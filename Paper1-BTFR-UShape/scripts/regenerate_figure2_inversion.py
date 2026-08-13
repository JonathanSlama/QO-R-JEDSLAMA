#!/usr/bin/env python3
"""
regenerate_figure2_inversion_CLEAN_v2.py - Revision v2 / Referee 2 minor 4

Changements par rapport a la v1 :
  - figsize 18 x 6.5 -> 15 x 6.0 (moins de compression a l'insertion LaTeX)
  - polices 14/15/12 -> 19/21/18
  - police effective sur le PDF imprime : ~4.7 pt -> ~7.7 pt

Aucun changement scientifique : seuls figsize et rcParams.

Sortie : Review_Scientific_report_v2/11_Manuscrit_Revise_Final/figure2_inversion.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# === FONT SIZES (Referee 2, minor 4) ===
plt.rcParams.update({
    'font.size': 19,
    'axes.titlesize': 21,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 22,
})

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TNG_RESULTS = PROJECT_ROOT / "data" / "tng300_stratified_results.csv"
OUTPUT_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(TNG_RESULTS)
    print(f"[TNG300] Loaded stratified results: {df['n'].sum():,} total galaxies")

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.0))

    # --- Panel A : Main gas categories ---
    ax = axes[0]
    main_cats = df[df['category'].isin(['Gas: Gas-rich', 'Gas: Intermediate', 'Gas: Gas-poor'])]
    names = main_cats['category'].str.replace('Gas: ', '').values
    a_vals = main_cats['a'].values
    a_errs = main_cats['a_err'].values
    n_vals = main_cats['n'].values

    colors = ['#3498db', '#f39c12', '#e74c3c']
    x_pos = range(len(names))
    bars = ax.bar(x_pos, a_vals, yerr=a_errs, color=colors,
                  edgecolor='black', capsize=8, alpha=0.85, linewidth=1.2)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{n}\n(N={nv:,})" for n, nv in zip(names, n_vals)])
    ax.set_ylabel('U-shape Coefficient $a$')
    ax.set_title('A) U-shape by Gas Fraction (TNG300)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel B : Gas-poor by mass ---
    ax = axes[1]
    mass_cats = df[df['category'].str.contains(r'Gas-poor \+')]
    names_b = mass_cats['category'].str.replace('Gas-poor + ', '').values
    a_vals_b = mass_cats['a'].values
    a_errs_b = mass_cats['a_err'].values
    n_vals_b = mass_cats['n'].values

    colors_b = ['#2ecc71', '#f39c12', '#9b59b6']
    x_pos = range(len(names_b))
    bars_b = ax.bar(x_pos, a_vals_b, yerr=a_errs_b, color=colors_b,
                    edgecolor='black', capsize=8, alpha=0.85, linewidth=1.2)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{n}\n(N={nv:,})" for n, nv in zip(names_b, n_vals_b)])
    ax.set_ylabel('U-shape Coefficient $a$')
    ax.set_title('B) Gas-Poor by Stellar Mass', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for i, (a, bar) in enumerate(zip(a_vals_b, bars_b)):
        if a < 0:
            bar.set_edgecolor('darkred')
            bar.set_linewidth(2.5)

    # --- Panel C : R-dominated analysis ---
    ax = axes[2]
    rdom_cats = df[df['category'].str.contains('R-dom|R-DOM')]
    names_c = rdom_cats['category'].values
    a_vals_c = rdom_cats['a'].values
    a_errs_c = rdom_cats['a_err'].values
    n_vals_c = rdom_cats['n'].values

    colors_c = plt.cm.Reds(np.linspace(0.3, 0.9, len(names_c)))
    x_pos = range(len(names_c))
    bars_c = ax.bar(x_pos, a_vals_c, yerr=a_errs_c, color=colors_c,
                    edgecolor='black', capsize=6, alpha=0.85, linewidth=1.2)
    ax.axhline(0, color='blue', linestyle='--', linewidth=2, alpha=0.7,
              label='Sign inversion')
    ax.set_xticks(x_pos)
    short_labels = [n.replace('TRUE ', '').replace('True ', '').replace('R-DOM', 'R-dom')
                    for n in names_c]
    ax.set_xticklabels(short_labels, rotation=30, ha='right')
    ax.set_ylabel('U-shape Coefficient $a$')
    ax.set_title('C) R-Dominated Systems', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "figure2_inversion.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n[OK] Saved: {output_path}")
    print("     figsize : 15 x 6.0 (etait 18 x 6.5)")
    print("     fonts   : axes=20, titles=21, ticks=18")


if __name__ == "__main__":
    main()

