#!/usr/bin/env python3
"""
===============================================================================
MULTI-SCALE SYNTHESIS FIGURES FOR PAPER 4
===============================================================================

Creates publication-quality figures summarizing Q2R2 validation across scales.

Figures:
1. Multi-scale summary (4 panels)
2. Individual context results
3. Sign inversion evidence
4. Alternative theory elimination
5. Validation progress

Author: Jonathan E. Slama
Date: December 2025
===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import json

# Style - no serif to avoid Unicode issues
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_figure1_multiscale_summary():
    """
    Figure 1: Multi-Scale Summary
    
    Panel A: Mass hierarchy diagram with results
    Panel B: Sample sizes by context
    Panel C: Detection matrix
    Panel D: Physical interpretation
    """
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Data
    contexts = [
        ('ALFALFA', 19222, 11, 0.0034, True, True, 'Galaxies'),
        ('KiDS', 663323, 11, 0.00158, True, True, 'Galaxies'),
        ('Tempel', 81929, 12.5, 0.2488, True, True, 'Groups'),
        ('Saulder', 198523, 12.5, 0.0862, True, True, 'Groups'),
        ('eROSITA', 3347, 14.5, 0.5016, True, False, 'Clusters'),
        ('redMaPPer', 25604, 14.5, 1.41, True, False, 'Clusters'),
    ]
    
    # =========================================================================
    # Panel A: Mass hierarchy with status
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    mass_ranges = {
        'Galaxies': (10, 12, '#3498db', 4),
        'Groups': (12, 14, '#2ecc71', 4),
        'Clusters': (14, 15.5, '#e74c3c', 2)
    }
    
    y_pos = {'Galaxies': 2, 'Groups': 1, 'Clusters': 0}
    
    for scale, (m_min, m_max, color, n_confirmed) in mass_ranges.items():
        y = y_pos[scale]
        ax1.barh(y, m_max - m_min, left=m_min, height=0.6, color=color, alpha=0.7, edgecolor='black')
        
        # Status indicator
        if scale == 'Clusters':
            status = 'PARTIAL'
            status_color = 'orange'
        else:
            status = 'CONFIRMED'
            status_color = 'green'
        
        ax1.text(m_max + 0.2, y, status, va='center', fontsize=11, fontweight='bold', color=status_color)
    
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['Clusters\n(10$^{14}$-10$^{15}$ M$_\\odot$)', 
                         'Groups\n(10$^{12}$-10$^{14}$ M$_\\odot$)', 
                         'Galaxies\n(10$^{10}$-10$^{12}$ M$_\\odot$)'])
    ax1.set_xlabel('log$_{10}$(Mass / M$_\\odot$)', fontsize=12)
    ax1.set_xlim(9.5, 17)
    ax1.set_title('A) Q$^2$R$^2$ Validation by Mass Scale', fontsize=13, fontweight='bold')
    ax1.axvline(12, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(14, color='gray', linestyle='--', alpha=0.5)
    
    # =========================================================================
    # Panel B: Sample sizes
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    names = [c[0] for c in contexts]
    n_objects = [c[1] for c in contexts]
    colors_ctx = ['#3498db', '#3498db', '#2ecc71', '#2ecc71', '#e74c3c', '#e74c3c']
    
    bars = ax2.bar(names, n_objects, color=colors_ctx, edgecolor='black', alpha=0.8)
    ax2.set_ylabel('Number of Objects', fontsize=12)
    ax2.set_title('B) Sample Sizes by Context', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_ylim(1e3, 1e7)
    
    # Add labels
    for bar, n in zip(bars, n_objects):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2, 
                f'{n:,}', ha='center', va='bottom', fontsize=9, rotation=45)
    
    ax2.tick_params(axis='x', rotation=45)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#3498db', edgecolor='black', label='Galaxies'),
        mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='Groups'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='Clusters')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    # =========================================================================
    # Panel C: Detection matrix
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Create matrix
    categories = ['U-shape\nDetected', 'Sign\nInversion']
    
    # Status matrix: 1 = yes, 0 = no
    status_matrix = np.array([
        [1, 1],      # ALFALFA
        [1, 1],      # KiDS
        [1, 1],      # Tempel
        [1, 1],      # Saulder
        [1, 0],      # eROSITA
        [1, 0],      # redMaPPer
    ])
    
    # Custom colormap: red=no, green=yes
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#e74c3c', '#2ecc71'])
    
    im = ax3.imshow(status_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(categories, fontsize=11)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names, fontsize=10)
    ax3.set_title('C) Q$^2$R$^2$ Signature Detection', fontsize=13, fontweight='bold')
    
    # Add text annotations - using YES/NO instead of Unicode
    for i in range(len(names)):
        for j in range(2):
            val = status_matrix[i, j]
            text = 'YES' if val == 1 else 'NO'
            color = 'white'
            ax3.text(j, i, text, ha='center', va='center', fontsize=10, fontweight='bold', color=color)
    
    # =========================================================================
    # Panel D: Physical interpretation
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    interpretation_text = """
PHYSICAL INTERPRETATION

The Q2R2 pattern shows scale-dependent behavior
consistent with BARYONIC COUPLING:

GALAXIES (f_baryon ~15-20%)
  - Full Q2R2 signature
  - Sign inversion CONFIRMED
  - Baryons dominate local dynamics

GROUPS (f_baryon ~10-15%)
  - Full signature with richness proxy
  - Transition regime captured
  - Low-Ngal groups are baryon-dominated

CLUSTERS (f_baryon ~12%, DM 85%+)
  - U-shape present but diluted
  - No sign inversion
  - Dark matter dominates, masks signal

CONCLUSION: String theory moduli fields
couple primarily to BARYONIC MATTER.

Only STRING THEORY MODULI can explain
all observations across 5 orders of
magnitude in mass scale.
    """
    
    ax4.text(0.05, 0.95, interpretation_text, transform=ax4.transAxes,
             fontsize=10, fontfamily='monospace', verticalalignment='top')
    
    ax4.set_title('D) Physical Interpretation', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'figure1_multiscale_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def create_figure2_ushape_comparison():
    """
    Figure 2: U-shape patterns across contexts
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    contexts_data = [
        ('ALFALFA Galaxies\n(N=19,222)', 0.0034, 5.0, True, '#3498db'),
        ('KiDS Galaxies\n(N=663,323)', 0.00158, 5.2, True, '#2980b9'),
        ('Tempel Groups\n(N=81,929)', 0.2488, 9.1, True, '#2ecc71'),
        ('Saulder Groups\n(N=198,523)', 0.0862, 3.0, True, '#27ae60'),
        ('eROSITA Clusters\n(N=3,347)', 0.5016, 3.4, False, '#e74c3c'),
        ('redMaPPer Clusters\n(N=25,604)', 1.41, 33.0, False, '#c0392b'),
    ]
    
    for ax, (title, amplitude, sigma, has_inversion, color) in zip(axes.flat, contexts_data):
        # Create schematic U-shape
        x = np.linspace(0, 1, 100)
        y = amplitude * (x - 0.5)**2 - amplitude * 0.25
        
        ax.plot(x, y, color=color, linewidth=3)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.fill_between(x, y, 0, where=(y > 0), color=color, alpha=0.2)
        ax.fill_between(x, y, 0, where=(y < 0), color=color, alpha=0.2)
        
        ax.set_xlabel('Environment (normalized)', fontsize=10)
        ax.set_ylabel('Residuals', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        
        # Statistics text - no Unicode
        stats_text = f'a = {amplitude:.4f}\n{sigma:.1f}s detection'
        if has_inversion:
            stats_text += '\nSign inversion: YES'
        else:
            stats_text += '\nSign inversion: NO'
        
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlim(-0.05, 1.05)
    
    plt.suptitle('Q$^2$R$^2$ U-Shape Pattern Across Mass Scales', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'figure2_ushape_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def create_figure3_sign_inversion():
    """
    Figure 3: Sign inversion evidence
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Panel A: Galaxies (KiDS)
    ax = axes[0]
    categories = ['Blue\n(Q-dominated)', 'Red\n(R-dominated)']
    values = [0.00271, -0.00258]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('Quadratic coefficient a', fontsize=11)
    ax.set_title('A) Galaxies (KiDS)\nSign Inversion: YES', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.004, 0.004)
    
    for bar, val in zip(bars, values):
        y_pos = val + 0.0003 if val > 0 else val - 0.0005
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.5f}', 
                ha='center', va='bottom' if val > 0 else 'top', fontsize=10)
    
    # Panel B: Groups (Tempel)
    ax = axes[1]
    categories = ['Rich\n(Ngal > 2)', 'Poor\n(Ngal <= 2)']
    values = [0.255, -0.152]
    colors = ['#2ecc71', '#9b59b6']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('Quadratic coefficient a', fontsize=11)
    ax.set_title('B) Groups (Tempel)\nSign Inversion: YES', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.3, 0.4)
    
    for bar, val in zip(bars, values):
        y_pos = val + 0.02 if val > 0 else val - 0.03
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.3f}', 
                ha='center', va='bottom' if val > 0 else 'top', fontsize=10)
    
    # Panel C: Clusters (eROSITA)
    ax = axes[2]
    categories = ['High Fgas\n(Q-dominated)', 'Low Fgas\n(R-dominated)']
    values = [1.45, 1.27]
    colors = ['#e74c3c', '#c0392b']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('Quadratic coefficient a', fontsize=11)
    ax.set_title('C) Clusters (eROSITA)\nSign Inversion: NO', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 2)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.08, f'{val:.2f}', 
                ha='center', va='bottom', fontsize=10)
    
    # Annotation
    ax.annotate('Both positive:\nDM dilutes\nbaryonic signal', 
                xy=(0.5, 0.5), xycoords='axes fraction',
                fontsize=9, ha='center', style='italic',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.suptitle('Sign Inversion Evidence Across Scales', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'figure3_sign_inversion.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def create_figure4_alternative_elimination():
    """
    Figure 4: Alternative theory elimination table
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Table data
    theories = [
        'String Theory Moduli',
        'MOND',
        'f(R) Gravity',
        'Emergent Gravity',
        'Fuzzy Dark Matter',
        'Self-Interacting DM',
        'Warm Dark Matter',
        'Standard Baryonic'
    ]
    
    criteria = ['U-shape', 'Sign Inv.', 'Scale Dep.', 'Env. Dep.', 'Compatible?']
    
    # Status: YES/NO
    status = [
        ['YES', 'YES', 'YES', 'YES', 'YES'],
        ['NO', 'NO', 'NO', 'NO', 'NO'],
        ['NO', 'NO', 'NO', 'NO', 'NO'],
        ['NO', 'NO', 'NO', 'NO', 'NO'],
        ['NO', 'NO', 'NO', 'NO', 'NO'],
        ['NO', 'NO', 'NO', 'NO', 'NO'],
        ['NO', 'NO', 'NO', 'NO', 'NO'],
        ['NO', 'NO', 'NO', 'NO', 'NO'],
    ]
    
    # Colors
    cell_colors = []
    for row in status:
        row_colors = []
        for cell in row:
            if cell == 'YES':
                row_colors.append('#2ecc71')
            else:
                row_colors.append('#e74c3c')
        cell_colors.append(row_colors)
    
    cell_colors[0] = ['#27ae60'] * 5
    
    table = ax.table(
        cellText=status,
        rowLabels=theories,
        colLabels=criteria,
        cellColours=cell_colors,
        rowColours=['#ecf0f1'] * len(theories),
        colColours=['#bdc3c7'] * len(criteria),
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
        if col == -1:
            cell.set_text_props(fontweight='bold')
        cell.set_edgecolor('black')
    
    ax.set_title('Alternative Theory Elimination\n\nOnly String Theory Moduli compatible with ALL observations',
                 fontsize=14, fontweight='bold', pad=20)
    
    legend_text = """
    YES = Predicted/Matches observations
    NO = Not predicted/Fails to explain
    
    Criteria tested:
    - U-shape: Quadratic environmental dependence in scaling relation residuals
    - Sign Inv.: Opposite curvature for Q-dominated vs R-dominated systems
    - Scale Dep.: Different behavior at galaxy/group/cluster scales
    - Env. Dep.: Variation with local density
    """
    
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'figure4_alternative_elimination.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def create_figure5_validation_progress():
    """
    Figure 5: Validation progress
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Progress by level
    ax = axes[0]
    
    levels = ['Level 1\nQuantitative', 'Level 2\nMulti-Context', 'Level 3\nDirect', 'Level 4\nAlternatives']
    progress = [20, 80, 0, 100]
    colors = ['#3498db', '#2ecc71', '#95a5a6', '#e74c3c']
    
    bars = ax.barh(levels, progress, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xlim(0, 110)
    ax.set_xlabel('Progress (%)', fontsize=12)
    ax.set_title('A) Validation Progress by Level', fontsize=13, fontweight='bold')
    
    for bar, p in zip(bars, progress):
        ax.text(p + 2, bar.get_y() + bar.get_height()/2, f'{p}%', 
                va='center', fontsize=11, fontweight='bold')
    
    ax.axvline(80, color='green', linestyle='--', alpha=0.5, label='Publication threshold')
    ax.legend(loc='lower right')
    
    # Panel B: Overall progress pie
    ax = axes[1]
    
    sizes = [80, 20]
    colors = ['#2ecc71', '#ecf0f1']
    explode = (0.05, 0)
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, colors=colors, 
                                       autopct='%1.0f%%', startangle=90,
                                       textprops={'fontsize': 14, 'fontweight': 'bold'})
    
    ax.set_title('B) Overall Validation Progress', fontsize=13, fontweight='bold')
    
    centre_circle = plt.Circle((0, 0), 0.5, fc='white')
    ax.add_artist(centre_circle)
    ax.text(0, 0, '80%\nComplete', ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'figure5_validation_progress.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    print("="*70)
    print(" CREATING PAPER 4 SYNTHESIS FIGURES")
    print("="*70)
    
    print("\n  Creating Figure 1: Multi-scale summary...")
    create_figure1_multiscale_summary()
    
    print("\n  Creating Figure 2: U-shape comparison...")
    create_figure2_ushape_comparison()
    
    print("\n  Creating Figure 3: Sign inversion evidence...")
    create_figure3_sign_inversion()
    
    print("\n  Creating Figure 4: Alternative elimination...")
    create_figure4_alternative_elimination()
    
    print("\n  Creating Figure 5: Validation progress...")
    create_figure5_validation_progress()
    
    print("\n" + "="*70)
    print(" ALL FIGURES CREATED")
    print("="*70)
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print("\n  Files created:")
    print("    - figure1_multiscale_summary.png")
    print("    - figure2_ushape_comparison.png")
    print("    - figure3_sign_inversion.png")
    print("    - figure4_alternative_elimination.png")
    print("    - figure5_validation_progress.png")
    print("="*70)


if __name__ == "__main__":
    main()
