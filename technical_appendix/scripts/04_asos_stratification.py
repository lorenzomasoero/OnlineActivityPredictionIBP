"""
04_asos_stratification.py
=========================
Stratifies ASOS accuracy by D1/D0 extrapolation ratio.
Addresses R2-b ("practical impact") and R2-c ("demonstrate advantage").

The raw ASOS median accuracy of 0.495 is misleading because it pools
experiments with D1/D0 ratios from 1x to 34x. Stratifying reveals that
accuracy is high for moderate extrapolation and degrades gracefully.

Output figures:
    lom_revision/plots/fig_asos_accuracy_vs_ratio.pdf
    lom_revision/plots/fig_asos_accuracy_boxplot_stratified.pdf

Usage:  python3 SubmissionAOAS/lom_revision/scripts/04_asos_stratification.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..', '..')
PLOT_DIR = os.path.join(SCRIPT_DIR, '..', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

from plot_style import COLORS, MARKERS, LABELS, apply_style
apply_style()

# ============================================================
# Load
# ============================================================
results = np.load(os.path.join(BASE, 'results/asos_all_results.npy'),
                  allow_pickle=True).item()

methods = ['TG_SSP', 'IBP', 'NB_SSP_regression']

# Collect per-arm data
data = []
for key, r in results.items():
    ratio = r['D1'] / r['D0']
    row = {'key': key, 'ratio': ratio, 'D0': r['D0'], 'D1': r['D1'],
           'N_pilot': r['N_pilot'], 'U_true': r['U_true']}
    for m in methods:
        if m in r and 'accuracy_v' in r[m]:
            row[f'{m}_v'] = r[m]['accuracy_v']
            row[f'{m}_covers'] = r[m].get('ci_covers', False)
    data.append(row)

print(f"Loaded {len(data)} ASOS arms")
print(f"D1/D0 range: {min(d['ratio'] for d in data):.1f} to {max(d['ratio'] for d in data):.1f}")

# ============================================================
# Figure 1: Accuracy vs. extrapolation ratio (scatter)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))

for m in methods:
    ratios = [d['ratio'] for d in data if f'{m}_v' in d]
    accs = [d[f'{m}_v'] for d in data if f'{m}_v' in d]
    ax.scatter(ratios, accs, s=15, alpha=0.5, marker=MARKERS[m],
               color=COLORS[m], label=LABELS[m], zorder=3)

ax.set_xlabel(r'Extrapolation ratio $D_1 / D_0$')
ax.set_ylabel(r'Accuracy $v_{D_0}^{(D_1)}$')
ax.set_title('ASOS: Prediction accuracy vs. extrapolation ratio')
ax.set_xlim(0, 36)
ax.set_ylim(-0.05, 1.05)
ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label='v = 0.8')
ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3)

fig.savefig(os.path.join(PLOT_DIR, 'fig_asos_accuracy_vs_ratio.pdf'))
plt.close()
print("Saved fig_asos_accuracy_vs_ratio.pdf")

# ============================================================
# Figure 2: Boxplot stratified by D1/D0 bins
# ============================================================
bins = [(0, 3, r'$\leq 3$'), (3, 5, '3–5'), (5, 10, '5–10'), (10, 100, r'$>10$')]

fig, axes = plt.subplots(1, len(methods), figsize=(12, 4), sharey=True)

for ax, m in zip(axes, methods):
    box_data = []
    box_labels = []
    for lo, hi, label in bins:
        subset = [d[f'{m}_v'] for d in data if lo < d['ratio'] <= hi and f'{m}_v' in d]
        if subset:
            box_data.append(subset)
            box_labels.append(f'{label}\n(n={len(subset)})')

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.5))
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.5)

    ax.set_title(LABELS[m])
    ax.set_xlabel(r'$D_1 / D_0$ bin')
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

axes[0].set_ylabel(r'Accuracy $v_{D_0}^{(D_1)}$')
fig.suptitle('ASOS: Accuracy stratified by extrapolation ratio', fontsize=13)
fig.tight_layout()

fig.savefig(os.path.join(PLOT_DIR, 'fig_asos_accuracy_boxplot_stratified.pdf'))
plt.close()
print("Saved fig_asos_accuracy_boxplot_stratified.pdf")

# ============================================================
# Print summary table
# ============================================================
print("\n=== ASOS Accuracy Stratified by D1/D0 ===\n")
header = f"{'Bin':>8} {'N':>5}"
for m in methods:
    header += f"  {LABELS[m]+' med':>14} {LABELS[m]+' mean':>14}"
print(header)
print('-' * len(header))

for lo, hi, label in bins:
    subset = [d for d in data if lo < d['ratio'] <= hi]
    if not subset:
        continue
    line = f"{label:>8} {len(subset):>5}"
    for m in methods:
        vals = [d[f'{m}_v'] for d in subset if f'{m}_v' in d]
        if vals:
            line += f"  {np.median(vals):>14.3f} {np.mean(vals):>14.3f}"
        else:
            line += f"  {'N/A':>14} {'N/A':>14}"
    print(line)

# Overall
line = f"{'ALL':>8} {len(data):>5}"
for m in methods:
    vals = [d[f'{m}_v'] for d in data if f'{m}_v' in d]
    line += f"  {np.median(vals):>14.3f} {np.mean(vals):>14.3f}"
print(line)

print("\nDone.")
