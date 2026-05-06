"""
11_asos_accuracy_figure.py
==========================
Regenerate ASOS accuracy boxplot using the new-users metric v_{D0}^{(D1)}
and the correct fitting methods (geometric MLE for TG-SSP).

Replaces the old PAPER_ASOS_accuracy.pdf which used cumulative total users.

Output:
    plots/PAPER_ASOS_accuracy_v2.pdf

Usage: python3 SubmissionAOAS/lom_revision/scripts/11_asos_accuracy_figure.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..', '..')
PLOT_DIR = os.path.join(SCRIPT_DIR, '..', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

from plot_style import COLORS, LABELS, apply_style
apply_style()

results = np.load(os.path.join(BASE, 'results/asos_all_results.npy'),
                  allow_pickle=True).item()

# Use TG_SSP_geom (geometric MLE) instead of TG_SSP (regression)
methods = {
    'NB_SSP_regression': 'NB-SSP',
    'TG_SSP_geom': 'TG-SSP',
    'IBP': 'IBP',
}

method_colors = {
    'NB_SSP_regression': COLORS['NB_SSP_regression'],
    'TG_SSP_geom': COLORS['TG_SSP'],
    'IBP': COLORS['IBP'],
}

# Collect accuracies
accs = {}
for m_key, m_label in methods.items():
    accs[m_label] = [results[k][m_key]['accuracy_v']
                     for k in results
                     if m_key in results[k]
                     and isinstance(results[k][m_key], dict)
                     and 'accuracy_v' in results[k][m_key]]

# Match the original figure style: vertical boxplot
fig, ax = plt.subplots(figsize=(5, 5))

labels = list(methods.values())
data = [accs[l] for l in labels]
colors = [method_colors[k] for k in methods.keys()]

bp = ax.boxplot(data, labels=labels, vert=True, patch_artist=True,
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=2),
                flierprops=dict(marker='.', markersize=3, alpha=0.3))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_ylabel(r'Prediction Accuracy $v_{7}^{(D_1)}$', fontsize=14)
ax.set_xlabel('Methods', fontsize=14)
ax.set_ylim([0, 1.02])
ax.set_yticks(np.linspace(0, 1, 6))
ax.set_yticklabels([f'{int(i)}%' for i in np.linspace(0, 100, 6)], fontsize=12)
ax.tick_params(axis='x', labelsize=13)
ax.grid(True, alpha=0.3, axis='y')

# Annotate medians
for i, (label, vals) in enumerate(zip(labels, data)):
    med = np.median(vals)
    ax.annotate(f'{med:.2f}', xy=(i + 1, med),
                xytext=(i + 1.3, med + 0.05),
                fontsize=9, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

fig.tight_layout()
outpath = os.path.join(PLOT_DIR, 'PAPER_ASOS_accuracy_v2.pdf')
fig.savefig(outpath)
plt.close()
print(f"Saved {outpath}")

# Print summary
for label, vals in zip(labels, data):
    print(f"  {label}: median={np.median(vals):.3f}, mean={np.mean(vals):.3f}, n={len(vals)}")
