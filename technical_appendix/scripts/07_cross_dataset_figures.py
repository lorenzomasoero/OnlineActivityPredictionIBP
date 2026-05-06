"""
07_cross_dataset_figures.py
===========================
Cross-dataset accuracy visuals: boxplots and ranking chart
spanning UCI, REES46, and ASOS.

Usage:  python3 SubmissionAOAS/lom_revision/scripts/07_cross_dataset_figures.py
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

methods = ['TG_SSP', 'IBP', 'NB_SSP_regression']

def get_accs(results, methods):
    """Extract per-arm accuracy for each method."""
    out = {m: [] for m in methods}
    for key in results:
        r = results[key]
        for m in methods:
            if m in r and isinstance(r[m], dict) and 'accuracy_v' in r[m]:
                out[m].append(r[m]['accuracy_v'])
    return out

# Load all results
uci = np.load(os.path.join(BASE, 'results/uci_all_results.npy'), allow_pickle=True).item()
rees_s = np.load(os.path.join(BASE, 'results/rees46_all_results.npy'), allow_pickle=True).item()
rees_k21 = np.load(os.path.join(BASE, 'results/rees46_rolling_k21_results.npy'), allow_pickle=True).item()
asos = np.load(os.path.join(BASE, 'results/asos_all_results.npy'), allow_pickle=True).item()

datasets = [
    ('UCI\n(n=13, ratio=3)', uci),
    ('REES46 sliding\n(n=7, ratio=1.8)', rees_s),
    ('REES46 k=21\n(n=186, ratio=3)', rees_k21),
    ('ASOS\n(n=144, ratio=8)', asos),
]

# ============================================================
# Figure 1: Grouped boxplots — accuracy by dataset and method
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

n_datasets = len(datasets)
n_methods = len(methods)
width = 0.22
positions = []

for d_idx, (dname, dres) in enumerate(datasets):
    accs = get_accs(dres, methods)
    for m_idx, m in enumerate(methods):
        pos = d_idx * (n_methods + 1) * width + m_idx * width
        positions.append(pos)
        vals = accs[m]
        if vals:
            bp = ax.boxplot([vals], positions=[pos], widths=width * 0.8,
                            patch_artist=True,
                            medianprops=dict(color='black', linewidth=1.5),
                            flierprops=dict(marker='.', markersize=2, alpha=0.3),
                            whiskerprops=dict(linewidth=0.8),
                            capprops=dict(linewidth=0.8))
            bp['boxes'][0].set_facecolor(COLORS[m])
            bp['boxes'][0].set_alpha(0.6)

# X-axis labels
group_centers = []
for d_idx in range(n_datasets):
    center = d_idx * (n_methods + 1) * width + (n_methods - 1) * width / 2
    group_centers.append(center)

ax.set_xticks(group_centers)
ax.set_xticklabels([d[0] for d in datasets], fontsize=9)
ax.set_ylabel(r'Accuracy $v_{D_0}^{(D_1)}$')
ax.set_title('Prediction accuracy across datasets and methods')
ax.set_ylim(-0.05, 1.1)
ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5)

# Legend
from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=COLORS[m], alpha=0.6, label=LABELS[m]) for m in methods]
ax.legend(handles=legend_patches, loc='lower left', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')

fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, 'fig_cross_dataset_boxplot.pdf'))
plt.close()
print("Saved fig_cross_dataset_boxplot.pdf")

# ============================================================
# Figure 2: Ranking frequency across all datasets combined
# ============================================================
# Pool all arms that have all 3 methods
all_arms = []
for dname, dres in datasets:
    for key in dres:
        r = dres[key]
        row = {}
        complete = True
        for m in methods:
            if m in r and isinstance(r[m], dict) and 'accuracy_v' in r[m]:
                row[m] = r[m]['accuracy_v']
            else:
                complete = False
        if complete:
            all_arms.append(row)

rank_counts = {m: {1: 0, 2: 0, 3: 0} for m in methods}
for row in all_arms:
    ranked = sorted(methods, key=lambda m: -row[m])
    for rank, m in enumerate(ranked, 1):
        rank_counts[m][rank] += 1

n_total = len(all_arms)

fig, ax = plt.subplots(figsize=(5, 4))

x = np.arange(len(methods))
w = 0.25
rank_colors = ['#2ca02c', '#ff7f0e', '#d62728']
rank_labels = ['Best', '2nd', '3rd']

for ri, (rank, color, label) in enumerate(zip([1, 2, 3], rank_colors, rank_labels)):
    fracs = [100 * rank_counts[m][rank] / n_total for m in methods]
    bars = ax.bar(x + ri * w, fracs, w, label=label, color=color, alpha=0.7)
    for bar, frac in zip(bars, fracs):
        if frac > 3:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{frac:.0f}%', ha='center', va='bottom', fontsize=7)

ax.set_ylabel('Fraction of experiments (%)')
ax.set_title(f'Ranking frequency (all datasets, n={n_total})')
ax.set_xticks(x + w)
ax.set_xticklabels([LABELS[m] for m in methods])
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 65)

fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, 'fig_cross_dataset_ranking.pdf'))
plt.close()
print("Saved fig_cross_dataset_ranking.pdf")

# ============================================================
# Figure 3: Median accuracy vs extrapolation ratio (all datasets)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))

dataset_configs = [
    ('UCI', uci, 3.0, 'o'),
    ('REES46 sliding', rees_s, 1.8, 's'),
    ('REES46 k=21', rees_k21, 3.0, '^'),
    ('ASOS', asos, 7.9, 'D'),
]

for m in methods:
    xs = []
    ys = []
    for dname, dres, ratio, _ in dataset_configs:
        accs = get_accs(dres, [m])[m]
        if accs:
            xs.append(ratio)
            ys.append(np.median(accs))
    ax.plot(xs, ys, marker='o', color=COLORS[m], linewidth=1.5,
            markersize=8, label=LABELS[m], linestyle='--', alpha=0.8)

# Annotate dataset names
for dname, dres, ratio, marker in dataset_configs:
    ax.annotate(dname, xy=(ratio, -0.08), fontsize=7, color='gray',
                ha='center', rotation=0)

ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel(r'Median extrapolation ratio $D_1 / D_0$')
ax.set_ylabel(r'Median accuracy $v_{D_0}^{(D_1)}$')
ax.set_title('Accuracy vs. extrapolation difficulty across datasets')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.15, 1.05)
ax.set_xlim(0, 10)

fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, 'fig_cross_dataset_accuracy_vs_ratio.pdf'))
plt.close()
print("Saved fig_cross_dataset_accuracy_vs_ratio.pdf")

# Print summary
print(f"\n=== Cross-Dataset Ranking ({n_total} total experiments) ===")
for m in methods:
    pct = 100 * rank_counts[m][1] / n_total
    print(f"  {LABELS[m]:15s}: best {pct:.0f}%")
