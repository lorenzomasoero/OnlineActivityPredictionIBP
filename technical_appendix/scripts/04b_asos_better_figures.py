"""
04b_asos_better_figures.py
==========================
Cleaner ASOS accuracy figures replacing the messy scatter.

(i)  Absolute accuracy: overlaid boxplots per method
(ii) Relative accuracy: fraction of arms where each method ranks 1st, 2nd, 3rd

Usage:  python3 SubmissionAOAS/lom_revision/scripts/04b_asos_better_figures.py
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

results = np.load(os.path.join(BASE, 'results/asos_all_results.npy'),
                  allow_pickle=True).item()

methods = ['TG_SSP', 'IBP', 'NB_SSP_regression']
method_labels = [LABELS[m] for m in methods]
method_colors = [COLORS[m] for m in methods]

# Collect per-arm accuracies
arm_data = []
for key, r in results.items():
    row = {'ratio': r['D1'] / r['D0']}
    for m in methods:
        if m in r and isinstance(r[m], dict) and 'accuracy_v' in r[m]:
            row[m] = r[m]['accuracy_v']
    if all(m in row for m in methods):
        arm_data.append(row)

print(f"Arms with all 3 methods: {len(arm_data)}")

# ============================================================
# Figure 1: Absolute accuracy — side-by-side boxplots
# ============================================================
fig, ax = plt.subplots(figsize=(5, 4.5))

box_data = [[d[m] for d in arm_data] for m in methods]
bp = ax.boxplot(box_data, labels=method_labels, patch_artist=True,
                widths=0.5,
                medianprops=dict(color='black', linewidth=2),
                flierprops=dict(marker='.', markersize=3, alpha=0.3))

for patch, color in zip(bp['boxes'], method_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label='v = 0.8')
ax.set_ylabel(r'Accuracy $v_{D_0}^{(D_1)}$')
ax.set_title('ASOS: Accuracy distribution by method (144 arms)')
ax.set_ylim(-0.05, 1.05)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Annotate medians
for i, m in enumerate(methods):
    vals = [d[m] for d in arm_data]
    med = np.median(vals)
    ax.annotate(f'med={med:.2f}', xy=(i + 1, med),
                xytext=(i + 1.35, med + 0.05),
                fontsize=8, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, 'fig_asos_accuracy_boxplot.pdf'))
plt.close()
print("Saved fig_asos_accuracy_boxplot.pdf")

# ============================================================
# Figure 2: Relative accuracy — ranking frequency
# ============================================================
rank_counts = {m: {1: 0, 2: 0, 3: 0} for m in methods}

for d in arm_data:
    accs = [(d[m], m) for m in methods]
    accs.sort(key=lambda x: -x[0])  # highest accuracy first
    for rank, (_, m) in enumerate(accs, 1):
        rank_counts[m][rank] += 1

n_arms = len(arm_data)

fig, ax = plt.subplots(figsize=(5, 4))

x = np.arange(len(methods))
width = 0.25
rank_colors = ['#2ca02c', '#ff7f0e', '#d62728']  # green, orange, red
rank_labels = ['Best', '2nd', '3rd']

for rank_idx, (rank, color, label) in enumerate(zip([1, 2, 3], rank_colors, rank_labels)):
    fracs = [100 * rank_counts[m][rank] / n_arms for m in methods]
    bars = ax.bar(x + rank_idx * width, fracs, width,
                  label=label, color=color, alpha=0.7)
    # Add percentage labels on bars
    for bar, frac in zip(bars, fracs):
        if frac > 5:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{frac:.0f}%', ha='center', va='bottom', fontsize=7)

ax.set_ylabel('Fraction of arms (%)')
ax.set_title('ASOS: How often does each method rank best?')
ax.set_xticks(x + width)
ax.set_xticklabels(method_labels)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 75)

fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, 'fig_asos_ranking_frequency.pdf'))
plt.close()
print("Saved fig_asos_ranking_frequency.pdf")

# ============================================================
# Figure 3: Accuracy vs ratio — binned means with error bars
#           (replaces the messy scatter)
# ============================================================
ratio_bins = [(0, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 40)]
bin_labels = [f'{lo}-{hi}' for lo, hi in ratio_bins]

fig, ax = plt.subplots(figsize=(7, 4))

for m_idx, m in enumerate(methods):
    means = []
    q25s = []
    q75s = []
    xs = []
    for lo, hi in ratio_bins:
        subset = [d[m] for d in arm_data if lo < d['ratio'] <= hi]
        if len(subset) >= 3:
            means.append(np.median(subset))
            q25s.append(np.percentile(subset, 25))
            q75s.append(np.percentile(subset, 75))
            xs.append((lo + hi) / 2)

    ax.errorbar(np.array(xs) + m_idx * 0.5, means,
                yerr=[np.array(means) - np.array(q25s),
                      np.array(q75s) - np.array(means)],
                marker=MARKERS[m], color=COLORS[m],
                linestyle='-', linewidth=1.5, capsize=3,
                label=LABELS[m], markersize=6)

ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel(r'Extrapolation ratio $D_1 / D_0$')
ax.set_ylabel(r'Median accuracy $v_{D_0}^{(D_1)}$')
ax.set_title('ASOS: Accuracy vs. extrapolation ratio (median ± IQR)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, 'fig_asos_accuracy_vs_ratio_binned.pdf'))
plt.close()
print("Saved fig_asos_accuracy_vs_ratio_binned.pdf")

# ============================================================
# Print summary
# ============================================================
print(f"\n=== Ranking Summary ({n_arms} arms) ===")
for m in methods:
    pct_best = 100 * rank_counts[m][1] / n_arms
    pct_top2 = 100 * (rank_counts[m][1] + rank_counts[m][2]) / n_arms
    print(f"  {LABELS[m]:15s}: best {pct_best:.0f}%, top-2 {pct_top2:.0f}%")
