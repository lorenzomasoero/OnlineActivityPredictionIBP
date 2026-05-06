"""
13_appendix_per_experiment_plots.py
====================================
Appendix per-experiment plots (tracker item B7):
  - fig_appendix_uci_per_experiment.pdf   : 13 UCI experiments (4×4 grid)
  - fig_appendix_rees46_per_window.pdf    : 7 REES46 sliding windows (2×4 grid)
  - fig_appendix_asos_selected.pdf        : 18 selected ASOS arms (3×6 grid)

Each panel shows: accuracy bar chart per method + CI coverage indicator.

Usage:
    cd SubmissionAOAS/lom_revision/scripts
    python3 13_appendix_per_experiment_plots.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..', '..')
PLOT_DIR = os.path.join(SCRIPT_DIR, '..', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

from plot_style import COLORS, LABELS, apply_style
apply_style()

METHODS = ['TG_SSP', 'IBP', 'NB_SSP_regression']
METHOD_SHORT = {'TG_SSP': 'TG', 'IBP': 'IBP', 'NB_SSP_regression': 'NB'}


def draw_panel(ax, r, title):
    """Draw a single per-experiment panel: accuracy bars + CI coverage dots."""
    accs = []
    covers = []
    colors = []
    for m in METHODS:
        if m in r and isinstance(r[m], dict) and 'accuracy_v' in r[m]:
            accs.append(r[m]['accuracy_v'])
            covers.append(r[m].get('ci_covers', None))
            colors.append(COLORS[m])
        else:
            accs.append(np.nan)
            covers.append(None)
            colors.append('#cccccc')

    x = np.arange(len(METHODS))
    bars = ax.bar(x, accs, color=colors, alpha=0.75, width=0.6, edgecolor='white')
    ax.axhline(0.8, color='gray', lw=0.6, ls=':', alpha=0.6)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_SHORT[m] for m in METHODS], fontsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_title(title, fontsize=8, pad=2)
    ax.grid(True, axis='y', alpha=0.25, lw=0.5)

    # CI coverage dot: green=covers, red=misses, gray=N/A
    for xi, cov in enumerate(covers):
        if cov is True:
            dot_color = '#2ca02c'
        elif cov is False:
            dot_color = '#d62728'
        else:
            continue
        ax.plot(xi, 1.06, 'o', color=dot_color, markersize=4, clip_on=False)

    # Annotate accuracy values
    for xi, acc in enumerate(accs):
        if not np.isnan(acc):
            ax.text(xi, acc + 0.02, f'{acc:.2f}', ha='center', va='bottom',
                    fontsize=6, color='#333333')


# ============================================================
# Figure 1: UCI — 13 experiments
# ============================================================
uci = np.load(os.path.join(BASE, 'results/uci_all_results.npy'), allow_pickle=True).item()

n_uci = 13
ncols = 4
nrows = (n_uci + ncols - 1) // ncols  # 4 rows

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.8, nrows * 2.6))
axes_flat = axes.flatten()

for i in range(n_uci):
    r = uci[i]
    d0, d1 = r['D0'], r['D1']
    draw_panel(axes_flat[i], r, f'Exp {i}  D₀={d0} D₁={d1}')

# Hide unused panels
for j in range(n_uci, len(axes_flat)):
    axes_flat[j].set_visible(False)

# Shared y-label
fig.text(0.01, 0.5, r'Accuracy $v_{D_0}^{(D_1)}$', va='center', rotation='vertical', fontsize=10)

# Legend
patches = [mpatches.Patch(color=COLORS[m], alpha=0.75, label=LABELS[m]) for m in METHODS]
ci_legend = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=6, label='CI covers'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=6, label='CI misses'),
]
fig.legend(handles=patches + ci_legend, loc='lower center', ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, -0.01), framealpha=0.9)

fig.suptitle('UCI Online Retail — per-experiment accuracy (D₀=7 pilot)', fontsize=11, y=1.01)
fig.tight_layout(rect=[0.03, 0.04, 1, 1])
out1 = os.path.join(PLOT_DIR, 'fig_appendix_uci_per_experiment.pdf')
fig.savefig(out1)
plt.close()
print(f"Saved {out1}")


# ============================================================
# Figure 2: REES46 sliding — 7 windows
# ============================================================
rees = np.load(os.path.join(BASE, 'results/rees46_all_results.npy'), allow_pickle=True).item()

n_rees = len(rees)  # 7
ncols = 4
nrows = (n_rees + ncols - 1) // ncols  # 2 rows

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.8, nrows * 2.6))
axes_flat = axes.flatten()

for i, key in enumerate(sorted(rees.keys())):
    r = rees[key]
    d0, d1 = r['D0'], r['D1']
    draw_panel(axes_flat[i], r, f'Window {key}  D₀={d0} D₁={d1}')

for j in range(n_rees, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.text(0.01, 0.5, r'Accuracy $v_{D_0}^{(D_1)}$', va='center', rotation='vertical', fontsize=10)
fig.legend(handles=patches + ci_legend, loc='lower center', ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
fig.suptitle('REES46 — per sliding-window accuracy', fontsize=11, y=1.01)
fig.tight_layout(rect=[0.03, 0.04, 1, 1])
out2 = os.path.join(PLOT_DIR, 'fig_appendix_rees46_per_window.pdf')
fig.savefig(out2)
plt.close()
print(f"Saved {out2}")


# ============================================================
# Figure 3: ASOS — selected arms (stratified by ratio)
# ============================================================
asos = np.load(os.path.join(BASE, 'results/asos_all_results.npy'), allow_pickle=True).item()

# Collect arms with all 3 methods
all_arms = []
for key, r in asos.items():
    if all(m in r and isinstance(r[m], dict) and 'accuracy_v' in r[m] for m in METHODS):
        all_arms.append((key, r, r['D1'] / r['D0']))

all_arms.sort(key=lambda x: x[2])  # sort by ratio

# Stratified selection: 3 arms per ratio bin
ratio_bins = [(0, 3), (3, 6), (6, 10), (10, 20), (20, 50), (50, 300)]
selected = []
for lo, hi in ratio_bins:
    bucket = [(k, r, rat) for k, r, rat in all_arms if lo < rat <= hi]
    # pick low, mid, high accuracy within bucket
    if len(bucket) == 0:
        continue
    bucket_sorted = sorted(bucket, key=lambda x: x[1]['NB_SSP_regression']['accuracy_v'])
    picks = []
    if len(bucket_sorted) >= 3:
        picks = [bucket_sorted[0], bucket_sorted[len(bucket_sorted) // 2], bucket_sorted[-1]]
    else:
        picks = bucket_sorted
    selected.extend(picks)

n_sel = len(selected)
ncols = 6
nrows = (n_sel + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.6))
axes_flat = axes.flatten() if nrows > 1 else axes

for i, (key, r, ratio) in enumerate(selected):
    d0, d1 = r['D0'], r['D1']
    draw_panel(axes_flat[i], r, f'{key}\nD₀={d0} D₁={d1} (×{ratio:.1f})')

for j in range(n_sel, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.text(0.01, 0.5, r'Accuracy $v_{D_0}^{(D_1)}$', va='center', rotation='vertical', fontsize=10)
fig.legend(handles=patches + ci_legend, loc='lower center', ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, -0.01), framealpha=0.9)
fig.suptitle(f'ASOS — selected arms ({n_sel} of 144), stratified by extrapolation ratio',
             fontsize=11, y=1.01)
fig.tight_layout(rect=[0.03, 0.04, 1, 1])
out3 = os.path.join(PLOT_DIR, 'fig_appendix_asos_selected.pdf')
fig.savefig(out3)
plt.close()
print(f"Saved {out3}")

print("\nB7 complete. Appendix per-experiment plots written to plots/.")
