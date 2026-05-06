"""
12_proprietary_accuracy_figure.py
=================================
Regenerate proprietary data accuracy boxplot using new-users metric.

Output:
    plots/PAPER_proprietary_28_accuracy_v2.pdf

Usage: python3 SubmissionAOAS/lom_revision/scripts/12_proprietary_accuracy_figure.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..', '..')
PLOT_DIR = os.path.join(SCRIPT_DIR, '..', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

from plot_style import COLORS, apply_style
apply_style()

data_path = os.path.join(SCRIPT_DIR, '../../../code_AISTATS/results/real_data/PAPER_amazon_fit_updated.npy')
results = np.load(data_path, allow_pickle=True).item()

D0 = 7

# Methods available in proprietary data — include all that were in the original figure
methods = {
    'SSP_regression': ('Be-SSP', '#e41a1c'),
    'SSP_geom': ('TG-SSP', '#d95f02'),
    'NBP_regression': ('NB-SSP', COLORS['NB_SSP_regression']),
    'IBP_regression': ('IBP', COLORS['IBP']),
    'BB': ('BB', '#984ea3'),
    'BG': ('BG', '#a65628'),
    'J': ('Jackknife', '#f781bf'),
    'GT': ('Good-Toulmin', '#999999'),
    'LP': ('LP', '#66c2a5'),
}

accs = {label: [] for _, (label, _) in methods.items()}

for exp_id, exp in results.items():
    for arm in ['C', 'T']:
        if arm not in exp:
            continue
        d = exp[arm]
        true_curve = d.get('True', None)
        if true_curve is None or len(true_curve) <= D0:
            continue

        N_pilot = true_curve[D0]
        U_true = true_curve[-1] - N_pilot
        if U_true <= 0:
            continue

        for m_key, (m_label, _) in methods.items():
            if m_key not in d:
                continue
            pred = d[m_key]
            # Handle special formats
            if m_key == 'J':
                # Jackknife: shape (4, D), use order 3 (index 2)
                if hasattr(pred, 'shape') and len(pred.shape) == 2:
                    pred_curve = pred[2]  # J3
                else:
                    continue
            elif m_key == 'GT':
                # Good-Toulmin: tuple (point_estimate, ...)
                if isinstance(pred, tuple):
                    pred_curve = pred[0]
                else:
                    pred_curve = pred
            else:
                pred_curve = pred
            
            if len(pred_curve) != len(true_curve):
                continue
            U_hat = pred_curve[-1] - N_pilot
            v = 1 - min(abs(U_true - U_hat) / max(U_true, 1), 1)
            accs[m_label].append(v)

# Plot
labels = [m_label for _, (m_label, _) in methods.items()]
colors = [color for _, (_, color) in methods.items()]
data = [accs[l] for l in labels]

fig, ax = plt.subplots(figsize=(10, 5))

bp = ax.boxplot(data, labels=labels, vert=True, patch_artist=True,
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=2),
                flierprops=dict(marker='.', markersize=3, alpha=0.3))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_ylabel(r'Prediction Accuracy $v_{7}^{(21)}$', fontsize=14)
ax.set_xlabel('Methods', fontsize=14)
ax.set_ylim([0, 1.02])
ax.set_yticks(np.linspace(0, 1, 6))
ax.set_yticklabels([f'{int(i)}%' for i in np.linspace(0, 100, 6)], fontsize=12)
ax.tick_params(axis='x', labelsize=13)
ax.grid(True, alpha=0.3, axis='y')

fig.tight_layout()
outpath = os.path.join(PLOT_DIR, 'PAPER_proprietary_28_accuracy_v2.pdf')
fig.savefig(outpath)
plt.close()
print(f"Saved {outpath}")

for label, vals in zip(labels, data):
    print(f"  {label}: median={np.median(vals):.3f}, mean={np.mean(vals):.3f}, n={len(vals)}")
