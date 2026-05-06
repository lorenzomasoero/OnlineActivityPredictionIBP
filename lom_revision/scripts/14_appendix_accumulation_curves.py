"""
14_appendix_accumulation_curves.py
==================================
Per-experiment accumulation curves for the appendix.
Each panel: truth (black) + method predictions over time.

Output:
    fig_appendix_uci_accumulation.pdf
    fig_appendix_rees46_accumulation.pdf

Usage: python3 SubmissionAOAS/lom_revision/scripts/14_appendix_accumulation_curves.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..', '..')
PLOT_DIR = os.path.join(SCRIPT_DIR, '..', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

from plot_style import COLORS, LINESTYLES, MARKERS, LABELS, apply_style
apply_style()

METHODS = ['TG_SSP', 'IBP', 'NB_SSP_regression']


def plot_accumulation_grid(results, exp_keys, title, filename, ncols=4):
    n = len(exp_keys)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8))
    axes_flat = axes.flatten() if nrows > 1 else (axes if ncols > 1 else [axes])

    for i, key in enumerate(exp_keys):
        ax = axes_flat[i]
        r = results[key]
        cum = np.array(r['cumulative_users'])
        D0 = r['D0']
        days = np.arange(len(cum))
        truth_new = cum - cum[D0]

        ax.plot(days, truth_new, 'k-', linewidth=1.5, label='Truth')
        ax.axvspan(0, D0, alpha=0.06, color='gray')
        ax.axvline(D0, color='gray', ls=':', lw=0.6, alpha=0.6)

        for m in METHODS:
            if m not in r or not isinstance(r[m], dict):
                continue
            mr = r[m]
            if 'U_trajectory' in mr:
                traj = np.array(mr['U_trajectory'])
                pred_days = np.arange(D0 + 1, D0 + 1 + len(traj))
                v = mr.get('accuracy_v', 0)
                ax.plot(pred_days, traj, color=COLORS[m],
                        linestyle=LINESTYLES[m], linewidth=1,
                        label=f'{LABELS[m]} ({v:.2f})')
            elif 'U_hat' in mr:
                D = D0 + r['D1']
                v = mr.get('accuracy_v', 0)
                ax.plot([D0, D], [0, mr['U_hat']], color=COLORS[m],
                        linestyle=':', alpha=0.6, linewidth=1)
                ax.scatter([D], [mr['U_hat']], color=COLORS[m],
                           marker=MARKERS[m], s=25, zorder=4,
                           label=f'{LABELS[m]} ({v:.2f})')

        ax.set_title(f'Exp {key} (N={r["N_pilot"]:,})', fontsize=8, pad=2)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=5, loc='upper left', framealpha=0.8)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 1])
    outpath = os.path.join(PLOT_DIR, filename)
    fig.savefig(outpath, bbox_inches='tight')
    plt.close()
    print(f"Saved {outpath}")


# UCI — has full trajectories
uci = np.load(os.path.join(BASE, 'results/uci_all_results.npy'),
              allow_pickle=True).item()
plot_accumulation_grid(uci, sorted(uci.keys()), 
                       'UCI Online Retail — accumulation curves (all 13 experiments)',
                       'fig_appendix_uci_accumulation.pdf', ncols=4)

# REES46 sliding — only final predictions (no trajectories)
rees = np.load(os.path.join(BASE, 'results/rees46_all_results.npy'),
               allow_pickle=True).item()
plot_accumulation_grid(rees, sorted(rees.keys()),
                       'REES46 — accumulation curves (7 sliding windows)',
                       'fig_appendix_rees46_accumulation.pdf', ncols=4)

print("\nDone.")
