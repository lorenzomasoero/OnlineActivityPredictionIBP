"""
case_studies.py
===============
Case study trajectory plots: truth vs. predictions for selected experiments.

Shows: "with 7 days of pilot, here's what each method predicted vs. what
actually happened."

Output:
    output/fig_case_study_uci.pdf
    output/fig_case_study_rees46.pdf
    output/fig_case_study_asos.pdf

Usage:
    python plotting/case_studies.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

RESULTS_DIR = os.path.join(REPO_ROOT, 'results')
PLOT_DIR = os.path.join(REPO_ROOT, 'output')
os.makedirs(PLOT_DIR, exist_ok=True)

from plotting.style import COLORS, MARKERS, LINESTYLES, LABELS, apply_style
apply_style()

methods = ['TG_SSP', 'IBP', 'NB_SSP_regression']

def plot_case_studies(results, exp_ids, dataset_name, filename,
                     has_trajectory=True):
    """Plot truth vs predictions for selected experiments."""
    n = len(exp_ids)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, exp_id in zip(axes, exp_ids):
        r = results[exp_id]
        cum = np.array(r['cumulative_users'])
        D0 = r['D0']
        D1 = r['D1']
        D = D0 + D1
        N_pilot = cum[D0]

        # Truth: absolute cumulative distinct users
        truth_days = np.arange(D + 1)

        # Plot truth (full curve)
        ax.plot(truth_days, cum[:D+1], color='black', linewidth=2,
                linestyle='-', label='Truth', zorder=5)

        # Pilot region shading
        ax.axvspan(0, D0, alpha=0.08, color='gray')
        ax.axvline(x=D0, color='gray', linestyle=':', alpha=0.7)

        # Predictions
        for m in methods:
            if m not in r or not isinstance(r[m], dict):
                continue
            mr = r[m]

            if has_trajectory and 'U_trajectory' in mr:
                traj = np.array(mr['U_trajectory'])
                pred_days = np.arange(D0 + 1, D0 + 1 + len(traj))
                ax.plot(pred_days, N_pilot + traj,
                        color=COLORS[m], linestyle=LINESTYLES[m],
                        marker=MARKERS[m], markevery=max(1, len(traj) // 6),
                        markersize=4, linewidth=1.2,
                        label=f'{LABELS[m]} (v={mr["accuracy_v"]:.2f})')
            elif 'U_hat' in mr:
                ax.scatter([D], [N_pilot + mr['U_hat']], color=COLORS[m],
                           marker=MARKERS[m], s=60, zorder=4,
                           label=f'{LABELS[m]} (v={mr["accuracy_v"]:.2f})')
                ax.plot([D0, D], [N_pilot, N_pilot + mr['U_hat']],
                        color=COLORS[m], linestyle=':', alpha=0.4)

            # CI error bar at the final day
            if 'U_ci' in mr:
                lo_raw, hi_raw = mr['U_ci']
                if has_trajectory and 'U_trajectory' in mr:
                    mid_abs = N_pilot + mr['U_trajectory'][-1]
                elif 'U_hat' in mr:
                    mid_abs = N_pilot + mr['U_hat']
                else:
                    continue
                lo_abs = N_pilot + lo_raw
                hi_abs = N_pilot + hi_raw
                yerr_lo = max(mid_abs - lo_abs, 0)
                yerr_hi = max(hi_abs - mid_abs, 0)
                ax.errorbar(D, mid_abs, yerr=[[yerr_lo], [yerr_hi]],
                            fmt='none', ecolor=COLORS[m], elinewidth=1.5,
                            capsize=3, capthick=1.2, alpha=0.7, zorder=3)

        ax.set_xlabel('Day')
        if ax == axes[0]:
            ax.set_ylabel('Cumulative distinct users')
        ax.set_title(f'Exp {exp_id}\n($N_{{pilot}}$={r["N_pilot"]:,}, '
                     f'$U_{{true}}$={r["U_true"]:,})',
                     fontsize=10)
        ax.legend(fontsize=7, loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{dataset_name}: Prediction trajectories vs. truth',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()
    print(f"Saved {filename}")


# ============================================================
# UCI case studies (have full trajectories)
# ============================================================
if __name__ == '__main__':
    uci = np.load(os.path.join(RESULTS_DIR, 'uci', 'uci_all_results.npy'),
                  allow_pickle=True).item()

    plot_case_studies(uci, [2, 6, 12], 'UCI Online Retail',
                      'fig_case_study_uci.pdf', has_trajectory=True)

    # ============================================================
    # REES46 case studies
    # ============================================================
    rees = np.load(os.path.join(RESULTS_DIR, 'rees46', 'rees46_all_results.npy'),
                   allow_pickle=True).item()

    plot_case_studies(rees, [0, 3, 6], 'REES46',
                      'fig_case_study_rees46.pdf', has_trajectory=False)

    # ============================================================
    # ASOS case studies
    # ============================================================
    asos = np.load(os.path.join(RESULTS_DIR, 'asos', 'asos_all_results.npy'),
                   allow_pickle=True).item()

    asos_keys = list(asos.keys())
    asos_by_d1 = sorted(asos_keys, key=lambda k: asos[k]['D1'])
    picks = [
        asos_by_d1[len(asos_by_d1) // 10],
        asos_by_d1[len(asos_by_d1) // 2],
        asos_by_d1[int(len(asos_by_d1) * 0.85)],
    ]

    plot_case_studies(asos, picks, 'ASOS',
                      'fig_case_study_asos.pdf', has_trajectory=False)

    print("\nAll case studies done.")
