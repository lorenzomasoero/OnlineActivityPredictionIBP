"""
10_case_studies.py
==================
Case study trajectory plots: truth vs. predictions for selected experiments.
Addresses T1.b (practical impact) and T2.e (decision-making on real data).

Shows: "with 7 days of pilot, here's what each method predicted vs. what
actually happened." This is the concrete evidence R2 asked for.

Output:
    fig_case_study_uci.pdf     — 3-panel UCI case studies
    fig_case_study_rees46.pdf  — 3-panel REES46 case studies (from cumulative curves)
    fig_case_study_asos.pdf    — 3-panel ASOS case studies

Usage:  python3 SubmissionAOAS/lom_revision/scripts/10_case_studies.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..', '..')
PLOT_DIR = os.path.join(SCRIPT_DIR, '..', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

from plot_style import COLORS, MARKERS, LINESTYLES, LABELS, apply_style
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

        # Predictions (stored as new users; shift up by N_pilot for absolute scale)
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
                # Only have final prediction — draw as a point at day D
                ax.scatter([D], [N_pilot + mr['U_hat']], color=COLORS[m],
                           marker=MARKERS[m], s=60, zorder=4,
                           label=f'{LABELS[m]} (v={mr["accuracy_v"]:.2f})')
                # Draw dashed line from pilot end to prediction
                ax.plot([D0, D], [N_pilot, N_pilot + mr['U_hat']],
                        color=COLORS[m], linestyle=':', alpha=0.4)

            # CI error bar at the final day (absolute scale)
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
uci = np.load(os.path.join(BASE, 'results/uci_all_results.npy'),
              allow_pickle=True).item()

# Pick: good pilot (exp 2, 249 pilot), medium (exp 6, 348 pilot), large (exp 12, 474 pilot)
plot_case_studies(uci, [2, 6, 12], 'UCI Online Retail',
                  'fig_case_study_uci.pdf', has_trajectory=True)

# ============================================================
# REES46 case studies (only final predictions, not trajectories)
# ============================================================
rees = np.load(os.path.join(BASE, 'results/rees46_all_results.npy'),
               allow_pickle=True).item()

# Pick experiments 0, 3, 6
plot_case_studies(rees, [0, 3, 6], 'REES46',
                  'fig_case_study_rees46.pdf', has_trajectory=False)

# ============================================================
# ASOS case studies (only final predictions)
# ============================================================
asos = np.load(os.path.join(BASE, 'results/asos_all_results.npy'),
               allow_pickle=True).item()

# Pick 3 arms with different D1 lengths
asos_keys = list(asos.keys())
asos_by_d1 = sorted(asos_keys, key=lambda k: asos[k]['D1'])
# Short, medium, long
picks = [
    asos_by_d1[len(asos_by_d1) // 10],      # ~10th percentile D1
    asos_by_d1[len(asos_by_d1) // 2],        # median D1
    asos_by_d1[int(len(asos_by_d1) * 0.85)], # ~85th percentile D1
]

plot_case_studies(asos, picks, 'ASOS',
                  'fig_case_study_asos.pdf', has_trajectory=False)

print("\nAll case studies done.")
