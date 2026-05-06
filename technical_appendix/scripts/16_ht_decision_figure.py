"""
16_ht_decision_figure.py
========================
Combined hitting-time decision figure:
  Top row:  2-3 trajectory case studies with decision annotations
  Bottom row: signed error boxplots framed as "planning error"

Designed to visually communicate "this impacts decision making."

Usage:
    python3 SubmissionAOAS/lom_revision/scripts/16_ht_decision_figure.py
"""

import numpy as np
import math
import sys
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..', '..')
PLOT_DIR = os.path.join(SCRIPT_DIR, '..', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

sys.path.append(os.path.join(BASE, '..', 'code_AISTATS', 'utils_folder'))
sys.path.append(SCRIPT_DIR)

from utils_GD import GD
from utils_IBP import IBP
from utils_NBP import NegBintSBSP
from utils_jack import predict_jack
from plot_style import COLORS, MARKERS, LABELS, apply_style

warnings.filterwarnings("ignore")
apply_style()
plt.rcParams.update({'font.size': 10})

ETA_VALUES = [1.5, 2.0, 3.0]
BNP_METHODS = ['NB_SSP_regression', 'TG_SSP', 'IBP']
ALL_METHODS = ['NB_SSP_regression', 'TG_SSP', 'IBP', 'linear', 'jackknife']

METHOD_DISPLAY = {
    'NB_SSP_regression': 'NB-SSP',
    'TG_SSP': 'TG-SSP',
    'IBP': 'IBP',
    'linear': 'Linear',
    'jackknife': 'Jackknife',
}

METHOD_COLORS = dict(COLORS)
METHOD_COLORS['linear'] = '#ff7f00'
METHOD_COLORS['jackknife'] = '#a65628'


def load_and_predict(dataset, k=None):
    """Load data and compute all hitting-time predictions."""
    results_dir = os.path.join(BASE, 'results')

    if dataset == 'rees46':
        data_dir = os.path.join(BASE, 'data', 'rees46_processed')
        experiments = list(np.load(os.path.join(data_dir, f'experiments_rolling_k{k}.npy'), allow_pickle=True))
        all_results = np.load(os.path.join(results_dir, f'rees46_rolling_k{k}_results.npy'), allow_pickle=True).item()
    else:
        all_results = np.load(os.path.join(results_dir, f'{dataset}_all_results.npy'), allow_pickle=True).item()
        experiments = []
        for exp_id, res in all_results.items():
            if res.get('N_pilot', 0) == 0:
                continue
            skip = any(m not in res or 'error' in res.get(m, {}) for m in BNP_METHODS)
            if skip:
                continue
            experiments.append({
                'exp_id': exp_id, 'D0': res['D0'], 'D1': res['D1'],
                'N_pilot': res['N_pilot'],
                'cumulative_users': np.array(res['cumulative_users']),
            })

    predictions = {}
    valid_experiments = []

    for exp in experiments:
        exp_id = exp['exp_id']
        if dataset == 'rees46':
            if exp['N_pilot'] == 0:
                continue
            res = all_results.get(exp_id, {})
            if any(m not in res or 'error' in res.get(m, {}) for m in BNP_METHODS):
                continue
        else:
            res = all_results[exp_id]

        D0, D1 = exp['D0'], exp['D1']
        N_pilot = exp['N_pilot']
        cum = np.array(exp['cumulative_users'])

        predictions[exp_id] = {}
        valid_experiments.append(exp)

        for eta in ETA_VALUES:
            M = math.ceil(eta * N_pilot)

            true_D = None
            for d in range(D0 + 1, len(cum)):
                if cum[d] >= M:
                    true_D = d
                    break

            pred = {'M': M, 'true_D': true_D, 'D0': D0, 'D1': D1, 'N_pilot': N_pilot}

            # NB-SSP
            try:
                nbp = NegBintSBSP()
                traj = nbp.mean_number_new_users(D0, D1, N_pilot, res['NB_SSP_regression']['params'])
                pred['NB_SSP_regression'] = next((D0 + ell + 1 for ell in range(D1) if N_pilot + traj[ell] >= M), None)
            except:
                pred['NB_SSP_regression'] = None

            # TG-SSP
            try:
                gd = GD()
                traj = gd.mean(D0, D1, N_pilot, res['TG_SSP']['params'])
                pred['TG_SSP'] = next((D0 + ell + 1 for ell in range(D1) if N_pilot + traj[ell] >= M), None)
            except:
                pred['TG_SSP'] = None

            # IBP
            try:
                ibp = IBP()
                traj = ibp.mean(D0, D1, res['IBP']['params'])
                pred['IBP'] = next((D0 + ell + 1 for ell in range(D1) if N_pilot + traj[ell] >= M), None)
            except:
                pred['IBP'] = None

            # Linear
            daily_rate = N_pilot / D0
            pred['linear'] = D0 + math.ceil((M - N_pilot) / daily_rate)

            # Jackknife
            try:
                sfs = np.array([N_pilot])
                cts = np.array(cum[:D0 + 1])
                preds_j = predict_jack(N=D0, M=D1, sfs=sfs, cts=cts, order=3)
                pred['jackknife'] = next((d for d in range(D0 + 1, D0 + D1 + 1) if preds_j[d] >= M), None)
            except:
                pred['jackknife'] = None

            predictions[exp_id][eta] = pred

    return predictions, valid_experiments, all_results


# ============================================================
# Top row: trajectory case studies with decision annotations
# ============================================================

def plot_trajectory_panel(ax, exp, predictions, all_results, eta=2.0):
    """Single trajectory panel with decision annotations."""
    exp_id = exp['exp_id']
    cum = np.array(exp['cumulative_users'])
    D0 = exp['D0']
    D1 = exp['D1']
    D = D0 + D1
    N_pilot = exp['N_pilot']
    M = math.ceil(eta * N_pilot)

    p = predictions[exp_id][eta]
    true_D = p['true_D']

    # Truth curve
    days = np.arange(D + 1)
    ax.plot(days, cum[:D+1], color='black', linewidth=2.5, zorder=5)

    # Pilot shading
    ax.axvspan(0, D0, alpha=0.06, color='gray')
    ax.axvline(x=D0, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)

    # Target line
    ax.axhline(y=M, color='#555555', linestyle='--', alpha=0.7, linewidth=1.2)

    # Methods to show: NB-SSP, Linear, and true
    show_methods = ['NB_SSP_regression', 'linear']

    # True hitting time — prominent marker
    if true_D is not None:
        ax.plot([true_D, true_D], [0, M], color='black', linestyle='-',
                alpha=0.25, linewidth=1)
        ax.scatter([true_D], [M], color='black', marker='D', s=50, zorder=7,
                   edgecolors='black', linewidths=0.8)

    # Predicted hitting times — vertical lines + markers + annotations
    y_offset_step = M * 0.08
    for i, method in enumerate(show_methods):
        pred_D = p[method]
        if pred_D is None:
            continue
        color = METHOD_COLORS[method]
        display = METHOD_DISPLAY[method]

        # Vertical line from x-axis to target
        ax.plot([pred_D, pred_D], [0, M], color=color, linestyle=':',
                alpha=0.5, linewidth=1.2)
        # Marker at target crossing
        ax.scatter([pred_D], [M], color=color, marker='o', s=40, zorder=6,
                   edgecolors='black', linewidths=0.5)

        # Annotation with day number
        y_annot = M * (0.55 - i * 0.18)
        ax.annotate(f'{display}: day {pred_D}',
                    xy=(pred_D, y_annot), fontsize=7.5, color=color,
                    fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=color, alpha=0.85, linewidth=0.8))

    # True hitting time annotation
    if true_D is not None:
        ax.annotate(f'True: day {true_D}',
                    xy=(true_D, M * 0.85), fontsize=7.5, color='black',
                    fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='black', alpha=0.85, linewidth=0.8))

        # Decision gap arrow between NB-SSP and Linear
        nb_D = p['NB_SSP_regression']
        lin_D = p['linear']
        if nb_D is not None and lin_D is not None and abs(nb_D - lin_D) >= 2:
            left = min(nb_D, lin_D)
            right = max(nb_D, lin_D)
            gap = abs(nb_D - lin_D)
            y_arrow = M * 0.15
            ax.annotate('', xy=(right, y_arrow), xytext=(left, y_arrow),
                        arrowprops=dict(arrowstyle='<->', color='#cc0000',
                                        lw=1.5, shrinkA=2, shrinkB=2))
            ax.text((left + right) / 2, y_arrow + M * 0.04,
                    f'{gap}d gap', fontsize=7, color='#cc0000',
                    ha='center', fontweight='bold')

    ax.set_xlabel('Day')
    ax.grid(True, alpha=0.15)
    ax.set_xlim(-0.5, D + 3)

    # Target label: place above the dashed line, inside the plot
    ylims = ax.get_ylim()
    offset = (ylims[1] - ylims[0]) * 0.025
    ax.text(D0 + 1, M + offset, f'Target ($\\eta={eta}$)',
            fontsize=9, va='bottom', ha='left', color='#555555', fontweight='bold')


# ============================================================
# Bottom row: planning error boxplots
# ============================================================

def plot_error_boxplots(ax, predictions, title, D0, D1, eta=2.0):
    """Boxplot of signed planning errors with decision framing."""

    data = []
    labels = []
    colors = []

    for method in ALL_METHODS:
        errors = []
        for exp_id, eta_dict in predictions.items():
            p = eta_dict[eta]
            true_D = p['true_D']
            pred_D = p[method]
            if true_D is None or pred_D is None:
                continue
            if method == 'linear' and D1 is not None:
                pred_D = min(pred_D, D0 + D1)
            errors.append(pred_D - true_D)

        data.append(errors)
        labels.append(METHOD_DISPLAY[method])
        colors.append(METHOD_COLORS.get(method, 'gray'))

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    widths=0.55, showfliers=True,
                    flierprops=dict(marker='.', markersize=3, alpha=0.3),
                    medianprops=dict(color='black', linewidth=1.8))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.4, linewidth=1.2)

    # Decision framing annotations
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]

    # "Too short" / "Too long" labels
    ax.text(0.02, 0.02, 'Plans too short\n(underpowered)',
            transform=ax.transAxes, fontsize=7, color='#cc0000',
            va='bottom', ha='left', fontstyle='italic', alpha=0.8)
    ax.text(0.02, 0.98, 'Plans too long\n(wasted resources)',
            transform=ax.transAxes, fontsize=7, color='#0066cc',
            va='top', ha='left', fontstyle='italic', alpha=0.8)

    ax.set_ylabel('Planning error (days)')
    ax.set_title(title, fontsize=10)
    ax.grid(axis='y', alpha=0.2)
    ax.tick_params(axis='x', rotation=15)


# ============================================================
# Main: combined figure
# ============================================================

if __name__ == '__main__':

    print("Loading ASOS data...")
    preds_asos, exps_asos, results_asos = load_and_predict('asos')

    print("Loading REES46 k=100...")
    preds_rees, exps_rees, _ = load_and_predict('rees46', k=100)

    # Find the ee6ff7_C experiment
    pick = None
    for exp in exps_asos:
        if exp['exp_id'] == 'ee6ff7_C':
            pick = exp
            break
    assert pick is not None, "ee6ff7_C not found"

    eid = pick['exp_id']
    pred = preds_asos[eid][2.0]
    print(f"Selected: {eid}, D1={pick['D1']}, N_pilot={pick['N_pilot']:,}, "
          f"true_D={pred['true_D']}, NB-SSP={pred['NB_SSP_regression']}, "
          f"Linear={pred['linear']}")

    # ---- Build the combined figure: 1 trajectory + 2 boxplots ----
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=2.5, label='Truth'),
        Line2D([0], [0], color='#555555', linestyle='--', linewidth=1.2,
               label='Target ($\\eta=2$)'),
        Line2D([0], [0], color='black', marker='D', markersize=6,
               linestyle='None', label='True $D_\\eta$'),
        Line2D([0], [0], color=METHOD_COLORS['NB_SSP_regression'], marker='o',
               markersize=6, linestyle='None', label='NB-SSP $\\hat{D}_\\eta$'),
        Line2D([0], [0], color=METHOD_COLORS['linear'], marker='o',
               markersize=6, linestyle='None', label='Linear $\\hat{D}_\\eta$'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2),
                              gridspec_kw={'width_ratios': [1.2, 1, 1]})

    # Panel (a): trajectory
    ax_traj = axes[0]
    plot_trajectory_panel(ax_traj, pick, preds_asos, results_asos, eta=2.0)
    ax_traj.set_ylabel('Cumulative distinct users')
    ax_traj.set_title(f'(a) ASOS case study\n($D_1={pick["D1"]}$, '
                      f'$N_{{pilot}}$={pick["N_pilot"]:,})', fontsize=9)
    ax_traj.legend(handles=legend_elements, fontsize=7, loc='upper left',
                   framealpha=0.9)

    # Panel (b): ASOS boxplot
    ax_asos = axes[1]
    plot_error_boxplots(ax_asos, preds_asos,
                        '(b) ASOS planning error ($\\eta=2$)',
                        D0=7, D1=None, eta=2.0)

    # Panel (c): REES46 boxplot
    ax_rees = axes[2]
    plot_error_boxplots(ax_rees, preds_rees,
                        '(c) REES46 planning error ($\\eta=2$)',
                        D0=7, D1=100, eta=2.0)

    fig.tight_layout()
    outpath = os.path.join(PLOT_DIR, 'fig_ht_decision.pdf')
    fig.savefig(outpath)
    plt.close()
    print(f"\nSaved: {outpath}")

    print("Done.")
