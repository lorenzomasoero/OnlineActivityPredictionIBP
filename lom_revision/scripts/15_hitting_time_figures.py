"""
15_hitting_time_figures.py
==========================
Three alternative figure options for the hitting-time MAE results.
Run from repo root:
    python3 SubmissionAOAS/lom_revision/scripts/15_hitting_time_figures.py

Produces:
    Option A: fig_ht_bar.pdf        — grouped bar chart of MAE by method x eta
    Option B: fig_ht_trajectories.pdf — trajectory case studies with hitting-time targets
    Option C: fig_ht_error_dist.pdf  — signed error distribution (boxplot) per method
"""

import numpy as np
import pandas as pd
import math
import sys
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
from plot_style import COLORS, MARKERS, LINESTYLES, LABELS, apply_style

warnings.filterwarnings("ignore")
apply_style()

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


# ============================================================
# Load data and compute predictions (reuse logic from compute_hitting_time_mae.py)
# ============================================================

def load_and_predict(dataset, k=None):
    """Load data and compute all hitting-time predictions."""
    results_dir = os.path.join(BASE, 'results')

    if dataset == 'rees46':
        data_dir = os.path.join(BASE, 'data', 'rees46_processed')
        experiments = np.load(os.path.join(data_dir, f'experiments_rolling_k{k}.npy'), allow_pickle=True)
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

    for exp in (experiments if dataset != 'rees46' else experiments):
        if dataset == 'rees46':
            exp_id = exp['exp_id']
            if exp['N_pilot'] == 0:
                continue
            res = all_results.get(exp_id, {})
            if any(m not in res or 'error' in res.get(m, {}) for m in BNP_METHODS):
                continue
        else:
            exp_id = exp['exp_id']
            res = all_results[exp_id]

        D0, D1 = exp['D0'], exp['D1']
        N_pilot = exp['N_pilot']
        cum = np.array(exp['cumulative_users'])

        predictions[exp_id] = {}
        valid_experiments.append(exp)

        for eta in ETA_VALUES:
            M = math.ceil(eta * N_pilot)

            # True hitting time
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
                pred['NB_SSP_regression_traj'] = traj
            except:
                pred['NB_SSP_regression'] = None

            # TG-SSP
            try:
                gd = GD()
                traj = gd.mean(D0, D1, N_pilot, res['TG_SSP']['params'])
                pred['TG_SSP'] = next((D0 + ell + 1 for ell in range(D1) if N_pilot + traj[ell] >= M), None)
                pred['TG_SSP_traj'] = traj
            except:
                pred['TG_SSP'] = None

            # IBP
            try:
                ibp = IBP()
                traj = ibp.mean(D0, D1, res['IBP']['params'])
                pred['IBP'] = next((D0 + ell + 1 for ell in range(D1) if N_pilot + traj[ell] >= M), None)
                pred['IBP_traj'] = traj
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
                pred['jackknife_traj'] = preds_j
            except:
                pred['jackknife'] = None

            predictions[exp_id][eta] = pred

    return predictions, valid_experiments


# ============================================================
# OPTION A: Grouped bar chart of MAE
# ============================================================

def plot_option_a(predictions_asos, predictions_rees, output_path):
    """Grouped bar chart: MAE by method, grouped by eta. Two panels (ASOS, REES46)."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, (preds, title, D0_D1) in zip(axes, [
        (predictions_asos, 'ASOS (144 arms)', (7, None)),
        (predictions_rees, 'REES46 $k=100$ (107 windows)', (7, 100)),
    ]):
        D0 = D0_D1[0]
        D1 = D0_D1[1]

        method_order = ALL_METHODS
        n_methods = len(method_order)
        n_eta = len(ETA_VALUES)
        bar_width = 0.15
        x = np.arange(n_eta)

        for i, method in enumerate(method_order):
            maes = []
            for eta in ETA_VALUES:
                errors = []
                for exp_id, eta_dict in preds.items():
                    p = eta_dict[eta]
                    true_D = p['true_D']
                    pred_D = p[method]
                    if true_D is None or pred_D is None:
                        continue
                    if method == 'linear' and D1 is not None:
                        pred_D = min(pred_D, D0 + D1)
                    errors.append(abs(pred_D - true_D))
                maes.append(np.mean(errors) if errors else 0)

            offset = (i - n_methods / 2 + 0.5) * bar_width
            ax.bar(x + offset, maes, bar_width,
                   color=METHOD_COLORS.get(method, 'gray'),
                   label=METHOD_DISPLAY[method], edgecolor='white', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f'$\\eta={e}$' for e in ETA_VALUES])
        ax.set_ylabel('MAE (days)')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)

    axes[0].legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close()
    print(f"Option A saved: {output_path}")


# ============================================================
# OPTION B: Trajectory case studies with hitting-time targets
# ============================================================

def plot_option_b(predictions, experiments, all_results, dataset_name, output_path, exp_picks=None):
    """
    For 3 selected experiments, show cumulative user curve + method predictions
    with horizontal lines at M_eta targets and vertical markers at predicted hitting times.
    """
    if exp_picks is None:
        # Pick 3 experiments spread across the range
        valid = [e for e in experiments if e['exp_id'] in predictions]
        valid.sort(key=lambda e: e['N_pilot'])
        n = len(valid)
        exp_picks = [valid[n // 10], valid[n // 2], valid[int(n * 0.85)]]

    fig, axes = plt.subplots(1, len(exp_picks), figsize=(5 * len(exp_picks), 4.5))
    if len(exp_picks) == 1:
        axes = [axes]

    eta_linestyles = {1.5: ':', 2.0: '--', 3.0: '-.'}
    eta_colors_gray = {1.5: '#999999', 2.0: '#666666', 3.0: '#333333'}

    for ax, exp in zip(axes, exp_picks):
        exp_id = exp['exp_id']
        cum = np.array(exp['cumulative_users'])
        D0 = exp['D0']
        D1 = exp['D1']
        D = D0 + D1
        N_pilot = exp['N_pilot']

        # Truth curve
        days = np.arange(D + 1)
        ax.plot(days, cum[:D+1], color='black', linewidth=2, label='Truth', zorder=5)

        # Pilot shading
        ax.axvspan(0, D0, alpha=0.08, color='gray')
        ax.axvline(x=D0, color='gray', linestyle=':', alpha=0.5)

        # Horizontal target lines
        for eta in ETA_VALUES:
            M = math.ceil(eta * N_pilot)
            if M <= cum[D] * 1.3:  # only draw if target is in a reasonable range
                ax.axhline(y=M, color=eta_colors_gray[eta], linestyle=eta_linestyles[eta],
                           alpha=0.6, linewidth=1)
                ax.text(D + 0.3, M, f'$\\eta={eta}$', fontsize=7, va='center',
                        color=eta_colors_gray[eta])

        # For eta=2.0 (the middle target), show predicted hitting times as vertical markers
        eta_show = 2.0
        M_show = math.ceil(eta_show * N_pilot)
        p = predictions[exp_id][eta_show]
        true_D = p['true_D']

        if true_D is not None:
            ax.plot([true_D, true_D], [0, M_show], color='black', linestyle='-',
                    alpha=0.3, linewidth=1)
            ax.scatter([true_D], [M_show], color='black', marker='*', s=80, zorder=6)

        for method in ['NB_SSP_regression', 'TG_SSP', 'linear']:
            pred_D = p[method]
            if pred_D is not None and pred_D <= D + 5:
                ax.scatter([pred_D], [M_show], color=METHOD_COLORS.get(method, 'gray'),
                           marker=MARKERS.get(method, 'o'), s=50, zorder=5,
                           edgecolors='black', linewidths=0.5)
                # Small vertical tick
                ax.plot([pred_D, pred_D], [M_show * 0.95, M_show * 1.05],
                        color=METHOD_COLORS.get(method, 'gray'), linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Day')
        if ax == axes[0]:
            ax.set_ylabel('Cumulative distinct users')
        ax.set_title(f'Exp {exp_id}\n($N_{{pilot}}$={N_pilot:,})', fontsize=10)
        ax.grid(True, alpha=0.2)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=2, label='Truth'),
        Line2D([0], [0], color='black', marker='*', markersize=8, linestyle='None', label=f'True $D_{{\\eta={eta_show}}}$'),
    ]
    for method in ['NB_SSP_regression', 'TG_SSP', 'linear']:
        legend_elements.append(
            Line2D([0], [0], color=METHOD_COLORS[method], marker=MARKERS.get(method, 'o'),
                   markersize=6, linestyle='None', label=f'{METHOD_DISPLAY[method]} $\\hat{{D}}_{{\\eta}}$')
        )
    axes[-1].legend(handles=legend_elements, fontsize=7, loc='lower right')

    fig.suptitle(f'{dataset_name}: Duration-to-target predictions ($\\eta=2$)',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close()
    print(f"Option B saved: {output_path}")


# ============================================================
# OPTION C: Signed error distribution (boxplot)
# ============================================================

def plot_option_c(predictions_asos, predictions_rees, output_path):
    """Boxplot of signed errors (D_hat - D_true) per method, one panel per dataset."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, (preds, title, D0_val, D1_val) in zip(axes, [
        (predictions_asos, 'ASOS (144 arms)', 7, None),
        (predictions_rees, 'REES46 $k=100$ (107 windows)', 7, 100),
    ]):
        # Collect signed errors for eta=2.0 (the middle target)
        eta = 2.0
        data_for_box = []
        labels_for_box = []
        colors_for_box = []

        for method in ALL_METHODS:
            errors = []
            for exp_id, eta_dict in preds.items():
                p = eta_dict[eta]
                true_D = p['true_D']
                pred_D = p[method]
                if true_D is None or pred_D is None:
                    continue
                if method == 'linear' and D1_val is not None:
                    pred_D = min(pred_D, D0_val + D1_val)
                errors.append(pred_D - true_D)  # signed error

            data_for_box.append(errors)
            labels_for_box.append(METHOD_DISPLAY[method])
            colors_for_box.append(METHOD_COLORS.get(method, 'gray'))

        bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                        widths=0.6, showfliers=True,
                        flierprops=dict(marker='.', markersize=3, alpha=0.4))

        for patch, color in zip(bp['boxes'], colors_for_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax.set_ylabel('Signed error $\\hat{D}_\\eta - D_\\eta$ (days)')
        ax.set_title(f'{title}, $\\eta={eta}$')
        ax.grid(axis='y', alpha=0.3)

        # Rotate labels
        ax.tick_params(axis='x', rotation=15)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close()
    print(f"Option C saved: {output_path}")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':

    print("Loading ASOS data...")
    preds_asos, exps_asos = load_and_predict('asos')
    asos_results = np.load(os.path.join(BASE, 'results', 'asos_all_results.npy'), allow_pickle=True).item()

    print("Loading REES46 k=100 data...")
    preds_rees, exps_rees = load_and_predict('rees46', k=100)

    print("\nGenerating Option A (bar chart)...")
    plot_option_a(preds_asos, preds_rees,
                  os.path.join(PLOT_DIR, 'fig_ht_bar.pdf'))

    print("Generating Option B (trajectory case studies) — ASOS...")
    plot_option_b(preds_asos, exps_asos, asos_results, 'ASOS',
                  os.path.join(PLOT_DIR, 'fig_ht_trajectories_asos.pdf'))

    print("Generating Option B (trajectory case studies) — REES46...")
    plot_option_b(preds_rees, exps_rees, None, 'REES46 $k=100$',
                  os.path.join(PLOT_DIR, 'fig_ht_trajectories_rees46.pdf'))

    print("Generating Option C (error distribution)...")
    plot_option_c(preds_asos, preds_rees,
                  os.path.join(PLOT_DIR, 'fig_ht_error_dist.pdf'))

    print("\nAll done. Check plots/ for:")
    print("  Option A: fig_ht_bar.pdf")
    print("  Option B: fig_ht_trajectories_asos.pdf, fig_ht_trajectories_rees46.pdf")
    print("  Option C: fig_ht_error_dist.pdf")
