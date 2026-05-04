"""
hitting_time_mae.py
===================
Hitting-time MAE experiment on REES46 rolling-window data.

For each experiment and each participation target eta in {1.5, 2.0, 3.0},
computes the true and predicted hitting time D_eta (first day cumulative users
reach ceil(eta * N_pilot)) for five methods:
    - NB-SSP (regression)
    - TG-SSP
    - IBP
    - Linear extrapolation
    - Jackknife (order 3)

Outputs:
    - CSV summary: results/hitting_time/hitting_time_mae_k{K}.csv
    - LaTeX table: printed to stdout
    - Scatter figure: output/fig_hitting_time_mae_k{K}.pdf

Usage:
    python experiments/evaluation/hitting_time_mae.py              # default k=21
    python experiments/evaluation/hitting_time_mae.py --k 50       # single k
    python experiments/evaluation/hitting_time_mae.py --k 21 50 100  # multiple k values
"""

import numpy as np
import pandas as pd
import math
import os
import sys
import warnings
import time
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from models.tg_ssp import GD
from models.ibp import IBP
from models.nb_ssp import NegBintSBSP
from models.jackknife import predict_jack
from plotting.style import COLORS, MARKERS, LABELS, apply_style

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(REPO_ROOT, 'data', 'preprocessed', 'rees46')
RESULTS_DIR = os.path.join(REPO_ROOT, 'results')
PLOTS_DIR = os.path.join(REPO_ROOT, 'output')

ETA_VALUES = [1.5, 2.0, 3.0]
METHODS = ['NB_SSP_regression', 'TG_SSP', 'IBP', 'linear', 'jackknife']
BNP_METHODS = ['NB_SSP_regression', 'TG_SSP', 'IBP']

METHOD_DISPLAY = {
    'NB_SSP_regression': 'NB-SSP (reg)',
    'TG_SSP': 'TG-SSP',
    'IBP': 'IBP',
    'linear': 'Linear extrap.',
    'jackknife': 'Jackknife (J3)',
}


# ===================================================================
# 1. Data loading
# ===================================================================

def load_data(data_dir, results_dir, k=21):
    """
    Load experiment metadata and fitted results for REES46 rolling-window data.

    Parameters
    ----------
    data_dir : str
        Path to preprocessed rees46 directory.
    results_dir : str
        Path to results/ directory.
    k : int
        Horizon (default 21).

    Returns
    -------
    experiments : list[dict]
        Filtered list of experiment dicts.
    results : dict
        Dict keyed by exp_id with fitted parameter sub-dicts.
    n_skipped : int
        Number of experiments skipped.
    """
    exp_path = os.path.join(data_dir, f"experiments_rolling_k{k}.npy")
    res_path = os.path.join(results_dir, 'rees46', f"rees46_rolling_k{k}_results.npy")

    all_experiments = np.load(exp_path, allow_pickle=True)
    all_results = np.load(res_path, allow_pickle=True).item()

    experiments = []
    n_skipped = 0

    for exp in all_experiments:
        exp_id = exp['exp_id']

        if exp['N_pilot'] == 0:
            n_skipped += 1
            continue

        res = all_results.get(exp_id, {})
        skip = False
        for method in BNP_METHODS:
            if method not in res or 'error' in res.get(method, {}):
                skip = True
                break
        if skip:
            n_skipped += 1
            continue

        experiments.append(exp)

    return experiments, all_results, n_skipped


def load_data_unified(dataset, results_dir):
    """
    Load experiment data for UCI or ASOS (single-file results, no k parameter).

    Parameters
    ----------
    dataset : str
        'uci' or 'asos'.
    results_dir : str
        Path to results/ directory.

    Returns
    -------
    experiments : list[dict]
        Filtered experiment dicts.
    results : dict
        Dict keyed by exp_id.
    n_skipped : int
    """
    res_path = os.path.join(results_dir, dataset, f"{dataset}_all_results.npy")
    all_results = np.load(res_path, allow_pickle=True).item()

    experiments = []
    n_skipped = 0

    for exp_id, res in all_results.items():
        N_pilot = res.get('N_pilot', 0)
        if N_pilot == 0:
            n_skipped += 1
            continue

        skip = False
        for method in BNP_METHODS:
            if method not in res or 'error' in res.get(method, {}):
                skip = True
                break
        if skip:
            n_skipped += 1
            continue

        cum = np.array(res['cumulative_users'])

        exp = {
            'exp_id': exp_id,
            'D0': res['D0'],
            'D1': res['D1'],
            'N_pilot': N_pilot,
            'U_true': res.get('U_true', res.get('N_total', cum[-1]) - N_pilot),
            'cumulative_users': cum,
        }
        experiments.append(exp)

    return experiments, all_results, n_skipped


# ===================================================================
# 2. True hitting time
# ===================================================================

def compute_true_hitting_time(cumulative_users, D0, eta):
    """
    Compute the true hitting time D_eta.

    Parameters
    ----------
    cumulative_users : array-like
        Cumulative user counts, length D0+D1+1.
    D0 : int
        Number of pilot days.
    eta : float
        Participation multiplier (> 1).

    Returns
    -------
    int or None
        First day d > D0 where cumulative_users[d] >= ceil(eta * N_pilot),
        or None if target not reached within the observation window.
    """
    N_pilot = cumulative_users[D0]
    M = math.ceil(eta * N_pilot)
    for d in range(D0 + 1, len(cumulative_users)):
        if cumulative_users[d] >= M:
            return d
    return None


# ===================================================================
# 3. BNP predicted hitting times
# ===================================================================

def predict_hitting_time_nbssp(params, D0, D1, N_pilot, M):
    """Predicted hitting time using NB-SSP mean trajectory."""
    try:
        nbp = NegBintSBSP()
        mean_traj = nbp.mean_number_new_users(D0, D1, N_pilot, params)
        for ell in range(D1):
            if N_pilot + mean_traj[ell] >= M:
                return D0 + ell + 1
        return None
    except Exception as e:
        warnings.warn(f"NB-SSP prediction failed: {e}")
        return None


def predict_hitting_time_tgssp(params, D0, D1, N_pilot, M):
    """Predicted hitting time using TG-SSP (GD) mean trajectory."""
    try:
        gd = GD()
        mean_traj = gd.mean(D0, D1, N_pilot, params)
        for ell in range(D1):
            if N_pilot + mean_traj[ell] >= M:
                return D0 + ell + 1
        return None
    except Exception as e:
        warnings.warn(f"TG-SSP prediction failed: {e}")
        return None


def predict_hitting_time_ibp(params, D0, D1, N_pilot, M):
    """Predicted hitting time using IBP mean trajectory."""
    try:
        ibp = IBP()
        mean_traj = ibp.mean(D0, D1, params)
        for ell in range(D1):
            if N_pilot + mean_traj[ell] >= M:
                return D0 + ell + 1
        return None
    except Exception as e:
        warnings.warn(f"IBP prediction failed: {e}")
        return None


# ===================================================================
# 4. Baseline predicted hitting times
# ===================================================================

def predict_hitting_time_linear(D0, N_pilot, M):
    """Predicted hitting time using linear extrapolation."""
    daily_rate = N_pilot / D0
    additional_days = math.ceil((M - N_pilot) / daily_rate)
    return D0 + additional_days


def predict_hitting_time_jackknife(cumulative_users, D0, D1, M, order=3):
    """Predicted hitting time using Jackknife estimator."""
    try:
        N_pilot = cumulative_users[D0]
        sfs = np.array([N_pilot])
        cts = np.array(cumulative_users[:D0 + 1])
        preds = predict_jack(N=D0, M=D1, sfs=sfs, cts=cts, order=order)
        for d in range(D0 + 1, D0 + D1 + 1):
            if preds[d] >= M:
                return d
        return None
    except Exception as e:
        warnings.warn(f"Jackknife prediction failed: {e}")
        return None


# ===================================================================
# 6. MAE aggregation
# ===================================================================

def compute_mae_table(predictions, eta_values, methods, D0, D1):
    """
    Compute MAE summary table across all experiments.

    Returns
    -------
    pd.DataFrame
        Rows: (method, eta), columns: mae, median_ae, n_valid, etc.
    """
    rows = []
    for method in methods:
        for eta in eta_values:
            errors = []
            n_censored_true = 0
            n_censored_pred = 0

            for exp_id, eta_dict in predictions.items():
                p = eta_dict[eta]
                true_D = p['true_D']
                pred_D = p[method]

                if true_D is None:
                    n_censored_true += 1
                    continue
                if pred_D is None:
                    n_censored_pred += 1
                    continue

                if method == 'linear':
                    pred_D = min(pred_D, D0 + D1)

                errors.append(abs(pred_D - true_D))

            mae = np.mean(errors) if errors else np.nan
            median_ae = np.median(errors) if errors else np.nan
            rows.append({
                'method': METHOD_DISPLAY.get(method, method),
                'eta': eta,
                'mae': mae,
                'median_ae': median_ae,
                'n_valid': len(errors),
                'n_censored_true': n_censored_true,
                'n_censored_pred': n_censored_pred,
            })

    return pd.DataFrame(rows)


# ===================================================================
# 7. Output — LaTeX table
# ===================================================================

def print_latex_table(mae_df, eta_values):
    """Print a LaTeX-formatted table of MAE values to stdout."""
    method_order = [METHOD_DISPLAY[m] for m in METHODS]
    pivot = mae_df.pivot(index='method', columns='eta', values='mae')
    pivot = pivot.reindex(method_order)

    best = {}
    for eta in eta_values:
        col = pivot[eta].dropna()
        if len(col) > 0:
            best[eta] = col.min()
        else:
            best[eta] = None

    print("\n\\begin{tabular}{l" + "c" * len(eta_values) + "}")
    print("\\toprule")
    header = "Method & " + " & ".join([f"$\\eta={e}$" for e in eta_values]) + " \\\\"
    print(header)
    print("\\midrule")

    for method_name in method_order:
        row_parts = [method_name]
        for eta in eta_values:
            val = pivot.loc[method_name, eta]
            if np.isnan(val):
                row_parts.append("--")
            elif best[eta] is not None and abs(val - best[eta]) < 1e-6:
                row_parts.append(f"\\textbf{{{val:.1f}}}")
            else:
                row_parts.append(f"{val:.1f}")
        print(" & ".join(row_parts) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}\n")


# ===================================================================
# 8. Output — scatter figure
# ===================================================================

def plot_scatter(predictions, output_path, eta_values=None):
    """Produce a 1x3 scatter figure: predicted vs true D_eta, one panel per eta."""
    if eta_values is None:
        eta_values = ETA_VALUES

    apply_style()

    colors = dict(COLORS)
    markers = dict(MARKERS)
    labels = dict(LABELS)

    colors['linear'] = '#ff7f00'
    colors['jackknife'] = '#a65628'
    markers['linear'] = 'P'
    markers['jackknife'] = 'X'
    labels['linear'] = 'Linear extrap.'
    labels['jackknife'] = 'Jackknife (J3)'

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    for ax_idx, eta in enumerate(eta_values):
        ax = axes[ax_idx]
        all_vals = []

        for method in METHODS:
            true_vals = []
            pred_vals = []
            for exp_id, eta_dict in predictions.items():
                p = eta_dict[eta]
                true_D = p['true_D']
                pred_D = p[method]
                if true_D is not None and pred_D is not None:
                    true_vals.append(true_D)
                    pred_vals.append(pred_D)

            if true_vals:
                ax.scatter(true_vals, pred_vals,
                           c=colors.get(method, 'gray'),
                           marker=markers.get(method, 'o'),
                           label=labels.get(method, method),
                           alpha=0.6, s=30, edgecolors='none')
                all_vals.extend(true_vals)
                all_vals.extend(pred_vals)

        if all_vals:
            lo = min(all_vals) - 1
            hi = max(all_vals) + 1
            ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, linewidth=1, label='$y=x$')
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

        ax.set_xlabel(r"True $D_\eta$ (days)")
        if ax_idx == 0:
            ax.set_ylabel(r"Predicted $\hat{D}_\eta$ (days)")
        ax.set_title(f"$\\eta = {eta}$")

        if ax_idx == 0:
            ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved figure: {output_path}")


# ===================================================================
# Main
# ===================================================================

def run_for_k(k, dataset='rees46'):
    """Run the full hitting-time MAE pipeline for a single horizon k."""
    t0 = time.time()

    if dataset == 'rees46':
        label = f"REES46 k={k}"
    else:
        label = dataset.upper()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Load data
    print("Loading data...")
    if dataset == 'rees46':
        experiments, results, n_skipped = load_data(DATA_DIR, RESULTS_DIR, k=k)
    else:
        experiments, results, n_skipped = load_data_unified(dataset, RESULTS_DIR)

    n_exp = len(experiments)
    print(f"Loaded {n_exp} valid experiments ({n_skipped} skipped)")

    # Prediction loop
    print("\nComputing hitting times...")
    predictions = {}

    for i, exp in enumerate(experiments):
        exp_id = exp['exp_id']
        D0 = exp['D0']
        D1 = exp['D1']
        N_pilot = exp['N_pilot']
        cumulative_users = np.array(exp['cumulative_users'])
        res = results[exp_id]

        predictions[exp_id] = {}

        for eta in ETA_VALUES:
            M = math.ceil(eta * N_pilot)
            true_D = compute_true_hitting_time(cumulative_users, D0, eta)

            pred = {'M': M, 'true_D': true_D}

            pred['NB_SSP_regression'] = predict_hitting_time_nbssp(
                res['NB_SSP_regression']['params'], D0, D1, N_pilot, M)
            pred['TG_SSP'] = predict_hitting_time_tgssp(
                res['TG_SSP']['params'], D0, D1, N_pilot, M)
            pred['IBP'] = predict_hitting_time_ibp(
                res['IBP']['params'], D0, D1, N_pilot, M)

            pred['linear'] = predict_hitting_time_linear(D0, N_pilot, M)
            pred['jackknife'] = predict_hitting_time_jackknife(
                cumulative_users, D0, D1, M)

            predictions[exp_id][eta] = pred

        if i % 20 == 0:
            elapsed = time.time() - t0
            print(f"  Experiment {i}/{n_exp} (exp_id={exp_id}) [{elapsed:.1f}s elapsed]")

    elapsed = time.time() - t0
    print(f"Predictions complete ({elapsed:.1f}s)")

    # MAE aggregation
    D0 = experiments[0]['D0']
    D1 = experiments[0]['D1']
    mae_df = compute_mae_table(predictions, ETA_VALUES, METHODS, D0, D1)

    print(f"\n=== Hitting-Time MAE Summary ({label}) ===\n")
    print(mae_df.to_string(index=False))

    # Save CSV
    if dataset == 'rees46':
        csv_name = f"hitting_time_mae_k{k}.csv"
        fig_name = f"fig_hitting_time_mae_k{k}.pdf"
    else:
        csv_name = f"hitting_time_mae_{dataset}.csv"
        fig_name = f"fig_hitting_time_mae_{dataset}.pdf"

    ht_results_dir = os.path.join(RESULTS_DIR, 'hitting_time')
    os.makedirs(ht_results_dir, exist_ok=True)
    csv_path = os.path.join(ht_results_dir, csv_name)
    mae_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # LaTeX table
    print(f"\n=== LaTeX Table ({label}) ===")
    print_latex_table(mae_df, ETA_VALUES)

    # Scatter figure
    fig_path = os.path.join(PLOTS_DIR, fig_name)
    plot_scatter(predictions, fig_path, ETA_VALUES)

    total_time = time.time() - t0
    print(f"\n{label} done in {total_time:.1f}s")

    return mae_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hitting-time MAE experiment')
    parser.add_argument('--k', type=int, nargs='*', default=[21],
                        help='Horizon values for REES46 (default: 21)')
    parser.add_argument('--dataset', type=str, nargs='*', default=['rees46'],
                        choices=['rees46', 'uci', 'asos'],
                        help='Datasets to run (default: rees46)')
    args = parser.parse_args()

    all_mae = {}
    for dataset in args.dataset:
        if dataset == 'rees46':
            for k in args.k:
                key = f"rees46_k{k}"
                all_mae[key] = run_for_k(k, dataset='rees46')
        else:
            all_mae[dataset] = run_for_k(k=None, dataset=dataset)

    if len(all_mae) > 1:
        print(f"\n{'='*60}")
        print(f"  COMBINED COMPARISON")
        print(f"{'='*60}")
        for key, df in all_mae.items():
            print(f"\n--- {key} ---")
            print(df.to_string(index=False))

    print("\nAll done.")
