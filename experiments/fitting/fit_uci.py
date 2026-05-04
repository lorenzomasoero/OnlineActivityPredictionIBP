"""
fit_uci.py
==========
Fits Be-SSP, TG-SSP, NB-SSP and competing methods on all 13 UCI Online Retail
experiment windows. Computes:
  - Point predictions for U_{D0}^{(D1)} (new users)
  - 95% credible intervals for U
  - Point predictions for T_{D0}^{(D1)} (total triggers, NB-SSP only)
  - Accuracy metrics v_{D0}^{(D1)}

Output:
    results/uci/uci_all_results.npy

Usage:
    python experiments/fitting/fit_uci.py
"""

import numpy as np
import os
import sys
import time
import warnings
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from models.common import *
from models.tg_ssp import GD
from models.ibp import IBP
from models.nb_ssp import NegBintSBSP
from models.beta_binomial import *

try:
    from models.beta_geometric import predict_beta_geom
    HAS_BETA_GEOM = True
except ImportError:
    HAS_BETA_GEOM = False
    print("Warning: beta_geometric not available (cmdstanpy/arviz dependency). Skipping BG model.")

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
DATA_DIR = os.path.join(REPO_ROOT, 'data', 'preprocessed', 'uci')
RESULTS_DIR = os.path.join(REPO_ROOT, 'results', 'uci')
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_ITS = 5        # number of optimization restarts
WIDTH = 0.95       # credible interval width
D0 = 7             # pilot days

# ============================================================
# Load experiment metadata
# ============================================================
experiments = np.load(os.path.join(DATA_DIR, "experiments_metadata.npy"), allow_pickle=True)
print(f"Loaded {len(experiments)} experiments")

# ============================================================
# Fitting loop
# ============================================================
all_results = {}

for exp in tqdm(experiments, desc="Fitting experiments"):
    exp_id = exp['exp_id']
    D1 = exp['D1']
    N_pilot = exp['N_pilot']
    U_true = exp['U_true']
    T_true = exp['T_true']
    cumulative_users = np.array(exp['cumulative_users'])

    # Load matrices
    matrix_full = np.load(os.path.join(DATA_DIR, f"matrix_exp_{exp_id}.npy"))
    matrix_pilot = np.load(os.path.join(DATA_DIR, f"matrix_pilot_exp_{exp_id}.npy"))

    D_total = matrix_full.shape[0]  # 28

    results = {
        'exp_id': exp_id,
        'D0': D0,
        'D1': D1,
        'N_pilot': N_pilot,
        'U_true': U_true,
        'T_true': T_true,
        'cumulative_users': cumulative_users,
    }

    # ----------------------------------------------------------
    # Compute sufficient statistics for the pilot
    # ----------------------------------------------------------
    pilot_binary = (matrix_pilot > 0).astype(int)
    days_active_per_user = pilot_binary.sum(axis=0)

    active_in_pilot = days_active_per_user > 0
    days_active_pilot_users = days_active_per_user[active_in_pilot]

    sfs = np.bincount(days_active_pilot_users)[1:]

    retrigger_counts_pilot = matrix_pilot[:, active_in_pilot].sum(axis=0)

    counts_short = cumulative_users[:D0+1]
    counts_long = cumulative_users

    # ----------------------------------------------------------
    # 1. TG-SSP (regression on first-trigger counts)
    # ----------------------------------------------------------
    try:
        st = time.time()
        gd = GD()
        gd_params = gd.regression(counts_short, NUM_ITS, 2, False)
        gd_pred_news = gd.mean(D0, D1, N_pilot, gd_params)
        gd_U_hat = gd_pred_news[-1]
        gd_lo, gd_hi = gd.credible_interval(D0, D1, N_pilot, gd_params, WIDTH)
        gd_U_lo, gd_U_hi = gd_lo[-1], gd_hi[-1]

        results['TG_SSP'] = {
            'params': gd_params.tolist(),
            'U_hat': float(gd_U_hat),
            'U_ci': (float(gd_U_lo), float(gd_U_hi)),
            'U_trajectory': gd_pred_news.tolist(),
            'time': time.time() - st,
        }
    except Exception as e:
        results['TG_SSP'] = {'error': str(e)}
        print(f"  Exp {exp_id} TG-SSP failed: {e}")

    # ----------------------------------------------------------
    # 2. IBP (3-parameter Indian Buffet Process)
    # ----------------------------------------------------------
    try:
        st = time.time()
        ibp = IBP()
        ibp_params = ibp.regression(counts_short, NUM_ITS, 2, False)
        ibp_pred_news = ibp.mean(D0, D1, ibp_params)
        ibp_U_hat = ibp_pred_news[-1]

        from scipy.stats import poisson as poisson_dist
        ibp_U_lo, ibp_U_hi = poisson_dist.interval(confidence=WIDTH, mu=ibp_pred_news[-1])

        results['IBP'] = {
            'params': ibp_params.tolist(),
            'U_hat': float(ibp_U_hat),
            'U_ci': (float(ibp_U_lo), float(ibp_U_hi)),
            'U_trajectory': ibp_pred_news.tolist(),
            'time': time.time() - st,
        }
    except Exception as e:
        results['IBP'] = {'error': str(e)}
        print(f"  Exp {exp_id} IBP failed: {e}")

    # ----------------------------------------------------------
    # 3. NB-SSP (full count matrix, maximum marginal likelihood)
    # ----------------------------------------------------------
    try:
        st = time.time()
        nbp = NegBintSBSP()
        nbp_params = nbp.fit_log_like(matrix_pilot, num_its=NUM_ITS)
        beta_, sigma_, tilting_, r_ = nbp_params

        nbp_pred_news = nbp.mean_number_new_users(D0, D1, N_pilot, nbp_params)
        nbp_U_hat = nbp_pred_news[-1]

        nbp_ci = nbp.ci_number_new_users(D0, D1, N_pilot, nbp_params, width=WIDTH)
        nbp_U_lo, nbp_U_hi = nbp_ci[-1]

        nbp_T_new = nbp.total_retriggers_new_users(D0, D1, N_pilot, nbp_params, N_MC=100, threshold=10)
        nbp_T_old = nbp.total_retriggers_old_users(D0, D1, retrigger_counts_pilot, nbp_params, N_MC=100)
        nbp_T_old_mean = nbp_T_old.mean(axis=0) if len(nbp_T_old) > 0 else np.zeros(D1)
        nbp_T_hat = float(nbp_T_new[-1] + nbp_T_old_mean[-1])

        results['NB_SSP'] = {
            'params': nbp_params.tolist(),
            'U_hat': float(nbp_U_hat),
            'U_ci': (float(nbp_U_lo), float(nbp_U_hi)),
            'U_trajectory': nbp_pred_news.tolist(),
            'T_hat': nbp_T_hat,
            'T_new_trajectory': nbp_T_new.tolist(),
            'time': time.time() - st,
        }
    except Exception as e:
        results['NB_SSP'] = {'error': str(e)}
        print(f"  Exp {exp_id} NB-SSP failed: {e}")

    # ----------------------------------------------------------
    # 4. NB-SSP with regression (r=1, i.e. Be-SSP equivalent)
    # ----------------------------------------------------------
    try:
        st = time.time()
        nbp_reg = NegBintSBSP()
        nbp_reg_params = nbp_reg.fit_regression(
            D0=0, N0=0,
            observed_counts=counts_short[1:D0+1],
            num_its=NUM_ITS
        )

        nbp_reg_pred = nbp_reg.mean_number_new_users(D0, D1, N_pilot, nbp_reg_params)
        nbp_reg_U_hat = nbp_reg_pred[-1]

        nbp_reg_ci = nbp_reg.ci_number_new_users(D0, D1, N_pilot, nbp_reg_params, width=WIDTH)
        nbp_reg_U_lo, nbp_reg_U_hi = nbp_reg_ci[-1]

        results['NB_SSP_regression'] = {
            'params': nbp_reg_params.tolist(),
            'U_hat': float(nbp_reg_U_hat),
            'U_ci': (float(nbp_reg_U_lo), float(nbp_reg_U_hi)),
            'U_trajectory': nbp_reg_pred.tolist(),
            'time': time.time() - st,
        }
    except Exception as e:
        results['NB_SSP_regression'] = {'error': str(e)}
        print(f"  Exp {exp_id} NB-SSP-reg failed: {e}")

    # ----------------------------------------------------------
    # 5. Beta-Geometric (Richardson et al. 2022)
    # ----------------------------------------------------------
    if HAS_BETA_GEOM:
        try:
            st = time.time()
            first_counts = np.array([counts_short[i+1] - counts_short[i] for i in range(len(counts_short)-1)], dtype=int)
            bg_pred = predict_beta_geom(first_counts, D0, D1)
            bg_U_hat = float(bg_pred[-1] - N_pilot) if bg_pred is not None else None

            results['BG'] = {
                'U_hat': bg_U_hat,
                'trajectory': bg_pred.tolist() if bg_pred is not None else None,
                'time': time.time() - st,
            }
        except Exception as e:
            results['BG'] = {'error': str(e)}
            print(f"  Exp {exp_id} BG failed: {e}")
    else:
        results['BG'] = {'error': 'beta_geom not available'}

    # ----------------------------------------------------------
    # Compute accuracy metrics
    # ----------------------------------------------------------
    for method in ['TG_SSP', 'IBP', 'NB_SSP', 'NB_SSP_regression', 'BG']:
        if method in results and 'U_hat' in results[method] and results[method]['U_hat'] is not None:
            U_hat = results[method]['U_hat']
            v = 1 - min(abs(U_true - U_hat) / max(U_true, 1), 1)
            results[method]['accuracy_v'] = float(v)

            if 'U_ci' in results[method]:
                lo, hi = results[method]['U_ci']
                results[method]['ci_covers'] = bool(lo <= U_true <= hi)

    # Total trigger accuracy for NB-SSP
    if 'NB_SSP' in results and 'T_hat' in results['NB_SSP']:
        T_hat = results['NB_SSP']['T_hat']
        v_T = 1 - min(abs(T_true - T_hat) / max(T_true, 1), 1)
        results['NB_SSP']['accuracy_v_T'] = float(v_T)

    all_results[exp_id] = results

# ============================================================
# Save results
# ============================================================
np.save(os.path.join(RESULTS_DIR, "uci_all_results.npy"), all_results)

# ============================================================
# Print summary
# ============================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

for method in ['TG_SSP', 'IBP', 'NB_SSP', 'NB_SSP_regression', 'BG']:
    accuracies = []
    coverages = []
    for exp_id, res in all_results.items():
        if method in res and 'accuracy_v' in res[method]:
            accuracies.append(res[method]['accuracy_v'])
        if method in res and 'ci_covers' in res[method]:
            coverages.append(res[method]['ci_covers'])

    if accuracies:
        acc_str = f"median v={np.median(accuracies):.3f}, mean={np.mean(accuracies):.3f}"
    else:
        acc_str = "no results"

    if coverages:
        cov_str = f"CI coverage={sum(coverages)}/{len(coverages)} ({100*sum(coverages)/len(coverages):.0f}%)"
    else:
        cov_str = ""

    print(f"  {method:20s}: {acc_str}  {cov_str}")

# NB-SSP total triggers
t_accs = []
for exp_id, res in all_results.items():
    if 'NB_SSP' in res and 'accuracy_v_T' in res['NB_SSP']:
        t_accs.append(res['NB_SSP']['accuracy_v_T'])
if t_accs:
    print(f"\n  NB-SSP total triggers: median v_T={np.median(t_accs):.3f}, mean={np.mean(t_accs):.3f}")

print(f"\nResults saved to {RESULTS_DIR}/uci_all_results.npy")
