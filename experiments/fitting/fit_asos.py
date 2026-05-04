"""
fit_asos.py
===========
Fits TG-SSP, IBP, and NB-SSP (regression) on all ASOS treatment arms.
ASOS data has first-trigger counts only (no per-user re-trigger matrix),
so we use regression-based fitting for all models.

Output:
    results/asos/asos_all_results.npy

Usage:
    python experiments/fitting/fit_asos.py
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

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
ASOS_PATH = os.path.join(REPO_ROOT, 'data', 'preprocessed', 'asos', 'PAPER_asos_data.npy')
RESULTS_DIR = os.path.join(REPO_ROOT, 'results', 'asos')
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_ITS = 5
WIDTH = 0.95
D0 = 7

# ============================================================
# Load ASOS data
# ============================================================
asos = np.load(ASOS_PATH, allow_pickle=True).item()
print(f"Loaded {len(asos)} experiments")

# ============================================================
# Fitting loop
# ============================================================
all_results = {}
arm_count = 0

for exp_id in tqdm(asos, desc="Fitting ASOS experiments"):
    exp_data = asos[exp_id]

    for arm in exp_data['first_trigger_counts_short']:
        counts_short = exp_data['first_trigger_counts_short'][arm].astype(int)
        counts_long = exp_data['first_trigger_counts_long'][arm].astype(int)

        key = f"{exp_id}_{arm}"
        arm_count += 1

        D0_actual = len(counts_short) - 1  # should be 7
        D1_actual = len(counts_long) - len(counts_short)
        N_pilot = int(counts_short[-1])
        N_total = int(counts_long[-1])
        U_true = N_total - N_pilot

        if D1_actual <= 0 or N_pilot <= 0:
            continue

        results = {
            'exp_id': exp_id,
            'arm': arm,
            'D0': D0_actual,
            'D1': D1_actual,
            'N_pilot': N_pilot,
            'N_total': N_total,
            'U_true': U_true,
            'cumulative_users': counts_long.tolist(),
        }

        # ----------------------------------------------------------
        # 1. TG-SSP (regression)
        # ----------------------------------------------------------
        try:
            st = time.time()
            gd = GD()
            gd_params = gd.regression(counts_short, NUM_ITS, 2, False)

            gd_pred = gd.mean(D0_actual, D1_actual, N_pilot, gd_params)
            gd_U_hat = gd_pred[-1]

            gd_lo, gd_hi = gd.credible_interval(D0_actual, D1_actual, N_pilot, gd_params, WIDTH)
            gd_U_lo, gd_U_hi = gd_lo[-1], gd_hi[-1]

            results['TG_SSP'] = {
                'params': gd_params.tolist(),
                'U_hat': float(gd_U_hat),
                'U_ci': (float(gd_U_lo), float(gd_U_hi)),
                'time': time.time() - st,
            }
        except Exception as e:
            results['TG_SSP'] = {'error': str(e)}

        # ----------------------------------------------------------
        # 2. IBP (regression)
        # ----------------------------------------------------------
        try:
            st = time.time()
            ibp = IBP()
            ibp_params = ibp.regression(counts_short, NUM_ITS, 2, False)

            ibp_pred = ibp.mean(D0_actual, D1_actual, ibp_params)
            ibp_U_hat = ibp_pred[-1]

            from scipy.stats import poisson as poisson_dist
            ibp_U_lo, ibp_U_hi = poisson_dist.interval(confidence=WIDTH, mu=ibp_pred[-1])

            results['IBP'] = {
                'params': ibp_params.tolist(),
                'U_hat': float(ibp_U_hat),
                'U_ci': (float(ibp_U_lo), float(ibp_U_hi)),
                'time': time.time() - st,
            }
        except Exception as e:
            results['IBP'] = {'error': str(e)}

        # ----------------------------------------------------------
        # 3. NB-SSP (regression, r free)
        # ----------------------------------------------------------
        try:
            st = time.time()
            nbp = NegBintSBSP()
            nbp_params = nbp.fit_regression(
                D0=0, N0=0,
                observed_counts=counts_short[1:],
                num_its=NUM_ITS
            )

            nbp_pred = nbp.mean_number_new_users(D0_actual, D1_actual, N_pilot, nbp_params)
            nbp_U_hat = nbp_pred[-1]

            nbp_ci = nbp.ci_number_new_users(D0_actual, D1_actual, N_pilot, nbp_params, width=WIDTH)
            nbp_U_lo, nbp_U_hi = nbp_ci[-1]

            results['NB_SSP_regression'] = {
                'params': nbp_params.tolist(),
                'U_hat': float(nbp_U_hat),
                'U_ci': (float(nbp_U_lo), float(nbp_U_hi)),
                'time': time.time() - st,
            }
        except Exception as e:
            results['NB_SSP_regression'] = {'error': str(e)}

        # ----------------------------------------------------------
        # Accuracy metrics
        # ----------------------------------------------------------
        for method in ['TG_SSP', 'IBP', 'NB_SSP_regression']:
            if method in results and 'U_hat' in results[method]:
                U_hat = results[method]['U_hat']
                v = 1 - min(abs(U_true - U_hat) / max(U_true, 1), 1)
                results[method]['accuracy_v'] = float(v)

                if 'U_ci' in results[method]:
                    lo, hi = results[method]['U_ci']
                    results[method]['ci_covers'] = bool(lo <= U_true <= hi)

        all_results[key] = results

# ============================================================
# Save
# ============================================================
np.save(os.path.join(RESULTS_DIR, "asos_all_results.npy"), all_results)

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"SUMMARY ({arm_count} treatment arms)")
print(f"{'='*60}")

for method in ['TG_SSP', 'IBP', 'NB_SSP_regression']:
    accuracies = []
    coverages = []
    errors = 0
    for key, res in all_results.items():
        if method in res:
            if 'accuracy_v' in res[method]:
                accuracies.append(res[method]['accuracy_v'])
            if 'ci_covers' in res[method]:
                coverages.append(res[method]['ci_covers'])
            if 'error' in res[method]:
                errors += 1

    if accuracies:
        print(f"  {method:20s}: median v={np.median(accuracies):.3f}, "
              f"mean={np.mean(accuracies):.3f}, "
              f"CI coverage={sum(coverages)}/{len(coverages)} ({100*sum(coverages)/len(coverages):.0f}%), "
              f"errors={errors}")
    else:
        print(f"  {method:20s}: no results (errors={errors})")

print(f"\nResults saved to {RESULTS_DIR}/asos_all_results.npy")
