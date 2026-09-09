"""
fit_rees46.py
=============
Fits TG-SSP, IBP, and NB-SSP (regression) on REES46 sliding-window experiments.
REES46 experiments are large (~800K pilot users), so we use regression only.
Saves intermediate results after each experiment for progress tracking.

Output:
    ../results/rees46_tmp/exp_{i}.npy   (intermediate, per-experiment)
    ../results/rees46_all_results.npy   (final, all experiments)

Usage:
    cd SubmissionAOAS/fitting
    python fit_rees46.py
"""

import numpy as np
import sys
import os
import time
import warnings

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))
np.random.seed(int(os.environ.get("SEED", "0")))

from utils_all import *
from utils_GD import *
from utils_IBP import *
from utils_NBP import *

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "rees46")
RESULTS_DIR = os.environ.get("OUTDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
TMP_DIR = os.path.join(RESULTS_DIR, "rees46_tmp")
os.makedirs(TMP_DIR, exist_ok=True)

NUM_ITS = 5
WIDTH = 0.95

# ============================================================
# Load
# ============================================================
experiments = np.load(os.path.join(DATA_DIR, "experiments_metadata.npy"), allow_pickle=True)
print(f"Loaded {len(experiments)} REES46 experiments")

# ============================================================
# Fitting loop
# ============================================================
all_results = {}

for exp in experiments:
    exp_id = exp['exp_id']
    D0 = exp['D0']
    D1 = exp['D1']
    N_pilot = exp['N_pilot']
    U_true = exp['U_true']
    T_true = exp['T_true']
    cumulative_users = np.array(exp['cumulative_users'])
    counts_short = cumulative_users[:D0+1].astype(int)

    print(f"\n--- Exp {exp_id}: N_pilot={N_pilot:,}, U_true={U_true:,} ---")

    results = {
        'exp_id': exp_id,
        'D0': D0,
        'D1': D1,
        'N_pilot': N_pilot,
        'U_true': U_true,
        'T_true': T_true,
        'cumulative_users': cumulative_users.tolist(),
    }

    # ----------------------------------------------------------
    # 1. TG-SSP (regression)
    # ----------------------------------------------------------
    try:
        st = time.time()
        gd = GD()
        gd_params = gd.regression(counts_short, NUM_ITS, 2, False)

        gd_pred = gd.mean(D0, D1, N_pilot, gd_params)
        gd_U_hat = gd_pred[-1]

        gd_lo, gd_hi = gd.credible_interval(D0, D1, N_pilot, gd_params, WIDTH)
        gd_U_lo, gd_U_hi = gd_lo[-1], gd_hi[-1]

        results['TG_SSP'] = {
            'params': gd_params.tolist(),
            'U_hat': float(gd_U_hat),
            'U_ci': (float(gd_U_lo), float(gd_U_hi)),
            'time': time.time() - st,
        }
        print(f"  TG-SSP: U_hat={gd_U_hat:,.0f}, CI=[{gd_U_lo:,.0f}, {gd_U_hi:,.0f}], time={results['TG_SSP']['time']:.1f}s")
    except Exception as e:
        results['TG_SSP'] = {'error': str(e)}
        print(f"  TG-SSP FAILED: {e}")

    # ----------------------------------------------------------
    # 2. IBP (regression)
    # ----------------------------------------------------------
    try:
        st = time.time()
        ibp = IBP()
        ibp_params = ibp.regression(counts_short, NUM_ITS, 2, False)

        ibp_pred = ibp.mean(D0, D1, ibp_params)
        ibp_U_hat = ibp_pred[-1]

        from scipy.stats import poisson as poisson_dist
        ibp_U_lo, ibp_U_hi = poisson_dist.interval(confidence=WIDTH, mu=ibp_pred[-1])

        results['IBP'] = {
            'params': ibp_params.tolist(),
            'U_hat': float(ibp_U_hat),
            'U_ci': (float(ibp_U_lo), float(ibp_U_hi)),
            'time': time.time() - st,
        }
        print(f"  IBP:    U_hat={ibp_U_hat:,.0f}, CI=[{ibp_U_lo:,.0f}, {ibp_U_hi:,.0f}], time={results['IBP']['time']:.1f}s")
    except Exception as e:
        results['IBP'] = {'error': str(e)}
        print(f"  IBP FAILED: {e}")

    # ----------------------------------------------------------
    # 3. NB-SSP (regression)
    # ----------------------------------------------------------
    try:
        st = time.time()
        nbp = NegBintSBSP()
        nbp_params = nbp.fit_regression(
            D0=0, N0=0,
            observed_counts=counts_short[1:],
            num_its=NUM_ITS
        )

        nbp_pred = nbp.mean_number_new_users(D0, D1, N_pilot, nbp_params)
        nbp_U_hat = nbp_pred[-1]

        nbp_ci = nbp.ci_number_new_users(D0, D1, N_pilot, nbp_params, width=WIDTH)
        nbp_U_lo, nbp_U_hi = nbp_ci[-1]

        results['NB_SSP_regression'] = {
            'params': nbp_params.tolist(),
            'U_hat': float(nbp_U_hat),
            'U_ci': (float(nbp_U_lo), float(nbp_U_hi)),
            'time': time.time() - st,
        }
        print(f"  NB-SSP: U_hat={nbp_U_hat:,.0f}, CI=[{nbp_U_lo:,.0f}, {nbp_U_hi:,.0f}], time={results['NB_SSP_regression']['time']:.1f}s")
    except Exception as e:
        results['NB_SSP_regression'] = {'error': str(e)}
        print(f"  NB-SSP FAILED: {e}")

    # ----------------------------------------------------------
    # Accuracy + coverage
    # ----------------------------------------------------------
    for method in ['TG_SSP', 'IBP', 'NB_SSP_regression']:
        if method in results and 'U_hat' in results[method]:
            U_hat = results[method]['U_hat']
            v = 1 - min(abs(U_true - U_hat) / max(U_true, 1), 1)
            results[method]['accuracy_v'] = float(v)
            if 'U_ci' in results[method]:
                lo, hi = results[method]['U_ci']
                results[method]['ci_covers'] = bool(lo <= U_true <= hi)

    # Save intermediate
    np.save(os.path.join(TMP_DIR, f"exp_{exp_id}.npy"), results)
    all_results[exp_id] = results
    print(f"  Saved tmp for exp {exp_id}")

# ============================================================
# Save final + summary
# ============================================================
np.save(os.path.join(RESULTS_DIR, "rees46_all_results.npy"), all_results)

print(f"\n{'='*60}")
print(f"SUMMARY ({len(experiments)} experiments)")
print(f"{'='*60}")

for method in ['TG_SSP', 'IBP', 'NB_SSP_regression']:
    accs = []
    covs = []
    for key, res in all_results.items():
        if method in res and 'accuracy_v' in res[method]:
            accs.append(res[method]['accuracy_v'])
        if method in res and 'ci_covers' in res[method]:
            covs.append(res[method]['ci_covers'])
    if accs:
        print(f"  {method:20s}: median v={np.median(accs):.3f}, mean={np.mean(accs):.3f}, "
              f"CI coverage={sum(covs)}/{len(covs)} ({100*sum(covs)/len(covs):.0f}%)")

print(f"\nResults saved to {RESULTS_DIR}/rees46_all_results.npy")
