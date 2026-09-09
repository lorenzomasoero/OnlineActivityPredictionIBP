"""
fit_rees46_rolling.py
=====================
Fits TG-SSP, IBP, NB-SSP on rolling-window REES46 experiments.
Reads experiments_rolling_k{K}.npy files produced by preprocess_rees46_rolling.py.

Usage:
    cd SubmissionAOAS/fitting
    python fit_rees46_rolling.py          # all k values
    python fit_rees46_rolling.py --k 21   # single k
"""

import numpy as np
import sys
import os
import time
import warnings
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))
np.random.seed(int(os.environ.get("SEED", "0")))
from utils_all import *
from utils_GD import *
from utils_IBP import *
from utils_NBP import *

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "rees46")
RESULTS_DIR = os.environ.get("OUTDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
NUM_ITS = 5
WIDTH = 0.95

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, nargs='*', default=None, help='k values to fit (default: all found)')
args = parser.parse_args()

# Find available k files
if args.k:
    k_values = args.k
else:
    import glob
    k_files = glob.glob(os.path.join(DATA_DIR, 'experiments_rolling_k*.npy'))
    k_values = sorted(int(f.split('_k')[1].split('.')[0]) for f in k_files)

print(f"Fitting rolling experiments for k={k_values}")

for k in k_values:
    infile = os.path.join(DATA_DIR, f'experiments_rolling_k{k}.npy')
    experiments = np.load(infile, allow_pickle=True)
    n_exp = len(experiments)
    print(f"\n{'='*60}")
    print(f"k={k}: {n_exp} experiments")
    print(f"{'='*60}")

    tmp_dir = os.path.join(RESULTS_DIR, f'rees46_rolling_k{k}')
    os.makedirs(tmp_dir, exist_ok=True)

    all_results = {}
    t_start = time.time()

    for exp in experiments:
        exp_id = exp['exp_id']
        D0 = exp['D0']
        D1 = exp['D1']
        N_pilot = exp['N_pilot']
        U_true = exp['U_true']
        N_total = exp['N_total']
        cumulative_users = np.array(exp['cumulative_users'])
        counts_short = cumulative_users[:D0+1].astype(int)

        results = {
            'exp_id': exp_id, 'k': k, 'D0': D0, 'D1': D1,
            'N_pilot': N_pilot, 'U_true': U_true, 'N_total': N_total,
        }

        # TG-SSP
        try:
            st = time.time()
            gd = GD()
            gd_params = gd.regression(counts_short, NUM_ITS, 2, False)
            gd_U_hat = gd.mean(D0, D1, N_pilot, gd_params)[-1]
            gd_lo, gd_hi = gd.credible_interval(D0, D1, N_pilot, gd_params, WIDTH)
            results['TG_SSP'] = {'params': gd_params.tolist(), 'U_hat': float(gd_U_hat),
                                 'U_ci': (float(gd_lo[-1]), float(gd_hi[-1])), 'time': time.time() - st}
        except Exception as e:
            results['TG_SSP'] = {'error': str(e)}

        # IBP
        try:
            st = time.time()
            ibp = IBP()
            ibp_params = ibp.regression(counts_short, NUM_ITS, 2, False)
            ibp_U_hat = ibp.mean(D0, D1, ibp_params)[-1]
            from scipy.stats import poisson as poisson_dist
            ibp_lo, ibp_hi = poisson_dist.interval(confidence=WIDTH, mu=ibp_U_hat)
            results['IBP'] = {'params': ibp_params.tolist(), 'U_hat': float(ibp_U_hat),
                              'U_ci': (float(ibp_lo), float(ibp_hi)), 'time': time.time() - st}
        except Exception as e:
            results['IBP'] = {'error': str(e)}

        # NB-SSP
        try:
            st = time.time()
            nbp = NegBintSBSP()
            nbp_params = nbp.fit_regression(D0=0, N0=0, observed_counts=counts_short[1:], num_its=NUM_ITS)
            nbp_U_hat = nbp.mean_number_new_users(D0, D1, N_pilot, nbp_params)[-1]
            nbp_ci = nbp.ci_number_new_users(D0, D1, N_pilot, nbp_params, width=WIDTH)
            results['NB_SSP_regression'] = {'params': nbp_params.tolist(), 'U_hat': float(nbp_U_hat),
                                            'U_ci': (float(nbp_ci[-1][0]), float(nbp_ci[-1][1])), 'time': time.time() - st}
        except Exception as e:
            results['NB_SSP_regression'] = {'error': str(e)}

        # Accuracy + coverage
        for method in ['TG_SSP', 'IBP', 'NB_SSP_regression']:
            if method in results and 'U_hat' in results[method]:
                U_hat = results[method]['U_hat']
                results[method]['accuracy_v'] = float(1 - min(abs(U_true - U_hat) / max(U_true, 1), 1))
                if 'U_ci' in results[method]:
                    lo, hi = results[method]['U_ci']
                    results[method]['ci_covers'] = bool(lo <= U_true <= hi)

        all_results[exp_id] = results

        if exp_id % 20 == 0:
            elapsed = time.time() - t_start
            rate = (exp_id + 1) / elapsed if elapsed > 0 else 0
            eta = (n_exp - exp_id - 1) / rate if rate > 0 else 0
            print(f"  exp {exp_id}/{n_exp}: NB-SSP v={results.get('NB_SSP_regression',{}).get('accuracy_v',0):.3f}  "
                  f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]")

    # Save
    outfile = os.path.join(RESULTS_DIR, f'rees46_rolling_k{k}_results.npy')
    np.save(outfile, all_results)

    # Summary
    elapsed = time.time() - t_start
    print(f"\n  k={k} done in {elapsed:.0f}s ({elapsed/n_exp:.1f}s/exp)")
    for method in ['TG_SSP', 'IBP', 'NB_SSP_regression']:
        accs = [r[method]['accuracy_v'] for r in all_results.values() if method in r and 'accuracy_v' in r[method]]
        covs = [r[method]['ci_covers'] for r in all_results.values() if method in r and 'ci_covers' in r[method]]
        if accs:
            print(f"  {method:22s}: median v={np.median(accs):.3f}, mean={np.mean(accs):.3f}, "
                  f"CI coverage={sum(covs)}/{len(covs)} ({100*sum(covs)/len(covs):.0f}%)")
    print(f"  Saved to {outfile}")

print("\nAll done.")
