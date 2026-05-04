"""
compute_dm.py
=============
Implements Algorithm 1 from the paper: posterior sampling for D_M
(number of follow-up days to see M new users).

Also implements the inversion-based interval as a faster alternative.

Usage:
    python experiments/evaluation/compute_dm.py
"""

import numpy as np
import os
import sys
from scipy.special import beta as betafn
from scipy.stats import nbinom
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from models.tg_ssp import GD
from models.nb_ssp import NegBintSBSP

# ============================================================
# Algorithm 1: Posterior sampling for D_M
# ============================================================

def get_psi(D0, D1, sigma, r=1):
    """Compute psi_r(D0, D1) = sigma * [B(r*D0+1, -sigma) - B(r*(D0+D1)+1, -sigma)]"""
    return sigma * (betafn(r * D0 + 1, -sigma) - betafn(r * (D0 + D1) + 1, -sigma))


def compute_trigger_time_pmf(sigma, D0, D_up):
    """
    Compute the PMF of first-trigger times for unobserved users.
    Pr(Y = y) proportional to B(1 - sigma, y) for y in {D0+1, ..., D0+D_up}
    """
    support = np.arange(D0 + 1, D0 + D_up + 1)
    log_probs = np.array([np.log(betafn(1 - sigma, y)) for y in support])
    log_probs -= log_probs.max()  # numerical stability
    probs = np.exp(log_probs)
    probs /= probs.sum()
    return support, probs


def sample_dm_algorithm1(D0, N_pilot, parameters, M, D_up=None, K=1000, r=1):
    """
    Algorithm 1: Posterior sampling for D_M.

    Parameters
    ----------
    D0 : int
        Pilot days.
    N_pilot : int
        Number of users in pilot.
    parameters : array-like
        (beta, sigma, tilting) for GD/TG-SSP, or (beta, sigma, tilting, r) for NB-SSP.
    M : int
        Target number of new users.
    D_up : int or None
        Upper bound for D_M (if None, estimated automatically).
    K : int
        Number of Monte Carlo iterations.
    r : float
        1 for Be-SSP/TG-SSP, >1 for NB-SSP.

    Returns
    -------
    dm_samples : ndarray
        Array of K samples from the posterior of D_M.
    """
    if len(parameters) == 4:
        beta_, sigma, tilting, r = parameters
    else:
        beta_, sigma, tilting = parameters
        r = 1

    # Estimate D_up if not provided
    if D_up is None:
        for d_candidate in range(1, 10000):
            psi_num = get_psi(D0, d_candidate, sigma, r)
            psi_den = beta_ + get_psi(0, D0 + d_candidate, sigma, r)
            p = psi_num / psi_den
            expected_U = (N_pilot + tilting + 1) * p / (1 - p)
            if expected_U >= M:
                D_up = 3 * d_candidate  # 3x safety margin as in the paper
                break
        if D_up is None:
            D_up = 10000  # fallback

    # Precompute the trigger-time PMF
    support, pmf = compute_trigger_time_pmf(sigma, D0, D_up)

    # Compute xi distribution parameters
    psi_dup_d0 = get_psi(D_up, D0, sigma, r)
    psi_total = get_psi(0, D0 + D_up, sigma, r)

    n_param = N_pilot + tilting + 1
    p_param = psi_dup_d0 / (beta_ + psi_total)

    dm_samples = np.full(K, np.nan)

    for k in range(K):
        # Step 1: Sample xi (number of new users triggering before D^up)
        xi = nbinom.rvs(n_param, 1 - p_param)

        if xi < M:
            # Not enough users even with D^up days
            dm_samples[k] = D_up + D0  # censored
            continue

        # Step 2: Sample trigger times
        trigger_times = np.random.choice(support, size=xi, p=pmf)

        # Step 3: Sort and take M-th order statistic
        trigger_times.sort()
        dm_samples[k] = trigger_times[M - 1]

    return dm_samples


def compute_dm_inversion(D0, N_pilot, parameters, M, D_up=None, width=0.95, Q=1000, r=1):
    """
    Inversion-based interval for D_M.

    Constructs a global credible band for the cumulative user trajectory,
    then slices at N_pilot + M to get an interval for D_M.

    Returns
    -------
    dm_point : float
        Point estimate (day when mean trajectory crosses M).
    dm_lo : float
        Lower bound of credible interval.
    dm_hi : float
        Upper bound of credible interval.
    dm_per_traj : ndarray
        Per-trajectory D_M samples.
    """
    if len(parameters) == 4:
        beta_, sigma, tilting, r = parameters
    else:
        beta_, sigma, tilting = parameters
        r = 1

    if D_up is None:
        for d_candidate in range(1, 10000):
            psi_num = get_psi(D0, d_candidate, sigma, r)
            psi_den = beta_ + get_psi(0, D0 + d_candidate, sigma, r)
            p = psi_num / psi_den
            expected_U = (N_pilot + tilting + 1) * p / (1 - p)
            if expected_U >= M:
                D_up = 3 * d_candidate
                break
        if D_up is None:
            D_up = 10000

    # Sample trajectories
    gamma_shape = N_pilot + tilting + 1
    gamma_rate = beta_ + get_psi(0, D0, sigma, r)
    delta_samples = np.random.gamma(gamma_shape, 1.0 / gamma_rate, size=Q)

    trajectories = np.zeros((Q, D_up))
    for q in range(Q):
        for ell in range(1, D_up + 1):
            rate = sigma * delta_samples[q] * betafn(1 - sigma, D0 + ell)
            trajectories[q, ell - 1] = np.random.poisson(rate)

    # Cumulative trajectories
    cum_trajectories = np.cumsum(trajectories, axis=1)

    # Mean trajectory
    mean_traj = cum_trajectories.mean(axis=0)

    # Point estimate: first day where mean >= M
    dm_point = None
    for ell in range(D_up):
        if mean_traj[ell] >= M:
            dm_point = D0 + ell + 1
            break
    if dm_point is None:
        dm_point = D0 + D_up

    # For each trajectory, check when it crosses M
    epsilon = 1 - width
    dm_per_traj = np.full(Q, D0 + D_up)
    for q in range(Q):
        for ell in range(D_up):
            if cum_trajectories[q, ell] >= M:
                dm_per_traj[q] = D0 + ell + 1
                break

    dm_lo = np.percentile(dm_per_traj, 100 * epsilon / 2)
    dm_hi = np.percentile(dm_per_traj, 100 * (1 - epsilon / 2))

    return float(dm_point), float(dm_lo), float(dm_hi), dm_per_traj


# ============================================================
# Main: compute D_M for UCI experiments
# ============================================================
if __name__ == '__main__':

    DATA_DIR = os.path.join(REPO_ROOT, 'data', 'preprocessed', 'uci')
    RESULTS_DIR = os.path.join(REPO_ROOT, 'results', 'uci')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load previous fitting results
    all_results = np.load(os.path.join(RESULTS_DIR, "uci_all_results.npy"), allow_pickle=True).item()
    experiments = np.load(os.path.join(DATA_DIR, "experiments_metadata.npy"), allow_pickle=True)

    # Compute D_M for selected experiments
    CASE_STUDIES = [1, 6, 12]
    M_MULTIPLIERS = [1.5, 2.0]

    dm_results = {}

    for exp_id in CASE_STUDIES:
        exp = experiments[exp_id]
        D0 = exp['D0']
        D1 = exp['D1']
        N_pilot = exp['N_pilot']
        cumulative_users = np.array(exp['cumulative_users'])

        dm_results[exp_id] = {'N_pilot': N_pilot, 'cumulative_users': cumulative_users.tolist()}

        for mult in M_MULTIPLIERS:
            M = int(mult * N_pilot)

            # True D_M: first day in follow-up where cumulative new users >= M
            true_dm = None
            for d in range(D0 + 1, len(cumulative_users)):
                if cumulative_users[d] - N_pilot >= M:
                    true_dm = d
                    break

            print(f"\nExp {exp_id}: M={M} ({mult}x N_pilot={N_pilot}), true D_M={'>' + str(D0+D1) if true_dm is None else true_dm}")

            dm_results[exp_id][f'M_{mult}'] = {
                'M': M,
                'true_dm': true_dm,
            }

            # Use NB-SSP regression parameters (best accuracy)
            if 'NB_SSP_regression' in all_results[exp_id] and 'params' in all_results[exp_id]['NB_SSP_regression']:
                params = all_results[exp_id]['NB_SSP_regression']['params']
                r_val = params[3] if len(params) == 4 else 1

                print(f"  Params: beta={params[0]:.2f}, sigma={params[1]:.4f}, c={params[2]:.2f}, r={r_val:.2f}")
                print(f"  Running Algorithm 1 (K=500)...")
                dm_samples = sample_dm_algorithm1(D0, N_pilot, params, M, K=500, r=r_val)
                dm_mean = np.nanmean(dm_samples)
                dm_median = np.nanmedian(dm_samples)
                dm_lo = np.nanpercentile(dm_samples, 2.5)
                dm_hi = np.nanpercentile(dm_samples, 97.5)

                print(f"  Algorithm 1: mean={dm_mean:.1f}, median={dm_median:.1f}, 95% CI=[{dm_lo:.1f}, {dm_hi:.1f}]")

                dm_results[exp_id][f'M_{mult}']['algo1'] = {
                    'mean': float(dm_mean),
                    'median': float(dm_median),
                    'ci_lo': float(dm_lo),
                    'ci_hi': float(dm_hi),
                    'params_used': params,
                }

                print(f"  Running inversion method (Q=500)...")
                inv_point, inv_lo, inv_hi, _ = compute_dm_inversion(D0, N_pilot, params, M, Q=500, r=r_val)

                print(f"  Inversion: point={inv_point:.0f}, 95% CI=[{inv_lo:.0f}, {inv_hi:.0f}]")

                dm_results[exp_id][f'M_{mult}']['inversion'] = {
                    'point': inv_point,
                    'ci_lo': inv_lo,
                    'ci_hi': inv_hi,
                }

    np.save(os.path.join(RESULTS_DIR, "uci_dm_results.npy"), dm_results)
    print(f"\nD_M results saved to {RESULTS_DIR}/uci_dm_results.npy")
