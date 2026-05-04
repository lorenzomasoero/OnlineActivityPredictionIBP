"""
Shared utility functions for Bayesian nonparametric models.

Provides mathematical helpers (log-Pochhammer, log-binomial, log-beta),
binary matrix manipulation utilities, and data generation functions used
across the model implementations.
"""

import os
import numpy as np
from scipy.special import gamma as spg
from scipy.special import beta as spb
from scipy.special import binom as spbin
from scipy.special import betaln as bln
from scipy.special import gammaln as gln
import scipy.stats as spst
from scipy import optimize


# ---------------------------------------------------------------------------
# Mathematical helpers
# ---------------------------------------------------------------------------

def log_poch(x, n):
    """
    Log of the Pochhammer symbol (rising factorial) of x of order n.

    Returns ln(Gamma(x + n) / Gamma(x)).

    See: http://mathworld.wolfram.com/PochhammerSymbol.html
    """
    return gln(x + n) - gln(x)


def log_beta(alpha, beta):
    """Return the logarithm of the beta function B(alpha, beta)."""
    return gln(alpha) + gln(beta) - gln(alpha + beta)


def log_binom(a, b):
    """Compute log(a choose b)."""
    return gln(a + 1) - gln(b + 1) - gln(a - b + 1)


def make_color_dict(color):
    """Return a matplotlib boxplot style dictionary for the given colour."""
    dict_ = {
        'patch_artist': True,
        'boxprops': dict(color=color, facecolor='w'),
        'capprops': dict(color=color),
        'flierprops': dict(color=color, markeredgecolor=color),
        'medianprops': dict(color='k'),
        'whiskerprops': dict(color=color),
    }
    return dict_


# ---------------------------------------------------------------------------
# Folder / path helpers
# ---------------------------------------------------------------------------

def create_folder(path):
    """Create *path* if it does not already exist."""
    if not os.path.exists(path):
        os.makedirs(path)


# ---------------------------------------------------------------------------
# Binary-matrix data generation and manipulation
# ---------------------------------------------------------------------------

def generate_bin_matrix_from_freqs(thetas, T, seed=0):
    """
    Generate a binary observation matrix from population frequencies.

    Parameters
    ----------
    thetas : array, shape (K,)
        Frequencies of variants in [0, 1].
    T : int
        Number of experiment days.
    seed : int
        Seed for the RNG.

    Returns
    -------
    X : array, shape (T, K)
        Binary matrix with X[t, n] ~ Bernoulli(thetas[n]).
    """
    np.random.seed(seed)
    X = np.random.binomial(1, np.repeat(thetas, T)).reshape(len(thetas), T).T
    return X


def generate_cts_from_bin_mat(X):
    """
    Compute the accumulation curve from a binary observation matrix.

    Parameters
    ----------
    X : array, shape (T, N), binary valued

    Returns
    -------
    cts : array, shape (T + 1,)
        t-th entry is the number of distinct customers observed in the
        first t days; the first entry is 0.
    """
    return np.concatenate([[0], np.count_nonzero(X.cumsum(axis=0), axis=1)])


def count_new_freq(binary_matrix, T, T2, r):
    """
    Count customers not observed in the first *T* days who appear with
    frequency *r* in *T2* additional days.
    """
    assert T + T2 <= binary_matrix.shape[0], 'T+T2 <= # rows'
    binary_cumsum = binary_matrix.cumsum(axis=0)
    yet_to_be_seen = (binary_cumsum[T] == 0)
    seen_with_frequency_at_T2 = np.array(
        [(binary_cumsum[T + t] == r) for t in range(T2)]
    )
    return np.sum(seen_with_frequency_at_T2 * yet_to_be_seen[np.newaxis, :], axis=1)


# ---------------------------------------------------------------------------
# Matrix utilities (from matrix_utils.py)
# ---------------------------------------------------------------------------

def count_news_from_matrix(matrix, start=0, end=None, frequency=None):
    """
    Count new columns (variants) appearing between rows *start* and *end*.

    Given an integer-valued matrix, for every row between ``start`` and
    ``end`` return the number of columns that are zero up to *start* and
    have cumulative count equal to *frequency* (or > 0 if *frequency* is
    ``None``) from *start* to the corresponding row.
    """
    idxs_news = np.where(matrix[:start].sum(axis=0) == 0)[0]
    submatrix = matrix[start:end, idxs_news]
    submatrix_cumsum = submatrix.cumsum(axis=0)
    if frequency is None:
        return (submatrix_cumsum > 0).sum(axis=1)
    return (submatrix_cumsum == frequency).sum(axis=1)


def count_total_abundance_previously_seen(matrix, pilot, follow):
    """
    Total abundance of previously-seen columns between rows *pilot* and
    *pilot + follow*.
    """
    submatrix = matrix[:pilot].reshape(pilot, matrix.shape[1])
    active = np.sum(submatrix.sum(axis=0) > 0)
    if active == 0:
        return 0
    return matrix[pilot:pilot + follow, :active].sum()


def count_total_abundance_yet_to_be_seen(matrix, pilot, follow):
    """
    Total abundance of yet-to-be-seen columns between rows *pilot* and
    *pilot + follow*.
    """
    submatrix = matrix[:pilot].reshape(pilot, matrix.shape[1])
    active = np.sum(submatrix.sum(axis=0) > 0)
    if active == 0:
        return 0
    return matrix[pilot:pilot + follow, active:].sum()
