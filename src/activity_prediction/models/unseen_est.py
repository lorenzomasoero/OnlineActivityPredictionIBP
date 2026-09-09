"""Linear-programming unseen-species estimator of Zou et al."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import binom


def _require_nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _estimate_rare_histogram(
    sample_size: int,
    sfs: np.ndarray,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from cvxopt import matrix, solvers
    except ImportError as error:
        raise ImportError(
            "The unseen-species estimator requires the optional dependency "
            "'cvxopt'. Install AoAS Code with its unseen-estimator extra."
        ) from error

    rare_cutoff = int(sample_size * kappa)
    rare_frequencies = list(sfs[:rare_cutoff].astype(int))

    grid_factor = 1.01
    grid_max = len(rare_frequencies) / sample_size
    grid_min = 1.0 / (sample_size * 100)
    maximum_population = 65_000_000

    target_moments = rare_frequencies + [0] * int(
        np.ceil(np.sqrt(len(rare_frequencies)))
    )
    number_moments = len(target_moments)
    grid = grid_min * np.power(
        grid_factor,
        np.arange(
            0,
            np.ceil(np.log(grid_max / grid_min) / np.log(grid_factor)) + 1,
        ),
    )
    grid_size = int(np.max(grid.shape))

    objective = np.zeros((1, grid_size + 2 * number_moments))
    objective[0, np.arange(grid_size, grid_size + 2 * number_moments, 2)] = (
        1.0 / np.sqrt(np.asarray(target_moments) + 1)
    )
    objective[
        0, np.arange(grid_size + 1, grid_size + 2 * number_moments, 2)
    ] = 1.0 / np.sqrt(np.asarray(target_moments) + 1)

    number_variables = grid_size + 2 * number_moments
    number_constraints = 2 * number_moments + number_variables + 1
    constraints = np.zeros((number_constraints, number_variables))
    bounds = np.zeros((number_constraints, 1))

    binomial_distributions = [binom(sample_size, probability) for probability in grid]
    for index in range(number_moments):
        constraints[2 * index, np.arange(grid_size)] = [
            distribution.pmf(index + 1)
            for distribution in binomial_distributions
        ]
        constraints[2 * index + 1, np.arange(grid_size)] = -constraints[
            2 * index, np.arange(grid_size)
        ]
        constraints[2 * index, grid_size + 2 * index] = -1
        constraints[2 * index + 1, grid_size + 2 * index + 1] = -1
        bounds[2 * index, 0] = target_moments[index]
        bounds[2 * index + 1, 0] = -target_moments[index]

    for index in range(number_variables):
        constraints[index + 2 * number_moments, index] = -1

    constraints[-1, range(grid_size)] = 1
    bounds[-1, 0] = maximum_population

    equality_constraint = np.zeros((1, number_variables))
    equality_constraint[0, range(grid_size)] = grid
    equality_bound = np.sum(
        np.asarray(rare_frequencies) * (1 + np.arange(len(rare_frequencies)))
    ) / sample_size

    solvers.options["show_progress"] = False
    for index in range(grid_size):
        constraints[:, index] /= grid[index]
        equality_constraint[0, index] /= grid[index]

    solution = solvers.lp(
        matrix(objective.T),
        matrix(constraints),
        matrix(bounds),
        matrix(equality_constraint),
        matrix(equality_bound),
    )

    scaled_histogram = np.asarray(solution["x"])[0:grid_size]
    histogram = np.asarray(
        [scaled_histogram[index] / grid[index] for index in range(grid_size)]
    )
    return histogram.reshape(histogram.size), grid


def predict_unseen_est(
    sample_size: int,
    extrapolation_size: int,
    sfs: Sequence[int] | np.ndarray,
    *,
    kappa: float = 0.5,
) -> np.ndarray:
    """Return the estimated accumulation curve through the extrapolation horizon."""
    sample_size = _require_nonnegative_integer(sample_size, name="sample_size")
    if sample_size == 0:
        raise ValueError("sample_size must be positive")
    extrapolation_size = _require_nonnegative_integer(
        extrapolation_size, name="extrapolation_size"
    )

    frequencies = np.asarray(sfs)
    if frequencies.ndim != 1:
        raise ValueError("sfs must be one-dimensional")
    if frequencies.size == 0:
        raise ValueError("sfs must not be empty")
    if not np.issubdtype(frequencies.dtype, np.number):
        raise TypeError("sfs must contain numeric counts")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("sfs must contain only finite values")
    if np.any(frequencies < 0) or np.any(frequencies != np.floor(frequencies)):
        raise ValueError("sfs must contain non-negative integer counts")
    if frequencies.size > sample_size:
        raise ValueError("sfs cannot have more entries than sample_size")
    frequencies = frequencies.astype(int, copy=False)

    if not np.isscalar(kappa) or not np.isfinite(kappa) or not 0 < kappa < 1:
        raise ValueError("kappa must lie strictly between 0 and 1")
    kappa = float(kappa)
    rare_cutoff = int(sample_size * kappa)
    if rare_cutoff == 0 or frequencies[:rare_cutoff].size == 0:
        raise ValueError("kappa selects no rare-frequency entries")

    rare_histogram, rare_grid = _estimate_rare_histogram(
        sample_size,
        frequencies,
        kappa,
    )

    empirical_histogram = frequencies[rare_cutoff:]
    empirical_grid = np.asarray(
        [frequency / sample_size for frequency in range(rare_cutoff, frequencies.size)]
    )
    grid = np.concatenate((rare_grid, empirical_grid))
    histogram = np.concatenate((rare_histogram, empirical_histogram))

    return np.asarray(
        [
            (histogram * (1 - (1 - grid) ** sample)).sum()
            for sample in range(sample_size + extrapolation_size + 1)
        ]
    )
