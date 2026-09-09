"""Bernoulli stable beta-scaled process (Be-SSP)."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.special import betaln, gammaln

from ._optimization import fit_differential_evolution
from ._ssp import (
    PARAMETER_BOUNDS,
    PARAMETER_NAMES,
    curve_fitting_objective,
    expected_new_users,
    new_user_interval,
    psi,
)


class BeSSP:
    """Be-SSP point prediction and empirical-Bayes fitting."""

    def expected_new_users(
        self,
        pilot_days: int,
        followup_days: int,
        observed_users: int,
        *,
        alpha: float,
        c: float,
        beta: float,
    ) -> np.ndarray:
        return expected_new_users(
            pilot_days,
            followup_days,
            observed_users,
            alpha=alpha,
            c=c,
            beta=beta,
        )

    def new_user_interval(
        self,
        pilot_days: int,
        followup_days: int,
        observed_users: int,
        coverage: float = 0.95,
        *,
        alpha: float,
        c: float,
        beta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return new_user_interval(
            pilot_days,
            followup_days,
            observed_users,
            coverage=coverage,
            alpha=alpha,
            c=c,
            beta=beta,
        )

    def fit_curve(
        self,
        cumulative_users: np.ndarray,
        *,
        start_day: int = 1,
        num_restarts: int = 5,
        norm: float = 2,
        seed: int | None = None,
    ) -> dict[str, float]:
        objective = curve_fitting_objective(
            cumulative_users,
            start_day=start_day,
            norm=norm,
        )
        return fit_differential_evolution(
            objective,
            PARAMETER_BOUNDS,
            PARAMETER_NAMES,
            num_restarts=num_restarts,
            seed=seed,
        )

    def fit_marginal_likelihood(
        self,
        sfs: np.ndarray,
        sample_size: int,
        *,
        num_restarts: int = 5,
        seed: int | None = None,
    ) -> dict[str, float]:
        spectrum = np.asarray(sfs, dtype=float)
        if spectrum.ndim != 1 or len(spectrum) == 0:
            raise ValueError("sfs must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(spectrum)) or np.any(spectrum < 0.0):
            raise ValueError("sfs must contain finite non-negative values")
        if not isinstance(sample_size, Integral) or isinstance(sample_size, (bool, np.bool_)):
            raise ValueError("sample_size must be an integer")
        sample_size = int(sample_size)
        if sample_size < 1:
            raise ValueError("sample_size must be positive")
        if len(spectrum) > sample_size:
            raise ValueError("sfs cannot have more entries than sample_size")
        if spectrum.sum() <= 0.0:
            raise ValueError("sfs must contain at least one observed user")

        objective = _marginal_likelihood_objective(spectrum, sample_size)
        return fit_differential_evolution(
            objective,
            PARAMETER_BOUNDS,
            PARAMETER_NAMES,
            num_restarts=num_restarts,
            seed=seed,
        )


def _marginal_likelihood_objective(
    sfs: np.ndarray,
    sample_size: int,
):
    observed_users = float(sfs.sum())
    frequencies = np.arange(1, len(sfs) + 1, dtype=float)

    def objective(parameters: np.ndarray) -> float:
        alpha, c, beta = parameters
        log_likelihood = observed_users * np.log(alpha)
        log_likelihood += (c + 1.0) * np.log(beta)
        log_likelihood -= (observed_users + c + 1.0) * np.log(
            beta + psi(0, sample_size, alpha=alpha)
        )
        log_likelihood += gammaln(observed_users + c + 1.0) - gammaln(c + 1.0)
        log_likelihood += np.inner(
            sfs,
            betaln(frequencies - alpha, sample_size - frequencies + 1.0),
        )
        return float(-log_likelihood)

    return objective


__all__ = ["BeSSP"]
