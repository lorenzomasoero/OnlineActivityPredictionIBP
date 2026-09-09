"""Negative-binomial stable-beta scaled process model from the paper."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import beta as beta_function
from scipy.special import betaln, gammaln
from scipy.stats import nbinom

from ._optimization import fit_differential_evolution


_PARAMETER_NAMES = ("alpha", "c", "beta", "r")
_CURVE_BOUNDS = (
    (1e-5, 0.99999),
    (1e-8, 1e5),
    (1e-8, 1e5),
    (1e-5, 1e3),
)
_MARGINAL_LIKELIHOOD_BOUNDS = (
    (1e-5, 0.9999),
    (1e-8, 1e5),
    (1e-8, 1e5),
    (1e-5, 1e3),
)


def _require_nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_parameters(
    *, alpha: float, c: float, beta: float, r: float
) -> tuple[float, float, float, float]:
    parameters = np.asarray((alpha, c, beta, r), dtype=float)
    if not np.all(np.isfinite(parameters)):
        raise ValueError("NB-SSP parameters must be finite")

    alpha, c, beta, r = map(float, parameters)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if c <= 0:
        raise ValueError("c must be positive")
    if beta <= 0:
        raise ValueError("beta must be positive")
    if r <= 0:
        raise ValueError("r must be positive")
    return alpha, c, beta, r


def _psi(
    alpha: float,
    r: float,
    start_days: int | np.ndarray,
    additional_days: int | np.ndarray,
) -> np.ndarray:
    """Evaluate the paper's ``psi_r(x, y)`` function."""
    start = np.asarray(start_days, dtype=float)
    additional = np.asarray(additional_days, dtype=float)
    return alpha * (
        beta_function(r * start + 1.0, -alpha)
        - beta_function(r * (start + additional) + 1.0, -alpha)
    )


def _validate_cumulative_users(
    cumulative_users: Sequence[float] | np.ndarray,
) -> np.ndarray:
    observed = np.asarray(cumulative_users, dtype=float)
    if observed.ndim != 1:
        raise ValueError("cumulative_users must be one-dimensional")
    if not np.all(np.isfinite(observed)):
        raise ValueError("cumulative_users must contain only finite values")
    if np.any(observed < 0):
        raise ValueError("cumulative_users must be non-negative")
    if np.any(observed != np.floor(observed)):
        raise ValueError("cumulative_users must contain integer counts")
    if np.any(np.diff(observed) < 0):
        raise ValueError("cumulative_users must be non-decreasing")
    return observed


def _validate_activity_matrix(activity_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(activity_matrix)
    if matrix.ndim != 2:
        raise ValueError("activity_matrix must be two-dimensional")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("activity_matrix must include at least one day and user")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError("activity_matrix must contain numeric counts")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("activity_matrix must contain only finite values")
    if np.any(matrix < 0) or np.any(matrix != np.floor(matrix)):
        raise ValueError("activity_matrix must contain non-negative integer counts")

    matrix = matrix.astype(float, copy=False)
    active_columns = matrix.sum(axis=0) > 0
    if not np.any(active_columns):
        raise ValueError("activity_matrix must contain at least one active user")
    return matrix[:, active_columns]


class NBSSP:
    """Negative-binomial stable-beta scaled process (NB-SSP)."""

    @staticmethod
    def expected_new_users(
        pilot_days: int,
        followup_days: int,
        observed_users: int,
        *,
        alpha: float,
        c: float,
        beta: float,
        r: float,
    ) -> np.ndarray:
        """Return cumulative expected new users for each follow-up horizon.

        This is Equation (3.8) in the paper with ``r_* = r``.
        """
        pilot_days = _require_nonnegative_integer(pilot_days, name="pilot_days")
        followup_days = _require_nonnegative_integer(
            followup_days, name="followup_days"
        )
        observed_users = _require_nonnegative_integer(
            observed_users, name="observed_users"
        )
        alpha, c, beta, r = _validate_parameters(
            alpha=alpha, c=c, beta=beta, r=r
        )

        if followup_days == 0:
            return np.empty(0, dtype=float)

        horizons = np.arange(1, followup_days + 1)
        posterior_rate = beta + _psi(alpha, r, 0, pilot_days)
        return (observed_users + c + 1.0) * _psi(
            alpha, r, pilot_days, horizons
        ) / posterior_rate

    @staticmethod
    def new_user_interval(
        pilot_days: int,
        followup_days: int,
        observed_users: int,
        *,
        alpha: float,
        c: float,
        beta: float,
        r: float,
        coverage: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return marginal credible intervals for cumulative new users."""
        pilot_days = _require_nonnegative_integer(pilot_days, name="pilot_days")
        followup_days = _require_nonnegative_integer(
            followup_days, name="followup_days"
        )
        observed_users = _require_nonnegative_integer(
            observed_users, name="observed_users"
        )
        alpha, c, beta, r = _validate_parameters(
            alpha=alpha, c=c, beta=beta, r=r
        )
        if not np.isscalar(coverage) or not np.isfinite(coverage):
            raise ValueError("coverage must be a finite scalar")
        coverage = float(coverage)
        if not 0 < coverage < 1:
            raise ValueError("coverage must lie in (0, 1)")

        if followup_days == 0:
            empty = np.empty(0, dtype=float)
            return empty, empty.copy()

        horizons = np.arange(1, followup_days + 1)
        followup_psi = _psi(alpha, r, pilot_days, horizons)
        paper_probability = followup_psi / (
            beta + _psi(alpha, r, 0, pilot_days + horizons)
        )
        lower, upper = nbinom.interval(
            coverage,
            n=observed_users + c + 1.0,
            p=1.0 - paper_probability,
        )
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

    @staticmethod
    def expected_total_triggers(
        pilot_days: int,
        followup_days: int,
        observed_users: int,
        pilot_total_triggers: int,
        *,
        alpha: float,
        c: float,
        beta: float,
        r: float,
    ) -> np.ndarray:
        """Return Proposition 3.7's expected total follow-up triggers."""
        pilot_days = _require_nonnegative_integer(pilot_days, name="pilot_days")
        if pilot_days == 0:
            raise ValueError("pilot_days must be positive")
        followup_days = _require_nonnegative_integer(
            followup_days, name="followup_days"
        )
        observed_users = _require_nonnegative_integer(
            observed_users, name="observed_users"
        )
        pilot_total_triggers = _require_nonnegative_integer(
            pilot_total_triggers, name="pilot_total_triggers"
        )
        if pilot_total_triggers < observed_users:
            raise ValueError(
                "pilot_total_triggers cannot be smaller than observed_users"
            )
        alpha, c, beta, r = _validate_parameters(
            alpha=alpha, c=c, beta=beta, r=r
        )

        if followup_days == 0:
            return np.empty(0, dtype=float)

        horizons = np.arange(1, followup_days + 1, dtype=float)
        posterior_rate = beta + _psi(alpha, r, 0, pilot_days)
        unseen_users_term = (
            (observed_users + c + 1.0)
            * alpha
            * r
            * horizons
            * beta_function(1.0 - alpha, r * pilot_days)
            / posterior_rate
        )
        observed_users_term = (horizons / pilot_days) * (
            pilot_total_triggers - alpha * observed_users
        )
        return unseen_users_term + observed_users_term

    @staticmethod
    def fit_curve(
        cumulative_users: Sequence[float] | np.ndarray,
        *,
        start_day: int = 1,
        num_restarts: int = 5,
        norm: float = 2,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Fit ``(alpha, c, beta, r)`` by the paper's curve criterion."""
        observed = _validate_cumulative_users(cumulative_users)
        start_day = _require_nonnegative_integer(start_day, name="start_day")
        if start_day == 0:
            raise ValueError("start_day must be positive")
        if start_day >= observed.size - 1:
            raise ValueError(
                "cumulative_users must include at least one day after start_day"
            )
        if not np.isscalar(norm) or not np.isfinite(norm) or norm <= 0:
            raise ValueError("norm must be a positive finite scalar")

        followup_days = observed.size - start_day - 1
        anchor = int(observed[start_day])
        target = observed[start_day + 1 :]

        def objective(parameters: np.ndarray) -> float:
            alpha, c, beta, r = parameters
            fitted = anchor + NBSSP.expected_new_users(
                start_day,
                followup_days,
                anchor,
                alpha=alpha,
                c=c,
                beta=beta,
                r=r,
            )
            return float(np.linalg.norm(fitted - target, ord=norm))

        return fit_differential_evolution(
            objective,
            _CURVE_BOUNDS,
            _PARAMETER_NAMES,
            num_restarts=num_restarts,
            seed=seed,
        )

    @staticmethod
    def fit_marginal_likelihood(
        activity_matrix: np.ndarray,
        *,
        num_restarts: int = 5,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Fit the NB-SSP using the exact marginal likelihood in Theorem 4.1."""
        matrix = _validate_activity_matrix(activity_matrix)
        pilot_days, observed_users = matrix.shape
        user_totals = matrix.sum(axis=0)

        def objective(parameters: np.ndarray) -> float:
            alpha, c, beta, r = parameters
            psi_pilot = _psi(alpha, r, 0, pilot_days)

            log_likelihood = observed_users * np.log(alpha)
            log_likelihood += (c + 1.0) * np.log(beta)
            log_likelihood += gammaln(observed_users + c + 1.0)
            log_likelihood -= gammaln(c + 1.0)
            log_likelihood -= (observed_users + c + 1.0) * np.log(
                beta + psi_pilot
            )

            log_likelihood += np.sum(
                gammaln(matrix + r) - gammaln(matrix + 1.0) - gammaln(r)
            )
            log_likelihood += np.sum(
                betaln(user_totals - alpha, r * pilot_days + 1.0)
            )
            return -float(log_likelihood)

        return fit_differential_evolution(
            objective,
            _MARGINAL_LIKELIHOOD_BOUNDS,
            _PARAMETER_NAMES,
            num_restarts=num_restarts,
            seed=seed,
        )
