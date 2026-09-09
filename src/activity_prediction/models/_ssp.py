"""Shared posterior-predictive calculations for Be-SSP and TG-SSP."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral, Real

import numpy as np
from scipy.special import betaln
from scipy.stats import nbinom


PARAMETER_NAMES = ("alpha", "c", "beta")
PARAMETER_BOUNDS = (
    (1e-6, 1.0 - 1e-6),
    (1e-8, 1e4),
    (1e-8, 1e4),
)


def validate_parameters(*, alpha: float, c: float, beta: float) -> None:
    """Validate the SB-SP hyperparameters in the paper's notation."""
    values = {"alpha": alpha, "c": c, "beta": beta}
    for name, value in values.items():
        if not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite real number")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if c <= 0.0:
        raise ValueError("c must be strictly positive")
    if beta <= 0.0:
        raise ValueError("beta must be strictly positive")


def _validate_horizon(name: str, value: int, *, allow_zero: bool) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    lower_bound = 0 if allow_zero else 1
    if value < lower_bound:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _validate_observed_users(observed_users: int) -> int:
    return _validate_horizon("observed_users", observed_users, allow_zero=True)


def psi(start_day: int, num_days: int, *, alpha: float) -> float:
    """Return ``psi_1(start_day, num_days)`` from the paper.

    The positive beta-function sum avoids evaluating ``B(x, -alpha)`` by
    analytic continuation:

    ``psi_1(x, y) = alpha * sum(B(1 - alpha, d), d=x+1,...,x+y)``.
    """
    start_day = _validate_horizon("start_day", start_day, allow_zero=True)
    num_days = _validate_horizon("num_days", num_days, allow_zero=True)
    if not isinstance(alpha, Real) or not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if num_days == 0:
        return 0.0
    days = np.arange(start_day + 1, start_day + num_days + 1, dtype=float)
    return float(alpha * np.exp(betaln(1.0 - alpha, days)).sum())


def expected_new_users(
    pilot_days: int,
    followup_days: int,
    observed_users: int,
    *,
    alpha: float,
    c: float,
    beta: float,
) -> np.ndarray:
    """Posterior means of new users at horizons 1 through ``followup_days``."""
    validate_parameters(alpha=alpha, c=c, beta=beta)
    pilot_days = _validate_horizon("pilot_days", pilot_days, allow_zero=True)
    followup_days = _validate_horizon("followup_days", followup_days, allow_zero=False)
    observed_users = _validate_observed_users(observed_users)

    days = np.arange(1, pilot_days + followup_days + 1, dtype=float)
    terms = alpha * np.exp(betaln(1.0 - alpha, days))
    cumulative = np.cumsum(terms)
    psi_pilot = cumulative[pilot_days - 1] if pilot_days else 0.0
    psi_followup = cumulative[pilot_days:] - psi_pilot
    return (observed_users + c + 1.0) * psi_followup / (beta + psi_pilot)


def new_user_interval(
    pilot_days: int,
    followup_days: int,
    observed_users: int,
    *,
    coverage: float,
    alpha: float,
    c: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Centered NB posterior-predictive intervals for each follow-up horizon."""
    validate_parameters(alpha=alpha, c=c, beta=beta)
    pilot_days = _validate_horizon("pilot_days", pilot_days, allow_zero=True)
    followup_days = _validate_horizon("followup_days", followup_days, allow_zero=False)
    observed_users = _validate_observed_users(observed_users)
    if not isinstance(coverage, Real) or not np.isfinite(coverage) or not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie strictly between 0 and 1")

    days = np.arange(1, pilot_days + followup_days + 1, dtype=float)
    terms = alpha * np.exp(betaln(1.0 - alpha, days))
    cumulative = np.cumsum(terms)
    psi_pilot = cumulative[pilot_days - 1] if pilot_days else 0.0
    psi_followup = cumulative[pilot_days:] - psi_pilot
    success_probability = (beta + psi_pilot) / (
        beta + psi_pilot + psi_followup
    )
    lower, upper = nbinom.interval(
        coverage,
        n=observed_users + c + 1.0,
        p=success_probability,
    )
    return np.asarray(lower), np.asarray(upper)


def curve_fitting_objective(
    cumulative_users: np.ndarray,
    *,
    start_day: int,
    norm: float,
) -> Callable[[np.ndarray], float]:
    """Build the paper's curve-fitting loss in ``(alpha, c, beta)`` order."""
    curve = np.asarray(cumulative_users, dtype=float)
    if curve.ndim != 1:
        raise ValueError("cumulative_users must be one-dimensional")
    if len(curve) < 3:
        raise ValueError("cumulative_users must contain day 0 and at least two observed days")
    if not np.all(np.isfinite(curve)):
        raise ValueError("cumulative_users must contain only finite values")
    if np.any(curve < 0.0) or np.any(np.diff(curve) < 0.0):
        raise ValueError("cumulative_users must be non-negative and non-decreasing")
    start_day = _validate_horizon("start_day", start_day, allow_zero=True)
    if start_day >= len(curve) - 1:
        raise ValueError("start_day must precede at least one observed follow-up day")
    if not isinstance(norm, Real) or (not np.isinf(norm) and (not np.isfinite(norm) or norm <= 0.0)):
        raise ValueError("norm must be positive or infinity")

    observed_users = int(curve[start_day])
    observed_increments = curve[start_day + 1 :] - curve[start_day]
    followup_days = len(observed_increments)

    def objective(parameters: np.ndarray) -> float:
        alpha, c, beta = parameters
        prediction = expected_new_users(
            start_day,
            followup_days,
            observed_users,
            alpha=alpha,
            c=c,
            beta=beta,
        )
        return float(np.linalg.norm(prediction - observed_increments, ord=norm))

    return objective


__all__ = [
    "PARAMETER_BOUNDS",
    "PARAMETER_NAMES",
    "curve_fitting_objective",
    "expected_new_users",
    "new_user_interval",
    "psi",
    "validate_parameters",
]
