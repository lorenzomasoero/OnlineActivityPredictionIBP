"""Three-parameter Indian buffet process model used in the paper."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import gammaln

from ._optimization import fit_differential_evolution


_PARAMETER_NAMES = ("mass", "concentration", "discount")
_CURVE_BOUNDS = (
    (1e-8, 1e5),
    (0.0, 10.0),
    (1e-5, 0.99999),
)
_MARGINAL_LIKELIHOOD_BOUNDS = (
    (1e-8, 1e5),
    (0.0, 1e3),
    (1e-5, 0.9999),
)


def _require_nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _log_rising_factorial(value: float, order: np.ndarray | int) -> np.ndarray:
    """Return ``log((value)_order)`` for non-negative integer orders."""
    return gammaln(value + order) - gammaln(value)


def _validate_parameters(
    *, mass: float, concentration: float, discount: float
) -> tuple[float, float, float]:
    parameters = np.asarray((mass, concentration, discount), dtype=float)
    if not np.all(np.isfinite(parameters)):
        raise ValueError("IBP parameters must be finite")

    mass, concentration, discount = map(float, parameters)
    if mass <= 0:
        raise ValueError("mass must be positive")
    if not 0 <= discount < 1:
        raise ValueError("discount must lie in [0, 1)")
    if concentration <= -discount:
        raise ValueError("concentration must be greater than -discount")
    return mass, concentration, discount


class IBP:
    """The paper's three-parameter Indian buffet process competitor."""

    @staticmethod
    def expected_new_users(
        pilot_days: int,
        followup_days: int,
        *,
        mass: float,
        concentration: float,
        discount: float,
    ) -> np.ndarray:
        """Return cumulative expected new users over the follow-up period."""
        pilot_days = _require_nonnegative_integer(pilot_days, name="pilot_days")
        followup_days = _require_nonnegative_integer(
            followup_days, name="followup_days"
        )
        mass, concentration, discount = _validate_parameters(
            mass=mass,
            concentration=concentration,
            discount=discount,
        )

        if followup_days == 0:
            return np.empty(0, dtype=float)

        orders = np.arange(pilot_days, pilot_days + followup_days)
        daily_means = mass * np.exp(
            _log_rising_factorial(concentration + discount, orders)
            - _log_rising_factorial(concentration + 1.0, orders)
        )
        return np.cumsum(daily_means)

    @staticmethod
    def fit_curve(
        cumulative_users: Sequence[float] | np.ndarray,
        *,
        start_day: int = 1,
        num_restarts: int = 5,
        norm: float = 2,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Fit the IBP mean curve to cumulative distinct-user counts."""
        observed = np.asarray(cumulative_users, dtype=float)
        if observed.ndim != 1:
            raise ValueError("cumulative_users must be one-dimensional")
        if not np.all(np.isfinite(observed)):
            raise ValueError("cumulative_users must contain only finite values")
        if np.any(observed < 0):
            raise ValueError("cumulative_users must be non-negative")
        if np.any(np.diff(observed) < 0):
            raise ValueError("cumulative_users must be non-decreasing")

        start_day = _require_nonnegative_integer(start_day, name="start_day")
        if start_day >= observed.size - 1:
            raise ValueError(
                "cumulative_users must include at least one day after start_day"
            )
        if not np.isscalar(norm) or not np.isfinite(norm) or norm <= 0:
            raise ValueError("norm must be a positive finite scalar")

        followup_days = observed.size - start_day - 1
        anchor = observed[start_day]
        target = observed[start_day + 1 :]

        def objective(parameters: np.ndarray) -> float:
            mass, concentration, discount = parameters
            fitted = anchor + IBP.expected_new_users(
                start_day,
                followup_days,
                mass=mass,
                concentration=concentration,
                discount=discount,
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
        sfs: Sequence[int] | np.ndarray,
        sample_size: int,
        *,
        num_restarts: int = 5,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Fit the IBP parameters from a site-frequency spectrum."""
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
        frequencies = frequencies.astype(float, copy=False)
        if frequencies.sum() <= 0:
            raise ValueError("sfs must contain at least one observed user")

        sample_size = _require_nonnegative_integer(sample_size, name="sample_size")
        if sample_size == 0:
            raise ValueError("sample_size must be positive")
        if frequencies.size > sample_size:
            raise ValueError("sfs cannot have more entries than sample_size")

        number_observed = float(frequencies.sum())
        spectrum_orders = np.arange(frequencies.size)
        reverse_sample_orders = np.arange(
            sample_size - frequencies.size, sample_size
        )[::-1]
        sample_orders = np.arange(sample_size)

        def objective(parameters: np.ndarray) -> float:
            mass, concentration, discount = parameters
            log_likelihood = number_observed * (
                np.log(mass)
                - _log_rising_factorial(concentration + 1.0, sample_size - 1)
            )
            log_likelihood -= mass * np.exp(
                _log_rising_factorial(concentration + discount, sample_orders)
                - _log_rising_factorial(concentration + 1.0, sample_orders)
            ).sum()
            log_likelihood += np.inner(
                frequencies,
                _log_rising_factorial(1.0 - discount, spectrum_orders),
            )
            log_likelihood += np.inner(
                frequencies,
                _log_rising_factorial(
                    concentration + discount, reverse_sample_orders
                ),
            )
            return -float(log_likelihood)

        return fit_differential_evolution(
            objective,
            _MARGINAL_LIKELIHOOD_BOUNDS,
            _PARAMETER_NAMES,
            num_restarts=num_restarts,
            seed=seed,
        )
