"""Private optimization helpers shared by the paper models."""

from collections.abc import Callable, Sequence

import numpy as np
from scipy import optimize


def fit_differential_evolution(
    objective: Callable[[np.ndarray], float],
    bounds: Sequence[tuple[float, float]],
    parameter_names: Sequence[str],
    *,
    num_restarts: int = 5,
    seed: int | None = None,
) -> dict[str, float]:
    """Minimize ``objective`` and return the best parameters by name.

    A small, explicit wrapper replaces the repeated optimization loops in the
    legacy model classes.  Independent seeds are generated for each restart so
    a supplied seed makes the complete fit reproducible.
    """

    if num_restarts < 1:
        raise ValueError("num_restarts must be at least one")
    if len(bounds) != len(parameter_names):
        raise ValueError("bounds and parameter_names must have the same length")

    generator = np.random.default_rng(seed)
    best_result: optimize.OptimizeResult | None = None

    for _ in range(num_restarts):
        restart_seed = int(generator.integers(0, np.iinfo(np.int32).max))
        result = optimize.differential_evolution(
            objective,
            bounds,
            seed=restart_seed,
            polish=True,
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    assert best_result is not None  # num_restarts is validated above
    return {
        name: float(value)
        for name, value in zip(parameter_names, best_result.x, strict=True)
    }
