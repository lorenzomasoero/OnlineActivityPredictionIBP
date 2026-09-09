"""Beta-binomial (BB) unseen-user predictor used as a paper benchmark."""

import numpy as np
from scipy.special import betaln, binom, gammaln

from ._optimization import fit_differential_evolution


def _log_pochhammer(value: float, order: np.ndarray) -> np.ndarray:
    return gammaln(value + order) - gammaln(value)


class BetaBinomial:
    """Beta-binomial model of Ionita-Laza et al. (2009)."""

    def expected_new_users(
        self,
        sample_size: int,
        followup_size: int,
        singletons: int,
        *,
        alpha: float,
        beta: float,
    ) -> np.ndarray:
        """Return expected unseen users after each additional observation."""

        if sample_size < 1 or followup_size < 0 or singletons < 0:
            raise ValueError("sample sizes must be positive and singletons nonnegative")
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        if followup_size == 0:
            return np.empty(0, dtype=float)

        steps = np.arange(1, followup_size + 1)
        survival_ratio = np.exp(
            _log_pochhammer(beta + sample_size, steps)
            - _log_pochhammer(alpha + beta + sample_size, steps)
        )
        scale = singletons * (beta + sample_size - 1) / (alpha * sample_size)
        return scale * (1 - survival_ratio)

    def fit_marginal_likelihood(
        self,
        sfs: np.ndarray,
        sample_size: int,
        *,
        num_restarts: int = 5,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Fit ``alpha`` and ``beta`` from the site-frequency spectrum."""

        spectrum = np.asarray(sfs, dtype=float)
        if sample_size < 1 or spectrum.ndim != 1 or np.any(spectrum < 0):
            raise ValueError("sfs must be a nonnegative vector and sample_size positive")
        if len(spectrum) > sample_size:
            raise ValueError("sfs cannot be longer than sample_size")

        frequencies = np.arange(1, sample_size + 1)

        def objective(values: np.ndarray) -> float:
            alpha, beta = values
            log_probabilities = (
                np.log(binom(sample_size, frequencies))
                + betaln(frequencies + alpha, sample_size - frequencies + beta)
                - betaln(alpha, beta)
            )
            log_probabilities -= np.logaddexp.reduce(log_probabilities)
            return -float(np.inner(spectrum, log_probabilities[: len(spectrum)]))

        return fit_differential_evolution(
            objective,
            ((1e-5, 1_000.0), (1e-5, 1_000.0)),
            ("alpha", "beta"),
            num_restarts=num_restarts,
            seed=seed,
        )
