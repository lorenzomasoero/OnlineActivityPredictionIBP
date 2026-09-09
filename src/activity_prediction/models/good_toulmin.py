"""Smoothed Good--Toulmin (GT) unseen-user estimator."""

import numpy as np
from scipy.stats import binom


def _missed_good_toulmin(
    sample_size: int,
    followup_size: int,
    sfs: np.ndarray,
) -> tuple[float, float]:
    signed_spectrum = (-1) ** np.arange(len(sfs)) * sfs
    ratio = followup_size / sample_size
    powers = ratio ** np.arange(1, len(sfs) + 1)

    if followup_size <= sample_size:
        return float(np.sum(signed_spectrum * powers)), float(np.sum(sfs * powers**2))

    truncation = int(0.5 * np.log(sample_size * ratio**2 / (ratio - 1)) / np.log(3))
    smoothing_probability = 2 / (ratio + 1)
    weights = 1 - binom.cdf(np.arange(len(sfs)), truncation, smoothing_probability)
    estimate = np.sum(signed_spectrum * powers * weights)
    variance = np.sum(np.abs(signed_spectrum) * powers**2 * weights**2)
    return float(estimate), float(variance)


def predict_good_toulmin(
    sample_size: int,
    followup_size: int,
    sfs: np.ndarray,
    cumulative_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative predictions and variances through the follow-up."""

    spectrum = np.asarray(sfs, dtype=float)
    counts = np.asarray(cumulative_counts, dtype=float)
    if sample_size < 1 or followup_size < 0:
        raise ValueError("sample_size must be positive and followup_size nonnegative")
    if spectrum.ndim != 1 or np.any(spectrum < 0) or len(spectrum) > sample_size:
        raise ValueError("sfs must be a nonnegative vector no longer than sample_size")
    if counts.ndim != 1 or len(counts) < sample_size + 1:
        raise ValueError("cumulative_counts must include days zero through sample_size")

    predictions = np.zeros(sample_size + followup_size + 1)
    variances = np.zeros_like(predictions)
    predictions[: sample_size + 1] = counts[: sample_size + 1]
    estimates = [
        _missed_good_toulmin(sample_size, step, spectrum)
        for step in range(1, followup_size + 1)
    ]
    predictions[sample_size + 1 :] = counts[sample_size] + [x[0] for x in estimates]
    variances[sample_size + 1 :] = [x[1] for x in estimates]
    return predictions, variances
