"""Evaluation metrics reported in the paper."""

import numpy as np


def prediction_accuracy(observed: float, predicted: float) -> float:
    """Return the paper's clipped relative-error accuracy ``v``."""

    if not np.isfinite(observed) or not np.isfinite(predicted):
        raise ValueError("observed and predicted must be finite")
    if observed <= 0:
        raise ValueError("observed must be positive")
    return float(1.0 - min(abs(observed - predicted) / observed, 1.0))
