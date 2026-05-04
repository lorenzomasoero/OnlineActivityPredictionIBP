"""
Bayesian nonparametric models for online activity prediction.

Models:
    GD (TG-SSP)      — Tilted Gibbs Stable-Beta Scaled Process
    IBP               — Indian Buffet Process
    NegBintSBSP (NB-SSP) — Negative Binomial Stable-Beta Scaled Process
    BB                — Beta-Binomial

Estimators:
    predict_jack      — Jackknife estimator
    predict_gt        — Good-Toulmin estimator
"""

from .tg_ssp import GD
from .ibp import IBP
from .nb_ssp import NegBintSBSP
from .beta_binomial import BB
from .jackknife import predict_jack, missed_jack
from .good_toulmin import predict_gt

__all__ = [
    'GD', 'IBP', 'NegBintSBSP', 'BB',
    'predict_jack', 'missed_jack', 'predict_gt',
]
