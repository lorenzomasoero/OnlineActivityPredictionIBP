"""Paper-aligned activity-prediction models.

Each public class corresponds to exactly one model name used in the paper.
Legacy implementation names such as ``GD`` and ``NegBintSBSP`` are
intentionally not exported.
"""

from .be_ssp import BeSSP
from .beta_binomial import BetaBinomial
from .good_toulmin import predict_good_toulmin
from .hierarchical_beta_geometric import HierarchicalBetaGeometric
from .ibp import IBP
from .jackknife import predict_jackknife
from .nb_ssp import NBSSP
from .tg_ssp import TGSSP
from .unseen_est import predict_unseen_est

__all__ = [
    "BeSSP",
    "TGSSP",
    "NBSSP",
    "IBP",
    "BetaBinomial",
    "HierarchicalBetaGeometric",
    "predict_good_toulmin",
    "predict_jackknife",
    "predict_unseen_est",
]
