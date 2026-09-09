"""Model-fitting entry points for public daily-activity datasets.

The functions are deliberately data-driven and perform no file I/O.  Dataset
scripts load public inputs, call one of these functions, and save the returned
records.  Hyperparameters are dictionaries using the notation in the paper.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from ..models.be_ssp import BeSSP
from ..models.beta_binomial import BetaBinomial
from ..models.good_toulmin import predict_good_toulmin
from ..models.hierarchical_beta_geometric import HierarchicalBetaGeometric
from ..models.ibp import IBP
from ..models.jackknife import predict_jackknife
from ..models.nb_ssp import NBSSP
from ..models.tg_ssp import TGSSP
from ..models.unseen_est import predict_unseen_est


def _validate_curve(cumulative_users: np.ndarray) -> np.ndarray:
    curve = np.asarray(cumulative_users, dtype=float)
    if curve.ndim != 1 or len(curve) < 3:
        raise ValueError("cumulative_users must contain day zero and two pilot days")
    if np.any(~np.isfinite(curve)) or np.any(curve < 0) or np.any(np.diff(curve) < 0):
        raise ValueError("cumulative_users must be finite, nonnegative, and nondecreasing")
    if np.any(curve != np.floor(curve)) or curve[0] != 0:
        raise ValueError("cumulative_users must be integer-valued and start at zero")
    return curve.astype(int)


def _validate_matrix(activity_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(activity_matrix)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("activity_matrix must be a nonempty days-by-users matrix")
    if not np.issubdtype(matrix.dtype, np.number) or np.any(~np.isfinite(matrix)):
        raise ValueError("activity_matrix must contain finite numeric counts")
    if np.any(matrix < 0) or np.any(matrix != np.floor(matrix)):
        raise ValueError("activity_matrix must contain nonnegative integer counts")
    return matrix.astype(int, copy=False)


def _curve_from_matrix(matrix: np.ndarray) -> np.ndarray:
    active = np.cumsum(matrix > 0, axis=0) > 0
    return np.concatenate(([0], np.count_nonzero(active, axis=1)))


def _prediction_record(
    parameters: dict[str, float],
    expected: np.ndarray,
    interval: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "parameters": parameters,
        "expected_new_users": np.asarray(expected, dtype=float),
    }
    if interval is not None:
        record["new_user_interval"] = tuple(
            np.asarray(endpoint, dtype=float) for endpoint in interval
        )
    return record


def fit_first_trigger_models(
    cumulative_users: np.ndarray,
    followup_days: int,
    *,
    coverage: float = 0.95,
    num_restarts: int = 5,
    seed: int | None = None,
) -> dict[str, dict[str, object]]:
    """Fit the models available when only first-trigger counts are retained.

    TG-SSP uses its first-trigger marginal likelihood, as stated for ASOS in
    the paper.  IBP and NB-SSP use curve fitting because their individual-user
    sufficient statistics are unavailable.
    """

    curve = _validate_curve(cumulative_users)
    if followup_days < 1:
        raise ValueError("followup_days must be positive")
    pilot_days = len(curve) - 1
    observed_users = int(curve[-1])
    first_trigger_counts = np.diff(curve)

    tg = TGSSP()
    tg_parameters = tg.fit_marginal_likelihood(
        first_trigger_counts, num_restarts=num_restarts, seed=seed
    )
    tg_expected = tg.expected_new_users(
        pilot_days, followup_days, observed_users, **tg_parameters
    )

    ibp = IBP()
    ibp_parameters = ibp.fit_curve(
        curve, num_restarts=num_restarts, seed=seed
    )
    ibp_expected = ibp.expected_new_users(
        pilot_days, followup_days, **ibp_parameters
    )

    nb = NBSSP()
    nb_parameters = nb.fit_curve(curve, num_restarts=num_restarts, seed=seed)
    nb_expected = nb.expected_new_users(
        pilot_days, followup_days, observed_users, **nb_parameters
    )

    return {
        "TG-SSP": _prediction_record(
            tg_parameters,
            tg_expected,
            tg.new_user_interval(
                pilot_days,
                followup_days,
                observed_users,
                coverage=coverage,
                **tg_parameters,
            ),
        ),
        "IBP": _prediction_record(
            ibp_parameters,
            ibp_expected,
            poisson.interval(coverage, ibp_expected),
        ),
        "NB-SSP (curve)": _prediction_record(
            nb_parameters,
            nb_expected,
            nb.new_user_interval(
                pilot_days,
                followup_days,
                observed_users,
                coverage=coverage,
                **nb_parameters,
            ),
        ),
    }


def fit_daily_activity_models(
    pilot_activity_matrix: np.ndarray,
    followup_days: int,
    *,
    coverage: float = 0.95,
    num_restarts: int = 5,
    seed: int | None = None,
    include_hbg: bool = False,
    include_lp: bool = False,
) -> dict[str, dict[str, object]]:
    """Fit paper models and benchmarks when per-user daily counts exist."""

    matrix = _validate_matrix(pilot_activity_matrix)
    if followup_days < 1:
        raise ValueError("followup_days must be positive")

    pilot_days = matrix.shape[0]
    active_columns = matrix.sum(axis=0) > 0
    active_matrix = matrix[:, active_columns]
    if active_matrix.shape[1] == 0:
        raise ValueError("pilot_activity_matrix has no active users")

    binary_matrix = (active_matrix > 0).astype(int)
    observed_users = active_matrix.shape[1]
    curve = _curve_from_matrix(binary_matrix)
    first_trigger_counts = np.diff(curve)
    active_days = binary_matrix.sum(axis=0)
    sfs = np.bincount(active_days, minlength=pilot_days + 1)[1:]

    be = BeSSP()
    be_parameters = be.fit_marginal_likelihood(
        sfs, pilot_days, num_restarts=num_restarts, seed=seed
    )
    be_expected = be.expected_new_users(
        pilot_days, followup_days, observed_users, **be_parameters
    )
    be_curve_parameters = be.fit_curve(
        curve, num_restarts=num_restarts, seed=seed
    )
    be_curve_expected = be.expected_new_users(
        pilot_days, followup_days, observed_users, **be_curve_parameters
    )

    tg = TGSSP()
    tg_parameters = tg.fit_marginal_likelihood(
        first_trigger_counts, num_restarts=num_restarts, seed=seed
    )
    tg_expected = tg.expected_new_users(
        pilot_days, followup_days, observed_users, **tg_parameters
    )
    tg_curve_parameters = tg.fit_curve(
        curve, num_restarts=num_restarts, seed=seed
    )
    tg_curve_expected = tg.expected_new_users(
        pilot_days, followup_days, observed_users, **tg_curve_parameters
    )

    nb = NBSSP()
    nb_parameters = nb.fit_marginal_likelihood(
        active_matrix, num_restarts=num_restarts, seed=seed
    )
    nb_expected = nb.expected_new_users(
        pilot_days, followup_days, observed_users, **nb_parameters
    )
    nb_curve_parameters = nb.fit_curve(
        curve, num_restarts=num_restarts, seed=seed
    )
    nb_curve_expected = nb.expected_new_users(
        pilot_days, followup_days, observed_users, **nb_curve_parameters
    )

    ibp = IBP()
    ibp_parameters = ibp.fit_marginal_likelihood(
        sfs, pilot_days, num_restarts=num_restarts, seed=seed
    )
    ibp_expected = ibp.expected_new_users(
        pilot_days, followup_days, **ibp_parameters
    )
    ibp_curve_parameters = ibp.fit_curve(
        curve, num_restarts=num_restarts, seed=seed
    )
    ibp_curve_expected = ibp.expected_new_users(
        pilot_days, followup_days, **ibp_curve_parameters
    )

    bb = BetaBinomial()
    bb_parameters = bb.fit_marginal_likelihood(
        sfs, pilot_days, num_restarts=num_restarts, seed=seed
    )
    bb_expected = bb.expected_new_users(
        pilot_days,
        followup_days,
        int(sfs[0]),
        **bb_parameters,
    )

    results = {
        "Be-SSP": _prediction_record(
            be_parameters,
            be_expected,
            be.new_user_interval(
                pilot_days,
                followup_days,
                observed_users,
                coverage=coverage,
                **be_parameters,
            ),
        ),
        "Be-SSP (curve)": _prediction_record(
            be_curve_parameters,
            be_curve_expected,
            be.new_user_interval(
                pilot_days,
                followup_days,
                observed_users,
                coverage=coverage,
                **be_curve_parameters,
            ),
        ),
        "TG-SSP": _prediction_record(
            tg_parameters,
            tg_expected,
            tg.new_user_interval(
                pilot_days,
                followup_days,
                observed_users,
                coverage=coverage,
                **tg_parameters,
            ),
        ),
        "TG-SSP (curve)": _prediction_record(
            tg_curve_parameters,
            tg_curve_expected,
            tg.new_user_interval(
                pilot_days,
                followup_days,
                observed_users,
                coverage=coverage,
                **tg_curve_parameters,
            ),
        ),
        "NB-SSP": _prediction_record(
            nb_parameters,
            nb_expected,
            nb.new_user_interval(
                pilot_days,
                followup_days,
                observed_users,
                coverage=coverage,
                **nb_parameters,
            ),
        ),
        "NB-SSP (curve)": _prediction_record(
            nb_curve_parameters,
            nb_curve_expected,
            nb.new_user_interval(
                pilot_days,
                followup_days,
                observed_users,
                coverage=coverage,
                **nb_curve_parameters,
            ),
        ),
        "IBP": _prediction_record(
            ibp_parameters,
            ibp_expected,
            poisson.interval(coverage, ibp_expected),
        ),
        "IBP (curve)": _prediction_record(
            ibp_curve_parameters,
            ibp_curve_expected,
            poisson.interval(coverage, ibp_curve_expected),
        ),
        "BB": _prediction_record(bb_parameters, bb_expected),
    }
    results["NB-SSP"]["expected_total_triggers"] = nb.expected_total_triggers(
        pilot_days,
        followup_days,
        observed_users,
        int(active_matrix.sum()),
        **nb_parameters,
    )

    good_toulmin, good_toulmin_variance = predict_good_toulmin(
        pilot_days, followup_days, sfs, curve
    )
    results["GT"] = {
        "cumulative_users": good_toulmin,
        "variance": good_toulmin_variance,
    }
    for order in range(1, 5):
        results[f"J{order}"] = {
            "cumulative_users": predict_jackknife(
                pilot_days, followup_days, sfs, curve, order=order
            )
        }

    if include_lp:
        results["LP"] = {
            "cumulative_users": predict_unseen_est(
                pilot_days, followup_days, sfs
            )
        }
    if include_hbg:
        results["HBG"] = {
            "cumulative_users": HierarchicalBetaGeometric().predict_cumulative_users(
                first_trigger_counts,
                followup_days,
                seed=seed,
            )
        }
    return results
