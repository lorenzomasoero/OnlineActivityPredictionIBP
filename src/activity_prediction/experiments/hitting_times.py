"""Duration-to-target calculations used in the paper."""

from collections.abc import Mapping
from math import ceil

import numpy as np

from ..models.ibp import IBP
from ..models.nb_ssp import NBSSP
from ..models.tg_ssp import TGSSP


def first_hitting_day(cumulative_users: np.ndarray, target_users: int) -> int | None:
    """Return the first observed day reaching ``target_users``."""

    curve = np.asarray(cumulative_users, dtype=float)
    if curve.ndim != 1 or len(curve) == 0 or np.any(np.diff(curve) < 0):
        raise ValueError("cumulative_users must be a nonempty nondecreasing vector")
    if target_users < 0:
        raise ValueError("target_users must be nonnegative")
    matches = np.flatnonzero(curve >= target_users)
    return int(matches[0]) if len(matches) else None


def model_hitting_day(
    model_name: str,
    parameters: Mapping[str, float],
    *,
    pilot_days: int,
    max_followup_days: int,
    observed_users: int,
    participation_multiplier: float,
) -> int | None:
    """Return the model mean-trajectory hitting day used in MAE analyses."""

    if participation_multiplier <= 1:
        raise ValueError("participation_multiplier must exceed one")
    target = ceil(participation_multiplier * observed_users)

    if model_name == "TG-SSP":
        expected = TGSSP().expected_new_users(
            pilot_days, max_followup_days, observed_users, **parameters
        )
    elif model_name in {"NB-SSP", "NB-SSP (curve)"}:
        expected = NBSSP().expected_new_users(
            pilot_days, max_followup_days, observed_users, **parameters
        )
    elif model_name == "IBP":
        expected = IBP().expected_new_users(
            pilot_days, max_followup_days, **parameters
        )
    else:
        raise ValueError(f"unsupported model_name: {model_name}")

    crossing = np.flatnonzero(observed_users + expected >= target)
    return pilot_days + int(crossing[0]) + 1 if len(crossing) else None
