import numpy as np

from activity_prediction.experiments import (
    first_hitting_day,
    model_hitting_day,
    prediction_accuracy,
)


def test_first_hitting_day_returns_first_crossing_or_none():
    curve = np.array([0, 4, 7, 11, 15])
    assert first_hitting_day(curve, 10) == 3
    assert first_hitting_day(curve, 20) is None


def test_model_hitting_day_uses_paper_model_names():
    day = model_hitting_day(
        "TG-SSP",
        {"alpha": 0.5, "c": 10.0, "beta": 2.0},
        pilot_days=7,
        max_followup_days=100,
        observed_users=100,
        participation_multiplier=1.5,
    )
    assert day is None or day > 7


def test_prediction_accuracy_matches_the_paper_definition():
    assert prediction_accuracy(100, 100) == 1.0
    assert prediction_accuracy(100, 80) == 0.8
    assert prediction_accuracy(100, 250) == 0.0
