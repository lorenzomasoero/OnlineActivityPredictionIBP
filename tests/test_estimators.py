import numpy as np

from activity_prediction.models import BetaBinomial, predict_good_toulmin, predict_jackknife


def test_beta_binomial_prediction_has_the_requested_horizon():
    prediction = BetaBinomial().expected_new_users(
        7,
        5,
        40,
        alpha=0.7,
        beta=3.0,
    )
    assert prediction.shape == (5,)
    assert np.all(np.diff(prediction) >= 0)


def test_jackknife_orders_and_good_toulmin_return_full_curves():
    sample_size = 7
    followup_size = 3
    sfs = np.array([30, 12, 6, 3, 1, 0, 0])
    observed = np.array([0, 20, 31, 39, 45, 49, 51, 52])

    for order in range(1, 5):
        prediction = predict_jackknife(
            sample_size,
            followup_size,
            sfs,
            observed,
            order=order,
        )
        assert prediction.shape == (sample_size + followup_size + 1,)
        np.testing.assert_array_equal(prediction[: sample_size + 1], observed)

    prediction, variance = predict_good_toulmin(
        sample_size,
        followup_size,
        sfs,
        observed,
    )
    assert prediction.shape == variance.shape == (sample_size + followup_size + 1,)
