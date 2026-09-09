import numpy as np
from scipy.special import beta as beta_function
from scipy.special import gammaln

from activity_prediction.models import BeSSP, IBP, NBSSP, TGSSP


def test_be_ssp_and_tg_ssp_share_the_fixed_parameter_prediction():
    parameters = {"alpha": 0.45, "c": 12.0, "beta": 3.0}
    be_prediction = BeSSP().expected_new_users(7, 14, 250, **parameters)
    tg_prediction = TGSSP().expected_new_users(7, 14, 250, **parameters)
    np.testing.assert_allclose(be_prediction, tg_prediction)


def test_nb_ssp_with_r_one_has_the_same_new_user_mean():
    parameters = {"alpha": 0.45, "c": 12.0, "beta": 3.0}
    shared_prediction = TGSSP().expected_new_users(7, 14, 250, **parameters)
    nb_prediction = NBSSP().expected_new_users(7, 14, 250, r=1.0, **parameters)
    np.testing.assert_allclose(nb_prediction, shared_prediction, rtol=1e-11)


def test_nb_ssp_total_trigger_mean_matches_the_paper_formula():
    alpha, c, beta, r = 0.4, 8.0, 2.5, 3.0
    pilot_days, observed_users, pilot_total = 7, 120, 420
    horizons = np.arange(1, 6, dtype=float)
    psi_pilot = alpha * (
        beta_function(1.0, -alpha)
        - beta_function(r * pilot_days + 1.0, -alpha)
    )
    expected = (
        (observed_users + c + 1)
        * alpha
        * r
        * horizons
        * beta_function(1 - alpha, r * pilot_days)
        / (beta + psi_pilot)
        + horizons / pilot_days * (pilot_total - alpha * observed_users)
    )
    actual = NBSSP().expected_total_triggers(
        pilot_days,
        len(horizons),
        observed_users,
        pilot_total,
        alpha=alpha,
        c=c,
        beta=beta,
        r=r,
    )
    np.testing.assert_allclose(actual, expected)


def test_ibp_expected_new_users_matches_rising_factorial_expression():
    mass, concentration, discount = 4.0, 2.0, 0.3
    pilot_days, followup_days = 5, 4
    orders = np.arange(pilot_days, pilot_days + followup_days)
    increments = mass * np.exp(
        gammaln(concentration + discount + orders)
        - gammaln(concentration + discount)
        - gammaln(concentration + 1 + orders)
        + gammaln(concentration + 1)
    )
    actual = IBP().expected_new_users(
        pilot_days,
        followup_days,
        mass=mass,
        concentration=concentration,
        discount=discount,
    )
    np.testing.assert_allclose(actual, np.cumsum(increments))


def test_predictive_intervals_have_one_entry_per_followup_day():
    parameters = {"alpha": 0.5, "c": 10.0, "beta": 2.0}
    for model in (BeSSP(), TGSSP()):
        lower, upper = model.new_user_interval(7, 6, 100, **parameters)
        assert lower.shape == upper.shape == (6,)
        assert np.all(lower <= upper)

    lower, upper = NBSSP().new_user_interval(7, 6, 100, r=2.0, **parameters)
    assert lower.shape == upper.shape == (6,)
    assert np.all(lower <= upper)
