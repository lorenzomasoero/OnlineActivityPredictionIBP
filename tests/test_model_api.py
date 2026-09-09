import inspect

import activity_prediction
from activity_prediction.models import BeSSP, BetaBinomial, IBP, NBSSP, TGSSP


def _public_methods(model_class):
    return {
        name
        for name, member in inspect.getmembers(model_class, inspect.isfunction)
        if not name.startswith("_")
    }


def test_classes_use_paper_names_without_legacy_aliases():
    assert activity_prediction.BeSSP is BeSSP
    assert activity_prediction.TGSSP is TGSSP
    assert activity_prediction.NBSSP is NBSSP
    assert activity_prediction.IBP is IBP
    assert activity_prediction.BetaBinomial is BetaBinomial
    assert not hasattr(activity_prediction, "GD")
    assert not hasattr(activity_prediction, "NegBintSBSP")
    assert not hasattr(activity_prediction, "BB")


def test_model_classes_have_only_the_curated_public_methods():
    assert _public_methods(BeSSP) == {
        "expected_new_users",
        "fit_curve",
        "fit_marginal_likelihood",
        "new_user_interval",
    }
    assert _public_methods(TGSSP) == {
        "expected_new_users",
        "fit_curve",
        "fit_marginal_likelihood",
        "new_user_interval",
    }
    assert _public_methods(NBSSP) == {
        "expected_new_users",
        "expected_total_triggers",
        "fit_curve",
        "fit_marginal_likelihood",
        "new_user_interval",
    }
    assert _public_methods(IBP) == {
        "expected_new_users",
        "fit_curve",
        "fit_marginal_likelihood",
    }
    assert _public_methods(BetaBinomial) == {
        "expected_new_users",
        "fit_marginal_likelihood",
    }
