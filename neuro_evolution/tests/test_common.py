import pytest

from sklearn.utils.estimator_checks import check_estimator

from neuro_evolution import NEATClassifier
from neuro_evolution import NEATRegressor


@pytest.mark.parametrize(
    "Estimator", [NEATClassifier, NEATRegressor]
)
def test_all_estimators(Estimator):
    # TODO: PASS the unit tests!!
    return check_estimator(Estimator)
