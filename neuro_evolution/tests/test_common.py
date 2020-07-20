import pytest

from sklearn.utils.estimator_checks import check_estimator

from neuro_evolution import NEATEstimator
from neuro_evolution import TemplateClassifier
from neuro_evolution import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [NEATEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
