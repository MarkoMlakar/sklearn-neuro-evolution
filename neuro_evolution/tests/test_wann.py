from sklearn.utils.estimator_checks import parametrize_with_checks
from neuro_evolution import WANNClassifier
from neuro_evolution import WANNRegressor


@parametrize_with_checks([WANNRegressor(), WANNClassifier()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)