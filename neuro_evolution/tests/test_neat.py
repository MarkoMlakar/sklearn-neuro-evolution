# NOTE: If some tests fail (or take too long)
# you need to tweak the NEATClassifier() and NEATRegressor() init parameters in _neat.py!

from sklearn.utils.estimator_checks import parametrize_with_checks
from neuro_evolution import NEATClassifier
from neuro_evolution import NEATRegressor

@parametrize_with_checks([NEATRegressor(), NEATClassifier()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)