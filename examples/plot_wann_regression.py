"""
============================
# Simple XOR regression example
============================

An example of :class:`neuro_evolution._wann.WANNRegressor`
"""
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from neuro_evolution import WANNRegressor

shared_weights = np.array((-2.0, -1.0, -0.5, 0.5, 1.0, 2.0))
num_of_shared_weights = len(shared_weights)
x_train = np.array([
                    [0, 0],
                    [1, 1],
                    [1, 0],
                    [0, 1],
])
y_train = np.logical_xor(x_train[ :, 0 ] > 0.5, x_train[ :, 1 ] > 0.5).astype(int)

x_test = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
y_test = np.array([1, 1, 0, 0])

# #############################################################################
# Fit regression model
regr = WANNRegressor(single_shared_weights=shared_weights,
                     number_of_generations=200,
                     pop_size=150,
                     activation_default='sigmoid',
                     activation_options='sigmoid tanh gauss relu sin inv identity',
                     fitness_threshold=0.92)

wann_genome = regr.fit(x_train, y_train)
print("Score: ", wann_genome.score(x_test, y_test))

wann_predictions = wann_genome.predict(x_test)
print("R2 score: ", r2_score(y_test, wann_predictions))
print("MSE", mean_squared_error(y_test, wann_predictions))