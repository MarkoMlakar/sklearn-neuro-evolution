"""
============================
# Simple XOR regression example
============================

An example of :class:`neuro_evolution._neat.NEATRegressor`
"""
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from neuro_evolution import NEATRegressor

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
regr = NEATRegressor(number_of_generations=1000,
                     fitness_threshold=0.95,
                     pop_size=150,
                     activation_mutate_rate=0.00,
                     activation_default='sigmoid')
neat_genome = regr.fit(x_train, y_train)
print("Score", neat_genome.score(x_test, y_test))

neat_predictions = neat_genome.predict(x_test)
print("R2 score: ", r2_score(y_test, neat_predictions))
print("MSE", mean_squared_error(y_test, neat_predictions))