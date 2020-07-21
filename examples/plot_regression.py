"""
============================
Plotting NEAT Regressor
============================

An example plot of :class:`neuro_evolution._neat.NEATRegressor`
"""
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

print(__doc__)

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#
# License: BSD 3 clause (C) INRIA


# #############################################################################
# Generate sample data
import numpy as np
import matplotlib.pyplot as plt
from neuro_evolution import NEATRegressor

np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X).ravel()

# Add noise to targets
y[::5] += 1 * (0.5 - np.random.rand(8))

# #############################################################################
# Fit regression model
regr = NEATRegressor(number_of_generations=500, pop_size=300,
                     compatibility_threshold=3.0, num_outputs=1, num_inputs=1, activation_default='sin',
                     activation_options='identity sin gauss tanh sigmoid abs relu square softplus clamped log exp '
                                        'cube hat',
                     activation_mutate_rate=0.20, fitness_threshold=0.98)
neat_genome = regr.fit(X, y)
neat_predictions = neat_genome.predict(X)
y_ = neat_predictions

print("R2 score: ", r2_score(y,y_))
print("MSE", mean_squared_error(y,y_))
print("RMSE", sqrt(mean_squared_error(y,y_)))

plt.plot(X, y, color='darkorange', label='data')
plt.plot(X,y_, color='navy', label='prediction')

plt.legend()
plt.title("NEATRegressor")

plt.tight_layout()
plt.show()

