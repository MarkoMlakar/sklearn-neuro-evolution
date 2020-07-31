"""
============================
Plotting WANN Regressor
============================

An example plot of :class:`neuro_evolution._wann.WANNRegressor`
"""
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from neuro_evolution import WANNRegressor

np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# #############################################################################
# Fit regression model
regr = WANNRegressor(single_shared_weights=[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
                     number_of_generations=1000,
                     pop_size=150,
                     activation_default='sin',
                     activation_options='sigmoid tanh gauss relu sin inv identity',
                     fitness_threshold=0.90)
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

