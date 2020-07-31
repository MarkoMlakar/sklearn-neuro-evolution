"""
============================
Plotting NEAT Regressor
============================

An example plot of :class:`neuro_evolution._neat.NEATRegressor`
"""
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
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
regr = NEATRegressor(number_of_generations=150,
                     fitness_threshold=0.95,
                     pop_size=64,
                     activation_default='sin',
                     activation_options='sigmoid tanh gauss relu sin inv',
                     activation_mutate_rate=0.50)
neat_genome = regr.fit(X, y)
neat_predictions = neat_genome.predict(X)
y_ = neat_predictions

print("R2 score: ", r2_score(y, y_))
print("MSE", mean_squared_error(y, y_))
print("RMSE", sqrt(mean_squared_error(y, y_)))

plt.plot(X, y, color='darkorange', label='data')
plt.plot(X, y_, color='navy', label='prediction')

plt.legend()
plt.title("NEATRegressor")

plt.tight_layout()
plt.show()
