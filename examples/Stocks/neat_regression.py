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
import pandas as pd

data = pd.read_csv('LeveledLogStockData.csv')
# print(data.columns)
# print(data)

# select an index to use to train network
index = data[ 'leveled log Nasdaq' ].values
n_samples = len(index)

# split into inputs and outputs
n_inputs = 5  # number of days to use as input
n_outputs = 1  # predict next day

x = [ ]
y = [ ]

for i in range(n_samples - n_inputs - 1):
    x.append(index[ i:i + n_inputs ])
    y.append([ index[ i + 1 ] ])

x = np.asarray(x)
y = np.asarray(y)

# print(x.shape, y.shape)

# hold out last samples for testing
n_train = int(n_samples * .9)
n_test = n_samples - n_train

print('train, test', n_train, n_test)

train_x = x[ 0:n_train ]
test_x = x[ n_train:-1 ]
train_y = y[ 0:n_train ]
test_y = y[ n_train:-1 ]
print('data split', train_x.shape, train_y.shape)
print('data split', test_x.shape, test_y.shape)

# shuffle training data?
z = np.arange(0, n_train - 1)
np.random.shuffle(z)

tx = train_x[ z[ ::-1 ] ]
ty = train_y[ z[ ::-1 ] ]

train_x = tx
train_y = ty

# #############################################################################
# Fit regression model
regr = NEATRegressor(number_of_generations=1000,
                     fitness_threshold=0.98,
                     pop_size=100,
                     compatibility_threshold=3.0,
                     num_outputs=1,
                     num_inputs=5,
                     activation_default='log',
                     activation_options='inv relu clamped cube tanh sigmoid log',
                     activation_mutate_rate=0.20,
                     conn_add_prob=0.60,
                     conn_delete_prob=0.60,
                     enabled_mutate_rate=0.30,
                     initial_connection='full_nodirect',
                     survival_threshold=0.30)
neat_genome = regr.fit(train_x, train_y)

neat_predictions = neat_genome.predict(test_x)
y_ = neat_predictions

print("R2 score: ", r2_score(test_y,y_))
print("MSE", mean_squared_error(test_y,y_))
print("RMSE", sqrt(mean_squared_error(test_y,y_)))

plt.plot(test_y, color='darkorange', label='Actual')
plt.plot(y_, color='navy', label='Predicted')

plt.legend()
plt.title("NEATRegressor")

plt.tight_layout()
plt.show()

