"""
============================
Plotting WANN Classifier
============================

An example plot of :class:`neuro_evolution._wann.WANNClassifier`
"""
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from neuro_evolution import WANNClassifier

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=123, n_samples=200)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

clf = WANNClassifier(single_shared_weights=[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
                     number_of_generations=150,
                     pop_size=150,
                     fitness_threshold=0.90,
                     activation_default='relu')

wann_genome = clf.fit(x_train, y_train)
y_predicted = wann_genome.predict(x_test)

fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for three-dimensional scattered points
train_z_data = y_train
train_x_data = x_train[:, 1]
train_y_data = x_train[:, 0]
ax.scatter3D(train_x_data, train_y_data, train_z_data, c='Blue')

test_z_data = y_predicted
test_x_data = x_test[:, 1]
test_y_data = x_test[:, 0]
ax.scatter3D(test_x_data, test_y_data, test_z_data, c='Red')
ax.legend(['Actual', 'Predicted'])
plt.show()

print(classification_report(y_test, y_predicted))
