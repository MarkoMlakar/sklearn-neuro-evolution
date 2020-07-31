"""
============================
Plotting WANN Classifier
============================

An example plot of :class:`neuro_evolution._wann.WANNClassifier`
"""
import numpy as np
from matplotlib import pyplot as plt
from neuro_evolution import WANNClassifier

X = [ [ 0, 0 ], [ 1, 1 ] ]
y = [ 0, 1 ]
clf = WANNClassifier(single_shared_weights=[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
                     number_of_generations=1000,
                     pop_size=150,
                     compatibility_threshold=3.0,
                     activation_options='identity sin gauss tanh sigmoid abs relu square softplus clamped log exp '
                                        'cube hat',
                     activation_default='tanh')
clf.fit(X, y)

rng = np.random.RandomState(0)
X_test = rng.rand(1000, 2)
y_pred = clf.predict(X_test)

X_0 = X_test[ y_pred == 0 ]
X_1 = X_test[ y_pred == 1 ]

p0 = plt.scatter(0, 0, c='red', s=150)
p1 = plt.scatter(1, 1, c='blue', s=150)

ax0 = plt.scatter(X_0[ :, 0 ], X_0[ :, 1 ], c='indianred', s=35)
ax1 = plt.scatter(X_1[ :, 0 ], X_1[ :, 1 ], c='deepskyblue', s=35)

leg = plt.legend([ p0, p1, ax0, ax1 ],
                 [ 'Point 0', 'Point 1', 'Class 0', 'Class 1' ],
                 loc='upper left', fancybox=True, scatterpoints=1)
leg.get_frame().set_alpha(0.5)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.xlim([ -.5, 1.5 ])

plt.show()
