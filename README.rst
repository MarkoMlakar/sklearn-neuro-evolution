.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_

.. |Travis| image:: https://travis-ci.org/scikit-learn-contrib/project-template.svg?branch=master
.. _Travis: https://travis-ci.org/scikit-learn-contrib/project-template

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/coy2qqaqr1rnnt5y/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/glemaitre/project-template

.. |Codecov| image:: https://codecov.io/gh/scikit-learn-contrib/project-template/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/scikit-learn-contrib/project-template

.. |CircleCI| image:: https://circleci.com/gh/scikit-learn-contrib/project-template.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/scikit-learn-contrib/project-template/tree/master

.. |ReadTheDocs| image:: https://readthedocs.org/projects/neuro-evolution/badge/?version=latest
.. _ReadTheDocs: https://neuro-evolution.readthedocs.io/en/latest/?badge=latest



sklearn-neuro-evolution
============================================================


.. _scikit-learn: https://scikit-learn.org
.. _neat: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
.. _wann: https://weightagnostic.github.io/



NEAT_ is a method developed by Kenneth O. Stanley for evolving arbitrary neural networks. It's an established topology search algorithm notable for its ability to optimize the weights and structure of networks simultaneously


Weight Agnostic Neural Networks (WANN_) is a method developed by Adam Gaier and David Ha in 2019. The algorithm is inspired by NEAT and focuses on evolving only the topology of the neural network without evolving the weights. It is a search method for topologies that can perform a task without explicit weight training. The end result is a minimal neural network topology where with a single shared weight parameter.


.. _sklearn-neuro-evolution: https://pypi.org/project/sklearn-neuro-evolution/
.. _neat-python: https://github.com/CodeReclaimers/neat-python
.. _weight-agnostic-neural-networks: https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease

sklearn-neuro-evolution_ package is based on a pure python implementation of NEAT called neat-python_ with the addition
of weight agnostic neural networks that are based on weight-agnostic-neural-networks_. It is compatible to use in the
Scikit-learn ecosystem

Installation
============================================================

.. code-block:: python

    pip install sklearn-neuro-evolution

NEAT Regression Example
============================================================

.. code-block:: python

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


NEAT Classification Example
============================================================

.. code-block:: python

    """
    ============================
    Plotting NEAT Classifier
    ============================

    An example plot of :class:`neuro_evolution._neat.NEATClassifier`
    """
    from matplotlib import pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from neuro_evolution import NEATClassifier

    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=123, n_samples=200)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    clf = NEATClassifier(number_of_generations=150,
                         fitness_threshold=0.90,
                         pop_size=150)

    neat_genome = clf.fit(x_train, y_train)
    y_predicted = neat_genome.predict(x_test)

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


WANN Regression Example
============================================================

.. code-block:: python

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

WANN Classification Example
============================================================
.. code-block:: python

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