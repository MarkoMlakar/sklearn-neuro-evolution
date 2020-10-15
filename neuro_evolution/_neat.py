"""Sklearn NEAT
"""

# Author: Marko Mlakar <markomlakar2@gmail.com>
# License: BSD-3-Clause License

from abc import ABCMeta, abstractmethod
from sklearn.metrics import accuracy_score, r2_score
import neat
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels
import multiprocessing
from ._config import NEATConfig, ConfigParser, GenomeConfig, ReproductionConfig, StagnationConfig, SpeciesConfig
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def softmax(x):
    """Compute softmax values for each sets of scores in x"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class BaseNEAT(BaseEstimator, metaclass=ABCMeta):
    """Base class for NEAT classification and regression.

    Warning: This class should not be used directly. Use derived classes instead!

    """

    @abstractmethod
    def __init__(self, number_of_generations, fitness_criterion, fitness_threshold, pop_size,
                 reset_on_extinction, no_fitness_termination, activation_default, activation_mutate_rate,
                 activation_options,
                 aggregation_default, aggregation_mutate_rate, aggregation_options, bias_init_mean,
                 bias_init_stdev, bias_max_value, bias_min_value, bias_mutate_power, bias_mutate_rate,
                 bias_replace_rate, compatibility_disjoint_coefficient, compatibility_weight_coefficient,
                 conn_add_prob, conn_delete_prob, enabled_default, enabled_mutate_rate, feed_forward,
                 initial_connection, node_add_prob, node_delete_prob, num_hidden,
                 response_init_mean, response_init_stdev, response_max_value, response_min_value,
                 response_mutate_power, response_mutate_rate, response_replace_rate, weight_init_mean,
                 weight_init_stdev, weight_max_value, weight_min_value, weight_mutate_power,
                 weight_mutate_rate, weight_replace_rate, compatibility_threshold, species_fitness_func,
                 max_stagnation, species_elitism, elitism, survival_threshold, statistic_reporter,
                 create_checkpoints, checkpoint_frequency):
        self.number_of_generations = number_of_generations
        self.fitness_criterion = fitness_criterion
        self.fitness_threshold = fitness_threshold
        self.pop_size = pop_size
        self.reset_on_extinction = reset_on_extinction
        self.no_fitness_termination = no_fitness_termination
        self.activation_default = activation_default
        self.activation_mutate_rate = activation_mutate_rate
        self.activation_options = activation_options
        self.aggregation_default = aggregation_default
        self.aggregation_mutate_rate = aggregation_mutate_rate
        self.aggregation_options = aggregation_options
        self.bias_init_mean = bias_init_mean
        self.bias_init_stdev = bias_init_stdev
        self.bias_max_value = bias_max_value
        self.bias_min_value = bias_min_value
        self.bias_mutate_power = bias_mutate_power
        self.bias_mutate_rate = bias_mutate_rate
        self.bias_replace_rate = bias_replace_rate
        self.compatibility_disjoint_coefficient = compatibility_disjoint_coefficient
        self.compatibility_weight_coefficient = compatibility_weight_coefficient
        self.conn_add_prob = conn_add_prob
        self.conn_delete_prob = conn_delete_prob
        self.enabled_default = enabled_default
        self.enabled_mutate_rate = enabled_mutate_rate
        self.feed_forward = feed_forward
        self.initial_connection = initial_connection
        self.node_add_prob = node_add_prob
        self.node_delete_prob = node_delete_prob
        self.num_hidden = num_hidden
        self.response_init_mean = response_init_mean
        self.response_init_stdev = response_init_stdev
        self.response_max_value = response_max_value
        self.response_min_value = response_min_value
        self.response_mutate_power = response_mutate_power
        self.response_mutate_rate = response_mutate_rate
        self.response_replace_rate = response_replace_rate
        self.weight_init_mean = weight_init_mean
        self.weight_init_stdev = weight_init_stdev
        self.weight_max_value = weight_max_value
        self.weight_min_value = weight_min_value
        self.weight_mutate_power = weight_mutate_power
        self.weight_mutate_rate = weight_mutate_rate
        self.weight_replace_rate = weight_replace_rate
        self.compatibility_threshold = compatibility_threshold
        self.species_fitness_func = species_fitness_func
        self.max_stagnation = max_stagnation
        self.species_elitism = species_elitism
        self.elitism = elitism
        self.survival_threshold = survival_threshold
        self.statistic_reporter = statistic_reporter
        self.create_checkpoints = create_checkpoints
        self.checkpoint_frequency = checkpoint_frequency

    @abstractmethod
    def _fitness_function(self, genomes, config):
        pass

    @abstractmethod
    def _validate_input(self, X, y):
        pass

    def _set_neat_config(self, num_inputs, num_outputs):
        _neat_config = NEATConfig(self.fitness_criterion, self.fitness_threshold, self.pop_size,
                                  self.reset_on_extinction, self.no_fitness_termination)

        _genome_config = GenomeConfig(self.activation_default, self.activation_mutate_rate, self.activation_options,
                                      self.aggregation_default, self.aggregation_mutate_rate, self.aggregation_options,
                                      self.bias_init_mean,
                                      self.bias_init_stdev, self.bias_max_value, self.bias_min_value,
                                      self.bias_mutate_power,
                                      self.bias_mutate_rate,
                                      self.bias_replace_rate, self.compatibility_disjoint_coefficient,
                                      self.compatibility_weight_coefficient,
                                      self.conn_add_prob, self.conn_delete_prob, self.enabled_default,
                                      self.enabled_mutate_rate,
                                      self.feed_forward,
                                      self.initial_connection, self.node_add_prob, self.node_delete_prob,
                                      self.num_hidden, num_inputs,
                                      num_outputs,
                                      self.response_init_mean, self.response_init_stdev, self.response_max_value,
                                      self.response_min_value,
                                      self.response_mutate_power, self.response_mutate_rate, self.response_replace_rate,
                                      self.weight_init_mean,
                                      self.weight_init_stdev, self.weight_max_value, self.weight_min_value,
                                      self.weight_mutate_power,
                                      self.weight_mutate_rate, self.weight_replace_rate)

        _reproduction_config = ReproductionConfig(self.elitism, self.survival_threshold)
        _species_config = SpeciesConfig(self.compatibility_threshold)
        _stagnation_config = StagnationConfig(self.species_fitness_func, self.max_stagnation, self.species_elitism)

        _config = ConfigParser(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                               neat.DefaultStagnation, _neat_config, _genome_config, _reproduction_config,
                               _species_config, _stagnation_config)
        return _config

    def _setup_neat(self):
        p_ = neat.Population(self.config_)
        self.stats_ = neat.StatisticsReporter()
        p_.add_reporter(self.stats_)
        if self.statistic_reporter:
            p_.add_reporter(neat.StdOutReporter(self.statistic_reporter))
        if self.create_checkpoints:
            p_.add_reporter(neat.Checkpointer(self.checkpoint_frequency))
        return p_

    def _more_tags(self):
        return {'poor_score': True}

    def _fit(self, X, y):
        X, y = self._validate_input(X,y)
        self.X_ = X
        self.y_ = y
        self.is_fitted_ = True
        self.pairwise_ = True
        if hasattr(self, "classes_"):
            self.config_ = self._set_neat_config(X.shape[1], len(self.classes_))
        else:
            self.config_ = self._set_neat_config(X.shape[1], 1)

        self.p_ = self._setup_neat()
        self.winner_genome_ = self.p_.run(self._fitness_function, self.number_of_generations)

        print(self.winner_genome_)

        return self


    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification).

        Returns
        -------
        self : NEATClassifier or NEATRegressor object with the best genome (neural network)
        """
        return self._fit(X, y)




class NEATClassifier(BaseNEAT, ClassifierMixin):
    """NEAT classifier
    """

    # TODO: Add model description

    def __init__(self, number_of_generations=100,
                 fitness_criterion='max',
                 fitness_threshold=0.95,
                 pop_size=150,
                 reset_on_extinction=0,
                 no_fitness_termination=0,
                 activation_default='sigmoid',
                 activation_mutate_rate=0.00,
                 activation_options='sigmoid relu tanh gauss inv hat clamped sin square abs exp identity',
                 aggregation_default='sum',
                 aggregation_mutate_rate=0.0,
                 aggregation_options='sum',
                 bias_init_mean=0.0,
                 bias_init_stdev=1.0,
                 bias_max_value=30.0,
                 bias_min_value=-30.0,
                 bias_mutate_power=0.5,
                 bias_mutate_rate=0.7,
                 bias_replace_rate=0.1,
                 compatibility_disjoint_coefficient=1.0,
                 compatibility_weight_coefficient=0.5,
                 conn_add_prob=0.2,
                 conn_delete_prob=0.0,
                 enabled_default=1,
                 enabled_mutate_rate=0.01,
                 feed_forward='true',
                 initial_connection='full_direct',
                 node_add_prob=0.2,
                 node_delete_prob=0.0,
                 num_hidden=0,
                 response_init_mean=1.0,
                 response_init_stdev=0.0,
                 response_max_value=30.0,
                 response_min_value=-30.0,
                 response_mutate_power=0.0,
                 response_mutate_rate=0.0,
                 response_replace_rate=0.0,
                 weight_init_mean=0.0,
                 weight_init_stdev=1.0,
                 weight_max_value=30,
                 weight_min_value=-30,
                 weight_mutate_power=0.5,
                 weight_mutate_rate=0.8,
                 weight_replace_rate=0.1,
                 compatibility_threshold=3.0,
                 species_fitness_func='max',
                 max_stagnation=20,
                 species_elitism=2,
                 elitism=2,
                 survival_threshold=0.2,
                 statistic_reporter=1,
                 create_checkpoints=0,
                 checkpoint_frequency=20):
        super().__init__(number_of_generations=number_of_generations,
                         fitness_criterion=fitness_criterion,
                         fitness_threshold=fitness_threshold,
                         pop_size=pop_size,
                         reset_on_extinction=reset_on_extinction,
                         no_fitness_termination=no_fitness_termination,
                         activation_default=activation_default,
                         activation_mutate_rate=activation_mutate_rate,
                         activation_options=activation_options,
                         aggregation_default=aggregation_default,
                         aggregation_mutate_rate=aggregation_mutate_rate,
                         aggregation_options=aggregation_options,
                         bias_init_mean=bias_init_mean,
                         bias_init_stdev=bias_init_stdev,
                         bias_max_value=bias_max_value,
                         bias_min_value=bias_min_value,
                         bias_mutate_power=bias_mutate_power,
                         bias_mutate_rate=bias_mutate_rate,
                         bias_replace_rate=bias_replace_rate,
                         compatibility_disjoint_coefficient=compatibility_disjoint_coefficient,
                         compatibility_weight_coefficient=compatibility_weight_coefficient,
                         conn_add_prob=conn_add_prob,
                         conn_delete_prob=conn_delete_prob,
                         enabled_default=enabled_default,
                         enabled_mutate_rate=enabled_mutate_rate,
                         feed_forward=feed_forward,
                         initial_connection=initial_connection,
                         node_add_prob=node_add_prob,
                         node_delete_prob=node_delete_prob,
                         num_hidden=num_hidden,
                         response_init_mean=response_init_mean,
                         response_init_stdev=response_init_stdev,
                         response_max_value=response_max_value,
                         response_min_value=response_min_value,
                         response_mutate_power=response_mutate_power,
                         response_mutate_rate=response_mutate_rate,
                         response_replace_rate=response_replace_rate,
                         weight_init_mean=weight_init_mean,
                         weight_init_stdev=weight_init_stdev,
                         weight_max_value=weight_max_value,
                         weight_min_value=weight_min_value,
                         weight_mutate_power=weight_mutate_power,
                         weight_mutate_rate=weight_mutate_rate,
                         weight_replace_rate=weight_replace_rate,
                         compatibility_threshold=compatibility_threshold,
                         species_fitness_func=species_fitness_func,
                         max_stagnation=max_stagnation,
                         species_elitism=species_elitism,
                         elitism=elitism,
                         survival_threshold=survival_threshold,
                         statistic_reporter=statistic_reporter,
                         create_checkpoints=create_checkpoints,
                         checkpoint_frequency=checkpoint_frequency)

    def predict(self, X):
        """Predict using the NEAT classifier

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray, shape (n_samples,) or (n_samples, n_classes)
            The predicted classes.
        """
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] is not self.config_.genome_config.num_inputs:
            raise ValueError

        net = neat.nn.FeedForwardNetwork.create(self.winner_genome_, self.config_)

        predictions = np.empty(X.shape[0], ).astype(int)
        for i in range(0, X.shape[0]):
            output = net.activate(X[i])
            softmax_result = softmax(output)
            if len(softmax_result) > 1:
                class_output = np.argmax(((softmax_result / np.max(softmax_result)) == 1))
            else:
                class_output = self.label_encoder_.transform(softmax_result)
            predictions[i] = class_output

        return self.label_encoder_.inverse_transform(predictions)

    def _fitness_function(self, genomes, config):
        """Evaluation function for calculating the fitness of genomes.
        Applies the softmax function for the classification output and calculates the accuracy score.

        Parameters
        ----------
        genomes : {list} of all genomes
        config : {ConfigParser} parsed config file

        Returns
        -------
        void : Assigns the fitness to each genome
        """
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            predictions = np.empty(self.X_.shape[0])
            for i in range(0, self.X_.shape[0]):
                output = net.activate(self.X_[i])
                softmax_result = softmax(output)
                if len(softmax_result) > 1:
                    class_output = np.argmax(((softmax_result / np.max(softmax_result)) == 1))
                else:
                    class_output = softmax_result[0]
                predictions[i] = class_output

            genome.fitness = accuracy_score(self.y_, predictions)

    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification).

        Returns
        -------
        self : NEATClassifier object with the best genome (neural network)
        """
        return self._fit(X, y)

    def _validate_input(self, X, y):
        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(y)
        self.classes_ = unique_labels(y)
        y = self.label_encoder_.transform(y)
        X, y = check_X_y(X, y)
        return X, y

class NEATRegressor(BaseNEAT, RegressorMixin):
    """NEAT regressor
    """
    def __init__(self, number_of_generations=100,
                 fitness_criterion='max',
                 fitness_threshold=0.95,
                 pop_size=150,
                 reset_on_extinction=0,
                 no_fitness_termination=0,
                 activation_default='relu',
                 activation_mutate_rate=0.0,
                 activation_options='sigmoid relu tanh gauss inv hat clamped sin square abs exp identity',
                 aggregation_default='sum',
                 aggregation_mutate_rate=0.0,
                 aggregation_options='sum',
                 bias_init_mean=0.0,
                 bias_init_stdev=1.0,
                 bias_max_value=30.0,
                 bias_min_value=-30.0,
                 bias_mutate_power=0.5,
                 bias_mutate_rate=0.7,
                 bias_replace_rate=0.1,
                 compatibility_disjoint_coefficient=1.0,
                 compatibility_weight_coefficient=0.5,
                 conn_add_prob=0.2,
                 conn_delete_prob=0.0,
                 enabled_default='true',
                 enabled_mutate_rate=0.01,
                 feed_forward='true',
                 initial_connection='full',
                 node_add_prob=0.2,
                 node_delete_prob=0.0,
                 num_hidden=0,
                 response_init_mean=1.0,
                 response_init_stdev=0.0,
                 response_max_value=30.0,
                 response_min_value=-30.0,
                 response_mutate_power=0.0,
                 response_mutate_rate=0.0,
                 response_replace_rate=0.0,
                 weight_init_mean=0.0,
                 weight_init_stdev=1.0,
                 weight_max_value=30,
                 weight_min_value=-30,
                 weight_mutate_power=0.5,
                 weight_mutate_rate=0.8,
                 weight_replace_rate=0.1,
                 compatibility_threshold=3.0,
                 species_fitness_func='max',
                 max_stagnation=20,
                 species_elitism=1,
                 elitism=2,
                 survival_threshold=0.2,
                 statistic_reporter=1,
                 create_checkpoints=0,
                 checkpoint_frequency=20):
        super().__init__(number_of_generations=number_of_generations,
                         fitness_criterion=fitness_criterion,
                         fitness_threshold=fitness_threshold,
                         pop_size=pop_size,
                         reset_on_extinction=reset_on_extinction,
                         no_fitness_termination=no_fitness_termination,
                         activation_default=activation_default,
                         activation_mutate_rate=activation_mutate_rate,
                         activation_options=activation_options,
                         aggregation_default=aggregation_default,
                         aggregation_mutate_rate=aggregation_mutate_rate,
                         aggregation_options=aggregation_options,
                         bias_init_mean=bias_init_mean,
                         bias_init_stdev=bias_init_stdev,
                         bias_max_value=bias_max_value,
                         bias_min_value=bias_min_value,
                         bias_mutate_power=bias_mutate_power,
                         bias_mutate_rate=bias_mutate_rate,
                         bias_replace_rate=bias_replace_rate,
                         compatibility_disjoint_coefficient=compatibility_disjoint_coefficient,
                         compatibility_weight_coefficient=compatibility_weight_coefficient,
                         conn_add_prob=conn_add_prob,
                         conn_delete_prob=conn_delete_prob,
                         enabled_default=enabled_default,
                         enabled_mutate_rate=enabled_mutate_rate,
                         feed_forward=feed_forward,
                         initial_connection=initial_connection,
                         node_add_prob=node_add_prob,
                         node_delete_prob=node_delete_prob,
                         num_hidden=num_hidden,
                         response_init_mean=response_init_mean,
                         response_init_stdev=response_init_stdev,
                         response_max_value=response_max_value,
                         response_min_value=response_min_value,
                         response_mutate_power=response_mutate_power,
                         response_mutate_rate=response_mutate_rate,
                         response_replace_rate=response_replace_rate,
                         weight_init_mean=weight_init_mean,
                         weight_init_stdev=weight_init_stdev,
                         weight_max_value=weight_max_value,
                         weight_min_value=weight_min_value,
                         weight_mutate_power=weight_mutate_power,
                         weight_mutate_rate=weight_mutate_rate,
                         weight_replace_rate=weight_replace_rate,
                         compatibility_threshold=compatibility_threshold,
                         species_fitness_func=species_fitness_func,
                         max_stagnation=max_stagnation,
                         species_elitism=species_elitism,
                         elitism=elitism,
                         survival_threshold=survival_threshold,
                         statistic_reporter=statistic_reporter,
                         create_checkpoints=create_checkpoints,
                         checkpoint_frequency=checkpoint_frequency)

    def predict(self, X):
        """Predict using the NEATRegressor model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self, [ 'X_', 'y_' ])
        X = check_array(X)
        net = neat.nn.FeedForwardNetwork.create(self.winner_genome_, self.config_)
        predictions = np.empty(X.shape[ 0 ], )
        for i in range(0, X.shape[ 0 ]):
            output = net.activate(X[ i ])
            predictions[ i ] = output[ 0 ]

        return predictions

    def _fitness_function(self, genomes, config):
        """Evaluation function for calculating the fitness of genomes. Calculates the r2_score

        Parameters
        ----------
        genomes : {list} of all genomes
        config : {ConfigParser} parsed config file

        Returns
        -------
        void : Assigns the fitness to each genome
        """
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            predictions = np.empty(self.y_.shape[0])
            for i in range(0, self.y_.shape[0]):
                output = net.activate(self.X_[i])
                predictions[i] = output[0]
            genome.fitness = r2_score(self.y_, predictions)

    def _validate_input(self, X, y):
        X, y = check_X_y(X, y)
        return X, y
