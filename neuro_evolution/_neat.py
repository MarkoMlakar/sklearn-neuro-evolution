"""Sklearn NEAT
"""

# Author: Marko Mlakar <markomlakar2@gmail.com>
# License: BSD-3-Clause License


from abc import ABCMeta, abstractmethod
import subprocess
import sys
from sklearn.metrics import accuracy_score, r2_score

subprocess.check_call([ sys.executable, "-m", "pip", "install", "neat-python" ])
import neat
from ._config import NEATConfig, ConfigParser, GenomeConfig, ReproductionConfig, StagnationConfig, SpeciesConfig
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


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
                 initial_connection, node_add_prob, node_delete_prob, num_hidden, num_inputs, num_outputs,
                 response_init_mean, response_init_stdev, response_max_value, response_min_value,
                 response_mutate_power, response_mutate_rate, response_replace_rate, weight_init_mean,
                 weight_init_stdev, weight_max_value, weight_min_value, weight_mutate_power,
                 weight_mutate_rate, weight_replace_rate, compatibility_threshold, species_fitness_func,
                 max_stagnation, species_elitism, elitism, survival_threshold, statistic_reporter,
                 create_checkpoints, checkpoint_frequency):
        self.neat = neat
        neat_config = NEATConfig(fitness_criterion, fitness_threshold, pop_size,
                                 reset_on_extinction, no_fitness_termination)
        genome_config = GenomeConfig(activation_default, activation_mutate_rate, activation_options,
                                     aggregation_default, aggregation_mutate_rate, aggregation_options, bias_init_mean,
                                     bias_init_stdev, bias_max_value, bias_min_value, bias_mutate_power,
                                     bias_mutate_rate,
                                     bias_replace_rate, compatibility_disjoint_coefficient,
                                     compatibility_weight_coefficient,
                                     conn_add_prob, conn_delete_prob, enabled_default, enabled_mutate_rate,
                                     feed_forward,
                                     initial_connection, node_add_prob, node_delete_prob, num_hidden, num_inputs,
                                     num_outputs,
                                     response_init_mean, response_init_stdev, response_max_value, response_min_value,
                                     response_mutate_power, response_mutate_rate, response_replace_rate,
                                     weight_init_mean,
                                     weight_init_stdev, weight_max_value, weight_min_value, weight_mutate_power,
                                     weight_mutate_rate, weight_replace_rate)

        reproduction_config = ReproductionConfig(elitism, survival_threshold)
        species_config = SpeciesConfig(compatibility_threshold)
        stagnation_config = StagnationConfig(species_fitness_func, max_stagnation, species_elitism)

        self.config = ConfigParser(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                   neat.DefaultStagnation, neat_config, genome_config, reproduction_config,
                                   species_config, stagnation_config)

        self.p = neat.Population(self.config)

        if statistic_reporter:
            self.p.add_reporter(neat.StdOutReporter(statistic_reporter))
            self.stats = neat.StatisticsReporter()
            self.p.add_reporter(self.stats)
            if create_checkpoints:
                self.p.add_reporter(neat.Checkpointer(checkpoint_frequency))

        self.number_of_generations = number_of_generations
        self.winner_genome = None

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
        self : returns the best genome (neural network)
        """
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.is_fitted_ = True
        self.winner_genome = self.p.run(self._fitness_function, self.number_of_generations)

        print(self.winner_genome)
        return self

    @abstractmethod
    def _fitness_function(self, genomes, config):
        pass


class NEATClassifier(BaseNEAT, ClassifierMixin):
    """NEAT classifier
    """

    # TODO: Add model description

    def __init__(self, number_of_generations=500,
                 fitness_criterion='max',
                 fitness_threshold=0.92,
                 pop_size=500,
                 reset_on_extinction=0,
                 no_fitness_termination=0,
                 activation_default='sigmoid',
                 activation_mutate_rate=0.00,
                 activation_options='sigmoid',
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
                 conn_add_prob=0.25,
                 conn_delete_prob=0.25,
                 enabled_default=1,
                 enabled_mutate_rate=0.01,
                 feed_forward='true',
                 initial_connection='full_direct',
                 node_add_prob=0.2,
                 node_delete_prob=0.2,
                 num_hidden=0,
                 num_inputs=2,
                 num_outputs=1,
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
                         num_inputs=num_inputs,
                         num_outputs=num_outputs,
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
        check_is_fitted(self, [ 'X_', 'y_' ])
        X = check_array(X)
        net = neat.nn.FeedForwardNetwork.create(self.winner_genome, self.config)
        predictions = np.empty(X.shape[ 0 ], )
        for i in range(0, X.shape[ 0 ]):
            output = net.activate(X[ i ])
            softmax_result = self.neat.math_util.softmax(np.clip(output, min(self.y_), max(self.y_)))
            class_output = np.argmax(((softmax_result / np.max(softmax_result)) == 1).astype(int))
            predictions[ i ] = class_output

        return predictions

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
            net = self.neat.nn.FeedForwardNetwork.create(genome, config)

            predictions = np.empty(self.X_.shape[0], )
            for i in range(0, self.X_.shape[0]):
                output = net.activate(self.X_[i])
                softmax_result = self.neat.math_util.softmax(output)
                class_output = np.argmax(((softmax_result / np.max(softmax_result)) == 1).astype(int))
                predictions[i] = class_output

            genome.fitness = accuracy_score(self.y_, predictions)


class NEATRegressor(BaseNEAT, RegressorMixin):
    """NEAT regressor
    """

    # TODO: Add model description

    def __init__(self, number_of_generations=10,
                 fitness_criterion='max',
                 fitness_threshold=0.92,
                 pop_size=150,
                 reset_on_extinction=0,
                 no_fitness_termination=0,
                 activation_default='sigmoid',
                 activation_mutate_rate=0.00,
                 activation_options='sigmoid',
                 aggregation_default='sum',
                 aggregation_mutate_rate=0.0,
                 aggregation_options='sum',
                 bias_init_mean=0.0,
                 bias_init_stdev=1.0,
                 bias_max_value=2.0,
                 bias_min_value=-2.0,
                 bias_mutate_power=0.5,
                 bias_mutate_rate=0.7,
                 bias_replace_rate=0.1,
                 compatibility_disjoint_coefficient=1.0,
                 compatibility_weight_coefficient=0.5,
                 conn_add_prob=0.5,
                 conn_delete_prob=0.5,
                 enabled_default=1,
                 enabled_mutate_rate=0.01,
                 feed_forward='true',
                 initial_connection='full_direct',
                 node_add_prob=0.2,
                 node_delete_prob=0.2,
                 num_hidden=0,
                 num_inputs=2,
                 num_outputs=1,
                 response_init_mean=1.0,
                 response_init_stdev=0.0,
                 response_max_value=10.0,
                 response_min_value=-10.0,
                 response_mutate_power=0.0,
                 response_mutate_rate=0.0,
                 response_replace_rate=0.1,
                 weight_init_mean=0.0,
                 weight_init_stdev=1.0,
                 weight_max_value=2,
                 weight_min_value=-2,
                 weight_mutate_power=0.5,
                 weight_mutate_rate=0.8,
                 weight_replace_rate=0.1,
                 compatibility_threshold=3.8,
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
                         bias_min_value=-bias_min_value,
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
                         num_inputs=num_inputs,
                         num_outputs=num_outputs,
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
        net = self.neat.nn.FeedForwardNetwork.create(self.winner_genome, self.config)
        predictions = np.empty(X.shape[ 0 ], )
        for i in range(0, X.shape[ 0 ]):
            output = net.activate(X[ i ])
            predictions[ i ] = output[0]

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
            net = self.neat.nn.FeedForwardNetwork.create(genome, config)

            predicted = np.empty(self.X_.shape[ 0 ], )
            for i in range(0, self.X_.shape[ 0 ]):
                output = net.activate(self.X_[ i ])
                predicted[i] = output[0]

            adjusted_r2_score = 1 - (1 - r2_score(self.y_, predicted)) * (len(self.y_) - 1) / (
                        len(self.y_) - self.X_.shape[ 1 ] - 1)

            genome.fitness = adjusted_r2_score

