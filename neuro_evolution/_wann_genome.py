import neat
from neat.activations import ActivationFunctionSet
from neat.aggregations import AggregationFunctionSet
from neat.config import ConfigParameter, write_pretty_params
from ._wann_genes import WannNodeGene, WannConnectionGene
import numpy as np
import random
from itertools import count
from neat.six_util import iteritems, iterkeys


class WannGenomeConfig(object):
    """Sets up and holds configuration information for the DefaultGenome class."""
    allowed_connectivity = [ 'unconnected', 'fs_neat_nohidden', 'fs_neat', 'fs_neat_hidden',
                             'full_nodirect', 'full', 'full_direct',
                             'partial_nodirect', 'partial', 'partial_direct' ]

    def __init__(self, params):
        # Create full set of available activation functions.
        self.activation_defs = ActivationFunctionSet()
        # ditto for aggregation functions - name difference for backward compatibility
        self.aggregation_function_defs = AggregationFunctionSet()
        self.aggregation_defs = self.aggregation_function_defs

        self._params = [ ConfigParameter('num_inputs', int),
                         ConfigParameter('num_outputs', int),
                         ConfigParameter('num_hidden', int),
                         ConfigParameter('feed_forward', bool),
                         ConfigParameter('compatibility_disjoint_coefficient', float),
                         ConfigParameter('compatibility_weight_coefficient', float),
                         ConfigParameter('conn_add_prob', float),
                         ConfigParameter('conn_delete_prob', float),
                         ConfigParameter('node_add_prob', float),
                         ConfigParameter('node_delete_prob', float),
                         ConfigParameter('single_structural_mutation', bool, 'false'),
                         ConfigParameter('structural_mutation_surer', str, 'default'),
                         ConfigParameter('initial_connection', str, 'unconnected')]

        # Gather configuration data from the gene classes.
        self.node_gene_type = params[ 'node_gene_type' ]
        self._params += self.node_gene_type.get_config_params()
        self.connection_gene_type = params[ 'connection_gene_type' ]
        self._params += self.connection_gene_type.get_config_params()

        # Use the configuration data to interpret the supplied parameters.
        for p in self._params:
            setattr(self, p.name, p.interpret(params))

        # By convention, input pins have negative keys, and the output
        # pins have keys 0,1,...
        self.input_keys = [ -i - 1 for i in range(self.num_inputs) ]
        self.output_keys = [ i for i in range(self.num_outputs) ]

        self.connection_fraction = None

        # Verify that initial connection type is valid.
        # pylint: disable=access-member-before-definition
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")

        assert self.initial_connection in self.allowed_connectivity

        # Verify structural_mutation_surer is valid.
        # pylint: disable=access-member-before-definition
        if self.structural_mutation_surer.lower() in [ '1', 'yes', 'true', 'on' ]:
            self.structural_mutation_surer = 'true'
        elif self.structural_mutation_surer.lower() in [ '0', 'no', 'false', 'off' ]:
            self.structural_mutation_surer = 'false'
        elif self.structural_mutation_surer.lower() == 'default':
            self.structural_mutation_surer = 'default'
        else:
            error_string = "Invalid structural_mutation_surer {!r}".format(
                self.structural_mutation_surer)
            raise RuntimeError(error_string)

        self.node_indexer = None

    def add_activation(self, name, func):
        self.activation_defs.add(name, func)

    def add_aggregation(self, name, func):
        self.aggregation_function_defs.add(name, func)

    def save(self, f):
        if 'partial' in self.initial_connection:
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")
            f.write('initial_connection      = {0} {1}\n'.format(self.initial_connection,
                                                                 self.connection_fraction))
        else:
            f.write('initial_connection      = {0}\n'.format(self.initial_connection))

        assert self.initial_connection in self.allowed_connectivity

        write_pretty_params(f, self, [ p for p in self._params
                                       if not 'initial_connection' in p.name ])

    def get_new_node_key(self, node_dict):
        if self.node_indexer is None:
            self.node_indexer = count(max(list(iterkeys(node_dict))) + 1)

        new_id = next(self.node_indexer)

        assert new_id not in node_dict

        return new_id

    def check_structural_mutation_surer(self):
        if self.structural_mutation_surer == 'true':
            return True
        elif self.structural_mutation_surer == 'false':
            return False
        elif self.structural_mutation_surer == 'default':
            return self.single_structural_mutation
        else:
            error_string = "Invalid structural_mutation_surer {!r}".format(
                self.structural_mutation_surer)
            raise RuntimeError(error_string)


class WannGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
        self.fitMax = 0.0

    @classmethod
    def parse_config(cls, param_dict):
        param_dict[ 'node_gene_type' ] = WannNodeGene
        param_dict[ 'connection_gene_type' ] = WannConnectionGene
        return WannGenomeConfig(param_dict)

    def configure_crossover(self, genome1, genome2, config):
        """ Configure a new genome by crossover from two parent genomes. """
        assert isinstance(genome1.fitness, (int, float))
        assert isinstance(genome2.fitness, (int, float))
        if genome1.rank < genome2.rank:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # Inherit connection genes
        for key, cg1 in iteritems(parent1.connections):
            cg2 = parent2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[ key ] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[ key ] = cg1.crossover(cg2)

        # Inherit node genes
        parent1_set = parent1.nodes
        parent2_set = parent2.nodes

        for key, ng1 in iteritems(parent1_set):
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[ key ] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[ key ] = ng1.crossover(ng2)

    def mutate(self, config):
        """ Mutates this genome. """
        topology_roulette = np.array(
            (config.conn_add_prob, config.conn_add_prob, config.activation_mutate_rate,
             config.conn_delete_prob,config.node_delete_prob, config.enabled_mutate_rate))

        spin = np.random.rand() * np.sum(topology_roulette)
        slot = topology_roulette[ 0 ]
        choice = topology_roulette.size
        for i in range(1, topology_roulette.size):
            if spin < slot:
                choice = i
                break
            else:
                slot += topology_roulette[ i ]

        # Add Connection
        if choice is 1:
            self.mutate_add_connection(config)

        # Add Node
        elif choice is 2:
            self.mutate_add_node(config)

        # Mutate Activation
        elif choice is 3:
            if len(self.nodes) > 0:
                random.choice(list(self.nodes.values())).mutate_activation(config)

        # Delete connection
        elif choice is 4:
            self.mutate_delete_connection()

        # Delete node
        elif choice is 5:
            self.mutate_delete_node(config)

        # Mutate Connection
        elif choice is 6:
            if len(self.connections) > 0:
                random.choice(list(self.connections.values())).mutate(config)
