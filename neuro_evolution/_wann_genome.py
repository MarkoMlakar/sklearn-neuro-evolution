import neat
from neat.six_util import iteritems
import numpy as np
import random


class WannGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
        self.fitMax = 0.0

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

        # TODO: Set this values as parameters!!
        topoRoulette = np.array((0.25, 0.25, 0.05, 0.50))

        spin = np.random.rand() * np.sum(topoRoulette)
        slot = topoRoulette[ 0 ]
        choice = topoRoulette.size
        for i in range(1, topoRoulette.size):
            if spin < slot:
                choice = i
                break
            else:
                slot += topoRoulette[ i ]

        # Add Connection
        if choice is 1:
            self.mutate_add_connection(config)

        # Add Node
        elif choice is 2:
            self.mutate_add_node(config)

        # Mutate Connection
        elif choice is 3:
            if len(self.connections) > 0:
                random.choice(list(self.connections.values())).mutate(config)

        # Mutate Activation
        elif choice is 4:
            if len(self.nodes) > 0:
                random.choice(list(self.nodes.values())).mutate(config)
