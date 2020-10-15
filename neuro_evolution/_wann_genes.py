from neat.genes import BaseGene, DefaultNodeGene, DefaultConnectionGene
from random import choice

class WannBaseGene(BaseGene):

    def __init__(self, key):
        super().__init__(key)

    def mutate_activation(self, config):

        # Mutate activation function
        options = config.activation_options
        new_activation = choice([x for x in options if x != self.activation])
        setattr(self, 'activation', new_activation)

        # TODO: Refactor this!
        # Removes the string attribute for mutating activation functions because we did that already
        other_attributes = self._gene_attributes.copy()
        del other_attributes[2]
        for a in other_attributes:
            v = getattr(self, a.name)
            setattr(self, a.name, a.mutate_value(v, config))


class WannNodeGene(WannBaseGene,DefaultNodeGene):

    def __init__(self, key):
        super().__init__(key)
        assert isinstance(key, int), "WannNodeGene key must be an int, not {!r}".format(key)
        WannBaseGene.__init__(self, key)



class WannConnectionGene(WannBaseGene, DefaultConnectionGene):
    def __init__(self, key):
        super().__init__(key)
        assert isinstance(key, tuple), "WannConnectionGene key must be a tuple, not {!r}".format(key)
        WannBaseGene.__init__(self, key)