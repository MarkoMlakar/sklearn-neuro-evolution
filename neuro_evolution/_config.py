class NEATConfig:
    def __init__(self, fitness_criterion, fitness_threshold, pop_size,
                 reset_on_extinction, no_fitness_termination):
        self.fitness_criterion = fitness_criterion
        self.fitness_threshold = fitness_threshold
        self.pop_size = pop_size
        self.reset_on_extinction = reset_on_extinction
        self.no_fitness_termination = no_fitness_termination


class GenomeConfig:
    def __init__(self, activation_default, activation_mutate_rate, activation_options,
                 aggregation_default, aggregation_mutate_rate, aggregation_options, bias_init_mean,
                 bias_init_stdev, bias_max_value, bias_min_value, bias_mutate_power, bias_mutate_rate,
                 bias_replace_rate, compatibility_disjoint_coefficient, compatibility_weight_coefficient,
                 conn_add_prob, conn_delete_prob, enabled_default, enabled_mutate_rate, feed_forward,
                 initial_connection, node_add_prob, node_delete_prob, num_hidden, num_inputs, num_outputs,
                 response_init_mean, response_init_stdev, response_max_value, response_min_value,
                 response_mutate_power, response_mutate_rate, response_replace_rate, weight_init_mean,
                 weight_init_stdev, weight_max_value, weight_min_value, weight_mutate_power,
                 weight_mutate_rate, weight_replace_rate):

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
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
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


class ReproductionConfig:
    def __init__(self, elitism, survival_threshold):
        self.elitism = elitism
        self.survival_threshold = survival_threshold


class StagnationConfig:
    def __init__(self, species_fitness_func, max_stagnation, species_elitism):
        self.species_fitness_func = species_fitness_func
        self.max_stagnation = max_stagnation
        self.species_elitism = species_elitism


class SpeciesConfig:
    def __init__(self, compatibility_threshold):
        self.compatibility_threshold = compatibility_threshold


class ConfigParser:
    def __init__(self, genome_type, reproduction_type, species_set_type, stagnation_type, neat_config, genome_config,
                 reproduction_config, species_config, stagnation_config):
        self.genome_type = genome_type
        self.reproduction_type = reproduction_type
        self.species_set_type = species_set_type
        self.stagnation_type = stagnation_type

        # Assign NEAT attributes
        for attr, value in neat_config.__dict__.items():
            setattr(self, attr, value)

        # Assign default Genome attributes
        genome_config_dict = {}
        for attr, value in genome_config.__dict__.items():
            genome_config_dict.update({attr: value})

        # Assign default Reproduction attributes
        reproduction_config_dict = {}
        for attr, value in reproduction_config.__dict__.items():
            reproduction_config_dict.update({attr: value})

        # # Assign default Species attributes
        species_config_dict = {}
        for attr, value in species_config.__dict__.items():
            species_config_dict.update({attr: value})

        # Assign default Stagnation attributes
        stagnation_config_dict = {}
        for attr, value in stagnation_config.__dict__.items():
            stagnation_config_dict.update({attr: value})

        self.genome_config = self.genome_type.parse_config(genome_config_dict)
        self.reproduction_config = self.reproduction_type.parse_config(reproduction_config_dict)
        self.species_set_config = self.species_set_type.parse_config(species_config_dict)
        self.stagnation_config = self.stagnation_type.parse_config(stagnation_config_dict)