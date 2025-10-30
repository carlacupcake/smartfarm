"""ga_run.py."""
import numpy as np

from .ga_bounds import DesignSpaceBounds
from .ga_member import Member
from .ga_params import GeneticAlgorithmParams
from .ga_population import Population
from .ga_result import GeneticAlgorithmResult

from ..model.model_carrying_capacities import ModelCarryingCapacities
from ..model.model_disturbances import ModelDisturbances
from ..model.model_growth_rates import ModelGrowthRates
from ..model.model_initial_conditions import ModelInitialConditions
from ..model.model_params import ModelParams
from ..model.model_typical_disturbances import ModelTypicalDisturbances


class GeneticAlgorithm:

    def __init__(
        self,
        bounds:               DesignSpaceBounds,
        ga_params:            GeneticAlgorithmParams,
        carrying_capacities:  ModelCarryingCapacities,
        disturbances:         ModelDisturbances,
        growth_rates:         ModelGrowthRates,
        initial_conditions:   ModelInitialConditions,
        model_params:         ModelParams,
        typical_disturbances: ModelTypicalDisturbances,
        gen_counter:          bool = True):

        self.bounds               = bounds
        self.ga_params            = ga_params
        self.carrying_capacities  = carrying_capacities
        self.disturbances         = disturbances
        self.growth_rates         = growth_rates
        self.initial_conditions   = initial_conditions
        self.model_params         = model_params
        self.typical_disturbances = typical_disturbances
        self.gen_counter          = gen_counter


    def run(self) -> GeneticAlgorithmResult:
        """
        Executes the Genetic Algorithm (GA) optimization process.

        Initializes a population, evaluates costs, and iteratively
        evolves the population through breeding and selection to minimize
        the cost function over multiple generations. The best and average costs
        for each generation are tracked, and the final population is returned
        alongside optimization results.

        Args:
            TODO

        Returns
        -------
            GeneticAlgorithmResult
        """

        # Unpack necessary attributes from self
        num_parents     = self.ga_params.num_parents
        num_kids        = self.ga_params.num_kids
        num_members     = self.ga_params.num_members
        num_generations = self.ga_params.num_generations

        lower_bounds = self.bounds.lower_bounds
        upper_bounds = self.bounds.upper_bounds

        # Initialize arrays to store the cost and original indices of each generation
        all_costs = np.ones((num_generations, num_members))

        # Initialize arrays to store best performer and parent avg
        lowest_costs = np.zeros(num_generations)     # best cost
        avg_parent_costs = np.zeros(num_generations) # avg cost of parents

        # Generation counter
        g = 0

        # Initialize array to store costs for current generation
        costs = np.zeros(num_members)

        # Randomly populate first generation
        population = Population(
            bounds               = self.bounds,
            ga_params            = self.ga_params,
            carrying_capacities  = self.carrying_capacities,
            disturbances         = self.disturbances,
            growth_rates         = self.growth_rates,
            initial_conditions   = self.initial_conditions,
            model_params         = self.model_params,
            typical_disturbances = self.typical_disturbances
        )
        population.set_random_values(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            start_member=0
        )

        # Calculate the costs of the first generation
        population.set_costs()

        # Sort the costs of the first generation
        [sorted_costs, sorted_indices] = population.sort_costs()
        all_costs[g, :] = sorted_costs.reshape(1, num_members)

        # Store the cost of the best performer and average cost of the parents
        lowest_costs[g] = np.min(sorted_costs)
        avg_parent_costs[g] = np.mean(sorted_costs[0:num_parents])

        # Update population based on sorted indices
        population.set_order_by_costs(sorted_indices)

        # Perform all later generations
        while g < num_generations:

            if self.gen_counter:
                print(f"Generation {g} of {num_generations}")

            # Retain the parents from the previous generation
            costs[0:num_parents] = sorted_costs[0:num_parents]

            # Select top parents from population to be breeders
            for p in range(0, num_parents, 2):
                phi1, phi2 = np.random.rand(2)
                kid1 = phi1 * population.values[p, :] + (1-phi1) * population.values[p+1, :]
                kid2 = phi2 * population.values[p, :] + (1-phi2) * population.values[p+1, :]

                # Append offspring to population, overwriting old population members
                population.values[num_parents+p,   :] = kid1
                population.values[num_parents+p+1, :] = kid2

                # Cast offspring to members and evaluate costs
                kid1 = Member(
                    ga_params            = self.ga_params,
                    bounds               = self.bounds,
                    carrying_capacities  = self.carrying_capacities,
                    disturbances         = self.disturbances,
                    growth_rates         = self.growth_rates,
                    initial_conditions   = self.initial_conditions,
                    model_params         = self.model_params,
                    typical_disturbances = self.typical_disturbances,
                    values               = kid1)
                kid2 = Member(
                    ga_params            = self.ga_params,
                    bounds               = self.bounds,
                    carrying_capacities  = self.carrying_capacities,
                    disturbances         = self.disturbances,
                    growth_rates         = self.growth_rates,
                    initial_conditions   = self.initial_conditions,
                    model_params         = self.model_params,
                    typical_disturbances = self.typical_disturbances,
                    values               = kid2)
                costs[num_parents+p]   = kid1.get_cost()
                costs[num_parents+p+1] = kid2.get_cost()

            # Randomly generate new members to fill the rest of the population
            parents_plus_kids = num_parents + num_kids
            population.set_random_values(
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                start_member=parents_plus_kids
            )

            # Calculate the costs of the gth generation
            population.set_costs()

            # Sort the costs for the gth generation
            [sorted_costs, sorted_indices] = population.sort_costs()
            all_costs[g, :] = sorted_costs.reshape(1, num_members)

            # Store the cost of the best performer and average cost of the parents
            lowest_costs[g] = np.min(sorted_costs)
            avg_parent_costs[g] = np.mean(sorted_costs[0:num_parents])

            # Update population based on sorted indices
            population.set_order_by_costs(sorted_indices)

            # Update the generation counter
            g = g + 1

        return GeneticAlgorithmResult(
            ga_params        = self.ga_params,
            final_population = population,
            lowest_costs     = lowest_costs,
            avg_parent_costs = avg_parent_costs
        )
