"""ga_population.py."""
import numpy as np
from typing import Optional

from .ga_bounds import DesignSpaceBounds
from .ga_member import Member
from .ga_params import GeneticAlgorithmParams

from ..model.model_carrying_capacities import ModelCarryingCapacities
from ..model.model_disturbances import ModelDisturbances
from ..model.model_growth_rates import ModelGrowthRates
from ..model.model_initial_conditions import ModelInitialConditions
from ..model.model_params import ModelParams
from ..model.model_typical_disturbances import ModelTypicalDisturbances


class Population:
    """
    Class to hold the population of members.

    The class also implements methods to generate the initial population
    and sort the members based on their costs.
    """

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
        values:               Optional[np.ndarray] = None,
        costs:                Optional[np.ndarray] = None) -> None:

        self.bounds               = bounds
        self.ga_params            = ga_params
        num_members               = ga_params.num_members
        self.carrying_capacities  = carrying_capacities
        self.disturbances         = disturbances
        self.growth_rates         = growth_rates
        self.initial_conditions   = initial_conditions
        self.model_params         = model_params
        self.typical_disturbances = typical_disturbances

        self.values = values
        if self.values is None:
            self.values = np.zeros((num_members, self.bounds.upper_bounds.shape[0]))

        self.costs = costs
        if self.costs is None:
            self.costs = np.zeros((num_members, 1))


    def get_unique_designs(self) -> "Population":
        """
        Retrieves the unique designs from the population based on their costs.

        Returns:
            list: A list containing three elements:
            - unique_values (ndarray), unique population members corresponding to unique_costs.
            - unique_eff_props (ndarray), effective properties corresponding to unique_costs.
            - unique_costs (ndarray)
        """
        self.set_costs()
        final_costs = self.costs
        rounded_costs = np.round(final_costs, decimals=3)

        # Obtain unique members and costs
        [unique_costs, unique_indices] = np.unique(rounded_costs, return_index=True)
        unique_values = self.values[unique_indices]
        unique_eff_props = self.eff_props[unique_indices]
        return [unique_values, unique_eff_props, unique_costs]


    def set_random_values(
        self,
        lower_bounds: np.ndarray = None,
        upper_bounds: np.ndarray = None,
        start_member: int = 0) -> "Population":
        """
        Sets random values for the properties of each member in the population.

        Args:
            lower_bounds (ndarray, optional)
            upper_bounds (ndarray, optional)
            start_member (int, optional): index of the first member to update in the population

        Returns
            self (Population)
        """

        # Fill in the population, not including the volume fractions
        population_size = len(self.values)
        for i in range(start_member, population_size):
            self.values[i, :] = np.random.uniform(lower_bounds, upper_bounds)

        return self


    def set_costs(self) -> "Population":
        """
        Calculates the costs for each member in the population.

        Returns:
            self (Population)
        """

        population_values = self.values
        num_members = self.ga_params.num_members
        costs = np.zeros(num_members)
        
        for i in range(num_members):
            this_member = Member(
                ga_params            = self.ga_params,
                carrying_capacities  = self.carrying_capacities,
                disturbances         = self.disturbances,
                growth_rates         = self.growth_rates,
                initial_conditions   = self.initial_conditions,
                model_params         = self.model_params,
                typical_disturbances = self.typical_disturbances,
                values               = population_values[i, :])
            costs[i] = this_member.get_cost()

        self.costs = costs
        return self


    def set_order_by_costs(self, sorted_indices: np.ndarray = None) ->  "Population":
        """
        Reorders the population based on the sorted indices of costs.

        Args:
            sorted_indices (ndarray)

        Returns:
            self (Population)
        """

        temporary = np.zeros((
            self.ga_params.num_members,
            len(self.bounds.upper_bounds)))
        
        for i in range(len(sorted_indices)):
            temporary[i,:] = self.values[int(sorted_indices[i]), :]

        self.values = temporary
        return self


    def sort_costs(self) -> list:
        """
        Sorts the costs and returns the sorted values along with their corresponding indices.

        Returns:
            A list containing two arrays:
            - sorted costs (ndarray)
            - sorted_indices (ndarray), indices that would sort the original `self.costs`.
        """

        sorted_costs   = np.sort(self.costs, axis=0)
        sorted_indices = np.argsort(self.costs, axis=0)
        return [sorted_costs, sorted_indices]
