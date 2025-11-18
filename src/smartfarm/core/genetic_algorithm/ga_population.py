"""ga_population.py."""
import boto3
import json
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    _LAMBDA_FUNCTION_NAME = "smartfarm-ga-eval"
    _LAMBDA_REGION_NAME   = "us-west-2"

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
        return [unique_values, unique_costs]


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
    

    def set_costs_with_lambda(
        self,
        lambda_client:        "boto3.client" = None,
        batch_size:           int = 1,
        max_parallel_batches: int = 128,
        verbose:              bool = True,
    ) -> "Population":
        """
        Calculate the costs for each member in the population by sending
        batched evaluation requests to AWS Lambda, with multiple batches
        invoked in parallel.

        Args:
            lambda_client:        Optional boto3 Lambda client.
            batch_size:           Number of members per Lambda invocation.
            max_parallel_batches: Max number of Lambda batches to run at once.
            verbose:              If True, prints timing info per batch.

        Returns:
            self (Population) with self.costs filled.
        """

        if lambda_client is None:
            lambda_client = boto3.client(
                "lambda",
                region_name=self._LAMBDA_REGION_NAME,
            )

        num_members = self.ga_params.num_members
        population_values = self.values  # shape: (num_members, n_params)

        # Serialize all members
        member_dicts = [
            self._member_values_to_dict(population_values[i, :])
            for i in range(num_members)
        ]

        # Build shared context once
        sim_context = self._build_sim_context_dict()

        # Prepare batches as (start, end) index pairs
        batches = [
            (start, min(start + batch_size, num_members))
            for start in range(0, num_members, batch_size)
        ]

        costs = np.zeros(num_members, dtype=float)

        def invoke_batch(start: int, end: int):
            """Invoke Lambda for members[start:end] and return (start, end, costs_array)."""
            batch = member_dicts[start:end]
            payload = {
                "members":     batch,
                "sim_context": sim_context,
            }

            t0 = time.time()
            response = lambda_client.invoke(
                FunctionName=self._LAMBDA_FUNCTION_NAME,
                InvocationType="RequestResponse",
                Payload=json.dumps(payload).encode("utf-8"),
            )
            t1 = time.time()

            raw_payload = response["Payload"].read().decode("utf-8")
            result = json.loads(raw_payload)

            body = result.get("body", result)
            if isinstance(body, str):
                body = json.loads(body)

            batch_costs = np.array(body["costs"], dtype=float)

            if batch_costs.shape[0] != (end - start):
                raise ValueError(
                    f"Lambda returned {batch_costs.shape[0]} costs for a batch "
                    f"of size {end - start} (start={start}, end={end})."
                )

            if verbose:
                print(
                    f"[Notebook] Batch {start}-{end-1} finished in {t1 - t0:.2f}s "
                    f"({end - start} members)"
                )

            return start, end, batch_costs

        # Fire batches in parallel
        max_workers = min(max_parallel_batches, len(batches))
        if verbose:
            print(
                f"[Notebook] Evaluating {num_members} members in {len(batches)} batches "
                f"of up to {batch_size}, with up to {max_workers} in parallel..."
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(invoke_batch, start, end)
                for (start, end) in batches
            ]

            for fut in as_completed(futures):
                start, end, batch_costs = fut.result()
                costs[start:end] = batch_costs

        self.costs = costs
        return self


    def set_order_by_costs(self, sorted_indices: np.ndarray) -> "Population":
        """
        Reorder population values (and costs) in-place based on sorted cost indices.
        """
        # Ensure indices are 1D integer array
        sorted_indices = np.asarray(sorted_indices).ravel().astype(int)

        self.values = self.values[sorted_indices, :]
        self.costs  = self.costs[sorted_indices]
        
        return self


    def sort_costs(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Sort costs in ascending order and return sorted costs and indices.

        Returns:
            sorted_costs (ndarray): shape (num_members,)
            sorted_indices (ndarray): shape (num_members,)
        """
        costs_1d       = self.costs.ravel()
        sorted_indices = np.argsort(costs_1d)
        sorted_costs   = costs_1d[sorted_indices]
        
        return sorted_costs, sorted_indices

    
    def _member_values_to_dict(self, values_row: np.ndarray) -> dict:
        """
        Convert one member's parameter vector into a JSON-serializable dict
        that the Lambda function knows how to interpret.
        """
        return {
            "values": values_row.tolist()
        }


    def _build_sim_context_dict(self) -> dict:
        """
        Build a JSON-serializable dict containing all simulation parameters
        that are shared across all members in the population.
        """

        model_params         = self.model_params
        disturbances         = self.disturbances
        typical_disturbances = self.typical_disturbances
        initial_conditions   = self.initial_conditions
        growth_rates         = self.growth_rates
        carrying_capacities  = self.carrying_capacities
        ga                   = self.ga_params

        sim_context = {
            # Time stepping / horizon
            "dt": float(model_params.dt),
            "total_time_steps": int(model_params.total_time_steps),
            "simulation_hours": int(model_params.simulation_hours),

            # Disturbances (hourly)
            "hourly_precipitation": np.asarray(disturbances.precipitation, dtype=float).tolist(),
            "hourly_temperature":   np.asarray(disturbances.temperature,   dtype=float).tolist(),
            "hourly_radiation":     np.asarray(disturbances.radiation,     dtype=float).tolist(),

            # Typical disturbances
            "W_typ": float(typical_disturbances.typical_water),
            "F_typ": float(typical_disturbances.typical_fertilizer),
            "T_typ": float(typical_disturbances.typical_temperature),
            "R_typ": float(typical_disturbances.typical_radiation),

            # Initial conditions
            "h0": float(initial_conditions.h0),
            "A0": float(initial_conditions.A0),
            "N0": float(initial_conditions.N0),
            "c0": float(initial_conditions.c0),
            "P0": float(initial_conditions.P0),

            # Growth rates
            "ah": float(growth_rates.ah),
            "aA": float(growth_rates.aA),
            "aN": float(growth_rates.aN),
            "ac": float(growth_rates.ac),
            "aP": float(growth_rates.aP),

            # Carrying capacities
            "kh": float(carrying_capacities.kh),
            "kA": float(carrying_capacities.kA),
            "kN": float(carrying_capacities.kN),
            "kc": float(carrying_capacities.kc),
            "kP": float(carrying_capacities.kP),

            # GA weights used in the cost
            "weight_fruit_biomass":  float(getattr(ga, "weight_fruit_biomass", 1.0)),
            "weight_irrigation":     float(getattr(ga, "weight_irrigation",    1.0)),
            "weight_fertilizer":     float(getattr(ga, "weight_fertilizer",    1.0)),
        }

        return sim_context
