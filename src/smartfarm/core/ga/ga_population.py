# ga_population.py
import boto3
import json
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from pure_eval import Evaluator

from .ga_bounds import DesignSpaceBounds
from .ga_member import Member
from .ga_params import GeneticAlgorithmParams

from ..model.model_carrying_capacities import ModelCarryingCapacities
from ..model.model_disturbances import ModelDisturbances
from ..model.model_growth_rates import ModelGrowthRates
from ..model.model_initial_conditions import ModelInitialConditions
from ..model.model_params import ModelParams
from ..model.model_typical_disturbances import ModelTypicalDisturbances
from ..model.model_sensitivities import ModelSensitivities

from ga_member_cpp import Evaluator


class Population:
    """
    Class to hold the population of members.

    The class also implements methods to generate the initial population
    and sort the members based on their costs.
    """

    _LAMBDA_FUNCTION_NAME = "smartfarm-ga-eval"
    _LAMBDA_REGION_NAME   = "us-west-1"

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
        sensitivities:        ModelSensitivities,
        values:               Optional[np.ndarray] = None,
        costs:                Optional[np.ndarray] = None) -> None:

        self.bounds               = bounds
        self.ga_params            = ga_params
        self.num_members          = values.shape[0] if values is not None else ga_params.num_members
        self.carrying_capacities  = carrying_capacities
        self.disturbances         = disturbances
        self.growth_rates         = growth_rates
        self.initial_conditions   = initial_conditions
        self.model_params         = model_params
        self.typical_disturbances = typical_disturbances
        self.model_sensitivities  = sensitivities

        self.values = values
        if self.values is None:
            self.values = np.zeros((self.num_members, self.bounds.upper_bounds.shape[0]))

        self.costs = costs
        if self.costs is None:
            self.costs = np.zeros((self.num_members, 1))


    def get_unique_designs(self):
        """
        Retrieves the unique designs from the population based on their member values.

        Returns:
            list: [unique_values (ndarray), unique_costs (ndarray)]
        """
        # Make sure costs are up to date
        self.set_costs_with_lambda(verbose=False)

        tol_digits = 10  # adjust as needed
        rounded_values = np.round(self.values, tol_digits)
        unique_values, unique_indices = np.unique(rounded_values, axis=0, return_index=True)

        # Grab the corresponding costs
        unique_costs = self.costs[unique_indices]

        # Sort the unique designs based on their costs
        unique_indices = np.argsort(unique_costs, axis=0)
        unique_values = unique_values[unique_indices]
        unique_costs = unique_costs[unique_indices]

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

        Returns:
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
        costs = np.zeros(self.num_members)
        
        for i in range(self.num_members):
            this_member = Member(
                ga_params            = self.ga_params,
                carrying_capacities  = self.carrying_capacities,
                disturbances         = self.disturbances,
                growth_rates         = self.growth_rates,
                initial_conditions   = self.initial_conditions,
                model_params         = self.model_params,
                typical_disturbances = self.typical_disturbances,
                sensitivities        = self.model_sensitivities,
                values               = population_values[i, :]
            )
            costs[i] = this_member.get_cost()

        self.costs = costs
        return self
    

    def set_costs_with_cpp(self) -> "Population":
        """
        Calculates the costs for each member in the population using the C++ Evaluator.

        Args:
            verbose: If True, print basic timing/debug info.

        Returns:
            self (Population)
        """
        # Build context dict for C++ evaluator
        context = self._build_sim_context_dict_cpp()

        # Ensure numpy-friendly types (pybind11 is happiest with contiguous float64)
        values = np.ascontiguousarray(self.values, dtype=np.float64)

        # Create evaluator (stores expanded disturbances/kernels etc. internally)
        evaluator = Evaluator(context)
        t0 = time.time()
        costs = evaluator.evaluate_population(values)
        t1 = time.time()
        print(f"Time taken to evaluate population with C++ evaluator: {t1 - t0} seconds")

        # Convert to a clean 1D numpy array
        costs = np.asarray(costs, dtype=np.float64).reshape(-1)

        if costs.shape[0] != values.shape[0]:
            raise RuntimeError(
                f"C++ evaluator returned {costs.shape[0]} costs, expected {values.shape[0]}."
            )

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

        population_values = self.values  # shape: (num_members, n_params)

        # Serialize all members
        member_dicts = [
            self._member_values_to_dict(population_values[i, :])
            for i in range(self.num_members)
        ]

        # Build shared context once
        context = self._build_sim_context_dict_lambda()

        # Prepare batches as (start, end) index pairs
        batches = [
            (start, min(start + batch_size, self.num_members))
            for start in range(0, self.num_members, batch_size)
        ]

        costs = np.zeros(self.num_members, dtype=float)

        def invoke_batch(start: int, end: int):
            """Invoke Lambda for members[start:end] and return (start, end, costs_array)."""
            batch = member_dicts[start:end]
            payload = {
                "members": batch,
                "context": context,
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
                f"[Notebook] Evaluating {self.num_members} members in {len(batches)} batches "
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
        
        Args:
            sorted_indices (np.ndarray):
                A 1D array of integer indices indicating the desired ordering of
                population members, typically produced by `np.argsort(self.costs)`.

        Returns:
            Population:
                The population instance with `values` and `costs` reordered
                according to the provided index sequence.
        """

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


    def _build_sim_context_dict_lambda(self) -> dict:
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
        model_sensitivities  = self.model_sensitivities
        ga                   = self.ga_params

        context = {

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

            # Nutrient absorption/metalysis sensitivities (sigmas)
            "sigma_W": float(model_sensitivities.sigma_W),
            "sigma_F": float(model_sensitivities.sigma_F),
            "sigma_T": float(model_sensitivities.sigma_T),
            "sigma_R": float(model_sensitivities.sigma_R),

            # GA weights used in the cost
            "weight_height":         float(getattr(ga, "weight_height",        1.0)),
            "weight_leaf_area":      float(getattr(ga, "weight_leaf_area",     1.0)),
            "weight_fruit_biomass":  float(getattr(ga, "weight_fruit_biomass", 1.0)),
            "weight_irrigation":     float(getattr(ga, "weight_irrigation",    1.0)),
            "weight_fertilizer":     float(getattr(ga, "weight_fertilizer",    1.0)),
        }

        return context
    

    def _build_sim_context_dict_cpp(self) -> dict:
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
        model_sensitivities  = self.model_sensitivities
        ga                   = self.ga_params

        # Build kernel_W here for C++ context
        dummy_member = Member(
            ga_params            = self.ga_params,
            carrying_capacities  = self.carrying_capacities,
            disturbances         = self.disturbances,
            growth_rates         = self.growth_rates,
            initial_conditions   = self.initial_conditions,
            model_params         = self.model_params,
            typical_disturbances = self.typical_disturbances,
            sensitivities        = self.model_sensitivities,
            values               = np.zeros_like(self.values[0, :])
        )

        context = {

            # Time stepping / horizon
            "dt": float(model_params.dt),
            "total_time_steps": int(model_params.total_time_steps),
            "simulation_hours": int(model_params.simulation_hours),

            # Sensitivity values
            "alpha":                float(model_sensitivities.alpha),
            "beta_divergence":      float(model_sensitivities.beta_divergence),
            "beta_nutrient_factor": float(model_sensitivities.beta_nutrient_factor),
            "epsilon":              float(model_sensitivities.epsilon),
            "sigma_W":              float(model_sensitivities.sigma_W),
            "sigma_F":              float(model_sensitivities.sigma_F),
            "sigma_T":              float(model_sensitivities.sigma_T),
            "sigma_R":              float(model_sensitivities.sigma_R),

            # Kernel attributes
            "kernel_W": np.asarray(dummy_member.kernel_W, dtype=np.float64),
            "kernel_F": np.asarray(dummy_member.kernel_F, dtype=np.float64),
            "kernel_T": np.asarray(dummy_member.kernel_T, dtype=np.float64),
            "kernel_R": np.asarray(dummy_member.kernel_R, dtype=np.float64),

            "fir_horizon_W": int(dummy_member.fir_horizon_W),
            "fir_horizon_F": int(dummy_member.fir_horizon_F),
            "fir_horizon_T": int(dummy_member.fir_horizon_T),
            "fir_horizon_R": int(dummy_member.fir_horizon_R),

            # Disturbances (hourly)
            "hourly_precipitation": np.asarray(disturbances.precipitation, dtype=np.float64),
            "hourly_temperature":   np.asarray(disturbances.temperature,   dtype=np.float64),
            "hourly_radiation":     np.asarray(disturbances.radiation,     dtype=np.float64),

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
            "weight_height":         float(getattr(ga, "weight_height",        1.0)),
            "weight_leaf_area":      float(getattr(ga, "weight_leaf_area",     1.0)),
            "weight_fruit_biomass":  float(getattr(ga, "weight_fruit_biomass", 1.0)),
            "weight_irrigation":     float(getattr(ga, "weight_irrigation",    1.0)),
            "weight_fertilizer":     float(getattr(ga, "weight_fertilizer",    1.0)),
        }

        return context
