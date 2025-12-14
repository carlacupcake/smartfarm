# lambda_mpc.py
import mpmath as mp
import numpy as np

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from typing import Optional, Dict, Tuple

from mpc_shadow.mpc import MPC
from mpc_shadow.mpc_params import MPCParams
from mpc_shadow.mpc_bounds import ControlInputBounds

from model_shadow.model_carrying_capacities import ModelCarryingCapacities
from model_shadow.model_disturbances import ModelDisturbances
from model_shadow.model_growth_rates import ModelGrowthRates
from model_shadow.model_initial_conditions import ModelInitialConditions
from model_shadow.model_params import ModelParams
from model_shadow.model_typical_disturbances import ModelTypicalDisturbances
from model_shadow.model_sensitivities import ModelSensitivities

#print(">>> LOADED lambda_mpc.py <<<")

def gaussian_kernel(
        mu: float,
        sigma_steps: float,
        length: int
    ) -> np.ndarray:
    """
    Construct a normalized discrete Gaussian kernel for modeling delayed
    nutrient absorption (temporal spreading) of an input signal.

    Args:
        mu (float):
            Center of the Gaussian in index units (e.g., time steps).
        sigma_steps (float):
            Standard deviation of the Gaussian in time steps, controlling how
            broadly the influence spreads across the kernel.
        length (int):
            Total number of discrete samples in the kernel.

    Returns:
        np.ndarray:
            A 1D array of length `length` containing non-negative values that
            sum to one, suitable for convolution or delayed-response modeling.
    """
    k = np.arange(length)
    g = np.exp(-0.5 * ((k - mu) / sigma_steps)**2)
    g /= g.sum() # normalize so area ~ 1; scale by instantaneous disturbance/control input later
    return g


def get_mu_from_sigma(
        sigma:    float,        # standard deviation of the Gaussian
        mu_guess: float = 100.0 # initial guess for mu for the root-finder
    ) -> float:
    """
    Compute the value of `mu` that satisfies a target Gaussian tail probability.
    This routine solves for `mu` such that the error function `erf(mu/(sqrt(2)*sigma))`
    equals 0.95, meaning it returns the point where approximately 95% of a
    zero-mean Gaussian lies between -mu and mu.

    Args:
        sigma (float):
            Standard deviation of the Gaussian distribution used in the
            implicit equation. Must be positive.
        mu_guess (float, optional):
            Initial estimate supplied to the root-finding algorithm. A good
            starting point can speed convergence, especially for large `sigma`.

    Returns:
        float:
            The solved value of `mu` for which `erf(mu/(sqrt(2)*sigma)) = 0.95`,
            computed using numerical root finding.
    """
    f = lambda mu: mp.erf(mu/(np.sqrt(2) * sigma)) - 0.95
    mu = float(mp.findroot(f, mu_guess))
    return mu


def compute_fir_horizon(
    kernel:         np.ndarray,
    mass_threshold: float = 0.95
    ) -> int:
    """
    Return the smallest L such that the first L taps of `kernel`
    contain at least `mass_threshold` of the total kernel mass.

    TODO
    """
    k = np.asarray(kernel, dtype=float)
    total = np.sum(k)
    if abs(total) < 1e-12:
        # Degenerate kernel; just fall back to full length
        return len(k)
    cum = np.cumsum(k) / total
    idx = np.where(cum >= mass_threshold)[0]
    if idx.size == 0:
        return len(k)
    return int(idx[0] + 1)  # +1 because length = index+1
        

def get_sim_inputs_from_hourly(
        hourly_array:     np.ndarray,
        dt:               float,
        simulation_hours: int,
        mode: str = 'copy' # 'copy' or 'split'
    ) -> np.ndarray:
    """
    Expand an hourly time series into a per–time-step simulation array. Each
    hour is either copied across all sub-steps (e.g. 1 inch over 5 steps -> 1, 1, 1, 1, 1)
    or evenly split across them (e.g. 1 inch over 5 steps -> 0.2, 0.2, 0.2, 0.2, 0.2),
    depending on the chosen mode.

    Args:
        hourly_array (array-like):
            Hourly input values to be upsampled for the simulation horizon.
        dt (float):
            Simulation time-step size in hours (e.g., 0.1 → 10 steps per hour).
        simulation_hours (int):
            Total number of hours in the simulation window; truncates the input.
        mode (str, optional):
            `'copy'` repeats the hourly value at each sub-step; `'split'`
            divides the hourly value evenly across sub-steps.

    Returns:
        ndarray:
            A 1D array of length `simulation_hours / dt` containing the
            interpolated per–time-step values.
    """

    # Initialize the output array
    total_time_steps = int(simulation_hours / dt)
    simulation_array = np.zeros(total_time_steps)

    # Truncate hourly_df to length simulation_hours
    hourly_array = hourly_array[:simulation_hours]

    # Time steps per hour
    time_steps_per_hour = int(1 / dt)

    # Loop over the hours and fill the simulation_df with the extra timesteps
    for i in range(simulation_hours):
        hourly_value = hourly_array[i]

        if mode == 'copy':
            simulation_array[i*time_steps_per_hour:(i+1)*time_steps_per_hour] = hourly_value
        elif mode == 'split':
            simulation_array[i*time_steps_per_hour:(i+1)*time_steps_per_hour] = hourly_value/time_steps_per_hour

    return simulation_array


def mpc_with_lambda(
        weights_dict: dict,
        ctx: dict
    ) -> float:
    """
    Evaluate a member’s cost inside an AWS Lambda environment by running the
    same forward-Euler plant-growth simulation as `get_cost`, using only the
    dictionaries provided by the Lambda handler.

    Args:
        member_dict (dict):
            Dictionary containing the member’s design variables under the key
            `"values"` in the order
            [irrigation_frequency, irrigation_amount,
             fertilizer_frequency, fertilizer_amount].
        ctx (dict):
            Dictionary providing all model parameters, disturbances,
            growth-rate values, carrying capacities, and cost-function weights
            required to run the simulation.

    Returns:
        float:
            The cost value (negative net revenue) computed from final fruit
            biomass minus weighted irrigation and fertilizer usage.
    """

    #print(">>> INSIDE get_cost_with_lambda <<<")

    # Time-stepping
    dt               = ctx["dt"] # hours/step
    total_time_steps = ctx["total_time_steps"]
    simulation_hours = ctx["simulation_hours"] # hours
    closed_form      = False

    # Disturbances (hourly)
    precipitation = ctx["precipitation"]
    temperature   = ctx["temperature"]
    radiation     = ctx["radiation"]
    
    # Set model params for time stepping
    model_params = ModelParams(
        dt               = dt,  # hours/step
        total_time_steps = total_time_steps,
        simulation_hours = simulation_hours, # hours
        closed_form      = closed_form,
        verbose          = False
    )

    # Set model growth rates
    growth_rates = ModelGrowthRates()

    # Set model carrying capacities
    carrying_capacities = ModelCarryingCapacities()

    # Set model sensitivities
    sensitivities = ModelSensitivities()

    # Set model initial conditions
    initial_conditions = ModelInitialConditions(
        h0=carrying_capacities.kh/simulation_hours, # m/hr
        A0=carrying_capacities.kA/simulation_hours, # m2/hr
        N0=carrying_capacities.kN/simulation_hours, # number/hr
        c0=carrying_capacities.kc/simulation_hours, # number/hr
        P0=carrying_capacities.kP/simulation_hours  # kg/hr
    )

    # Typical Distrubances
    default_typical_disturbances = ModelTypicalDisturbances()
    typical_disturbances = ModelTypicalDisturbances(
        typical_water       = default_typical_disturbances.typical_water * dt,
        typical_fertilizer  = default_typical_disturbances.typical_fertilizer * dt,
        typical_temperature = default_typical_disturbances.typical_temperature * dt,
        typical_radiation   = default_typical_disturbances.typical_radiation * dt
    )

    # Set ModelDisturbances
    disturbances = ModelDisturbances(
        precipitation = 0.5*precipitation, # imagine it's a drought year
        radiation     = radiation,
        temperature   = temperature
    )

    # Set MPC Params and Bounds
    mpc_params = MPCParams(
        weight_irrigation    = weights_dict["weight_irrigation"],
        weight_fertilizer    = weights_dict["weight_fertilizer"],
        weight_fruit_biomass = weights_dict["weight_fruit_biomass"],
        weight_cumulative_average_water       = weights_dict["weight_cumulative_average_water"],
        weight_cumulative_average_fertilizer  = weights_dict["weight_cumulative_average_fertilizer"],
        weight_cumulative_average_temperature = weights_dict["weight_cumulative_average_temperature"],
        weight_cumulative_average_radiation   = weights_dict["weight_cumulative_average_radiation"]
    )
    bounds = ControlInputBounds()

    # Construct an instance of the Member class and get the cost
    mpc = MPC(
        carrying_capacities  = carrying_capacities,
        disturbances         = disturbances,
        growth_rates         = growth_rates,
        initial_conditions   = initial_conditions,
        model_params         = model_params,
        typical_disturbances = typical_disturbances,
        sensitivities        = sensitivities,
        mpc_params           = mpc_params,
        bounds               = bounds
    )

    # Run the MPC
    mpc_result = mpc.run()

    # Extract results
    P = mpc_result["P"]
    irrigation = mpc_result["irrigation"]
    fertilizer = mpc_result["fertilizer"]

    return P, irrigation, fertilizer
