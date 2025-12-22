# mpc_helpers.py
import io
import json
import boto3
import numpy as np
import pandas as pd

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

from model_shadow.model_helpers import get_sim_inputs_from_hourly


def mpc_from_context(
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
    # Time-stepping
    dt               = ctx["dt"] # hours/step
    total_time_steps = ctx["total_time_steps"]
    simulation_hours = ctx["simulation_hours"] # hours
    closed_form      = False

    # Disturbances from s3
    data_bucket = ctx["data_bucket"]
    data_key    = ctx["data_key"]

    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=data_bucket, Key=data_key)
    csv_bytes = obj["Body"].read()

    hourly = pd.read_csv(io.BytesIO(csv_bytes))

    precipitation = get_sim_inputs_from_hourly(
        hourly_array     = 0.5 * hourly["Hourly Precipitation (in)"].to_numpy(), # drought
        dt               = dt,
        simulation_hours = simulation_hours,
        mode             = "split",
    )

    temperature = get_sim_inputs_from_hourly(
        hourly_array     = hourly["Temperature (C)"].to_numpy(),
        dt               = dt,
        simulation_hours = simulation_hours,
        mode             = "split",
    )

    radiation = get_sim_inputs_from_hourly(
        hourly_array     = hourly["Hourly Radiation (W/m2)"].to_numpy(),
        dt               = dt,
        simulation_hours = simulation_hours,
        mode             = "split",
    )

    disturbances = ModelDisturbances(
        precipitation=precipitation,
        radiation=radiation,
        temperature=temperature,
    )
    
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
    final_fruit_biomass = mpc_result["P"][-1]
    sum_irrigation = np.sum(mpc_result["irrigation"])
    sum_fertilizer = np.sum(mpc_result["fertilizer"])

    # Report whether MPC succeeded or not
    solve_status = mpc_result.get("termination", "unknown")
    if solve_status.lower() in ("optimal", "locally optimal", "ok"):
        mpc_status = "optimal"
    else:
        mpc_status = f"solver_{solve_status}"

    return {
        "final_fruit_biomass": final_fruit_biomass,
        "sum_irrigation":      sum_irrigation,
        "sum_fertilizer":      sum_fertilizer,
        "mpc_status":          mpc_status,
    }
