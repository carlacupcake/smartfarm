# mpc_ec2_run.py
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from mpc.mpc_helpers import mpc_from_context, get_sim_inputs_from_hourly


# Time stepping / horizon
dt               = 1.0     # hours/step
simulation_hours = 2900    # hours
closed_form      = False
verbose          = False
total_time_steps = int(simulation_hours / dt)

# Hourly precipitation, radiation, and temperature from CSV
hourly_disturbances = pd.read_csv("io/inputs/hourly_prcp_rad_temp_iowa.csv")

precipitation = get_sim_inputs_from_hourly(
    hourly_array     = 0.5 * hourly_disturbances["Hourly Precipitation (in)"].to_numpy(),  # drought year
    dt               = dt,
    simulation_hours = simulation_hours,
    mode             = "split",
)

temperature = get_sim_inputs_from_hourly(
    hourly_array     = hourly_disturbances["Temperature (C)"].to_numpy(),
    dt               = dt,
    simulation_hours = simulation_hours,
    mode             = "split",
)

radiation = get_sim_inputs_from_hourly(
    hourly_array     = hourly_disturbances["Hourly Radiation (W/m2)"].to_numpy(),
    dt               = dt,
    simulation_hours = simulation_hours,
    mode             = "split",
)

# Build the context passed to each MPC run
context = {
    # Time stepping / horizon
    "dt": float(dt),
    "total_time_steps": int(total_time_steps),
    "simulation_hours": int(simulation_hours),
    "closed_form": bool(closed_form),

    # Disturbances (time step)
    "precipitation": np.asarray(precipitation, dtype=float).tolist(),
    "temperature":   np.asarray(temperature,   dtype=float).tolist(),
    "radiation":     np.asarray(radiation,     dtype=float).tolist(),
}


# Generate weight combinations as list of dictionaries
weight_irrigation_options                     = [0.01, 0.1, 1.0, 10.0]
weight_fertilizer_options                     = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
weight_fruit_biomass_options                  = [1.0, 10.0, 100.0, 1000.0]
weight_cumulative_average_water_options       = [1.0]
weight_cumulative_average_fertilizer_options  = [1.0]
weight_cumulative_average_temperature_options = [1.0]
weight_cumulative_average_radiation_options   = [1.0]

weights = []
for weight_irrigation in weight_irrigation_options:
    for weight_fertilizer in weight_fertilizer_options:
        for weight_fruit_biomass in weight_fruit_biomass_options:
            for weight_cumulative_average_water in weight_cumulative_average_water_options:
                for weight_cumulative_average_fertilizer in weight_cumulative_average_fertilizer_options:
                    for weight_cumulative_average_temperature in weight_cumulative_average_temperature_options:
                        for weight_cumulative_average_radiation in weight_cumulative_average_radiation_options:
                            weights.append(
                                {
                                    "weight_irrigation":                     weight_irrigation,
                                    "weight_fertilizer":                     weight_fertilizer,
                                    "weight_fruit_biomass":                  weight_fruit_biomass,
                                    "weight_cumulative_average_water":       weight_cumulative_average_water,
                                    "weight_cumulative_average_fertilizer":  weight_cumulative_average_fertilizer,
                                    "weight_cumulative_average_temperature": weight_cumulative_average_temperature,
                                    "weight_cumulative_average_radiation":   weight_cumulative_average_radiation,
                                }
                            )
num_weights = len(weights)


# Worker function for a single MPC run
def run_single_mpc(idx_and_weight):
    """
    Run a single MPC simulation for a given weight dictionary.

    Args:
        idx_and_weight (tuple): (index, weights_dict)

    Returns:
        (int, dict):
            index, results_dict with summary metrics + raw arrays.
    """
    idx, w = idx_and_weight
    print(f"=== Running MPC for weight set {idx} ===", flush=True)

    P, irrigation, fertilizer = mpc_from_context(w, context)

    P = np.asarray(P, dtype=float)
    irrigation = np.asarray(irrigation, dtype=float)
    fertilizer = np.asarray(fertilizer, dtype=float)

    result = {
        "index": idx,
        "weights": w,
        "final_biomass": float(P[-1]),
        "total_irrigation": float(irrigation.sum()),
        "total_fertilizer": float(fertilizer.sum()),
        # optional: keep the full time series
        # "P_series": P.tolist(),
        # "irrigation_series": irrigation.tolist(),
        # "fertilizer_series": fertilizer.tolist(),
    }
    return idx, result


# Parallel execution on EC2
if __name__ == "__main__":
    # Choose a reasonable number of workers
    max_procs = multiprocessing.cpu_count()
    # You can tune this down if Ipopt is heavy, e.g. max_procs // 2
    max_workers = min(max_procs, num_weights)

    print(f"Launching {num_weights} MPC sims with up to {max_workers} workers...")

    # Prepare container for results
    results_list = [None] * num_weights

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_single_mpc, (i, weights[i]))
            for i in range(num_weights)
        ]

        for fut in as_completed(futures):
            idx, res = fut.result()
            results_list[idx] = res
            print(
                f"Done weight set {idx}: "
                f"final_biomass={res['final_biomass']:.3f}, "
                f"total_irrigation={res['total_irrigation']:.3f}, "
                f"total_fertilizer={res['total_fertilizer']:.3f}",
                flush=True,
            )

    # Convert to a DataFrame and save, if desired
    df_results = pd.DataFrame(results_list)
    df_results.to_csv("io/outputs/mpc_weight_tuning_results.csv", index=False)
    print("Saved results to mpc_weight_tuning_results.csv")
