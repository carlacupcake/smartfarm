# model_helpers.py
import numpy as np
import warnings # temporary

def get_nutrient_factor(x, mu):
    # Set up a context manager to temporarily treat RuntimeWarnings as exceptions
    with warnings.catch_warnings():
        # Specifically filter RuntimeWarnings to be raised as an exception
        warnings.filterwarnings("error", category=RuntimeWarning)
        try:
            # The calculation where the warning might occur
            exp_arg = 1/100*(x - mu)
            exp_val = np.exp(exp_arg)
            nu = 1/2 * (1 / (1 + exp_val) + 1)
            return nu
        except RuntimeWarning as e:
            # This block will now catch the promoted RuntimeWarning
            print(f"RuntimeWarning occurred: {e}")
            print(f'x: {x}, mu: {mu}, exp arg: {exp_arg}')
            # You might want to return a specific value or re-raise
            return np.nan # Or a sensible default value
        except Exception as e:
            # This catches any other non-Runtime exceptions
            print(f"Other Error occurred: {e}")
            return np.nan # Or a sensible default value
        

def get_sim_inputs_from_hourly(
        hourly_array,
        dt,
        simulation_hours,
        mode='copy' # 'copy' or 'split'
    ):

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
