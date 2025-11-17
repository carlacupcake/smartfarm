# model_helpers.py
import numpy as np

def get_nutrient_factor(x, mu, sensitivity=0.5):
    
    # Sensitivity parameter between 0 and 1
    sigma_min = 0.1 * mu
    sigma_max = 100 * mu
    sigma = sigma_max**(1 - sensitivity) * sigma_min**(sensitivity)
    exp_arg = -(x - mu)**2/(2*sigma**2)
    nu = np.exp(exp_arg)
    
    return nu
        

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
