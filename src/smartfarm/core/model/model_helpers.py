# model_helpers.py
import numpy as np
import matplotlib.pyplot as plt
from core.model.model_growth_rates import ModelGrowthRates
from core.model.model_carrying_capacities import ModelCarryingCapacities

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

def plot_crop_growth_results(
        hs: np.ndarray,
        As: np.ndarray,
        Ns: np.ndarray,
        cs: np.ndarray,
        Ps: np.ndarray,
        dt: float = 1.0,
        labels: list = None) -> None:

    #time = np.arange(len(h))
    fig, axs = plt.subplots(5, 1, figsize=(10, 15))
    time = np.arange(0, len(hs[0])) * dt

    # Plant Height
    for i, h in enumerate(hs):
        axs[0].plot(time, h, label=labels[i])
    axs[0].set_xlabel('Time (hr)')
    axs[0].set_ylabel('Plant Height (m)')
    axs[0].set_title(f'Plant Height vs.Time')
    axs[0].legend()
    axs[0].grid(True)

    # Leaf Area
    for i, A in enumerate(As):
        axs[1].plot(time, A, label=labels[i])
    axs[1].set_xlabel('Time (hr)')
    axs[1].set_ylabel('Leaf Area (m²)')
    axs[1].set_title(f'Leaf Area vs.Time')
    axs[1].legend()
    axs[1].grid(True)

    # Number of Leaves
    for i, N in enumerate(Ns):
        axs[2].plot(time, N, label=labels[i])
    axs[2].set_xlabel('Time (hr)')
    axs[2].set_ylabel('Number of Leaves (unitless)')
    axs[2].set_title(f'Number of Leaves vs.Time')
    axs[2].legend()
    axs[2].grid(True)

    # Spikelet Count
    for i, c in enumerate(cs):
        axs[3].plot(time, c, label=labels[i])
    axs[3].set_xlabel('Time (hr)')
    axs[3].set_ylabel('Spikelet Count (unitless)')
    axs[3].set_title(f'Spikelet Count vs.Time')
    axs[3].legend()
    axs[3].grid(True)

    # Fruit Biomass
    for i, P in enumerate(Ps):
        axs[4].plot(time, P, label=labels[i])
    axs[4].set_xlabel('Time (hr)')
    axs[4].set_ylabel('Fruit Biomass (kg)')
    axs[4].set_title(f'Fruit Biomass vs.Time')
    axs[4].legend()
    axs[4].grid(True)

    fig.suptitle(f'Hourly Plant Growth over Season', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_nutrient_factor_evolution(
        nuWs: np.ndarray,
        nuFs: np.ndarray,
        nuTs: np.ndarray,
        nuRs: np.ndarray,
        dt: float = 1.0,
        labels: list = None
    ) -> None:

    fig, axs = plt.subplots(4, 1, figsize=(10, 15))
    time = np.arange(0, len(nuWs[0])) * dt

    # Water Nutrient Factor
    for i, nuW_values in enumerate(nuWs):
        axs[0].plot(time, nuW_values, label=labels[i])
    axs[0].axhline(y=1, color='red', linestyle='--', linewidth=1.5)
    axs[0].set_xlabel('Time (hr)')
    axs[0].set_ylabel('Water Nutrient Factor (unitless)')
    axs[0].set_title(f'Water Nutrient Factor vs.Time')
    axs[0].legend()
    axs[0].grid(True)

    # Fertilizer Nutrient Factor
    for i, nuF_values in enumerate(nuFs):
        axs[1].plot(time, nuF_values, label=labels[i])
    axs[1].axhline(y=1, color='red', linestyle='--', linewidth=1.5)
    axs[1].set_xlabel('Time (hr)')
    axs[1].set_ylabel('Fertilizer Nutrient Factor (unitless)')
    axs[1].set_title(f'Fertilizer Nutrient Factor vs.Time')
    axs[1].legend()
    axs[1].grid(True)

    # Temperature Nutrient Factor
    for i, nuT_values in enumerate(nuTs):
        axs[2].plot(time, nuT_values, label=labels[i])
    axs[2].axhline(y=1, color='red', linestyle='--', linewidth=1.5)
    axs[2].set_xlabel('Time (hr)')
    axs[2].set_ylabel('Temperature Nutrient Factor (unitless)')
    axs[2].set_title(f'Temperature Nutrient Factor vs.Time')
    axs[2].legend()
    axs[2].grid(True)

    # Radiation Nutrient Factor
    for i, nuR_values in enumerate(nuRs):
        axs[3].plot(time, nuR_values, label=labels[i])
    axs[3].axhline(y=1, color='red', linestyle='--', linewidth=1.5)
    axs[3].set_xlabel('Time (hr)')
    axs[3].set_ylabel('Radiation Nutrient Factor (unitless)')
    axs[3].set_title(f'Radiation Nutrient Factor vs.Time')
    axs[3].legend()
    axs[3].grid(True)

    fig.suptitle(f'Hourly Nutrient Factors over Season', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_growth_rate_evolution(
        growth_rates: ModelGrowthRates,
        ah_hats: np.ndarray,
        aA_hats: np.ndarray,
        aN_hats: np.ndarray,
        ac_hats: np.ndarray,
        aP_hats: np.ndarray,
        dt: float = 1.0,
        labels: list = None
    ) -> None:

    fig, axs = plt.subplots(5, 1, figsize=(10, 15))
    time = np.arange(0, len(ah_hats[0])) * dt

    # Plant Height Growth Rate
    for i, ah_hat in enumerate(ah_hats):
        axs[0].plot(time, ah_hat, label=labels[i])
    axs[0].axhline(y=growth_rates.ah, color='red', linestyle='--', linewidth=1.5)
    axs[0].set_xlabel('Time (hr)')
    axs[0].set_ylabel('Growth Rate (1/hr)')
    axs[0].set_title(f'Plant Height Growth Rate vs.Time')
    axs[0].legend()
    axs[0].grid(True)

    # Leaf Area Growth Rate
    for i, aA_hat in enumerate(aA_hats):
        axs[1].plot(time, aA_hat, label=labels[i])
    axs[1].axhline(y=growth_rates.aA, color='red', linestyle='--', linewidth=1.5)
    axs[1].set_xlabel('Time (hr)')
    axs[1].set_ylabel('Growth Rate (1/hr)')
    axs[1].set_title(f'Leaf Area Growth Rate vs.Time')
    axs[1].legend()
    axs[1].grid(True)

    # Number of Leaves Growth Rate
    for i, aN_hat in enumerate(aN_hats):
        axs[2].plot(time, aN_hat, label=labels[i])
    axs[2].axhline(y=growth_rates.aN, color='red', linestyle='--', linewidth=1.5)
    axs[2].set_xlabel('Time (hr)')
    axs[2].set_ylabel('Growth Rate (1/hr)')
    axs[2].set_title(f'Number of Leaves Growth Rate vs.Time')
    axs[2].legend()
    axs[2].grid(True)

    # Spikelet Count Growth Rate
    for i, ac_hat in enumerate(ac_hats):
        axs[3].plot(time, ac_hat, label=labels[i])
    axs[3].axhline(y=growth_rates.ac, color='red', linestyle='--', linewidth=1.5)
    axs[3].set_xlabel('Time (hr)')
    axs[3].set_ylabel('Growth Rate (1/hr)')
    axs[3].set_title(f'Spikelet Count Growth Rate vs.Time')
    axs[3].legend()
    axs[3].grid(True)

    # Fruit Biomass Growth Rate
    for i, aP_hat in enumerate(aP_hats):
        axs[4].plot(time, aP_hat, label=labels[i])
    axs[4].axhline(y=growth_rates.aP, color='red', linestyle='--', linewidth=1.5)
    axs[4].set_xlabel('Time (hr)')
    axs[4].set_ylabel('Growth Rate (1/hr)')
    axs[4].set_title(f'Fruit Biomass Growth Rate vs.Time')
    axs[4].legend()
    axs[4].grid(True)

    fig.suptitle(f'Hourly Plant Growth Rates over Season', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_carrying_capacity_evolution(
        carrying_capacities: ModelCarryingCapacities,
        kh_hats: np.ndarray,
        kA_hats: np.ndarray,
        kN_hats: np.ndarray,
        kc_hats: np.ndarray,
        kP_hats: np.ndarray,
        dt: float = 1.0,
        labels: list = None
    ) -> None:

    fig, axs = plt.subplots(5, 1, figsize=(10, 15))
    time = np.arange(0, len(kh_hats[0])) * dt

    # Plant Height Carrying Capacity
    for i, kh_hat in enumerate(kh_hats):
        axs[0].plot(time, kh_hat, label=labels[i])
    axs[0].axhline(y=carrying_capacities.kh, color='red', linestyle='--', linewidth=1.5)
    axs[0].set_xlabel('Time (hr)')
    axs[0].set_ylabel('Carrying Capacity (cm)')
    axs[0].set_title(f'Plant Height Carrying Capacity vs.Time')
    axs[0].legend()
    axs[0].grid(True)

    # Leaf Area Carrying Capacity
    for i, kA_hat in enumerate(kA_hats):
        axs[1].plot(time, kA_hat, label=labels[i])
    axs[1].axhline(y=carrying_capacities.kA, color='red', linestyle='--', linewidth=1.5)
    axs[1].set_xlabel('Time (hr)')
    axs[1].set_ylabel('Carrying Capacity (cm²)')
    axs[1].set_title(f'Leaf Area Carrying Capacity vs.Time')
    axs[1].legend()
    axs[1].grid(True)

    # Number of Leaves Carrying Capacity
    for i, kN_hat in enumerate(kN_hats):
        axs[2].plot(time, kN_hat, label=labels[i])
    axs[2].axhline(y=carrying_capacities.kN, color='red', linestyle='--', linewidth=1.5)
    axs[2].set_xlabel('Time (hr)')
    axs[2].set_ylabel('Carrying Capacity (1/hr)')
    axs[2].set_title(f'Number of Leaves Carrying Capacity vs.Time')
    axs[2].legend()
    axs[2].grid(True)

    # Spikelet Count Carrying Capacity
    for i, kc_hat in enumerate(kc_hats):
        axs[3].plot(time, kc_hat, label=labels[i])
    axs[3].axhline(y=carrying_capacities.kc, color='red', linestyle='--', linewidth=1.5)
    axs[3].set_xlabel('Time (hr)')
    axs[3].set_ylabel('Carrying Capacity (1/hr)')
    axs[3].set_title(f'Spikelet Count Carrying Capacity vs.Time')
    axs[3].legend()
    axs[3].grid(True)

    # Fruit Biomass Carrying Capacity
    for i, kP_hat in enumerate(kP_hats):
        axs[4].plot(time, kP_hat, label=labels[i])
    axs[4].axhline(y=carrying_capacities.kP, color='red', linestyle='--', linewidth=1.5)
    axs[4].set_xlabel('Time (hr)')
    axs[4].set_ylabel('Carrying Capacity (1/hr)')
    axs[4].set_title(f'Fruit Biomass Carrying Capacity vs.Time')
    axs[4].legend()
    axs[4].grid(True)

    fig.suptitle(f'Hourly Plant Carrying Capacities over Season', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
