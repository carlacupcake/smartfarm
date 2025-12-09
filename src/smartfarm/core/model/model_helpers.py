# model_helpers.py
import numpy as np
import matplotlib.pyplot as plt
from core.model.model_growth_rates import ModelGrowthRates
from core.model.model_carrying_capacities import ModelCarryingCapacities


def get_nutrient_factor(x, mu, sensitivity=0.7):
    """
    Compute a bounded nutrient factor based on a Gaussian-like curve
    centered at `mu`. The `sensitivity` parameter controls how sharply the
    factor penalizes deviations from the optimal value.

    Args:
        x (float or ndarray):
            Current cumulative nutrient level normalized by the typical cumulative
            value at that time point.
        mu (float):
            Optimal or “target” value at which the nutrient factor peaks (nu = 1).
        sensitivity (float, optional):
            Controls curvature of the response (0 → broad tolerance, 1 → sharp
            sensitivity); must lie between 0 and 1.

    Returns:
        float or ndarray:
            Nutrient factor `nu` ∈ (0, 1], representing how supportive the
            current conditions are relative to the optimum.
    """

    sigma_min = 0.1 * mu
    sigma_max = 100 * mu
    sigma = 1/4 * sigma_min * sigma_max * mu**2 *(1 - sensitivity**2)
    exp_arg = -(x - mu)**2/(2*sigma**2)
    nu = np.exp(exp_arg)
    
    return nu


def get_nutrient_factor_abs(x, mu, sensitivity=1.0):

    nu = np.clip(sensitivity * np.abs(x/mu) + (1 - sensitivity), 0, 1)
    #nu = np.clip(sensitivity * (1 - np.sqrt((x-mu)**2/mu**2)) + (1 - sensitivity), 0.1, 1)
    
    return nu
        

def get_sim_inputs_from_hourly(
        hourly_array,
        dt,
        simulation_hours,
        mode='copy' # 'copy' or 'split'
    ):
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


def logistic_step(x, a, k, dt, eps=1e-12):
    """
    Advance a logistic-growth state variable one time step using the closed-form
    solution of the logistic ODE dy/dt = ay(1 − y/k). Small eps values prevent
    division by zero or singularities.

    Args:
        y (float):
            Current state value (height, biomass, etc.).
        a (float):
            Growth-rate parameter in the logistic equation.
        k (float):
            Carrying capacity; values below `eps` are clipped.
        dt (float):
            Time-step size.
        eps (float, optional):
            Minimum allowed value for `y` and `k` to maintain numerical stability.

    Returns:
        float:
            The analytically updated state value at time t + dt.
    """

    k = max(k, eps)
    x = max(x, eps)
    exp_term = np.exp(-a * dt)
    return k / (1.0 + (k / x - 1.0) * exp_term)


def plot_crop_growth_results(
        hs: list[np.ndarray],
        As: list[np.ndarray],
        Ns: list[np.ndarray],
        cs: list[np.ndarray],
        Ps: list[np.ndarray],
        dt: float = 1.0,
        labels: list[str] = None) -> None:
    """
    Plot time-series trajectories of key crop-growth variables—plant height,
    leaf area, leaf count, spikelet count, and fruit biomass—for one or more
    simulated scenarios.

    Args:
        hs (list[np.ndarray]):
            List of plant-height trajectories, where each element is a 1D
            NumPy array for a single simulation scenario.
        As (list[np.ndarray]):
            List of leaf-area trajectories.
        Ns (list[np.ndarray]):
            List of leaf-count trajectories.
        cs (list[np.ndarray]):
            List of spikelet-count trajectories.
        Ps (list[np.ndarray]):
            List of fruit-biomass trajectories.
        dt (float, optional):
            Time step (in hours) between samples in each trajectory; used to
            construct the time axis. Defaults to 1.0.
        labels (list[str], optional):
            List of labels corresponding to each trajectory set; used in legend
            entries. Must be the same length as the input lists.

    Returns:
        None:
            Displays a Matplotlib figure with five stacked subplots showing
            the evolution of all growth variables over time.
    """

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
        nuWs: list[np.ndarray],
        nuFs: list[np.ndarray],
        nuTs: list[np.ndarray],
        nuRs: list[np.ndarray],
        dt: float = 1.0,
        labels: list[str] = None
    ) -> None:
    """
    Plot the evolution of nutrient-response factors—water, fertilizer,
    temperature, and radiation—for one or more simulated scenarios over time.

    Args:
        nuWs (list[np.ndarray]):
            List of water nutrient-factor trajectories, each a 1D NumPy array.
        nuFs (list[np.ndarray]):
            List of fertilizer nutrient-factor trajectories.
        nuTs (list[np.ndarray]):
            List of temperature nutrient-factor trajectories.
        nuRs (list[np.ndarray]):
            List of radiation nutrient-factor trajectories.
        dt (float, optional):
            Time step (in hours) used to construct the time axis. Defaults to 1.0.
        labels (list[str], optional):
            Labels for each scenario, used in plot legends. Must match the number
            of trajectories in each input list.

    Returns:
        None:
            Displays a Matplotlib figure with four stacked subplots showing the
            evolution of all nutrient factors across the simulation horizon.
    """

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
        ah_hats: list[np.ndarray],
        aA_hats: list[np.ndarray],
        aN_hats: list[np.ndarray],
        ac_hats: list[np.ndarray],
        aP_hats: list[np.ndarray],
        dt: float = 1.0,
        labels: list[str] = None
    ) -> None:
    """
    Plot the time evolution of adjusted logistic growth-rate coefficients for
    plant height, leaf area, leaf count, spikelet count, and fruit biomass
    across one or more simulated scenarios.

    Args:
        growth_rates (ModelGrowthRates):
            Baseline (unadjusted) growth-rate parameters, used to draw reference
            horizontal lines in each subplot.
        ah_hats (list[np.ndarray]):
            Adjusted height growth-rate trajectories for each scenario.
        aA_hats (list[np.ndarray]):
            Adjusted leaf-area growth-rate trajectories.
        aN_hats (list[np.ndarray]):
            Adjusted leaf-count growth-rate trajectories.
        ac_hats (list[np.ndarray]):
            Adjusted spikelet-count growth-rate trajectories.
        aP_hats (list[np.ndarray]):
            Adjusted fruit-biomass growth-rate trajectories.
        dt (float, optional):
            Time step (in hours) used for constructing the time axis. Defaults to 1.0.
        labels (list[str], optional):
            Labels corresponding to each scenario, used for legends.

    Returns:
        None:
            Displays a Matplotlib figure with five stacked subplots showing
            adjusted growth-rate trajectories relative to their baseline values.
    """

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
        kh_hats: list[np.ndarray],
        kA_hats: list[np.ndarray],
        kN_hats: list[np.ndarray],
        kc_hats: list[np.ndarray],
        kP_hats: list[np.ndarray],
        dt: float = 1.0,
        labels: list[str] = None
    ) -> None:
    """
    Plot the time evolution of adjusted carrying-capacity parameters for
    plant height, leaf area, leaf count, spikelet count, and fruit biomass
    across one or more simulated scenarios.

    Args:
        carrying_capacities (ModelCarryingCapacities):
            Baseline carrying-capacity parameters used to draw reference lines
            in each subplot.
        kh_hats (list[np.ndarray]):
            Adjusted height carrying-capacity trajectories for each scenario.
        kA_hats (list[np.ndarray]):
            Adjusted leaf-area carrying-capacity trajectories.
        kN_hats (list[np.ndarray]):
            Adjusted leaf-count carrying-capacity trajectories.
        kc_hats (list[np.ndarray]):
            Adjusted spikelet-count carrying-capacity trajectories.
        kP_hats (list[np.ndarray]):
            Adjusted fruit-biomass carrying-capacity trajectories.
        dt (float, optional):
            Time step (in hours) used for constructing the time axis.
            Defaults to 1.0.
        labels (list[str], optional):
            Scenario labels for use in plot legends. Must match the number of
            trajectories in each list.

    Returns:
        None:
            Displays a Matplotlib figure with five stacked subplots showing the
            evolution of all carrying-capacity parameters relative to their
            baseline values.
    """

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
