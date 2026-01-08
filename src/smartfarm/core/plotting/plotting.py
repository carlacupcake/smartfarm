# plotting.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from typing import Optional

from core.model.model_disturbances import ModelDisturbances
from core.model.model_growth_rates import ModelGrowthRates
from core.model.model_carrying_capacities import ModelCarryingCapacities
from core.model.model_params import ModelParams

from core.plotting.plotting_colors import PlottingColors
from core.plotting.plotting_params import PlottingParams

def get_hour_formatter(dt: float = 1.0):
    """
    Create a FuncFormatter that converts step values to hours.

    Args:
        dt: Time step in hours (default 1.0)

    Returns:
        A matplotlib FuncFormatter that scales x-axis values by dt
    """
    def scale_to_hours(x, pos):
        return f'{x * dt:g}'
    return ticker.FuncFormatter(scale_to_hours)


def add_adaptive_legend(ax, loc: str = 'lower right', max_per_col: int = 5) -> None:
    """
    Add a legend that adapts to multiple columns based on the number of items.

    Args:
        ax: Matplotlib Axes object
        loc: Legend location (default 'lower right')
        max_per_col: Maximum items per column before adding another column
    """
    handles, labels = ax.get_legend_handles_labels()
    n_items = len(labels)

    if n_items == 0:
        return

    # Calculate number of columns needed
    ncol = max(1, (n_items + max_per_col - 1) // max_per_col)

    ax.legend(loc=loc, ncol=ncol, fontsize=9, framealpha=0.9)


def apply_time_formatter(fig_or_axes, dt: float = 1.0) -> None:
    """
    Apply hour-scaling formatter to all x-axes in a figure or list of axes.

    Args:
        fig_or_axes: A matplotlib Figure, single Axes, or array of Axes
        dt: Time step in hours (default 1.0)
    """
    formatter = get_hour_formatter(dt)

    # Handle different input types
    if hasattr(fig_or_axes, 'get_axes'):
        # It's a Figure
        axes = fig_or_axes.get_axes()
    elif hasattr(fig_or_axes, '__iter__'):
        # It's an array/list of axes
        axes = np.array(fig_or_axes).flatten()
    else:
        # It's a single Axes
        axes = [fig_or_axes]

    for ax in axes:
        ax.xaxis.set_major_formatter(get_hour_formatter(dt))
        ax.xaxis.set_major_locator(ticker.AutoLocator())


def setup_plotting_styles() -> None:
    """
    Set default plotting styles and parameters.

    Args:
        None
    Returns:
        colors (PlottingColors):
            Apply default plotting color schemes.
        suptitle_y_position (float):
            Set the y-position of the suptitle.
        tight_layout_rect (list):
            Set the rectangle for the tight layout.
    """
    style  = PlottingParams()
    colors = PlottingColors()
    colors.apply_as_default()
    
    suptitle_y_position = style.suptitle_y_position
    tight_layout_rect   = style.tight_layout_rect

    return colors, suptitle_y_position, tight_layout_rect


def plot_hourly_inputs(
        input_disturbances: ModelDisturbances,
        hourly_irrigation:  Optional[np.ndarray] = None,
        hourly_fertilizer:  Optional[np.ndarray] = None
    ) -> None:
    """
    Plot the time-series trajectories of all input disturbances and control
    inputs for a single simulation scenario.

    Args:
        input_disturbances (InputDisturbances):
            Object containing hourly precipitation, solar radiation, and temperature
            data.
        hourly_irrigation (np.ndarray):
            Array of hourly irrigation inputs.
        hourly_fertilizer (np.ndarray):
            Array of hourly fertilizer inputs.
    Returns:
        None:
            Displays a Matplotlib figure with five stacked subplots showing
            the evolution of all input disturbances and control inputs over time.
    """
    # Setup plotting styles and parameters
    _, suptitle_y_position, tight_layout_rect = setup_plotting_styles()
    if hourly_irrigation is None:
        num_subplots = 3
    else:
        num_subplots = 5
    fig, axs = plt.subplots(num_subplots, 1, figsize=(9, num_subplots*3))
    timesteps = len(input_disturbances.precipitation)
    time = np.arange(timesteps)

    # Apply hour formatting to all axes
    model_params = ModelParams()
    simulation_hours = model_params.simulation_hours
    dt = simulation_hours/timesteps
    apply_time_formatter(axs, dt=dt)

    # Precipitation
    axs[0].plot(time, input_disturbances.precipitation)
    axs[0].set_xlabel('Time (hr)')
    axs[0].set_ylabel('Precipitation (in)')
    axs[0].set_title('Precipitation vs. Time')

    # Solar Radiation
    axs[1].plot(time, input_disturbances.radiation)
    axs[1].set_xlabel('Time (hr)')
    axs[1].set_ylabel('Radiation (W)')
    axs[1].set_title('Solar Radiation vs. Time')

    # Temperature
    axs[2].plot(time, input_disturbances.temperature)
    axs[2].set_xlabel('Time (hr)')
    axs[2].set_ylabel(r'Temperature (\textdegree C)')
    axs[2].set_title('Temperature vs. Time')

    # Irrigation Events
    if hourly_irrigation is not None:
        axs[3].plot(time, hourly_irrigation)
        axs[3].set_xlabel('Time (hr)')
        axs[3].set_ylabel('Irrigation (in)')
        axs[3].set_title('Irrigation vs. Time')

    # Fertilizer Events
    if hourly_fertilizer is not None:
        axs[4].plot(time, hourly_fertilizer)
        axs[4].set_xlabel('Time (hr)')
        axs[4].set_ylabel('Fertilizer (lbs)')
        axs[4].set_title('Fertilizer vs. Time')

    if hourly_irrigation is None:
        fig.suptitle(f'Hourly Disturbances', y=suptitle_y_position)
    else:
        fig.suptitle(f'Hourly Disturbances and Control Inputs', y=suptitle_y_position)

    plt.tight_layout(rect=tight_layout_rect)
    plt.show()


def plot_control_inputs(
        irrigation: np.ndarray,
        fertilizer: np.ndarray) -> None:
    """
    Plot the time-series trajectories of all control inputs for a single simulation
    scenario.

    Args:
        irrigation (np.ndarray):
            Array of irrigation inputs.
        fertilizer (np.ndarray):
            Array of fertilizer inputs.
    Returns:
        None:
            Displays a Matplotlib figure with two stacked subplots showing the
            evolution of all control inputs over time.
    """

    # Setup plotting styles and parameters
    _, suptitle_y_position, tight_layout_rect = setup_plotting_styles()
    num_subplots = 2
    fig, axs = plt.subplots(num_subplots, 1, figsize=(9, num_subplots*3))
    timesteps = len(irrigation)
    time = np.arange(timesteps)

    # Apply hour formatting to all axes
    apply_time_formatter(axs, dt=1.0)

    # Irrigation Events
    axs[0].plot(time, irrigation)
    axs[0].set_xlabel('Time (hours)')
    axs[0].set_ylabel('Irrigation (in)')
    axs[0].set_title(f'Irrigation Events vs. Time')

    # Fertilizer Events
    axs[1].plot(time, fertilizer)
    axs[1].set_xlabel('Time (hours)')
    axs[1].set_ylabel('Fertilizer (lbs)')
    axs[1].set_title(f'Fertilizer Events vs. Time')

    fig.suptitle('Control Inputs by Time Step', y=suptitle_y_position)
    plt.tight_layout(rect=tight_layout_rect)
    plt.show()


def plot_crop_growth_results(
        hs: list[np.ndarray],
        As: list[np.ndarray],
        Ns: list[np.ndarray],
        cs: list[np.ndarray],
        Ps: list[np.ndarray],
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
        labels (list[str], optional):
            List of labels corresponding to each trajectory set; used in legend
            entries. Must be the same length as the input lists.

    Returns:
        None:
            Displays a Matplotlib figure with five stacked subplots showing
            the evolution of all growth variables over time.
    """

    # Setup plotting styles and parameters
    _, suptitle_y_position, tight_layout_rect = setup_plotting_styles()
    num_plots = 5
    fig, axs = plt.subplots(num_plots, 1, figsize=(9, 3 * num_plots))
    timesteps = len(hs[0])
    time = np.arange(timesteps)

    # Apply hour formatting to all axes
    model_params = ModelParams()
    simulation_hours = model_params.simulation_hours
    dt = simulation_hours/timesteps
    apply_time_formatter(axs, dt=dt)

    # Plant Height
    for i, h in enumerate(hs):
        axs[0].plot(time, h, label=labels[i] if labels else None)
    axs[0].set_xlabel('Time (hours)')
    axs[0].set_ylabel('Plant Height (m)')
    axs[0].set_title('Plant Height vs. Time')
    add_adaptive_legend(axs[0])

    # Leaf Area
    for i, A in enumerate(As):
        axs[1].plot(time, A, label=labels[i] if labels else None)
    axs[1].set_xlabel('Time (hours)')
    axs[1].set_ylabel(r'Leaf Area (m\textsuperscript{2})')
    axs[1].set_title('Leaf Area vs. Time')
    add_adaptive_legend(axs[1])

    # Number of Leaves
    for i, N in enumerate(Ns):
        axs[2].plot(time, N, label=labels[i] if labels else None)
    axs[2].set_xlabel('Time (hours)')
    axs[2].set_ylabel('Number of Leaves (unitless)')
    axs[2].set_title('Number of Leaves vs. Time')
    add_adaptive_legend(axs[2])

    # Spikelet Count
    for i, c in enumerate(cs):
        axs[3].plot(time, c, label=labels[i] if labels else None)
    axs[3].set_xlabel('Time (hours)')
    axs[3].set_ylabel('Spikelet Count (unitless)')
    axs[3].set_title('Spikelet Count vs. Time')
    add_adaptive_legend(axs[3])

    # Fruit Biomass
    for i, P in enumerate(Ps):
        axs[4].plot(time, P, label=labels[i] if labels else None)
    axs[4].set_xlabel('Time (hours)')
    axs[4].set_ylabel('Fruit Biomass (kg)')
    axs[4].set_title('Fruit Biomass vs. Time')
    add_adaptive_legend(axs[4])

    fig.suptitle('Plant Growth over Season', y=suptitle_y_position)
    plt.tight_layout(rect=tight_layout_rect)
    plt.show()


def plot_applied_vs_absorbed(
        irrigation:          np.ndarray,
        precipitation:       np.ndarray,
        delayed_water:       np.ndarray,
        fertilizer:          np.ndarray,
        delayed_fertilizer:  np.ndarray,
        temperature:         np.ndarray,
        delayed_temperature: np.ndarray,
        radiation:           np.ndarray,
        delayed_radiation:   np.ndarray
    ) -> None:
    """
    Plot the time-series trajectories of all applied inputs vs. the delayed
    values (which represent what is actually absorbed).

    Args:
        irrigation (np.ndarray):
            Array of irrigation inputs vs. time steps.
        precipitation (np.ndarray):
            Array of precipitation inputs vs. time steps.
        delayed_water (np.ndarray):
            Array of delayed water inputs vs. time steps.
        fertilizer (np.ndarray):
            Array of fertilizer inputs vs. time steps.
        delayed_fertilizer (np.ndarray):
            Array of delayed fertilizer inputs vs. time steps.
        temperature (np.ndarray):
            Array of temperature inputs vs. time steps.
        delayed_temperature (np.ndarray):
            Array of delayed temperature inputs vs. time steps.
        radiation (np.ndarray):
            Array of radiation inputs vs. time steps.
        delayed_radiation (np.ndarray):
            Array of delayed radiation inputs vs. time steps.

    Returns:
        None:
            Displays a Matplotlib figure with four stacked subplots showing the
            evolution of applied and absorbed water, fertilizer, temperature, 
            and radiation inputs over time.
    """
    # Setup plotting styles and parameters
    colors, suptitle_y_position, tight_layout_rect = setup_plotting_styles()
    num_subplots = 4
    fig, axs = plt.subplots(num_subplots, 1, figsize=(9, num_subplots*3))
    timesteps = len(delayed_water)
    time = np.arange(timesteps)

    # Apply hour formatting to all axes
    model_params = ModelParams()
    simulation_hours = model_params.simulation_hours
    dt = simulation_hours/timesteps
    apply_time_formatter(axs, dt=dt)

    # Delayed Water
    axs[0].plot(precipitation, label='precipitation')
    axs[0].plot(time, delayed_water, label='water absorbed')
    axs[0].plot(irrigation, label='irrigation applied')
    axs[0].set_xlabel('Time (hours)')
    axs[0].set_ylabel('Water (in)')
    axs[0].set_title('Applied and Absorbed Water vs. Time')
    axs[0].legend(loc='upper right')

    # Delayed Fertilizer
    axs[1].plot(fertilizer, label='applied')
    axs[1].set_xlabel('Time (hours)')
    axs[1].set_ylabel('Fertilizer Applied (lbs)')
    axs[1].tick_params(axis='y', colors=colors.vivid_green)
    axs[1].yaxis.label.set_color(colors.vivid_green) 
    axs[1].set_title('Applied and Absorbed Fertilizer vs. Time')

    ax2 = axs[1].twinx() # Create a second y-axis sharing the same x-axis
    ax2.plot(            # Plot the absorbed trace there
        time,
        delayed_fertilizer,
        label='absorbed',
        color=colors.vivid_red
    ) 
    ax2.set_ylabel('Fertilizer Absorbed (lbs)')
    ax2.tick_params(axis='y', colors=colors.vivid_red)
    ax2.yaxis.label.set_color(colors.vivid_red)

    # Delayed Temperature
    axs[2].plot(temperature, label='applied')
    axs[2].plot(time, delayed_temperature, label='absorbed')
    axs[2].set_xlabel('Time (hours)')
    axs[2].set_ylabel(r'Temperature (\textdegree C)')
    axs[2].set_title('Applied and Absorbed Temperature vs. Time')
    axs[2].legend(loc='upper right')

    # Delayed Radiation
    axs[3].plot(radiation, label='applied')
    axs[3].plot(time, delayed_radiation, label='absorbed')
    axs[3].set_xlabel('Time (hours)')
    axs[3].set_ylabel(r'Radiation (W/m\textsuperscript{2})')
    axs[3].set_title('Applied and Absorbed Radiation vs. Time')
    axs[3].legend(loc='upper right')

    fig.suptitle('Applied and Absorbed Inputs and Disturbances over Season', y=suptitle_y_position)
    plt.tight_layout(rect=tight_layout_rect)
    plt.show()


def plot_cumulative_values(
        cumulative_water:       np.ndarray,
        cumulative_fertilizer:  np.ndarray,
        cumulative_temperature: np.ndarray,
        cumulative_radiation:   np.ndarray,
        typical_disturbances:   ModelDisturbances
    ) -> None:
    """
    Plot the time-series trajectories of cumulative water, fertilizer, temperature,
    and radiation inputs, along with typical values for comparison.
    
    Args:
        cumulative_water (np.ndarray):
            Array of cumulative water inputs over time.
        cumulative_fertilizer (np.ndarray):
            Array of cumulative fertilizer inputs over time.
        cumulative_temperature (np.ndarray):
            Array of cumulative temperature inputs over time.
        cumulative_radiation (np.ndarray):
            Array of cumulative radiation inputs over time.
        typical_disturbances (ModelDisturbances):
            Typical disturbance values (scalars) for comparison.

    Returns:
        None:
            Displays a Matplotlib figure with four stacked subplots showing the
            evolution of cumulative water, fertilizer, temperature, and radiation
            inputs over time, along with typical values for comparison.
    """

    # Setup plotting styles and parameters
    colors, suptitle_y_position, tight_layout_rect = setup_plotting_styles()
    num_subplots = 4
    fig, axs = plt.subplots(num_subplots, 1, figsize=(9, num_subplots*3))
    timesteps = len(cumulative_water)
    time = np.arange(timesteps)

    # Apply hour formatting to all axes
    model_params = ModelParams()
    simulation_hours = model_params.simulation_hours
    dt = simulation_hours/timesteps
    apply_time_formatter(axs, dt=dt)

    R_typ = typical_disturbances.typical_radiation   * time
    T_typ = typical_disturbances.typical_temperature * time
    W_typ = typical_disturbances.typical_water       * time
    F_typ = typical_disturbances.typical_fertilizer  * time 

    # Cumulative Water
    axs[0].plot(time, cumulative_water)
    axs[0].plot(W_typ, linestyle='--', color=colors.vivid_red)
    axs[0].set_xlabel('Time (hours)')
    axs[0].set_ylabel('Cumulative Water (in)')
    axs[0].set_title('Cumulative Water vs. Time')

    # Cumulative Fertilizer
    axs[1].plot(time, cumulative_fertilizer)
    axs[1].plot(F_typ, linestyle='--', color=colors.vivid_red)
    axs[1].set_xlabel('Time (hours)')
    axs[1].set_ylabel('Cumulative Fertilizer (lbs)')
    axs[1].set_title('Cumulative Fertilizer vs. Time')

    # Cumulative Temperature
    axs[2].plot(time, cumulative_temperature)
    axs[2].plot(T_typ, linestyle='--', color=colors.vivid_red)
    axs[2].set_xlabel('Time (hours)')
    axs[2].set_ylabel(r'Cumulative Temperature (\textdegree C)')
    axs[2].set_title('Cumulative Temperature vs. Time')

    # Cumulative Radiation
    axs[3].plot(time, cumulative_radiation)
    axs[3].plot(R_typ, linestyle='--', color=colors.vivid_red)
    axs[3].set_xlabel('Time (hours)')
    axs[3].set_ylabel(r'Cumulative Radiation (W/m\textsuperscript{2})')
    axs[3].set_title('Cumulative Radiation vs. Time')

    fig.suptitle('Cumulative Values over Season', y=suptitle_y_position)
    plt.tight_layout(rect=tight_layout_rect)
    plt.show()


def plot_actual_vs_expected_cumulatives(
        delta_cumulative_water:       np.ndarray,
        delta_cumulative_fertilizer:  np.ndarray,
        delta_cumulative_temperature: np.ndarray,
        delta_cumulative_radiation:   np.ndarray
    ) -> None:
    """
    Plot the time-series trajectories of the differences between actual and typical cumulative values.

    Args:
        delta_cumulative_water (np.ndarray):
            Array of differences between actual and typical cumulative water inputs.
        delta_cumulative_fertilizer (np.ndarray):
            Array of differences between actual and typical cumulative fertilizer inputs.
        delta_cumulative_temperature (np.ndarray):
            Array of differences between actual and typical cumulative temperature inputs.
        delta_cumulative_radiation (np.ndarray):
            Array of differences between actual and typical cumulative radiation inputs.
    Returns:
        None:
            Displays a Matplotlib figure with four stacked subplots showing the
            evolution of the differences between actual and typical cumulative values
            over time.
    """
    # Setup plotting styles and parameters
    _, suptitle_y_position, tight_layout_rect = setup_plotting_styles()
    num_subplots = 4
    fig, axs = plt.subplots(num_subplots, 1, figsize=(9, num_subplots*3))
    timesteps = len(delta_cumulative_water)
    time = np.arange(timesteps)

    # Apply hour formatting to all axes
    model_params = ModelParams()
    simulation_hours = model_params.simulation_hours
    dt = simulation_hours/timesteps
    apply_time_formatter(axs, dt=dt)

    # Delta Cumulative Water
    axs[0].plot(time, delta_cumulative_water)
    axs[0].set_ylim(bottom=0, top=None) 
    axs[0].set_xlabel('Time (hours)')
    axs[0].set_ylabel(r'$\Delta$ (normed)')
    axs[0].set_title(r'$\Delta$ Cumulative Water vs. Time')

    # Delta Cumulative Fertilizer
    axs[1].plot(time, delta_cumulative_fertilizer)
    axs[1].set_ylim(bottom=0, top=None) 
    axs[1].set_xlabel('Time (hours)')
    axs[1].set_ylabel(r'$\Delta$ (normed)')
    axs[1].set_title(r'$\Delta$ Cumulative Fertilizer vs. Time')

    # Delta Cumulative Temperature
    axs[2].plot(time, delta_cumulative_temperature)
    axs[2].set_ylim(bottom=0, top=None) 
    axs[2].set_xlabel('Time (hours)')
    axs[2].set_ylabel(r'$\Delta$ (normed)')
    axs[2].set_title(r'$\Delta$ Cumulative Temperature vs. Time')

    # Delta Cumulative Radiation
    axs[3].plot(time, delta_cumulative_radiation)
    axs[3].set_ylim(bottom=0, top=None) 
    axs[3].set_xlabel('Time (hours)')
    axs[3].set_ylabel(r'$\Delta$ (normed)')
    axs[3].set_title(r'$\Delta$ Cumulative Radiation vs. Time')

    fig.suptitle('Differences between Actual and Typical Cumulative Values', y=suptitle_y_position)
    plt.tight_layout(rect=tight_layout_rect)
    plt.show()


def plot_nutrient_factor_evolution(
        nuWs: list[np.ndarray],
        nuFs: list[np.ndarray],
        nuTs: list[np.ndarray],
        nuRs: list[np.ndarray],
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
    # Setup plotting styles and parameters
    colors, suptitle_y_position, tight_layout_rect = setup_plotting_styles()
    num_subplots = 4
    fig, axs = plt.subplots(num_subplots, 1, figsize=(9, num_subplots*3))
    timesteps = len(nuWs[0])
    time = np.arange(timesteps)

    # Apply hour formatting to all axes
    model_params = ModelParams()
    simulation_hours = model_params.simulation_hours
    dt = simulation_hours/timesteps
    apply_time_formatter(axs, dt=dt)

    # Water Nutrient Factor
    for i, nuW_values in enumerate(nuWs):
        axs[0].plot(time, nuW_values, label=labels[i] if labels else None)
    axs[0].axhline(y=1, linestyle='--', color=colors.vivid_red)
    axs[0].set_ylim(bottom=0, top=None)
    axs[0].set_xlabel('Time (hours)')
    axs[0].set_ylabel('Nutrient Factor (unitless)')
    axs[0].set_title('Water Nutrient Factor vs. Time')
    add_adaptive_legend(axs[0])

    # Fertilizer Nutrient Factor
    for i, nuF_values in enumerate(nuFs):
        axs[1].plot(time, nuF_values, label=labels[i] if labels else None)
    axs[1].axhline(y=1, linestyle='--', color=colors.vivid_red)
    axs[1].set_ylim(bottom=0, top=None)
    axs[1].set_xlabel('Time (hours)')
    axs[1].set_ylabel('Nutrient Factor (unitless)')
    axs[1].set_title('Fertilizer Nutrient Factor vs. Time')
    add_adaptive_legend(axs[1])

    # Temperature Nutrient Factor
    for i, nuT_values in enumerate(nuTs):
        axs[2].plot(time, nuT_values, label=labels[i] if labels else None)
    axs[2].axhline(y=1, linestyle='--', color=colors.vivid_red)
    axs[2].set_ylim(bottom=0, top=None)
    axs[2].set_xlabel('Time (hours)')
    axs[2].set_ylabel('Nutrient Factor (unitless)')
    axs[2].set_title('Temperature Nutrient Factor vs. Time')
    add_adaptive_legend(axs[2])

    # Radiation Nutrient Factor
    for i, nuR_values in enumerate(nuRs):
        axs[3].plot(time, nuR_values, label=labels[i] if labels else None)
    axs[3].axhline(y=1, linestyle='--', color=colors.vivid_red)
    axs[3].set_ylim(bottom=0, top=None)
    axs[3].set_xlabel('Time (hours)')
    axs[3].set_ylabel('Nutrient Factor (unitless)')
    axs[3].set_title('Radiation Nutrient Factor vs. Time')
    add_adaptive_legend(axs[3])

    fig.suptitle('Nutrient Factors over Season', y=suptitle_y_position)
    plt.tight_layout(rect=tight_layout_rect)
    plt.show()


def plot_growth_rate_evolution(
        growth_rates: ModelGrowthRates,
        ah_hats: list[np.ndarray],
        aA_hats: list[np.ndarray],
        aN_hats: list[np.ndarray],
        ac_hats: list[np.ndarray],
        aP_hats: list[np.ndarray],
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
    # Setup plotting styles and parameters
    colors, suptitle_y_position, tight_layout_rect = setup_plotting_styles()
    num_subplots = 5
    fig, axs = plt.subplots(num_subplots, 1, figsize=(9, num_subplots*3))
    timesteps = len(ah_hats[0])
    time = np.arange(timesteps)

    # Apply hour formatting to all axes
    model_params = ModelParams()
    simulation_hours = model_params.simulation_hours
    dt = simulation_hours/timesteps
    apply_time_formatter(axs, dt=dt)

    # Plant Height Growth Rate
    for i, ah_hat in enumerate(ah_hats):
        axs[0].plot(time, ah_hat, label=labels[i] if labels else None)
    axs[0].axhline(y=growth_rates.ah, linestyle='--', color=colors.vivid_red)
    axs[0].set_ylim(bottom=0, top=None)
    axs[0].set_xlabel('Time (hours)')
    axs[0].set_ylabel('Growth Rate (1/hr)')
    axs[0].set_title('Plant Height Growth Rate vs. Time')
    add_adaptive_legend(axs[0])

    # Leaf Area Growth Rate
    for i, aA_hat in enumerate(aA_hats):
        axs[1].plot(time, aA_hat, label=labels[i] if labels else None)
    axs[1].axhline(y=growth_rates.aA, linestyle='--', color=colors.vivid_red)
    axs[1].set_ylim(bottom=0, top=None)
    axs[1].set_xlabel('Time (hours)')
    axs[1].set_ylabel('Growth Rate (1/hr)')
    axs[1].set_title('Leaf Area Growth Rate vs. Time')
    add_adaptive_legend(axs[1])

    # Number of Leaves Growth Rate
    for i, aN_hat in enumerate(aN_hats):
        axs[2].plot(time, aN_hat, label=labels[i] if labels else None)
    axs[2].axhline(y=growth_rates.aN, linestyle='--', color=colors.vivid_red)
    axs[2].set_ylim(bottom=0, top=None)
    axs[2].set_xlabel('Time (hours)')
    axs[2].set_ylabel('Growth Rate (1/hr)')
    axs[2].set_title('Number of Leaves Growth Rate vs. Time')
    add_adaptive_legend(axs[2])

    # Spikelet Count Growth Rate
    for i, ac_hat in enumerate(ac_hats):
        axs[3].plot(time, ac_hat, label=labels[i] if labels else None)
    axs[3].axhline(y=growth_rates.ac, linestyle='--', color=colors.vivid_red)
    axs[3].set_ylim(bottom=0, top=None)
    axs[3].set_xlabel('Time (hours)')
    axs[3].set_ylabel('Growth Rate (1/hr)')
    axs[3].set_title('Spikelet Count Growth Rate vs. Time')
    add_adaptive_legend(axs[3])

    # Fruit Biomass Growth Rate
    for i, aP_hat in enumerate(aP_hats):
        axs[4].plot(time, aP_hat, label=labels[i] if labels else None)
    axs[4].axhline(y=growth_rates.aP, linestyle='--', color=colors.vivid_red)
    axs[4].set_ylim(bottom=0, top=None)
    axs[4].set_xlabel('Time (hours)')
    axs[4].set_ylabel('Growth Rate (1/hr)')
    axs[4].set_title('Fruit Biomass Growth Rate vs. Time')
    add_adaptive_legend(axs[4])

    fig.suptitle('Plant Growth Rates over Season', y=suptitle_y_position)
    plt.tight_layout(rect=tight_layout_rect)
    plt.show()


def plot_carrying_capacity_evolution(
        carrying_capacities: ModelCarryingCapacities,
        kh_hats: list[np.ndarray],
        kA_hats: list[np.ndarray],
        kN_hats: list[np.ndarray],
        kc_hats: list[np.ndarray],
        kP_hats: list[np.ndarray],
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
    # Setup plotting styles and parameters
    colors, suptitle_y_position, tight_layout_rect = setup_plotting_styles()
    num_subplots = 5
    fig, axs = plt.subplots(num_subplots, 1, figsize=(9, num_subplots*3))
    timesteps = len(kh_hats[0])
    time = np.arange(timesteps)

    # Apply hour formatting to all axes
    model_params = ModelParams()
    simulation_hours = model_params.simulation_hours
    dt = simulation_hours/timesteps
    apply_time_formatter(axs, dt=dt)

    # Plant Height Carrying Capacity
    for i, kh_hat in enumerate(kh_hats):
        axs[0].plot(time, kh_hat, label=labels[i] if labels else None)
    axs[0].axhline(y=carrying_capacities.kh, linestyle='--', color=colors.vivid_red)
    axs[0].set_ylim(bottom=0, top=None)
    axs[0].set_xlabel('Time (hours)')
    axs[0].set_ylabel('Carrying Capacity (m)')
    axs[0].set_title('Plant Height Carrying Capacity vs. Time')
    add_adaptive_legend(axs[0])

    # Leaf Area Carrying Capacity
    for i, kA_hat in enumerate(kA_hats):
        axs[1].plot(time, kA_hat, label=labels[i] if labels else None)
    axs[1].axhline(y=carrying_capacities.kA, linestyle='--', color=colors.vivid_red)
    axs[1].set_ylim(bottom=0, top=None)
    axs[1].set_xlabel('Time (hours)')
    axs[1].set_ylabel(r'Carrying Capacity (m\textsuperscript{2})')
    axs[1].set_title('Leaf Area Carrying Capacity vs. Time')
    add_adaptive_legend(axs[1])

    # Number of Leaves Carrying Capacity
    for i, kN_hat in enumerate(kN_hats):
        axs[2].plot(time, kN_hat, label=labels[i] if labels else None)
    axs[2].axhline(y=carrying_capacities.kN, linestyle='--', color=colors.vivid_red)
    axs[2].set_ylim(bottom=0, top=None)
    axs[2].set_xlabel('Time (hours)')
    axs[2].set_ylabel('Carrying Capacity (count)')
    axs[2].set_title('Number of Leaves Carrying Capacity vs. Time')
    add_adaptive_legend(axs[2])

    # Spikelet Count Carrying Capacity
    for i, kc_hat in enumerate(kc_hats):
        axs[3].plot(time, kc_hat, label=labels[i] if labels else None)
    axs[3].axhline(y=carrying_capacities.kc, linestyle='--', color=colors.vivid_red)
    axs[3].set_ylim(bottom=0, top=None)
    axs[3].set_xlabel('Time (hours)')
    axs[3].set_ylabel('Carrying Capacity (count)')
    axs[3].set_title('Spikelet Count Carrying Capacity vs. Time')
    add_adaptive_legend(axs[3])

    # Fruit Biomass Carrying Capacity
    for i, kP_hat in enumerate(kP_hats):
        axs[4].plot(time, kP_hat, label=labels[i] if labels else None)
    axs[4].axhline(y=carrying_capacities.kP, linestyle='--', color=colors.vivid_red)
    axs[4].set_ylim(bottom=0, top=None)
    axs[4].set_xlabel('Time (hours)')
    axs[4].set_ylabel('Carrying Capacity (kg)')
    axs[4].set_title('Fruit Biomass Carrying Capacity vs. Time')
    add_adaptive_legend(axs[4])

    fig.suptitle('Plant Carrying Capacities over Season', y=suptitle_y_position)
    plt.tight_layout(rect=tight_layout_rect)
    plt.show()
