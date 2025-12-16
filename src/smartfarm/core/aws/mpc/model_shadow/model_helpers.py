# model_helpers.py
import mpmath as mp
import numpy as np


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


def logistic_step(
        x:   float, # evaluation point
        a:   float, # growth rate
        k:   float, # carrying capacity
        dt:  float, # time step size
        eps: float = 1e-12 # small value to prevent numerical issues
    ) -> float:
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
