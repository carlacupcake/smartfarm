# ga_lambda_helpers.py
import mpmath as mp
import numpy as np


def compute_fir_horizon(
    kernel:         np.ndarray,
    mass_threshold: float = 0.95
    ) -> int:
    """
    Return the smallest L such that the first L taps of `kernel`
    contain at least `mass_threshold` of the total kernel mass.

    Args:
        kernel (np.ndarray):
            1D array representing the FIR kernel.
        mass_threshold (float):
            Cumulative mass threshold in (0, 1].
    Returns:
        int:
            The computed FIR horizon length.
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


def smooth_with_ema(x, beta, y_prev0):
    """
    Compute an exponential moving average (EMA) of a 1D signal using an
    explicit forward scan. Implements the first-order recursive filter

        y[t] = beta * y[t-1] + (1 - beta) * x[t]

    with an externally provided initial condition `y_prev0 = y[-1]`.
    It is mathematically equivalent to an IIR low-pass filter and is
    useful when exact step-by-step state propagation is required (e.g.,
    for matching MPC / CFTOC dynamics or logging internal model state).

    Args:
        x (np.ndarray)
            One-dimensional input signal to be smoothed.

        beta (float)
            Exponential decay factor in (0, 1). Larger values place more weight
            on past history (slower response); smaller values emphasize recent
            samples.

        y_prev0 (float)
            Initial EMA state corresponding to the value at time step t = -1.

    Returns:
        y (np.ndarray)
            EMA filtered signal.
    
    Example usage:
        x = anomaly[k]
        y = smoothed divergence
        yprev0 = 0.0 (assume divergence should be zero)
    """
    y = np.empty_like(x, dtype=float)
    y_prev = float(y_prev0)
    for t in range(x.shape[0]):
        y_prev = beta * y_prev + (1.0 - beta) * x[t]
        y[t] = y_prev
    return y


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


def get_cost_with_lambda(
        member_dict: dict,
        enriched_ctx: dict
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
        enriched_ctx (dict):
            Dictionary providing all model parameters, disturbances,
            growth-rate values, carrying capacities, and cost-function weights
            required to run the simulation.

    Returns:
        float:
            The cost value (negative net revenue) computed from final fruit
            biomass minus weighted irrigation and fertilizer usage.
    """

    #print(">>> INSIDE get_cost_with_lambda <<<")

    # Design variables (same order as bounds)
    irrigation_frequency, irrigation_amount, fertilizer_frequency, fertilizer_amount = [
        float(x) for x in member_dict["values"]
    ]

    # Unpack model parameters
    dt               = enriched_ctx["dt"]
    total_time_steps = enriched_ctx["total_time_steps"]
    simulation_hours = enriched_ctx["simulation_hours"]

    # Unpack typical disturbances
    W_typ = enriched_ctx["W_typ"]
    F_typ = enriched_ctx["F_typ"]
    T_typ = enriched_ctx["T_typ"]
    R_typ = enriched_ctx["R_typ"]

    # Unpack initial conditions
    h = enriched_ctx["h0"]
    A = enriched_ctx["A0"]
    N = enriched_ctx["N0"]
    c = enriched_ctx["c0"]
    P = enriched_ctx["P0"]

    # Unpack growth rates
    ah = enriched_ctx["ah"]
    aA = enriched_ctx["aA"]
    aN = enriched_ctx["aN"]
    ac = enriched_ctx["ac"]
    aP = enriched_ctx["aP"]

    # Unpack carrying capacities
    kh = enriched_ctx["kh"]
    kA = enriched_ctx["kA"]
    kN = enriched_ctx["kN"]
    kc = enriched_ctx["kc"]
    kP = enriched_ctx["kP"]

    # Unpack GA cost function weights
    weight_irrigation    = enriched_ctx["weight_irrigation"]
    weight_fertilizer    = enriched_ctx["weight_fertilizer"]
    weight_height        = enriched_ctx["weight_height"]
    weight_leaf_area     = enriched_ctx["weight_leaf_area"]
    weight_fruit_biomass = enriched_ctx["weight_fruit_biomass"]

    # Build hourly control series from design variables and input disturbances
    hourly_irrigation = np.zeros(simulation_hours)
    step_if = max(1, int(np.ceil(irrigation_frequency)))
    hourly_irrigation[::step_if] = irrigation_amount
    irrigation = get_sim_inputs_from_hourly(hourly_irrigation, dt, simulation_hours, mode='split')

    hourly_fertilizer = np.zeros(simulation_hours)
    step_ff = max(1, int(np.ceil(fertilizer_frequency)))
    hourly_fertilizer[::step_ff] = fertilizer_amount
    fertilizer = get_sim_inputs_from_hourly(hourly_fertilizer, dt, simulation_hours, mode='split')

    precipitation = np.asarray(enriched_ctx["precipitation"])
    temperature   = np.asarray(enriched_ctx["temperature"])
    radiation     = np.asarray(enriched_ctx["radiation"])

    # # Precompute FIR kernels from existing sensitivities to model delayed absorption/metalysis
    kernel_W = enriched_ctx["kernel_W"]
    kernel_F = enriched_ctx["kernel_F"]
    kernel_T = enriched_ctx["kernel_T"]
    kernel_R = enriched_ctx["kernel_R"]

    fir_horizon_W = compute_fir_horizon(kernel_W)
    fir_horizon_F = compute_fir_horizon(kernel_F)
    fir_horizon_T = compute_fir_horizon(kernel_T)
    fir_horizon_R = compute_fir_horizon(kernel_R)

    # Set history of disturbances before time zero
    water_history       = np.ones(fir_horizon_W, dtype=float) * W_typ
    fertilizer_history  = np.ones(fir_horizon_F, dtype=float) * F_typ
    temperature_history = np.ones(fir_horizon_T, dtype=float) * T_typ
    radiation_history   = np.ones(fir_horizon_R, dtype=float) * R_typ

    # Initialize cumulative disturbances
    cumulative_water       = 0.0
    cumulative_fertilizer  = 0.0
    cumulative_temperature = 0.0
    cumulative_radiation   = 0.0

    # Initialize cumulative divergences
    cumulative_divergence_water       = 0.0
    cumulative_divergence_fertilizer  = 0.0
    cumulative_divergence_temperature = 0.0
    cumulative_divergence_radiation   = 0.0

    # Initialize nutrient factors
    nuW = 1.0
    nuF = 1.0
    nuT = 1.0
    nuR = 1.0

    # Run the season simulation for the given member
    for t in range(total_time_steps - 1):

        # Unpack control inputs
        W = irrigation[t]
        F = fertilizer[t]

        # Unpack disturbances
        S = precipitation[t]
        T = temperature[t]
        R = radiation[t]

        # Unpack FIR kernels (truncate to FIR horizon)
        kernel_W = kernel_W[:fir_horizon_W]
        kernel_F = kernel_F[:fir_horizon_F]
        kernel_T = kernel_T[:fir_horizon_T]
        kernel_R = kernel_R[:fir_horizon_R]

        # Update FIR histories (shift left, append new sample)
        water_history       = np.roll(water_history,       -1)
        fertilizer_history  = np.roll(fertilizer_history,  -1)
        temperature_history = np.roll(temperature_history, -1)
        radiation_history   = np.roll(radiation_history,   -1)

        water_history[-1]       = W + S   # irrigation + precipitation
        fertilizer_history[-1]  = F
        temperature_history[-1] = T
        radiation_history[-1]   = R

        # Convolve input disturbances with FIR kernels to model delayed absorption/metalysis
        delayed_water       = np.dot(kernel_W, water_history)
        delayed_fertilizer  = np.dot(kernel_F, fertilizer_history)
        delayed_temperature = np.dot(kernel_T, temperature_history)
        delayed_radiation   = np.dot(kernel_R, radiation_history)
        
        # Update cumulative delayed values
        cumulative_water       = cumulative_water       + delayed_water
        cumulative_fertilizer  = cumulative_fertilizer  + delayed_fertilizer
        cumulative_temperature = cumulative_temperature + delayed_temperature
        cumulative_radiation   = cumulative_radiation   + delayed_radiation

        # Calculate the differences between the expected and actual cumulative values
        epsilon = 1e-6  # small value to prevent division by zero
        water_anomaly       = max(np.abs(W_typ * (t-1) - cumulative_water)       / (W_typ * t + epsilon), epsilon)
        fertilizer_anomaly  = max(np.abs(F_typ * (t-1) - cumulative_fertilizer)  / (F_typ * t + epsilon), epsilon)
        temperature_anomaly = max(np.abs(T_typ * (t-1) - cumulative_temperature) / (T_typ * t + epsilon), epsilon)
        radiation_anomaly   = max(np.abs(R_typ * (t-1) - cumulative_radiation)   / (R_typ * t + epsilon), epsilon)

        # Recursive cumulative divergence update
        beta_divergence = 0.95
        cumulative_divergence_water       = beta_divergence * cumulative_divergence_water       + (1.0 - beta_divergence) * water_anomaly
        cumulative_divergence_fertilizer  = beta_divergence * cumulative_divergence_fertilizer  + (1.0 - beta_divergence) * fertilizer_anomaly
        cumulative_divergence_temperature = beta_divergence * cumulative_divergence_temperature + (1.0 - beta_divergence) * temperature_anomaly
        cumulative_divergence_radiation   = beta_divergence * cumulative_divergence_radiation   + (1.0 - beta_divergence) * radiation_anomaly

        # Raw nutrient factors
        alpha = 3.0
        nuW_raw = np.exp(-alpha * cumulative_divergence_water)
        nuF_raw = np.exp(-alpha * cumulative_divergence_fertilizer)
        nuT_raw = np.exp(-alpha * cumulative_divergence_temperature)
        nuR_raw = np.exp(-alpha * cumulative_divergence_radiation)

        # Final, smoothed nutrient factors
        beta_nutrient_factor = 0.05
        nuW = beta_nutrient_factor * nuW + (1 - beta_nutrient_factor) * nuW_raw
        nuF = beta_nutrient_factor * nuF + (1 - beta_nutrient_factor) * nuF_raw
        nuT = beta_nutrient_factor * nuT + (1 - beta_nutrient_factor) * nuT_raw
        nuR = beta_nutrient_factor * nuR + (1 - beta_nutrient_factor) * nuR_raw

        # Calculate the instantaneous adjusted growth rates and carrying capacities
        ah_hat = np.clip(ah * (nuF * nuT * nuR)**(1/3), 0, 2 * ah)
        aA_hat = np.clip(aA * (nuF * nuT * nuR)**(1/3), 0, 2 * aA)
        aN_hat = np.clip(aN, 0, 2 * aN)
        ac_hat = np.clip(ac * ( (1/nuT) * (1/nuR) )**(1/2), 0, 2 * ac)
        aP_hat = np.clip(aP * (nuT * nuR)**(1/2), 0, 2 * aP)

        kh_hat = np.clip(kh * (nuF * nuT * nuR)**(1/3), h, 2 * kh)
        kA_hat = np.clip(kA * (nuW * nuF * nuT * nuR * (kh_hat/kh))**(1/5), A, 2 * kA)
        kN_hat = np.clip(kN * (nuT * nuR)**(1/2), N, 2 * kN)
        kc_hat = np.clip(kc * (nuW * (1/nuT) * (1/nuR))**(1/3), c, 2 * kc)
        kP_hat = np.clip(kP * (nuW * nuF * nuT * nuR * (kh_hat/kh) * (kA_hat/kA) * (kc_hat/kc))**(1/7), P, 2 * kP)

        # Logistic-style updates
        h = logistic_step(h, ah_hat, kh_hat, dt)
        A = logistic_step(A, aA_hat, kA_hat, dt)
        N = logistic_step(N, aN_hat, kN_hat, dt)
        c = logistic_step(c, ac_hat, kc_hat, dt)
        P = logistic_step(P, aP_hat, kP_hat, dt)

    # Combined objective (negative because GA minimizes)
    profit = weight_fruit_biomass * P + weight_height * h + weight_leaf_area * A
    expenses = (weight_irrigation * np.sum(irrigation)
                + weight_fertilizer * np.sum(fertilizer))
    revenue = profit - expenses
    cost = -revenue # GA minimizes cost, but we want to maximize revenue

    return float(cost)
