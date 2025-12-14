# ga_lambda_helpers.py
import mpmath as mp
import numpy as np

#print(">>> LOADED ga_lambda_helpers.py <<<")

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
        sigma_dt: float # standard deviation of the Gaussian in time steps
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

    Returns:
        mu (float):
            The solved value of `mu` for which `erf(mu/(sqrt(2)*sigma)) = 0.95`,
            computed using numerical root finding.
    """
    MU_FACTOR_95 = 1.959963984540054 # precomputed constant: sqrt(2) * erfinv(0.95)
    mu = MU_FACTOR_95 * sigma_dt
    return mu


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
        x (float):
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
    closed_form      = enriched_ctx.get("closed_form", True)

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
    weight_fruit_biomass = enriched_ctx["weight_fruit_biomass"]
    weight_irrigation    = enriched_ctx["weight_irrigation"]
    weight_fertilizer    = enriched_ctx["weight_fertilizer"]

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

    # Convolve input disturbances with Gaussian kernels to model delayed absorption/metalysis
    kernel_W = enriched_ctx["kernel_W"]
    kernel_F = enriched_ctx["kernel_F"]
    kernel_T = enriched_ctx["kernel_T"]
    kernel_R = enriched_ctx["kernel_R"]

    delayed_water       = np.convolve(irrigation + precipitation, kernel_W, mode="full")[:total_time_steps]
    delayed_fertilizer  = np.convolve(fertilizer,  kernel_F, mode="full")[:total_time_steps]
    delayed_temperature = np.convolve(temperature, kernel_T, mode="full")[:total_time_steps]
    delayed_radiation   = np.convolve(radiation,   kernel_R, mode="full")[:total_time_steps]

    # Calculate the cumulative values over time from the delayed values
    cumulative_water       = np.cumsum(delayed_water)
    cumulative_fertilizer  = np.cumsum(delayed_fertilizer)
    cumulative_temperature = np.cumsum(delayed_temperature)
    cumulative_radiation   = np.cumsum(delayed_radiation)

    # Calculate the differences between the expected and actual cumulative values
    t_idx_0_to_N = np.asarray(enriched_ctx["t_idx_0_to_N"]) # 0..N-1
    t_idx_1_to_N = np.asarray(enriched_ctx["t_idx_1_to_N"]) # 1..N
    delta_Ws = np.abs((W_typ * t_idx_0_to_N - cumulative_water)       / (W_typ * t_idx_1_to_N))
    delta_Fs = np.abs((F_typ * t_idx_0_to_N - cumulative_fertilizer)  / (F_typ * t_idx_1_to_N))
    delta_Ts = np.abs((T_typ * t_idx_0_to_N - cumulative_temperature) / (T_typ * t_idx_1_to_N))
    delta_Rs = np.abs((R_typ * t_idx_0_to_N - cumulative_radiation)   / (R_typ * t_idx_1_to_N))

    # Calculate the cumulative deltas over time
    delta_cumulative_water       = np.cumsum(delta_Ws) / t_idx_1_to_N
    delta_cumulative_fertilizer  = np.cumsum(delta_Fs) / t_idx_1_to_N
    delta_cumulative_temperature = np.cumsum(delta_Ts) / t_idx_1_to_N
    delta_cumulative_radiation   = np.cumsum(delta_Rs) / t_idx_1_to_N

    # Then use the cumulative deltas to calculate the nutrient factors for each time step
    nuW = np.clip(1 - np.abs(delta_cumulative_water), 0, 1)
    nuF = np.clip(1 - np.abs(delta_cumulative_fertilizer), 0, 1)
    nuT = np.clip(1 - np.abs(delta_cumulative_temperature), 0, 1)
    nuR = np.clip(1 - np.abs(delta_cumulative_radiation), 0, 1)

    # Calculate the instantaneous adjusted growth rates and carrying capacities
    ah_hat = np.clip(ah * (nuF * nuT * nuR)**(1/3), 0, 2 * ah)
    aA_hat = np.clip(aA * (nuF * nuT * nuR)**(1/3), 0, 2 * aA)
    aN_hat = np.clip(aN, 0, 2 * aN) * np.ones(total_time_steps)
    ac_hat = np.clip(ac * ( (1/nuT) * (1/nuR) )**(1/2), 0, 2 * ac)
    aP_hat = np.clip(aP * (nuT * nuR)**(1/2), 0, 2 * aP)

    kh_hat = np.clip(kh * (nuF * nuT * nuR)**(1/3), 0, 2 * kh)
    kA_hat = np.clip(kA * (nuW * nuF * nuT * nuR * (kh_hat/kh))**(1/5), 0, 2 * kA)
    kN_hat = np.clip(kN * (nuT * nuR)**(1/2), 0, 2 * kN) * np.ones(total_time_steps)
    kc_hat = np.clip(kc * (nuW * (1/nuT) * (1/nuR))**(1/3), 0, 2 * kc)
    kP_hat = np.clip(kP * (nuW * nuF * nuT * nuR * (kh_hat/kh) * (kA_hat/kA) * (kc_hat/kc))**(1/7), 0, 2 * kP)

    # Run the season simulation for the given member
    for t in range(total_time_steps - 1):

        # Logistic-style updates
        if closed_form:
            h = logistic_step(h, ah_hat[t], kh_hat[t], dt)
            A = logistic_step(A, aA_hat[t], kA_hat[t], dt)
            N = logistic_step(N, aN_hat[t], kN_hat[t], dt)
            c = logistic_step(c, ac_hat[t], kc_hat[t], dt)
            P = logistic_step(P, aP_hat[t], kP_hat[t], dt)

        else:
            # Forward Euler integration
            h = h + dt * (ah_hat[t] * h * (1 - h/max(kh_hat[t], 1e-9)))
            A = A + dt * (aA_hat[t] * A * (1 - A/max(kA_hat[t], 1e-9)))
            N = N + dt * (aN_hat[t] * N * (1 - N/max(kN_hat[t], 1e-9)))
            c = c + dt * (ac_hat[t] * c * (1 - c/max(kc_hat[t], 1e-9)))
            P = P + dt * (aP_hat[t] * P * (1 - P/max(kP_hat[t], 1e-9)))

            # Enforce non-negativity explicitly
            h = max(h, 0.0)
            A = max(A, 0.0)
            N = max(N, 0.0)
            c = max(c, 0.0)
            P = max(P, 0.0)

    # Combined objective (negative because GA minimizes)
    profit = weight_fruit_biomass * P
    expenses = (weight_irrigation * np.sum(hourly_irrigation)
                + weight_fertilizer * np.sum(hourly_fertilizer))
    revenue = profit - expenses
    cost = -revenue # GA minimizes cost, but we want to maximize revenue

    return float(cost)
