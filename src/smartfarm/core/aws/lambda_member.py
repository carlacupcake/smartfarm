# lambda_member.py
import numpy as np

print(">>> LOADED lambda_member.py <<<")

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

    print(">>> INSIDE member_get_cost_with_lambda <<<")

    # Design variables (same order as bounds)
    irrigation_frequency, irrigation_amount, fertilizer_frequency, fertilizer_amount = [
        float(x) for x in member_dict["values"]
    ]

    # Unpack model parameters
    dt               = ctx["dt"]
    total_time_steps = ctx["total_time_steps"]
    simulation_hours = ctx["simulation_hours"]

    # Unpack input disturbances
    hourly_precipitation = np.array(ctx["hourly_precipitation"], dtype=float)
    hourly_temperature   = np.array(ctx["hourly_temperature"],   dtype=float)
    hourly_radiation     = np.array(ctx["hourly_radiation"],     dtype=float)

    # Unpack typical disturbances
    W_typ = ctx["W_typ"]
    F_typ = ctx["F_typ"]
    T_typ = ctx["T_typ"]
    R_typ = ctx["R_typ"]

    # Unpack initial conditions
    h0 = ctx["h0"]
    A0 = ctx["A0"]
    N0 = ctx["N0"]
    c0 = ctx["c0"]
    P0 = ctx["P0"]

    # Unpack growth rates
    ah = ctx["ah"]
    aA = ctx["aA"]
    aN = ctx["aN"]
    ac = ctx["ac"]
    aP = ctx["aP"]

    # Unpack carrying capacities
    kh = ctx["kh"]
    kA = ctx["kA"]
    kN = ctx["kN"]
    kc = ctx["kc"]
    kP = ctx["kP"]

    # Unpack GA cost function weights
    weight_fruit_biomass = ctx["weight_fruit_biomass"]
    weight_irrigation    = ctx["weight_irrigation"]
    weight_fertilizer    = ctx["weight_fertilizer"]

    # Build hourly control series from design variables and input disturbances
    hourly_irrigation = np.zeros(simulation_hours)
    step_if = max(1, int(np.ceil(irrigation_frequency)))
    hourly_irrigation[::step_if] = irrigation_amount
    irrigation = get_sim_inputs_from_hourly(hourly_irrigation, dt, simulation_hours)

    hourly_fertilizer = np.zeros(simulation_hours)
    step_ff = max(1, int(np.ceil(fertilizer_frequency)))
    hourly_fertilizer[::step_ff] = fertilizer_amount
    fertilizer = get_sim_inputs_from_hourly(hourly_fertilizer, dt, simulation_hours)

    precipitation = get_sim_inputs_from_hourly(
        hourly_array     = hourly_precipitation,
        dt               = dt,
        simulation_hours = simulation_hours,
        mode             = 'split')
    temperature = get_sim_inputs_from_hourly(
        hourly_array     = hourly_temperature,
        dt               = dt,
        simulation_hours = simulation_hours,
        mode             = 'copy')
    radiation = get_sim_inputs_from_hourly(
        hourly_array     = hourly_radiation,
        dt               = dt,
        simulation_hours = simulation_hours,
        mode             = 'copy')

    # Initialize storage for state variables
    h = np.full(total_time_steps, h0)
    A = np.full(total_time_steps, A0)
    N = np.full(total_time_steps, N0)
    c = np.full(total_time_steps, c0)
    P = np.full(total_time_steps, P0)

    # Initialize storage of cumulative water and fertilizer values
    cumulative_radiation   = np.zeros(total_time_steps)
    cumulative_temperature = np.zeros(total_time_steps)
    cumulative_water       = np.zeros(total_time_steps)
    cumulative_fertilizer  = np.zeros(total_time_steps)

    # Run the season simulation for the given member
    for t in range(total_time_steps - 1):
        S = precipitation[t] 
        T = temperature[t]
        R = radiation[t]
        W = irrigation[t]
        F = fertilizer[t]

        RC = cumulative_radiation[t] + R
        cumulative_radiation[t+1] = RC

        TC = cumulative_temperature[t] + T
        cumulative_temperature[t+1] = TC

        WC = cumulative_water[t] + W + S
        cumulative_water[t+1] = WC

        FC = cumulative_fertilizer[t] + F
        cumulative_fertilizer[t+1] = FC

        # Nutrient factors (bounded, nonnegative)
        nuW = get_nutrient_factor(x=WC/(W_typ * (t+1)), mu=1, sensitivity=0.95)
        nuF = get_nutrient_factor(x=FC/(F_typ * (t+1)), mu=1, sensitivity=0.95)
        nuT = get_nutrient_factor(x=TC/(T_typ * (t+1)), mu=1, sensitivity=0.95)
        nuR = get_nutrient_factor(x=RC/(R_typ * (t+1)), mu=1, sensitivity=0.95)

        # Growth rates
        ah_hat = np.clip(ah * (nuF * nuT * nuR)**(1/3), 0, 10 * ah)
        aA_hat = np.clip(aA * (nuF * nuT * nuR)**(1/3), 0, 10 * aA)
        aN_hat = np.clip(aN, 0, 10 * aN)
        ac_hat = np.clip(ac * ( (1/nuT) * (1/nuR) )**(1/2), 0, 10 * ac)
        aP_hat = np.clip(aP * (nuT * nuR)**(1/2), 0, 10 * aP)

        # Carrying capacities
        kh_hat = np.clip(kh * (nuF * nuT * nuR)**(1/3), 0, 10 * kh)
        kA_hat = np.clip(kA * (nuW * nuF * nuT * nuR * (kh_hat/kh))**(1/5), 0, 10 * kA)
        kN_hat = np.clip(kN * (nuT * nuR)**(1/2), 0, 10 * kN)
        kc_hat = np.clip(kc * (nuW * (1/nuT) * (1/nuR))**(1/3), 0, 10 * kc)
        kP_hat = np.clip(kP * (nuW * nuF * nuT * nuR * (kh_hat/kh) * (kA_hat/kA) * (kc_hat/kc))**(1/7), 0, 10 * kP)

        # Logistic-style updates
        h[t+1] = h[t] + dt * (ah_hat * h[t] * (1 - h[t]/max(kh_hat, 1e-9)))
        A[t+1] = A[t] + dt * (aA_hat * A[t] * (1 - A[t]/max(kA_hat, 1e-9)))
        N[t+1] = N[t] + dt * (aN_hat * N[t] * (1 - N[t]/max(kN_hat, 1e-9)))
        c[t+1] = c[t] + dt * (ac_hat * c[t] * (1 - c[t]/max(kc_hat, 1e-9)))
        P[t+1] = P[t] + dt * (aP_hat * P[t] * (1 - P[t]/max(kP_hat, 1e-9)))

        # Enforce non-negativity explicitly
        h[t+1] = max(h[t+1], 0.0)
        A[t+1] = max(A[t+1], 0.0)
        N[t+1] = max(N[t+1], 0.0)
        c[t+1] = max(c[t+1], 0.0)
        P[t+1] = max(P[t+1], 0.0)

    profit = weight_fruit_biomass * P[-1]
    expenses = (weight_irrigation * np.sum(hourly_irrigation)
                + weight_fertilizer * np.sum(hourly_fertilizer))
    revenue = profit - expenses
    cost = -revenue # GA minimizes cost, but we want to maximize revenue
    
    return float(cost)


def get_closed_form_cost_with_lambda(
        member_dict: dict,
        ctx: dict
    ) -> float:
    """
    Evaluate a member’s cost inside an AWS Lambda environment by running the
    same closed-form plant-growth simulation as `get_closed_form_cost`, using only the
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

    print(">>> INSIDE member_get_cost_with_lambda <<<")

    # Design variables (same order as bounds)
    irrigation_frequency, irrigation_amount, fertilizer_frequency, fertilizer_amount = [
        float(x) for x in member_dict["values"]
    ]

    # Unpack model parameters
    dt               = ctx["dt"]
    total_time_steps = ctx["total_time_steps"]
    simulation_hours = ctx["simulation_hours"]

    # Unpack input disturbances
    hourly_precipitation = np.array(ctx["hourly_precipitation"], dtype=float)
    hourly_temperature   = np.array(ctx["hourly_temperature"],   dtype=float)
    hourly_radiation     = np.array(ctx["hourly_radiation"],     dtype=float)

    # Unpack typical disturbances
    W_typ = ctx["W_typ"]
    F_typ = ctx["F_typ"]
    T_typ = ctx["T_typ"]
    R_typ = ctx["R_typ"]

    # Unpack initial conditions
    h0 = ctx["h0"]
    A0 = ctx["A0"]
    N0 = ctx["N0"]
    c0 = ctx["c0"]
    P0 = ctx["P0"]

    # Unpack growth rates
    ah = ctx["ah"]
    aA = ctx["aA"]
    aN = ctx["aN"]
    ac = ctx["ac"]
    aP = ctx["aP"]

    # Unpack carrying capacities
    kh = ctx["kh"]
    kA = ctx["kA"]
    kN = ctx["kN"]
    kc = ctx["kc"]
    kP = ctx["kP"]

    # Unpack GA cost function weights
    weight_fruit_biomass = ctx["weight_fruit_biomass"]
    weight_irrigation    = ctx["weight_irrigation"]
    weight_fertilizer    = ctx["weight_fertilizer"]

    # Build hourly control series from design variables and input disturbances
    hourly_irrigation = np.zeros(simulation_hours)
    step_if = max(1, int(np.ceil(irrigation_frequency)))
    hourly_irrigation[::step_if] = irrigation_amount
    irrigation = get_sim_inputs_from_hourly(hourly_irrigation, dt, simulation_hours)

    hourly_fertilizer = np.zeros(simulation_hours)
    step_ff = max(1, int(np.ceil(fertilizer_frequency)))
    hourly_fertilizer[::step_ff] = fertilizer_amount
    fertilizer = get_sim_inputs_from_hourly(hourly_fertilizer, dt, simulation_hours)

    precipitation = get_sim_inputs_from_hourly(
        hourly_array     = hourly_precipitation,
        dt               = dt,
        simulation_hours = simulation_hours,
        mode             = 'split')
    temperature = get_sim_inputs_from_hourly(
        hourly_array     = hourly_temperature,
        dt               = dt,
        simulation_hours = simulation_hours,
        mode             = 'copy')
    radiation = get_sim_inputs_from_hourly(
        hourly_array     = hourly_radiation,
        dt               = dt,
        simulation_hours = simulation_hours,
        mode             = 'copy')

    # Initialize storage for state variables
    h = np.full(total_time_steps, h0)
    A = np.full(total_time_steps, A0)
    N = np.full(total_time_steps, N0)
    c = np.full(total_time_steps, c0)
    P = np.full(total_time_steps, P0)

    # Initialize storage of cumulative water and fertilizer values
    cumulative_radiation   = np.zeros(total_time_steps)
    cumulative_temperature = np.zeros(total_time_steps)
    cumulative_water       = np.zeros(total_time_steps)
    cumulative_fertilizer  = np.zeros(total_time_steps)

    # Run the season simulation for the given member
    for t in range(total_time_steps - 1):
        S = precipitation[t] 
        T = temperature[t]
        R = radiation[t]
        W = irrigation[t]
        F = fertilizer[t]

        RC = cumulative_radiation[t] + R
        cumulative_radiation[t+1] = RC

        TC = cumulative_temperature[t] + T
        cumulative_temperature[t+1] = TC

        WC = cumulative_water[t] + W + S
        cumulative_water[t+1] = WC

        FC = cumulative_fertilizer[t] + F
        cumulative_fertilizer[t+1] = FC

        # Nutrient factors (bounded, nonnegative)
        nuW = get_nutrient_factor(x=WC/(W_typ * (t+1)), mu=1, sensitivity=0.95)
        nuF = get_nutrient_factor(x=FC/(F_typ * (t+1)), mu=1, sensitivity=0.95)
        nuT = get_nutrient_factor(x=TC/(T_typ * (t+1)), mu=1, sensitivity=0.95)
        nuR = get_nutrient_factor(x=RC/(R_typ * (t+1)), mu=1, sensitivity=0.95)

        # Growth rates (FIX: use parentheses in fractional powers)
        ah_hat = ah * (nuF * nuT * nuR)**(1/3)
        aA_hat = aA * (nuF * nuT * nuR)**(1/3)
        aN_hat = aN
        ac_hat = ac * ( (1/nuT) * (1/nuR) )**(1/2)
        aP_hat = aP * (nuT * nuR)**(1/2)

        # Carrying capacities
        kh_hat = kh * (nuF * nuT * nuR)**(1/3)
        kA_hat = kA * (nuW * nuF * nuT * nuR * (kh_hat/kh))**(1/5)
        kN_hat = kN * (nuT * nuR)**(1/2)
        kc_hat = kc * (nuW * (1/nuT) * (1/nuR))**(1/3)
        kP_hat = kP * (nuW * nuF * nuT * nuR * (kh_hat/kh) * (kA_hat/kA) * (kc_hat/kc))**(1/7)

        # Logistic-style updates
        h[t+1] = logistic_step(h[t], ah_hat, kh_hat, dt)
        A[t+1] = logistic_step(A[t], aA_hat, kA_hat, dt)
        N[t+1] = logistic_step(N[t], aN_hat, kN_hat, dt)
        c[t+1] = logistic_step(c[t], ac_hat, kc_hat, dt)
        P[t+1] = logistic_step(P[t], aP_hat, kP_hat, dt)

    profit = weight_fruit_biomass * P[-1]
    expenses = (weight_irrigation * np.sum(hourly_irrigation)
                + weight_fertilizer * np.sum(hourly_fertilizer))
    revenue = profit - expenses
    cost = -revenue # GA minimizes cost, but we want to maximize revenue
    
    return float(cost)
