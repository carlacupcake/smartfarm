# lambda_member.py
import numpy as np

print(">>> LOADED lambda_member.py <<<")

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


def member_get_cost_with_lambda(
        member_dict: dict,
        ctx: dict
    ) -> float:

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
        nuW = get_nutrient_factor(x=WC/(W_typ * (t+1)), mu=1, sensitivity=0.7)
        nuF = get_nutrient_factor(x=FC/(F_typ * (t+1)), mu=1, sensitivity=0.7)
        nuT = get_nutrient_factor(x=TC/(T_typ * (t+1)), mu=1, sensitivity=0.7)
        nuR = get_nutrient_factor(x=RC/(R_typ * (t+1)), mu=1, sensitivity=0.7)

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
        h[t+1] = h[t] + dt * (ah_hat * h[t] * (1 - h[t]/max(kh_hat, 1e-9)))
        A[t+1] = A[t] + dt * (aA_hat * A[t] * (1 - A[t]/max(kA_hat, 1e-9)))
        N[t+1] = N[t] + dt * (aN_hat * N[t] * (1 - N[t]/max(kN_hat, 1e-9)))
        c[t+1] = c[t] + dt * (ac_hat * c[t] * (1 - c[t]/max(kc_hat, 1e-9)))
        P[t+1] = P[t] + dt * (aP_hat * P[t] * (1 - P[t]/max(kP_hat, 1e-9)))

    profit = weight_fruit_biomass * P[-1]
    expenses = (weight_irrigation * np.sum(hourly_irrigation)
                + weight_fertilizer * np.sum(hourly_fertilizer))
    revenue = profit - expenses
    cost = -revenue # GA minimizes cost, but we want to maximize revenue
    
    return float(cost)
