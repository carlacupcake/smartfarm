# ga_member.py
import pandas as pd
import mpmath as mp
import numpy as np
from typing import Optional

from .ga_params import GeneticAlgorithmParams
from ..model.model_helpers import get_nutrient_factor, get_nutrient_factor_abs, get_sim_inputs_from_hourly, logistic_step

from ..model.model_carrying_capacities import ModelCarryingCapacities
from ..model.model_disturbances import ModelDisturbances
from ..model.model_growth_rates import ModelGrowthRates
from ..model.model_initial_conditions import ModelInitialConditions
from ..model.model_params import ModelParams
from ..model.model_typical_disturbances import ModelTypicalDisturbances

class Member:
    """
    Class to represent a member of the population in genetic algorithm optimization.
    Stores the functions for cost function calculation.
    """

    def __init__(
        self,
        ga_params:            GeneticAlgorithmParams,
        carrying_capacities:  ModelCarryingCapacities,
        disturbances:         ModelDisturbances,
        growth_rates:         ModelGrowthRates,
        initial_conditions:   ModelInitialConditions,
        model_params:         ModelParams,
        typical_disturbances: ModelTypicalDisturbances,
        values:               Optional[np.ndarray] = None,
    ):

        self.ga_params            = ga_params
        self.carrying_capacities  = carrying_capacities
        self.disturbances         = disturbances
        self.growth_rates         = growth_rates
        self.initial_conditions   = initial_conditions
        self.model_params         = model_params
        self.typical_disturbances = typical_disturbances
        self.values = values


    def get_cost(self) -> float:
        """
        Compute the objective cost for this member by simulating plant growth
        over a full season and evaluating net economic performance.

        The member’s decision variables are interpreted as:
            - irrigation frequency
            - irrigation amount
            - fertilizer frequency
            - fertilizer amount

        These controls are converted to hourly irrigation and fertilization
        schedules, combined with exogenous disturbances (precipitation,
        temperature, radiation), and used to drive a coupled logistic-style
        growth model for plant height, leaf area, leaf count, stress metric,
        and fruit biomass. The system is integrated in time using a
        forward-Euler scheme with time step `dt`.

        The final fruit biomass and total irrigation/fertilizer use are then
        converted into a scalar objective: net revenue (profit minus input
        costs). The cost returned by this function is the negative of that
        net revenue so that the genetic algorithm can be formulated as a
        minimization problem.

        Args:
            None explicitly. This method uses the member’s internal state:
                - self.values: design variables
                - self.model_params: time-stepping and horizon settings
                - self.disturbances: precipitation, temperature, radiation
                - self.typical_disturbances: typical/normalized disturbance scales
                - self.initial_conditions: initial plant state
                - self.growth_rates: base logistic growth rates
                - self.carrying_capacities: base carrying capacities
                - self.ga_params: economic weights for revenue and costs

        Returns:
            cost (float): Objective value for this member, defined as the
                negative net revenue (weighted final fruit biomass minus
                weighted irrigation and fertilizer usage).
        """

        # Unpack in the same order as bounds: [irrig_freq, irrig_amt, fert_freq, fert_amt]
        irrigation_frequency, irrigation_amount, fertilizer_frequency, fertilizer_amount = [
            float(x) for x in self.values
        ]

        # Unpack model parameters
        dt = self.model_params.dt
        total_time_steps = self.model_params.total_time_steps
        simulation_hours = self.model_params.simulation_hours

        # Unpack disturbances
        hourly_precipitation = self.disturbances.precipitation
        hourly_temperature   = self.disturbances.temperature
        hourly_radiation     = self.disturbances.radiation

        # Unpack typical disturbances
        W_typ = self.typical_disturbances.typical_water
        F_typ = self.typical_disturbances.typical_fertilizer
        T_typ = self.typical_disturbances.typical_temperature
        R_typ = self.typical_disturbances.typical_radiation

        # Unpack initial conditions
        h0 = self.initial_conditions.h0
        A0 = self.initial_conditions.A0
        N0 = self.initial_conditions.N0
        c0 = self.initial_conditions.c0
        P0 = self.initial_conditions.P0

        # Unpack growth rates
        ah = self.growth_rates.ah
        aA = self.growth_rates.aA
        aN = self.growth_rates.aN
        ac = self.growth_rates.ac
        aP = self.growth_rates.aP

        # Unpack carrying capacities
        kh = self.carrying_capacities.kh
        kA = self.carrying_capacities.kA
        kN = self.carrying_capacities.kN
        kc = self.carrying_capacities.kc
        kP = self.carrying_capacities.kP

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
            nuW = get_nutrient_factor(x=WC/(W_typ * (t+1)), mu=1, sensitivity=0.9)
            nuF = get_nutrient_factor(x=FC/(F_typ * (t+1)), mu=1, sensitivity=0.9)
            nuT = get_nutrient_factor(x=TC/(T_typ * (t+1)), mu=1, sensitivity=0.9)
            nuR = get_nutrient_factor(x=RC/(R_typ * (t+1)), mu=1, sensitivity=0.9)

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

        # Combined objective (negative because GA minimizes)
        profit = self.ga_params.weight_fruit_biomass * P[-1]
        expenses = (self.ga_params.weight_irrigation * np.sum(hourly_irrigation)
                    + self.ga_params.weight_fertilizer * np.sum(hourly_fertilizer))
        revenue = profit - expenses
        cost = -revenue # GA minimizes cost, but we want to maximize revenue
        
        return float(cost)
    

    def get_closed_form_cost(self) -> float:
        """
        Compute the cost for this member using the same plant-growth
        season model as `get_cost`, but replacing the Forward Euler integration
        with the analytic closed-form solution of the logistic equation at each
        time step.

        Args:
            None explicitly. This method relies on internal fields, including:
            - self.values (decision variables),
            - model parameters,
            - disturbances,
            - initial conditions,
            - growth rates and carrying capacities,
            - GA economic weights.

        Returns:
            cost (float): Negative net revenue, computed from final fruit biomass
            minus weighted irrigation and fertilizer usage.
        """

        # Unpack in the same order as bounds: [irrig_freq, irrig_amt, fert_freq, fert_amt]
        irrigation_frequency, irrigation_amount, fertilizer_frequency, fertilizer_amount = [
            float(x) for x in self.values
        ]

        # Unpack model parameters
        dt = self.model_params.dt
        total_time_steps = self.model_params.total_time_steps
        simulation_hours = self.model_params.simulation_hours

        # Unpack disturbances
        hourly_precipitation = self.disturbances.precipitation
        hourly_temperature   = self.disturbances.temperature
        hourly_radiation     = self.disturbances.radiation

        # Unpack typical disturbances
        W_typ = self.typical_disturbances.typical_water
        F_typ = self.typical_disturbances.typical_fertilizer
        T_typ = self.typical_disturbances.typical_temperature
        R_typ = self.typical_disturbances.typical_radiation

        # Unpack initial conditions
        h0 = self.initial_conditions.h0
        A0 = self.initial_conditions.A0
        N0 = self.initial_conditions.N0
        c0 = self.initial_conditions.c0
        P0 = self.initial_conditions.P0

        # Unpack growth rates
        ah = self.growth_rates.ah
        aA = self.growth_rates.aA
        aN = self.growth_rates.aN
        ac = self.growth_rates.ac
        aP = self.growth_rates.aP

        # Unpack carrying capacities
        kh = self.carrying_capacities.kh
        kA = self.carrying_capacities.kA
        kN = self.carrying_capacities.kN
        kc = self.carrying_capacities.kc
        kP = self.carrying_capacities.kP

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
            nuW = get_nutrient_factor(x=WC/(W_typ * (t+1)), mu=1, sensitivity=0.8)
            nuF = get_nutrient_factor(x=FC/(F_typ * (t+1)), mu=1, sensitivity=0.8)
            nuT = get_nutrient_factor(x=TC/(T_typ * (t+1)), mu=1, sensitivity=0.8)
            nuR = get_nutrient_factor(x=RC/(R_typ * (t+1)), mu=1, sensitivity=0.8)

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
            h[t+1] = logistic_step(h[t], ah_hat, kh_hat, dt)
            A[t+1] = logistic_step(A[t], aA_hat, kA_hat, dt)
            N[t+1] = logistic_step(N[t], aN_hat, kN_hat, dt)
            c[t+1] = logistic_step(c[t], ac_hat, kc_hat, dt)
            P[t+1] = logistic_step(P[t], aP_hat, kP_hat, dt)

        # Combined objective (negative because GA minimizes)
        profit = self.ga_params.weight_fruit_biomass * P[-1]
        expenses = (self.ga_params.weight_irrigation * np.sum(hourly_irrigation)
                    + self.ga_params.weight_fertilizer * np.sum(hourly_fertilizer))
        revenue = profit - expenses
        cost = -revenue # GA minimizes cost, but we want to maximize revenue
        
        return float(cost)
    

    def get_cost_verbose(
            self,
            nuW_sens = 0.95,
            nuF_sens = 0.95,
            nuT_sens = 0.95,
            nuR_sens = 0.95
        ) -> float:
        """
        Compute the cost for this member using the same plant-growth
        season model as `get_cost`, but return additional internal state for
        debugging and analysis purposes.

        Args:
            nuW_sens (float, optional):
                Sensitivity parameter for the water-based nutrient factor.
            nuF_sens (float, optional):
                Sensitivity parameter for the fertilizer-based nutrient factor.
            nuT_sens (float, optional):
                Sensitivity parameter for the temperature-based nutrient factor.
            nuR_sens (float, optional):
                Sensitivity parameter for the radiation-based nutrient factor.
                These allow debugging how each environmental driver influences
                growth-rate and carrying-capacity adjustments.

        Returns:
            A tuple containing the full simulated internal state across all
            time steps. The elements are, in order:

                h (ndarray):
                    Plant height trajectory.
                A (ndarray):
                    Leaf area trajectory.
                N (ndarray):
                    Leaf count trajectory.
                c (ndarray):
                    Flower spikelet trajectory.
                P (ndarray):
                    Fruit biomass trajectory.

                nuW_values, nuF_values, nuT_values, nuR_values (ndarray):
                    Time series of nutrient-factor values for water, fertilizer,
                    temperature, and radiation.

                ah_hat_values, aA_hat_values, aN_hat_values,
                ac_hat_values, aP_hat_values (ndarray):
                    Adjusted growth-rates at each time step.

                kh_hat_values, kA_hat_values, kN_hat_values,
                kc_hat_values, kP_hat_values (ndarray):
                    Adjusted carrying capacities at each time step.
        """

        # Unpack in the same order as bounds: [irrig_freq, irrig_amt, fert_freq, fert_amt]
        irrigation_frequency, irrigation_amount, fertilizer_frequency, fertilizer_amount = [float(x) for x in self.values]

        # Unpack model parameters
        dt = self.model_params.dt
        total_time_steps = self.model_params.total_time_steps
        simulation_hours = self.model_params.simulation_hours

        # Unpack disturbances
        hourly_precipitation = self.disturbances.precipitation
        hourly_temperature   = self.disturbances.temperature
        hourly_radiation     = self.disturbances.radiation

        # Unpack typical disturbances
        W_typ = self.typical_disturbances.typical_water
        F_typ = self.typical_disturbances.typical_fertilizer
        T_typ = self.typical_disturbances.typical_temperature
        R_typ = self.typical_disturbances.typical_radiation

        # Unpack initial conditions
        h0 = self.initial_conditions.h0
        A0 = self.initial_conditions.A0
        N0 = self.initial_conditions.N0
        c0 = self.initial_conditions.c0
        P0 = self.initial_conditions.P0

        # Unpack growth rates
        ah = self.growth_rates.ah
        aA = self.growth_rates.aA
        aN = self.growth_rates.aN
        ac = self.growth_rates.ac
        aP = self.growth_rates.aP

        # Unpack carrying capacities
        kh = self.carrying_capacities.kh
        kA = self.carrying_capacities.kA
        kN = self.carrying_capacities.kN
        kc = self.carrying_capacities.kc
        kP = self.carrying_capacities.kP

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

        # Initialize storage of nutrient factors
        nuW_values = np.zeros(total_time_steps)
        nuF_values = np.zeros(total_time_steps)
        nuT_values = np.zeros(total_time_steps)
        nuR_values = np.zeros(total_time_steps)

        # Intialize storage of adjusted growth rates
        ah_hat_values = np.zeros(total_time_steps)
        aA_hat_values = np.zeros(total_time_steps)
        aN_hat_values = np.zeros(total_time_steps)
        ac_hat_values = np.zeros(total_time_steps)
        aP_hat_values = np.zeros(total_time_steps)

        # Initialize storage of adjusted carrying capacities
        kh_hat_values = np.zeros(total_time_steps)
        kA_hat_values = np.zeros(total_time_steps)
        kN_hat_values = np.zeros(total_time_steps)
        kc_hat_values = np.zeros(total_time_steps)
        kP_hat_values = np.zeros(total_time_steps)

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
            nuW = get_nutrient_factor(x=WC/(W_typ * (t+1)), mu=1, sensitivity=nuW_sens)
            nuF = get_nutrient_factor(x=FC/(F_typ * (t+1)), mu=1, sensitivity=nuF_sens)
            nuT = get_nutrient_factor(x=TC/(T_typ * (t+1)), mu=1, sensitivity=nuT_sens)
            nuR = get_nutrient_factor(x=RC/(R_typ * (t+1)), mu=1, sensitivity=nuR_sens)

            nuW_values[t] = nuW
            nuF_values[t] = nuF
            nuT_values[t] = nuT
            nuR_values[t] = nuR

            # Growth rates
            ah_hat = np.clip(ah * (nuF * nuT * nuR)**(1/3), 0, 10 * ah)
            aA_hat = np.clip(aA * (nuF * nuT * nuR)**(1/3), 0, 10 * aA)
            aN_hat = np.clip(aN, 0, 10 * aN)
            ac_hat = np.clip(ac * ( (1/nuT) * (1/nuR) )**(1/2), 0, 10 * ac)
            aP_hat = np.clip(aP * (nuT * nuR)**(1/2), 0, 10 * aP)

            ah_hat_values[t] = ah_hat
            aA_hat_values[t] = aA_hat
            aN_hat_values[t] = aN_hat
            ac_hat_values[t] = ac_hat
            aP_hat_values[t] = aP_hat

            # Carrying capacities
            kh_hat = np.clip(kh * (nuF * nuT * nuR)**(1/3), 0, 10 * kh)
            kA_hat = np.clip(kA * (nuW * nuF * nuT * nuR * (kh_hat/kh))**(1/5), 0, 10 * kA)
            kN_hat = np.clip(kN * (nuT * nuR)**(1/2), 0, 10 * kN)
            kc_hat = np.clip(kc * (nuW * (1/nuT) * (1/nuR))**(1/3), 0, 10 * kc)
            kP_hat = np.clip(kP * (nuW * nuF * nuT * nuR * (kh_hat/kh) * (kA_hat/kA) * (kc_hat/kc))**(1/7), 0, 10 * kP)

            kh_hat_values[t] = kh_hat
            kA_hat_values[t] = kA_hat
            kN_hat_values[t] = kN_hat
            kc_hat_values[t] = kc_hat
            kP_hat_values[t] = kP_hat

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

        return h, A, N, c, P, nuW_values, nuF_values, nuT_values, nuR_values, ah_hat_values, aA_hat_values, aN_hat_values, ac_hat_values, aP_hat_values, kh_hat_values, kA_hat_values, kN_hat_values, kc_hat_values, kP_hat_values
    

    def get_closed_form_cost_verbose(
            self,
            sigma_W = 10,
            sigma_F = 500,
            sigma_T = 30,
            sigma_R = 30 
        ) -> float:
        """
        Compute the cost for this member using the same plant-growth
        season model as `get_closed_form_cost`, but return additional internal 
        state for debugging and analysis purposes.

        Args:
            nuW_sens (float, optional):
                Sensitivity parameter for the water-based nutrient factor.
            nuF_sens (float, optional):
                Sensitivity parameter for the fertilizer-based nutrient factor.
            nuT_sens (float, optional):
                Sensitivity parameter for the temperature-based nutrient factor.
            nuR_sens (float, optional):
                Sensitivity parameter for the radiation-based nutrient factor.
                These allow debugging how each environmental driver influences
                growth-rate and carrying-capacity adjustments.

        Returns:
            A tuple containing the full simulated internal state across all
            time steps. The elements are, in order:

                h (ndarray):
                    Plant height trajectory.
                A (ndarray):
                    Leaf area trajectory.
                N (ndarray):
                    Leaf count trajectory.
                c (ndarray):
                    Flower spikelet trajectory.
                P (ndarray):
                    Fruit biomass trajectory.

                nuW_values, nuF_values, nuT_values, nuR_values (ndarray):
                    Time series of nutrient-factor values for water, fertilizer,
                    temperature, and radiation.

                ah_hat_values, aA_hat_values, aN_hat_values,
                ac_hat_values, aP_hat_values (ndarray):
                    Adjusted growth-rates at each time step.

                kh_hat_values, kA_hat_values, kN_hat_values,
                kc_hat_values, kP_hat_values (ndarray):
                    Adjusted carrying capacities at each time step.
        """

        # Unpack in the same order as bounds: [irrig_freq, irrig_amt, fert_freq, fert_amt]
        irrigation_frequency, irrigation_amount, fertilizer_frequency, fertilizer_amount = [float(x) for x in self.values]

        # Unpack model parameters
        dt = self.model_params.dt
        total_time_steps = self.model_params.total_time_steps
        simulation_hours = self.model_params.simulation_hours

        # Unpack disturbances
        hourly_precipitation = self.disturbances.precipitation
        hourly_temperature   = self.disturbances.temperature
        hourly_radiation     = self.disturbances.radiation

        # Unpack typical disturbances
        W_typ = self.typical_disturbances.typical_water
        F_typ = self.typical_disturbances.typical_fertilizer
        T_typ = self.typical_disturbances.typical_temperature
        R_typ = self.typical_disturbances.typical_radiation

        # Unpack initial conditions
        h0 = self.initial_conditions.h0
        A0 = self.initial_conditions.A0
        N0 = self.initial_conditions.N0
        c0 = self.initial_conditions.c0
        P0 = self.initial_conditions.P0

        # Unpack growth rates
        ah = self.growth_rates.ah
        aA = self.growth_rates.aA
        aN = self.growth_rates.aN
        ac = self.growth_rates.ac
        aP = self.growth_rates.aP

        # Unpack carrying capacities
        kh = self.carrying_capacities.kh
        kA = self.carrying_capacities.kA
        kN = self.carrying_capacities.kN
        kc = self.carrying_capacities.kc
        kP = self.carrying_capacities.kP

        # Build hourly control series from design variables and input disturbances
        hourly_irrigation = np.zeros(simulation_hours)
        step_if = max(1, int(np.ceil(irrigation_frequency)))
        hourly_irrigation[::step_if] = irrigation_amount
        irrigation = get_sim_inputs_from_hourly(hourly_irrigation, dt, simulation_hours, mode='split')

        hourly_fertilizer = np.zeros(simulation_hours)
        step_ff = max(1, int(np.ceil(fertilizer_frequency)))
        hourly_fertilizer[::step_ff] = fertilizer_amount
        fertilizer = get_sim_inputs_from_hourly(hourly_fertilizer, dt, simulation_hours, mode='split')

        precipitation = get_sim_inputs_from_hourly(
            hourly_array     = hourly_precipitation,
            dt               = dt,
            simulation_hours = simulation_hours,
            mode             = 'split')
        temperature = get_sim_inputs_from_hourly(
            hourly_array     = hourly_temperature,
            dt               = dt,
            simulation_hours = simulation_hours,
            mode             = 'split')
            #mode             = 'copy')
        radiation = get_sim_inputs_from_hourly(
            hourly_array     = hourly_radiation,
            dt               = dt,
            simulation_hours = simulation_hours,
            #mode             = 'copy')
            mode             = 'split')

        # Initialize storage for state variables
        h = np.full(total_time_steps, h0)
        A = np.full(total_time_steps, A0)
        N = np.full(total_time_steps, N0)
        c = np.full(total_time_steps, c0)
        P = np.full(total_time_steps, P0)

        # Initialize storage of nutrient factors
        nuW_values = np.zeros(total_time_steps)
        nuF_values = np.zeros(total_time_steps)
        nuT_values = np.zeros(total_time_steps)
        nuR_values = np.zeros(total_time_steps)

        # Intialize storage of adjusted growth rates
        ah_hat_values = np.zeros(total_time_steps)
        aA_hat_values = np.zeros(total_time_steps)
        aN_hat_values = np.zeros(total_time_steps)
        ac_hat_values = np.zeros(total_time_steps)
        aP_hat_values = np.zeros(total_time_steps)

        # Initialize storage of adjusted carrying capacities
        kh_hat_values = np.zeros(total_time_steps)
        kA_hat_values = np.zeros(total_time_steps)
        kN_hat_values = np.zeros(total_time_steps)
        kc_hat_values = np.zeros(total_time_steps)
        kP_hat_values = np.zeros(total_time_steps)

        # Initialize storage of inputs and disturbances that take delayed absorption/metalysis into account
        delayed_water       = np.zeros(total_time_steps)
        delayed_fertilizer  = np.zeros(total_time_steps)
        delayed_temperature = np.zeros(total_time_steps)
        delayed_radiation   = np.zeros(total_time_steps)

        # Initialize storage of cumulative water and fertilizer values
        cumulative_radiation   = np.zeros(total_time_steps)
        cumulative_temperature = np.zeros(total_time_steps)
        cumulative_water       = np.zeros(total_time_steps)
        cumulative_fertilizer  = np.zeros(total_time_steps)

        # Initialize storage of deltas between expected and actual cumulative values
        delta_cumulative_water       = np.zeros(total_time_steps)
        delta_cumulative_fertilizer  = np.zeros(total_time_steps)
        delta_cumulative_temperature = np.zeros(total_time_steps)
        delta_cumulative_radiation   = np.zeros(total_time_steps)

        # TEMP
        delta_Ws = np.zeros(total_time_steps)
        delta_Fs = np.zeros(total_time_steps)
        delta_Ts = np.zeros(total_time_steps)
        delta_Rs = np.zeros(total_time_steps)

        # Pre-calculate the mu values that correspond to 95% absorption for each sigma ("sensitivity")
        def solve_for_mu(sigma, mu_guess=100):
                f = lambda mu: mp.erf(mu/(np.sqrt(2) * sigma)) - 0.95
                mu = mp.findroot(f, mu_guess)
                return mu

        mu_W = float(solve_for_mu(sigma_W/dt))
        mu_F = float(solve_for_mu(sigma_F/dt))
        mu_T = float(solve_for_mu(sigma_T/dt))
        mu_R = float(solve_for_mu(sigma_R/dt))

        # Run the season simulation for the given member
        for t in range(total_time_steps - 1):
        #for t in range(10):
            S = precipitation[t] 
            T = temperature[t]
            R = radiation[t]
            W = irrigation[t]
            F = fertilizer[t]

            # Update future radiation, temperature, water, fertilizer based on delayed absorption/metalysis
            f = lambda x, mu, sigma, area: area / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((x - mu) / sigma)**2)

            time = np.linspace(0, total_time_steps - t, total_time_steps - t)
            delayed_water[t:]       = delayed_water[t:]       + f(time, mu_W, sigma_W/dt, W + S)
            delayed_fertilizer[t:]  = delayed_fertilizer[t:]  + f(time, mu_F, sigma_F/dt, F)
            delayed_temperature[t:] = delayed_temperature[t:] + f(time, mu_T, sigma_T/dt, T)
            delayed_radiation[t:]   = delayed_radiation[t:]   + f(time, mu_R, sigma_R/dt, R)

            # Update cumulative values
            cumulative_water[t+1]       = cumulative_water[t]       + delayed_water[t]
            cumulative_fertilizer[t+1]  = cumulative_fertilizer[t]  + delayed_fertilizer[t]
            cumulative_temperature[t+1] = cumulative_temperature[t] + delayed_temperature[t]
            cumulative_radiation[t+1]   = cumulative_radiation[t]   + delayed_radiation[t]

            # Calculate the differences in cumulative values and expectation
            delta_W = (W_typ * (t+1) - cumulative_water[t+1])/ (W_typ * (t+1))
            delta_F = (F_typ * (t+1) - cumulative_fertilizer[t+1]) / (F_typ * (t+1))
            delta_T = (T_typ * (t+1) - cumulative_temperature[t+1]) / (T_typ * (t+1))
            delta_R = (R_typ * (t+1) - cumulative_radiation[t+1]) / (R_typ * (t+1))

            delta_Ws[t+1] = delta_W
            delta_Fs[t+1] = delta_F
            delta_Ts[t+1] = delta_T
            delta_Rs[t+1] = delta_R

            # Update the cumulative deltas
            delta_cumulative_water[t+1]       = (delta_cumulative_water[t]       + delta_W)/2
            delta_cumulative_fertilizer[t+1]  = (delta_cumulative_fertilizer[t]  + delta_F)/2
            delta_cumulative_temperature[t+1] = (delta_cumulative_temperature[t] + delta_T)/2
            delta_cumulative_radiation[t+1]   = (delta_cumulative_radiation[t]   + delta_R)/2

            # Nutrient factors (bounded, nonnegative)
            nuW = np.abs(1 - delta_cumulative_water[t])
            nuF = np.abs(1 - delta_cumulative_fertilizer[t])
            nuT = np.abs(1 - delta_cumulative_temperature[t])
            nuR = np.abs(1 - delta_cumulative_radiation[t])

            nuW_values[t] = nuW
            nuF_values[t] = nuF
            nuT_values[t] = nuT
            nuR_values[t] = nuR

            # Growth rates
            ah_hat = np.clip(ah * (nuF * nuT * nuR)**(1/3), 0, 10 * ah)
            aA_hat = np.clip(aA * (nuF * nuT * nuR)**(1/3), 0, 10 * aA)
            aN_hat = np.clip(aN, 0, 10 * aN)
            ac_hat = np.clip(ac * ( (1/nuT) * (1/nuR) )**(1/2), 0, 10 * ac)
            aP_hat = np.clip(aP * (nuT * nuR)**(1/2), 0, 10 * aP)

            ah_hat_values[t] = ah_hat
            aA_hat_values[t] = aA_hat
            aN_hat_values[t] = aN_hat
            ac_hat_values[t] = ac_hat
            aP_hat_values[t] = aP_hat

            # Carrying capacities
            kh_hat = np.clip(kh * (nuF * nuT * nuR)**(1/3), 0, 10 * kh)
            kA_hat = np.clip(kA * (nuW * nuF * nuT * nuR * (kh_hat/kh))**(1/5), 0, 10 * kA)
            kN_hat = np.clip(kN * (nuT * nuR)**(1/2), 0, 10 * kN)
            kc_hat = np.clip(kc * (nuW * (1/nuT) * (1/nuR))**(1/3), 0, 10 * kc)
            kP_hat = np.clip(kP * (nuW * nuF * nuT * nuR * (kh_hat/kh) * (kA_hat/kA) * (kc_hat/kc))**(1/7), 0, 10 * kP)

            kh_hat_values[t] = kh_hat
            kA_hat_values[t] = kA_hat
            kN_hat_values[t] = kN_hat
            kc_hat_values[t] = kc_hat
            kP_hat_values[t] = kP_hat

            # Logistic-style updates
            h[t+1] = logistic_step(h[t], ah_hat, kh_hat, dt)
            A[t+1] = logistic_step(A[t], aA_hat, kA_hat, dt)
            N[t+1] = logistic_step(N[t], aN_hat, kN_hat, dt)
            c[t+1] = logistic_step(c[t], ac_hat, kc_hat, dt)
            P[t+1] = logistic_step(P[t], aP_hat, kP_hat, dt)

        # Save all the data of interest to a csv for further analysis
        df = pd.DataFrame({
            'h': h.flatten(),
            'A': A.flatten(),
            'N': N.flatten(),
            'c': c.flatten(),
            'P': P.flatten(),
            'delayed_water': delayed_water.flatten(),
            'delayed_fertilizer': delayed_fertilizer.flatten(),
            'delayed_temperature': delayed_temperature.flatten(),
            'delayed_radiation': delayed_radiation.flatten(),
            'cumulative_radiation': cumulative_radiation.flatten(),
            'cumulative_temperature': cumulative_temperature.flatten(),
            'cumulative_water': cumulative_water.flatten(),
            'cumulative_fertilizer': cumulative_fertilizer.flatten(),
            'delta_cumulative_water': delta_cumulative_water.flatten(),
            'delta_cumulative_fertilizer': delta_cumulative_fertilizer.flatten(),
            'delta_cumulative_temperature': delta_cumulative_temperature.flatten(),
            'delta_cumulative_radiation': delta_cumulative_radiation.flatten(),
            'delta_Ws': delta_Ws.flatten(),
            'delta_Fs': delta_Fs.flatten(),
            'delta_Ts': delta_Ts.flatten(),
            'delta_Rs': delta_Rs.flatten(),
            'nuW_values': nuW_values.flatten(),
            'nuF_values': nuF_values.flatten(),
            'nuT_values': nuT_values.flatten(),
            'nuR_values': nuR_values.flatten(),
            'ah_hat_values': ah_hat_values.flatten(),
            'aA_hat_values': aA_hat_values.flatten(),
            'aN_hat_values': aN_hat_values.flatten(),
            'ac_hat_values': ac_hat_values.flatten(),
            'aP_hat_values': aP_hat_values.flatten(),
            'kh_hat_values': kh_hat_values.flatten(),
            'kA_hat_values': kA_hat_values.flatten(),
            'kN_hat_values': kN_hat_values.flatten(),
            'kc_hat_values': kc_hat_values.flatten(),
            'kP_hat_values': kP_hat_values.flatten(),
        })

        df.to_csv("output_get_closed_form_cost_verbose.csv", index=False)

        return 
