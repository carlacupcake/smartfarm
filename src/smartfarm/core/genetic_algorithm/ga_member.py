"""ga_member.py."""
import numpy as np
from typing import Optional

from .ga_bounds import DesignSpaceBounds
from .ga_params import GeneticAlgorithmParams
from ..model.model_helpers import get_nutrient_factor, get_sim_inputs_from_hourly

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
        TODO add doc string

        Args:
            TODO

        Returns:
            cost (float)
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
            nuW = get_nutrient_factor(x=WC/(W_typ * (t+1)), mu=1, sensitivity=0.5)
            nuF = get_nutrient_factor(x=FC/(F_typ * (t+1)), mu=1, sensitivity=0.01)
            nuT = get_nutrient_factor(x=TC/(T_typ * (t+1)), mu=1, sensitivity=0.5)
            nuR = get_nutrient_factor(x=RC/(R_typ * (t+1)), mu=1, sensitivity=0.5)

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

        # Combined objective (negative because GA minimizes)
        profit = self.ga_params.weight_fruit_biomass * P[-1]
        expenses = (self.ga_params.weight_irrigation * np.sum(irrigation)
                    + self.ga_params.weight_fertilizer * np.sum(fertilizer))
        revenue = profit - expenses
        cost = -revenue # GA minimizes cost, but we want to maximize revenue
        return float(cost)
    

    def get_cost_verbose(self) -> float:
        """
        TODO add doc string

        Args:
            TODO

        Returns:
            cost (float)
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
            nuW = get_nutrient_factor(x=WC/(W_typ * (t+1)), mu=1, sensitivity=0.7)
            nuF = get_nutrient_factor(x=FC/(F_typ * (t+1)), mu=1, sensitivity=0.1)
            nuT = get_nutrient_factor(x=TC/(T_typ * (t+1)), mu=1, sensitivity=0.5)
            nuR = get_nutrient_factor(x=RC/(R_typ * (t+1)), mu=1, sensitivity=0.7)

            nuW_values[t] = nuW
            nuF_values[t] = nuF
            nuT_values[t] = nuT
            nuR_values[t] = nuR

            # Growth rates (FIX: use parentheses in fractional powers)
            ah_hat = ah * (nuF * nuT * nuR)**(1/3)
            aA_hat = aA * (nuF * nuT * nuR)**(1/3)
            aN_hat = aN
            ac_hat = ac * ( (1/nuT) * (1/nuR) )**(1/2)
            aP_hat = aP * (nuT * nuR)**(1/2)

            ah_hat_values[t] = ah_hat
            aA_hat_values[t] = aA_hat
            aN_hat_values[t] = aN_hat
            ac_hat_values[t] = ac_hat
            aP_hat_values[t] = aP_hat

            # Carrying capacities
            kh_hat = kh * (nuF * nuT * nuR)**(1/3)
            kA_hat = kA * (nuW * nuF * nuT * nuR * (kh_hat/kh))**(1/5)
            kN_hat = kN * (nuT * nuR)**(1/2)
            kc_hat = kc * (nuW * (1/nuT) * (1/nuR))**(1/3)
            kP_hat = kP * (nuW * nuF * nuT * nuR * (kh_hat/kh) * (kA_hat/kA) * (kc_hat/kc))**(1/7)

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

        return h, A, N, c, P, nuW_values, nuF_values, nuT_values, nuR_values, ah_hat_values, aA_hat_values, aN_hat_values, ac_hat_values, aP_hat_values, kh_hat_values, kA_hat_values, kN_hat_values, kc_hat_values, kP_hat_values
