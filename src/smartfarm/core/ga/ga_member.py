# ga_member.py
import pandas as pd
import mpmath as mp
import numpy as np
from typing import Optional

from .ga_params import GeneticAlgorithmParams
from ..model.model_helpers import ema_scan, gaussian_kernel, get_mu_from_sigma, get_sim_inputs_from_hourly, logistic_step 

from ..model.model_carrying_capacities import ModelCarryingCapacities
from ..model.model_disturbances import ModelDisturbances
from ..model.model_growth_rates import ModelGrowthRates
from ..model.model_initial_conditions import ModelInitialConditions
from ..model.model_params import ModelParams
from ..model.model_typical_disturbances import ModelTypicalDisturbances
from ..model.model_sensitivities import ModelSensitivities


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
        sensitivities:        ModelSensitivities,
        values:               Optional[np.ndarray] = None,
    ):

        self.ga_params            = ga_params
        self.carrying_capacities  = carrying_capacities
        self.disturbances         = disturbances
        self.growth_rates         = growth_rates
        self.initial_conditions   = initial_conditions
        self.model_params         = model_params
        self.typical_disturbances = typical_disturbances
        self.sensitivities        = sensitivities
        self.values               = values

    def get_cost(
            self,
            irrigation: Optional[np.ndarray] = None,
            fertilizer: Optional[np.ndarray] = None,
            ) -> float:
        """
        Compute the objective cost for this member by simulating plant growth
        over a full season and evaluating net economic performance.

        The member’s decision variables are interpreted as:
            - irrigation frequency
            - irrigation amount
            - fertilizer frequency
            - fertilizer amount

        These controls are translated into hourly irrigation and fertilization
        schedules, combined with exogenous disturbances (precipitation,
        temperature, radiation), and used to drive a coupled logistic-style
        growth model for height, leaf area, leaf count, flower spikelet count, and
        fruit biomass. The system may be integrated either via a closed-form
        logistic update or using Forward Euler time stepping.

        The final fruit biomass and total irrigation/fertilizer application are
        converted into a scalar objective: net revenue. The returned cost is
        the negative of net revenue so that the genetic algorithm minimizes cost
        while maximizing profit.

        Args:
            None: all inputs are stored as member attributes.
            TODO: Optionally override irrigation and fertilizer inputs.

        Returns:
            float:
                Scalar objective cost, defined as the negative net revenue for
                this simulated growth season.
        """

        # Unpack in the same order as bounds
        irrigation_frequency, irrigation_amount, fertilizer_frequency, fertilizer_amount = [float(x) for x in self.values]

        # Unpack model parameters
        dt = self.model_params.dt
        total_time_steps = self.model_params.total_time_steps
        simulation_hours = self.model_params.simulation_hours
        closed_form      = self.model_params.closed_form
        verbose          = self.model_params.verbose

        # Unpack sensitivities
        alpha = self.sensitivities.alpha
        beta = self.sensitivities.beta
        epsilon = self.sensitivities.epsilon

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

        # Unpack sensitivities
        sigma_W = self.sensitivities.sigma_W
        sigma_F = self.sensitivities.sigma_F
        sigma_T = self.sensitivities.sigma_T
        sigma_R = self.sensitivities.sigma_R

        # Build control inputs and input disturbances by time step from hourly data
        if irrigation is not None:
            irrigation = irrigation
        else:
            hourly_irrigation = np.zeros(simulation_hours)
            step_if = max(1, int(np.ceil(irrigation_frequency)))
            hourly_irrigation[::step_if] = irrigation_amount
            irrigation = get_sim_inputs_from_hourly(hourly_irrigation, dt, simulation_hours, mode='split')

        if fertilizer is not None:
            fertilizer = fertilizer
        else:
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
        radiation = get_sim_inputs_from_hourly(
            hourly_array     = hourly_radiation,
            dt               = dt,
            simulation_hours = simulation_hours,
            mode             = 'split')

        # Initialize storage for state variables
        h = np.full(total_time_steps, h0)
        A = np.full(total_time_steps, A0)
        N = np.full(total_time_steps, N0)
        c = np.full(total_time_steps, c0)
        P = np.full(total_time_steps, P0)

        # Pre-calculate the mu values that correspond to 95% absorption for each sigma ("sensitivity")
        mu_W = get_mu_from_sigma(sigma_W/dt)
        mu_F = get_mu_from_sigma(sigma_F/dt)
        mu_T = get_mu_from_sigma(sigma_T/dt)
        mu_R = get_mu_from_sigma(sigma_R/dt)

        # Convolve input disturbances with Gaussian kernels to model delayed absorption/metalysis
        kernel_W = gaussian_kernel(mu_W, sigma_W/dt, total_time_steps)
        kernel_F = gaussian_kernel(mu_F, sigma_F/dt, total_time_steps)
        kernel_T = gaussian_kernel(mu_T, sigma_T/dt, total_time_steps)
        kernel_R = gaussian_kernel(mu_R, sigma_R/dt, total_time_steps)

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
        water_anomaly       = np.sqrt(((delayed_water - W_typ) / (W_typ + epsilon))**2 + epsilon)
        fertilizer_anomaly  = np.sqrt(((delayed_fertilizer - F_typ) / (F_typ + epsilon))**2 + epsilon)
        temperature_anomaly = np.sqrt(((delayed_temperature - T_typ) / (abs(T_typ) + epsilon))**2 + epsilon)
        radiation_anomaly   = np.sqrt(((delayed_radiation - R_typ) / (R_typ + epsilon))**2 + epsilon)

        # Calculate the smoothed cumulative divergences over time        
        cumulative_divergence_water       = ema_scan(water_anomaly,       beta, 0.0)
        cumulative_divergence_fertilizer  = ema_scan(fertilizer_anomaly,  beta, 0.0)
        cumulative_divergence_temperature = ema_scan(temperature_anomaly, beta, 0.0)
        cumulative_divergence_radiation   = ema_scan(radiation_anomaly,   beta, 0.0)

        # Then use the cumulative deltas to calculate the nutrient factors for each time step
        nuW_raw = np.exp(-alpha * cumulative_divergence_water)
        nuF_raw = np.exp(-alpha * cumulative_divergence_fertilizer)
        nuT_raw = np.exp(-alpha * cumulative_divergence_temperature)
        nuR_raw = np.exp(-alpha * cumulative_divergence_radiation)

        # Then smooth with EMA to get final nutrient factors
        nuW = ema_scan(nuW_raw, beta, 1.0)
        nuF = ema_scan(nuF_raw, beta, 1.0)
        nuT = ema_scan(nuT_raw, beta, 1.0)
        nuR = ema_scan(nuR_raw, beta, 1.0)

        # Initialize storage for instantaneous adjusted growth rates and carrying capacities
        ah_hat = np.zeros(total_time_steps)
        aA_hat = np.zeros(total_time_steps)
        aN_hat = np.zeros(total_time_steps)
        ac_hat = np.zeros(total_time_steps)
        aP_hat = np.zeros(total_time_steps)

        kh_hat = np.zeros(total_time_steps)
        kA_hat = np.zeros(total_time_steps)
        kN_hat = np.zeros(total_time_steps)
        kc_hat = np.zeros(total_time_steps)
        kP_hat = np.zeros(total_time_steps)

        # Run the season simulation for the given member
        for t in range(total_time_steps - 1):

            # Calculate the instantaneous adjusted growth rates and carrying capacities
            ah_hat[t] = np.clip(ah * (nuF[t] * nuT[t] * nuR[t])**(1/3), 0, 2 * ah)
            aA_hat[t] = np.clip(aA * (nuF[t] * nuT[t] * nuR[t])**(1/3), 0, 2 * aA)
            aN_hat[t] = np.clip(aN, 0, 2 * aN)
            ac_hat[t] = np.clip(ac * ( (1/nuT[t]) * (1/nuR[t]) )**(1/2), 0, 2 * ac)
            aP_hat[t] = np.clip(aP * (nuT[t] * nuR[t])**(1/2), 0, 2 * aP)

            kh_hat[t] = np.clip(kh * (nuF[t] * nuT[t] * nuR[t])**(1/3), h[t], 2 * kh)
            kA_hat[t] = np.clip(kA * (nuW[t] * nuF[t] * nuT[t] * nuR[t] * (kh_hat[t]/kh))**(1/5), A[t], 2 * kA)
            kN_hat[t] = np.clip(kN * (nuT[t] * nuR[t])**(1/2), N[t], 2 * kN)
            kc_hat[t] = np.clip(kc * (nuW[t] * (1/nuT[t]) * (1/nuR[t]))**(1/3), c[t], 2 * kc)
            kP_hat[t] = np.clip(kP * (nuW[t] * nuF[t] * nuT[t] * nuR[t] * (kh_hat[t]/kh) * (kA_hat[t]/kA) * (kc_hat[t]/kc))**(1/7), P[t], 2 * kP)

            # Logistic-style updates
            if closed_form:
                h[t+1] = logistic_step(h[t], ah_hat[t], kh_hat[t], dt)
                A[t+1] = logistic_step(A[t], aA_hat[t], kA_hat[t], dt)
                N[t+1] = logistic_step(N[t], aN_hat[t], kN_hat[t], dt)
                c[t+1] = logistic_step(c[t], ac_hat[t], kc_hat[t], dt)
                P[t+1] = logistic_step(P[t], aP_hat[t], kP_hat[t], dt)

            else:
                # Forward Euler integration
                h[t+1] = h[t] + dt * (ah_hat[t] * h[t] * (1 - h[t]/max(kh_hat[t], 1e-9)))
                A[t+1] = A[t] + dt * (aA_hat[t] * A[t] * (1 - A[t]/max(kA_hat[t], 1e-9)))
                N[t+1] = N[t] + dt * (aN_hat[t] * N[t] * (1 - N[t]/max(kN_hat[t], 1e-9)))
                c[t+1] = c[t] + dt * (ac_hat[t] * c[t] * (1 - c[t]/max(kc_hat[t], 1e-9)))
                P[t+1] = P[t] + dt * (aP_hat[t] * P[t] * (1 - P[t]/max(kP_hat[t], 1e-9)))

                # Enforce non-negativity explicitly
                h[t+1] = max(h[t+1], 0.0)
                A[t+1] = max(A[t+1], 0.0)
                N[t+1] = max(N[t+1], 0.0)
                c[t+1] = max(c[t+1], 0.0)
                P[t+1] = max(P[t+1], 0.0)

        # Combined objective (negative because GA minimizes)
        profit = self.ga_params.weight_fruit_biomass * P[-1] + self.ga_params.weight_height * h[-1] + self.ga_params.weight_leaf_area * A[-1]
        expenses = (self.ga_params.weight_irrigation * np.sum(irrigation)
                    + self.ga_params.weight_fertilizer * np.sum(fertilizer))
        revenue = profit - expenses
        cost = -revenue # GA minimizes cost, but we want to maximize revenue

        # Save all the data of interest to a csv for further analysis if verbose is True
        if verbose:

            df = pd.DataFrame({
                'h': h.flatten(),
                'A': A.flatten(),
                'N': N.flatten(),
                'c': c.flatten(),
                'P': P.flatten(),
                'delayed_water':       delayed_water.flatten(),
                'delayed_fertilizer':  delayed_fertilizer.flatten(),
                'delayed_temperature': delayed_temperature.flatten(),
                'delayed_radiation':   delayed_radiation.flatten(),
                'cumulative_water':       cumulative_water.flatten(),
                'cumulative_fertilizer':  cumulative_fertilizer.flatten(),
                'cumulative_temperature': cumulative_temperature.flatten(),
                'cumulative_radiation':   cumulative_radiation.flatten(),
                'cumulative_divergence_water':       cumulative_divergence_water.flatten(),
                'cumulative_divergence_fertilizer':  cumulative_divergence_fertilizer.flatten(),
                'cumulative_divergence_temperature': cumulative_divergence_temperature.flatten(),
                'cumulative_divergence_radiation':   cumulative_divergence_radiation.flatten(),
                'nuW': nuW.flatten(),
                'nuF': nuF.flatten(),
                'nuT': nuT.flatten(),
                'nuR': nuR.flatten(),
                'ah_hat': ah_hat.flatten(),
                'aA_hat': aA_hat.flatten(),
                'aN_hat': aN_hat.flatten(),
                'ac_hat': ac_hat.flatten(),
                'aP_hat': aP_hat.flatten(),
                'kh_hat': kh_hat.flatten(),
                'kA_hat': kA_hat.flatten(),
                'kN_hat': kN_hat.flatten(),
                'kc_hat': kc_hat.flatten(),
                'kP_hat': kP_hat.flatten(),
            })

            df.to_csv("output_get_cost.csv", index=False)

        return float(cost)
