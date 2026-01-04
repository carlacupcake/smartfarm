# ga_member.py
import pandas as pd
import mpmath as mp
import numpy as np
from typing import Optional

from .ga_params import GeneticAlgorithmParams
from ..model.model_helpers import *

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

        # Precompute FIR kernels from existing sensitivities
        dt = self.model_params.dt

        sigma_W = self.sensitivities.sigma_W
        sigma_F = self.sensitivities.sigma_F
        sigma_T = self.sensitivities.sigma_T
        sigma_R = self.sensitivities.sigma_R

        mu_W = get_mu_from_sigma(sigma_W / dt)
        mu_F = get_mu_from_sigma(sigma_F / dt)
        mu_T = get_mu_from_sigma(sigma_T / dt)
        mu_R = get_mu_from_sigma(sigma_R / dt)

        self.kernel_W = gaussian_kernel(mu_W, sigma_W / dt, self.model_params.total_time_steps)
        self.kernel_F = gaussian_kernel(mu_F, sigma_F / dt, self.model_params.total_time_steps)
        self.kernel_T = gaussian_kernel(mu_T, sigma_T / dt, self.model_params.total_time_steps)
        self.kernel_R = gaussian_kernel(mu_R, sigma_R / dt, self.model_params.total_time_steps)

        self.fir_horizon_W = compute_fir_horizon(self.kernel_W)
        self.fir_horizon_F = compute_fir_horizon(self.kernel_F)
        self.fir_horizon_T = compute_fir_horizon(self.kernel_T)
        self.fir_horizon_R = compute_fir_horizon(self.kernel_R)


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
        verbose          = self.model_params.verbose

        # Unpack sensitivities
        alpha                = self.sensitivities.alpha
        beta_divergence      = self.sensitivities.beta_divergence
        beta_nutrient_factor = self.sensitivities.beta_nutrient_factor
        epsilon              = self.sensitivities.epsilon

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

        # Set history of disturbances before time zero
        water_history       = np.ones(self.fir_horizon_W, dtype=float) * W_typ
        fertilizer_history  = np.ones(self.fir_horizon_F, dtype=float) * F_typ
        temperature_history = np.ones(self.fir_horizon_T, dtype=float) * T_typ
        radiation_history   = np.ones(self.fir_horizon_R, dtype=float) * R_typ

        # Initialize storage for delayed disturbances
        delayed_water       = np.zeros(total_time_steps)
        delayed_fertilizer  = np.zeros(total_time_steps)
        delayed_temperature = np.zeros(total_time_steps)
        delayed_radiation   = np.zeros(total_time_steps)

        # Initialize storage for cumulative disturbances
        cumulative_water       = np.zeros(total_time_steps)
        cumulative_fertilizer  = np.zeros(total_time_steps)
        cumulative_temperature = np.zeros(total_time_steps)
        cumulative_radiation   = np.zeros(total_time_steps)

        # Initialize storage for anomalies
        water_anomaly       = np.zeros(total_time_steps)
        fertilizer_anomaly  = np.zeros(total_time_steps)
        temperature_anomaly = np.zeros(total_time_steps)
        radiation_anomaly   = np.zeros(total_time_steps)

        # Initialize storage for cumulative divergences
        cumulative_divergence_water       = np.zeros(total_time_steps)
        cumulative_divergence_fertilizer  = np.zeros(total_time_steps)
        cumulative_divergence_temperature = np.zeros(total_time_steps)
        cumulative_divergence_radiation   = np.zeros(total_time_steps)

        # Initialize storage for nutrient factors
        nuW = np.zeros(total_time_steps)
        nuF = np.zeros(total_time_steps)
        nuT = np.zeros(total_time_steps)
        nuR = np.zeros(total_time_steps)

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
        for t in range(1, total_time_steps - 1):

            # Unpack control inputs
            W = irrigation[t]
            F = fertilizer[t]

            # Unpack disturbances
            S = precipitation[t]
            T = temperature[t]
            R = radiation[t]

            # Unpack FIR kernels (truncate to FIR horizon)
            fir_horizon_W = self.fir_horizon_W
            fir_horizon_F = self.fir_horizon_F
            fir_horizon_T = self.fir_horizon_T
            fir_horizon_R = self.fir_horizon_R

            kernel_W = self.kernel_W[:fir_horizon_W]
            kernel_F = self.kernel_F[:fir_horizon_F]
            kernel_T = self.kernel_T[:fir_horizon_T]
            kernel_R = self.kernel_R[:fir_horizon_R]

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
            delayed_water[t]       = np.dot(kernel_W, water_history)
            delayed_fertilizer[t]  = np.dot(kernel_F, fertilizer_history)
            delayed_temperature[t] = np.dot(kernel_T, temperature_history)
            delayed_radiation[t]   = np.dot(kernel_R, radiation_history)
            
            # Update cumulative delayed values
            cumulative_water[t+1]       = cumulative_water[t]       + delayed_water[t]
            cumulative_fertilizer[t+1]  = cumulative_fertilizer[t]  + delayed_fertilizer[t]
            cumulative_temperature[t+1] = cumulative_temperature[t] + delayed_temperature[t]
            cumulative_radiation[t+1]   = cumulative_radiation[t]   + delayed_radiation[t]

            # Calculate the differences between the expected and actual cumulative values
            water_anomaly[t]       = max(np.abs(W_typ * (t-1) - cumulative_water[t])       / (W_typ * t + epsilon), epsilon)
            fertilizer_anomaly[t]  = max(np.abs(F_typ * (t-1) - cumulative_fertilizer[t])  / (F_typ * t + epsilon), epsilon)
            temperature_anomaly[t] = max(np.abs(T_typ * (t-1) - cumulative_temperature[t]) / (T_typ * t + epsilon), epsilon)
            radiation_anomaly[t]   = max(np.abs(R_typ * (t-1) - cumulative_radiation[t])   / (R_typ * t + epsilon), epsilon)

            # Recursive cumulative divergence update
            cumulative_divergence_water[t]       = beta_divergence * cumulative_divergence_water[t-1]       + (1.0 - beta_divergence) * water_anomaly[t]
            cumulative_divergence_fertilizer[t]  = beta_divergence * cumulative_divergence_fertilizer[t-1]  + (1.0 - beta_divergence) * fertilizer_anomaly[t]
            cumulative_divergence_temperature[t] = beta_divergence * cumulative_divergence_temperature[t-1] + (1.0 - beta_divergence) * temperature_anomaly[t]
            cumulative_divergence_radiation[t]   = beta_divergence * cumulative_divergence_radiation[t-1]   + (1.0 - beta_divergence) * radiation_anomaly[t]

            # Raw nutrient factors
            nuW_raw = np.exp(-alpha * cumulative_divergence_water[t])
            nuF_raw = np.exp(-alpha * cumulative_divergence_fertilizer[t])
            nuT_raw = np.exp(-alpha * cumulative_divergence_temperature[t])
            nuR_raw = np.exp(-alpha * cumulative_divergence_radiation[t])

            # Final, smoothed nutrient factors
            nuW[t] = (1.0 - beta_nutrient_factor) * nuW[t-1] + beta_nutrient_factor * nuW_raw
            nuF[t] = (1.0 - beta_nutrient_factor) * nuF[t-1] + beta_nutrient_factor * nuF_raw
            nuT[t] = (1.0 - beta_nutrient_factor) * nuT[t-1] + beta_nutrient_factor * nuT_raw
            nuR[t] = (1.0 - beta_nutrient_factor) * nuR[t-1] + beta_nutrient_factor * nuR_raw

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
            h[t+1] = logistic_step(h[t], ah_hat[t], kh_hat[t], dt)
            A[t+1] = logistic_step(A[t], aA_hat[t], kA_hat[t], dt)
            N[t+1] = logistic_step(N[t], aN_hat[t], kN_hat[t], dt)
            c[t+1] = logistic_step(c[t], ac_hat[t], kc_hat[t], dt)
            P[t+1] = logistic_step(P[t], aP_hat[t], kP_hat[t], dt)

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
                'raw_water':       irrigation.flatten() + precipitation.flatten(),
                'raw_fertilizer':  fertilizer.flatten(),
                'raw_temperature': temperature.flatten(),
                'raw_radiation':   radiation.flatten(),
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
