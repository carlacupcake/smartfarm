# ga_member.py
import pandas as pd
import mpmath as mp
import numpy as np
from typing import Optional

from .ga_params import GeneticAlgorithmParams
from ..model.model_helpers import gaussian_kernel, get_mu_from_sigma, get_sim_inputs_from_hourly, logistic_step

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

        # ------------------------------------------------------------
        # Match MPC/single_time_step physics:
        # - FIR histories updated step-by-step (NOT full convolution)
        # - Histories initialized to typical steady-state values
        # - Anomaly computed on delayed signals (local)
        # - Divergence updated via EMA (no 1/k)
        # ------------------------------------------------------------

        # Pre-calculate the mu values that correspond to 95% absorption for each sigma ("sensitivity")
        mu_W = get_mu_from_sigma(sigma_W/dt)
        mu_F = get_mu_from_sigma(sigma_F/dt)
        mu_T = get_mu_from_sigma(sigma_T/dt)
        mu_R = get_mu_from_sigma(sigma_R/dt)

        # Build Gaussian FIR kernels (truncate to final horizon)
        kernel_W = gaussian_kernel(mu_W, sigma_W / dt, total_time_steps)
        kernel_F = gaussian_kernel(mu_F, sigma_F / dt, total_time_steps)
        kernel_T = gaussian_kernel(mu_T, sigma_T / dt, total_time_steps)
        kernel_R = gaussian_kernel(mu_R, sigma_R / dt, total_time_steps)

        '''
        # Convolve input disturbances with Gaussian kernels to model delayed absorption/metalysis
        kernel_W = gaussian_kernel(mu_W, sigma_W/dt, total_time_steps)
        kernel_F = gaussian_kernel(mu_F, sigma_F/dt, total_time_steps)
        kernel_T = gaussian_kernel(mu_T, sigma_T/dt, total_time_steps)
        kernel_R = gaussian_kernel(mu_R, sigma_R/dt, total_time_steps)
        '''

        # Initialize FIR histories to typical steady-state values (removes startup transient)
        water_history       = np.ones(total_time_steps, dtype=float) * W_typ
        fertilizer_history  = np.ones(total_time_steps, dtype=float) * F_typ
        temperature_history = np.ones(total_time_steps, dtype=float) * T_typ
        radiation_history   = np.ones(total_time_steps, dtype=float) * R_typ

        # Storage for delayed and cumulative signals (so verbose CSV stays useful)
        delayed_water       = np.zeros(total_time_steps, dtype=float)
        delayed_fertilizer  = np.zeros(total_time_steps, dtype=float)
        delayed_temperature = np.zeros(total_time_steps, dtype=float)
        delayed_radiation   = np.zeros(total_time_steps, dtype=float)

        cumulative_water       = np.zeros(total_time_steps, dtype=float)
        cumulative_fertilizer  = np.zeros(total_time_steps, dtype=float)
        cumulative_temperature = np.zeros(total_time_steps, dtype=float)
        cumulative_radiation   = np.zeros(total_time_steps, dtype=float)

        # Start cumulative values at 0
        cumulative_water_prev = 0.0
        cumulative_fertilizer_prev = 0.0
        cumulative_temperature_prev = 0.0
        cumulative_radiation_prev = 0.0

        # EMA divergence states
        cumulative_divergence_water_prev = 0.0
        cumulative_divergence_fertilizer_prev = 0.0
        cumulative_divergence_temperature_prev = 0.0
        cumulative_divergence_radiation_prev = 0.0

        # EMA beta
        beta = self.sensitivities.beta

        # Storage for anomalies/divergences and nutrient factors
        water_anomaly       = np.zeros(total_time_steps, dtype=float)
        fertilizer_anomaly  = np.zeros(total_time_steps, dtype=float)
        temperature_anomaly = np.zeros(total_time_steps, dtype=float)
        radiation_anomaly   = np.zeros(total_time_steps, dtype=float)

        cumulative_divergence_water       = np.zeros(total_time_steps, dtype=float)
        cumulative_divergence_fertilizer  = np.zeros(total_time_steps, dtype=float)
        cumulative_divergence_temperature = np.zeros(total_time_steps, dtype=float)
        cumulative_divergence_radiation   = np.zeros(total_time_steps, dtype=float)

        nuW = np.zeros(total_time_steps, dtype=float)
        nuF = np.zeros(total_time_steps, dtype=float)
        nuT = np.zeros(total_time_steps, dtype=float)
        nuR = np.zeros(total_time_steps, dtype=float)

        alpha = float(self.sensitivities.alpha)
        eps = 1e-9  # local numeric safety; keep separate from MPC eps if you want

        for t in range(total_time_steps):
            # Update FIR histories (shift left, append new sample)
            water_history       = np.roll(water_history,       -1)
            fertilizer_history  = np.roll(fertilizer_history,  -1)
            temperature_history = np.roll(temperature_history, -1)
            radiation_history   = np.roll(radiation_history,   -1)

            # Match single_time_step convention:
            # water input into FIR is irrigation + precipitation (already combined)
            water_history[-1]       = float(irrigation[t] + precipitation[t])
            fertilizer_history[-1]  = float(fertilizer[t])
            temperature_history[-1] = float(temperature[t])
            radiation_history[-1]   = float(radiation[t])

            # Compute delayed signals via FIR dot-product (like MPC)
            delayed_water[t]       = float(np.dot(kernel_W, water_history))
            delayed_fertilizer[t]  = float(np.dot(kernel_F, fertilizer_history))
            delayed_temperature[t] = float(np.dot(kernel_T, temperature_history))
            delayed_radiation[t]   = float(np.dot(kernel_R, radiation_history))

            # Update cumulative delayed values (for logging / interpretability)
            cumulative_water_prev       += delayed_water[t]
            cumulative_fertilizer_prev  += delayed_fertilizer[t]
            cumulative_temperature_prev += delayed_temperature[t]
            cumulative_radiation_prev   += delayed_radiation[t]

            cumulative_water[t]       = cumulative_water_prev
            cumulative_fertilizer[t]  = cumulative_fertilizer_prev
            cumulative_temperature[t] = cumulative_temperature_prev
            cumulative_radiation[t]   = cumulative_radiation_prev

            # Local anomaly on delayed signals (this is what you put into CFTOC)
            water_anomaly[t]       = np.sqrt(((delayed_water[t]       - W_typ) / (W_typ + eps))**2 + eps)
            fertilizer_anomaly[t]  = np.sqrt(((delayed_fertilizer[t]  - F_typ) / (F_typ + eps))**2 + eps)
            temperature_anomaly[t] = np.sqrt(((delayed_temperature[t] - T_typ) / (abs(T_typ) + eps))**2 + eps)
            radiation_anomaly[t]   = np.sqrt(((delayed_radiation[t]   - R_typ) / (R_typ + eps))**2 + eps)

            # EMA divergence update (same structure as CFTOC)
            cumulative_divergence_water_prev       = beta * cumulative_divergence_water_prev       + (1.0 - beta) * water_anomaly[t]
            cumulative_divergence_fertilizer_prev  = beta * cumulative_divergence_fertilizer_prev  + (1.0 - beta) * fertilizer_anomaly[t]
            cumulative_divergence_temperature_prev = beta * cumulative_divergence_temperature_prev + (1.0 - beta) * temperature_anomaly[t]
            cumulative_divergence_radiation_prev   = beta * cumulative_divergence_radiation_prev   + (1.0 - beta) * radiation_anomaly[t]

            cumulative_divergence_water[t]       = cumulative_divergence_water_prev
            cumulative_divergence_fertilizer[t]  = cumulative_divergence_fertilizer_prev
            cumulative_divergence_temperature[t] = cumulative_divergence_temperature_prev
            cumulative_divergence_radiation[t]   = cumulative_divergence_radiation_prev

            # Nutrient factors (match your current MPC choice; if you're using bounded version there, mirror it here)
            nuW[t] = np.exp(-alpha * cumulative_divergence_water[t])
            nuF[t] = np.exp(-alpha * cumulative_divergence_fertilizer[t])
            nuT[t] = np.exp(-alpha * cumulative_divergence_temperature[t])
            nuR[t] = np.exp(-alpha * cumulative_divergence_radiation[t])

        # Now compute adjusted growth rates / carrying capacities per step (same as before)
        ah_hat = np.clip(ah * (nuF * nuT * nuR)**(1/3), 0, 2 * ah)
        aA_hat = np.clip(aA * (nuF * nuT * nuR)**(1/3), 0, 2 * aA)
        aN_hat = np.clip(aN, 0, 2 * aN) * np.ones(total_time_steps)
        ac_hat = np.clip(ac * ((1/np.maximum(nuT, eps)) * (1/np.maximum(nuR, eps)))**(1/2), 0, 2 * ac)
        aP_hat = np.clip(aP * (nuT * nuR)**(1/2), 0, 2 * aP)

        kh_hat = np.clip(kh * (nuF * nuT * nuR)**(1/3), 0, 2 * kh)
        kA_hat = np.clip(kA * (nuW * nuF * nuT * nuR * (kh_hat/kh))**(1/5), 0, 2 * kA)
        kN_hat = np.clip(kN * (nuT * nuR)**(1/2), 0, 2 * kN) * np.ones(total_time_steps)
        kc_hat = np.clip(kc * (nuW * (1/np.maximum(nuT, eps)) * (1/np.maximum(nuR, eps)))**(1/3), 0, 2 * kc)
        kP_hat = np.clip(kP * (nuW * nuF * nuT * nuR * (kh_hat/kh) * (kA_hat/kA) * (kc_hat/kc))**(1/7), 0, 2 * kP)

        # Run the season simulation for the given member
        for t in range(total_time_steps - 1):

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
        profit = self.ga_params.weight_fruit_biomass * P[-1]
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
                'water_anomaly':       water_anomaly.flatten(),
                'fertilizer_anomaly':  fertilizer_anomaly.flatten(),
                'temperature_anomaly': temperature_anomaly.flatten(),
                'radiation_anomaly':   radiation_anomaly.flatten(),
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
