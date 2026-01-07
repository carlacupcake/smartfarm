# mpc.py
import logging
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from typing import Optional, Dict, Tuple

from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds

from ..model.model_helpers import *
from ..model.model_carrying_capacities import ModelCarryingCapacities
from ..model.model_disturbances import ModelDisturbances
from ..model.model_growth_rates import ModelGrowthRates
from ..model.model_initial_conditions import ModelInitialConditions
from ..model.model_params import ModelParams
from ..model.model_typical_disturbances import ModelTypicalDisturbances
from ..model.model_sensitivities import ModelSensitivities

from .mpc_params import MPCParams
from .mpc_bounds import ControlInputBounds

class MPC:
    """
    Nonlinear MPC controller for the plant growth model.

    This class solves a constrained finite-time optimal control (CFTOC)
    problem over a horizon of length N at each step.

    The control inputs are per-time-step irrigation and fertilizer amounts.
    Disturbances are precipitation, temperature, and solar radiation.
    The state variables are plant height, leaf area, number of leaves, flower spikelet count, and fruit biomass.
    """

    def __init__(
        self,
        carrying_capacities:  ModelCarryingCapacities,
        disturbances:         ModelDisturbances,
        growth_rates:         ModelGrowthRates,
        initial_conditions:   ModelInitialConditions,
        model_params:         ModelParams,
        typical_disturbances: ModelTypicalDisturbances,
        sensitivities:        ModelSensitivities,
        mpc_params:           MPCParams,
        bounds:               ControlInputBounds
    ):

        self.carrying_capacities  = carrying_capacities
        self.disturbances         = disturbances
        self.growth_rates         = growth_rates
        self.initial_conditions   = initial_conditions
        self.model_params         = model_params
        self.typical_disturbances = typical_disturbances
        self.sensitivities        = sensitivities
        self.mpc_params           = mpc_params
        self.bounds               = bounds

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

        dt = int(round(24.0 / self.model_params.dt))

        self.kernel_W_daily = self.kernel_W[::dt] # TODO used?
        self.kernel_F_daily = self.kernel_F[::dt]
        self.kernel_T_daily = self.kernel_T[::dt]
        self.kernel_R_daily = self.kernel_R[::dt]

        self.fir_horizon_W_daily = compute_fir_horizon(self.kernel_W_daily)
        self.fir_horizon_F_daily = compute_fir_horizon(self.kernel_F_daily)
        self.fir_horizon_T_daily = compute_fir_horizon(self.kernel_T_daily)
        self.fir_horizon_R_daily = compute_fir_horizon(self.kernel_R_daily)
    

    def run(self) -> Dict[str, np.ndarray]:
        """
        Run receding-horizon MPC on a *daily* control grid while simulating
        the plant on an *hourly* grid with the existing single_time_step().
        """

        dt                = 1.0 # hours
        hours_per_day     = int(24)
        steps_per_horizon = int(self.mpc_params.daily_horizon * hours_per_day)
        total_time_steps  = int(self.model_params.total_time_steps)
        total_days        = int(total_time_steps // hours_per_day)
        horizon           = int(self.mpc_params.daily_horizon)

        # Initial state and cumulative state
        h0 = self.initial_conditions.h0
        A0 = self.initial_conditions.A0
        N0 = self.initial_conditions.N0
        c0 = self.initial_conditions.c0
        P0 = self.initial_conditions.P0

        x = np.array([h0, A0, N0, c0, P0], dtype=float)
        C = np.zeros(4, dtype=float)  # cumulative water, fertilizer, temperature, radiation
        extra_state = None

        # Storage for full-hourly trajectories
        h = np.full(total_time_steps, h0)
        A = np.full(total_time_steps, A0)
        N = np.full(total_time_steps, N0)
        c = np.full(total_time_steps, c0)
        P = np.full(total_time_steps, P0)

        irrigation = np.zeros(total_time_steps)
        fertilizer = np.zeros(total_time_steps)

        # Unpack hourly disturbances (already defined)
        hourly_precipitation = self.disturbances.precipitation
        hourly_temperature   = self.disturbances.temperature
        hourly_radiation     = self.disturbances.radiation

        # Main loop over days
        step_idx   = 0    # hourly index
        u_prev = None # at the start, there are no previous control inputs

        for d in range(total_days):
            print(f"[Daily MPC] Day {d+1}/{total_days}, hour index {step_idx}")

            # Build a DAILY forecast for the next `horizon`
            # Aggregate hourly disturbances into per-day means or sums.
            day_start     = step_idx
            day_end       = min(step_idx + steps_per_horizon, total_time_steps)

            # Slice hourly arrays
            horizon_precipitation = hourly_precipitation[day_start:day_end].tolist()
            horizon_temperature   = hourly_temperature[day_start:day_end].tolist()
            horizon_radiation     = hourly_radiation[day_start:day_end].tolist()

            # Pad the end of the hourly arrays in order to have len = total_time_steps
            while len(horizon_precipitation) < total_time_steps:
                horizon_precipitation.append(0.0)
                horizon_temperature.append(horizon_temperature[-1])
                horizon_radiation.append(horizon_radiation[-1])

            # Aggregate to per-day (simple mean here; you might prefer sums for water)
            daily_precipitation = []
            daily_temperature   = []
            daily_radiation    = []
            for i in range(horizon):
                s = i * hours_per_day
                e = s + hours_per_day
                daily_precipitation.append(np.sum(horizon_precipitation[s:e])) # * 24 ?? TODO
                daily_temperature.append(np.mean(horizon_temperature[s:e]))
                daily_radiation.append(np.mean(horizon_radiation[s:e]))
                
            # If near the end and we had fewer than horizon, pad with last value
            while len(daily_precipitation) < horizon:
                daily_precipitation.append(daily_precipitation[-1])
                daily_temperature.append(daily_temperature[-1])
                daily_radiation.append(daily_radiation[-1])

            forecast = {
                "precipitation": np.array(daily_precipitation[:horizon]),
                "temperature":   np.array(daily_temperature[:horizon]),
                "radiation":     np.array(daily_radiation[:horizon]),
            }

            # Solve DAILY CFTOC to get plan for next horizon
            irrigation_plan, fertilizer_plan = self.solve_cftoc(
                x0=x,
                C0=C,
                forecast=forecast,
                u_prev=u_prev,
            )

            # Apply ONLY the first day's controls as constants over that day
            irrigation_control_action = float(irrigation_plan[0])
            fertilizer_control_action = float(fertilizer_plan[0])
            u_prev = np.array([irrigation_control_action, fertilizer_control_action])

            # Simulate that day hour by hour with single_time_step()
            for _ in range(hours_per_day):
                if step_idx >= total_time_steps - 1:
                    break

                d_k = {
                    "precipitation": hourly_precipitation[step_idx],
                    "temperature":   hourly_temperature[step_idx],
                    "radiation":     hourly_radiation[step_idx],
                }
                u_k = u_prev / hours_per_day  # distribute daily amount over hours

                x, C, extra_state = self.single_time_step(
                    x=x,
                    u=u_k,
                    d=d_k,
                    C=C,
                    extra_state=extra_state,
                )

                irrigation[step_idx] = u_k[0]
                fertilizer[step_idx] = u_k[1]

                h[step_idx+1] = x[0]
                A[step_idx+1] = x[1]
                N[step_idx+1] = x[2]
                c[step_idx+1] = x[3]
                P[step_idx+1] = x[4]

                step_idx += 1

        # Trim to actual simulated steps
        n_steps = step_idx
        result = {
            "h":          h[:n_steps+1],
            "A":          A[:n_steps+1],
            "N":          N[:n_steps+1],
            "c":          c[:n_steps+1],
            "P":          P[:n_steps+1],
            "irrigation": irrigation[:n_steps],
            "fertilizer": fertilizer[:n_steps],
        }
        if extra_state is not None and "log" in extra_state:
            result["logs"] = {k: np.array(v) for k, v in extra_state["log"].items()}
        else:
            result["logs"] = {}

        return result
    

    def solve_cftoc(
        self,
        x0: np.ndarray,
        C0: np.ndarray,
        forecast: Dict[str, np.ndarray],
        u_prev: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Coarse CFTOC over a DAILY horizon.

        - horizon: e.g. 7
        - dt: 24 hours (or treat as 1 "day-unit" and rescale a's)
        """
        # Unpack time parameters and other model constants
        horizon = self.mpc_params.daily_horizon
        dt      = 24.0

        # Unpack sensitivities
        alpha                = self.sensitivities.alpha
        beta_divergence      = self.sensitivities.beta_divergence
        beta_nutrient_factor = self.sensitivities.beta_nutrient_factor
        epsilon              = self.sensitivities.epsilon

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

        # Unpack typical disturbances
        W_typ = self.typical_disturbances.typical_water        * dt # to get daily value
        F_typ = self.typical_disturbances.typical_fertilizer   * dt # to get daily value
        T_typ = self.typical_disturbances.typical_temperature
        R_typ = self.typical_disturbances.typical_radiation

        # Build model       
        model    = pyo.ConcreteModel()
        model.xk = pyo.RangeSet(0, horizon)      # states 0...N
        model.uk = pyo.RangeSet(0, horizon - 1)  # controls 0...N-1

        # Daily disturbances as Params
        model.precipitation = pyo.Param(
            model.uk,
            initialize=lambda m, d: float(forecast["precipitation"][d]),
            mutable=False,
        )
        model.temperature = pyo.Param(
            model.uk,
            initialize=lambda m, d: float(forecast["temperature"][d]),
            mutable=False,
        )
        model.radiation = pyo.Param(
            model.uk,
            initialize=lambda m, d: float(forecast["radiation"][d]),
            mutable=False,
        )

        # Time indices for kernels
        model.wk = pyo.RangeSet(0, self.fir_horizon_W_daily - 1)
        model.fk = pyo.RangeSet(0, self.fir_horizon_F_daily - 1)
        model.tk = pyo.RangeSet(0, self.fir_horizon_T_daily - 1)
        model.rk = pyo.RangeSet(0, self.fir_horizon_R_daily - 1)

        # Kernels as parameters
        model.kernel_W = pyo.Param(model.wk, initialize=lambda model, j: float(self.kernel_W_daily[j]), mutable=False)
        model.kernel_F = pyo.Param(model.fk, initialize=lambda model, j: float(self.kernel_F_daily[j]), mutable=False)
        model.kernel_T = pyo.Param(model.tk, initialize=lambda model, j: float(self.kernel_T_daily[j]), mutable=False)
        model.kernel_R = pyo.Param(model.rk, initialize=lambda model, j: float(self.kernel_R_daily[j]), mutable=False)

        # States
        model.h = pyo.Var(model.xk, bounds=(self.initial_conditions.h0, 10*self.carrying_capacities.kh))
        model.A = pyo.Var(model.xk, bounds=(self.initial_conditions.A0, 10*self.carrying_capacities.kA))
        model.N = pyo.Var(model.xk, bounds=(self.initial_conditions.N0, 10*self.carrying_capacities.kN))
        model.c = pyo.Var(model.xk, bounds=(self.initial_conditions.c0, 10*self.carrying_capacities.kc))
        model.P = pyo.Var(model.xk, bounds=(self.initial_conditions.P0, 10*self.carrying_capacities.kP))

        # Augmented states: cumulative delayed signals
        model.cumulative_water       = pyo.Var(model.xk)
        model.cumulative_fertilizer  = pyo.Var(model.xk)
        model.cumulative_temperature = pyo.Var(model.xk)
        model.cumulative_radiation   = pyo.Var(model.xk)

        # Control inputs
        model.uW = pyo.Var(model.uk, bounds=self.bounds.irrigation_bounds)   # irrigation
        model.uF = pyo.Var(model.uk, bounds=self.bounds.fertilizer_bounds)   # fertilizer

        # Cumulative-average deviations (EMA-smoothed; nonnegative)
        model.cumulative_divergence_water       = pyo.Var(model.uk, bounds=(0.0, None))
        model.cumulative_divergence_fertilizer  = pyo.Var(model.uk, bounds=(0.0, None))
        model.cumulative_divergence_temperature = pyo.Var(model.uk, bounds=(0.0, None))
        model.cumulative_divergence_radiation   = pyo.Var(model.uk, bounds=(0.0, None))

        # Initial conditions for state variables
        x0 = np.asarray(x0).flatten()
        model.h[0].fix(float(x0[0]))
        model.A[0].fix(float(x0[1]))
        model.N[0].fix(float(x0[2]))
        model.c[0].fix(float(x0[3]))
        model.P[0].fix(float(x0[4]))

        # Initial conditions for cumulative states
        C0 = np.asarray(C0).flatten()
        model.cumulative_water[0].fix(float(C0[0]))
        model.cumulative_fertilizer[0].fix(float(C0[1]))
        model.cumulative_temperature[0].fix(float(C0[2]))
        model.cumulative_radiation[0].fix(float(C0[3]))

        # Forward initialize state variable values across horizon based on closed-form solutions
        for k in range(horizon):
            
            # Initialize h values
            model.h[k+1].value = float(pyo.value(
                kh / (1.0 + ((kh - model.h[k]) / (model.h[k] + epsilon)) * pyo.exp(-ah * dt))
            ))

            # Initialize A values
            model.A[k+1].value = float(pyo.value(
                kA / (1.0 + ((kA - model.A[k]) / (model.A[k] + epsilon)) * pyo.exp(-aA * dt))
            ))

            # Initialize N values
            model.N[k+1].value = float(pyo.value(
                kN / (1.0 + ((kN - model.N[k])/(model.N[k] + epsilon)) * pyo.exp(-aN * dt))
            ))

            # Initialize c values
            model.c[k+1].value = float(pyo.value(
                kc / (1.0 + ((kc - model.c[k]) / (model.c[k] + epsilon)) * pyo.exp(-ac * dt))
            ))

            # Initialize P values
            model.P[k+1].value = float(pyo.value(
                kP / (1.0 + ((kP - model.P[k]) / (model.P[k] + epsilon)) * pyo.exp(-aP * dt))
            ))

        # Forward initialize cumulative values across horizon
        for k in model.xk:
            model.cumulative_water[k].value       = float(C0[0])
            model.cumulative_fertilizer[k].value  = float(C0[1])
            model.cumulative_temperature[k].value = float(C0[2])
            model.cumulative_radiation[k].value   = float(C0[3])

        # Initial guesses for controls
        irrigation_lower_bound, irrigation_upper_bound = self.bounds.irrigation_bounds
        fertilizer_lower_bound, fertilizer_upper_bound = self.bounds.fertilizer_bounds
        irrigation_amount_guess = self.bounds.irrigation_amount_guess * dt
        fertilizer_amount_guess = self.bounds.fertilizer_amount_guess * dt

        if u_prev is not None:
            u_prev = np.asarray(u_prev).flatten()
            irrigation_amount = float(np.clip(u_prev[0], irrigation_lower_bound, irrigation_upper_bound))
            fertilizer_amount = float(np.clip(u_prev[1], fertilizer_lower_bound, fertilizer_upper_bound))
            for k in model.uk:
                model.uW[k].value = irrigation_amount
                model.uF[k].value = fertilizer_amount
        else:
            for k in model.uk:
                model.uW[k].value = irrigation_amount_guess
                model.uF[k].value = fertilizer_amount_guess

        # -------------------------------------------------------------------------
        # Pad FIR sums with typical values for missing pre-history terms
        #
        # At early k, k-j < 0 terms are missing. Instead of dropping them (which causes
        # startup transients), we fill them with "typical" steady-state values.
        #
        # Interpretation (daily):
        # - Water: (uW + precipitation) has typical total W_typ per day
        # - Fertilizer: uF has typical total F_typ per day
        # - Temperature: T has typical T_typ (degC)
        # - Radiation: R has typical R_typ (W/m^2)
        # -------------------------------------------------------------------------

        # FIR convolution expressions for delayed signals (with padding)
        def padded_delayed_water_rule(m, k):
            return sum(
                m.kernel_W[j] * (
                    (m.uW[k - j] + m.precipitation[k - j]) if (k - j) >= 0 else W_typ
                )
                for j in m.wk
            )
        model.delayed_water = pyo.Expression(model.uk, rule=padded_delayed_water_rule)

        def padded_delayed_fertilizer_rule(m, k):
            return sum(
                m.kernel_F[j] * (m.uF[k - j] if (k - j) >= 0 else F_typ)
                for j in m.fk
            )
        model.delayed_fertilizer = pyo.Expression(model.uk, rule=padded_delayed_fertilizer_rule)

        def padded_delayed_temperature_rule(m, k):
            return sum(
                m.kernel_T[j] * (m.temperature[k - j] if (k - j) >= 0 else T_typ)
                for j in m.tk
            )
        model.delayed_temperature = pyo.Expression(model.uk, rule=padded_delayed_temperature_rule)

        def padded_delayed_radiation_rule(m, k):
            return sum(
                m.kernel_R[j] * (m.radiation[k - j] if (k - j) >= 0 else R_typ)
                for j in m.rk
            )
        model.delayed_radiation = pyo.Expression(model.uk, rule=padded_delayed_radiation_rule)

        # Set next cumulative values to satisfy the update equations
        for k in model.uk:
            model.cumulative_water[k + 1].value       = model.cumulative_water[k].value       + float(pyo.value(model.delayed_water[k]))
            model.cumulative_fertilizer[k + 1].value  = model.cumulative_fertilizer[k].value  + float(pyo.value(model.delayed_fertilizer[k]))
            model.cumulative_temperature[k + 1].value = model.cumulative_temperature[k].value + float(pyo.value(model.delayed_temperature[k]))
            model.cumulative_radiation[k + 1].value   = model.cumulative_radiation[k].value   + float(pyo.value(model.delayed_radiation[k]))

        # Cumulative updates
        def cumulative_water_update_rule(model, k):
            if k == horizon:
                return pyo.Constraint.Skip
            return model.cumulative_water[k + 1] == model.cumulative_water[k] + model.delayed_water[k]
        model.cumulative_water_update = pyo.Constraint(model.uk, rule=cumulative_water_update_rule)

        def cumulative_fertilizer_update_rule(model, k):
            if k == horizon:
                return pyo.Constraint.Skip
            return model.cumulative_fertilizer[k + 1] == model.cumulative_fertilizer[k] + model.delayed_fertilizer[k]
        model.cumulative_fertilizer_update = pyo.Constraint(model.uk, rule=cumulative_fertilizer_update_rule)

        def cumulative_temperature_update_rule(model, k):
            if k == horizon:
                return pyo.Constraint.Skip
            return model.cumulative_temperature[k + 1] == model.cumulative_temperature[k] + model.delayed_temperature[k]
        model.cumulative_temperature_update = pyo.Constraint(model.uk, rule=cumulative_temperature_update_rule)

        def cumulative_radiation_update_rule(model, k):
            if k == horizon:
                return pyo.Constraint.Skip
            return model.cumulative_radiation[k + 1] == model.cumulative_radiation[k] + model.delayed_radiation[k]
        model.cumulative_radiation_update = pyo.Constraint(model.uk, rule=cumulative_radiation_update_rule)

        # Anomaly expressions for tracking divergence from typical
        def water_anomaly_rule(model, k):
            anomaly = model.delayed_water[k] - W_typ
            denom = W_typ + epsilon
            return pyo.sqrt((anomaly / denom) ** 2 + epsilon)
        model.water_anomaly = pyo.Expression(model.uk, rule=water_anomaly_rule)

        def fertilizer_anomaly_rule(model, k):
            anomaly = model.delayed_fertilizer[k] - F_typ
            denom = F_typ + epsilon
            return pyo.sqrt((anomaly / denom) ** 2 + epsilon)
        model.fertilizer_anomaly = pyo.Expression(model.uk, rule=fertilizer_anomaly_rule)

        def temperature_anomaly_rule(model, k):
            anomaly = model.delayed_temperature[k] - T_typ
            denom = abs(T_typ) + epsilon
            return pyo.sqrt((anomaly / denom) ** 2 + epsilon)
        model.temperature_anomaly = pyo.Expression(model.uk, rule=temperature_anomaly_rule)

        def radiation_anomaly_rule(model, k):
            anomaly = model.delayed_radiation[k] - R_typ
            denom = R_typ + epsilon
            return pyo.sqrt((anomaly / denom) ** 2 + epsilon)
        model.radiation_anomaly = pyo.Expression(model.uk, rule=radiation_anomaly_rule)

        # Forward initialize divergence Vars to satisfy their recursion constraints
        def smooth_with_ema_pyomo(divergence, anomaly):
            divergence[0].value = 0.0
            for k in range(1, horizon):
                divergence[k].value = beta_divergence * float(divergence[k - 1].value) + (1.0 - beta_divergence) * float(pyo.value(anomaly[k]))

        smooth_with_ema_pyomo(model.cumulative_divergence_water,       model.water_anomaly)
        smooth_with_ema_pyomo(model.cumulative_divergence_fertilizer,  model.fertilizer_anomaly)
        smooth_with_ema_pyomo(model.cumulative_divergence_temperature, model.temperature_anomaly)
        smooth_with_ema_pyomo(model.cumulative_divergence_radiation,   model.radiation_anomaly)

        # EMA smoothing for forgetting divergences from typical
        def cumulative_divergence_water_rule(model, k):
            if k == 0:
                # NEW: start divergences at typical steady-state (0.0)
                return model.cumulative_divergence_water[0] == 0.0
            return model.cumulative_divergence_water[k] == (
                beta_divergence * model.cumulative_divergence_water[k - 1]
                + (1.0 - beta_divergence) * model.water_anomaly[k]
            )
        model.cumulative_divergence_water_constraint = pyo.Constraint(model.uk, rule=cumulative_divergence_water_rule)

        def cumulative_divergence_fertilizer_rule(model, k):
            if k == 0:
                return model.cumulative_divergence_fertilizer[0] == 0.0
            return model.cumulative_divergence_fertilizer[k] == (
                beta_divergence * model.cumulative_divergence_fertilizer[k - 1]
                + (1.0 - beta_divergence) * model.fertilizer_anomaly[k]
            )
        model.cumulative_divergence_fertilizer_constraint = pyo.Constraint(model.uk, rule=cumulative_divergence_fertilizer_rule)

        def cumulative_divergence_temperature_rule(model, k):
            if k == 0:
                return model.cumulative_divergence_temperature[0] == 0.0
            return model.cumulative_divergence_temperature[k] == (
                beta_divergence * model.cumulative_divergence_temperature[k - 1]
                + (1.0 - beta_divergence) * model.temperature_anomaly[k]
            )
        model.cumulative_divergence_temperature_constraint = pyo.Constraint(model.uk, rule=cumulative_divergence_temperature_rule)

        def cumulative_divergence_radiation_rule(model, k):
            if k == 0:
                return model.cumulative_divergence_radiation[0] == 0.0
            return model.cumulative_divergence_radiation[k] == (
                beta_divergence * model.cumulative_divergence_radiation[k - 1]
                + (1.0 - beta_divergence) * model.radiation_anomaly[k]
            )
        model.cumulative_divergence_radiation_constraint = pyo.Constraint(model.uk, rule=cumulative_divergence_radiation_rule)

        # Helper to enforce k_hat >= state without non-differentiable max()
        def smoothmax_pyomo(a, b, delta=1e-8):
            return 0.5 * (a + b + pyo.sqrt((a - b) * (a - b) + delta))

        # Raw nutrient factor Expressions
        def nuW_raw_rule(model, k):
            return pyo.exp(-alpha * model.cumulative_divergence_water[k])
        model.nuW_raw = pyo.Expression(model.uk, rule=nuW_raw_rule)

        def nuF_raw_rule(model, k):
            return pyo.exp(-alpha * model.cumulative_divergence_fertilizer[k])
        model.nuF_raw = pyo.Expression(model.uk, rule=nuF_raw_rule)

        def nuT_raw_rule(model, k):
            return pyo.exp(-alpha * model.cumulative_divergence_temperature[k])
        model.nuT_raw = pyo.Expression(model.uk, rule=nuT_raw_rule)

        def nuR_raw_rule(model, k):
            return pyo.exp(-alpha * model.cumulative_divergence_radiation[k])
        model.nuR_raw = pyo.Expression(model.uk, rule=nuR_raw_rule)

        # Smoothed nutrient factors
        # Vars are easiest because EMA is recursive in time.
        model.nuW = pyo.Var(model.uk, bounds=(0.0, 1.0))
        model.nuF = pyo.Var(model.uk, bounds=(0.0, 1.0))
        model.nuT = pyo.Var(model.uk, bounds=(0.0, 1.0))
        model.nuR = pyo.Var(model.uk, bounds=(0.0, 1.0))

        # Anchor EMA at the first control index
        uk0 = min(model.uk)

        def nuW_ema_rule(model, k):
            if k == uk0:
                return model.nuW[k] == model.nuW_raw[k]
            return model.nuW[k] == beta_nutrient_factor * model.nuW[k-1] +  (1.0 - beta_nutrient_factor) * model.nuW_raw[k]
        model.nuW_ema = pyo.Constraint(model.uk, rule=nuW_ema_rule)

        def nuF_ema_rule(model, k):
            if k == uk0:
                return model.nuF[k] == model.nuF_raw[k]
            return model.nuF[k] == beta_nutrient_factor * model.nuF[k-1] +  (1.0 - beta_nutrient_factor) * model.nuF_raw[k]
        model.nuF_ema = pyo.Constraint(model.uk, rule=nuF_ema_rule)

        def nuT_ema_rule(model, k):
            if k == uk0:
                return model.nuT[k] == model.nuT_raw[k]
            return model.nuT[k] == beta_nutrient_factor * model.nuT[k-1] +  (1.0 - beta_nutrient_factor) * model.nuT_raw[k]
        model.nuT_ema = pyo.Constraint(model.uk, rule=nuT_ema_rule)

        def nuR_ema_rule(model, k):
            if k == uk0:
                return model.nuR[k] == model.nuR_raw[k]
            return model.nuR[k] == beta_nutrient_factor * model.nuR[k-1] +  (1.0 - beta_nutrient_factor) * model.nuR_raw[k]
        model.nuR_ema = pyo.Constraint(model.uk, rule=nuR_ema_rule)

        # State variable dynamics as constraints
        # Daily plant dynamics via closed-form logistic step over dt
        def plant_height_dynamics_rule(model, k):
            if k == horizon:
                return pyo.Constraint.Skip

            nu_h = (model.nuF[k] * model.nuT[k] * model.nuR[k] + epsilon)**(1.0/3.0)

            ah_hat = ah * nu_h
            kh_hat = kh * nu_h

            # Enforce kh_hat >= current h[k] (smoothly)
            kh_eff = smoothmax_pyomo(kh_hat, model.h[k])

            h_next = kh_eff / (1.0 + (kh_eff - model.h[k]) / (model.h[k] + epsilon) * pyo.exp(-ah_hat * dt))
            return model.h[k+1] == h_next

        model.plant_height_dynamics_constraint = pyo.Constraint(model.xk, rule=plant_height_dynamics_rule)

        def leaf_area_dynamics_rule(model, k):
            if k == horizon:
                return pyo.Constraint.Skip

            nu_aA = (model.nuF[k] * model.nuT[k] * model.nuR[k] + epsilon)**(1.0 / 3.0)
            nu_kA = (model.nuW[k] * model.nuF[k] * model.nuT[k] * model.nuR[k] + epsilon)**(1.0 / 4.0)

            aA_hat = aA * nu_aA
            kA_hat = kA * nu_kA

            # Enforce kA_hat >= current A[k]
            kA_eff = smoothmax_pyomo(kA_hat, model.A[k])

            A_next = kA_eff / (1.0 + (kA_eff - model.A[k]) / (model.A[k] + epsilon) * pyo.exp(-aA_hat * dt))
            return model.A[k+1] == A_next

        model.leaf_area_dynamics_constraint = pyo.Constraint(model.xk, rule=leaf_area_dynamics_rule)

        def number_leaves_dynamics_rule(model, k):
            if k == horizon:
                return pyo.Constraint.Skip

            # Enforce kN >= current N[k] (kN is constant, but we "floor" it forward)
            kN_eff = smoothmax_pyomo(kN, model.N[k])

            ratio = (kN_eff - model.N[k]) / (model.N[k] + epsilon)
            N_next = kN_eff / (1.0 + ratio * pyo.exp(-aN * dt))
            return model.N[k+1] == N_next

        model.number_leaves_dynamics_constraint = pyo.Constraint(model.xk, rule=number_leaves_dynamics_rule)

        def number_spikelets_dynamics_rule(model, k):
            if k == horizon:
                return pyo.Constraint.Skip

            nu_ac = (1.0 / model.nuT[k] * 1.0 / model.nuR[k] + epsilon)**(1.0/2.0)
            nu_kc = (model.nuW[k] * 1.0 / model.nuT[k] * 1.0 / model.nuR[k] + epsilon)**(1.0 / 3.0)

            ac_hat = ac * nu_ac
            kc_hat = kc * nu_kc

            # Enforce kc_hat >= current c[k]
            kc_eff = smoothmax_pyomo(kc_hat, model.c[k])

            c_next = kc_eff / (1.0 + (kc_eff - model.c[k]) / (model.c[k] + epsilon) * pyo.exp(-ac_hat * dt))
            return model.c[k+1] == c_next

        model.number_spikelets_dynamics_constraint = pyo.Constraint(model.xk, rule=number_spikelets_dynamics_rule)

        def fruit_biomass_dynamics_rule(model, k):
            if k == horizon:
                return pyo.Constraint.Skip

            nu_aP = (model.nuT[k] * model.nuR[k] + epsilon)**(1.0/2.0)
            aP_hat = aP * nu_aP

            nu_kh = (model.nuF[k] * model.nuT[k] * model.nuR[k] + epsilon)**(1.0/3.0)
            kh_hat = kh * nu_kh
            kh_eff = smoothmax_pyomo(kh_hat, model.h[k])  # keep consistent with height constraint

            nu_kA = (model.nuW[k] * model.nuF[k] * model.nuT[k] * model.nuR[k] * (kh_eff / kh) + epsilon)**(1.0/5.0)
            kA_hat = kA * nu_kA
            kA_eff = smoothmax_pyomo(kA_hat, model.A[k])

            nu_kc = (model.nuW[k] * (1.0 / model.nuT[k]) * (1.0 / model.nuR[k]) + epsilon)**(1.0/3.0)
            kc_hat = kc * nu_kc
            kc_eff = smoothmax_pyomo(kc_hat, model.c[k])

            nu_kP = (
                model.nuW[k] * model.nuF[k] * model.nuT[k] * model.nuR[k]
                * (kh_eff / kh) * (kA_eff / kA) * (kc_eff / kc) + epsilon
            )**(1.0/7.0)
            kP_hat = kP * nu_kP

            # Enforce kP_hat >= current P[k]
            kP_eff = smoothmax_pyomo(kP_hat, model.P[k])

            P_next = kP_eff / (1.0 + (kP_eff - model.P[k]) / (model.P[k] + epsilon) * pyo.exp(-aP_hat * dt))
            return model.P[k+1] == P_next

        model.fruit_biomass_dynamics_constraint = pyo.Constraint(model.xk, rule=fruit_biomass_dynamics_rule)

        # Objective: daily resource usage + running fruit reward + terminal fruit
        def objective_rule(model):
            stage_cost     = 0.0

            for k in model.uk:
                # Resource costs (per-day)
                stage_cost += (
                    self.mpc_params.weight_irrigation   * (model.uW[k] / W_typ)**2 # encourage sparsity
                    + self.mpc_params.weight_fertilizer * (model.uF[k] / F_typ)**2 # encourage sparsity
                )

                stage_cost += (
                    self.mpc_params.weight_water_anomaly      * model.cumulative_divergence_water[k]**2 +
                    self.mpc_params.weight_fertilizer_anomaly * model.cumulative_divergence_fertilizer[k]**2
                )

            terminal_reward = -self.mpc_params.weight_fruit_biomass * model.P[horizon] / kP
            terminal_reward -= self.mpc_params.weight_height * model.h[horizon] / kh
            terminal_reward -= self.mpc_params.weight_leaf_area * model.A[horizon] / k

            return stage_cost + terminal_reward

        model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # Solve
        solver_name = self.mpc_params.solver
        solver = pyo.SolverFactory(solver_name)
        solver_options = self.mpc_params.solver_options or {} # handle case where solver_options is None

        for k, v in solver_options.items():
            solver.options[k] = v

        results = solver.solve(model, tee=False, load_solutions=False)

        status = results.solver.status
        term   = results.solver.termination_condition
        print(f"[CFTOC] Solver status: {status}, termination: {term}")

        # Decide if we trust this solution
        good_status = (
            status == SolverStatus.ok
            and term in (
                TerminationCondition.optimal,
                TerminationCondition.locallyOptimal,
                TerminationCondition.feasible,
                TerminationCondition.maxIterations,
            )
        )

        if not good_status:
            print("[CFTOC] WARNING: bad solver status, using fallback control plan.")

            # Fallback: hold last control or use nominal guess
            if u_prev is not None:
                u_prev = np.asarray(u_prev).flatten()
                irrigation_amount = float(np.clip(u_prev[0], irrigation_lower_bound, irrigation_upper_bound))
                fertilizer_amount = float(np.clip(u_prev[1], fertilizer_lower_bound, fertilizer_upper_bound))
            else:
                # Use nominal guesses
                irrigation_amount = float(irrigation_amount_guess)
                fertilizer_amount = float(fertilizer_amount_guess)

            irrigation_plan = np.full(horizon, irrigation_amount)
            fertilizer_plan = np.full(horizon, fertilizer_amount)

            return irrigation_plan, fertilizer_plan

        # Only load solution when status is OK
        try:
            model.solutions.load_from(results)
        except Exception as e:
            print("[CFTOC] ERROR loading solution despite good status, falling back:", e)

            if u_prev is not None:
                u_prev = np.asarray(u_prev).flatten()
                irrigation_amount = float(np.clip(u_prev[0], irrigation_lower_bound, irrigation_upper_bound))
                fertilizer_amount = float(np.clip(u_prev[1], fertilizer_lower_bound, fertilizer_upper_bound))
            else:
                irrigation_amount = float(irrigation_amount_guess * dt)
                fertilizer_amount = float(fertilizer_amount_guess * dt)

            irrigation_plan = np.full(horizon, irrigation_amount)
            fertilizer_plan = np.full(horizon, fertilizer_amount)

            return irrigation_plan, fertilizer_plan

        # Extract optimal controls and clamp to bounds
        irrigation_plan = np.array([
            max(irrigation_lower_bound, min(pyo.value(model.uW[k]), irrigation_upper_bound))
            for k in model.uk
        ])
        fertilizer_plan = np.array([
            max(fertilizer_lower_bound, min(pyo.value(model.uF[k]), fertilizer_upper_bound))
            for k in model.uk
        ])

        return irrigation_plan, fertilizer_plan


    def single_time_step(
        self,
        x,
        u,
        d,
        C,
        extra_state
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        One-step plant update for the MPC loop.

        Args:
            x:
                Current state vector [h, A, N, c, P] at step k.
            u:
                Control vector [uW, uF] at step k.
            d:
                Dict with disturbances at step k:
                    {
                        "precipitation": float,
                        "temperature":   float,
                        "radiation":     float
                    }
            C:
                Cumulative delayed vector [Cw, Cf, Ct, Cr] at step k.
            extra_state:
                Dict containing FIR histories and nutrient-history.
                On the very first call, this can be None.

        Returns:
            x (next):
                Next state vector [h, A, N, c, P].
            C (next):
                Next cumulative vector.
            extra_state (next):
                Updated histories for use on the next step.
        """

        # Unpack model parameters
        dt                   = self.model_params.dt
        alpha                = self.sensitivities.alpha
        beta_divergence      = self.sensitivities.beta_divergence
        beta_nutrient_factor = self.sensitivities.beta_nutrient_factor
        epsilon              = self.sensitivities.epsilon

        # Unpack typical disturbances
        W_typ = self.typical_disturbances.typical_water
        F_typ = self.typical_disturbances.typical_fertilizer
        T_typ = self.typical_disturbances.typical_temperature
        R_typ = self.typical_disturbances.typical_radiation

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

        # Unpack FIR kernels (truncate to FIR horizon)
        fir_horizon_W = self.fir_horizon_W
        fir_horizon_F = self.fir_horizon_F
        fir_horizon_T = self.fir_horizon_T
        fir_horizon_R = self.fir_horizon_R

        kernel_W = self.kernel_W[:fir_horizon_W]
        kernel_F = self.kernel_F[:fir_horizon_F]
        kernel_T = self.kernel_T[:fir_horizon_T]
        kernel_R = self.kernel_R[:fir_horizon_R]

        # Ensure inputs are numpy arrays
        x = np.asarray(x, dtype=float).copy()
        C = np.asarray(C, dtype=float).copy()
        u = np.asarray(u, dtype=float).copy()

        # Unpack states
        h, A, N, c, P = x

        # Unpack cumulative values
        cumulative_water, cumulative_fertilizer, cumulative_temperature, cumulative_radiation = C

        # Unpack control inputs
        W, F = u  # irrigation, fertilizer

        # Unpack disturbances
        S = float(d["precipitation"])
        T = float(d["temperature"])
        R = float(d["radiation"])

        # Initialize histories if needed
        if extra_state is None:
            extra_state = {
                "water_history":       np.ones(fir_horizon_W, dtype=float) * W_typ,
                "fertilizer_history":  np.ones(fir_horizon_F, dtype=float) * F_typ,
                "temperature_history": np.ones(fir_horizon_T, dtype=float) * T_typ,
                "radiation_history":   np.ones(fir_horizon_R, dtype=float) * R_typ,
                "step":                0,
                "log": {
                    "delayed_water":                     [0.0],
                    "delayed_fertilizer":                [0.0],
                    "delayed_temperature":               [0.0],
                    "delayed_radiation":                 [0.0],
                    "cumulative_water":                  [float(cumulative_water)],
                    "cumulative_fertilizer":             [float(cumulative_fertilizer)],
                    "cumulative_temperature":            [float(cumulative_temperature)],
                    "cumulative_radiation":              [float(cumulative_radiation)],
                    "water_anomaly":                     [0.0],
                    "fertilizer_anomaly":                [0.0],
                    "temperature_anomaly":               [0.0],
                    "radiation_anomaly":                 [0.0],
                    "cumulative_divergence_water":       [0.0],
                    "cumulative_divergence_fertilizer":  [0.0],
                    "cumulative_divergence_temperature": [0.0],
                    "cumulative_divergence_radiation":   [0.0],
                    "nuW":                               [1.0],
                    "nuF":                               [1.0],
                    "nuT":                               [1.0],
                    "nuR":                               [1.0],
                }
            }

        water_history       = extra_state["water_history"]
        fertilizer_history  = extra_state["fertilizer_history"]
        temperature_history = extra_state["temperature_history"]
        radiation_history   = extra_state["radiation_history"]

        # Update FIR histories (shift left, append new sample)
        water_history       = np.roll(water_history,       -1)
        fertilizer_history  = np.roll(fertilizer_history,  -1)
        temperature_history = np.roll(temperature_history, -1)
        radiation_history   = np.roll(radiation_history,   -1)

        water_history[-1]       = W + S   # irrigation + precipitation
        fertilizer_history[-1]  = F
        temperature_history[-1] = T
        radiation_history[-1]   = R

        extra_state["water_history"]       = water_history
        extra_state["fertilizer_history"]  = fertilizer_history
        extra_state["temperature_history"] = temperature_history
        extra_state["radiation_history"]   = radiation_history

        # Compute delayed signals via FIR (convolution outputs)
        delayed_water       = np.dot(kernel_W, water_history)
        delayed_fertilizer  = np.dot(kernel_F, fertilizer_history)
        delayed_temperature = np.dot(kernel_T, temperature_history)
        delayed_radiation   = np.dot(kernel_R, radiation_history)

        extra_state["log"]["delayed_water"].append(delayed_water)
        extra_state["log"]["delayed_fertilizer"].append(delayed_fertilizer)
        extra_state["log"]["delayed_temperature"].append(delayed_temperature)
        extra_state["log"]["delayed_radiation"].append(delayed_radiation)

        # Update cumulative delayed values
        cumulative_water       = cumulative_water       + delayed_water
        cumulative_fertilizer  = cumulative_fertilizer  + delayed_fertilizer
        cumulative_temperature = cumulative_temperature + delayed_temperature
        cumulative_radiation   = cumulative_radiation   + delayed_radiation

        extra_state["log"]["cumulative_water"].append(cumulative_water)
        extra_state["log"]["cumulative_fertilizer"].append(cumulative_fertilizer)
        extra_state["log"]["cumulative_temperature"].append(cumulative_temperature)
        extra_state["log"]["cumulative_radiation"].append(cumulative_radiation)

        # Update step counter (like time index k)
        step = extra_state.get("step", 0) + 1
        extra_state["step"] = step

        # Cumulative average deviations
        k_idx = step - 1
        water_anomaly       = max(np.abs(W_typ * k_idx - cumulative_water)       / (W_typ * (k_idx + 1) + epsilon), epsilon)
        fertilizer_anomaly  = max(np.abs(F_typ * k_idx - cumulative_fertilizer)  / (F_typ * (k_idx + 1) + epsilon), epsilon)
        temperature_anomaly = max(np.abs(T_typ * k_idx - cumulative_temperature) / (T_typ * (k_idx + 1) + epsilon), epsilon)
        radiation_anomaly   = max(np.abs(R_typ * k_idx - cumulative_radiation)   / (R_typ * (k_idx + 1) + epsilon), epsilon)

        extra_state["log"]["water_anomaly"].append(water_anomaly)
        extra_state["log"]["fertilizer_anomaly"].append(fertilizer_anomaly)
        extra_state["log"]["temperature_anomaly"].append(temperature_anomaly)
        extra_state["log"]["radiation_anomaly"].append(radiation_anomaly)

        # Previous cumulative divergences
        prev_cumulative_divergence_water       = extra_state["log"]["cumulative_divergence_water"][-1]
        prev_cumulative_divergence_fertilizer  = extra_state["log"]["cumulative_divergence_fertilizer"][-1]
        prev_cumulative_divergence_temperature = extra_state["log"]["cumulative_divergence_temperature"][-1]
        prev_cumulative_divergence_radiation   = extra_state["log"]["cumulative_divergence_radiation"][-1]

        # Recursive cumulative divergence update
        cumulative_divergence_water       = beta_divergence * prev_cumulative_divergence_water       + (1.0 - beta_divergence) * water_anomaly
        cumulative_divergence_fertilizer  = beta_divergence * prev_cumulative_divergence_fertilizer  + (1.0 - beta_divergence) * fertilizer_anomaly
        cumulative_divergence_temperature = beta_divergence * prev_cumulative_divergence_temperature + (1.0 - beta_divergence) * temperature_anomaly
        cumulative_divergence_radiation   = beta_divergence * prev_cumulative_divergence_radiation   + (1.0 - beta_divergence) * radiation_anomaly

        extra_state["log"]["cumulative_divergence_water"].append(cumulative_divergence_water)
        extra_state["log"]["cumulative_divergence_fertilizer"].append(cumulative_divergence_fertilizer)
        extra_state["log"]["cumulative_divergence_temperature"].append(cumulative_divergence_temperature)
        extra_state["log"]["cumulative_divergence_radiation"].append(cumulative_divergence_radiation)

        # Raw nutrient factors
        nuW_raw = np.exp(-alpha * cumulative_divergence_water)
        nuF_raw = np.exp(-alpha * cumulative_divergence_fertilizer)
        nuT_raw = np.exp(-alpha * cumulative_divergence_temperature)
        nuR_raw = np.exp(-alpha * cumulative_divergence_radiation)

        # Previous EMA nutrient factors
        prev_nuW = extra_state["log"]["nuW"][-1]
        prev_nuF = extra_state["log"]["nuF"][-1]
        prev_nuT = extra_state["log"]["nuT"][-1]
        prev_nuR = extra_state["log"]["nuR"][-1]

        # Final, smoothed nutrient factors (EMA: weight on previous value is beta)
        nuW = beta_nutrient_factor * prev_nuW + (1.0 - beta_nutrient_factor) * nuW_raw
        nuF = beta_nutrient_factor * prev_nuF + (1.0 - beta_nutrient_factor) * nuF_raw
        nuT = beta_nutrient_factor * prev_nuT + (1.0 - beta_nutrient_factor) * nuT_raw
        nuR = beta_nutrient_factor * prev_nuR + (1.0 - beta_nutrient_factor) * nuR_raw

        extra_state["log"]["nuW"].append(nuW)
        extra_state["log"]["nuF"].append(nuF)
        extra_state["log"]["nuT"].append(nuT)
        extra_state["log"]["nuR"].append(nuR)

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

        # Closed-form logistic step (matches GA and CFTOC constraints)
        h_next = logistic_step(h, ah_hat, kh_hat, dt)
        A_next = logistic_step(A, aA_hat, kA_hat, dt)
        N_next = logistic_step(N, aN_hat, kN_hat, dt)
        c_next = logistic_step(c, ac_hat, kc_hat, dt)
        P_next = logistic_step(P, aP_hat, kP_hat, dt)

        # Enforce non-negativity explicitly
        h_next = max(h_next, 0.0)
        A_next = max(A_next, 0.0)
        N_next = max(N_next, 0.0)
        c_next = max(c_next, 0.0)
        P_next = max(P_next, 0.0)

        # Pack x, C, and extra_state for return
        x = np.array([h_next, A_next, N_next, c_next, P_next])

        C = np.array([
            cumulative_water,
            cumulative_fertilizer,
            cumulative_temperature,
            cumulative_radiation
        ])

        return x, C, extra_state
