# mpc.py
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from typing import Optional, Dict, Tuple

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

        # Precompute Gaussian kernels from existing sensitivities TODO FIR instead??
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


    def run(
        self
    ) -> Dict[str, np.ndarray]:
        """
        Run receding-horizon, move-blocked MPC over the full growing season.

        Returns:
            trajectories:
                Dict containing time series for states and controls:
                    "h", "A", "N", "c", "P", "uW", "uF"
                and optional logs.
        """
        # Unpack model parameters
        total_time_steps = self.model_params.total_time_steps
        dt               = self.model_params.dt
        horizon          = int(self.mpc_params.hourly_horizon * int(1 / dt))  # convert hours to time steps

        # Unpack initial conditions
        h0 = self.initial_conditions.h0
        A0 = self.initial_conditions.A0
        N0 = self.initial_conditions.N0
        c0 = self.initial_conditions.c0
        P0 = self.initial_conditions.P0

        # Initialize storage for state variables
        h = np.full(total_time_steps, h0)
        A = np.full(total_time_steps, A0)
        N = np.full(total_time_steps, N0)
        c = np.full(total_time_steps, c0)
        P = np.full(total_time_steps, P0)

        # Set initial conditions for the MPC loop
        x = np.array([h0, A0, N0, c0, P0])
        C = np.zeros(4)
        extra_state = None

        # Initialize storage for control inputs
        irrigation = np.zeros(total_time_steps)
        fertilizer = np.zeros(total_time_steps)

        # Unpack disturbances
        hourly_precipitation = self.disturbances.precipitation
        hourly_temperature   = self.disturbances.temperature
        hourly_radiation     = self.disturbances.radiation

        precipitation = get_sim_inputs_from_hourly(
            hourly_array     = hourly_precipitation,
            dt               = self.model_params.dt,
            simulation_hours = self.model_params.simulation_hours,
            mode             = 'split')
        temperature = get_sim_inputs_from_hourly(
            hourly_array     = hourly_temperature,
            dt               = self.model_params.dt,
            simulation_hours = self.model_params.simulation_hours,
            mode             = 'split')
        radiation = get_sim_inputs_from_hourly(
            hourly_array     = hourly_radiation,
            dt               = self.model_params.dt,
            simulation_hours = self.model_params.simulation_hours,
            mode             = 'split')

        # Initialize plan & index
        irrigation_plan = None
        fertilizer_plan = None
        plan_idx = 0

        irrigation_guess = self.bounds.irrigation_amount_guess
        fertilizer_guess = self.bounds.fertilizer_amount_guess

        for k in range(total_time_steps - 1):

            print(f"Step {k}/{total_time_steps}")

            # Decide whether to re-optimize
            need_new_plan = (
                irrigation_plan is None or
                plan_idx >= len(irrigation_plan) or
                (k % self.mpc_params.reoptimization_interval == 0)
            )

            if need_new_plan:
                # Build local forecast
                k_end = min(k + horizon, total_time_steps)
                local_horizon = k_end - k

                forecasted_precipitation = precipitation[k:k_end]
                forecasted_temperature   = temperature[k:k_end]
                forecasted_radiation     = radiation[k:k_end]

                if local_horizon < horizon:
                    pad = horizon - local_horizon
                    forecasted_precipitation = np.pad(forecasted_precipitation, (0, pad), mode="edge")
                    forecasted_temperature   = np.pad(forecasted_temperature,   (0, pad), mode="edge")
                    forecasted_radiation     = np.pad(forecasted_radiation,     (0, pad), mode="edge")

                forecast = {
                    "precipitation": forecasted_precipitation,
                    "temperature":   forecasted_temperature,
                    "radiation":     forecasted_radiation,
                }

                try:
                    irrigation_plan, fertilizer_plan = self.solve_cftoc(
                        x0=x,
                        C0=C,
                        forecast=forecast,
                        u_prev=(irrigation_guess, fertilizer_guess),
                    )
                    # Warm start for the NEXT horizon
                    irrigation_lower_bound, irrigation_upper_bound = self.bounds.irrigation_bounds
                    fertilizer_lower_bound, fertilizer_upper_bound = self.bounds.fertilizer_bounds

                    # First element of the new plan as guess for next CFTOC
                    irrigation_guess = float(np.clip(irrigation_plan[0], irrigation_lower_bound, irrigation_upper_bound))
                    fertilizer_guess = float(np.clip(fertilizer_plan[0], fertilizer_lower_bound, fertilizer_upper_bound))

                    plan_idx = 0
                except RuntimeError as e:
                    print(f"[MPC] CFTOC failed at step {k}: {e}")
                    # Fallback: set controls to zero
                    irrigation_plan = np.zeros(horizon)
                    fertilizer_plan = np.zeros(horizon)
                    plan_idx = 0

            # Apply next element of the current plan 
            u_k = np.array([irrigation_plan[plan_idx], fertilizer_plan[plan_idx]])
            plan_idx += 1

            d_k = {
                "precipitation": precipitation[k],
                "temperature":   temperature[k],
                "radiation":     radiation[k],
            }

            x, C, extra_state = self.single_time_step(
                x=x,
                u=u_k,
                d=d_k,
                C=C,
                extra_state=extra_state,
            )

            # Store applied controls
            irrigation[k] = u_k[0]
            fertilizer[k] = u_k[1]

            # Store next state
            h[k + 1] = x[0]
            A[k + 1] = x[1]
            N[k + 1] = x[2]
            c[k + 1] = x[3]
            P[k + 1] = x[4]

        # Determine how many steps were actually simulated
        n_steps = k

        # Slice trajectories to the executed steps (optionally +1 for final state)
        h_out = h[:n_steps+1]
        A_out = A[:n_steps+1]
        N_out = N[:n_steps+1]
        c_out = c[:n_steps+1]
        P_out = P[:n_steps+1]
        irrigation_out = irrigation[:n_steps]
        fertilizer_out = fertilizer[:n_steps]

        # Extract logs from extra_state (if we ever ran single_time_step)
        log_dict = {}
        if extra_state is not None and "log" in extra_state:
            for key, lst in extra_state["log"].items():
                log_dict[key] = np.array(lst)

        result = {
            "h":  h_out,
            "A":  A_out,
            "N":  N_out,
            "c":  c_out,
            "P":  P_out,
            "irrigation": irrigation_out,
            "fertilizer": fertilizer_out,
        }
        result["logs"] = log_dict

        return result


    def solve_cftoc(
        self,
        x0:       np.ndarray,
        C0:       np.ndarray,
        forecast: Dict[str, np.ndarray],
        u_prev:   Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the nonlinear CFTOC over the horizon using Pyomo.

        Args:
            x0:
                Current plant state at the beginning of the horizon:
                    x = [h, A, N, c, P]
            C0:
                Current cumulative states at the beginning of the horizon:
                    C = [cumulative_water, cumulative_fertilizer,
                         cumulative_temperature, cumulative_radiation]
                These approximate the cumulative delayed water/fertilizer/
                temperature/radiation up to the present.
            forecast:
                Dictionary of disturbance forecasts over the horizon:
                    forecast["precip"]      : ndarray of length horizon
                    forecast["temperature"] : ndarray of length horizon
                    forecast["radiation"]   : ndarray of length horizon
            u_prev:
                Optional previous control used at the last real time step. 
                This is useful for penalize changes in control (smoothing).

        Returns:
            (irrigation_plan, fertilizer_plan):
                Two ndarrays of horizon length (in time steps, not hours) with the optimal 
                control sequence.
        """
        # Unpack model parameters
        dt      = self.model_params.dt
        horizon = int(self.mpc_params.hourly_horizon * (1 / dt))  # convert hours to time steps
        model.alpha   = pyo.Param(initialize=self.sensitivities.alpha, mutable=False)
        model.epsilon = pyo.Param(initialize=self.model_params.epsilon, mutable=False)

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
        W_typ = self.typical_disturbances.typical_water
        F_typ = self.typical_disturbances.typical_fertilizer
        T_typ = self.typical_disturbances.typical_temperature
        R_typ = self.typical_disturbances.typical_radiation

        # Build Pyomo model
        model    = pyo.ConcreteModel()
        model.xk = pyo.RangeSet(0, horizon)     # states   indexed 0..N
        model.uk = pyo.RangeSet(0, horizon - 1) # controls indexed 0..N-1
        model.kk = pyo.RangeSet(0, horizon - 1) # kernel indices

        # Disturbance forecasts as parameters
        model.precipitation = pyo.Param(model.uk, initialize=lambda model, k: float(forecast["precipitation"][k]), mutable=False)
        model.temperature   = pyo.Param(model.uk, initialize=lambda model, k: float(forecast["temperature"][k]),   mutable=False)
        model.radiation     = pyo.Param(model.uk, initialize=lambda model, k: float(forecast["radiation"][k]),     mutable=False)

        # Kernels as parameters
        model.kernel_W = pyo.Param(model.kk, initialize=lambda model, j: float(self.kernel_W[j]), mutable=False)
        model.kernel_F = pyo.Param(model.kk, initialize=lambda model, j: float(self.kernel_F[j]), mutable=False)
        model.kernel_T = pyo.Param(model.kk, initialize=lambda model, j: float(self.kernel_T[j]), mutable=False)
        model.kernel_R = pyo.Param(model.kk, initialize=lambda model, j: float(self.kernel_R[j]), mutable=False)

        # Decision variables: control inputs
        model.uW = pyo.Var(model.uk, bounds=self.bounds.irrigation_bounds)
        model.uF = pyo.Var(model.uk, bounds=self.bounds.fertilizer_bounds)

        # State variables
        model.h = pyo.Var(model.xk) # plant height
        model.A = pyo.Var(model.xk) # leaf area
        model.N = pyo.Var(model.xk) # number of leaves
        model.c = pyo.Var(model.xk) # flower spikelet count
        model.P = pyo.Var(model.xk) # fruit biomass

        # Augmented states: cumulative delayed signals
        model.cumulative_water       = pyo.Var(model.xk)
        model.cumulative_fertilizer  = pyo.Var(model.xk)
        model.cumulative_temperature = pyo.Var(model.xk)
        model.cumulative_radiation   = pyo.Var(model.xk)

        # Delayed signals (convolution outputs)
        model.delayed_water       = pyo.Var(model.uk)
        model.delayed_fertilizer  = pyo.Var(model.uk)
        model.delayed_temperature = pyo.Var(model.uk)
        model.delayed_radiation   = pyo.Var(model.uk)

        # Per-step deviation of actual cumulative values from expectation
        model.cumulative_average_water       = pyo.Var(model.uk)
        model.cumulative_average_fertilizer  = pyo.Var(model.uk)
        model.cumulative_average_temperature = pyo.Var(model.uk)
        model.cumulative_average_radiation   = pyo.Var(model.uk)

        # Cumulative-average deviations
        model.cumulative_divergence_water       = pyo.Var(model.uk)
        model.cumulative_divergence_fertilizer  = pyo.Var(model.uk)
        model.cumulative_divergence_temperature = pyo.Var(model.uk)
        model.cumulative_divergence_radiation   = pyo.Var(model.uk)

        # Initial conditions (states & augmented states)
        x0 = np.asarray(x0).flatten()
        C0 = np.asarray(C0).flatten()

        model.h[0].fix(float(x0[0]))
        model.A[0].fix(float(x0[1]))
        model.N[0].fix(float(x0[2]))
        model.c[0].fix(float(x0[3]))
        model.P[0].fix(float(x0[4]))

        model.cumulative_water[0].fix(float(C0[0]))
        model.cumulative_fertilizer[0].fix(float(C0[1]))
        model.cumulative_temperature[0].fix(float(C0[2]))
        model.cumulative_radiation[0].fix(float(C0[3]))

        # Initial guesses for controls
        irrigation_lower_bound, irrigation_upper_bound = self.bounds.irrigation_bounds
        fertilizer_lower_bound, fertilizer_upper_bound = self.bounds.fertilizer_bounds
        if u_prev is not None:
            u_prev = np.asarray(u_prev).flatten()
            uW0 = float(np.clip(u_prev[0], irrigation_lower_bound, irrigation_upper_bound))
            uF0 = float(np.clip(u_prev[1], fertilizer_lower_bound, fertilizer_upper_bound))
            for k in model.uk:
                model.uW[k].value = uW0
                model.uF[k].value = uF0
        else:
            for k in model.uk:
                model.uW[k].value = self.bounds.irrigation_amount_guess
                model.uF[k].value = self.bounds.fertilizer_amount_guess

        # FIR convolution constraints for delayed signals
        def delayed_water_rule(model, k):
            return model.delayed_water[k] == sum(
                model.kernel_W[j] * (model.uW[k - j] + model.precipitation[k - j])
                for j in model.kk if k - j >= 0
            )
        model.delayed_water_constraint = pyo.Constraint(model.uk, rule=delayed_water_rule)

        def delayed_fertilizer_rule(model, k):
            return model.delayed_fertilizer[k] == sum(
                model.kernel_F[j] * model.uF[k - j]
                for j in model.kk if k - j >= 0
            )
        model.delayed_fertilizer_constraint = pyo.Constraint(model.uk, rule=delayed_fertilizer_rule)

        def delayed_temperature_rule(model, k):
            return model.delayed_temperature[k] == sum(
                model.kernel_T[j] * model.temperature[k - j]
                for j in model.kk if k - j >= 0
            )
        model.delayed_temperature_constraint = pyo.Constraint(model.uk, rule=delayed_temperature_rule)

        def delayed_radiation_rule(model, k):
            return model.delayed_radiation[k] == sum(
                model.kernel_R[j] * model.radiation[k - j]
                for j in model.kk if k - j >= 0
            )
        model.delayed_radiation_constraint = pyo.Constraint(model.uk, rule=delayed_radiation_rule)

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

        # Cumulative average updates
        def cumulative_average_water_rule(model, k):
            k_idx = k  # matches np.arange indexing
            num   = W_typ * k_idx - model.cumulative_water[k]
            denom = W_typ * (k_idx + 1) + model.epsilon
            return model.cumulative_average_water[k] == pyo.sqrt((num / denom) ** 2 + model.epsilon)
        model.cumulative_average_water_constraint = pyo.Constraint(model.uk, rule=cumulative_average_water_rule)

        def cumulative_average_fertilizer_rule(model, k):
            k_idx = k
            num   = F_typ * k_idx - model.cumulative_fertilizer[k]
            denom = F_typ * (k_idx + 1) + model.epsilon
            return model.cumulative_average_fertilizer[k] == pyo.sqrt((num / denom) ** 2 + model.epsilon)
        model.cumulative_average_fertilizer_constraint = pyo.Constraint(model.uk, rule=cumulative_average_fertilizer_rule)

        def cumulative_average_temperature_rule(model, k):
            k_idx = k
            num   = T_typ * k_idx - model.cumulative_temperature[k]
            denom = T_typ * (k_idx + 1) + model.epsilon
            return model.cumulative_average_temperature[k] == pyo.sqrt((num / denom) ** 2 + model.epsilon)
        model.cumulative_average_temperature_constraint = pyo.Constraint(model.uk, rule=cumulative_average_temperature_rule)

        def cumulative_average_radiation_rule(model, k):
            k_idx = k
            num   = R_typ * k_idx - model.cumulative_radiation[k]
            denom = R_typ * (k_idx + 1) + model.epsilon
            return model.cumulative_average_radiation[k] == pyo.sqrt((num / denom) ** 2 + model.epsilon)
        model.cumulative_average_radiation_constraint = pyo.Constraint(model.uk, rule=cumulative_average_radiation_rule)

        # Cumulative divergence updates
        def cumulative_divergence_water_rule(model, k):
            if k == 0:
                return model.cumulative_divergence_water[0] == model.cumulative_average_water[0]
            return model.cumulative_divergence_water[k] == (
                k * model.cumulative_divergence_water[k-1] + model.cumulative_average_water[k]
            ) / (k + 1)
        model.cumulative_divergence_water_constraint = pyo.Constraint(model.uk, rule=cumulative_divergence_water_rule)

        def cumulative_divergence_fertilizer_rule(model, k):
            if k == 0:
                return model.cumulative_divergence_fertilizer[0] == model.cumulative_average_fertilizer[0]
            return model.cumulative_divergence_fertilizer[k] == (
                k * model.cumulative_divergence_fertilizer[k-1] + model.cumulative_average_fertilizer[k]
            ) / (k + 1)
        model.cumulative_divergence_fertilizer_constraint = pyo.Constraint(model.uk, rule=cumulative_divergence_fertilizer_rule)

        def cumulative_divergence_temperature_rule(model, k):
            if k == 0:
                return model.cumulative_divergence_temperature[0] == model.cumulative_average_temperature[0]
            return model.cumulative_divergence_temperature[k] == (
                k * model.cumulative_divergence_temperature[k-1] + model.cumulative_average_temperature[k]
            ) / (k + 1)
        model.cumulative_divergence_temperature_constraint = pyo.Constraint(model.uk, rule=cumulative_divergence_temperature_rule)

        def cumulative_divergence_radiation_rule(model, k):
            if k == 0:
                return model.cumulative_divergence_radiation[0] == model.cumulative_average_radiation[0]
            return model.cumulative_divergence_radiation[k] == (
                k * model.cumulative_divergence_radiation[k-1] + model.cumulative_average_radiation[k]
            ) / (k + 1)
        model.cumulative_divergence_radiation_constraint = pyo.Constraint(model.uk, rule=cumulative_divergence_radiation_rule)

        # Nutrient factor updates
        def nuW_rule(model, k):
            return pyo.exp(- model.sensitivity_alpha * model.cumulative_divergence_water[k])
        model.nuW = pyo.Expression(model.uk, rule=nuW_rule)

        def nuF_rule(model, k):
            return pyo.exp(- model.sensitivity_alpha * model.cumulative_divergence_fertilizer[k])
        model.nuF = pyo.Expression(model.uk, rule=nuF_rule)
        
        def nuT_rule(model, k):
            return pyo.exp(- model.sensitivity_alpha * model.cumulative_divergence_temperature[k])
        model.nuT = pyo.Expression(model.uk, rule=nuT_rule)
        
        def nuR_rule(model, k):
            return pyo.exp(- model.sensitivity_alpha * model.cumulative_divergence_radiation[k])
        model.nuR = pyo.Expression(model.uk, rule=nuR_rule)

        # Plant dynamics (Forward Euler logistic-like update)
        def plant_height_dynamics_rule(model, k):
            if k == horizon:
                return pyo.Constraint.Skip

            nu_kh = model.nuF[k] * model.nuT[k] * model.nuR[k] + model.epsilon

            ah_hat = ah * (nu_kh) ** (1.0 / 3.0)
            kh_hat = kh * (nu_kh) ** (1.0 / 3.0)

            return model.h[k + 1] == model.h[k] + dt * (
                ah_hat * model.h[k] * (1.0 - model.h[k] / (kh_hat + 1e-9))
            )
        model.plant_height_dynamics_constraint = pyo.Constraint(model.xk, rule=plant_height_dynamics_rule)

        def leaf_area_dynamics_rule(model, k):
            if k == horizon:
                return pyo.Constraint.Skip
            
            nu_aA = model.nuF[k] * model.nuT[k] * model.nuR[k] + model.epsilon
            nu_kA = model.nuW[k] * model.nuF[k] * model.nuT[k] * model.nuR[k] + model.epsilon

            aA_hat = aA * (nu_aA) ** (1.0 / 3.0)
            kA_hat = kA * (nu_kA) ** (1.0 / 5.0)

            return model.A[k + 1] == model.A[k] + dt * (
                aA_hat * model.A[k] * (1.0 - model.A[k] / (kA_hat + 1e-9))
            )
        model.leaf_area_dynamics_constraint = pyo.Constraint(model.xk, rule=leaf_area_dynamics_rule)

        def number_leaves_dynamics_rule(model, k):
            if k == horizon:
                return pyo.Constraint.Skip
            # For now: keep N dynamics simpler (no nu dependence)
            return model.N[k + 1] == model.N[k] + dt * (
                aN * model.N[k] * (1.0 - model.N[k] / (kN + 1e-9))
            )
        model.number_leaves_dynamics_constraint = pyo.Constraint(model.xk, rule=number_leaves_dynamics_rule)

        def number_spikelets_dynamics_rule(model, k):
            if k == horizon:
                return pyo.Constraint.Skip
            
            nu_ac = (1.0 / (model.nuT[k] + 1e-9)) * (1.0 / (model.nuR[k] + 1e-9)) + model.epsilon
            nu_kc = model.nuW[k] * nu_ac + model.epsilon

            ac_hat = ac * (nu_ac) ** 0.5
            kc_hat = kc * (nu_kc) ** (1.0 / 3.0)

            return model.c[k + 1] == model.c[k] + dt * (
                ac_hat * model.c[k] * (1.0 - model.c[k] / (kc_hat + 1e-9))
            )
        model.number_spikelets_dynamics_constraint = pyo.Constraint(model.xk, rule=number_spikelets_dynamics_rule)

        def fruit_biomass_dynamics_rule(model, k):
            if k == horizon:
                return pyo.Constraint.Skip
            
            nu_aP = model.nuT[k] * model.nuR[k] + model.epsilon
            aP_hat = aP * (nu_aP) ** 0.5

            nu_kh = model.nuF[k] * model.nuT[k] * model.nuR[k] + model.epsilon
            kh_hat = kh * (nu_kh) ** (1.0 / 3.0)

            nu_kA = model.nuW[k] * model.nuF[k] * model.nuT[k] * model.nuR[k] + model.epsilon
            kA_hat = kA * (nu_kA) ** (1.0 / 5.0)

            nu_kc = model.nuW[k] * (1.0 / (model.nuT[k] + 1e-9)) * (1.0 / (model.nuR[k] + 1e-9)) + model.epsilon
            kc_hat = kc * (nu_kc) ** (1.0 / 3.0)

            phi_full = (model.nuW[k] * model.nuF[k] * model.nuT[k] * model.nuR[k]
                        * (kh_hat / (kh + 1e-9)) * (kA_hat / (kA + 1e-9)) * (kc_hat / (kc + 1e-9)))
            phi_full_shift = phi_full + model.epsilon

            kP_hat = kP * (phi_full_shift) ** (1.0 / 4.0)

            return model.P[k + 1] == model.P[k] + dt * (
                aP_hat * model.P[k] * (1.0 - model.P[k] / (kP_hat + 1e-9))
            )
        model.fruit_biomass_dynamics_constraint = pyo.Constraint(model.xk, rule=fruit_biomass_dynamics_rule)

        # Objective (not exactly in dollars in order to keep scaling reasonable)
        def objective_rule(model):
            stage_cost = 0.0

            for k in model.uk:
                # 1) Resource usage penalty
                stage_cost += (
                    self.mpc_params.weight_irrigation * model.uW[k] +
                    self.mpc_params.weight_fertilizer * (model.uF[k])**2 # so that fertilizer sparing (and sparsing) is more encouraged
                )

                # 2) Penalize cumulative-average deviations from typical profiles
                stage_cost += (
                    self.mpc_params.weight_cumulative_average_water       * model.cumulative_divergence_water[k]**2 +
                    self.mpc_params.weight_cumulative_average_fertilizer  * model.cumulative_divergence_fertilizer[k]**2 +
                    self.mpc_params.weight_cumulative_average_temperature * model.cumulative_divergence_temperature[k]**2 +
                    self.mpc_params.weight_cumulative_average_radiation   * model.cumulative_divergence_radiation[k]**2
                )

            # 3) Terminal reward on fruit at the end of the horizon
            terminal_reward = -self.mpc_params.weight_fruit_biomass * model.P[horizon]

            return stage_cost + terminal_reward

        model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # Solve
        solver_name = self.mpc_params.solver  # e.g. "ipopt"
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

        irrigation_lower_bound, irrigation_upper_bound = self.bounds.irrigation_bounds
        fertilizer_lower_bound, fertilizer_upper_bound = self.bounds.fertilizer_bounds

        if not good_status:
            print("[CFTOC] WARNING: bad solver status, using fallback control plan.")

            # Fallback: hold last control or use nominal guess
            if u_prev is not None:
                u_prev = np.asarray(u_prev).flatten()
                uW0 = float(np.clip(u_prev[0], irrigation_lower_bound, irrigation_upper_bound))
                uF0 = float(np.clip(u_prev[1], fertilizer_lower_bound, fertilizer_upper_bound))
            else:
                # Use nominal guesses
                uW0 = float(self.bounds.irrigation_amount_guess)
                uF0 = float(self.bounds.fertilizer_amount_guess)

            irrigation_plan = np.full(horizon, uW0)
            fertilizer_plan = np.full(horizon, uF0)
            return irrigation_plan, fertilizer_plan

        # Only load solution when status is OK
        try:
            model.solutions.load_from(results)
        except Exception as e:
            print("[CFTOC] ERROR loading solution despite good status, falling back:", e)

            if u_prev is not None:
                u_prev = np.asarray(u_prev).flatten()
                uW0 = float(np.clip(u_prev[0], irrigation_lower_bound, irrigation_upper_bound))
                uF0 = float(np.clip(u_prev[1], fertilizer_lower_bound, fertilizer_upper_bound))
            else:
                uW0 = float(self.bounds.irrigation_amount_guess)
                uF0 = float(self.bounds.fertilizer_amount_guess)

            irrigation_plan = np.full(horizon, uW0)
            fertilizer_plan = np.full(horizon, uF0)
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
        dt = self.model_params.dt

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
                "water_history":                np.zeros(fir_horizon_W),
                "fertilizer_history":           np.zeros(fir_horizon_F),
                "temperature_history":          np.zeros(fir_horizon_T),
                "radiation_history":            np.zeros(fir_horizon_R),
                "step":                         0,
                "log": {
                    "delayed_water":                     [0.0],
                    "delayed_fertilizer":                [0.0],
                    "delayed_temperature":               [0.0],
                    "delayed_radiation":                 [0.0],
                    "cumulative_water":                  [0.0],
                    "cumulative_fertilizer":             [0.0],
                    "cumulative_temperature":            [0.0],
                    "cumulative_radiation":              [0.0],
                    "cumulative_average_water":          [0.0],
                    "cumulative_average_fertilizer":     [0.0],
                    "cumulative_average_temperature":    [0.0],
                    "cumulative_average_radiation":      [0.0],
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
        k_idx = step - 1  # to mirror np.arange starting at 0
        cumulative_average_water       = max(np.abs(W_typ * k_idx - cumulative_water)       / (W_typ * (k_idx + 1) + self.model_params.epsilon), self.model_params.epsilon)
        cumulative_average_fertilizer  = max(np.abs(F_typ * k_idx - cumulative_fertilizer)  / (F_typ * (k_idx + 1) + self.model_params.epsilon), self.model_params.epsilon)
        cumulative_average_temperature = max(np.abs(T_typ * k_idx - cumulative_temperature) / (T_typ * (k_idx + 1) + self.model_params.epsilon), self.model_params.epsilon)
        cumulative_average_radiation   = max(np.abs(R_typ * k_idx - cumulative_radiation)   / (R_typ * (k_idx + 1) + self.model_params.epsilon), self.model_params.epsilon)

        extra_state["log"]["cumulative_average_water"].append(cumulative_average_water)
        extra_state["log"]["cumulative_average_fertilizer"].append(cumulative_average_fertilizer)
        extra_state["log"]["cumulative_average_temperature"].append(cumulative_average_temperature)
        extra_state["log"]["cumulative_average_radiation"].append(cumulative_average_radiation)

        # Previous cumulative divergences
        prev_cumulative_divergence_water       = extra_state["log"]["cumulative_divergence_water"][-1]
        prev_cumulative_divergence_fertilizer  = extra_state["log"]["cumulative_divergence_fertilizer"][-1]
        prev_cumulative_divergence_temperature = extra_state["log"]["cumulative_divergence_temperature"][-1]
        prev_cumulative_divergence_radiation   = extra_state["log"]["cumulative_divergence_radiation"][-1]

        # Recursive cumulative divergence update
        cumulative_divergence_water       = (k_idx * prev_cumulative_divergence_water       + cumulative_average_water) / (k_idx + 1)
        cumulative_divergence_fertilizer  = (k_idx * prev_cumulative_divergence_fertilizer  + cumulative_average_fertilizer) / (k_idx + 1)
        cumulative_divergence_temperature = (k_idx * prev_cumulative_divergence_temperature + cumulative_average_temperature) / (k_idx + 1)
        cumulative_divergence_radiation   = (k_idx * prev_cumulative_divergence_radiation   + cumulative_average_radiation) / (k_idx + 1)

        extra_state["log"]["cumulative_divergence_water"].append(cumulative_divergence_water)
        extra_state["log"]["cumulative_divergence_fertilizer"].append(cumulative_divergence_fertilizer)
        extra_state["log"]["cumulative_divergence_temperature"].append(cumulative_divergence_temperature)
        extra_state["log"]["cumulative_divergence_radiation"].append(cumulative_divergence_radiation)

        # Nutrient factors
        nuW = np.exp(-self.sensitivities.alpha * cumulative_divergence_water)
        nuF = np.exp(-self.sensitivities.alpha * cumulative_divergence_fertilizer)
        nuT = np.exp(-self.sensitivities.alpha * cumulative_divergence_temperature)
        nuR = np.exp(-self.sensitivities.alpha * cumulative_divergence_radiation)

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

        kh_hat = np.clip(kh * (nuF * nuT * nuR)**(1/3), 0, 2 * kh)
        kA_hat = np.clip(kA * (nuW * nuF * nuT * nuR * (kh_hat/kh))**(1/5), 0, 2 * kA)
        kN_hat = np.clip(kN * (nuT * nuR)**(1/2), 0, 2 * kN)
        kc_hat = np.clip(kc * (nuW * (1/nuT) * (1/nuR))**(1/3), 0, 2 * kc)
        kP_hat = np.clip(kP * (nuW * nuF * nuT * nuR * (kh_hat/kh) * (kA_hat/kA) * (kc_hat/kc))**(1/7), 0, 2 * kP)

        # Forward Euler integration, to match the CFTOC model
        h_next = h + dt * (ah_hat * h * (1.0 - h / max(kh_hat, 1e-9)))
        A_next = A + dt * (aA_hat * A * (1.0 - A / max(kA_hat, 1e-9)))
        N_next = N + dt * (aN_hat * N * (1.0 - N / max(kN_hat, 1e-9)))
        c_next = c + dt * (ac_hat * c * (1.0 - c / max(kc_hat, 1e-9)))
        P_next = P + dt * (aP_hat * P * (1.0 - P / max(kP_hat, 1e-9)))

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
