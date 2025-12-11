# mpc.py

import numpy as np
import pyomo.environ as pyo
from typing import Optional, Callable, Dict, Tuple

from ..model.model_helpers import gaussian_kernel, get_mu_from_sigma, get_sim_inputs_from_hourly
from ..model.model_carrying_capacities import ModelCarryingCapacities
from ..model.model_disturbances import ModelDisturbances
from ..model.model_growth_rates import ModelGrowthRates
from ..model.model_initial_conditions import ModelInitialConditions
from ..model.model_params import ModelParams
from ..model.model_typical_disturbances import ModelTypicalDisturbances
from ..model.model_sensitivities import ModelSensitivities


class MPC:
    """
    Nonlinear MPC controller for the plant growth model.

    This class solves a constrained finite-time optimal control (CFTOC)
    problem over a horizon of length N at each step.

    The control inputs are per-time-step irrigation and fertilizer amounts.
    Disturbances are precipitation, temperature, and solar radiation.
    The state variables are plant height, leaf area, number of leaves, flower spikelet count, and fruit biomass.
    """

    '''
    Example usage: 
    mpc = MPC(
        carrying_capacities=carrying_capacities,
        disturbances=disturbances,
        growth_rates=growth_rates,
        initial_conditions=initial_conditions,
        model_params=model_params,
        typical_disturbances=typical_disturbances,
        sensitivities=sensitivities,
        horizon_N=48,
        fir_length=24,
        plant_step_fn=lambda x, u, d, C, extra_state: default_plant_step_fn(
            mpc, x, u, d, C, extra_state
        ),
    )
    '''

    def __init__(
        self,
        carrying_capacities:  ModelCarryingCapacities,
        disturbances:         ModelDisturbances,
        growth_rates:         ModelGrowthRates,
        initial_conditions:   ModelInitialConditions,
        model_params:         ModelParams,
        typical_disturbances: ModelTypicalDisturbances,
        sensitivities:        ModelSensitivities,
        horizon_N:            int,
        fir_length:           int = 24, # divide by dt to get steps
        w_P:                  float = 1.0,
        w_W:                  float = 1e-2,
        w_F:                  float = 1e-2,
        uW_bounds:            Tuple[float, float] = (0.0, 0.5),
        uF_bounds:            Tuple[float, float] = (0.0, 0.1),
        solver:               str = "ipopt",
        solver_options:       Optional[Dict[str, float]] = None,
        plant_step_fn:        Optional[Callable] = None
    ):
        """
        Args:
            carrying_capacities, disturbances, growth_rates, initial_conditions,
            model_params, typical_disturbances, sensitivities:
                Same objects you pass into Member.

            horizon_N:
                Prediction horizon length in time steps.

            fir_length:
                Length (in steps) of the FIR kernel used to approximate the
                delayed absorption of water/fertilizer/temperature/radiation.

            w_P, w_W, w_F:
                Weights in the MPC objective:
                    J = -w_P * P_N + sum_k (w_W * u_W[k] + w_F * u_F[k])

            uW_bounds, uF_bounds:
                (min, max) bounds for irrigation and fertilizer per time step.

            solver:
                Pyomo NLP solver name (e.g. "ipopt").

            solver_options:
                Dict of solver options passed to the solver.

            plant_step_fn:
                Optional callable for closed-loop simulation:
                    x_next, extra_state = plant_step_fn(x, u, d, extra_state)
                You can pass in a wrapper around your existing forward
                simulation logic if you like.
        """
        self.carrying_capacities  = carrying_capacities
        self.disturbances         = disturbances
        self.growth_rates         = growth_rates
        self.initial_conditions   = initial_conditions
        self.model_params         = model_params
        self.typical_disturbances = typical_disturbances
        self.sensitivities        = sensitivities

        self.N         = horizon_N
        self.L_fir     = fir_length
        self.w_P       = w_P
        self.w_W       = w_W
        self.w_F       = w_F
        self.uW_bounds = uW_bounds
        self.uF_bounds = uF_bounds
        self.solver    = solver
        self.solver_options = solver_options or {}
        self.plant_step_fn   = plant_step_fn

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

        self.kernel_W = gaussian_kernel(mu_W, sigma_W / dt, self.L_fir)
        self.kernel_F = gaussian_kernel(mu_F, sigma_F / dt, self.L_fir)
        self.kernel_T = gaussian_kernel(mu_T, sigma_T / dt, self.L_fir)
        self.kernel_R = gaussian_kernel(mu_R, sigma_R / dt, self.L_fir)


    def solve_cftoc(
        self,
        x0: np.ndarray,
        C0: np.ndarray,
        forecast: Dict[str, np.ndarray],
        u_prev: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the nonlinear CFTOC over the horizon [0, ..., N-1] using Pyomo.

        Args:
            x0:
                Current plant state at the beginning of the horizon:
                    x0 = [h0, A0, N0, c0, P0]

            C0:
                Current cumulative states at the beginning of the horizon:
                    C0 = [Cw0, Cf0, Ct0, Cr0]
                These approximate the cumulative delayed water/fertilizer/
                temperature/radiation up to the present.

            forecast:
                Dictionary of disturbance forecasts over the horizon:
                    forecast["precip"]      : ndarray of length N
                    forecast["temperature"] : ndarray of length N
                    forecast["radiation"]   : ndarray of length N

            u_prev:
                Optional previous control [uW_prev, uF_prev] used at the last
                real time step. This is useful if you want to penalize changes
                in control (smoothing).

        Returns:
            (uW_opt, uF_opt):
                Two ndarrays of length N with the optimal control sequence over
                the horizon. In closed-loop MPC you typically apply only
                uW_opt[0], uF_opt[0] to the plant.
        """

        N  = self.N
        dt = self.model_params.dt

        # Unpack model parameters
        ah = self.growth_rates.ah
        aA = self.growth_rates.aA
        aN = self.growth_rates.aN
        ac = self.growth_rates.ac
        aP = self.growth_rates.aP

        kh = self.carrying_capacities.kh
        kA = self.carrying_capacities.kA
        kN = self.carrying_capacities.kN
        kc = self.carrying_capacities.kc
        kP = self.carrying_capacities.kP

        W_typ = self.typical_disturbances.typical_water
        F_typ = self.typical_disturbances.typical_fertilizer
        T_typ = self.typical_disturbances.typical_temperature
        R_typ = self.typical_disturbances.typical_radiation

        # Build Pyomo model
        model = pyo.ConcreteModel()

        model.K  = pyo.RangeSet(0, N)              # states indexed 0..N
        model.UK = pyo.RangeSet(0, N - 1)          # controls indexed 0..N-1
        model.J  = pyo.RangeSet(0, self.L_fir - 1) # FIR kernel indices

        # Disturbance forecasts as parameters
        model.precipitation = pyo.Param(model.UK, initialize=lambda model, k: float(forecast["precipitation"][k]), mutable=False)
        model.temperature   = pyo.Param(model.UK, initialize=lambda model, k: float(forecast["temperature"][k]),   mutable=False)
        model.radiation     = pyo.Param(model.UK, initialize=lambda model, k: float(forecast["radiation"][k]),     mutable=False)

        # FIR kernels as parameters
        model.gW = pyo.Param(model.J, initialize=lambda model, j: float(self.kernel_W[j]), mutable=False)
        model.gF = pyo.Param(model.J, initialize=lambda model, j: float(self.kernel_F[j]), mutable=False)
        model.gT = pyo.Param(model.J, initialize=lambda model, j: float(self.kernel_T[j]), mutable=False)
        model.gR = pyo.Param(model.J, initialize=lambda model, j: float(self.kernel_R[j]), mutable=False)

        # Decision variables: controls
        model.uW = pyo.Var(model.UK, bounds=self.uW_bounds)  # irrigation
        model.uF = pyo.Var(model.UK, bounds=self.uF_bounds)  # fertilizer

        # State variables: plant states
        model.h = pyo.Var(model.K)
        model.A = pyo.Var(model.K)
        model.N = pyo.Var(model.K)
        model.c = pyo.Var(model.K)
        model.P = pyo.Var(model.K)

        # Augmented states: cumulative delayed signals
        model.Cw = pyo.Var(model.K)
        model.Cf = pyo.Var(model.K)
        model.Ct = pyo.Var(model.K)
        model.Cr = pyo.Var(model.K)

        # Delayed signals (FIR outputs)
        model.W_del = pyo.Var(model.UK)
        model.F_del = pyo.Var(model.UK)
        model.T_del = pyo.Var(model.UK)
        model.R_del = pyo.Var(model.UK)

        # Nutrient factors (algebraic)
        model.nuW = pyo.Var(model.UK, bounds=(0.0, 1.0))
        model.nuF = pyo.Var(model.UK, bounds=(0.0, 1.0))
        model.nuT = pyo.Var(model.UK, bounds=(0.0, 1.0))
        model.nuR = pyo.Var(model.UK, bounds=(0.0, 1.0))

        # Initial conditions (state & cumulated)
        x0 = np.asarray(x0).flatten()
        C0 = np.asarray(C0).flatten()

        model.h[0].fix(float(x0[0]))
        model.A[0].fix(float(x0[1]))
        model.N[0].fix(float(x0[2]))
        model.c[0].fix(float(x0[3]))
        model.P[0].fix(float(x0[4]))

        model.Cw[0].fix(float(C0[0]))
        model.Cf[0].fix(float(C0[1]))
        model.Ct[0].fix(float(C0[2]))
        model.Cr[0].fix(float(C0[3]))

        # FIR convolution constraints for delayed signals
        def delayed_water_rule(model, k):
            return model.W_del[k] == sum(
                model.gW[j] * (model.uW[k - j] + model.precipitation[k - j])
                for j in model.J if k - j >= 0
            )
        model.delayed_water_con = pyo.Constraint(model.UK, rule=delayed_water_rule)

        def delayed_fert_rule(model, k):
            return model.F_del[k] == sum(
                model.gF[j] * model.uF[k - j]
                for j in model.J if k - j >= 0
            )
        model.delayed_fert_con = pyo.Constraint(model.UK, rule=delayed_fert_rule)

        def delayed_temp_rule(model, k):
            return model.T_del[k] == sum(
                model.gT[j] * model.temperature[k - j]
                for j in model.J if k - j >= 0
            )
        model.delayed_temp_con = pyo.Constraint(model.UK, rule=delayed_temp_rule)

        def delayed_rad_rule(model, k):
            return model.R_del[k] == sum(
                model.gR[j] * model.radiation[k - j]
                for j in model.J if k - j >= 0
            )
        model.delayed_rad_con = pyo.Constraint(model.UK, rule=delayed_rad_rule)

        # Cumulative updates
        def Cw_update_rule(model, k):
            if k == N:
                return pyo.Constraint.Skip
            return model.Cw[k + 1] == model.Cw[k] + model.W_del[k]
        model.Cw_update = pyo.Constraint(model.UK, rule=Cw_update_rule)

        def Cf_update_rule(model, k):
            if k == N:
                return pyo.Constraint.Skip
            return model.Cf[k + 1] == model.Cf[k] + model.F_del[k]
        model.Cf_update = pyo.Constraint(model.UK, rule=Cf_update_rule)

        def Ct_update_rule(model, k):
            if k == N:
                return pyo.Constraint.Skip
            return model.Ct[k + 1] == model.Ct[k] + model.T_del[k]
        model.Ct_update = pyo.Constraint(model.UK, rule=Ct_update_rule)

        def Cr_update_rule(model, k):
            if k == N:
                return pyo.Constraint.Skip
            return model.Cr[k + 1] == model.Cr[k] + model.R_del[k]
        model.Cr_update = pyo.Constraint(model.UK, rule=Cr_update_rule)

        # Nutrient factor constraints (smooth Gaussian-like dependence on
        # normalized cumulative deviation). This is where you can swap in
        # your exact nu formulas if you want.
        alpha_W = 0.333
        alpha_F = 0.0333
        alpha_T = 0.333
        alpha_R = 0.333

        def nuW_rule(model, k):
            # Avoid division by zero using (k+1)
            norm = (model.Cw[k] / (W_typ * (k + 1))) if W_typ > 0 else 1.0
            # Equivalent of exp( -alpha * (norm - 1)^2 )
            return model.nuW[k] == pyo.exp(-alpha_W * (norm - 1.0) ** 2)
        model.nuW_con = pyo.Constraint(model.UK, rule=nuW_rule)

        def nuF_rule(model, k):
            norm = (model.Cf[k] / (F_typ * (k + 1))) if F_typ > 0 else 1.0
            return model.nuF[k] == pyo.exp(-alpha_F * (norm - 1.0) ** 2)
        model.nuF_con = pyo.Constraint(model.UK, rule=nuF_rule)

        def nuT_rule(model, k):
            norm = (model.Ct[k] / (T_typ * (k + 1))) if T_typ > 0 else 1.0
            return model.nuT[k] == pyo.exp(-alpha_T * (norm - 1.0) ** 2)
        model.nuT_con = pyo.Constraint(model.UK, rule=nuT_rule)

        def nuR_rule(model, k):
            norm = (model.Cr[k] / (R_typ * (k + 1))) if R_typ > 0 else 1.0
            return model.nuR[k] == pyo.exp(-alpha_R * (norm - 1.0) ** 2)
        model.nuR_con = pyo.Constraint(model.UK, rule=nuR_rule)

        # Plant dynamics (Forward Euler logistic-like update)
        def h_dyn_rule(model, k):
            if k == N:
                return pyo.Constraint.Skip
            ah_hat = ah * (model.nuF[k] * model.nuT[k] * model.nuR[k]) ** (1.0 / 3.0)
            kh_hat = kh * (model.nuF[k] * model.nuT[k] * model.nuR[k]) ** (1.0 / 3.0)
            return model.h[k + 1] == model.h[k] + dt * (
                ah_hat * model.h[k] * (1.0 - model.h[k] / (kh_hat + 1e-9)) # avoid divide by zero
            )
        model.h_dyn = pyo.Constraint(model.K, rule=h_dyn_rule)

        def A_dyn_rule(model, k):
            if k == N:
                return pyo.Constraint.Skip
            aA_hat = aA * (model.nuF[k] * model.nuT[k] * model.nuR[k]) ** (1.0 / 3.0)
            kA_hat = kA * (model.nuW[k] * model.nuF[k] * model.nuT[k] * model.nuR[k]) ** (1.0 / 5.0)
            return model.A[k + 1] == model.A[k] + dt * (
                aA_hat * model.A[k] * (1.0 - model.A[k] / (kA_hat + 1e-9))
            )
        model.A_dyn = pyo.Constraint(model.K, rule=A_dyn_rule)

        def N_dyn_rule(model, k):
            if k == N:
                return pyo.Constraint.Skip
            # For now: keep N dynamics simpler (no nu dependence)
            return model.N[k + 1] == model.N[k] + dt * (
                aN * model.N[k] * (1.0 - model.N[k] / (kN + 1e-9))
            )
        model.N_dyn = pyo.Constraint(model.K, rule=N_dyn_rule)

        def c_dyn_rule(model, k):
            if k == N:
                return pyo.Constraint.Skip
            ac_hat = ac * ((1.0 / (model.nuT[k] + 1e-9)) * (1.0 / (model.nuR[k] + 1e-9))) ** 0.5
            kc_hat = kc * (model.nuW[k] * (1.0 / (model.nuT[k] + 1e-9)) * (1.0 / (model.nuR[k] + 1e-9))) ** (1.0 / 3.0)
            return model.c[k + 1] == model.c[k] + dt * (
                ac_hat * model.c[k] * (1.0 - model.c[k] / (kc_hat + 1e-9))
            )
        model.c_dyn = pyo.Constraint(model.K, rule=c_dyn_rule)

        def P_dyn_rule(model, k):
            if k == N:
                return pyo.Constraint.Skip
            aP_hat = aP * (model.nuT[k] * model.nuR[k]) ** 0.5
            kP_hat = kP * (model.nuW[k] * model.nuF[k] * model.nuT[k] * model.nuR[k]) ** (1.0 / 4.0)
            return model.P[k + 1] == model.P[k] + dt * (
                aP_hat * model.P[k] * (1.0 - model.P[k] / (kP_hat + 1e-9))
            )
        model.P_dyn = pyo.Constraint(model.K, rule=P_dyn_rule)

        # Objective: final fruit minus resource usage
        def objective_rule(model):
            stage_cost = sum(
                self.w_W * model.uW[k] + self.w_F * model.uF[k]
                for k in model.UK
            )
            #terminal_reward = -self.w_P * model.P[N]  # negative because we minimize
            return stage_cost #+ terminal_reward
        model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # Solve
        solver = pyo.SolverFactory(self.solver)
        for k, v in self.solver_options.items():
            solver.options[k] = v
        results = solver.solve(model, tee=False)

        # Extract optimal control sequences
        uW_opt = np.array([pyo.value(model.uW[k]) for k in model.UK])
        uF_opt = np.array([pyo.value(model.uF[k]) for k in model.UK])

        return uW_opt, uF_opt

    # ------------------------------------------------------------------
    # 3. Receding-horizon MPC loop
    # ------------------------------------------------------------------

    def run_mpc(
        self
    ) -> Dict[str, np.ndarray]:
        """
        Run receding-horizon MPC over the full growing season.

        This does NOT hard-code the plant update; instead it calls
        self.plant_step_fn(x, u, d, extra_state), which you can implement
        as a wrapper around your existing forward simulation logic.

        Args:
            total_steps:
                Total number of real time steps in the season.

            full_forecast:
                Dict with full-length disturbance arrays of length total_steps:
                    "precip", "temperature", "radiation"

            x0:
                Initial state vector [h0, A0, N0, c0, P0]. If None, uses
                values from self.initial_conditions.

            C0:
                Initial cumulative vector [Cw0, Cf0, Ct0, Cr0]. If None, starts
                from zeros.

        Returns:
            trajectories:
                Dict containing time series for states and controls:
                    "h", "A", "N", "c", "P", "uW", "uF"
        """
        x0 = np.array([
            self.initial_conditions.h0,
            self.initial_conditions.A0,
            self.initial_conditions.N0,
            self.initial_conditions.c0,
            self.initial_conditions.P0,
        ])
        C0 = np.zeros(4)

        x = x0.copy()
        C = C0.copy()
        extra_state = None  # whatever your plant_step_fn needs

        h_hist = np.zeros(self.model_params.total_time_steps + 1)
        A_hist = np.zeros(self.model_params.total_time_steps + 1)
        N_hist = np.zeros(self.model_params.total_time_steps + 1)
        c_hist = np.zeros(self.model_params.total_time_steps + 1)
        P_hist = np.zeros(self.model_params.total_time_steps + 1)
        uW_hist = np.zeros(self.model_params.total_time_steps)
        uF_hist = np.zeros(self.model_params.total_time_steps)

        h_hist[0], A_hist[0], N_hist[0], c_hist[0], P_hist[0] = x

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

        for k in range(total_steps):
            print(f"Step {k}/{total_steps}")
            # Build a local forecast window for the horizon
            k_end = min(k + self.N, total_steps)
            N_loc = k_end - k

            # If near the end, we shrink the horizon
            forecasted_precip      = precipitation[k:k_end]
            forecasted_temperature = temperature[k:k_end]
            forecasted_radiation   = radiation[k:k_end]

            # If horizon shorter than N, pad with last value to match N
            if N_loc < self.N:
                pad = self.N - N_loc
                forecasted_precip      = np.pad(forecasted_precip,      (0, pad), mode="edge")
                forecasted_temperature = np.pad(forecasted_temperature, (0, pad), mode="edge")
                forecasted_radiation   = np.pad(forecasted_radiation,   (0, pad), mode="edge")

            forecast = {
                "precipitation": forecasted_precip,
                "temperature":   forecasted_temperature,
                "radiation":     forecasted_radiation,
            }

            # Solve CFTOC for this horizon
            uW_opt, uF_opt = self.solve_cftoc(
                x0=x,
                C0=C,
                forecast=forecast,
            )

            # Apply first control
            uW_k = uW_opt[0]
            uF_k = uF_opt[0]

            # Disturbance at real step k
            d_k = {
                "precipitation": precipitation[k],
                "temperature":   temperature[k],
                "radiation":   radiation[k],
            }

            # Advance plant one step using your plant integrator
            x_next, C_next, extra_state = self.plant_step_fn(
                x=x,
                u=np.array([uW_k, uF_k]),
                d=d_k,
                C=C,
                extra_state=extra_state,
            )

            # Record
            uW_hist[k] = uW_k
            uF_hist[k] = uF_k

            h_hist[k + 1] = x_next[0]
            A_hist[k + 1] = x_next[1]
            N_hist[k + 1] = x_next[2]
            c_hist[k + 1] = x_next[3]
            P_hist[k + 1] = x_next[4]

            # Update for next step
            x = x_next
            C = C_next

        return {
            "h":  h_hist,
            "A":  A_hist,
            "N":  N_hist,
            "c":  c_hist,
            "P":  P_hist,
            "uW": uW_hist,
            "uF": uF_hist,
        }


    def default_plant_step_fn(mpc, x, u, d, C, extra_state):
        """
        One-step plant update for the MPC loop.

        Args:
            mpc:
                The MPC object (to pull dt, parameters, kernels, etc.).

            x:
                Current state vector [h, A, N, c, P] at step k.

            u:
                Control vector [uW, uF] at step k.

            d:
                Dict with disturbances at step k:
                    {
                        "precip":      float,
                        "temperature": float,
                        "radiation":   float
                    }

            C:
                Cumulative delayed vector [Cw, Cf, Ct, Cr] at step k.

            extra_state:
                Dict containing FIR histories:
                    {
                        "w_hist": np.ndarray of length L,
                        "f_hist": np.ndarray of length L,
                        "t_hist": np.ndarray of length L,
                        "r_hist": np.ndarray of length L,
                    }
                On the very first call, this can be None; we’ll initialize zeros.

        Returns:
            x_next:
                Next state vector [h, A, N, c, P].

            C_next:
                Next cumulative vector [Cw, Cf, Ct, Cr].

            extra_state_next:
                Updated histories for use on the next step.
        """

        # Unpack
        dt = mpc.model_params.dt

        ah = mpc.growth_rates.ah
        aA = mpc.growth_rates.aA
        aN = mpc.growth_rates.aN
        ac = mpc.growth_rates.ac
        aP = mpc.growth_rates.aP

        kh = mpc.carrying_capacities.kh
        kA = mpc.carrying_capacities.kA
        kN = mpc.carrying_capacities.kN
        kc = mpc.carrying_capacities.kc
        kP = mpc.carrying_capacities.kP

        W_typ = mpc.typical_disturbances.typical_water
        F_typ = mpc.typical_disturbances.typical_fertilizer
        T_typ = mpc.typical_disturbances.typical_temperature
        R_typ = mpc.typical_disturbances.typical_radiation

        kernel_W = mpc.kernel_W
        kernel_F = mpc.kernel_F
        kernel_T = mpc.kernel_T
        kernel_R = mpc.kernel_R
        L = mpc.L_fir

        x = np.asarray(x, dtype=float).copy()
        C = np.asarray(C, dtype=float).copy()
        u = np.asarray(u, dtype=float).copy()

        h, A, N, c, P = x
        Cw, Cf, Ct, Cr = C
        uW, uF = u

        precip      = float(d["precip"])
        temperature = float(d["temperature"])
        radiation   = float(d["radiation"])

        # Initialize FIR histories if needed
        if extra_state is None:
            extra_state = {
                "w_hist": np.zeros(L),
                "f_hist": np.zeros(L),
                "t_hist": np.zeros(L),
                "r_hist": np.zeros(L),
            }

        w_hist = extra_state["w_hist"]
        f_hist = extra_state["f_hist"]
        t_hist = extra_state["t_hist"]
        r_hist = extra_state["r_hist"]

        # ------------------------------------------------------------------
        # 1. Update FIR histories (shift left, append new sample)
        # ------------------------------------------------------------------
        # Total water input: irrigation + precip
        w_in = uW + precip
        f_in = uF
        t_in = temperature
        r_in = radiation

        w_hist = np.roll(w_hist, -1)
        f_hist = np.roll(f_hist, -1)
        t_hist = np.roll(t_hist, -1)
        r_hist = np.roll(r_hist, -1)

        w_hist[-1] = w_in
        f_hist[-1] = f_in
        t_hist[-1] = t_in
        r_hist[-1] = r_in

        # ------------------------------------------------------------------
        # 2. Compute delayed signals via FIR
        # ------------------------------------------------------------------
        W_del_k = np.dot(kernel_W, w_hist)
        F_del_k = np.dot(kernel_F, f_hist)
        T_del_k = np.dot(kernel_T, t_hist)
        R_del_k = np.dot(kernel_R, r_hist)

        # ------------------------------------------------------------------
        # 3. Update cumulative delayed quantities
        # ------------------------------------------------------------------
        Cw_next = Cw + W_del_k
        Cf_next = Cf + F_del_k
        Ct_next = Ct + T_del_k
        Cr_next = Cr + R_del_k
        C_next = np.array([Cw_next, Cf_next, Ct_next, Cr_next])

        # We need a notion of "effective time index" for the Gaussian-like nu.
        # One simple way: infer number of steps from total cumulative magnitude,
        # or just pass in a step counter separately. For now, we assume a
        # pseudo-time index grows by +1 each call. You could also keep this in
        # extra_state.
        step = extra_state.get("step", 0) + 1
        extra_state["step"] = step

        # ------------------------------------------------------------------
        # 4. Compute nuW, nuF, nuT, nuR (same structure as CFTOC)
        # ------------------------------------------------------------------
        # Use the same alpha values as in the Pyomo model
        alpha_W = 4.0
        alpha_F = 4.0
        alpha_T = 4.0
        alpha_R = 4.0

        def safe_norm(C_val, C_typ, step, eps=1e-9):
            if C_typ <= eps:
                return 1.0
            return float(C_val / (C_typ * step))

        norm_W = safe_norm(Cw_next, W_typ, step)
        norm_F = safe_norm(Cf_next, F_typ, step)
        norm_T = safe_norm(Ct_next, T_typ, step)
        norm_R = safe_norm(Cr_next, R_typ, step)

        nuW = np.exp(-alpha_W * (norm_W - 1.0) ** 2)
        nuF = np.exp(-alpha_F * (norm_F - 1.0) ** 2)
        nuT = np.exp(-alpha_T * (norm_T - 1.0) ** 2)
        nuR = np.exp(-alpha_R * (norm_R - 1.0) ** 2)

        # ------------------------------------------------------------------
        # 5. Compute adjusted growth rates and carrying capacities
        #    (parallel to what we did in solve_cftoc)
        # ------------------------------------------------------------------
        ah_hat = ah * (nuF * nuT * nuR) ** (1.0 / 3.0)
        kh_hat = kh * (nuF * nuT * nuR) ** (1.0 / 3.0)

        aA_hat = aA * (nuF * nuT * nuR) ** (1.0 / 3.0)
        kA_hat = kA * (nuW * nuF * nuT * nuR) ** (1.0 / 5.0)

        # For N, keep it simple (no nu dependence)
        aN_hat = aN
        kN_hat = kN

        ac_hat = ac * ((1.0 / (nuT + 1e-9)) * (1.0 / (nuR + 1e-9))) ** 0.5
        kc_hat = kc * (nuW * (1.0 / (nuT + 1e-9)) * (1.0 / (nuR + 1e-9))) ** (1.0 / 3.0)

        aP_hat = aP * (nuT * nuR) ** 0.5
        kP_hat = kP * (nuW * nuF * nuT * nuR) ** (1.0 / 4.0)

        # ------------------------------------------------------------------
        # 6. Forward Euler updates (to match the Pyomo dynamics)
        # ------------------------------------------------------------------
        h_next = h + dt * (ah_hat * h * (1.0 - h / (kh_hat + 1e-9)))
        A_next = A + dt * (aA_hat * A * (1.0 - A / (kA_hat + 1e-9)))
        N_next = N + dt * (aN_hat * N * (1.0 - N / (kN_hat + 1e-9)))
        c_next = c + dt * (ac_hat * c * (1.0 - c / (kc_hat + 1e-9)))
        P_next = P + dt * (aP_hat * P * (1.0 - P / (kP_hat + 1e-9)))

        # Enforce non-negativity
        h_next = max(h_next, 0.0)
        A_next = max(A_next, 0.0)
        N_next = max(N_next, 0.0)
        c_next = max(c_next, 0.0)
        P_next = max(P_next, 0.0)

        x_next = np.array([h_next, A_next, N_next, c_next, P_next])

        # Pack updated histories
        extra_state_next = {
            "w_hist": w_hist,
            "f_hist": f_hist,
            "t_hist": t_hist,
            "r_hist": r_hist,
            "step":   step,
        }

        return x_next, C_next, extra_state_next

