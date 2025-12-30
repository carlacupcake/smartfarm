#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// Helper functions
static inline double clip(
    double x,
    double lo,
    double hi
) {
    return std::min(std::max(x, lo), hi);
}

static inline double logistic_step(
    double x,
    double a,
    double k,
    double dt,
    double eps = 1e-12
) {
    k = std::max(k, eps);
    x = std::max(x, eps);
    const double exp_term = std::exp(-a * dt);
    return k / (1.0 + (k / x - 1.0) * exp_term);
}


// mode='split': expand hourly_array into per-time-step array by evenly splitting each hour evenly across substeps
static std::vector<double> get_sim_inputs_from_hourly(
    const std::vector<double>& hourly,
    double dt,
    int simulation_hours,
    int total_time_steps
) {
    // total_time_steps is expected to match simulation_hours/dt, but just in case...
    const double steps_per_hour_f = 1.0 / dt;
    const int steps_per_hour = std::max(1, (int)std::llround(steps_per_hour_f));

    std::vector<double> out((size_t)total_time_steps, 0.0);

    // Fill hour by hour; split the hourly value evenly across substeps.
    const int hours_to_use = std::min(simulation_hours, (int)hourly.size());
    int idx = 0;

    for (int h = 0; h < hours_to_use && idx < total_time_steps; ++h) {
        const double per_step = hourly[h] / (double)steps_per_hour;
        for (int s = 0; s < steps_per_hour && idx < total_time_steps; ++s) {
            out[(size_t)idx++] = per_step;
        }
    }

    // If rounding mismatch leaves remaining steps, repeat last hour’s split (or zeros if none)
    const double last = (hours_to_use > 0) ? (hourly[hours_to_use - 1] / (double)steps_per_hour) : 0.0;
    while (idx < total_time_steps) out[(size_t)idx++] = last;

    return out;
}

// Ring buffer dot product: kernel[0..h-1] dot history (oldest->newest)
// We store history in a ring; "head" is index of oldest element.
static inline double ring_dot(
    const std::vector<double>& kernel,
    const std::vector<double>& ring,
    int head)
{
    const int n = (int)kernel.size();
    double acc = 0.0;
    // ring[(head + i) % n] gives i-th oldest
    for (int i = 0; i < n; ++i) {
        acc += kernel[(size_t)i] * ring[(size_t)((head + i) % n)];
    }
    return acc;
}

static std::vector<double> dict_get_vec(const py::dict& d, const std::string& key) {
    if (!d.contains(key.c_str())) {
        throw std::runtime_error("Missing context key: " + key);
    }
    // Accept Python list or numpy array
    py::object obj = d[key.c_str()];
    // Convert to numpy array then to std::vector
    py::array arr = py::array::ensure(obj);
    if (!arr) throw std::runtime_error("Context key '" + key + "' could not be converted to numpy array.");

    if (arr.ndim() != 1) throw std::runtime_error("Context key '" + key + "' must be 1D.");
    auto buf = arr.request();
    const ssize_t n = buf.shape[0];

    std::vector<double> out((size_t)n);
    // assume convertible to double
    const char* p = static_cast<const char*>(buf.ptr);
    // handle common numeric dtypes by using pybind to cast per element (safe, a bit slower)
    // If you want max speed, enforce float64 in Python and memcpy.
    for (ssize_t i = 0; i < n; ++i) {
        out[(size_t)i] = py::cast<double>(arr[py::int_(i)]);
    }
    return out;
}

static inline double dict_get_double(const py::dict& d, const std::string& key) {
    if (!d.contains(key.c_str())) throw std::runtime_error("Missing context key: " + key);
    return py::cast<double>(d[key.c_str()]);
}

static inline int dict_get_int(const py::dict& d, const std::string& key) {
    if (!d.contains(key.c_str())) throw std::runtime_error("Missing context key: " + key);
    return py::cast<int>(d[key.c_str()]);
}

// Cost function evaluator class
class Evaluator {
public:
    explicit Evaluator(const py::dict& context)
    : dt(dict_get_double(context,            "dt")),
      total_time_steps(dict_get_int(context, "total_time_steps")),
      simulation_hours(dict_get_int(context, "simulation_hours")),

      alpha(dict_get_double(context,                "alpha")),
      beta_divergence(dict_get_double(context,      "beta_divergence")),
      beta_nutrient_factor(dict_get_double(context, "beta_nutrient_factor")),
      epsilon(dict_get_double(context,              "epsilon")),

      // typical disturbances
      W_typ(dict_get_double(context, "W_typ")),
      F_typ(dict_get_double(context, "F_typ")),
      T_typ(dict_get_double(context, "T_typ")),
      R_typ(dict_get_double(context, "R_typ")),

      // initial conditions
      h0(dict_get_double(context, "h0")),
      A0(dict_get_double(context, "A0")),
      N0(dict_get_double(context, "N0")),
      c0(dict_get_double(context, "c0")),
      P0(dict_get_double(context, "P0")),

      // growth rates
      ah(dict_get_double(context, "ah")),
      aA(dict_get_double(context, "aA")),
      aN(dict_get_double(context, "aN")),
      ac(dict_get_double(context, "ac")),
      aP(dict_get_double(context, "aP")),

      // carrying capacities
      kh(dict_get_double(context, "kh")),
      kA(dict_get_double(context, "kA")),
      kN(dict_get_double(context, "kN")),
      kc(dict_get_double(context, "kc")),
      kP(dict_get_double(context, "kP")),

      // GA weights
      w_height(dict_get_double(context,    "weight_height")),
      w_leaf_area(dict_get_double(context, "weight_leaf_area")),
      w_fruit(dict_get_double(context,     "weight_fruit_biomass")),
      w_irrig(dict_get_double(context,     "weight_irrigation")),
      w_fert(dict_get_double(context,      "weight_fertilizer"))
    {
        // disturbances (hourly)
        hourly_precipitation = dict_get_vec(context, "hourly_precipitation");
        hourly_temperature   = dict_get_vec(context, "hourly_temperature");
        hourly_radiation     = dict_get_vec(context, "hourly_radiation");

        // FIR kernels + horizons
        kernel_W = dict_get_vec(context, "kernel_W");
        kernel_F = dict_get_vec(context, "kernel_F");
        kernel_T = dict_get_vec(context, "kernel_T");
        kernel_R = dict_get_vec(context, "kernel_R");

        fir_W = dict_get_int(context, "fir_horizon_W");
        fir_F = dict_get_int(context, "fir_horizon_F");
        fir_T = dict_get_int(context, "fir_horizon_T");
        fir_R = dict_get_int(context, "fir_horizon_R");

        if (fir_W <= 0 || fir_F <= 0 || fir_T <= 0 || fir_R <= 0) {
            throw std::runtime_error("FIR horizons must be positive.");
        }

        // Truncate kernels to horizons
        if ((int)kernel_W.size() < fir_W) throw std::runtime_error("kernel_W shorter than fir_horizon_W");
        if ((int)kernel_F.size() < fir_F) throw std::runtime_error("kernel_F shorter than fir_horizon_F");
        if ((int)kernel_T.size() < fir_T) throw std::runtime_error("kernel_T shorter than fir_horizon_T");
        if ((int)kernel_R.size() < fir_R) throw std::runtime_error("kernel_R shorter than fir_horizon_R");

        kernel_W.resize((size_t)fir_W);
        kernel_F.resize((size_t)fir_F);
        kernel_T.resize((size_t)fir_T);
        kernel_R.resize((size_t)fir_R);

        // Pre-expand disturbances to per-step arrays once (shared across members)
        precipitation = get_sim_inputs_from_hourly(hourly_precipitation, dt, simulation_hours, total_time_steps);
        temperature   = get_sim_inputs_from_hourly(hourly_temperature,   dt, simulation_hours, total_time_steps);
        radiation     = get_sim_inputs_from_hourly(hourly_radiation,     dt, simulation_hours, total_time_steps);

        // Scratch buffers can be reused per member
        alloc_scratch();
    }

    double evaluate_member(py::array_t<double, py::array::c_style | py::array::forcecast> values_row) {
        auto v = values_row.request();
        if (v.ndim != 1 || v.shape[0] < 4) {
            throw std::runtime_error("values_row must be shape (>=4,).");
        }
        const double irrigation_frequency  = ((double*)v.ptr)[0];
        const double irrigation_amount     = ((double*)v.ptr)[1];
        const double fertilizer_frequency  = ((double*)v.ptr)[2];
        const double fertilizer_amount     = ((double*)v.ptr)[3];

        return evaluate_member_impl(irrigation_frequency, irrigation_amount,
                                    fertilizer_frequency, fertilizer_amount);
    }

    py::array_t<double> evaluate_population(py::array_t<double, py::array::c_style | py::array::forcecast> values_matrix) {
        auto m = values_matrix.request();
        if (m.ndim != 2 || m.shape[1] < 4) {
            throw std::runtime_error("values_matrix must be shape (num_members, >=4).");
        }
        const ssize_t n = m.shape[0];
        const ssize_t p = m.shape[1];
        const double* data = (const double*)m.ptr;

        py::array_t<double> out(n);
        auto o = out.request();
        double* outp = (double*)o.ptr;

        for (ssize_t i = 0; i < n; ++i) {
            const double irrigation_frequency  = data[i*p + 0];
            const double irrigation_amount     = data[i*p + 1];
            const double fertilizer_frequency  = data[i*p + 2];
            const double fertilizer_amount     = data[i*p + 3];
            outp[i] = evaluate_member_impl(irrigation_frequency, irrigation_amount,
                                           fertilizer_frequency, fertilizer_amount);
        }

        return out;
    }

private:
    // Context scalars
    double dt;
    int total_time_steps;
    int simulation_hours;

    double alpha, beta_divergence, beta_nutrient_factor, epsilon;

    double W_typ, F_typ, T_typ, R_typ;

    double h0, A0, N0, c0, P0;

    double ah, aA, aN, ac, aP;
    double kh, kA, kN, kc, kP;

    double w_height, w_leaf_area, w_fruit, w_irrig, w_fert;

    // Context arrays
    std::vector<double> hourly_precipitation, hourly_temperature, hourly_radiation;
    std::vector<double> precipitation, temperature, radiation;

    std::vector<double> kernel_W, kernel_F, kernel_T, kernel_R;
    int fir_W, fir_F, fir_T, fir_R;

    // Scratch buffers
    std::vector<double> irrigation, fertilizer;
    std::vector<double> h, A, N, c, P;
    std::vector<double> delayed_W, delayed_F, delayed_T, delayed_R;
    std::vector<double> cum_W, cum_F, cum_T, cum_R;
    std::vector<double> cumdiv_W, cumdiv_F, cumdiv_T, cumdiv_R;
    std::vector<double> nuW, nuF, nuT, nuR;

    // Ring histories
    std::vector<double> hist_W, hist_F, hist_T, hist_R;
    int head_W = 0, head_F = 0, head_T = 0, head_R = 0;

    void alloc_scratch() {
        irrigation.assign((size_t)total_time_steps, 0.0);
        fertilizer.assign((size_t)total_time_steps, 0.0);

        h.assign((size_t)total_time_steps, h0);
        A.assign((size_t)total_time_steps, A0);
        N.assign((size_t)total_time_steps, N0);
        c.assign((size_t)total_time_steps, c0);
        P.assign((size_t)total_time_steps, P0);

        delayed_W.assign((size_t)total_time_steps, 0.0);
        delayed_F.assign((size_t)total_time_steps, 0.0);
        delayed_T.assign((size_t)total_time_steps, 0.0);
        delayed_R.assign((size_t)total_time_steps, 0.0);

        cum_W.assign((size_t)total_time_steps, 0.0);
        cum_F.assign((size_t)total_time_steps, 0.0);
        cum_T.assign((size_t)total_time_steps, 0.0);
        cum_R.assign((size_t)total_time_steps, 0.0);

        cumdiv_W.assign((size_t)total_time_steps, 0.0);
        cumdiv_F.assign((size_t)total_time_steps, 0.0);
        cumdiv_T.assign((size_t)total_time_steps, 0.0);
        cumdiv_R.assign((size_t)total_time_steps, 0.0);

        nuW.assign((size_t)total_time_steps, 0.0);
        nuF.assign((size_t)total_time_steps, 0.0);
        nuT.assign((size_t)total_time_steps, 0.0);
        nuR.assign((size_t)total_time_steps, 0.0);

        // histories initialized to typical values like Python :contentReference[oaicite:7]{index=7}
        hist_W.assign((size_t)fir_W, W_typ);
        hist_F.assign((size_t)fir_F, F_typ);
        hist_T.assign((size_t)fir_T, T_typ);
        hist_R.assign((size_t)fir_R, R_typ);

        head_W = head_F = head_T = head_R = 0;
    }

    void build_controls(double irrigation_frequency, double irrigation_amount,
                        double fertilizer_frequency, double fertilizer_amount)
    {
        // Mirror Python:
        // step_if = max(1, ceil(irrigation_frequency)); hourly_irrigation[::step_if] = irrigation_amount
        // then get_sim_inputs_from_hourly(..., mode='split')
        const int hours = simulation_hours;

        std::vector<double> hourly_irrig((size_t)hours, 0.0);
        std::vector<double> hourly_fert((size_t)hours, 0.0);

        const int step_if = std::max(1, (int)std::ceil(irrigation_frequency));
        const int step_ff = std::max(1, (int)std::ceil(fertilizer_frequency));

        for (int h = 0; h < hours; h += step_if) hourly_irrig[(size_t)h] = irrigation_amount;
        for (int h = 0; h < hours; h += step_ff) hourly_fert[(size_t)h]  = fertilizer_amount;

        irrigation = get_sim_inputs_from_hourly(hourly_irrig, dt, simulation_hours, total_time_steps);
        fertilizer = get_sim_inputs_from_hourly(hourly_fert, dt, simulation_hours, total_time_steps);
    }

    // Main per-member evaluation
    double evaluate_member_impl(double irrigation_frequency, double irrigation_amount,
                                double fertilizer_frequency, double fertilizer_amount)
    {
        // reset scratch
        std::fill(h.begin(), h.end(), h0);
        std::fill(A.begin(), A.end(), A0);
        std::fill(N.begin(), N.end(), N0);
        std::fill(c.begin(), c.end(), c0);
        std::fill(P.begin(), P.end(), P0);

        std::fill(delayed_W.begin(), delayed_W.end(), 0.0);
        std::fill(delayed_F.begin(), delayed_F.end(), 0.0);
        std::fill(delayed_T.begin(), delayed_T.end(), 0.0);
        std::fill(delayed_R.begin(), delayed_R.end(), 0.0);

        std::fill(cum_W.begin(), cum_W.end(), 0.0);
        std::fill(cum_F.begin(), cum_F.end(), 0.0);
        std::fill(cum_T.begin(), cum_T.end(), 0.0);
        std::fill(cum_R.begin(), cum_R.end(), 0.0);

        std::fill(cumdiv_W.begin(), cumdiv_W.end(), 0.0);
        std::fill(cumdiv_F.begin(), cumdiv_F.end(), 0.0);
        std::fill(cumdiv_T.begin(), cumdiv_T.end(), 0.0);
        std::fill(cumdiv_R.begin(), cumdiv_R.end(), 0.0);

        std::fill(nuW.begin(), nuW.end(), 0.0);
        std::fill(nuF.begin(), nuF.end(), 0.0);
        std::fill(nuT.begin(), nuT.end(), 0.0);
        std::fill(nuR.begin(), nuR.end(), 0.0);

        std::fill(hist_W.begin(), hist_W.end(), W_typ);
        std::fill(hist_F.begin(), hist_F.end(), F_typ);
        std::fill(hist_T.begin(), hist_T.end(), T_typ);
        std::fill(hist_R.begin(), hist_R.end(), R_typ);
        head_W = head_F = head_T = head_R = 0;

        build_controls(irrigation_frequency, irrigation_amount,
                       fertilizer_frequency, fertilizer_amount);

        // Run loop
        for (int t = 1; t < total_time_steps - 1; ++t) {
            const double W = irrigation[(size_t)t];
            const double F = fertilizer[(size_t)t];

            const double S = precipitation[(size_t)t];
            const double T = temperature[(size_t)t];
            const double R = radiation[(size_t)t];

            // Update ring histories with newest sample
            // Replace oldest at head, then advance head
            hist_W[(size_t)head_W] = W + S;
            hist_F[(size_t)head_F] = F;
            hist_T[(size_t)head_T] = T;
            hist_R[(size_t)head_R] = R;

            head_W = (head_W + 1) % fir_W;
            head_F = (head_F + 1) % fir_F;
            head_T = (head_T + 1) % fir_T;
            head_R = (head_R + 1) % fir_R;

            // Delayed signals via dot(kernel, history)
            delayed_W[(size_t)t] = ring_dot(kernel_W, hist_W, head_W);
            delayed_F[(size_t)t] = ring_dot(kernel_F, hist_F, head_F);
            delayed_T[(size_t)t] = ring_dot(kernel_T, hist_T, head_T);
            delayed_R[(size_t)t] = ring_dot(kernel_R, hist_R, head_R);

            // Cumulative delayed values
            cum_W[(size_t)(t+1)] = cum_W[(size_t)t] + delayed_W[(size_t)t];
            cum_F[(size_t)(t+1)] = cum_F[(size_t)t] + delayed_F[(size_t)t];
            cum_T[(size_t)(t+1)] = cum_T[(size_t)t] + delayed_T[(size_t)t];
            cum_R[(size_t)(t+1)] = cum_R[(size_t)t] + delayed_R[(size_t)t];

            // Anomalies (max(|typ*(t-1) - cum[t]|/(typ*t+eps), eps))
            const double water_anom =
                std::max(std::abs(W_typ * (double)(t-1) - cum_W[(size_t)t]) / (W_typ * (double)t + epsilon), epsilon);
            const double fert_anom =
                std::max(std::abs(F_typ * (double)(t-1) - cum_F[(size_t)t]) / (F_typ * (double)t + epsilon), epsilon);
            const double temp_anom =
                std::max(std::abs(T_typ * (double)(t-1) - cum_T[(size_t)t]) / (T_typ * (double)t + epsilon), epsilon);
            const double rad_anom =
                std::max(std::abs(R_typ * (double)(t-1) - cum_R[(size_t)t]) / (R_typ * (double)t + epsilon), epsilon);

            // Recursive cumulative divergence update
            cumdiv_W[(size_t)t] = beta_divergence * cumdiv_W[(size_t)(t-1)] + (1.0 - beta_divergence) * water_anom;
            cumdiv_F[(size_t)t] = beta_divergence * cumdiv_F[(size_t)(t-1)] + (1.0 - beta_divergence) * fert_anom;
            cumdiv_T[(size_t)t] = beta_divergence * cumdiv_T[(size_t)(t-1)] + (1.0 - beta_divergence) * temp_anom;
            cumdiv_R[(size_t)t] = beta_divergence * cumdiv_R[(size_t)(t-1)] + (1.0 - beta_divergence) * rad_anom;

            // Raw nutrient factors then EMA smoothing
            const double nuW_raw = std::exp(-alpha * cumdiv_W[(size_t)t]);
            const double nuF_raw = std::exp(-alpha * cumdiv_F[(size_t)t]);
            const double nuT_raw = std::exp(-alpha * cumdiv_T[(size_t)t]);
            const double nuR_raw = std::exp(-alpha * cumdiv_R[(size_t)t]);

            nuW[(size_t)t] = (1.0 - beta_nutrient_factor) * nuW[(size_t)(t-1)] + beta_nutrient_factor * nuW_raw;
            nuF[(size_t)t] = (1.0 - beta_nutrient_factor) * nuF[(size_t)(t-1)] + beta_nutrient_factor * nuF_raw;
            nuT[(size_t)t] = (1.0 - beta_nutrient_factor) * nuT[(size_t)(t-1)] + beta_nutrient_factor * nuT_raw;
            nuR[(size_t)t] = (1.0 - beta_nutrient_factor) * nuR[(size_t)(t-1)] + beta_nutrient_factor * nuR_raw;

            // Instantaneous adjusted growth rates and carrying capacities
            // (with clipping + lower bound at current state)
            const double g_1_3 = std::pow(nuF[(size_t)t] * nuT[(size_t)t] * nuR[(size_t)t], 1.0/3.0);
            const double g_1_2_TR = std::pow(nuT[(size_t)t] * nuR[(size_t)t], 1.0/2.0);

            const double ah_hat = clip(ah * g_1_3, 0.0, 2.0 * ah);
            const double aA_hat = clip(aA * g_1_3, 0.0, 2.0 * aA);
            const double aN_hat = clip(aN,         0.0, 2.0 * aN);

            const double ac_hat = clip(ac * std::pow((1.0/nuT[(size_t)t]) * (1.0/nuR[(size_t)t]), 1.0/2.0), 0.0, 2.0 * ac);
            const double aP_hat = clip(aP * g_1_2_TR, 0.0, 2.0 * aP);

            const double kh_hat = clip(kh * g_1_3, h[(size_t)t], 2.0 * kh);

            const double kA_hat = clip(
                kA * std::pow(nuW[(size_t)t] * nuF[(size_t)t] * nuT[(size_t)t] * nuR[(size_t)t] * (kh_hat/kh), 1.0/5.0),
                A[(size_t)t], 2.0 * kA
            );

            const double kN_hat = clip(kN * g_1_2_TR, N[(size_t)t], 2.0 * kN);

            const double kc_hat = clip(
                kc * std::pow(nuW[(size_t)t] * (1.0/nuT[(size_t)t]) * (1.0/nuR[(size_t)t]), 1.0/3.0),
                c[(size_t)t], 2.0 * kc
            );

            const double kP_hat = clip(
                kP * std::pow(
                    nuW[(size_t)t] * nuF[(size_t)t] * nuT[(size_t)t] * nuR[(size_t)t]
                    * (kh_hat/kh) * (kA_hat/kA) * (kc_hat/kc),
                    1.0/7.0
                ),
                P[(size_t)t], 2.0 * kP
            );

            // Logistic-style updates
            h[(size_t)(t+1)] = logistic_step(h[(size_t)t], ah_hat, kh_hat, dt);
            A[(size_t)(t+1)] = logistic_step(A[(size_t)t], aA_hat, kA_hat, dt);
            N[(size_t)(t+1)] = logistic_step(N[(size_t)t], aN_hat, kN_hat, dt);
            c[(size_t)(t+1)] = logistic_step(c[(size_t)t], ac_hat, kc_hat, dt);
            P[(size_t)(t+1)] = logistic_step(P[(size_t)t], aP_hat, kP_hat, dt);
        }

        // Cost (negative revenue)
        const double profit   = w_fruit * P.back() + w_height * h.back() + w_leaf_area * A.back();
        double sum_irrig = 0.0, sum_fert = 0.0;
        for (int t = 0; t < total_time_steps; ++t) {
            sum_irrig += irrigation[(size_t)t];
            sum_fert  += fertilizer[(size_t)t];
        }
        const double expenses = (w_irrig * sum_irrig) + (w_fert * sum_fert);
        const double revenue  = profit - expenses;
        const double cost     = -revenue;
        return cost;
    }
};

PYBIND11_MODULE(_core, m) {
    m.doc() = "SmartFarm GA member cost evaluator (C++/pybind11)";

    py::class_<Evaluator>(m, "Evaluator")
        .def(py::init<const py::dict&>(), py::arg("context"))
        .def("evaluate_member", &Evaluator::evaluate_member, py::arg("values_row"),
             "Evaluate one member: values_row shape (>=4,). Returns cost.")
        .def("evaluate_population", &Evaluator::evaluate_population, py::arg("values_matrix"),
             "Evaluate many members: values_matrix shape (n,>=4). Returns costs shape (n,).");
}
