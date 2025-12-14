# lambda_handler.py
import json
import numpy as np
import os
import time
from smartfarm.core.aws.ga.ga_lambda_helpers import *

PRICE_PER_GB_SECOND = 0.0000166667  # USD per GB-second (approx; region-specific)

def lambda_handler(event, context):
    """
    AWS Lambda entry point for batch evaluation of GA members. The handler
    runs either the Forward Euler or closed-form plant-growth simulation for
    each member, logs compute usage, and returns all computed costs.

    Args:
        event (dict):
            Invocation payload containing:
                - "members": list of member dictionaries, each holding
                  decision variables under the key "values".
                - "context": dictionary of all model parameters,
                  disturbances, growth rates, carrying capacities, and
                  economic weights needed by the simulation.
        context (LambdaContext):
            AWS execution context providing metadata such as the request ID.

    Returns:
        dict:
            A response payload with:
                - "statusCode": HTTP status (200 on success).
                - "body": JSON string containing {"costs": [...]} where each
                  entry is the simulated cost for a corresponding member.
    """
    
    start = time.time()
    members = event["members"]
    ctx     = event["context"]

    #print(f"[Lambda] Starting evaluation for {num_members} members. "
    #      f"RequestId={context.aws_request_id}")

    # Perform common operations on the context before processing with Lambda
    enriched_ctx = ctx.copy()

    # Time stepping / horizon
    dt = ctx["dt"]
    total_time_steps = ctx["total_time_steps"]
    simulation_hours = ctx["simulation_hours"]

    # Nutrient absorption/metalysis sensitivities (sigmas)
    sigma_W = ctx["sigma_W"]
    sigma_F = ctx["sigma_F"]
    sigma_T = ctx["sigma_T"]
    sigma_R = ctx["sigma_R"]

    # Pre-calculate the mu values that correspond to 95% absorption for each sigma ("sensitivity")
    enriched_ctx["mu_W"] = get_mu_from_sigma(sigma_W/dt)
    enriched_ctx["mu_F"] = get_mu_from_sigma(sigma_F/dt)
    enriched_ctx["mu_T"] = get_mu_from_sigma(sigma_T/dt)
    enriched_ctx["mu_R"] = get_mu_from_sigma(sigma_R/dt)

    # Pre-calculate the convolution kernels for each disturbance type
    enriched_ctx["kernel_W"] = gaussian_kernel(enriched_ctx["mu_W"], sigma_W/dt, total_time_steps)
    enriched_ctx["kernel_F"] = gaussian_kernel(enriched_ctx["mu_F"], sigma_F/dt, total_time_steps)
    enriched_ctx["kernel_T"] = gaussian_kernel(enriched_ctx["mu_T"], sigma_T/dt, total_time_steps)
    enriched_ctx["kernel_R"] = gaussian_kernel(enriched_ctx["mu_R"], sigma_R/dt, total_time_steps)

    # Disturbances (hourly)
    hourly_precipitation = np.array(ctx["hourly_precipitation"], dtype=float)
    hourly_temperature   = np.array(ctx["hourly_temperature"],   dtype=float)
    hourly_radiation     = np.array(ctx["hourly_radiation"],     dtype=float)

    # Disturbances (time steps)
    enriched_ctx["precipitation"] = get_sim_inputs_from_hourly(hourly_precipitation, dt, simulation_hours, mode="split").tolist()
    enriched_ctx["temperature"]   = get_sim_inputs_from_hourly(hourly_temperature,   dt, simulation_hours, mode="split").tolist()
    enriched_ctx["radiation"]     = get_sim_inputs_from_hourly(hourly_radiation,     dt, simulation_hours, mode="split").tolist()

    # 4) Precompute time indices
    t_idx_0_to_N = np.arange(total_time_steps).tolist()        # 0..N-1
    t_idx_1_to_N = np.arange(1, total_time_steps + 1).tolist() # 1..N
    enriched_ctx["t_idx_0_to_N"] = t_idx_0_to_N
    enriched_ctx["t_idx_1_to_N"] = t_idx_1_to_N

    # Evaluate each member with Lambda
    costs = []
    for _, m in enumerate(members):

        # Evaluate the member cost 
        cost = get_cost_with_lambda(m, enriched_ctx)

        # Store the cost
        costs.append(float(cost))
        #print(f"[Lambda] Finished member {idx+1}/{num_members}")

    duration = time.time() - start  # seconds

    # Memory config from environment
    mem_mb = int(os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", "1024"))
    gb_seconds = (mem_mb / 1024.0) * duration
    est_cost = gb_seconds * PRICE_PER_GB_SECOND

    print(
        f"[Lambda] Done. Duration={duration:.3f}s, "
        f"Mem={mem_mb}MB, GB-s={gb_seconds:.6f}, "
        f"Est cost=${est_cost:.6f} (before free tier)"
    )

    return {
        "statusCode": 200,
        "body": json.dumps({"costs": costs})
    }
