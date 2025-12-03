# lambda_handler.py
import json
import os
import time
from lambda_member import get_cost_with_lambda, get_closed_form_cost_with_lambda

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
                - "sim_context": dictionary of all model parameters,
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
    sim_ctx = event["sim_context"]

    num_members = len(members)
    print(f"[Lambda] Starting evaluation for {num_members} members. "
          f"RequestId={context.aws_request_id}")

    costs = []
    for idx, m in enumerate(members):
        # Evaluate the member cost using the Forward Euler simulation
        #cost = get_cost_with_lambda(m, sim_ctx)

        # Evaluate the member cost using the closed-form solution
        cost = get_closed_form_cost_with_lambda(m, sim_ctx)

        # Store the cost
        costs.append(float(cost))
        print(f"[Lambda] Finished member {idx+1}/{num_members}")

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
