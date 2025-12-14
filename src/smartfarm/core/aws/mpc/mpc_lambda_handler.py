# lambda_handler.py
import json
import numpy as np
import os
import time

from mpc_lambda_helpers import mpc_with_lambda

PRICE_PER_GB_SECOND = 0.0000166667  # USD per GB-second (approx; region-specific)

def lambda_handler(event, context):
    """
    AWS Lambda entry point for batch evaluation of MPC instances. The handler
    returns the irrigation and fertilizer plans as well as the fruit biomass 
    evolution and the weights used to achieve the result.

    Args:
        event (dict):
            Invocation payload containing:
                - "weights": list of weight dictionaries, each holding
                  weight_irrigation, weight_fertilizer, weight_fruit_biomass,
                    weight_cumulative_average_water,
                    weight_cumulative_average_fertilizer,
                    weight_cumulative_average_temperature,
                    weight_cumulative_average_radiation
                - "context": dictionary of all model parameters,
                  disturbances, growth rates, carrying capacities, etc.
        context (LambdaContext):
            AWS execution context providing metadata such as the request ID.

    Returns:
        dict:
            A response payload with:
                - "statusCode": HTTP status (200 on success).
                - "body": JSON string containing {"results": [...]}.
    """
    
    start = time.time()
    weights = event["weights"]
    ctx     = event["context"]

    #print(f"[Lambda] Starting evaluation for {num_weights} weights. "
    #      f"RequestId={context.aws_request_id}")

    # Evaluate each member with Lambda
    results = {}
    for i, m in enumerate(weights):

        # Evaluate the member cost 
        result = mpc_with_lambda(m, ctx)

        # Store the result as the value to the corresponding weights key
        results[i] = {}
        results[i]["weights"] = m
        results[i]["P"] = result[0]
        results[i]["irrigation"] = result[1]
        results[i]["fertilizer"] = result[2]

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
        "body": json.dumps({"results": results})
    }
