# lambda_handler.py
import json
import os
import time
from lambda_member import member_get_cost_with_lambda

PRICE_PER_GB_SECOND = 0.0000166667  # USD per GB-second (approx; region-specific)

def lambda_handler(event, context):
    start = time.time()

    members = event["members"]
    sim_ctx = event["sim_context"]

    num_members = len(members)
    print(f"[Lambda] Starting evaluation for {num_members} members. "
          f"RequestId={context.aws_request_id}")

    costs = []
    for idx, m in enumerate(members):
        cost = member_get_cost_with_lambda(m, sim_ctx)
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

