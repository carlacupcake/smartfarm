# mpc_lambda_handler.py
import json
import boto3
import traceback

from mpc_helpers import mpc_from_context

s3 = boto3.client("s3")


def lambda_handler(event, context):
    # Unpack event
    run_id       = event["run_id"]
    idx          = event["idx"]
    weights_dict = event["weights_dict"]
    ctx          = event["context"]

    # Where results go
    data_bucket = ctx["data_bucket"]          # e.g. "smartfarm-mpc-results"
    data_prefix = ctx.get("data_prefix", "runs")

    stage = "start"

    try:
        stage = "mpc_call"

        mpc_out = mpc_from_context(weights_dict, ctx)
        # Expecting something like:
        # {
        #   "final_fruit_biomass": ...,
        #   "sum_irrigation": ...,
        #   "sum_fertilizer": ...,
        #   "mpc_status": "optimal" / "infeasible" / ...
        # }

        result_doc = {
            "status":              "ok", # the *Lambda* completed without exception
            "stage":               "completed_mpc",
            "run_id":              run_id,
            "idx":                 idx,
            "weights":             weights_dict,
            "final_fruit_biomass": float(mpc_out["final_fruit_biomass"]),
            "sum_irrigation":      float(mpc_out["sum_irrigation"]),
            "sum_fertilizer":      float(mpc_out["sum_fertilizer"]),
            "mpc_status":          mpc_out.get("mpc_status", "unknown"),
        }

    except Exception as e:
        # Anything that blows up before we compute outputs lands here
        result_doc = {
            "status":        "error",       # the *Lambda* raised an exception
            "failed_stage":  stage,   # e.g. "start", "mpc_call"
            "run_id":        run_id,
            "idx":           idx,
            "weights":       weights_dict,
            "error_type":    type(e).__name__,
            "error_message": str(e),
            "traceback":     traceback.format_exc(),
        }

    key = f"{data_prefix}/{run_id}/result_{idx:03d}.json"

    s3.put_object(
        Bucket=data_bucket,
        Key=key,
        Body=json.dumps(result_doc).encode("utf-8"),
        ContentType="application/json",
    )

    return {
        "statusCode": 200,
        "body": json.dumps({
            "written_key": key,
            "status": result_doc["status"],
        }),
    }
