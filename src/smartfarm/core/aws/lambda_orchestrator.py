# lambda_orchestrator.py
import json, boto3, numpy as np
from ga_core import fitness, GAContext

s3 = boto3.client("s3")  # S3 is global; bucket is in us-west-1

def _read_json_s3(uri):
    bucket, key = uri[5:].split("/", 1)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())

def _write_json_s3(uri, payload):
    bucket, key = uri[5:].split("/", 1)
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(payload).encode(), ContentType="application/json")

def _np(x): return np.array(x, dtype=float)

def s3_uri(key): return f"s3://{BUCKET}/{key}"

def upload_json_to_s3(obj, key):
    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(obj).encode("utf-8"),
                  ContentType="application/json")

def list_results(prefix):
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return [c["Key"] for c in resp.get("Contents", [])] if "Contents" in resp else []

# notebook_orchestrator.py  — Lambda version
import time, json, boto3, numpy as np
from botocore.exceptions import ClientError

# discover your account id (for convenience; not strictly required)
sts = boto3.client("sts")

s3  = boto3.client("s3",     region_name=AWS_REGION)     # bucket is in us-west-1
lam = boto3.client("lambda", region_name=AWS_REGION)     # your Lambda is in us-west-1



def make_ctx_dict_from_locals():
    # Build GAContext-compatible dict from variables you already have in the notebook
    return {
        "dt": dt, "simulation_hours": simulation_hours, "total_time_steps": total_time_steps,
        "h0": h0, "A0": A0, "N0": N0, "c0": c0, "P0": P0,
        "precipitation": list(map(float, precipitation)),
        "temperature":   list(map(float, temperature)),
        "radiation":     list(map(float, radiation)),
        "WC_opt": WC_opt, "FC_opt": FC_opt, "T_typ": T_typ, "R_typ": R_typ,
        "ah": ah, "aA": aA, "aN": aN, "ac": ac, "aP": aP,
        "kh": kh, "kA": kA, "kN": kN, "kc": kc, "kP": kP,
        "w1": w1, "w2": w2,
    }

def evaluate_population_lambda_chunked(population: np.ndarray, generation: int,
                                       chunk_size: int = 25, max_inflight: int = 500) -> np.ndarray:
    n = len(population)

    # 1) Upload spec once
    spec_key = f"{ROOT_PREFIX}/spec/gen_{generation}.json"
    spec = {"population": [list(map(float, row)) for row in population],
            "ctx": make_ctx_dict_from_locals()}
    upload_json_to_s3(spec, spec_key)
    spec_s3 = s3_uri(spec_key)
    out_prefix = f"s3://{BUCKET}/{ROOT_PREFIX}/results"

    # 2) Determine chunk starts
    chunks = [(start, min(chunk_size, n - start)) for start in range(0, n, chunk_size)]
    result_prefix = f"{ROOT_PREFIX}/results/gen_{generation}/"
    want = {f"{result_prefix}part_{start}.json" for (start, _) in chunks}

    # 3) Fire async invokes with throttling
    payload_base = {"spec_s3": spec_s3, "out_prefix": out_prefix, "generation": generation}
    submitted = 0
    inflight = 0
    while submitted < len(chunks):
        while inflight < max_inflight and submitted < len(chunks):
            start, count = chunks[submitted]
            ev = dict(payload_base, start=start, count=count)
            lam.invoke(FunctionName=FUNCTION_NAME, InvocationType="Event",
                       Payload=json.dumps(ev).encode("utf-8"))
            submitted += 1
            inflight += 1
        time.sleep(0.25)
        have = set(list_results(result_prefix))
        done = len(want.intersection(have))
        inflight = max(0, submitted - done)
    print("Submitted events this gen:", submitted)

    # 4) Wait until all chunk outputs exist
    have = set()
    backoff = 0.25
    while have != want:
        have = set(list_results(result_prefix)).intersection(want)
        time.sleep(backoff)
        backoff = min(backoff * 1.5, 2.0)

    # 5) Assemble full cost vector
    costs = np.zeros(n, dtype=float)
    for (start, _count) in chunks:
        rec = read_json_s3_key(f"{result_prefix}part_{start}.json")
        for i, c in zip(rec["indices"], rec["costs"]):
            costs[i] = float(c)
    return costs

def lambda_handler(event, _):
    spec_s3    = event["spec_s3"]              # s3://.../spec/gen_N.json
    out_prefix = event["out_prefix"]           # s3://.../results
    gen        = int(event["generation"])
    start      = int(event.get("start", 0))    # chunk start index
    count      = int(event.get("count", 1))    # how many to do in this chunk

    # Back-compat: support old single-index or new chunked API
    if "index" in event:
        start = int(event["index"])
        count = 1
    else:
        start = int(event.get("start", 0))
        count = int(event.get("count", 1))

    spec = _read_json_s3(spec_s3)
    pop  = [ _np(v) for v in spec["population"] ]
    c    = spec["ctx"]
    ctx  = GAContext(
        dt=c["dt"], simulation_hours=c["simulation_hours"], total_time_steps=c["total_time_steps"],
        h0=c["h0"], A0=c["A0"], N0=c["N0"], c0=c["c0"], P0=c["P0"],
        precipitation=_np(c["precipitation"]), temperature=_np(c["temperature"]), radiation=_np(c["radiation"]),
        WC_opt=c["WC_opt"], FC_opt=c["FC_opt"], T_typ=c["T_typ"], R_typ=c["R_typ"],
        ah=c["ah"], aA=c["aA"], aN=c["aN"], ac=c["ac"], aP=c["aP"],
        kh=c["kh"], kA=c["kA"], kN=c["kN"], kc=c["kc"], kP=c["kP"],
        w1=c["w1"], w2=c["w2"]
    )

    end = min(start + count, len(pop))
    indices, costs = [], []
    for i in range(start, end):
        indices.append(i)
        costs.append(float(fitness(pop[i], ctx)))

    _write_json_s3(f"{out_prefix}/gen_{gen}/part_{start}.json",
                   {"indices": indices, "costs": costs})
    return {"ok": True, "start": start, "end": end, "n": len(indices)}
