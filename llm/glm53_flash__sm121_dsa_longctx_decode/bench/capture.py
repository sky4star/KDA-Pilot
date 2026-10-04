#!/usr/bin/env python3
"""Capture a decode-only torch-profiler trace at a given context length.

Flow: warm the radix prefix cache with an identical prompt (so the measured
request reuses the whole prefix and the trace contains decode steps, not a
100K prefill), then start_profile -> measured request -> stop_profile, then
`docker cp` the newest /tmp trace out of the server container.

  python3 capture.py --len 100000 --max-new 16
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request

DEFAULT_KEY = os.environ.get("SGLANG_API_KEY", "REDACTED_SET_SGLANG_API_KEY")
CONTAINER = os.environ.get("MOE_CONTAINER", "dspark-tp2-rank0")


def make_input_ids(n):
    return [100 + (i % 50) for i in range(n)]


def post(base, path, key, payload, timeout=1800):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode(errors="replace")


def generate(base, key, ids, max_new, timeout=1800):
    t0 = time.perf_counter()
    body = post(base, "/generate", key,
                {"input_ids": ids, "sampling_params": {"max_new_tokens": max_new, "temperature": 0.0}},
                timeout=timeout)
    return time.perf_counter() - t0, body


def newest_trace():
    out = subprocess.check_output(
        ["docker", "exec", CONTAINER, "bash", "-lc",
         "ls -t /tmp/*TP-0.trace.json.gz 2>/dev/null | head -1"]).decode().strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", type=int, required=True)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--host", default="http://10.100.8.1:8940")
    ap.add_argument("--key", default=DEFAULT_KEY)
    ap.add_argument("--dest", default="/home/spark/profiles/longctx")
    ap.add_argument("--tag", default="overlay")
    args = ap.parse_args()

    ids = make_input_ids(args.len)
    before = newest_trace()
    print(f"len={args.len} warmup (fills radix cache)...", flush=True)
    t, warm = generate(args.host, args.key, ids, 2)
    print(f"  warmup wall={t:.1f}s resp={warm[:160]}", flush=True)
    time.sleep(1.0)

    print("start_profile", flush=True)
    post(args.host, "/start_profile", args.key, {"record_shapes": False, "detailed_annotations": True})
    time.sleep(0.5)

    print(f"measured request max_new={args.max_new}...", flush=True)
    t, body = generate(args.host, args.key, ids, args.max_new)
    try:
        m = json.loads(body)["meta_info"]
    except Exception:
        m = {}
    print(f"  wall={t:.2f}s e2e={m.get('e2e_latency',0)/1000:.2f}s completion={m.get('completion_tokens')} "
          f"cached={m.get('cached_tokens')} accept={m.get('spec_accept_length')}", flush=True)
    time.sleep(1.5)

    print("stop_profile", flush=True)
    post(args.host, "/stop_profile", args.key, {})
    time.sleep(3.0)

    tr = newest_trace()
    if not tr or tr == before:
        print(f"WARN: no new trace (before={before} now={tr})", flush=True)
        return
    os.makedirs(args.dest, exist_ok=True)
    name = f"L{args.len}_{args.tag}.trace.json.gz"
    subprocess.check_call(["docker", "cp", f"{CONTAINER}:{tr}", os.path.join(args.dest, name)])
    print(f"copied -> {os.path.join(args.dest, name)} ({os.path.getsize(os.path.join(args.dest, name))} bytes)", flush=True)


if __name__ == "__main__":
    main()
