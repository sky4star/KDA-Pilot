#!/usr/bin/env python3
"""Capture a cold-prefill torch-profiler trace (unique prompt => no cache hit)."""
import argparse, json, os, subprocess, time, urllib.request

KEY = "REDACTED_SET_SGLANG_API_KEY"
BASE = "http://10.100.8.1:8940"
CONTAINER = "dspark-tp2-rank0"


def ids_for(L, salt=99):
    return [100 + ((i * 7 + salt * 13 + i // 97) % 50) for i in range(L)]


def post(path, payload, timeout=3600):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode(errors="replace")


def newest():
    return subprocess.check_output(["docker", "exec", CONTAINER, "bash", "-lc",
        "ls -t /tmp/*TP-0.trace.json.gz 2>/dev/null | head -1"]).decode().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", type=int, required=True)
    ap.add_argument("--salt", type=int, default=99)
    args = ap.parse_args()
    before = newest()
    print("start_profile", flush=True)
    post("/start_profile", {"record_shapes": False, "detailed_annotations": True})
    time.sleep(0.3)
    t0 = time.perf_counter()
    body = post("/generate", {"input_ids": ids_for(args.len, args.salt),
                              "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}})
    print(f"  prefill wall={time.perf_counter()-t0:.1f}s", flush=True)
    time.sleep(1.0)
    print("stop_profile", flush=True)
    post("/stop_profile", {})
    time.sleep(3.0)
    tr = newest()
    dest = f"/home/spark/profiles/longctx/P{args.len}.trace.json.gz"
    if not tr or tr == before:
        print("WARN no new trace", before, tr); return
    subprocess.check_call(["docker", "cp", f"{CONTAINER}:{tr}", dest])
    print(f"copied -> {dest} ({os.path.getsize(dest)} bytes)", flush=True)


if __name__ == "__main__":
    main()
