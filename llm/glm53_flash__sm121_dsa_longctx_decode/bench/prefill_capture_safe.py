#!/usr/bin/env python3
"""SAFE cold-prefill profiler capture: cap forwards with num_steps so the trace
stays small (the previous full-prefill dump crashed the scheduler, exit 255).

  python3 prefill_capture_safe.py --len 8192 --num-steps 6

The profiler auto-stops after num_steps forwards; we then wait for the trace to
appear in /tmp and copy it out. No explicit /stop_profile (avoids the large-dump
path that crashed).
"""
import argparse, json, os, subprocess, time, urllib.request

KEY = "REDACTED_SET_SGLANG_API_KEY"
BASE = "http://10.100.8.1:8940"
CONTAINER = "dspark-tp2-rank0"


def ids_for(L, salt=123):
    return [100 + ((i * 7 + salt * 13 + i // 97) % 50) for i in range(L)]


def post(path, payload, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode(errors="replace")


def traces():
    out = subprocess.run(["docker", "exec", CONTAINER, "bash", "-lc",
        "ls -t /tmp/*TP-0.trace.json.gz 2>/dev/null || true"],
        capture_output=True, text=True).stdout
    return out.split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", type=int, required=True)
    ap.add_argument("--num-steps", type=int, default=6)
    ap.add_argument("--salt", type=int, default=123)
    args = ap.parse_args()
    before = set(traces())
    print(f"start_profile num_steps={args.num_steps} (no detailed_annotations)", flush=True)
    post("/start_profile", {"num_steps": args.num_steps, "record_shapes": False})
    time.sleep(0.3)
    t0 = time.perf_counter()
    post("/generate", {"input_ids": ids_for(args.len, args.salt),
                       "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}})
    print(f"  request wall={time.perf_counter()-t0:.1f}s", flush=True)
    # the profiler auto-stops and dumps after num_steps; wait for a new trace
    tr = None
    for _ in range(60):
        time.sleep(2)
        new = [t for t in traces() if t not in before]
        if new:
            tr = new[0]; break
    if not tr:
        print("WARN: no new trace appeared", flush=True); return
    time.sleep(2)
    dest = f"/home/spark/profiles/longctx/PF{args.len}_ns{args.num_steps}.trace.json.gz"
    subprocess.check_call(["docker", "cp", f"{CONTAINER}:{tr}", dest])
    print(f"copied -> {dest} ({os.path.getsize(dest)} bytes)", flush=True)


if __name__ == "__main__":
    main()
