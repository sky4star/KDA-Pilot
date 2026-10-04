#!/usr/bin/env python3
"""Cold-prefill per-chunk timing from the server log.

Sends a unique 100K prompt (no cache hit) and parses each `Prefill batch`
line's token/s. Per-chunk time = chunk_tokens / throughput; the growth from the
first chunk (context~0) to the last (context~100K) is the attention component.
Timing only -- no torch profiler (which crashes this box on long prefill).
"""
import json, re, subprocess, sys, time, urllib.request

KEY = "REDACTED_SET_SGLANG_API_KEY"
BASE = "http://10.100.8.1:8940"
CONTAINER = "dspark-tp2-rank0"


def ids_for(L, salt=7777):
    return [100 + ((i * 7 + salt * 13 + (i * i) % 11) % 50) for i in range(L)]


def log_lines(since_iso):
    out = subprocess.run(["docker", "logs", "--since", since_iso, CONTAINER],
                         capture_output=True, text=True).stdout
    return out.splitlines()


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    # local/container timezone offset: just take all lines after a marker via tail count
    before = len(subprocess.run(["docker", "logs", CONTAINER], capture_output=True, text=True).stdout.splitlines())
    t0 = time.perf_counter()
    req = urllib.request.Request(BASE + "/generate",
        data=json.dumps({"input_ids": ids_for(L), "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}}).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=3600) as r:
        m = json.loads(r.read().decode())["meta_info"]
    wall = time.perf_counter() - t0
    all_lines = subprocess.run(["docker", "logs", CONTAINER], capture_output=True, text=True).stdout.splitlines()
    new = all_lines[before:]
    rows = []
    for ln in new:
        mt = re.search(r"#new-token:\s*(\d+).*?throughput \(token/s\):\s*([0-9.]+)", ln)
        if mt:
            rows.append((int(mt.group(1)), float(mt.group(2))))
    print(f"L={L} wall={wall:.1f}s completion={m.get('completion_tokens')} chunks={len(rows)}")
    # show first/last chunk timings
    tot = 0.0
    for i, (nt, tps) in enumerate(rows):
        dt = nt / tps  # seconds for this chunk (approx, includes queue)
        tot += dt
        if i < 4 or i >= len(rows) - 5:
            print(f"  chunk {i+1:3d}: tokens={nt:5d} {tps:8.1f} tok/s  ~{dt*1000:6.1f} ms  (cum ctx ~{sum(r[0] for r in rows[:i]):6d})")
    print(f"  sum(chunk times)={tot:.1f}s")
    if rows:
        print(f"  first chunk {rows[0][0]/rows[0][1]*1000:.1f} ms | last chunk {rows[-1][0]/rows[-1][1]*1000:.1f} ms "
              f"| growth x{(rows[-1][0]/rows[-1][1])/(rows[0][0]/rows[0][1]):.2f}")


if __name__ == "__main__":
    main()
