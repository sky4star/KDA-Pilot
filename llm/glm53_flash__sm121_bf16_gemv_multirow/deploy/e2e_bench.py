#!/usr/bin/env python3
"""Single-request decode A/B client for the GLM-5.3-Flash DSpark server.

Uses the native /generate endpoint with explicit input_ids (exact prompt length)
and streaming to separate TTFT from decode. Reports median decode tok/s.

  e2e_bench.py --base http://10.100.8.1:8940 --key ... [--prompt 64] [--out 128]
"""
import argparse
import json
import os
import statistics
import time
import urllib.request

DEFAULT_KEY = os.environ.get("SGLANG_API_KEY", "REPLACE_WITH_API_KEY")


def make_input_ids(n):
    return [100 + (i % 50) for i in range(n)]


PROMPTS = [
    ("Write a detailed, well-commented Python implementation of a red-black tree "
     "with insert, delete, and search. Then explain the time complexity of each "
     "operation and give a worked example with at least five insertions."),
    ("Explain how a transformer language model computes attention. Derive the "
     "scaled dot-product attention formula, describe multi-head attention, and "
     "discuss the memory cost of the KV cache during autoregressive decoding."),
    ("Prove that the square root of 2 is irrational. Then prove the infinitude of "
     "the primes using Euclid's argument, and state the fundamental theorem of "
     "arithmetic with a brief proof sketch."),
]
DEFAULT_TEXT = PROMPTS[0]


def one_request(base, key, prompt_len, max_new, text=None, timeout=600):
    payload = {
        "text" if text else "input_ids": text if text else make_input_ids(prompt_len),
        "sampling_params": {"max_new_tokens": max_new, "temperature": 0.0},
        "stream": True,
    }
    req = urllib.request.Request(
        base + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    t0 = time.perf_counter()
    ttft = None
    last = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body in ("", "[DONE]"):
                continue
            try:
                j = json.loads(body)
            except Exception:
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunks += 1
            last = j
    t1 = time.perf_counter()
    meta = (last or {}).get("meta_info", {}) if isinstance(last, dict) else {}
    ctok = meta.get("completion_tokens")
    if ctok is None:
        ctok = chunks
    return {
        "ttft_ms": (ttft or 0.0) * 1e3,
        "e2e_ms": (t1 - t0) * 1e3,
        "decode_ms": (t1 - t0 - (ttft or 0.0)) * 1e3,
        "completion_tokens": ctok,
        "chunks": chunks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://10.100.8.1:8940")
    ap.add_argument("--key", default=DEFAULT_KEY)
    ap.add_argument("--prompt", type=int, default=64)
    ap.add_argument("--out", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--text", action="store_true", help="use a real text prompt")
    ap.add_argument("--prompt-idx", type=int, default=0)
    args = ap.parse_args()

    text = PROMPTS[args.prompt_idx] if args.text else None
    for _ in range(args.warmup):
        r = one_request(args.base, args.key, args.prompt, 16, text=text)
    print("warmup done")

    rows = []
    for _ in range(args.runs):
        rows.append(one_request(args.base, args.key, args.prompt, args.out, text=text))

    def med(k):
        return statistics.median(r[k] for r in rows)

    tps = statistics.median(
        (r["completion_tokens"] - 1) / (r["decode_ms"] / 1000.0)
        for r in rows if r["decode_ms"] > 0
    )
    e2e_tps = statistics.median(
        r["completion_tokens"] / (r["e2e_ms"] / 1000.0) for r in rows
    )
    summary = {
        "prompt": args.prompt, "out": args.out, "runs": args.runs,
        "median_ttft_ms": round(med("ttft_ms"), 1),
        "median_e2e_ms": round(med("e2e_ms"), 1),
        "median_decode_ms": round(med("decode_ms"), 1),
        "median_completion_tokens": med("completion_tokens"),
        "median_decode_tok_s": round(tps, 2),
        "median_e2e_tok_s": round(e2e_tps, 2),
        "rows": rows,
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
