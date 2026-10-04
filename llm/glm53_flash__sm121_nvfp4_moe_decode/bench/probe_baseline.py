#!/usr/bin/env python3
"""Standalone probe: time the production NVFP4 grouped-MoE call and compare its
weight traffic against the GB10 DRAM ceiling.

No SGLang import: builds the FlashInfer `cutlass_fused_moe` payload directly,
mirroring `nvfp4_moe_harness.py` (which mirrors
`sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py::_run_flashinfer_cutlass`).

Run inside the serving image:
    python3 bench/probe_baseline.py --M 1 8 --E 144
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from flashinfer import fp4_quantize
from flashinfer.fused_moe import cutlass_fused_moe

N = 2048
K = 4096
BLK = 16
FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0


def round_up(x, y):
    return (x + y - 1) // y * y


def build(M, E, topk, seed=7):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    a = (torch.randn((M, K), device=dev, dtype=torch.bfloat16, generator=g) / 10)
    # packed NVFP4 weights (checkpoint layout, E x rows x K/2 uint8) + block scales.
    w1_q = torch.empty((E, 2 * N, K // 2), device=dev, dtype=torch.uint8)
    w2_q = torch.empty((E, K, N // 2), device=dev, dtype=torch.uint8)
    w1_bs = torch.empty((E, round_up(2 * N, 128), round_up(K // BLK, 4)), device=dev, dtype=torch.float8_e4m3fn)
    w2_bs = torch.empty((E, round_up(K, 128), round_up(N // BLK, 4)), device=dev, dtype=torch.float8_e4m3fn)
    w1_gs = torch.empty((E,), device=dev, dtype=torch.float32)
    w2_gs = torch.empty((E,), device=dev, dtype=torch.float32)
    gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / (2.0 / 10.0)
    for e in range(E):
        w1 = (torch.randn((2 * N, K), device=dev, dtype=torch.bfloat16, generator=g) / 10)
        w2 = (torch.randn((K, N), device=dev, dtype=torch.bfloat16, generator=g) / 10)
        w1_gs[e] = gs
        w2_gs[e] = gs
        w1_q[e], w1_bs[e] = fp4_quantize(w1, w1_gs[e])
        w2_q[e], w2_bs[e] = fp4_quantize(w2, w2_gs[e])
        del w1, w2
    score = torch.randn((M, E), device=dev, dtype=torch.bfloat16, generator=g)
    tw, ti = torch.topk(score, topk, dim=-1, sorted=False)
    tw = torch.softmax(tw.float(), dim=-1)
    a_gs = torch.ones((E,), device=dev, dtype=torch.float32)
    return dict(a=a, tw=tw, ti=ti.to(torch.int), w1_q=w1_q.view(torch.long) if False else w1_q,
                w2_q=w2_q, w1_bs=w1_bs, w1_gs=w1_gs, w2_bs=w2_bs, w2_gs=w2_gs, a_gs=a_gs)


def call(p):
    return cutlass_fused_moe(
        p["a"], p["ti"], p["tw"],
        p["w1_q"].view(torch.long), p["w2_q"].view(torch.long), p["a"].dtype,
        quant_scales=[p["a_gs"], p["w1_bs"].view(torch.int32), (1 / p["w1_gs"]),
                      p["a_gs"], p["w2_bs"].view(torch.int32), (1 / p["w2_gs"])],
    )[0]


def timed(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); e.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--E", type=int, default=144)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("probe_baseline.json"))
    args = ap.parse_args()
    print(f"torch {torch.__version__} device {torch.cuda.get_device_name(0)} E={args.E} topk={args.topk}")
    rows = []
    for M in args.M:
        p = build(M, args.E, args.topk)
        try:
            out = call(p); torch.cuda.synchronize()
        except Exception as ex:
            print(f"M={M}: baseline call failed: {ex!r}")
            continue
        us = timed(lambda: call(p))
        # routed expert weight bytes (each selected expert's w13+w2 read once)
        wt_bytes = args.topk * (2 * N * (K // 2) + K * (N // 2))
        rows.append(dict(M=M, E=args.E, topk=args.topk, us=us,
                         out=str(tuple(out.shape)),
                         expert_weight_mb=wt_bytes / 1e6,
                         gbps=wt_bytes / (us * 1e-6) / 1e9))
        print(f"M={M:2d}  {us:8.1f} us   routed_w={wt_bytes/1e6:6.1f} MB  "
              f"({args.topk} experts)  achieved={wt_bytes/(us*1e-6)/1e9:6.1f} GB/s")
    args.out.write_text(json.dumps(rows, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    raise SystemExit(main())
