#!/usr/bin/env python3
"""Roofline probe for GLM-5.3-Flash prefill on GB10 (sm_121).

Step 1 (per handoff section 7): measure, before writing any kernel:
  A. GB10 bf16 peak TFLOPS via large square cuBLAS GEMMs.
  B. projection-like GEMMs at M=4096 for the real (K,N) shapes -> MFU.
  C. MoE routed NVFP4 at M=4096 (flashinfer cutlass_fused_moe) -> effective MFU.

Run inside the serving image:
    python3 bench/roofline.py
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def flush_l2():
    if not hasattr(flush_l2, "buf"):
        flush_l2.buf = torch.empty(int(2.2 * 25.2e6), dtype=torch.int8, device="cuda")
    flush_l2.buf.zero_()


def bench(fn, iters, warmup=10, cold=False):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        if cold:
            flush_l2()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(ts), min(ts), max(ts)


def gemm_bf16(M, K, N, iters=50, cold=True):
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) / 10
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) / 10
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    fn = lambda: torch.mm(a, b, out=out)
    med, lo, hi = bench(fn, iters, cold=cold)
    flops = 2.0 * M * N * K
    return dict(M=M, K=K, N=N, us=med, tflops=flops / (med * 1e-6) / 1e12,
                gbps=(M * K + K * N + M * N) * 2 / (med * 1e-6) / 1e9, min_us=lo, max_us=hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--peak", action="store_true", help="only run peak GEMMs")
    args = ap.parse_args()
    dev = "cuda"
    print(f"torch {torch.__version__} {torch.cuda.get_device_name(0)} cap {torch.cuda.get_device_capability(0)}")
    print(f"cudnn {torch.backends.cudnn.version()} cublas available {torch.backends.cuda.is_built()}")

    rows = []
    print("\n== A. bf16 peak (square, compute-bound) ==")
    for n in [4096, 8192, 12288, 16384]:
        r = gemm_bf16(n, n, n, iters=args.iters, cold=False)
        rows.append(("peak", r))
        print(f"  {n:6d}^3  {r['us']:9.1f} us  {r['tflops']:7.1f} TFLOPS  ({r['gbps']:6.0f} GB/s)")

    if args.peak:
        Path("/tmp/roofline_peak.json").write_text(json.dumps(rows, indent=1))
        return

    print("\n== B. M=4096 projection-like bf16 GEMM  (K,N) ==")
    proj_shapes = [
        (4096, 4096), (4096, 2048), (4096, 12288), (4096, 24576),
        (4096, 1536), (1536, 16384), (512, 4096), (4096, 512),
        (4096, 512 * 384), (4096, 154880),
    ]
    for K, N in proj_shapes:
        try:
            r = gemm_bf16(4096, K, N, iters=max(20, args.iters // 2))
        except Exception as ex:
            print(f"  M=4096 K={K:6d} N={N:6d}  FAILED {ex!r}")
            continue
        rows.append(("proj", r))
        print(f"  M=4096 K={K:6d} N={N:6d}  {r['us']:9.1f} us  {r['tflops']:7.1f} TFLOPS  ({r['gbps']:6.0f} GB/s)")

    print("\n== C. full model bf16 weight bytes per forward ==")
    try:
        from pathlib import Path as P
        idx = P("/models/GLM-5.3-Flash-NVFP4/model.safetensors.index.json")
    except Exception:
        pass

    out = P(__file__).with_name("roofline.json")
    out.write_text(json.dumps(rows, indent=1))
    print("\nwrote", out)


if __name__ == "__main__":
    raise SystemExit(main())
