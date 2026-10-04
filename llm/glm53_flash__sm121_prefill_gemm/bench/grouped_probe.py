#!/usr/bin/env python3
"""Probe: can a generic grouped NVFP4 GEMM (cuDNN, via grouped_mm_fp4) beat the
FlashInfer CUTLASS grouped kernel at the M=4096 prefill shape?

Synthetic grouped problem: E experts, uniform token distribution, total
cum_m = M*topk rows.  This measures only the GEMM, not routing/gather.
"""
from __future__ import annotations
import argparse, statistics
import torch
from flashinfer import nvfp4_quantize, SfLayout, grouped_mm_fp4


def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(ts)


def make(cum_m, E, N, K, dev="cuda"):
    a = (torch.randn(cum_m, K, device=dev, dtype=torch.bfloat16) / 10)
    a_gs = (448 * 6) / a.float().abs().nan_to_num().max()
    a_fp4, a_sf = nvfp4_quantize(a, a_gs, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    a_sf = a_sf.view(torch.float8_e4m3fn)
    # expert weights [E, N, K]
    w = (torch.randn(E, N, K, device=dev, dtype=torch.bfloat16) / 10)
    w_gs = (448 * 6) / w.float().abs().nan_to_num().max()
    w_fp4, w_sf = nvfp4_quantize(w, w_gs, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    w_sf = w_sf.reshape(E, N, K // 16).contiguous().view(torch.float8_e4m3fn)
    counts = torch.full((E,), cum_m // E, device=dev, dtype=torch.int32)
    counts[: cum_m % E] += 1
    indptr = torch.zeros(E + 1, device=dev, dtype=torch.int32)
    indptr[1:] = torch.cumsum(counts, 0)
    alpha = torch.tensor([1.0 / (a_gs * w_gs).item()], device=dev, dtype=torch.float32)
    return a_fp4, a_sf, w_fp4, w_sf, indptr, alpha


def run(M, E, topk, N, K, backend, label):
    cum_m = M * topk
    a_fp4, a_sf, w_fp4, w_sf, indptr, alpha = make(cum_m, E, N, K)
    out = torch.empty(cum_m, N, device="cuda", dtype=torch.bfloat16)

    def fn():
        grouped_mm_fp4(a_fp4, w_fp4, a_sf, w_sf, indptr, alpha, out,
                       out_dtype=torch.bfloat16, block_size=16, backend=backend)

    try:
        us = bench(fn)
    except Exception as ex:
        print(f"  {label:6s} {backend:8s} FAILED {type(ex).__name__}: {str(ex)[:160]}")
        return
    flops = 2.0 * cum_m * N * K
    print(f"  {label:6s} {backend:8s} M={M} E={E} N={N} K={K}  {us:9.1f} us  {flops/(us*1e-6)/1e12:7.1f} TFLOPS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["cudnn"])
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--E", type=int, default=144)
    args = ap.parse_args()
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    for bk in args.backends:
        run(args.M, args.E, args.topk, 2 * 2048, 4096, bk, "w13")
        run(args.M, args.E, args.topk, 4096, 2048, bk, "w2")


if __name__ == "__main__":
    raise SystemExit(main())
