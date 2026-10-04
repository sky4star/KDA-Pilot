#!/usr/bin/env python3
"""Dense NVFP4 GEMM peak on GB10 + backend comparison (mm_fp4)."""
from __future__ import annotations
import argparse, statistics
import torch
from flashinfer import nvfp4_quantize, mm_fp4, SfLayout


def bench(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(ts)


def run(M, K, N, backend):
    dev = "cuda"
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16) / 10
    b = torch.randn(N, K, device=dev, dtype=torch.bfloat16) / 10  # row-major [N,K]
    a_gs = (448 * 6) / a.float().abs().nan_to_num().max()
    b_gs = (448 * 6) / b.float().abs().nan_to_num().max()
    a_fp4, a_sf = nvfp4_quantize(a, a_gs, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    b_fp4, b_sf = nvfp4_quantize(b, b_gs, sfLayout=SfLayout.layout_128x4,
                                 do_shuffle=(backend == "trtllm"))
    alpha = torch.tensor(1.0 / (a_gs * b_gs).item(), device=dev, dtype=torch.float32)
    out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)

    def fn():
        mm_fp4(a_fp4, b_fp4.T, a_sf, b_sf.T, alpha, torch.bfloat16, out,
               block_size=16, use_8x4_sf_layout=False, backend=backend)

    try:
        us = bench(fn)
    except Exception as ex:
        print(f"  {backend:8s} M={M} K={K} N={N} FAILED {ex!r}")
        return
    flops = 2.0 * M * N * K
    print(f"  {backend:8s} M={M:6d} K={K:6d} N={N:6d}  {us:9.1f} us  {flops/(us*1e-6)/1e12:7.1f} TFLOPS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["b12x", "cutlass", "cudnn", "cute-dsl"])
    ap.add_argument("--shapes", nargs="+", default=["4096,4096,4096", "8192,8192,8192", "16384,8192,8192", "16384,4096,4096"])
    args = ap.parse_args()
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    for bk in args.backends:
        print(f"== backend={bk} ==")
        for sh in args.shapes:
            M, K, N = map(int, sh.split(","))
            run(M, K, N, bk)


if __name__ == "__main__":
    raise SystemExit(main())
